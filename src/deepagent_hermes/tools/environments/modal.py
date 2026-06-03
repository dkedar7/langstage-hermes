"""Modal cloud-sandbox terminal backend — lazy SDK adapter.

Uses the official ``modal`` Python SDK (declared as the ``[modal]`` extra).
Lazy-imported inside :meth:`init_session` so installing ``deepagent-hermes``
without the extra doesn't pull in Modal's hefty dependency tree.

Modal authentication is handled by the SDK itself via the
``MODAL_TOKEN_ID`` + ``MODAL_TOKEN_SECRET`` env vars (their convention).
We don't pass the tokens explicitly — the SDK reads them on first use —
but :func:`_check_modal_auth` warns clearly when they're missing so the
failure isn't buried inside an opaque SDK exception.

One sandbox per session: :meth:`init_session` constructs the sandbox;
every :meth:`execute` reuses it via ``sandbox.exec``; :meth:`cleanup`
calls ``sandbox.terminate()``.

The SDK call surface mirrors the public Modal docs as of mid-2026
(https://modal.com/docs/reference/modal.Sandbox):

    import modal
    app = modal.App.lookup("deepagent-hermes", create_if_missing=True)
    sandbox = modal.Sandbox.create("python:3.13-slim", app=app)
    process = sandbox.exec("bash", "-c", "echo hi")
    output = process.stdout.read()
    exit_code = process.wait()

Where the docs are ambiguous (e.g. the Image construction API has shifted
across releases), the code is annotated with
``# TODO(modal-api-verify): ...`` for confirmation at integration time.
"""

from __future__ import annotations

import io
import os
from typing import Any

from deepagent_hermes.tools.environments.base import (
    BaseEnvironment,
    ProcessHandle,
)

_INSTALL_HINT = (
    "ModalEnvironment requires the 'modal' package. "
    "Install with: pip install deepagent-hermes[modal]"
)

_APP_NAME = "deepagent-hermes"
_DEFAULT_IMAGE = "python:3.13-slim"


def _import_sdk() -> Any:
    """Lazy import the Modal SDK, raising a helpful error if missing."""
    try:
        import modal  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - covered by test_modal_raises_without_sdk
        raise ImportError(_INSTALL_HINT) from exc
    return modal


def _check_modal_auth() -> tuple[str | None, str | None]:
    """Return (token_id, token_secret) from env — Modal's documented convention.

    The SDK reads these itself; we don't need to pass them anywhere. But we
    surface a clear error from :meth:`init_session` when neither is set,
    rather than letting the SDK throw its own (often less obvious) auth
    failure deep inside ``Sandbox.create``.
    """
    return (
        os.environ.get("MODAL_TOKEN_ID"),
        os.environ.get("MODAL_TOKEN_SECRET"),
    )


# ── ProcessHandle adapter ─────────────────────────────────────────────


class _ModalProcessHandle:
    """Wrap a ``modal.container_process.ContainerProcess`` as a ``ProcessHandle``.

    Modal's process object exposes ``stdout`` / ``stderr`` streams and a
    ``wait()`` for the exit code, but not a Popen-style ``poll`` / ``kill``
    interface. We adapt: ``poll`` peeks at the cached exit code if ``wait``
    has already returned; otherwise returns ``None``. ``kill`` is best-effort
    via the SDK's ``terminate`` if available.

    For the base class's drain loop we eagerly read stdout+stderr inside
    :meth:`wait` and stash them in a single in-memory buffer exposed via the
    ``stdout`` property — matches Daytona's blocking-handle shape.
    """

    def __init__(self, process: Any) -> None:
        self._process = process
        self._exit_code: int | None = None
        self._stdout_buf: io.BytesIO | None = None

    def _materialize_output(self) -> None:
        """Read both streams into one buffer; called once on wait()."""
        if self._stdout_buf is not None:
            return
        try:
            # TODO(modal-api-verify): ``stdout.read()`` returns bytes in
            # current SDK; older versions returned str. Normalise to bytes.
            stdout_data = self._process.stdout.read()
            if isinstance(stdout_data, str):
                stdout_data = stdout_data.encode("utf-8", errors="replace")
        except Exception:
            stdout_data = b""
        try:
            stderr_data = self._process.stderr.read()
            if isinstance(stderr_data, str):
                stderr_data = stderr_data.encode("utf-8", errors="replace")
        except Exception:
            stderr_data = b""
        # Merge stderr after stdout so chronology is approximately preserved
        # (Modal interleaves them on the wire, but the SDK exposes them split).
        combined = stdout_data + (b"\n" + stderr_data if stderr_data else b"")
        self._stdout_buf = io.BytesIO(combined)

    def poll(self) -> int | None:
        return self._exit_code

    def kill(self) -> None:
        # Modal SDK exposes process.terminate() or kill() depending on version.
        for attr in ("terminate", "kill"):
            fn = getattr(self._process, attr, None)
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass
                return

    def wait(self, timeout: float | None = None) -> int:
        # TODO(modal-api-verify): ``process.wait()`` returns int exit code in
        # current SDK; some snapshots return a Result object with .exit_code.
        raw = self._process.wait()
        if isinstance(raw, int):
            self._exit_code = raw
        else:
            self._exit_code = int(getattr(raw, "exit_code", getattr(raw, "returncode", 0)))
        self._materialize_output()
        return self._exit_code

    @property
    def stdout(self):
        if self._stdout_buf is None:
            self._materialize_output()
        return self._stdout_buf

    @property
    def returncode(self) -> int | None:
        return self._exit_code


# ── Environment ────────────────────────────────────────────────────────


class ModalEnvironment(BaseEnvironment):
    """Run commands in a Modal sandbox, one sandbox per session."""

    # Modal cold-starts (image build + sandbox boot) frequently take 30-60s.
    _snapshot_timeout: int = 180

    def __init__(
        self,
        session_id: str,
        *,
        image: str = _DEFAULT_IMAGE,
        app_name: str = _APP_NAME,
    ) -> None:
        super().__init__(session_id=session_id)
        # Validate SDK is importable at construction so callers get the
        # actionable ImportError immediately, not on first execute().
        self._sdk = _import_sdk()
        self._image_spec = image
        self._app_name = app_name
        self._app: Any | None = None
        self._sandbox: Any | None = None

    # ── one-time sandbox provisioning ─────────────────────────────────

    def init_session(self) -> None:
        """Look up the Modal app, construct the sandbox, then run the bootstrap.

        We override the base class to set up ``self._sandbox`` before the
        snapshot bootstrap ``_run_bash`` call runs (it needs the sandbox to
        exist).
        """
        if self._initialized:
            return

        token_id, token_secret = _check_modal_auth()
        if not (token_id and token_secret):
            raise RuntimeError(
                "Modal auth missing. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET "
                "(see https://modal.com/settings/tokens), or run `modal token new`."
            )

        modal = self._sdk

        # TODO(modal-api-verify): App lookup API. Current docs use
        # ``modal.App.lookup(name, create_if_missing=True)``; older releases
        # used ``modal.Stub.lookup``. We probe for App first.
        app_factory = getattr(modal, "App", None) or getattr(modal, "Stub", None)
        if app_factory is None:
            raise ImportError(
                "modal SDK has no App or Stub class — the SDK API may have changed. "
                "Please file an issue against deepagent-hermes with your modal version."
            )
        lookup = getattr(app_factory, "lookup", None)
        if lookup is not None:
            self._app = lookup(self._app_name, create_if_missing=True)
        else:
            # Fallback: construct directly. Some very old SDKs only had
            # the constructor.
            self._app = app_factory(self._app_name)

        # TODO(modal-api-verify): Image construction. Current SDK uses
        # ``modal.Image.from_registry(image_spec)``; some snapshots used
        # ``modal.Image.from_dockerhub`` or accepted a bare string in
        # ``Sandbox.create(image=...)``.
        image_obj: Any
        Image = getattr(modal, "Image", None)
        if Image is not None and hasattr(Image, "from_registry"):
            image_obj = Image.from_registry(self._image_spec)
        else:
            # Last-resort: pass the string and let Sandbox.create decide.
            image_obj = self._image_spec

        # TODO(modal-api-verify): Sandbox.create signature. Current docs show
        # ``modal.Sandbox.create(image=..., app=...)`` — when no command is
        # given the sandbox just runs an idle shell, which is what we want
        # since each exec() call spawns its own process.
        Sandbox = getattr(modal, "Sandbox", None)
        if Sandbox is None or not hasattr(Sandbox, "create"):
            raise ImportError(
                "modal SDK has no Sandbox.create — the SDK API may have changed."
            )
        self._sandbox = Sandbox.create(image=image_obj, app=self._app)

        # Now run the standard base-class snapshot bootstrap inside the sandbox.
        super().init_session()

    # ── command spawn ─────────────────────────────────────────────────

    def _run_bash(
        self,
        cmd: str,
        *,
        login: bool = False,
        timeout: int = 60,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Spawn ``sandbox.exec("bash", "-l", "-c", cmd)`` and adapt the handle."""
        if self._sandbox is None:
            raise RuntimeError("Modal sandbox not initialized — call init_session() first.")

        argv = ["bash", "-l", "-c", cmd] if login else ["bash", "-c", cmd]
        # TODO(modal-api-verify): exact exec signature. Docs show
        # ``sandbox.exec(*cmd)`` returning a ContainerProcess.
        process = self._sandbox.exec(*argv)
        return _ModalProcessHandle(process)

    # ── cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Terminate the sandbox; best-effort, swallow SDK errors."""
        if self._sandbox is not None:
            for attr in ("terminate", "stop", "kill"):
                fn = getattr(self._sandbox, attr, None)
                if fn is not None:
                    try:
                        fn()
                    except Exception:
                        pass
                    break
            self._sandbox = None
        self._app = None
        self._initialized = False


__all__ = ["ModalEnvironment"]
