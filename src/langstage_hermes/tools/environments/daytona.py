"""Daytona cloud-sandbox terminal backend — lazy SDK adapter.

Uses Daytona's published Python SDK (``daytona-sdk``, declared as the
``[daytona]`` extra). The SDK is imported lazily inside :meth:`init_session`
so installing ``langstage-hermes`` without the extra still keeps the rest of
the package importable; instantiating this class without the SDK raises a
clear ``ImportError`` pointing at the extra.

API key resolution prefers Daytona's own convention (``DAYTONA_API_KEY``) and
falls back to a namespaced override (``DEEPAGENT_HERMES_DAYTONA_API_KEY``) so
callers running multiple agent installs side-by-side can scope keys.

One sandbox per session: :meth:`init_session` creates the sandbox once;
every :meth:`execute` reuses it; :meth:`cleanup` deletes it. The wrapped
bash command (snapshot + cwd persistence) is shipped via
``sandbox.process.exec(...)`` — Daytona's spec-documented API for running a
shell-string in the sandbox.

The SDK call surface mirrors the public Daytona docs at
https://www.daytona.io/docs/python-sdk/ as of mid-2026:

    from daytona_sdk import Daytona, CreateSandboxParams
    client = Daytona(api_key=...)
    sandbox = client.create(CreateSandboxParams(language="python"))
    result = sandbox.process.exec(command="echo hi")
    output, exit_code = result.result, result.exit_code

Where the docs are ambiguous (e.g. exact attribute names on the response
object across SDK versions) the code is annotated with
``# TODO(daytona-api-verify): ...`` for confirmation at integration time.
"""

from __future__ import annotations

import os
import time
from typing import Any

from langstage_hermes.tools.environments.base import (
    BaseEnvironment,
    ProcessHandle,
)

_INSTALL_HINT = "DaytonaEnvironment requires the 'daytona-sdk' package. Install with: pip install langstage-hermes[daytona]"


def _resolve_api_key() -> str | None:
    """Pick up Daytona credentials from env, preferring their convention.

    Order:
      1. ``DAYTONA_API_KEY`` (Daytona's own env var — what their SDK auto-reads)
      2. ``DEEPAGENT_HERMES_DAYTONA_API_KEY`` (namespaced override)
    """
    return (
        os.environ.get("DAYTONA_API_KEY")
        or os.environ.get("LANGSTAGE_HERMES_DAYTONA_API_KEY")
        or os.environ.get("DEEPAGENT_HERMES_DAYTONA_API_KEY")
    )


def _import_sdk() -> Any:
    """Lazy import the Daytona SDK, raising a helpful error if missing.

    Imported on first use rather than at module load so that
    ``from langstage_hermes.tools.environments import daytona`` stays free of
    the heavyweight optional dep when callers only want the class object for
    a type check or factory lookup.
    """
    try:
        import daytona_sdk  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - covered by test_daytona_raises_without_sdk
        raise ImportError(_INSTALL_HINT) from exc
    return daytona_sdk


# ── ProcessHandle adapter for blocking SDK calls ──────────────────────


class _BlockingResultHandle:
    """Wrap an already-completed SDK call as a ``ProcessHandle``.

    Daytona's ``sandbox.process.exec()`` is synchronous and returns the full
    result in one shot (no streaming stdout). To plug that into the base
    class's ``_drain`` loop (which expects ``poll``/``wait``/``stdout``) we
    fake a finished-process handle: ``poll`` and ``returncode`` immediately
    return the recorded exit code; ``stdout`` is a one-shot in-memory stream
    holding the captured output bytes.
    """

    def __init__(self, output: str, exit_code: int) -> None:
        import io

        self._exit_code = exit_code
        # Encode and wrap so the base class's generic drain path (iterates
        # bytes) gets data in the same shape as a real Popen.stdout.
        self._stdout = io.BytesIO(output.encode("utf-8", errors="replace"))

    def poll(self) -> int | None:
        return self._exit_code

    def kill(self) -> None:  # pragma: no cover - nothing to kill, call already returned
        pass

    def wait(self, timeout: float | None = None) -> int:
        return self._exit_code

    @property
    def stdout(self):
        return self._stdout

    @property
    def returncode(self) -> int | None:
        return self._exit_code


# ── Environment ────────────────────────────────────────────────────────


class DaytonaEnvironment(BaseEnvironment):
    """Run commands in a Daytona cloud sandbox, one sandbox per session.

    Sandbox lifecycle:
      - :meth:`init_session` (called by the base class on first execute)
        constructs the Daytona client, creates a sandbox, and runs the
        snapshot-bootstrap script inside it.
      - Subsequent :meth:`execute` calls reuse the sandbox via
        ``sandbox.process.exec``.
      - :meth:`cleanup` calls ``sandbox.delete()``.
    """

    # Cloud cold start: image pull + container provisioning easily takes
    # tens of seconds on first call.
    _snapshot_timeout: int = 180

    def __init__(
        self,
        session_id: str,
        *,
        language: str = "python",
        api_key: str | None = None,
    ) -> None:
        super().__init__(session_id=session_id)
        # Eagerly validate the SDK is importable so we fail fast at construction
        # time instead of buried in the first execute() call. The actual client
        # construction is deferred to init_session() so tests can mock the SDK
        # surface without paying the construction cost.
        self._sdk = _import_sdk()
        self._language = language
        self._api_key = api_key or _resolve_api_key()
        self._client: Any | None = None
        self._sandbox: Any | None = None

    # ── one-time sandbox provisioning ─────────────────────────────────

    def init_session(self) -> None:
        """Create the Daytona client + sandbox, then run the snapshot bootstrap.

        Overrides the base class so we can build the sandbox before the
        snapshot bootstrap runs — the base class's ``init_session`` calls
        ``_run_bash``, which needs ``self._sandbox`` to exist.
        """
        if self._initialized:
            return

        Daytona = self._sdk.Daytona
        # TODO(daytona-api-verify): The exact name of the params class varies
        # between SDK releases. ``CreateSandboxParams`` is the documented
        # current name; older releases used ``CreateWorkspaceParams``. If the
        # SDK exposes a different name, surface a clear hint.
        CreateSandboxParams = getattr(self._sdk, "CreateSandboxParams", None)
        if CreateSandboxParams is None:
            CreateSandboxParams = getattr(self._sdk, "CreateWorkspaceParams", None)
        if CreateSandboxParams is None:
            raise ImportError(
                "daytona-sdk is installed but neither CreateSandboxParams nor "
                "CreateWorkspaceParams is exposed. The SDK API may have changed; "
                "please file an issue against langstage-hermes with your SDK version."
            )

        # Daytona SDK reads DAYTONA_API_KEY from env automatically, but we
        # pass explicitly when set so the resolution order in _resolve_api_key
        # (incl. our namespaced override) is honoured.
        if self._api_key:
            self._client = Daytona(api_key=self._api_key)
        else:
            # No key: rely on the SDK's own env-var resolution / config file
            # (~/.daytona/config). Will raise inside the SDK if no auth found.
            self._client = Daytona()

        # TODO(daytona-api-verify): docs show ``client.create(params)``; some
        # versions name this ``create_sandbox`` / ``create_workspace``. Probe
        # for whichever exists.
        create_fn = (
            getattr(self._client, "create", None)
            or getattr(self._client, "create_sandbox", None)
            or getattr(self._client, "create_workspace", None)
        )
        if create_fn is None:
            raise ImportError(
                "daytona-sdk client has no create()/create_sandbox()/create_workspace() method. The SDK API may have changed."
            )
        self._sandbox = create_fn(CreateSandboxParams(language=self._language))

        # Now run the standard base-class snapshot bootstrap inside the sandbox.
        super().init_session()

    # ── command spawn (synchronous SDK call → fake handle) ────────────

    def _run_bash(
        self,
        cmd: str,
        *,
        login: bool = False,
        timeout: int = 60,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Execute ``cmd`` in the sandbox and return a finished-process handle.

        Daytona's ``process.exec`` is synchronous and returns the full result,
        so we run it inline and wrap the result in :class:`_BlockingResultHandle`.
        That preserves the contract that the base class's ``_drain`` expects.

        ``stdin_data`` is unused: the base class uses ``_stdin_mode = "pipe"``
        by default; SDK backends that need stdin should embed it as a heredoc
        inside ``cmd`` rather than rely on a stdin pipe. (Switching to
        ``_stdin_mode = "heredoc"`` is a follow-up — current callers don't use
        stdin_data.)
        """
        if self._sandbox is None:
            # Defensive: init_session() should have been called by execute().
            raise RuntimeError("Daytona sandbox not initialized — call init_session() first.")

        # bash -l vs bash -c happens inside the sandbox; the SDK's exec takes
        # a shell-string and runs it under bash by default.
        shell_cmd = f"bash -l -c {_shquote(cmd)}" if login else f"bash -c {_shquote(cmd)}"

        start = time.monotonic()
        # TODO(daytona-api-verify): exact signature of process.exec varies —
        # docs show ``exec(command=...)``; some versions also accept
        # ``timeout=...``. Pass both via kwargs and let unsupported kwargs
        # raise so users learn about mismatches early.
        try:
            response = self._sandbox.process.exec(command=shell_cmd, timeout=timeout)
        except TypeError:
            # Older SDK signature without timeout kwarg.
            response = self._sandbox.process.exec(command=shell_cmd)

        # TODO(daytona-api-verify): response attribute names. The docs show
        # ``response.result`` (stdout) and ``response.exit_code``. Some SDK
        # versions expose ``stdout``/``output`` instead. Try the most common
        # in order.
        output = getattr(response, "result", None) or getattr(response, "stdout", None) or getattr(response, "output", None) or ""
        exit_code = getattr(response, "exit_code", None)
        if exit_code is None:
            exit_code = getattr(response, "returncode", None)
        if exit_code is None:
            exit_code = 0

        # Annotate slow calls in the captured output so users notice when
        # they're hitting the SDK timeout vs. a fast-fail.
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            output = f"{output}\n[Daytona exec hit {timeout}s timeout]"
            exit_code = 124

        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return _BlockingResultHandle(output=output, exit_code=int(exit_code))

    # ── cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Delete the sandbox + clear cached client state.

        Best-effort: SDK delete failures are swallowed (logged would need a
        logger, deferred to the agent-side logging layer). If the SDK isn't
        even importable any more — e.g. uninstalled mid-session — we silently
        drop the handle.
        """
        if self._sandbox is not None:
            try:
                # TODO(daytona-api-verify): client-level delete vs. sandbox-level.
                # Some versions expose ``sandbox.delete()``; others want
                # ``client.delete(sandbox)``. Try sandbox-level first.
                delete_fn = getattr(self._sandbox, "delete", None)
                if delete_fn is not None:
                    delete_fn()
                elif self._client is not None:
                    client_delete = getattr(self._client, "delete", None) or getattr(self._client, "remove", None)
                    if client_delete is not None:
                        client_delete(self._sandbox)
            except Exception:
                pass
            self._sandbox = None
        self._client = None
        self._initialized = False


# ── helpers ───────────────────────────────────────────────────────────


def _shquote(s: str) -> str:
    """Single-quote-safe shell quoting (POSIX).

    Re-inlined from stdlib ``shlex.quote`` to keep this module's import
    surface flat (and to make the SDK-vs-stdlib boundary obvious in diffs).
    """
    import shlex

    return shlex.quote(s)


__all__ = ["DaytonaEnvironment"]
