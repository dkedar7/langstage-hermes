"""`DockerEnvironment` — runs commands inside a per-session Docker container.

Mirrors the Hermes design (``hermes-agent/tools/environments/docker.py``) but
strips it to the smallest workable shape: one container per session, started
lazily on ``init_session()``, ``docker exec``'d for every command, ``docker
stop``'d on ``cleanup()``. Pure ``subprocess`` — no Docker SDK dep.

Design choices:

* **Container-per-session.** Starting a container per ``execute()`` call would
  add ~1s of latency to every command. We pay the start cost once per session
  and amortize it across many commands. ``--rm`` on ``docker run`` means we
  don't need an explicit ``docker rm`` — ``docker stop`` triggers the delete.

* **Snapshot lives inside the container.** The base class wraps every command
  with ``source <snap>; cd <cwd>; eval <cmd>; pwd > <cwd>``. For the local
  backend those files live in the host's tempdir; for Docker we redirect them
  to ``/tmp/`` *inside* the container by overriding ``_snapshot_path()`` and
  ``_cwd_path()`` to return container-side paths. The bash wrapping in the
  base class doesn't care that the paths point at a different filesystem — it
  just shells them in via ``shlex.quote``.

* **CWD readback via ``docker exec cat``.** The base class' ``get_cwd()`` and
  ``execute()`` both read the cwd marker through ``Path.read_text``, which
  would fail because the path is inside the container. We override
  ``execute()`` to copy the cwd value back from the container after each call
  via a quick ``docker exec ... cat /tmp/cwd``.

* **Timeout = 124 convention.** Matches GNU ``timeout`` and the local backend.
  When ``subprocess.run(..., timeout=N)`` fires on ``docker exec``, we kill
  the process group and return exit_code 124.

* **Naming.** ``deepagent-hermes-{session_id}`` truncated to 63 chars (Docker's
  container-name limit). Non-alnum chars in session ids get sanitized to ``-``.

Environment variables consulted at construction time:

* ``DEEPAGENT_HERMES_DOCKER_IMAGE`` — image to run (default ``python:3.13-slim``)
* ``DEEPAGENT_HERMES_DOCKER_WORKSPACE`` — optional host path to bind-mount at
  ``/workspace`` inside the container
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from deepagent_hermes.tools.environments.base import (
    BaseEnvironment,
    ExecuteResponse,
    ProcessHandle,
)

# Docker container-name constraints: 63 chars, [a-zA-Z0-9][a-zA-Z0-9_.-]*
_NAME_PREFIX = "deepagent-hermes-"
_MAX_NAME_LEN = 63
_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")

# Snapshot + cwd marker live INSIDE the container at /tmp/. Namespaced names so
# parallel sessions (with the same /tmp visible inside their own containers
# anyway) keep the same naming scheme as the host-side equivalents.
_CONTAINER_SNAP = "/tmp/deepagent-hermes-snap.sh"
_CONTAINER_CWD = "/tmp/deepagent-hermes-cwd.txt"


def _sanitize_container_name(session_id: str) -> str:
    """Build a Docker-legal container name from ``session_id``.

    Strategy:
      1. Replace any char not in ``[A-Za-z0-9_.-]`` with ``-``.
      2. Prefix with ``deepagent-hermes-``.
      3. Truncate to 63 chars total (Docker's max name length).

    Truncation prefers keeping the prefix + a session-id tail rather than the
    head, so the human-meaningful part of the session id stays visible.
    """
    cleaned = _INVALID_NAME_CHARS.sub("-", session_id) or "session"
    budget = _MAX_NAME_LEN - len(_NAME_PREFIX)
    if len(cleaned) > budget:
        cleaned = cleaned[-budget:]
    return f"{_NAME_PREFIX}{cleaned}"


def _docker_available() -> bool:
    """Return True iff ``docker`` is on PATH and ``docker info`` succeeds.

    Used by tests to skip the whole suite cleanly when the daemon isn't
    reachable. Kept module-level so tests can import it directly.
    """
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0


class DockerEnvironment(BaseEnvironment):
    """Run commands inside a per-session Docker container via ``docker exec``.

    Lifecycle:
      * ``__init__`` resolves image / workspace mount config and computes the
        container name. **No** docker invocation yet — construction stays cheap
        and side-effect free.
      * ``init_session`` (called lazily by the base class on first ``execute``)
        starts the container with ``docker run -d --rm ... sleep infinity``,
        then runs the snapshot bootstrap via ``docker exec``.
      * ``execute`` (inherited) wraps the user command, then ``_run_bash``
        ``docker exec``'s it. We override ``execute`` only to refresh ``self.cwd``
        from the container after each call.
      * ``cleanup`` issues ``docker stop`` (``--rm`` handles deletion).
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id=session_id)

        # Image + optional bind mount come from env vars per the spec. Read
        # once at construction so a test that sets the env then constructs
        # gets a deterministic image, even if the env is later mutated.
        self._image = os.environ.get(
            "DEEPAGENT_HERMES_DOCKER_IMAGE", "python:3.13-slim"
        )
        workspace = os.environ.get("DEEPAGENT_HERMES_DOCKER_WORKSPACE")
        self._workspace_host: str | None = workspace if workspace else None

        self._container_name = _sanitize_container_name(session_id)
        # Track whether *we* started the container so cleanup() doesn't try to
        # stop a container that init_session() never managed to launch.
        self._container_started = False

    # ── snapshot / cwd paths override ────────────────────────────────

    def _snapshot_path(self) -> Path:
        """Return the container-internal path for the snapshot file.

        Base class uses this only via ``shlex.quote(str(...))`` when building
        the bash wrapper, so a ``Path`` pointing at an in-container path is
        fine — it's never opened from the host side.
        """
        return Path(_CONTAINER_SNAP)

    def _cwd_path(self) -> Path:
        """Return the container-internal path for the cwd marker file.

        Same caveat as :meth:`_snapshot_path` — this path is only string-
        formatted into bash wrappers, never opened on the host.
        """
        return Path(_CONTAINER_CWD)

    # ── container lifecycle ──────────────────────────────────────────

    def _start_container(self) -> None:
        """Launch the per-session container via ``docker run -d --rm``.

        Idempotent: if a container with our chosen name is already running
        (e.g. from a previous Python process in the same session id) we skip
        the run and reuse it. This is a convenience for interactive debugging
        and is NOT a cross-process reuse contract — the spec asks for one
        container per session and that's what we provide.
        """
        # Conservative reuse check: if a container with our name already exists
        # (any state), don't try to create another one — docker would error
        # with a name conflict. If it's stopped we let it stay stopped; the
        # caller can ``cleanup()`` and reconstruct with a fresh session id.
        probe = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name=^{re.escape(self._container_name)}$",
             "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            # Already exists — treat as started. ``docker exec`` will fail
            # loudly if it's not actually running, which is the right signal.
            self._container_started = True
            return

        cmd: list[str] = [
            "docker", "run", "-d", "--rm",
            "--name", self._container_name,
        ]
        if self._workspace_host:
            # Resolve to absolute path; Docker rejects relative ``-v`` sources.
            host_abs = os.path.abspath(os.path.expanduser(self._workspace_host))
            cmd.extend(["-v", f"{host_abs}:/workspace"])
        cmd.extend([self._image, "sleep", "infinity"])

        # Image pull on first run can be slow; give it a generous timeout.
        # 180s covers a clean pull of python:3.13-slim on most home connections.
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            # Surface stderr to the caller — without it debugging a missing
            # image / daemon-down condition is a guessing game.
            raise RuntimeError(
                f"docker run failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        self._container_started = True

    def init_session(self) -> None:
        """Start the container, then run the base-class snapshot bootstrap.

        Order matters: the snapshot bootstrap calls ``_run_bash`` which
        ``docker exec``'s into the container, so the container must be up
        before we touch the snapshot. If the container start fails we leave
        ``_initialized = False`` and propagate the error — the base class'
        ``init_session`` only swallows snapshot failures, not container ones.
        """
        if self._initialized:
            return
        if not self._container_started:
            self._start_container()
        super().init_session()

    # ── bash exec ────────────────────────────────────────────────────

    def _run_bash(
        self,
        cmd: str,
        *,
        login: bool = False,
        timeout: int = 60,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """``docker exec`` the wrapped bash command inside our session container.

        Returns a live ``subprocess.Popen`` so the base class' ``_drain``
        fast-path works without modification. The ``-i`` flag is only added
        when ``stdin_data`` is provided — otherwise docker exec attaches a
        TTY-less stdin that some images object to.
        """
        if not self._container_started:
            # init_session() should have started it, but defend against the
            # base class invoking _run_bash before init_session in some future
            # refactor. Better an explicit error here than a confusing
            # "container not found" from docker exec.
            raise RuntimeError(
                "DockerEnvironment._run_bash called before container start; "
                "call init_session() first."
            )

        exec_cmd: list[str] = ["docker", "exec"]
        if stdin_data is not None:
            exec_cmd.append("-i")
        exec_cmd.append(self._container_name)
        # ``login`` is a no-op on the snapshot bootstrap path for Docker —
        # ``bash -l`` inside a slim image just sources /etc/profile, which is
        # typically empty. We still honor it for parity with LocalEnvironment.
        if login:
            exec_cmd.extend(["bash", "-l", "-c", cmd])
        else:
            exec_cmd.extend(["bash", "-c", cmd])

        popen_kwargs: dict = {}
        # On POSIX, put the docker-cli child in its own process group so we
        # can kill the whole tree (the cli + its grandchildren) on timeout.
        # No Windows equivalent we care about — CREATE_NO_WINDOW just hides
        # the console flash.
        import sys as _sys
        if _sys.platform == "win32":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
        else:
            popen_kwargs["preexec_fn"] = os.setsid  # type: ignore[attr-defined]

        proc = subprocess.Popen(
            exec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            text=False,
            **popen_kwargs,
        )

        if stdin_data is not None:
            self._pipe_stdin(proc, stdin_data)

        return proc

    # ── stdin pipe helper (copy of LocalEnvironment's) ───────────────

    @staticmethod
    def _pipe_stdin(proc: subprocess.Popen, data: str | bytes) -> None:
        """Write ``data`` to ``proc.stdin`` on a daemon thread.

        Same pattern as ``LocalEnvironment._pipe_stdin`` — daemonized so a
        slow / non-reading child can't wedge the main thread, errors swallowed
        because the child may have exited before we got around to writing.
        Duplicated rather than imported because pulling LocalEnvironment in
        here would couple the two backends unnecessarily.
        """
        import threading

        def _write() -> None:
            try:
                raw = data.encode("utf-8") if isinstance(data, str) else data
                stdin = proc.stdin
                if stdin is None:
                    return
                try:
                    stdin.write(raw)  # type: ignore[arg-type]
                finally:
                    try:
                        stdin.close()
                    except Exception:
                        pass
            except (BrokenPipeError, OSError):
                pass

        threading.Thread(target=_write, daemon=True).start()

    # ── execute override (cwd readback) ──────────────────────────────

    def execute(
        self,
        command: str,
        *,
        timeout: int = 60,
        stdin_data: str | None = None,
    ) -> ExecuteResponse:
        """Run ``command`` and refresh ``self.cwd`` from the container.

        The base class' ``execute`` reads the cwd marker via
        ``Path.read_text``, which can't see into the container. We let the
        base class do its thing (it'll silently fail the read), then issue a
        small ``docker exec cat`` to pull the real cwd value back.
        """
        resp = super().execute(command, timeout=timeout, stdin_data=stdin_data)
        # Best-effort cwd readback — failure here just means the next
        # ``get_cwd()`` returns the previous value, which is the safe default.
        if self._container_started:
            try:
                cat = subprocess.run(
                    ["docker", "exec", self._container_name,
                     "cat", _CONTAINER_CWD],
                    capture_output=True, text=True, timeout=10,
                )
                if cat.returncode == 0:
                    val = cat.stdout.strip()
                    if val:
                        self.cwd = val
            except (subprocess.SubprocessError, OSError):
                pass
        return resp

    def get_cwd(self) -> str:
        """Return the container's most-recently-recorded working directory.

        Overrides the base class because the cwd marker lives inside the
        container — a ``Path.read_text`` on the host would fail.
        """
        if self._container_started:
            try:
                cat = subprocess.run(
                    ["docker", "exec", self._container_name,
                     "cat", _CONTAINER_CWD],
                    capture_output=True, text=True, timeout=10,
                )
                if cat.returncode == 0:
                    val = cat.stdout.strip()
                    if val:
                        self.cwd = val
            except (subprocess.SubprocessError, OSError):
                pass
        return self.cwd

    # ── cleanup ──────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Stop the per-session container.

        ``--rm`` on ``docker run`` means the container is deleted automatically
        when it stops, so we don't need a follow-up ``docker rm``. Errors are
        logged-and-swallowed: cleanup is best-effort, and a stuck container
        with a unique session-id name doesn't block the process from exiting.

        Idempotent: subsequent ``cleanup()`` calls become no-ops.
        """
        if not self._container_started:
            return

        # Best-effort stop with a 30s outer timeout — the inner ``-t 10`` gives
        # the container 10s to SIGTERM cleanly before docker SIGKILLs it. The
        # outer timeout catches a hung docker CLI itself.
        try:
            subprocess.run(
                ["docker", "stop", "-t", "10", self._container_name],
                capture_output=True,
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError):
            pass
        self._container_started = False
        self._initialized = False


__all__ = [
    "DockerEnvironment",
    "_docker_available",
    "_sanitize_container_name",
]
