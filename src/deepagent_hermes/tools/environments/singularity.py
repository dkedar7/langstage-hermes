"""Singularity / Apptainer terminal backend — real subprocess implementation.

Singularity (now upstreamed as Apptainer after the 2021 governance fork) is a
CLI tool, not an SDK, so this backend is just a thin wrapper around
``singularity exec --bind <ws>:/workspace <image> bash -c '...'``. Same
spawn-per-call snapshot pattern as :class:`LocalEnvironment`, but each bash
invocation runs inside a container.

Both binary names are accepted — ``shutil.which("singularity")`` first, then
``apptainer`` — because the project rename happened mid-deployment and most
HPC sites still have either binary on PATH (often both, with one as an alias
for the other).

**Image:** controlled via ``DEEPAGENT_HERMES_SINGULARITY_IMAGE`` (default
``docker://python:3.13-slim`` — Singularity / Apptainer transparently pulls
Docker images). Can also be a path to a local ``.sif`` file.

**Workspace bind:** the host directory referenced by
``DEEPAGENT_HERMES_SINGULARITY_WORKSPACE`` (default: host tempdir) is bind-mounted
to ``/workspace`` inside the container. The session snapshot + CWD marker live
in this bind mount so they survive across ``singularity exec`` invocations
(each exec is a fresh container — no in-container persistence without an
overlay).

**Out of scope for v0.1.0:** persistent overlays (``--overlay``), instance
mode (``singularity instance start/stop``), and the credential-mount /
file-sync machinery the upstream Hermes backend has. Those layered on top of
real production deployments and were never going to be a dependency-free
out-of-the-box experience anyway. The baseline ``exec``-per-call here gives
you correctness and isolation; persistence is a follow-up.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from deepagent_hermes.tools.environments.base import BaseEnvironment, ProcessHandle

_DEFAULT_IMAGE = "docker://python:3.13-slim"
_WORKSPACE_MOUNT = "/workspace"


def _find_singularity() -> str | None:
    """Return path to ``singularity`` or ``apptainer``, or ``None`` if neither found.

    Apptainer is the post-2021 fork name; most installations expose one or
    the other (sometimes both, often as symlinks).
    """
    for binary in ("singularity", "apptainer"):
        found = shutil.which(binary)
        if found:
            return found
    return None


def _resolve_workspace() -> Path:
    """Resolve the host directory bind-mounted into the container.

    Reads ``DEEPAGENT_HERMES_SINGULARITY_WORKSPACE`` env override; falls back
    to the host tempdir so unconfigured callers still get a working sandbox.
    """
    override = os.environ.get("DEEPAGENT_HERMES_SINGULARITY_WORKSPACE")
    if override:
        path = Path(override)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.gettempdir())


class SingularityEnvironment(BaseEnvironment):
    """Run commands inside a Singularity / Apptainer container, spawn-per-call.

    The snapshot file and CWD marker are written into the host-side workspace
    directory (which is bind-mounted into the container) so they survive
    across exec invocations. We override :meth:`_snapshot_path` and
    :meth:`_cwd_path` to put both artifacts in the bind-mount rather than
    ``/tmp``; that way the base class's ``source <snap>`` / ``cd $(cat <cwd>)``
    wrapper just works inside the container.
    """

    # Container cold-starts (image pull on first exec) can take a while.
    _snapshot_timeout: int = 120

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id=session_id)
        # Resolve binary + image + workspace eagerly so failure modes are
        # surfaced at construction time, not deep inside the first execute().
        # _find_singularity() may return None on hosts without the CLI; we
        # store the None and let _run_bash() raise FileNotFoundError so the
        # base class can format it as a normal ExecuteResponse with exit 127.
        self._singularity = _find_singularity()
        self._image = os.environ.get("DEEPAGENT_HERMES_SINGULARITY_IMAGE", _DEFAULT_IMAGE)
        self._workspace = _resolve_workspace()

    # ── snapshot / cwd paths live inside the bind mount ───────────────

    def _snapshot_path(self) -> Path:
        """Snapshot lives in the host workspace dir (bind-mounted into container).

        The base class default points at the host tempdir, which is NOT visible
        inside the container — so we override to the workspace, which IS
        bind-mounted at ``/workspace``. The wrapped command running inside the
        container then sees this file at ``/workspace/<filename>``.
        """
        return self._workspace / f"deepagent-hermes-snap-{self.session_id}.sh"

    def _cwd_path(self) -> Path:
        return self._workspace / f"deepagent-hermes-cwd-{self.session_id}.txt"

    # ── command wrapping (translate host paths to container paths) ────

    def _wrap_command(self, command: str) -> str:
        """Wrap the user command so it references container-side paths.

        The base class's :meth:`_wrap_command` uses the host filesystem paths
        from ``_snapshot_path()`` / ``_cwd_path()``. Inside the container the
        bind-mounted workspace is at ``/workspace``, so we rewrite the host
        ``self._workspace`` prefix to ``/workspace`` in the wrapped script.
        """
        wrapped = super()._wrap_command(command)
        host_prefix = str(self._workspace).replace("\\", "/")
        # When the snapshot lands in ``C:/.../Temp/...sh`` on Windows or
        # ``/var/folders/.../...sh`` on macOS, the container sees it at
        # ``/workspace/...sh`` instead. Plain string replace is fine because
        # both prefixes are absolute paths that don't collide with command
        # contents in practice.
        return wrapped.replace(host_prefix, _WORKSPACE_MOUNT)

    # ── bash spawn inside container ───────────────────────────────────

    def _run_bash(
        self,
        cmd: str,
        *,
        login: bool = False,
        timeout: int = 60,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Spawn ``singularity exec --bind <ws>:/workspace <image> bash -c <cmd>``.

        ``login`` triggers ``bash -l`` inside the container (full profile
        load). Same stdin-via-daemon-thread pattern as :class:`LocalEnvironment`
        to dodge pipe-buffer deadlocks.
        """
        binary = self._singularity or _find_singularity()
        if not binary:
            raise FileNotFoundError(
                "Neither 'singularity' nor 'apptainer' found on PATH. "
                "Install Apptainer (https://apptainer.org/) or Singularity "
                "(https://sylabs.io/singularity/), or use LocalEnvironment."
            )

        # Build the singularity exec argv.
        host_ws = str(self._workspace)
        bind_spec = f"{host_ws}:{_WORKSPACE_MOUNT}"
        bash_args = ["bash", "-l", "-c", cmd] if login else ["bash", "-c", cmd]
        args = [binary, "exec", "--bind", bind_spec, self._image, *bash_args]

        popen_kwargs: dict = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            # POSIX: own process group so we can kill the whole tree on timeout.
            popen_kwargs["preexec_fn"] = os.setsid  # type: ignore[attr-defined]

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            text=False,
            **popen_kwargs,
        )

        if stdin_data is not None:
            self._pipe_stdin(proc, stdin_data)

        return proc

    @staticmethod
    def _pipe_stdin(proc: subprocess.Popen, data: str | bytes) -> None:
        """Async-write ``data`` to ``proc.stdin`` via a daemon thread.

        Identical to :class:`LocalEnvironment`'s helper — we re-inline rather
        than import to keep the backends decoupled.
        """
        import threading

        def _write() -> None:
            try:
                raw = data.encode("utf-8") if isinstance(data, str) else data
                stdin = proc.stdin
                if stdin is None:
                    return
                try:
                    stdin.write(raw)
                finally:
                    try:
                        stdin.close()
                    except Exception:
                        pass
            except (BrokenPipeError, OSError):
                pass

        threading.Thread(target=_write, daemon=True).start()

    # ── cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Remove the per-session snapshot + cwd marker files from the workspace.

        We don't tear down a long-lived container because the ``exec``-per-call
        model never starts one. The bind-mounted workspace itself is not
        deleted (it may be user-owned or shared across sessions).
        """
        for path in (self._snapshot_path(), self._cwd_path()):
            try:
                path.unlink()
            except (FileNotFoundError, OSError):
                pass


__all__ = ["SingularityEnvironment", "_find_singularity"]
