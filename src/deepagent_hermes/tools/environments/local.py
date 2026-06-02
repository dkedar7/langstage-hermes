"""`LocalEnvironment` — runs commands directly on the host via ``subprocess.Popen``.

Full implementation of the :class:`BaseEnvironment` protocol for v0.1.0.

**Windows note (carry-over from Hermes):** the wrapper script the base class
emits is plain bash — relies on ``source``, ``cd``, ``eval``, and ``pwd -P``.
On Windows that means we need Git Bash (or any other ``bash.exe`` on PATH) to
execute it. The shipping recommendation is the same as Hermes's: install Git
for Windows. When ``bash`` is not on PATH, ``execute()`` returns a structured
error rather than crashing; the test fixture skips with a clear reason.

The PowerShell fallback the SPEC mentions (translate the snapshot to PS1)
isn't implemented in v0.1.0 — it's a much larger lift than it looks because
bash's ``export -p`` / ``declare -f`` output has no clean PowerShell analog.
Documenting the bash dependency is the pragmatic answer for now.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from deepagent_hermes.tools.environments.base import BaseEnvironment, ProcessHandle


def _find_bash() -> str | None:
    """Locate ``bash`` on the host, returning ``None`` when it's not available.

    Search order:
      1. ``$HERMES_BASH_PATH`` env override (parity with upstream's
         ``HERMES_GIT_BASH_PATH`` knob, but namespaced).
      2. ``shutil.which("bash")`` — picks up Git Bash on Windows via PATH.
      3. POSIX hard-coded paths (``/usr/bin/bash``, ``/bin/bash``).
      4. Windows fallback: standard Git for Windows install locations.
    """
    override = os.environ.get("DEEPAGENT_HERMES_BASH_PATH") or os.environ.get(
        "HERMES_BASH_PATH"
    )
    if override and Path(override).is_file():
        return override

    found = shutil.which("bash")
    if found:
        return found

    if sys.platform == "win32":
        candidates = [
            os.path.join(
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                "Git", "bin", "bash.exe",
            ),
            os.path.join(
                os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                "Git", "bin", "bash.exe",
            ),
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""),
                "Programs", "Git", "bin", "bash.exe",
            ),
        ]
        for c in candidates:
            if c and Path(c).is_file():
                return c
        return None

    # POSIX last-resort.
    for path in ("/usr/bin/bash", "/bin/bash"):
        if Path(path).is_file():
            return path
    return None


class LocalEnvironment(BaseEnvironment):
    """Run commands on the local host via ``subprocess.Popen``.

    Snapshot + cwd marker live in the host's temp dir (see base class).
    On Windows this requires a usable ``bash.exe`` — Git for Windows is the
    standard answer. When bash is missing, :meth:`_run_bash` raises
    ``FileNotFoundError`` and ``BaseEnvironment.execute`` reports it as
    exit_code 127.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id=session_id)
        # Resolve bash eagerly so the failure mode is clear at construction
        # time, not buried inside the first execute() call. ``None`` is OK
        # here — the test fixture skips before init_session() runs.
        self._bash_path = _find_bash()

    # ── bash spawn ────────────────────────────────────────────────────

    def _run_bash(
        self,
        cmd: str,
        *,
        login: bool = False,
        timeout: int = 60,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Spawn ``bash -c cmd`` (or ``bash -l -c`` for login mode).

        Returns a ``subprocess.Popen`` instance, which satisfies the
        :class:`ProcessHandle` protocol natively.

        ``stdin_data`` is written asynchronously via a daemon thread to avoid
        pipe-buffer deadlocks. On Windows we open stdin in binary mode and
        encode ourselves to bypass the text-mode ``\\n`` -> ``\\r\\n``
        translation (the same bug Hermes hit on every write_file call).
        """
        bash = self._bash_path or _find_bash()
        if not bash:
            raise FileNotFoundError(
                "bash not found on PATH. On Windows install Git for Windows "
                "(https://git-scm.com/download/win) or set "
                "DEEPAGENT_HERMES_BASH_PATH=<bash.exe>."
            )

        args: list[str] = [bash, "-l", "-c", cmd] if login else [bash, "-c", cmd]

        # Windows-specific kwargs: hide the console window so terminal calls
        # don't flash a cmd prompt every time. ``CREATE_NO_WINDOW`` is
        # documented stable since 3.7.
        popen_kwargs: dict = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
        else:
            # POSIX: put the child in its own process group so we can kill
            # the whole tree (background processes + grandchildren) on
            # timeout / interrupt. Mirrors Hermes's setsid trick.
            popen_kwargs["preexec_fn"] = os.setsid  # type: ignore[attr-defined]

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            text=False,  # we handle decoding ourselves to dodge CRLF translation
            **popen_kwargs,
        )

        if stdin_data is not None:
            self._pipe_stdin(proc, stdin_data)

        return proc

    # ── stdin pipe helper ─────────────────────────────────────────────

    @staticmethod
    def _pipe_stdin(proc: subprocess.Popen, data: str | bytes) -> None:
        """Write ``data`` to ``proc.stdin`` on a daemon thread.

        Daemon-threaded to avoid blocking the main thread when the child
        isn't reading stdin fast enough. Errors (BrokenPipeError, OSError)
        are swallowed — the child may have legitimately exited before we
        got around to writing, in which case the write is a no-op.
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

    # ── cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Remove the per-session snapshot + cwd marker files.

        Local doesn't hold container / SDK resources so cleanup is just
        scrubbing the host-side artifacts.
        """
        for path in (self._snapshot_path(), self._cwd_path()):
            try:
                path.unlink()
            except (FileNotFoundError, OSError):
                pass


__all__ = ["LocalEnvironment", "_find_bash"]
