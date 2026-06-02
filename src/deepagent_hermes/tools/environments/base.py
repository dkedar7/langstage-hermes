"""`BaseEnvironment` — ABC for terminal-execution backends (SPEC §12).

Six concrete backends slot in behind this protocol: ``LocalEnvironment`` (full
impl in v0.1.0), plus stubs for Docker, SSH, Daytona, Modal, Singularity.

The model mirrors Hermes's ``tools/environments/base.py`` design exactly:
spawn-per-call, with a one-shot **session snapshot** capturing env vars +
functions + aliases + shellopts at init, and a CWD marker file the wrapped
bash script writes after every command so directory changes persist across
calls. See SPEC §12 for the rationale (TL;DR: stateless processes simplify
backends, the snapshot gives back the "stateful shell" feel for free).

This module imports only stdlib + ``typing`` — no langchain / deepagents / SDK
deps — so tests can exercise the ABC and the dataclass without a full agent
install.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol, runtime_checkable

# CWD-marker / snapshot prefixes are namespaced to avoid colliding with a
# parallel Hermes install on the same machine (snapshots live in TEMPDIR).
_SNAP_PREFIX = "deepagent-hermes-snap"
_CWD_PREFIX = "deepagent-hermes-cwd"


# ── ProcessHandle protocol ────────────────────────────────────────────


@runtime_checkable
class ProcessHandle(Protocol):
    """Duck type every backend's ``_run_bash()`` must return.

    ``subprocess.Popen`` satisfies this natively. SDK backends (Modal,
    Daytona) typically wrap their blocking ``exec`` calls in a threaded
    adapter exposing the same surface (see Hermes's ``_ThreadedProcessHandle``
    in the reference implementation).
    """

    def poll(self) -> int | None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = ...) -> int: ...

    @property
    def stdout(self) -> IO[str] | None: ...

    @property
    def returncode(self) -> int | None: ...


# ── ExecuteResponse dataclass ─────────────────────────────────────────


@dataclass
class ExecuteResponse:
    """Result of a single ``execute()`` call.

    Attributes:
        output: Combined stdout+stderr captured from the child process.
            Already had the CWD marker stripped if one was present.
        exit_code: Process exit code; ``-1`` if the process didn't return a
            real code (timeout / interrupt path).
        truncated: Whether the captured output was truncated to fit a size
            limit. v0.1.0 always sets this to ``False``; reserved for future
            output-truncation middleware.
        duration_ms: Wall-clock time from ``execute()`` entry to return,
            in milliseconds.
    """

    output: str
    exit_code: int
    truncated: bool = False
    duration_ms: float = 0.0


# ── BaseEnvironment ABC ───────────────────────────────────────────────


class BaseEnvironment(ABC):
    """Protocol for terminal-execution backends.

    Subclasses MUST implement :meth:`_run_bash` and :meth:`cleanup`. The base
    class supplies the unified ``init_session`` / ``execute`` / ``get_cwd``
    flow that makes the six backends behave identically from the agent's POV.

    The instantiation model is **per-session**: the agent factory creates one
    ``BaseEnvironment`` per ``session_id`` (== langgraph thread id), reuses it
    for every terminal call, and ``cleanup()``s it on session end.
    """

    # Hook for backends that need to embed stdin as a heredoc inside the
    # command itself (Modal, Daytona) rather than piping. Default = pipe.
    _stdin_mode: str = "pipe"

    # Snapshot creation gets its own timeout — slow cold-starts (Docker pull,
    # Modal sandbox warm-up) shouldn't be capped at the per-command default.
    _snapshot_timeout: int = 30

    def __init__(self, session_id: str) -> None:
        """Store ``session_id`` and pre-compute the per-session artifact paths.

        ``init_session()`` must be called before the first ``execute()``;
        we don't auto-invoke it in ``__init__`` because some backends
        (Docker, Modal) want explicit control over when the container /
        sandbox warm-up happens.
        """
        self.session_id = session_id
        self._initialized = False
        # Subclasses override cwd via ``execute(cwd=...)``; default to
        # whatever Python sees at instantiation time.
        self.cwd: str = os.getcwd()

    # ── abstract surface (must be overridden) ─────────────────────────

    @abstractmethod
    def _run_bash(
        self,
        cmd: str,
        *,
        login: bool = False,
        timeout: int = 60,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Spawn a bash process to run ``cmd``.

        Backends are free to use ``subprocess.Popen`` (local, docker, ssh) or
        an SDK adapter (modal, daytona). The returned object must satisfy the
        :class:`ProcessHandle` protocol.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release backend resources (container, instance, SSH connection)."""
        ...

    # ── session snapshot ──────────────────────────────────────────────

    def _snapshot_path(self) -> Path:
        """Return the on-host path where the env-var / function snapshot lives.

        Uses the host's temp dir so we don't pollute the user's home / cwd.
        Filename is ``<prefix>-<session_id>.sh`` for parallel-session safety.
        """
        return Path(tempfile.gettempdir()) / f"{_SNAP_PREFIX}-{self.session_id}.sh"

    def _cwd_path(self) -> Path:
        """Return the on-host path where bash writes ``pwd -P`` after each call."""
        return Path(tempfile.gettempdir()) / f"{_CWD_PREFIX}-{self.session_id}.txt"

    def init_session(self) -> None:
        """Capture env vars / functions / aliases / shellopts into a snapshot.

        Idempotent: returns immediately if already initialized. On failure the
        snapshot is left ``_initialized = False`` so subsequent ``execute()``
        calls fall back to a login shell.
        """
        if self._initialized:
            return

        snap = self._snapshot_path()
        cwd_file = self._cwd_path()

        # The bash snippet that captures session state. Mirrors Hermes's
        # bootstrap exactly: export -p (env), declare -f (functions, filtered
        # to drop bash internals starting with `_<not_underscore>`), alias -p,
        # and a couple of shopts so the resulting source-able script doesn't
        # bail on undefined vars or fail-fast on unrelated errors when
        # re-sourced before each command.
        snap_q = shlex.quote(str(snap))
        cwd_q = shlex.quote(str(cwd_file))
        bootstrap = (
            f"export -p > {snap_q}\n"
            f"declare -f | grep -vE '^_[^_]' >> {snap_q}\n"
            f"alias -p >> {snap_q} 2>/dev/null || true\n"
            f"echo 'shopt -s expand_aliases' >> {snap_q}\n"
            f"echo 'set +e' >> {snap_q}\n"
            f"echo 'set +u' >> {snap_q}\n"
            f"pwd -P > {cwd_q} 2>/dev/null || true\n"
        )

        try:
            proc = self._run_bash(bootstrap, login=True, timeout=self._snapshot_timeout)
            # Drain output so the pipe doesn't deadlock; ignore content.
            try:
                proc.wait(timeout=self._snapshot_timeout)
            except Exception:
                pass
            self._initialized = True
        except Exception:
            # Snapshot creation is best-effort. Failure just means subsequent
            # execute() calls run with `bash -l` for full profile loading.
            self._initialized = False

    # ── command wrapping ──────────────────────────────────────────────

    def _wrap_command(self, command: str) -> str:
        """Wrap a user command so snapshot is sourced, cwd is restored, and pwd is recorded.

        The shape::

            source <snap> ; cd "$(cat <cwd> 2>/dev/null || echo .)" ;
            eval '<command>' ; __ec=$? ; pwd -P > <cwd> ; exit $__ec

        ``eval`` keeps complex one-liners (pipes, redirections, glob
        expansion) working exactly as the user typed them. The exit code
        from the user's command is the script's exit code.
        """
        snap_q = shlex.quote(str(self._snapshot_path()))
        cwd_q = shlex.quote(str(self._cwd_path()))
        # Single-quote-safe escape: ' -> '\''
        escaped = command.replace("'", "'\\''")

        parts: list[str] = []
        if self._initialized:
            parts.append(f"source {snap_q} >/dev/null 2>&1 || true")
        # Restore cwd from the marker file; fall back to "." when missing so
        # the first command after init still runs cleanly.
        parts.append(f'cd "$(cat {cwd_q} 2>/dev/null || echo .)" 2>/dev/null || true')
        parts.append(f"eval '{escaped}'")
        parts.append("__deepagent_ec=$?")
        # Re-dump env vars so ``export FOO=bar`` in this command is visible
        # to subsequent commands when they re-source the snapshot. Mirrors
        # Hermes's "last-writer-wins" behavior — for concurrent sessions the
        # session_id namespacing on the snapshot path keeps them disjoint.
        if self._initialized:
            parts.append(f"export -p > {snap_q} 2>/dev/null || true")
        parts.append(f"pwd -P > {cwd_q} 2>/dev/null || true")
        parts.append("exit $__deepagent_ec")
        return "\n".join(parts)

    # ── execution ─────────────────────────────────────────────────────

    def execute(
        self,
        command: str,
        *,
        timeout: int = 60,
        stdin_data: str | None = None,
    ) -> ExecuteResponse:
        """Run ``command`` end-to-end and return an :class:`ExecuteResponse`.

        Implementation notes:
          - Auto-calls ``init_session()`` on first invocation.
          - Wraps the command with snapshot sourcing + cwd persistence.
          - Falls back to ``bash -l`` (login shell) when the snapshot is
            unavailable, so the user's profile still loads.
          - Times out after ``timeout`` seconds; on timeout the child is
            killed and ``exit_code = 124`` (GNU ``timeout`` convention).
        """
        start = time.monotonic()
        self.init_session()

        wrapped = self._wrap_command(command)
        login = not self._initialized

        try:
            proc = self._run_bash(
                wrapped, login=login, timeout=timeout, stdin_data=stdin_data
            )
        except FileNotFoundError as exc:
            return ExecuteResponse(
                output=f"[error] {exc}",
                exit_code=127,
                truncated=False,
                duration_ms=(time.monotonic() - start) * 1000.0,
            )

        output, exit_code = self._drain(proc, timeout)
        # Refresh tracked cwd from marker file after each command. Done after
        # drain so we read the post-command pwd, not the pre-command one.
        try:
            cwd_text = self._cwd_path().read_text(encoding="utf-8").strip()
            if cwd_text:
                self.cwd = cwd_text
        except OSError:
            pass

        duration_ms = (time.monotonic() - start) * 1000.0
        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=False,
            duration_ms=duration_ms,
        )

    # ── process wait/drain (subclasses may override for select-based draining) ─

    def _drain(self, proc: ProcessHandle, timeout: int) -> tuple[str, int]:
        """Wait for ``proc`` to exit (or timeout), returning ``(output, exit_code)``.

        Default implementation uses ``subprocess.Popen.communicate`` because
        it's portable across Linux/macOS/Windows. The select-based draining
        Hermes uses on POSIX (to defang grandchild-pipe-holding via
        ``setsid + & disown``) is a nice-to-have for the local backend that
        wraps long-running daemons; we defer it to a backend-specific
        override rather than putting it in the base class.
        """
        # Fast path: real subprocess.Popen → use communicate() with a deadline
        # so we don't deadlock on output pipe back-pressure.
        if isinstance(proc, subprocess.Popen):
            try:
                stdout_b, _ = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                # Best-effort drain of whatever was buffered.
                try:
                    stdout_b, _ = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    stdout_b = b"" if isinstance(proc.stdout, type(None)) else b""
                output = _decode(stdout_b) + f"\n[Command timed out after {timeout}s]"
                return output, 124
            return _decode(stdout_b), proc.returncode if proc.returncode is not None else -1

        # Generic ProcessHandle path: poll loop with interrupt-safety.
        deadline = time.monotonic() + timeout
        interrupted = threading.Event()
        chunks: list[str] = []

        # Spawn a drain thread so blocking readline() on a long-running
        # command doesn't starve our poll loop.
        def _read():
            try:
                stream = proc.stdout
                if stream is None:
                    return
                for line in stream:
                    if interrupted.is_set():
                        return
                    chunks.append(line if isinstance(line, str) else line.decode("utf-8", "replace"))
            except Exception:
                pass

        t = threading.Thread(target=_read, daemon=True)
        t.start()

        while proc.poll() is None:
            if time.monotonic() > deadline:
                interrupted.set()
                try:
                    proc.kill()
                except Exception:
                    pass
                t.join(timeout=2)
                return ("".join(chunks) + f"\n[Command timed out after {timeout}s]", 124)
            time.sleep(0.05)

        t.join(timeout=2)
        exit_code = proc.returncode if proc.returncode is not None else -1
        return ("".join(chunks), exit_code)

    # ── public helpers ────────────────────────────────────────────────

    def get_cwd(self) -> str:
        """Return the most-recently-recorded working directory.

        Reads the CWD marker file (written by the wrapped command) on every
        call so multiple ``BaseEnvironment`` instances sharing a session_id
        observe the same state.
        """
        try:
            text = self._cwd_path().read_text(encoding="utf-8").strip()
            if text:
                self.cwd = text
        except OSError:
            pass
        return self.cwd

    @staticmethod
    def new_session_id() -> str:
        """Convenience helper for callers that want a fresh id quickly."""
        return uuid.uuid4().hex[:12]


# ── helpers ────────────────────────────────────────────────────────────


def _decode(data: bytes | str | None) -> str:
    """Best-effort UTF-8 decode that never raises."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive
        return ""


__all__ = [
    "BaseEnvironment",
    "ProcessHandle",
    "ExecuteResponse",
]
