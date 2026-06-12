"""`SshEnvironment` — runs commands on a remote host over a persistent SSH transport.

Design notes
------------

* **Long-lived transport.** A single ``paramiko.SSHClient`` is opened at
  :meth:`init_session` and reused for every :meth:`execute` call. Each command
  goes through ``client.exec_command(wrapped_bash)`` which spawns a fresh
  remote bash on the same underlying TCP connection — much cheaper than the
  spawn-per-call ``ssh user@host ...`` model the reference Hermes uses
  (``hermes-agent/tools/environments/ssh.py``) because we skip the
  fork+OpenSSH-client startup on every step.

* **Snapshot pattern.** Same as :class:`LocalEnvironment` (see
  ``base.py::_wrap_command``), except the snapshot script and CWD marker file
  live on the *remote* host under ``/tmp``. The snapshot is sourced before
  every command and re-dumped after, so env vars / functions / aliases
  persist across calls just like a sticky shell.

* **Config via env vars.** No constructor sprawl — pull connection details
  from ``DEEPAGENT_HERMES_SSH_*`` (host/key/password/timeout). Constructor
  kwargs override env, which makes the rare programmatic-config case ergonomic
  without forcing every test or call site to thread six parameters through.

* **Lazy paramiko import.** ``paramiko`` is behind the ``[ssh]`` extra so the
  base install stays slim. We try-import at module load and raise an
  informative ``ImportError`` in ``__init__`` when it's missing — failing at
  construction is more user-friendly than blowing up halfway through a tool
  call.

* **Reconnect on broken pipe.** SSH transports occasionally drop on idle or
  network blips. We catch ``paramiko.SSHException`` (and ``EOFError``,
  ``OSError`` for the same failure mode at lower layers) on
  ``exec_command``, reopen the client once, and retry. One retry only — if
  the reconnect itself also fails we surface the error rather than looping.
"""

from __future__ import annotations

import io
import os
import shlex
import sys
import threading
import time
from pathlib import PurePosixPath
from typing import IO

from langstage_hermes.tools.environments.base import BaseEnvironment, ProcessHandle

# Lazy import — paramiko is optional. We sniff at module load so tests can
# patch ``sys.modules['paramiko'] = None`` and exercise the missing-dep path.
try:  # pragma: no cover - import guard
    import paramiko as _paramiko  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - import guard
    _paramiko = None  # type: ignore[assignment]


_INSTALL_HINT = "paramiko is required for SshEnvironment. Install with: pip install 'langstage-hermes[ssh]'"


# ── ProcessHandle adapter ─────────────────────────────────────────────


class _SshProcessHandle:
    """Adapter that wraps a paramiko ``Channel`` in the ProcessHandle protocol.

    ``BaseEnvironment._drain`` only needs ``poll`` / ``kill`` / ``wait`` /
    ``stdout`` / ``returncode``. Paramiko's ``Channel`` already exposes
    ``recv_exit_status`` (blocking) and ``exit_status_ready`` (non-blocking),
    plus a ``ChannelFile`` for stdout — wiring them up is just a thin shim.

    We collect stdout+stderr into a single ``io.BytesIO`` for the base class's
    drain loop. Reading is done on a daemon thread so a long-running remote
    command doesn't starve the polling loop — same pattern as
    ``LocalEnvironment._pipe_stdin``.
    """

    def __init__(self, channel, stdin=None, stdout=None, stderr=None, stdin_data: str | None = None):
        self._channel = channel
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self._returncode: int | None = None
        self._buffer = io.BytesIO()
        self._buffer_lock = threading.Lock()

        # Pump stdin if the caller passed data. Daemon thread so we don't
        # block on a remote process that's not reading.
        if stdin_data is not None and stdin is not None:
            self._pipe_stdin(stdin, stdin_data)

        # Drain stdout + stderr into our buffer in the background. The base
        # class's _drain expects a readable .stdout, so we expose self as the
        # file-like via the ``stdout`` property below.
        self._drain_thread = threading.Thread(target=self._drain_streams, daemon=True)
        self._drain_thread.start()

    @staticmethod
    def _pipe_stdin(stdin: IO[bytes], data: str) -> None:
        def _write() -> None:
            try:
                raw = data.encode("utf-8") if isinstance(data, str) else data
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

    def _drain_streams(self) -> None:
        """Copy remote stdout/stderr into our internal buffer."""
        try:
            if self._stdout is not None:
                for chunk in iter(lambda: self._stdout.read(4096), b""):
                    if not chunk:
                        break
                    with self._buffer_lock:
                        self._buffer.write(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8", "replace"))
        except Exception:
            pass
        try:
            if self._stderr is not None:
                data = self._stderr.read()
                if data:
                    with self._buffer_lock:
                        self._buffer.write(data if isinstance(data, bytes) else data.encode("utf-8", "replace"))
        except Exception:
            pass

    # ── ProcessHandle protocol ────────────────────────────────────────

    def poll(self) -> int | None:
        if self._returncode is not None:
            return self._returncode
        try:
            if self._channel.exit_status_ready():
                self._returncode = int(self._channel.recv_exit_status())
                return self._returncode
        except Exception:
            return None
        return None

    def kill(self) -> None:
        try:
            self._channel.close()
        except Exception:
            pass
        # If close() didn't surface an exit code, mark as killed so callers
        # don't loop forever waiting on poll().
        if self._returncode is None:
            self._returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            rc = self.poll()
            if rc is not None:
                # Make sure the background drain finishes too so we return
                # complete output. Cap at 1s — buffered remote output should
                # be flushed almost immediately after exit.
                self._drain_thread.join(timeout=1.0)
                return rc
            if deadline is not None and time.monotonic() > deadline:
                # Mirror subprocess.Popen.wait timeout semantics — raise so
                # the base class's drain loop can fall through to kill().
                import subprocess as _sp

                raise _sp.TimeoutExpired(cmd="ssh-exec", timeout=timeout or 0)
            time.sleep(0.05)

    @property
    def stdout(self) -> IO[bytes]:
        """Expose the buffer as a readable stream.

        ``BaseEnvironment._drain`` iterates ``proc.stdout`` line-by-line in
        its generic path. We rewind the buffer and hand it back — by the time
        ``_drain`` reads from us the drain thread has populated whatever is
        available, and the line iterator stops at EOF.

        For the subprocess.Popen fast path the base class never touches this
        property; it goes straight to ``communicate``. Since we're not a
        Popen the generic path runs.
        """
        with self._buffer_lock:
            data = self._buffer.getvalue()
        return io.BytesIO(data)

    @property
    def returncode(self) -> int | None:
        return self._returncode


# ── SshEnvironment ────────────────────────────────────────────────────


def _parse_host(spec: str) -> tuple[str | None, str, int]:
    """Parse ``[user@]host[:port]`` into ``(user, host, port)``.

    Returns ``user=None`` when the spec omits it — caller falls back to the
    current login user, matching OpenSSH's default behavior.
    """
    user: str | None = None
    if "@" in spec:
        user, _, spec = spec.partition("@")
    if ":" in spec:
        host, _, port_s = spec.rpartition(":")
        try:
            port = int(port_s)
        except ValueError:
            host = spec
            port = 22
    else:
        host = spec
        port = 22
    return (user or None), host, port


class SshEnvironment(BaseEnvironment):
    """Run commands on a remote host over a persistent paramiko transport.

    Configuration (constructor kwargs override env vars):
        host:       ``DEEPAGENT_HERMES_SSH_HOST`` — ``[user@]host[:port]``
        key_path:   ``DEEPAGENT_HERMES_SSH_KEY``  — default ``~/.ssh/id_rsa``
        password:   ``DEEPAGENT_HERMES_SSH_PASSWORD`` — fallback only
        timeout:    ``DEEPAGENT_HERMES_SSH_TIMEOUT`` — connect timeout, default 10s
    """

    # SSH commands stream stdin via paramiko, not via Popen pipes.
    _stdin_mode: str = "pipe"

    def __init__(
        self,
        session_id: str,
        *,
        host: str | None = None,
        user: str | None = None,
        port: int | None = None,
        key_path: str | None = None,
        password: str | None = None,
        timeout: float | None = None,
    ) -> None:
        # Fail fast if the optional dep is missing — clearer than a confusing
        # AttributeError later when we reach for ``_paramiko.SSHClient``.
        # We re-check sys.modules so tests can simulate the missing-dep path
        # with ``monkeypatch.setitem(sys.modules, 'paramiko', None)`` without
        # having to actually uninstall paramiko.
        mod = sys.modules.get("paramiko", _paramiko)
        if mod is None:
            raise ImportError(_INSTALL_HINT)
        self._paramiko = mod

        super().__init__(session_id=session_id)

        # Resolve config: kwarg > env > default.
        raw_host = (
            host
            if host is not None
            else (os.environ.get("LANGSTAGE_HERMES_SSH_HOST") or os.environ.get("DEEPAGENT_HERMES_SSH_HOST"))
        )
        if not raw_host:
            raise ValueError("SshEnvironment requires DEEPAGENT_HERMES_SSH_HOST (e.g. 'user@host:22') or the host= kwarg.")

        parsed_user, parsed_host, parsed_port = _parse_host(raw_host)
        self.host = parsed_host
        self.user = user or parsed_user or os.environ.get("USER") or os.environ.get("USERNAME") or "root"
        self.port = int(port) if port is not None else parsed_port

        self.key_path = (
            key_path
            if key_path is not None
            else (
                os.environ.get("LANGSTAGE_HERMES_SSH_KEY")
                or os.environ.get("DEEPAGENT_HERMES_SSH_KEY", os.path.expanduser("~/.ssh/id_rsa"))
            )
        )
        self.password = (
            password
            if password is not None
            else (os.environ.get("LANGSTAGE_HERMES_SSH_PASSWORD") or os.environ.get("DEEPAGENT_HERMES_SSH_PASSWORD"))
        )

        if timeout is not None:
            self.timeout = float(timeout)
        else:
            env_to = os.environ.get("LANGSTAGE_HERMES_SSH_TIMEOUT") or os.environ.get("DEEPAGENT_HERMES_SSH_TIMEOUT")
            self.timeout = float(env_to) if env_to else 10.0

        self._client = None  # set on first init_session / execute
        self._connect_lock = threading.Lock()

    # ── snapshot / cwd path overrides (remote tmp, not host tmp) ──────

    def _snapshot_path(self) -> PurePosixPath:  # type: ignore[override]
        """Remote path — always POSIX, lives in ``/tmp`` on the SSH target.

        Overrides ``BaseEnvironment._snapshot_path`` which would otherwise
        return a host-local ``Path`` under ``tempfile.gettempdir()``. The
        wrapper script needs the remote path so ``source`` works on the
        other side.
        """
        return PurePosixPath(f"/tmp/langstage-hermes-snap-{self.session_id}.sh")

    def _cwd_path(self) -> PurePosixPath:  # type: ignore[override]
        """Remote CWD marker — POSIX path in remote ``/tmp``."""
        return PurePosixPath(f"/tmp/langstage-hermes-cwd-{self.session_id}.txt")

    # ── connection management ─────────────────────────────────────────

    def _connect(self):
        """Open (or reopen) the SSH client. Holds a lock to serialize reconnects."""
        with self._connect_lock:
            # Tear down any half-dead client first.
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

            client = self._paramiko.SSHClient()
            # AutoAddPolicy mirrors what OpenSSH does with
            # StrictHostKeyChecking=accept-new (first-seen-trust). Production
            # deployments should swap this for a known_hosts-backed policy.
            try:
                client.set_missing_host_key_policy(self._paramiko.AutoAddPolicy())
            except Exception:
                pass

            connect_kwargs: dict = {
                "hostname": self.host,
                "port": self.port,
                "username": self.user,
                "timeout": self.timeout,
                # Disable banner_timeout's interactive prompts; we want fast failure.
                "allow_agent": True,
                "look_for_keys": True,
            }
            if self.key_path and os.path.isfile(self.key_path):
                connect_kwargs["key_filename"] = self.key_path
            if self.password:
                connect_kwargs["password"] = self.password

            client.connect(**connect_kwargs)
            self._client = client
            return client

    def _ensure_client(self):
        """Return the live client, opening one if we don't have it yet."""
        if self._client is None:
            return self._connect()
        return self._client

    # ── execute / get_cwd overrides (CWD marker lives remotely) ───────

    def execute(self, command: str, *, timeout: int = 60, stdin_data: str | None = None):
        """Run ``command`` end-to-end, then refresh ``self.cwd`` via a remote read.

        We mirror ``BaseEnvironment.execute`` exactly but skip its post-run
        ``self._cwd_path().read_text(...)`` step — the marker lives on the
        remote host, not on the local filesystem, so a Path.read_text() call
        would either find nothing or (worse, on Windows) collide with an
        unrelated local ``/tmp`` file. Instead we ``cat`` the remote marker
        through the same SSH client.
        """
        import time as _time

        from langstage_hermes.tools.environments.base import ExecuteResponse

        start = _time.monotonic()
        self.init_session()

        wrapped = self._wrap_command(command)
        login = not self._initialized

        try:
            proc = self._run_bash(wrapped, login=login, timeout=timeout, stdin_data=stdin_data)
        except FileNotFoundError as exc:
            return ExecuteResponse(
                output=f"[error] {exc}",
                exit_code=127,
                truncated=False,
                duration_ms=(_time.monotonic() - start) * 1000.0,
            )

        output, exit_code = self._drain(proc, timeout)
        self._refresh_remote_cwd()

        duration_ms = (_time.monotonic() - start) * 1000.0
        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=False,
            duration_ms=duration_ms,
        )

    def get_cwd(self) -> str:
        """Return the most recent remote working directory."""
        self._refresh_remote_cwd()
        return self.cwd

    def _refresh_remote_cwd(self) -> None:
        """Best-effort: read the remote CWD marker via a short exec_command."""
        if self._client is None:
            return
        try:
            cwd_q = shlex.quote(str(self._cwd_path()))
            _stdin, stdout, _stderr = self._client.exec_command(f"cat {cwd_q} 2>/dev/null || true", timeout=5)
            data = stdout.read()
            text = data.decode("utf-8", "replace").strip() if isinstance(data, bytes) else str(data).strip()
            if text:
                self.cwd = text
        except Exception:
            # Best-effort — never fail an execute() because we couldn't
            # refresh cached CWD state.
            pass

    # ── command execution ─────────────────────────────────────────────

    def _exec(self, wrapped: str, *, timeout: float | None, stdin_data: str | None):
        """Run ``exec_command`` with one reconnect-on-broken-pipe retry.

        Reconnect is gated on ``paramiko.SSHException`` (covers broken
        transports, channel-open failures, auth-key rotation) plus the
        lower-level ``EOFError`` / ``OSError`` paramiko sometimes raises when
        the TCP connection is reset mid-call.
        """
        client = self._ensure_client()
        try:
            return client.exec_command(wrapped, timeout=timeout)
        except (self._paramiko.SSHException, EOFError, OSError):
            # One retry — full reconnect, then re-run. If this fails too the
            # exception propagates; the base class converts it into an
            # ExecuteResponse with a non-zero exit code via the FileNotFoundError
            # path... actually no, only FileNotFoundError gets that treatment;
            # other exceptions bubble up to the caller. That's fine — a
            # repeated SSH failure isn't a "user typo", it's an outage.
            client = self._connect()
            return client.exec_command(wrapped, timeout=timeout)

    def _run_bash(
        self,
        cmd: str,
        *,
        login: bool = False,
        timeout: int = 60,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        """Run ``cmd`` on the remote host via a fresh ``exec_command`` channel.

        We always wrap in ``bash -c`` (or ``bash -l -c`` for login) so the
        snapshot-sourcing script the base class hands us runs under a real
        bash — the user's default remote shell could be zsh/fish/dash and the
        snapshot script uses bash-isms (``declare -f``, ``export -p``).
        """
        # The base class already wraps with source/cd/eval/pwd. We just need to
        # ensure it runs under bash. shlex.quote keeps embedded single quotes safe.
        if login:
            remote_cmd = f"bash -l -c {shlex.quote(cmd)}"
        else:
            remote_cmd = f"bash -c {shlex.quote(cmd)}"

        stdin, stdout, stderr = self._exec(remote_cmd, timeout=float(timeout), stdin_data=stdin_data)
        channel = stdout.channel

        return _SshProcessHandle(
            channel=channel,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            stdin_data=stdin_data,
        )

    # ── cleanup ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Best-effort: rm snapshot artifacts on the remote, then close the client.

        Snapshot cleanup is best-effort because the remote host may already be
        unreachable when we tear down (network outage, ctrl-C on the agent).
        Closing the client is the important part — it releases the TCP socket
        and any agent-forwarded sockets paramiko opened.
        """
        if self._client is not None:
            try:
                snap = shlex.quote(str(self._snapshot_path()))
                cwd = shlex.quote(str(self._cwd_path()))
                try:
                    self._client.exec_command(f"rm -f {snap} {cwd}", timeout=5)
                except Exception:
                    pass
            finally:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None


__all__ = ["SshEnvironment"]
