"""Tests for :class:`SshEnvironment` — mocked paramiko, no real network I/O.

Strategy: stub the entire ``paramiko`` module via ``MagicMock`` and verify the
SSH environment wires it up correctly — connection params, ``exec_command``
invocation shape, snapshot/cwd marker file paths in the wrapped script,
reconnect-on-broken-pipe, and cleanup. Integration tests against a real SSH
target are skipped behind the ``needs_ssh`` marker; nothing in this file
opens a socket.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ── Helpers ──────────────────────────────────────────────────────────


def _install_fake_paramiko(monkeypatch) -> MagicMock:
    """Inject a fake paramiko module into ``sys.modules`` and re-import ssh.

    Returns the fake module so tests can assert on it. The SshEnvironment
    captures whichever ``paramiko`` is in ``sys.modules`` at instantiation
    time, so we install our mock before constructing the env.
    """
    fake = MagicMock(name="paramiko")

    # paramiko.SSHException needs to be an actual exception type — the
    # SshEnvironment catches it. MagicMock's auto-attrs are not real classes.
    class _FakeSSHException(Exception):
        pass

    fake.SSHException = _FakeSSHException
    fake.AutoAddPolicy = MagicMock(return_value=MagicMock(name="AutoAddPolicy"))

    monkeypatch.setitem(sys.modules, "paramiko", fake)
    # Also overwrite the module-level cached _paramiko in our ssh module so
    # the kwarg-resolution branch (when sys.modules and module-level both
    # have something) picks up our mock.
    from langstage_hermes.tools.environments import ssh as ssh_mod

    monkeypatch.setattr(ssh_mod, "_paramiko", fake, raising=False)
    return fake


def _make_channel_mock(exit_status: int = 0) -> MagicMock:
    """Build a mock paramiko Channel with an exit status ready."""
    channel = MagicMock(name="Channel")
    channel.exit_status_ready.return_value = True
    channel.recv_exit_status.return_value = exit_status
    return channel


def _make_stdout_mock(channel, data: bytes = b"") -> MagicMock:
    """ChannelFile-like: ``.channel`` attr, ``.read()`` returns data then b''."""
    stdout = MagicMock(name="stdout")
    stdout.channel = channel
    # First read returns the data, subsequent reads return b'' (EOF).
    stdout.read.side_effect = [data, b""]
    return stdout


def _exec_command_returning(exit_status: int = 0, output: bytes = b"") -> MagicMock:
    """Build an ``exec_command`` callable that returns (stdin, stdout, stderr).

    Each call gets a fresh tuple so multi-command tests don't share state.
    """

    def _factory(*_args, **_kwargs):
        channel = _make_channel_mock(exit_status=exit_status)
        stdin = MagicMock(name="stdin")
        stdout = _make_stdout_mock(channel, data=output)
        stderr = MagicMock(name="stderr")
        stderr.read.return_value = b""
        return (stdin, stdout, stderr)

    return MagicMock(side_effect=_factory)


# ── 1. Config validation ─────────────────────────────────────────────


def test_init_requires_host_env(monkeypatch):
    """Without DEEPAGENT_HERMES_SSH_HOST (or host= kwarg) we raise."""
    _install_fake_paramiko(monkeypatch)
    monkeypatch.delenv("DEEPAGENT_HERMES_SSH_HOST", raising=False)

    from langstage_hermes.tools.environments.ssh import SshEnvironment

    with pytest.raises(ValueError, match="DEEPAGENT_HERMES_SSH_HOST"):
        SshEnvironment(session_id="t1")


# ── 2. Connect call shape ────────────────────────────────────────────


def test_init_session_uses_paramiko_client(monkeypatch):
    """init_session opens an SSHClient and calls .connect() with parsed params."""
    fake = _install_fake_paramiko(monkeypatch)
    monkeypatch.setenv("DEEPAGENT_HERMES_SSH_HOST", "alice@example.com:2222")
    monkeypatch.setenv("DEEPAGENT_HERMES_SSH_KEY", "/tmp/fake_key")
    monkeypatch.delenv("DEEPAGENT_HERMES_SSH_PASSWORD", raising=False)

    # SSHClient() returns a fresh mock each call; .exec_command must work.
    client_instance = MagicMock(name="SSHClient.instance")
    client_instance.exec_command = _exec_command_returning(output=b"ok\n")
    fake.SSHClient = MagicMock(return_value=client_instance)

    # Pretend the key file exists so the connect kwargs include key_filename.
    monkeypatch.setattr("os.path.isfile", lambda p: p == "/tmp/fake_key")

    from langstage_hermes.tools.environments.ssh import SshEnvironment

    env = SshEnvironment(session_id="t2")
    env.init_session()

    fake.SSHClient.assert_called()
    client_instance.connect.assert_called_once()
    call_kwargs = client_instance.connect.call_args.kwargs
    assert call_kwargs["hostname"] == "example.com"
    assert call_kwargs["port"] == 2222
    assert call_kwargs["username"] == "alice"
    assert call_kwargs["key_filename"] == "/tmp/fake_key"
    assert call_kwargs["timeout"] == 10.0


# ── 3. exec_command receives the wrapped bash command ────────────────


def test_execute_runs_via_exec_command(monkeypatch):
    """A single execute() call should hand bash -c '<wrapped>' to exec_command.

    The wrapped script must contain the canonical pieces: source <snap>,
    cd "$(cat <cwd_file>...)", eval '<user_cmd>', and pwd -P > <cwd_file>.
    """
    fake = _install_fake_paramiko(monkeypatch)
    monkeypatch.setenv("DEEPAGENT_HERMES_SSH_HOST", "user@host")

    client_instance = MagicMock(name="SSHClient.instance")
    client_instance.exec_command = _exec_command_returning(output=b"hello\n")
    fake.SSHClient = MagicMock(return_value=client_instance)
    monkeypatch.setattr("os.path.isfile", lambda p: False)

    from langstage_hermes.tools.environments.ssh import SshEnvironment

    env = SshEnvironment(session_id="t3")
    env.execute("echo hello")

    # First call is the snapshot bootstrap, subsequent is our echo. Check that
    # at least one exec_command call carries the expected shape.
    calls = client_instance.exec_command.call_args_list
    assert calls, "exec_command was never called"

    # The execute() call wraps the user's command in source+cd+eval+pwd.
    # Inspect the *positional* arg (the remote command string).
    found_wrapper = False
    for call in calls:
        cmd_arg = call.args[0] if call.args else call.kwargs.get("command", "")
        if "eval " in cmd_arg and "langstage-hermes-snap-t3" in cmd_arg and "langstage-hermes-cwd-t3" in cmd_arg:
            assert "source " in cmd_arg
            assert "cd " in cmd_arg
            assert "pwd -P" in cmd_arg
            assert "bash -c" in cmd_arg or "bash -l -c" in cmd_arg
            found_wrapper = True
            break
    assert found_wrapper, f"no wrapped exec_command found; got: {calls}"


# ── 4. CWD marker file references in the wrapped script ──────────────


def test_cwd_persists_via_marker_file(monkeypatch):
    """Each execute() call must read the CWD marker via cat and rewrite it."""
    fake = _install_fake_paramiko(monkeypatch)
    monkeypatch.setenv("DEEPAGENT_HERMES_SSH_HOST", "user@host")

    client_instance = MagicMock(name="SSHClient.instance")
    client_instance.exec_command = _exec_command_returning()
    fake.SSHClient = MagicMock(return_value=client_instance)
    monkeypatch.setattr("os.path.isfile", lambda p: False)

    from langstage_hermes.tools.environments.ssh import SshEnvironment

    env = SshEnvironment(session_id="t4")
    env.execute("cd /var/log")
    env.execute("pwd")

    # Look at the 2nd user-execute (after snapshot). Both should reference the
    # marker file via `cat <cwd_path>`.
    user_calls = [c for c in client_instance.exec_command.call_args_list if "eval " in (c.args[0] if c.args else "")]
    assert len(user_calls) >= 2, f"expected 2 user execs, got {len(user_calls)}"

    marker = "langstage-hermes-cwd-t4"
    for c in user_calls:
        cmd = c.args[0]
        assert "cat " in cmd and marker in cmd, f"call should read cwd marker via cat, got: {cmd!r}"
        assert "pwd -P" in cmd and marker in cmd, f"call should write cwd marker via pwd -P, got: {cmd!r}"


# ── 5. cleanup closes the client ─────────────────────────────────────


def test_cleanup_closes_client(monkeypatch):
    """cleanup() should call .close() on the active SSHClient."""
    fake = _install_fake_paramiko(monkeypatch)
    monkeypatch.setenv("DEEPAGENT_HERMES_SSH_HOST", "user@host")

    client_instance = MagicMock(name="SSHClient.instance")
    client_instance.exec_command = _exec_command_returning()
    fake.SSHClient = MagicMock(return_value=client_instance)
    monkeypatch.setattr("os.path.isfile", lambda p: False)

    from langstage_hermes.tools.environments.ssh import SshEnvironment

    env = SshEnvironment(session_id="t5")
    env.execute("echo x")
    env.cleanup()

    client_instance.close.assert_called()


# ── 6. Missing paramiko raises a helpful ImportError ─────────────────


def test_paramiko_missing_raises_helpful(monkeypatch):
    """If paramiko is unavailable, instantiating SshEnvironment must give a clear hint."""
    monkeypatch.setenv("DEEPAGENT_HERMES_SSH_HOST", "user@host")
    monkeypatch.setitem(sys.modules, "paramiko", None)

    # Also wipe the cached module-level _paramiko so the check picks up our None.
    from langstage_hermes.tools.environments import ssh as ssh_mod

    monkeypatch.setattr(ssh_mod, "_paramiko", None, raising=False)

    with pytest.raises(ImportError, match=r"pip install.*langstage-hermes\[ssh\]"):
        ssh_mod.SshEnvironment(session_id="t6")


# ── 7. Reconnect once on broken pipe ─────────────────────────────────


def test_reconnect_on_broken_pipe(monkeypatch):
    """A paramiko.SSHException on exec_command should trigger one reconnect + retry."""
    fake = _install_fake_paramiko(monkeypatch)
    monkeypatch.setenv("DEEPAGENT_HERMES_SSH_HOST", "user@host")

    # We track connect calls and have exec_command fail once then succeed.
    call_log: dict = {"connects": 0, "execs": 0}

    def _stdin_stdout_stderr(exit_status: int = 0):
        channel = _make_channel_mock(exit_status=exit_status)
        stdin = MagicMock(name="stdin")
        stdout = _make_stdout_mock(channel, data=b"")
        stderr = MagicMock(name="stderr")
        stderr.read.return_value = b""
        return (stdin, stdout, stderr)

    def _exec_command(*_args, **_kwargs):
        call_log["execs"] += 1
        if call_log["execs"] == 1:
            # First call: simulate a transport drop.
            raise fake.SSHException("broken pipe")
        return _stdin_stdout_stderr()

    client_instance = MagicMock(name="SSHClient.instance")
    client_instance.exec_command = MagicMock(side_effect=_exec_command)

    def _new_client():
        call_log["connects"] += 1
        return client_instance

    fake.SSHClient = MagicMock(side_effect=_new_client)
    monkeypatch.setattr("os.path.isfile", lambda p: False)

    from langstage_hermes.tools.environments.ssh import SshEnvironment

    env = SshEnvironment(session_id="t7")
    # Manually drive _exec to avoid intermixing with snapshot init.
    env._connect()  # establishes the initial client
    assert call_log["connects"] == 1

    # This call should fail once, reconnect, then succeed.
    _stdin, stdout, _stderr = env._exec("bash -c 'echo hi'", timeout=10.0, stdin_data=None)
    assert call_log["execs"] == 2
    assert call_log["connects"] == 2  # one initial + one retry-reconnect
    assert stdout is not None
