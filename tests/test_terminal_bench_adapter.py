"""Smoke test for the Harbor adapter without booting a real container.

We construct a fake :class:`BaseEnvironment` that records every
``exec`` call and returns canned ``ExecResult`` objects, then drive
:class:`HarborSandboxBackend` through ``read``/``write``/``ls``/``execute``
to confirm the sync→async bridge works and the protocol contract is
honoured.

This test is intentionally kept out of the default ``examples/`` layout
because Harbor isn't a hard dependency of the package — the test is
skipped when ``harbor`` isn't importable.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("harbor")

# Make the adapter importable as a module without a setup.py entry.
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES))


class FakeExecResult:
    """Minimal stand-in for ``harbor.environments.base.ExecResult``.

    We don't import the real one because it's a pydantic model that
    validates inside ``__init__``; constructing it directly is cheap
    enough but reusing pydantic isn't necessary for the bridge to work
    — the adapter only reads ``.stdout``, ``.stderr``, ``.return_code``.
    """

    def __init__(self, stdout: str = "", stderr: str = "", return_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


class FakeEnv:
    """Records exec() calls and returns canned responses keyed by command."""

    def __init__(self) -> None:
        self.session_id = "fake-session"
        self.calls: list[tuple[str, dict]] = []
        # The default user is set by Harbor's orchestrator; the adapter
        # doesn't read it, but BaseSandbox.read/ls etc. do call exec
        # without an explicit user, which our stub ignores.
        self.default_user: str | int | None = None

    async def exec(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        # Cheap pattern match: writes succeed silently, reads return canned.
        if command.startswith("test -f"):
            return FakeExecResult(return_code=0)
        if "base64 -d > " in command:
            return FakeExecResult(return_code=0)
        if command.startswith("base64 -w0 "):
            # Base64 of "hello world\n"
            return FakeExecResult(stdout="aGVsbG8gd29ybGQK", return_code=0)
        if command == "echo hi":
            return FakeExecResult(stdout="hi\n", return_code=0)
        return FakeExecResult(stdout="", return_code=0)


def test_harbor_backend_execute_returns_exec_response():
    """``execute`` should ferry env.exec output into ExecuteResponse."""
    from terminal_bench import HarborSandboxBackend

    async def run():
        env = FakeEnv()
        loop = asyncio.get_running_loop()
        backend = HarborSandboxBackend(env, loop)
        # ``execute`` is sync; the bridge spawns a thread that schedules
        # back onto our loop. ``asyncio.to_thread`` is the official path.
        result = await asyncio.to_thread(backend.execute, "echo hi")
        assert result.exit_code == 0
        assert "hi" in result.output

    asyncio.run(run())


def test_harbor_backend_upload_download_roundtrip():
    """upload_files+download_files should land in env.exec as base64 ops."""
    from terminal_bench import HarborSandboxBackend

    async def run():
        env = FakeEnv()
        loop = asyncio.get_running_loop()
        backend = HarborSandboxBackend(env, loop)

        up = await asyncio.to_thread(backend.upload_files, [("/tmp/x.txt", b"hello world\n")])
        assert up[0].error is None
        # Last upload command should be the base64-pipe write.
        last_up_cmd = env.calls[-1][0]
        assert "base64 -d > " in last_up_cmd
        assert "/tmp/x.txt" in last_up_cmd

        down = await asyncio.to_thread(backend.download_files, ["/tmp/x.txt"])
        assert down[0].error is None
        assert down[0].content == b"hello world\n"

    asyncio.run(run())


def test_harbor_backend_download_missing_file_returns_not_found():
    """download_files should surface ``file_not_found`` for ``test -f`` failures."""
    from terminal_bench import HarborSandboxBackend

    class MissingEnv(FakeEnv):
        async def exec(self, command: str, **kwargs):
            self.calls.append((command, kwargs))
            if command.startswith("test -f"):
                return FakeExecResult(return_code=1)
            return FakeExecResult(return_code=0)

    async def run():
        env = MissingEnv()
        loop = asyncio.get_running_loop()
        backend = HarborSandboxBackend(env, loop)
        down = await asyncio.to_thread(backend.download_files, ["/nope.txt"])
        assert down[0].content is None
        assert down[0].error == "file_not_found"

    asyncio.run(run())


def test_langstage_hermes_agent_metadata():
    """``name()`` and ``version()`` should match the installed package."""
    from terminal_bench import DeepagentHermesAgent

    assert DeepagentHermesAgent.name() == "langstage-hermes"
    # version() returns None if the package isn't importable; if it IS
    # importable, the value must be a non-empty string.
    instance = DeepagentHermesAgent.__new__(DeepagentHermesAgent)
    v = instance.version()
    assert v is None or (isinstance(v, str) and v)
