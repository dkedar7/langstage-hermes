"""Tests for :class:`LocalEnvironment` — snapshot, execute, persistent cwd.

Bash is a hard runtime requirement (the wrapper script is plain POSIX shell).
On Windows that means Git Bash; the test fixture skips with a clear reason
when ``bash`` isn't on PATH so a thin Windows dev env doesn't fail the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from deepagent_hermes.tools.environments.base import ExecuteResponse
from deepagent_hermes.tools.environments.local import LocalEnvironment, _find_bash


# ── Fixture ──────────────────────────────────────────────────────────


@pytest.fixture
def env() -> LocalEnvironment:
    """Construct a fresh ``LocalEnvironment`` per test, skipping when bash is missing."""
    if _find_bash() is None:
        pytest.skip("bash not found on PATH")
    e = LocalEnvironment(session_id="test-" + os.urandom(4).hex())
    yield e
    e.cleanup()


# ── Basic execution ──────────────────────────────────────────────────


def test_echo_returns_expected_output(env: LocalEnvironment) -> None:
    """A trivial echo should round-trip with a clean exit code."""
    env.init_session()
    resp = env.execute("echo hello")
    assert isinstance(resp, ExecuteResponse)
    # bash adds a trailing newline; we strip for comparison since CRLF
    # on Windows Git Bash can confuse a literal "hello\n" check.
    assert resp.output.strip() == "hello"
    assert resp.exit_code == 0


def test_exit_code_propagates(env: LocalEnvironment) -> None:
    resp = env.execute("exit 3")
    assert resp.exit_code == 3


def test_init_session_is_idempotent(env: LocalEnvironment) -> None:
    """Calling init_session twice must not blow up the snapshot."""
    env.init_session()
    first_state = env._initialized
    env.init_session()
    assert env._initialized == first_state


# ── CWD persistence across calls ─────────────────────────────────────


def test_cwd_persists_across_execute_calls(
    env: LocalEnvironment, tmp_path: Path
) -> None:
    """``cd`` in one call must be visible to ``pwd`` in the next call.

    Uses ``tmp_path`` rather than ``/tmp`` so the test works on Windows
    (where ``/tmp`` doesn't exist natively, though Git Bash maps it).
    """
    env.init_session()

    target = tmp_path.resolve()
    # On Windows the bash inside Git Bash returns /c/Users/... style paths,
    # which our base class records verbatim. We compare basenames to dodge
    # the MSYS-vs-Windows path-shape difference.
    expected_basename = target.name

    env.execute(f'cd "{target.as_posix()}"')
    resp = env.execute("pwd")
    pwd_output = resp.output.strip().splitlines()[-1] if resp.output.strip() else ""

    # Either the full path matches, or at minimum the basename does
    # (covers MSYS path translation under Git Bash on Windows).
    assert pwd_output.endswith(expected_basename), (
        f"expected pwd to end with {expected_basename!r}, got {pwd_output!r}\n"
        f"full output: {resp.output!r}"
    )


def test_env_vars_persist_across_execute_calls(env: LocalEnvironment) -> None:
    """``export FOO=bar`` in one call must be readable as ``$FOO`` in the next."""
    env.init_session()
    if not env._initialized:
        pytest.skip("snapshot init failed in this environment")

    env.execute('export DEEPAGENT_TEST_VAR="persisted_value"')
    resp = env.execute('echo "$DEEPAGENT_TEST_VAR"')
    assert "persisted_value" in resp.output


# ── Cleanup ──────────────────────────────────────────────────────────


def test_cleanup_removes_artifacts(env: LocalEnvironment) -> None:
    env.init_session()
    env.execute("echo x")
    snap = env._snapshot_path()
    cwd = env._cwd_path()
    # At least one artifact should exist after a real run.
    assert snap.exists() or cwd.exists()

    env.cleanup()
    assert not snap.exists()
    assert not cwd.exists()


# ── Timeout / error paths ────────────────────────────────────────────


def test_timeout_returns_124(env: LocalEnvironment) -> None:
    """Long-running commands killed by the timeout should return exit_code 124."""
    env.init_session()
    resp = env.execute("sleep 5", timeout=1)
    assert resp.exit_code == 124
    assert "timed out" in resp.output.lower()


def test_response_has_duration(env: LocalEnvironment) -> None:
    env.init_session()
    resp = env.execute("echo timed")
    assert resp.duration_ms >= 0.0
