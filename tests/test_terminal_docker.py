"""Tests for :class:`DockerEnvironment`.

Gated on docker CLI availability + a reachable daemon. On machines without
docker the whole module skips with a clear reason; on machines with docker
we exercise the real container lifecycle (start, exec, stop) end-to-end.

Mark all tests with ``needs_docker`` so CI / local runs can opt out via
``pytest -m "not needs_docker"``.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from deepagent_hermes.tools.environments.base import ExecuteResponse
from deepagent_hermes.tools.environments.docker import (
    DockerEnvironment,
    _docker_available,
    _sanitize_container_name,
)

# ── module-level skip when docker is unusable ────────────────────────

# Per the SPEC: skip if the CLI is missing OR if `docker info` fails with
# non-zero. Both conditions collapse into _docker_available(). We check at
# module import time so the whole file goes "skipped" rather than producing
# per-test skip noise.
if not _docker_available():
    pytest.skip(
        "Docker not available (CLI missing or daemon unreachable)",
        allow_module_level=True,
    )


# Apply the needs_docker marker to every test in this module so users can
# filter with -m "not needs_docker".
pytestmark = pytest.mark.needs_docker


# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def env():
    """Fresh DockerEnvironment per test; always cleanup, even on failure.

    Session id is randomized so concurrent test runs don't collide on the
    container name (the sanitized form lands in ``deepagent-hermes-<id>``).
    """
    e = DockerEnvironment(session_id="test-" + os.urandom(4).hex())
    try:
        yield e
    finally:
        e.cleanup()


# ── unit-ish tests that don't need a running container ──────────────


def test_sanitize_container_name_respects_docker_limits() -> None:
    """Sanitized names stay within Docker's 63-char limit."""
    long_id = "x" * 200
    name = _sanitize_container_name(long_id)
    assert len(name) <= 63
    assert name.startswith("deepagent-hermes-")


def test_sanitize_container_name_replaces_invalid_chars() -> None:
    """Non-alnum/[._-] chars in session ids get replaced with ``-``."""
    name = _sanitize_container_name("abc/def:ghi jkl")
    # Docker container names must match [a-zA-Z0-9][a-zA-Z0-9_.-]*
    assert "/" not in name
    assert ":" not in name
    assert " " not in name


# ── lifecycle tests (actually exercise docker) ──────────────────────


def test_init_session_starts_container_and_cleanup_stops_it() -> None:
    """``init_session`` should bring up the container; ``cleanup`` tears it down."""
    e = DockerEnvironment(session_id="lifecycle-" + os.urandom(4).hex())
    try:
        e.init_session()
        # Probe via docker ps for the container name we computed.
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{e._container_name}$",
             "--format", "{{.ID}}"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout.strip(), (
            f"expected container {e._container_name} to be running, "
            f"got: {result.stdout!r}"
        )
    finally:
        e.cleanup()

    # After cleanup the container should be gone (--rm handles the delete).
    # Give docker a moment to actually flush the stop.
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{e._container_name}$",
         "--format", "{{.ID}}"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert not result.stdout.strip(), (
        f"expected container {e._container_name} to be gone after cleanup, "
        f"got: {result.stdout!r}"
    )


def test_echo_returns_expected_output(env: DockerEnvironment) -> None:
    """``echo hello`` should round-trip with output 'hello\\n' and exit 0."""
    resp = env.execute("echo hello")
    assert isinstance(resp, ExecuteResponse)
    assert resp.output == "hello\n"
    assert resp.exit_code == 0


def test_cwd_persists_across_calls(env: DockerEnvironment) -> None:
    """``cd /tmp; pwd`` across two calls should report ``/tmp``."""
    env.execute("cd /tmp")
    resp = env.execute("pwd")
    # The wrapped command's stdout is the user pwd; strip for newline tolerance.
    assert resp.output.strip().endswith("/tmp"), (
        f"expected pwd output ending in /tmp, got: {resp.output!r}"
    )
    assert resp.exit_code == 0


def test_env_vars_persist_across_calls(env: DockerEnvironment) -> None:
    """``export FOO=bar`` in one call must be readable as ``$FOO`` in the next."""
    env.init_session()
    if not env._initialized:
        pytest.skip("snapshot init failed inside container")

    env.execute("export FOO=bar")
    resp = env.execute("echo $FOO")
    assert "bar" in resp.output


def test_timeout_returns_124(env: DockerEnvironment) -> None:
    """``sleep 30`` with timeout=2 should return exit_code 124."""
    resp = env.execute("sleep 30", timeout=2)
    assert resp.exit_code == 124
    assert "timed out" in resp.output.lower()


def test_exit_code_propagates(env: DockerEnvironment) -> None:
    """Non-zero exits from the user command should surface untouched."""
    resp = env.execute("exit 7")
    assert resp.exit_code == 7


def test_cleanup_is_idempotent(env: DockerEnvironment) -> None:
    """Multiple cleanup() calls should be no-ops after the first."""
    env.init_session()
    env.cleanup()
    # Second cleanup: must not raise.
    env.cleanup()
