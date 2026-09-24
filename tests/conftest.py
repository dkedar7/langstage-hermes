"""Shared pytest fixtures for langstage-hermes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# ── Home-directory sandbox ───────────────────────────────────────────
#
# No test may ever read or write a developer's real hermes home, whatever their
# shell exports (a real HERMES_HOME, LANGSTAGE_HERMES_HOME or legacy
# DEEPAGENT_HERMES_HOME). Two layers:
#
# 1. At conftest import (before any test module imports langstage_hermes), every
#    home variable, plus HOME / USERPROFILE (so the `~/.langstage-hermes` default
#    and core's `~/.langstage` config are sandboxed too), is pointed at a
#    session-wide temp dir. This covers anything resolved at import time.
# 2. The autouse `_sandbox_hermes_home` fixture re-points them at a fresh per-test
#    dir with monkeypatch. A test that sets its own home (monkeypatch.setenv, or
#    the `tmp_hermes_home` fixture) still wins, because its setup runs after this.
#
# All three home variables point at the SAME dir, so a test that sets only
# HERMES_HOME must also clear LANGSTAGE_HERMES_HOME (it outranks HERMES_HOME).

_HOME_VARS = ("HERMES_HOME", "LANGSTAGE_HERMES_HOME", "DEEPAGENT_HERMES_HOME")
_USER_HOME_VARS = ("HOME", "USERPROFILE")

_SESSION_SANDBOX = Path(tempfile.mkdtemp(prefix="langstage-hermes-test-home-"))
(_SESSION_SANDBOX / "user").mkdir()
for _var in _HOME_VARS:
    os.environ[_var] = str(_SESSION_SANDBOX / "hermes_home")
for _var in _USER_HOME_VARS:
    os.environ[_var] = str(_SESSION_SANDBOX / "user")


@pytest.fixture(autouse=True)
def _sandbox_hermes_home(monkeypatch, tmp_path_factory) -> Path:
    """Point every home variable at a fresh per-test temp dir (see module comment)."""
    base = tmp_path_factory.mktemp("sandbox_home")
    hermes = base / "hermes_home"
    user = base / "user"
    user.mkdir()
    for var in _HOME_VARS:
        monkeypatch.setenv(var, str(hermes))
    for var in _USER_HOME_VARS:
        monkeypatch.setenv(var, str(user))
    return hermes


@pytest.fixture
def tmp_hermes_home(monkeypatch, tmp_path: Path) -> Path:
    """Isolated HERMES_HOME / LANGSTAGE_HERMES_HOME pointing at a tmp dir.

    Use this in any test that touches the on-disk skill/memory/cron/state layout.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "memories").mkdir()
    (home / "cron").mkdir()
    (home / "logs").mkdir()
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(home))
    monkeypatch.delenv("DEEPAGENT_HERMES_HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Empty working directory for FilesystemBackend tests."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def disable_anthropic(monkeypatch):
    """Pretend no Anthropic key — avoids accidental network calls."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
