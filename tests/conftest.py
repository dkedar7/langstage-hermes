"""Shared pytest fixtures for deepagent-hermes."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_hermes_home(monkeypatch, tmp_path: Path) -> Path:
    """Isolated HERMES_HOME / DEEPAGENT_HERMES_HOME pointing at a tmp dir.

    Use this in any test that touches the on-disk skill/memory/cron/state layout.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "memories").mkdir()
    (home / "cron").mkdir()
    (home / "logs").mkdir()
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(home))
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
