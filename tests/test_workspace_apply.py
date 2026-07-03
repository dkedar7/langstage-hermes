"""create_hermes_agent roots at the shared workspace source of truth (ADR 0005).

The chat path calls ``core.apply_workspace()`` and the factory now defaults to
``core.workspace_root()`` instead of ``cwd`` — so ``--workspace`` / toml / env
reaches the agent's ``FilesystemBackend``. That closes the gap where the chat path
built the factory as ``create_hermes_agent(cfg)`` (no ``workspace=``) and silently
rooted the agent at the launch dir; only ``verify`` used to forward it.
"""

from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langstage_core import apply_workspace
from langstage_core.host import workspace as ws_mod

from langstage_hermes.agent import create_hermes_agent
from langstage_hermes.config import HermesConfig


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Per-test HERMES_HOME so state.db / skills are isolated."""
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_active_workspace(monkeypatch):
    """apply_workspace sets process-global state; isolate it per test."""
    saved = ws_mod._ACTIVE
    monkeypatch.setattr(ws_mod, "_ACTIVE", None)
    monkeypatch.delenv("LANGSTAGE_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("DEEPAGENT_WORKSPACE_ROOT", raising=False)
    yield
    ws_mod._ACTIVE = saved


def _capture_backend_root(monkeypatch) -> dict:
    """Record the root_dir the FilesystemBackend is constructed with."""
    import deepagents.backends.filesystem as fsmod

    captured: dict[str, Any] = {}
    real = fsmod.FilesystemBackend

    def spy(*args, **kwargs):
        captured["root_dir"] = kwargs.get("root_dir")
        return real(*args, **kwargs)

    monkeypatch.setattr(fsmod, "FilesystemBackend", spy)
    return captured


def _fake_model():
    return FakeListChatModel(responses=["ok"])


def test_factory_defaults_to_applied_workspace(home, tmp_path, monkeypatch):
    ws = tmp_path / "applied_ws"
    apply_workspace(ws)  # what the chat path does before building the agent
    captured = _capture_backend_root(monkeypatch)
    with patch("langstage_hermes.agent._init_chat_model", return_value=_fake_model()):
        create_hermes_agent(HermesConfig.resolve())  # no explicit workspace=
    assert captured["root_dir"] == str(ws.resolve())


def test_explicit_workspace_still_wins(home, tmp_path, monkeypatch):
    apply_workspace(tmp_path / "applied_ws")
    explicit = tmp_path / "explicit_ws"
    captured = _capture_backend_root(monkeypatch)
    with patch("langstage_hermes.agent._init_chat_model", return_value=_fake_model()):
        create_hermes_agent(HermesConfig.resolve(), workspace=str(explicit))
    assert captured["root_dir"] == str(explicit.resolve())
