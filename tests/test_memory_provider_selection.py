"""Selecting the bundled markdown provider builds the agent (gh #37).

`memory.provider="markdown"` is the only documented way to enable the bundled
MarkdownProvider, but it hard-crashed the agent build with a KeyError: the
runtime factory never imported the builtin plugin (only the `plugins` CLI did),
so "markdown" was never registered. And get_provider's docstring promised a
degrade-to-noop fallback the factory didn't implement. These tests lock in both:
markdown builds, an unknown name degrades to noop instead of crashing.
"""

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langstage_hermes.agent import create_hermes_agent
from langstage_hermes.config import HermesConfig
from langstage_hermes.memory.provider import available_providers, ensure_builtin_providers


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(tmp_path))
    return tmp_path


def _stub() -> FakeListChatModel:
    return FakeListChatModel(responses=["YES"])


def test_ensure_builtin_providers_registers_markdown():
    ensure_builtin_providers()
    assert "markdown" in available_providers()


def test_markdown_provider_builds_the_agent(home, tmp_path, monkeypatch):
    """The exact #37 repro: memory.provider=markdown no longer KeyErrors at build."""
    monkeypatch.setenv("LANGSTAGE_HERMES_MEMORY_PROVIDER", "markdown")
    cfg = HermesConfig.resolve()
    assert cfg.memory_provider == "markdown"
    # Must not raise KeyError("No memory provider registered as 'markdown'").
    graph = create_hermes_agent(cfg, workspace=tmp_path, session_id="s", model=_stub(), aux_model=_stub())
    assert graph is not None


def test_unknown_provider_degrades_to_noop(home, tmp_path, monkeypatch, caplog):
    """An unregistered provider name degrades to noop with a warning, not a crash."""
    monkeypatch.setenv("LANGSTAGE_HERMES_MEMORY_PROVIDER", "does-not-exist")
    cfg = HermesConfig.resolve()
    import logging

    with caplog.at_level(logging.WARNING):
        graph = create_hermes_agent(cfg, workspace=tmp_path, session_id="s", model=_stub(), aux_model=_stub())
    assert graph is not None
    assert any("falling back to the no-op provider" in r.message for r in caplog.records)


def test_default_provider_is_noop(home, tmp_path, monkeypatch):
    """Regression: the default (unset) provider still builds via noop."""
    monkeypatch.delenv("LANGSTAGE_HERMES_MEMORY_PROVIDER", raising=False)
    monkeypatch.delenv("DEEPAGENT_HERMES_MEMORY_PROVIDER", raising=False)
    cfg = HermesConfig.resolve()
    assert cfg.memory_provider in ("", None)
    graph = create_hermes_agent(cfg, workspace=tmp_path, session_id="s", model=_stub(), aux_model=_stub())
    assert graph is not None
