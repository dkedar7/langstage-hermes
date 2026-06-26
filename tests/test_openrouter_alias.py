"""OPENROUTER_API_KEY is wired to the OpenAI client for openai:* models (gh #33).

The README advertises OPENROUTER_API_KEY as a drop-in for OPENAI_API_KEY, but
ChatOpenAI only reads OPENAI_API_KEY — so the documented path failed with
"Missing credentials" at build. _alias_openrouter_key bridges it.
"""

import os

from langstage_hermes.agent import _alias_openrouter_key


def test_aliases_openrouter_key_for_openai_model(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-DUMMY")

    _alias_openrouter_key("openai:openai/gpt-4o-mini")

    assert os.environ["OPENAI_API_KEY"] == "sk-or-v1-DUMMY"
    assert os.environ["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_does_not_override_explicit_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-DUMMY")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    _alias_openrouter_key("openai:gpt-4o-mini")

    # A real OPENAI_API_KEY wins; we don't clobber it or force the base URL.
    assert os.environ["OPENAI_API_KEY"] == "sk-real-openai"
    assert "OPENAI_BASE_URL" not in os.environ


def test_noop_for_non_openai_models(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-DUMMY")

    _alias_openrouter_key("anthropic:claude-sonnet-4-6")

    assert "OPENAI_API_KEY" not in os.environ
