"""`chat` checks the aux model's key too, like verify/doctor (gh #189).

#96 gave ``verify``/``doctor`` the aux-model key preflight, but ``chat`` kept its own
main-model-only gate from #76. The README's first OpenAI/OpenRouter snippet,
``chat --model openai:openai/gpt-4o-mini``, overrides only the main model, so the
default ``anthropic:*`` aux model ran the reflection review with no key and chat
said nothing, while ``verify`` on the same config exited 1. Chat now warns at
startup (it doesn't exit: chatting works, only skill/memory creation fails).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import _warn_aux_model_key, cli

_AUX_MSG = "aux model is anthropic:* but ANTHROPIC_API_KEY not set"


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)  # no stray langstage.toml from the repo
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_BASE_URL",
        "LANGSTAGE_HERMES_HOME",
        "DEEPAGENT_HERMES_HOME",
        "LANGSTAGE_HERMES_MODEL_DEFAULT",
        "DEEPAGENT_HERMES_MODEL_DEFAULT",
        "LANGSTAGE_HERMES_MODEL_AUX",
        "DEEPAGENT_HERMES_MODEL_AUX",
        "LANGSTAGE_AGENT_SPEC",
        "DEEPAGENT_AGENT_SPEC",
    ):
        monkeypatch.delenv(k, raising=False)


class _FakeGraph:
    def invoke(self, *args, **kwargs):
        return {"messages": []}


def _no_build(monkeypatch):
    """Skip building the real agent: the check runs before it, and the [openai] extra may be absent."""
    monkeypatch.setattr("langstage_hermes.cli._instantiate_factory", lambda target, cfg: _FakeGraph())


def test_chat_with_openai_main_model_warns_about_unkeyed_anthropic_aux(monkeypatch, tmp_path):
    """The issue's repro: chat starts (exit 0) but says the aux model has no key."""
    _isolate(monkeypatch, tmp_path)
    _no_build(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")

    r = CliRunner().invoke(cli, ["chat", "--model", "openai:openai/gpt-4o-mini"], input="/quit\n")

    assert r.exit_code == 0, r.output
    assert _AUX_MSG in r.output
    assert "LANGSTAGE_HERMES_MODEL_AUX" in r.output


def test_chat_with_keyed_aux_model_is_silent(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    _no_build(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_AUX", "openai:openai/gpt-4o-mini")

    r = CliRunner().invoke(cli, ["chat", "--model", "openai:openai/gpt-4o-mini"], input="/quit\n")

    assert r.exit_code == 0, r.output
    assert "aux model is" not in r.output


@pytest.mark.parametrize(
    ("aux", "main", "env", "warns"),
    [
        ("anthropic:claude-haiku-4-5-20251001", "openai:openai/gpt-4o-mini", {"OPENAI_API_KEY": "x"}, True),
        ("anthropic:claude-haiku-4-5-20251001", "openai:openai/gpt-4o-mini", {"ANTHROPIC_API_KEY": "x"}, False),
        ("openai:openai/gpt-4o-mini", "anthropic:claude-sonnet-4-5", {"ANTHROPIC_API_KEY": "x"}, True),
        ("openai:openai/gpt-4o-mini", "anthropic:claude-sonnet-4-5", {"OPENROUTER_API_KEY": "x"}, False),
        # Same model as main: the main gate already covered it.
        ("anthropic:claude-sonnet-4-5", "anthropic:claude-sonnet-4-5", {}, False),
        ("", "openai:openai/gpt-4o-mini", {}, False),
        ("ollama:llama3", "openai:openai/gpt-4o-mini", {}, False),  # no key we know of
    ],
)
def test_warn_aux_model_key(monkeypatch, capsys, aux, main, env, warns):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    _warn_aux_model_key(aux, main)  # never exits

    err = capsys.readouterr().err
    assert ("aux model is" in err) is warns
