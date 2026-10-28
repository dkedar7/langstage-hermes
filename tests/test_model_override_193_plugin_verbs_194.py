"""gh #193: `/model` in chat actually switches the model; gh #194: `plugins enable/disable`
never report success for something they didn't do."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from langstage_hermes.agent import create_hermes_agent
from langstage_hermes.cli import _slash_model, cli
from langstage_hermes.config import HermesConfig


class _Fake(FakeListChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _reply(graph: Any, override: str | None, thread: str) -> str:
    out = graph.invoke(
        {"messages": [HumanMessage("hi")], "model_override": override},
        config={"configurable": {"thread_id": thread}},
    )
    return out["messages"][-1].content


# ── #193: the override reaches the model call ───────────────────────────


def test_model_override_runs_the_override_model(home: Path, monkeypatch: pytest.MonkeyPatch):
    built: list[str] = []

    def fake_init(model_id: str | None) -> Any:
        built.append(model_id or "")
        return _Fake(responses=[f"reply from {model_id}"])

    monkeypatch.setattr("langstage_hermes.agent._init_chat_model", fake_init)
    cfg = HermesConfig.resolve()
    cfg.model_default = "openai:main-model"
    cfg.model_aux = None
    graph = create_hermes_agent(cfg)

    assert _reply(graph, None, "t1") == "reply from openai:main-model"
    assert _reply(graph, "anthropic:other-model", "t2") == "reply from anthropic:other-model"
    # The same id as the default is not rebuilt; a repeated override reuses its model.
    assert _reply(graph, "openai:main-model", "t3") == "reply from openai:main-model"
    _reply(graph, "anthropic:other-model", "t4")
    assert built.count("anthropic:other-model") == 1


def test_model_override_applies_to_a_byo_model(home: Path, monkeypatch: pytest.MonkeyPatch):
    # The issue's repro: a caller-supplied model, then an override id.
    monkeypatch.setattr("langstage_hermes.agent._init_chat_model", lambda mid: _Fake(responses=[f"reply from {mid}"]))
    fake = _Fake(responses=["reply from the ORIGINAL model"])
    graph = create_hermes_agent(model=fake, aux_model=fake)
    assert _reply(graph, "anthropic:claude-haiku-4-5", "t1") == "reply from anthropic:claude-haiku-4-5"
    assert _reply(graph, None, "t2") == "reply from the ORIGINAL model"


# ── #193: /model refuses what it can't honor ────────────────────────────


class _BuiltIn:
    langstage_hermes_config = object()


def _state(agent: Any = None) -> dict[str, Any]:
    return {"model_override": None, "cfg": HermesConfig.resolve(), "agent": agent or _BuiltIn()}


def test_slash_model_sets_a_usable_model(home: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr("langstage_hermes.agent._init_chat_model", lambda mid: _Fake(responses=["ok"]))
    st = _state()
    _slash_model("openai:gpt-test", st)
    assert st["model_override"] == "openai:gpt-test"
    assert "model override set" in capsys.readouterr().out


def test_slash_model_refuses_a_missing_key(home: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    st = _state()
    _slash_model("anthropic:claude-haiku-4-5", st)
    out = capsys.readouterr().out
    assert st["model_override"] is None
    assert "model not changed" in out and "ANTHROPIC_API_KEY" in out


def test_slash_model_refuses_a_model_that_cannot_be_built(home: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    def boom(mid: str) -> Any:
        raise ValueError("Unable to infer model provider")

    monkeypatch.setattr("langstage_hermes.agent._init_chat_model", boom)
    st = _state()
    _slash_model("nosuchprovider:x", st)
    assert st["model_override"] is None
    assert "model not changed: Unable to infer model provider" in capsys.readouterr().out


def test_slash_model_refuses_for_a_spec_agent(home: Path, capsys):
    st = _state(agent=object())
    _slash_model("openai:gpt-test", st)
    assert st["model_override"] is None
    assert "only applies to the built-in agent" in capsys.readouterr().out


# ── #194: plugins enable/disable ────────────────────────────────────────


@pytest.mark.parametrize("verb", ["enable", "disable"])
def test_plugin_verbs_exit_1_and_say_how(home: Path, verb: str):
    result = CliRunner().invoke(cli, ["plugins", verb, "markdown-provider"])
    assert result.exit_code == 1
    assert "doesn't edit your config" in result.output
    assert "[plugins] disabled" in result.output
    assert "TBD" not in result.output
    assert not (home / "config.toml").exists()


@pytest.mark.parametrize("verb", ["enable", "disable"])
def test_plugin_verbs_reject_an_unknown_name(home: Path, verb: str):
    result = CliRunner().invoke(cli, ["plugins", verb, "nosuchplugin"])
    assert result.exit_code == 1
    assert "No plugin named 'nosuchplugin'" in result.output


def test_plugin_help_no_longer_claims_to_write_config():
    out = CliRunner().invoke(cli, ["plugins", "--help"]).output
    assert "TBD" not in out
    assert "Add ``name``" not in out
