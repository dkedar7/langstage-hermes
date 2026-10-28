"""`/yolo`, `/verbose` and `/reload` act on the session they report on, and a cron job whose
agent can't be imported records a failure (follow-ups to gh #193)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langstage_hermes.cli import _run_agent_turn_agui, _slash_reload, _slash_verbose, _slash_yolo
from langstage_hermes.config import HermesConfig


class _ToolFake(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


class _Fake(FakeListChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _gated_graph(ran: list[str]) -> Any:
    @tool
    def danger(target: str) -> str:
        """Do something that needs approval."""
        ran.append(target)
        return f"did {target}"

    model = _ToolFake(
        disable_streaming=True,
        messages=iter(
            [
                AIMessage("", tool_calls=[{"name": "danger", "args": {"target": "prod"}, "id": "c1"}]),
                AIMessage("all done"),
            ]
        ),
    )
    return create_agent(
        model,
        tools=[danger],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"danger": True})],
        checkpointer=InMemorySaver(),
    )


def _state(**kw: Any) -> dict[str, Any]:
    return {"session_id": "s1", "cfg": HermesConfig.resolve(), "verbose": False, "yolo": False, **kw}


# ── /yolo ───────────────────────────────────────────────────────────────


def test_yolo_approves_a_tool_review_pause(home: Path, capsys):
    ran: list[str] = []
    st = _state()
    _slash_yolo("", st)
    assert st["yolo"] is True
    _run_agent_turn_agui(_gated_graph(ran), "go", st)
    out = capsys.readouterr().out
    assert ran == ["prod"]
    assert "auto-approved (/yolo): danger" in out
    assert "all done" in out


def test_without_yolo_the_pause_stops_the_turn(home: Path, capsys):
    ran: list[str] = []
    _run_agent_turn_agui(_gated_graph(ran), "go", _state())
    out = capsys.readouterr().out
    assert ran == []
    assert "paused for approval" in out and "/yolo" in out
    assert "all done" not in out


def test_yolo_never_answers_a_question_interrupt():
    from langstage_hermes.cli import _yolo_decision

    assert _yolo_decision({"action_requests": ["What is your name?"], "allowed_decisions": ["approve"]}) is None
    assert _yolo_decision({"action_requests": [{"name": "t"}], "allowed_decisions": ["reject"]}) is None
    got = _yolo_decision({"action_requests": [{"name": "a"}, {"name": "b"}], "allowed_decisions": ["approve"]})
    assert got["resume"] == {"decisions": [{"type": "approve"}, {"type": "approve"}]}


# ── /verbose ────────────────────────────────────────────────────────────


def test_verbose_shows_tool_args_and_results(home: Path, capsys):
    st = _state(yolo=True)
    _slash_verbose("", st)
    assert st["verbose"] is True
    _run_agent_turn_agui(_gated_graph([]), "go", st)
    out = capsys.readouterr().out
    assert 'args: {"target": "prod"}' in out
    assert "success: did prod" in out


def test_quiet_by_default(home: Path, capsys):
    _run_agent_turn_agui(_gated_graph([]), "go", _state(yolo=True))
    out = capsys.readouterr().out
    assert "args:" not in out and "did prod" not in out


# ── /reload ─────────────────────────────────────────────────────────────


def test_reload_rebuilds_the_builtin_agent_and_keeps_the_conversation(home: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr("langstage_hermes.agent._init_chat_model", lambda mid: _Fake(responses=["hello back"]))
    from langstage_hermes.agent import create_hermes_agent

    overrides = {"model_default": "openai:from-cli-flag"}
    cfg = HermesConfig.resolve(overrides=overrides)
    old = create_hermes_agent(cfg)
    sid = old.langstage_hermes_session_id
    thread = {"configurable": {"thread_id": sid}}
    old.invoke({"messages": [HumanMessage("remember this")]}, config=thread)
    st = _state(session_id=sid, cfg=cfg, agent=old, cli_overrides=overrides, _agui_agent=object())

    _slash_reload("", st)

    out = capsys.readouterr().out
    assert "agent rebuilt" in out
    assert st["agent"] is not old
    assert "_agui_agent" not in st
    assert st["cfg"].model_default == "openai:from-cli-flag"  # --model survives /reload
    kept = [m.content for m in st["agent"].get_state(thread).values["messages"]]
    assert "remember this" in kept and "hello back" in kept


def test_reload_says_a_spec_graph_is_not_rebuilt(home: Path, capsys):
    spec_graph = object()
    st = _state(agent=spec_graph, cli_overrides={})
    _slash_reload("", st)
    assert st["agent"] is spec_graph
    assert "isn't rebuilt" in capsys.readouterr().out


def test_reload_failure_keeps_the_running_agent(home: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    class _BuiltIn:
        langstage_hermes_config = object()

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("bad config")

    monkeypatch.setattr("langstage_hermes.agent.create_hermes_agent", boom)
    agent = _BuiltIn()
    st = _state(agent=agent, cli_overrides={})
    _slash_reload("", st)
    assert st["agent"] is agent
    assert "reload failed, still running the previous agent: bad config" in capsys.readouterr().out


# ── cron ────────────────────────────────────────────────────────────────


def test_cron_job_without_an_agent_is_a_failure(monkeypatch: pytest.MonkeyPatch):
    from langstage_hermes.cron.scheduler import _build_cron_response

    monkeypatch.setitem(sys.modules, "langstage_hermes.agent", None)  # import raises ImportError
    ok, text = _build_cron_response({"id": "j1"}, prompt="summarise the news")
    assert ok is False
    assert text.startswith("[agent unavailable:")
    assert "summarise the news" not in text


# ── turns share one event loop ──────────────────────────────────────────


def test_turns_reuse_one_open_event_loop(home: Path):
    # A loop closed after every turn broke every other openai:* turn with
    # "Event loop is closed" (pooled HTTP connections are bound to their loop).
    import asyncio

    from langstage_hermes import agui_stream

    seen: list[Any] = []

    from langchain.agents.middleware.types import AgentMiddleware

    class _LoopSpy(AgentMiddleware):
        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            seen.append(asyncio.get_running_loop())
            return await handler(request)

    graph = create_agent(_Fake(responses=["one", "two"]), tools=[], middleware=[_LoopSpy()], checkpointer=InMemorySaver())
    st = _state()
    _run_agent_turn_agui(graph, "first", st)
    _run_agent_turn_agui(graph, "second", st)
    assert len(seen) == 2 and seen[0] is seen[1]
    assert not agui_stream._session_loop().is_closed()
