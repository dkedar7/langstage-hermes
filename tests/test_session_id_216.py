"""gh #216: the recorded session is the one the factory reports, and
``session_search`` never hands the live conversation back as recall.

Before the fix, a host agent's state had no ``session_id``, so the recorder
minted its own ``sess-<uuid>``. ``hermes_middleware(session_id=...)`` and
``.session_id`` then named a session missing from ``state.db``, and
``session_search`` (filtering on the factory's id) returned the current chat.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langstage_hermes import HermesConfig, hermes_middleware

LIVE = "lineage-check zebra"
LEFT_OUT = "All matching sessions are in the current session's lineage"


class _Scripted(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


class _Aux(FakeListChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


def _search_call() -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "session_search", "args": {"query": "zebra"}, "id": "ss-1"}])


def _model(*turns: AIMessage) -> _Scripted:
    return _Scripted(disable_streaming=True, messages=iter([*turns, *[AIMessage("done")] * 6]))


@pytest.fixture
def home(tmp_hermes_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return tmp_hermes_home


def _search_result(result: dict[str, Any]) -> str:
    return next(m.content for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "session_search")


def _session_ids(store: Any) -> set[str]:
    return {s["id"] if isinstance(s, dict) else s[0] for s in store.list_recent_sessions(limit=50)}


@pytest.mark.parametrize("given", ["my-session", None])
def test_embedded_records_under_the_reported_session(home: Path, given: str | None) -> None:
    mws = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10), session_id=given)
    try:
        agent = create_agent(
            _model(), tools=[], system_prompt="You are Acme's agent.", middleware=[*mws], checkpointer=InMemorySaver()
        )
        out = agent.invoke({"messages": [{"role": "user", "content": LIVE}]}, {"configurable": {"thread_id": "t1"}})
        if given:
            assert mws.session_id == given
        assert out["session_id"] == mws.session_id
        assert _session_ids(mws.store) == {mws.session_id}
    finally:
        mws.close()


def test_embedded_search_leaves_out_the_live_conversation(home: Path) -> None:
    mws = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10), session_id="my-session")
    try:
        agent = create_agent(
            _model(_search_call()),
            tools=[],
            system_prompt="You are Acme's agent.",
            middleware=[*mws],
            checkpointer=InMemorySaver(),
        )
        out = agent.invoke({"messages": [{"role": "user", "content": LIVE}]}, {"configurable": {"thread_id": "t1"}})
        assert LEFT_OUT in _search_result(out)
        # The tool called directly (no graph state) falls back to the reported session.
        search = next(t for m in mws for t in getattr(m, "tools", []) if t.name == "session_search")
        assert LEFT_OUT in search.invoke({"query": "zebra"})
    finally:
        mws.close()


def test_embedded_search_still_finds_a_past_session(home: Path) -> None:
    past = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10), session_id="past")
    try:
        create_agent(_model(), tools=[], middleware=[*past], checkpointer=InMemorySaver()).invoke(
            {"messages": [{"role": "user", "content": "an older zebra question"}]}, {"configurable": {"thread_id": "p"}}
        )
    finally:
        past.close()
    mws = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10), session_id="now")
    try:
        agent = create_agent(_model(_search_call()), tools=[], middleware=[*mws], checkpointer=InMemorySaver())
        out = agent.invoke({"messages": [{"role": "user", "content": LIVE}]}, {"configurable": {"thread_id": "t1"}})
        found = _search_result(out)
        assert "`past`" in found and "`now`" not in found
    finally:
        mws.close()


def test_state_session_id_still_wins(home: Path) -> None:
    """A host that threads its own session id through state keeps it."""
    mws = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10), session_id="default")
    try:
        agent = create_agent(_model(_search_call()), tools=[], middleware=[*mws], checkpointer=InMemorySaver())
        out = agent.invoke(
            {"messages": [{"role": "user", "content": LIVE}], "session_id": "per-thread"}, {"configurable": {"thread_id": "t1"}}
        )
        assert out["session_id"] == "per-thread"
        assert "per-thread" in _session_ids(mws.store)
        assert LEFT_OUT in _search_result(out)
    finally:
        mws.close()


def test_full_agent_hosted_without_session_id_leaves_out_live_chat(home: Path) -> None:
    """The full agent under a host (web, langstage-cli) whose input has no session_id.

    The recorder mints a per-thread id; session_search must filter on that id,
    not the factory's, or the live chat comes back as a past session.
    """
    from langstage_hermes.agent import create_hermes_agent

    graph = create_hermes_agent(
        HermesConfig.resolve(), model=_model(_search_call()), aux_model=_Aux(responses=["ok"] * 10), workspace=os.getcwd()
    )
    out = graph.invoke({"messages": [{"role": "user", "content": LIVE}]}, {"configurable": {"thread_id": "host-1"}})
    assert out["session_id"]
    assert LEFT_OUT in _search_result(out)
