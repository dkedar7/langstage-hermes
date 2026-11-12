"""gh #218: a skill change made during a run is audited under that run's session.

Before the fix the library's audit context was set once, at build time, so a
run with a per-run ``session_id`` in its input (the README's way to keep
conversations apart) still wrote ``hermes_middleware().session_id`` into
``skill_mutations``. That is a session ``state.db`` never recorded, so
``audit log`` couldn't lead back to the conversation. The review fork and the
curator, which share the library, had the same gap.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langstage_hermes import hermes_middleware
from langstage_hermes.curator import CuratorMiddleware, _save_curator_state
from langstage_hermes.reflection import ReflectionMiddleware
from langstage_hermes.skills.tools import _skill_manage_impl

BODY = "# Order lookup\n\n1. Call lookup_order.\n2. Report the status.\n"


class _Scripted(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


class _Aux(FakeListChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


def _create_call(name: str = "order-lookup") -> AIMessage:
    args = {"action": "create", "name": name, "description": "Look up an order status by id.", "body": BODY}
    return AIMessage("", tool_calls=[{"name": "skill_manage", "id": "c1", "args": args}])


def _model(*turns: AIMessage) -> _Scripted:
    return _Scripted(disable_streaming=True, messages=iter([*turns, *[AIMessage("done")] * 6]))


@pytest.fixture
def home(tmp_hermes_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return tmp_hermes_home


def _rows(mws: Any, name: str = "order-lookup") -> list[Any]:
    return mws.library.audit_log.list(skill_name=name)


def _sessions(store: Any) -> set[str]:
    return {s["id"] if isinstance(s, dict) else s[0] for s in store.list_recent_sessions(limit=50)}


def _create(library: Any, name: str, session_id: str | None = None) -> None:
    _skill_manage_impl(
        library,
        action="create",
        name=name,
        description="Look up an order status by id.",
        body=BODY,
        category="",
        old_str="",
        new_str="",
        frontmatter_data=None,
        tool_call_id="c-" + name,
        session_id=session_id,
    )


def test_per_run_session_id_reaches_the_audit_row(home: Path) -> None:
    """The issue's repro: the audit row names the session state.db recorded."""
    mws = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10))
    try:
        agent = create_agent(_model(_create_call()), tools=[], middleware=[*mws], checkpointer=InMemorySaver())
        out = agent.invoke(
            {"messages": [{"role": "user", "content": "save that as a skill"}], "session_id": "chat-7"},
            {"configurable": {"thread_id": "t1"}},
        )
        assert out["session_id"] == "chat-7"
        (row,) = _rows(mws)
        assert row.session_id == "chat-7" != mws.session_id
        assert row.session_id in _sessions(mws.store)
        assert (row.source, row.tool_call_id) == ("agent", "c1")
    finally:
        mws.close()


def test_without_a_per_run_id_the_row_carries_the_reported_session(home: Path) -> None:
    mws = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10), session_id="default-session")
    try:
        agent = create_agent(_model(_create_call()), tools=[], middleware=[*mws], checkpointer=InMemorySaver())
        agent.invoke({"messages": [{"role": "user", "content": "save it"}]}, {"configurable": {"thread_id": "t1"}})
        (row,) = _rows(mws)
        assert row.session_id == "default-session"
        assert row.session_id in _sessions(mws.store)
    finally:
        mws.close()


def test_two_runs_on_one_list_keep_their_own_sessions(home: Path) -> None:
    mws = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10))
    try:
        agent = create_agent(
            _model(_create_call("first-skill"), AIMessage("saved"), _create_call("second-skill")),
            tools=[],
            middleware=[*mws],
            checkpointer=InMemorySaver(),
        )
        for sid, thread in (("chat-a", "ta"), ("chat-b", "tb")):
            run_input = {"messages": [{"role": "user", "content": "save"}], "session_id": sid}
            agent.invoke(run_input, {"configurable": {"thread_id": thread}})
        assert [r.session_id for r in _rows(mws, "first-skill")] == ["chat-a"]
        assert [r.session_id for r in _rows(mws, "second-skill")] == ["chat-b"]
    finally:
        mws.close()


def test_concurrent_runs_sharing_a_library_do_not_cross_attribute(home: Path) -> None:
    """The per-run session is context-scoped, not a field on the shared library."""
    mws = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10), session_id="default-session")
    barrier = threading.Barrier(2)
    try:

        def run(sid: str) -> None:
            with mws.library.mutation_session(sid):
                barrier.wait()
                _create(mws.library, f"skill-{sid}")

        threads = [threading.Thread(target=run, args=(sid,)) for sid in ("left", "right")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert [r.session_id for r in _rows(mws, "skill-left")] == ["left"]
        assert [r.session_id for r in _rows(mws, "skill-right")] == ["right"]
    finally:
        mws.close()


class _ReviewGraph:
    """Stands in for the review fork: it edits the shared library, with no session in its own state."""

    def __init__(self, library: Any) -> None:
        self.library = library

    def invoke(self, _input: Any) -> dict[str, Any]:
        _create(self.library, "reviewed-skill")
        return {"messages": []}

    async def ainvoke(self, input: Any) -> dict[str, Any]:
        return self.invoke(input)


@pytest.mark.parametrize("use_async", [False, True])
def test_review_fork_changes_carry_the_parent_run_session(home: Path, use_async: bool) -> None:
    import asyncio

    mws = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10), session_id="default-session")
    try:
        mw = ReflectionMiddleware(library=mws.library, store=mws.store, model=None, review_graph=_ReviewGraph(mws.library))
        state = {"messages": [], "pending_review_kind": "skills", "session_id": "chat-7"}
        if use_async:
            asyncio.run(mw.aafter_agent(state))
        else:
            mw.after_agent(state)
        (row,) = _rows(mws, "reviewed-skill")
        assert row.session_id == "chat-7"
    finally:
        mws.close()


class _CuratorGraph:
    def __init__(self, library: Any) -> None:
        self.library = library

    def invoke(self, _input: Any) -> dict[str, Any]:
        _create(self.library, "curated-skill")
        return {"messages": [AIMessage("consolidated")]}


def test_curator_pass_changes_carry_the_run_session(home: Path) -> None:
    mws = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10), session_id="default-session")
    try:
        long_ago = time.time() - 400 * 3600
        _save_curator_state(mws.store, {"last_run_at": long_ago, "last_user_activity": long_ago})
        mw = CuratorMiddleware(mws.library, mws.store, curator_graph=_CuratorGraph(mws.library))
        mw.before_agent({"messages": [], "session_id": "chat-7"})
        (row,) = _rows(mws, "curated-skill")
        assert row.session_id == "chat-7"
    finally:
        mws.close()


def test_full_agent_audits_under_the_per_run_session(home: Path) -> None:
    from langstage_hermes import HermesConfig
    from langstage_hermes.agent import create_hermes_agent

    graph = create_hermes_agent(
        HermesConfig.resolve(), model=_model(_create_call()), aux_model=_Aux(responses=["ok"] * 10), workspace=str(home)
    )
    out = graph.invoke(
        {"messages": [{"role": "user", "content": "save that as a skill"}], "session_id": "chat-9"},
        {"configurable": {"thread_id": "t1"}},
    )
    assert out["session_id"] == "chat-9"
    from langstage_hermes.skills.audit import SkillAuditLog

    log = SkillAuditLog(home / "state.db")
    try:
        (row,) = log.list(skill_name="order-lookup")
        assert row.session_id == "chat-9"
    finally:
        log.close()
