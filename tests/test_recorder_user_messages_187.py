"""The session store records what the user typed (gh #187).

``HermesStateRecorderMiddleware`` wrote the agent's replies (``after_model``) and
tool output (``wrap_tool_call``) but nothing wrote the ``HumanMessage``, so
``search`` and the agent's own ``session_search`` could never find the user's
words. These tests drive the real ``create_hermes_agent`` wiring with a keyless
scripted model, the path a real turn takes, rather than seeding rows through
``record_message``.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langstage_hermes import HermesConfig, create_hermes_agent
from langstage_hermes.demo import DemoReviewModel, _close_agent_resources, _ScriptedModel
from langstage_hermes.store.recorder import HermesStateRecorderMiddleware
from langstage_hermes.store.sqlite_fts import SqliteFtsStore


class _OneToolThenAnswer(_ScriptedModel):
    """Each turn: one ``ls`` call, then an answer."""

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any):
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        turn = sum(isinstance(m, HumanMessage) for m in messages)
        if not any(isinstance(m, ToolMessage) for m in messages[last_human:]):
            return self._result(AIMessage(content="", tool_calls=[{"name": "ls", "id": f"ls-{turn}", "args": {"path": "."}}]))
        return self._result(AIMessage(content=f"Answer {turn}."))


def _rows(home: Path, sid: str) -> list[tuple[str, str | None]]:
    con = sqlite3.connect(home / "state.db")
    try:
        return con.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (sid,)).fetchall()
    finally:
        con.close()


def _agent(home: Path, sid: str, *, memory_interval: int = 10) -> Any:
    ws = home / "ws"
    ws.mkdir(exist_ok=True)
    cfg = HermesConfig.resolve(overrides={"memory_nudge_interval": memory_interval})
    return create_hermes_agent(cfg, workspace=ws, session_id=sid, model=_OneToolThenAnswer(), aux_model=DemoReviewModel())


def _turn(text: str, sid: str) -> dict:
    return {"messages": [{"role": "user", "content": text}], "session_id": sid, "iteration_budget_remaining": 30}


def test_each_user_turn_is_one_user_row_in_order(tmp_hermes_home: Path):
    sid = "r187"
    agent = _agent(tmp_hermes_home, sid)
    cfg = {"configurable": {"thread_id": sid}}
    try:
        agent.invoke(_turn("What did the helm rollback change?", sid), config=cfg)
        agent.invoke(_turn("And the canary weights?", sid), config=cfg)
    finally:
        _close_agent_resources(agent)

    rows = _rows(tmp_hermes_home, sid)
    assert [r for r, _ in rows] == ["user", "assistant", "tool", "assistant"] * 2
    assert [c for r, c in rows if r == "user"] == ["What did the helm rollback change?", "And the canary weights?"]

    store = SqliteFtsStore(tmp_hermes_home / "state.db")
    try:
        hits = store.search_messages("helm rollback")
        assert any(h["role"] == "user" and h["session_id"] == sid for h in hits)
    finally:
        store.close()


def test_async_turn_records_the_user_message(tmp_hermes_home: Path):
    sid = "r187-async"
    agent = _agent(tmp_hermes_home, sid)
    try:
        asyncio.run(agent.ainvoke(_turn("profile the slow script", sid), config={"configurable": {"thread_id": sid}}))
    finally:
        _close_agent_resources(agent)
    assert _rows(tmp_hermes_home, sid)[0] == ("user", "profile the slow script")


def test_review_prompt_is_not_recorded_as_the_user(tmp_hermes_home: Path):
    """The reflection review sends its own HumanMessage to the aux model; that is not user input."""
    sid = "r187-review"
    agent = _agent(tmp_hermes_home, sid, memory_interval=1)
    try:
        agent.invoke(_turn("list the files", sid), config={"configurable": {"thread_id": sid}})
    finally:
        _close_agent_resources(agent)
    assert [c for r, c in _rows(tmp_hermes_home, sid) if r == "user"] == ["list the files"]


def test_replayed_history_is_not_recorded_again(tmp_path: Path):
    """Only the HumanMessages after the last agent/tool message are this turn's input.

    The checkpointed history comes back in state every turn; its user messages were
    recorded on their own turns and must not be written again.
    """
    store = SqliteFtsStore(tmp_path / "state.db")
    try:
        history = [HumanMessage(content="first question"), AIMessage(content="first answer")]
        state = {"session_id": "s", "messages": [*history, HumanMessage(content="second question")]}
        rec = HermesStateRecorderMiddleware(store)
        rec.before_agent(state, None)
        rec.before_agent(state, None)  # re-run on the same state
    finally:
        store.close()
    con = sqlite3.connect(tmp_path / "state.db")
    try:
        rows = con.execute("SELECT role, content FROM messages ORDER BY id").fetchall()
    finally:
        con.close()
    assert rows == [("user", "second question")]
