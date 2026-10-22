"""The recorder must persist what the user typed (gh #187).

``search`` and ``session_search`` read the ``messages`` table. Before this test,
only assistant and tool rows reached it; every user turn was missing, so the
highest-signal text in a conversation was unsearchable.
"""

from __future__ import annotations

import sqlite3

from langchain_core.messages import AIMessage, HumanMessage

from langstage_hermes.store.recorder import HermesStateRecorderMiddleware
from langstage_hermes.store.sqlite_fts import SqliteFtsStore


def _rows(home) -> list[tuple[str, str | None]]:
    conn = sqlite3.connect(str(home / "state.db"))
    try:
        return [
            (row[0], row[1])
            for row in conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                ("sess-187",),
            )
        ]
    finally:
        conn.close()


def test_before_agent_records_each_user_turn_once(tmp_hermes_home):
    store = SqliteFtsStore(db_path=tmp_hermes_home / "state.db")
    try:
        recorder = HermesStateRecorderMiddleware(store)
        state = {
            "session_id": "sess-187",
            "messages": [HumanMessage(content="Investigate the helm rollback last week", id="human-1")],
        }

        recorder.before_agent(state, runtime=None)
        recorder.before_agent(state, runtime=None)

        assert _rows(tmp_hermes_home) == [("user", "Investigate the helm rollback last week")]

        state["messages"] = [
            *state["messages"],
            AIMessage(content="The rollback came from the release gate", id="ai-1"),
        ]
        recorder.after_model(state, runtime=None)
        assert _rows(tmp_hermes_home)[-1] == ("assistant", "The rollback came from the release gate")
    finally:
        store.close()


def test_user_turn_is_findable_through_search(tmp_hermes_home):
    store = SqliteFtsStore(db_path=tmp_hermes_home / "state.db")
    try:
        recorder = HermesStateRecorderMiddleware(store)
        recorder.before_agent(
            {
                "session_id": "sess-187",
                "messages": [HumanMessage(content="Why did the deploy gate block the release?", id="human-2")],
            },
            runtime=None,
        )

        results = store.search_messages("deploy", role_filter=["user"])
        assert results, "the user's words should be indexed"
        assert "deploy" in (results[0].get("snippet") or "")
    finally:
        store.close()
