"""Tests for the keyless ``langstage-hermes search`` CLI over the FTS5 store (gh #79).

The FTS5 session store is a headline feature but had no human reader — only the
in-chat ``session_search`` tool (needs a live model + key). These tests pin the
three documented modes (DISCOVERY / SCROLL / BROWSE), the ``--json`` surface,
that hits carry session_id + message_id, and graceful handling of an
absent/empty store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.search.session_search import search_sessions_structured
from langstage_hermes.store.sqlite_fts import SqliteFtsStore


@pytest.fixture
def populated_home(tmp_hermes_home: Path) -> tuple[Path, dict[str, int]]:
    """A HERMES_HOME whose state.db holds a few searchable sessions."""
    ids: dict[str, int] = {}
    store = SqliteFtsStore(db_path=str(tmp_hermes_home / "state.db"))
    store.ensure_session("sess-a", source="user", title="Docker setup chat")
    ids["first_a"] = store.record_message("sess-a", "user", "set up docker compose for the api")
    ids["mid_a"] = store.record_message("sess-a", "assistant", "I'll write docker-compose.yml with api + redis")
    for i in range(5):
        store.record_message("sess-a", "user", f"tweak {i}: dockerfile layer caching")
        store.record_message("sess-a", "assistant", f"applied tweak {i} to the dockerfile")
    ids["last_a"] = store.record_message("sess-a", "assistant", "all dockerfile tweaks merged, deploy ready")

    store.ensure_session("sess-b", source="user", title="Kubernetes deploy")
    store.record_message("sess-b", "user", "deploy on kubernetes cluster")
    store.record_message("sess-b", "assistant", "kubernetes manifests written")

    # A reflection-fork (source=tool) — hidden by default in browse.
    store.ensure_session("sess-bg", source="tool", title="bg review")
    store.record_message("sess-bg", "user", "internal review of docker setup")
    store.close()
    return tmp_hermes_home, ids


# ── DISCOVERY ──────────────────────────────────────────────────────


def test_discovery_human(populated_home):
    res = CliRunner().invoke(cli, ["search", "docker"])
    assert res.exit_code == 0, res.output
    assert "sess-a" in res.output
    assert "#" in res.output  # a message id is printed
    assert "BM25" in res.output


def test_discovery_json_has_actionable_ids(populated_home):
    res = CliRunner().invoke(cli, ["search", "docker", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["mode"] == "discovery"
    assert data["count"] >= 1
    hit = next(h for h in data["results"] if h["session_id"] == "sess-a")
    assert isinstance(hit["message_id"], int)
    assert hit["session_id"] == "sess-a"


def test_discovery_limit_caps_results(populated_home):
    res = CliRunner().invoke(cli, ["search", "docker OR kubernetes", "--limit", "1", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["count"] == 1


# ── SCROLL ─────────────────────────────────────────────────────────


def test_scroll_shows_anchor(populated_home):
    _, ids = populated_home
    res = CliRunner().invoke(cli, ["search", "--session", "sess-a", "--around", str(ids["mid_a"]), "--window", "2"])
    assert res.exit_code == 0, res.output
    assert "sess-a" in res.output
    assert f"#{ids['mid_a']}" in res.output
    assert "anchor" in res.output


def test_scroll_json_marks_anchor(populated_home):
    _, ids = populated_home
    res = CliRunner().invoke(cli, ["search", "--session", "sess-a", "--around", str(ids["mid_a"]), "--window", "2", "--json"])
    data = json.loads(res.output)
    assert data["mode"] == "scroll"
    anchors = [m for m in data["messages"] if m["anchor"]]
    assert len(anchors) == 1
    assert anchors[0]["message_id"] == ids["mid_a"]


def test_scroll_missing_message_is_clean_error(populated_home):
    res = CliRunner().invoke(cli, ["search", "--session", "sess-a", "--around", "999999"])
    assert res.exit_code == 0, res.output
    assert "not in session" in res.output


# ── BROWSE ─────────────────────────────────────────────────────────


def test_browse_lists_sessions_and_hides_tool_forks(populated_home):
    res = CliRunner().invoke(cli, ["search", "--browse"])
    assert res.exit_code == 0, res.output
    assert "sess-a" in res.output
    assert "sess-b" in res.output
    # source=tool reflection forks stay hidden by default
    assert "sess-bg" not in res.output


def test_browse_when_no_query_is_default(populated_home):
    """No query + no scroll args → BROWSE (parity with the tool's inference)."""
    res = CliRunner().invoke(cli, ["search", "--json"])
    data = json.loads(res.output)
    assert data["mode"] == "browse"
    assert data["count"] >= 2


# ── empty / absent store ───────────────────────────────────────────


def test_absent_store_is_graceful(tmp_hermes_home):
    """No state.db yet → a clear message, exit 0, no traceback, no DB created."""
    res = CliRunner().invoke(cli, ["search", "anything"])
    assert res.exit_code == 0, res.output
    assert "No session store yet" in res.output
    assert not (tmp_hermes_home / "state.db").exists()  # read command must not create it


def test_absent_store_json_mode(tmp_hermes_home):
    res = CliRunner().invoke(cli, ["search", "anything", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["mode"] == "empty"


# ── structured helper directly ─────────────────────────────────────


def test_structured_scroll_rejects_current_lineage(populated_home):
    _, ids = populated_home
    store = SqliteFtsStore(db_path=str(populated_home[0] / "state.db"))
    try:
        out = search_sessions_structured(
            store,
            session_id="sess-a",
            around_message_id=ids["mid_a"],
            current_session_id="sess-a",
        )
    finally:
        store.close()
    assert out["mode"] == "scroll"
    assert "current session lineage" in out["error"]
