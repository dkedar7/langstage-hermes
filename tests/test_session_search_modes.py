"""Tests for ``session_search`` — DISCOVERY / SCROLL / BROWSE modes."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepagent_hermes.search.session_search import (
    make_session_search_tool,
    run_session_search,
)
from deepagent_hermes.store.sqlite_fts import SqliteFtsStore

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_hermes_home: Path) -> SqliteFtsStore:
    s = SqliteFtsStore()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def seeded_store(store: SqliteFtsStore) -> tuple[SqliteFtsStore, dict[str, int]]:
    """Three sessions with known docker / kubernetes content plus
    a lineage pair (sess-old → sess-new via parent_session_id).

    Returns the store plus a dict of message ids the tests rely on:
    {
        "docker_compose_id": <id of the auth assistant turn in sess-a>,
        "first_a": <first message id in sess-a>,
        ...
    }
    """
    ids: dict[str, int] = {}

    store.ensure_session("sess-a", source="user", title="Docker setup chat")
    ids["first_a"] = store.record_message("sess-a", "user", "set up docker compose for the api")
    ids["docker_assistant"] = store.record_message(
        "sess-a",
        "assistant",
        "I'll write docker-compose.yml with the api + redis services",
    )
    for i in range(8):
        store.record_message("sess-a", "user", f"tweak {i}: dockerfile layer caching")
        store.record_message("sess-a", "assistant", f"applied tweak {i} to the dockerfile")
    ids["docker_resolution"] = store.record_message("sess-a", "assistant", "all dockerfile tweaks merged, deploy ready")

    store.ensure_session("sess-b", source="user", title="Kubernetes deploy")
    store.record_message("sess-b", "user", "deploy on kubernetes")
    store.record_message("sess-b", "assistant", "kubernetes manifests written for the cluster")
    for i in range(6):
        store.record_message("sess-b", "user", f"scale replicas to {i + 1} on the cluster")
        store.record_message("sess-b", "assistant", f"replicas now {i + 1} in kubernetes")

    # Lineage: sess-old → sess-new (compression-style split)
    store.ensure_session("sess-old", source="user", title="Long context")
    store.record_message("sess-old", "user", "compare docker swarm vs kubernetes scheduler")
    store.record_message(
        "sess-old",
        "assistant",
        "swarm is simpler; kubernetes more capable",
    )
    store.ensure_session(
        "sess-new",
        source="user",
        parent_session_id="sess-old",
        title="Compressed continuation",
    )
    store.record_message(
        "sess-new",
        "user",
        "ok continue: also bring in nomad in that comparison",
    )

    # Background reflection-fork session — should be hidden by default.
    store.ensure_session("sess-bg", source="tool", title="bg review")
    store.record_message("sess-bg", "user", "internal review: reflect on docker setup decisions")

    return store, ids


# ---------------------------------------------------------------------------
# DISCOVERY mode
# ---------------------------------------------------------------------------


def test_discovery_returns_relevant_sessions(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, _ = seeded_store
    out = run_session_search(store, query="docker-compose.yml")
    assert "## session_search (discover)" in out
    # The docker-heavy session must surface
    assert "sess-a" in out
    # Should include the window with anchor marker
    assert "← anchor" in out


def test_discovery_emits_bookends_for_mid_session_match(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    """A query that only matches a row near the END of the session must
    produce a non-empty ``bookend_start`` (opening of conversation)."""
    store, _ = seeded_store
    # 'merged' only appears in the very last assistant message of sess-a.
    out = run_session_search(store, query="merged")
    assert "## session_search (discover)" in out
    assert "sess-a" in out
    assert "bookend_start" in out


def test_discovery_dedupes_lineage(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, _ = seeded_store
    out = run_session_search(store, query="kubernetes")
    # sess-old + sess-new are one lineage. They should NOT both surface
    # as separate entries; only the lineage root (sess-old).
    n_root_headers = out.count("### Session `sess-old`")
    n_child_headers = out.count("### Session `sess-new`")
    # The anchor lives in the message — which exact id matched depends
    # on FTS5 ranking — but we must not get both root + child as separate
    # session entries.
    assert (n_root_headers + n_child_headers) <= 1, out


def test_discovery_excludes_tool_sources_by_default(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, _ = seeded_store
    out = run_session_search(store, query="reflect")
    # sess-bg is tagged source='tool' and must be filtered out
    assert "sess-bg" not in out


def test_discovery_excludes_current_session_lineage(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, _ = seeded_store
    out = run_session_search(
        store,
        query="docker",
        current_session_id="sess-a",
    )
    # The current session's lineage root is sess-a itself.
    assert "Session `sess-a`" not in out


def test_discovery_empty_query_falls_through_to_browse(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, _ = seeded_store
    out = run_session_search(store, query="   ")
    assert "(browse)" in out


def test_discovery_no_matches_returns_friendly_message(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, _ = seeded_store
    out = run_session_search(store, query="quetzalcoatl-flavoured zarf")
    assert "No matching sessions found" in out


def test_discovery_handles_cjk(store: SqliteFtsStore) -> None:
    store.ensure_session("cjk-s", source="user", title="日本語 session")
    store.record_message(
        "cjk-s",
        "user",
        "今日は素晴らしい日本語をテストしています",
    )
    out = run_session_search(store, query="日本語をテスト")
    assert "## session_search (discover)" in out
    assert "cjk-s" in out


# ---------------------------------------------------------------------------
# SCROLL mode
# ---------------------------------------------------------------------------


def test_scroll_returns_window_around_anchor(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, ids = seeded_store
    anchor = ids["docker_assistant"]
    out = run_session_search(
        store,
        session_id="sess-a",
        around_message_id=anchor,
        window=3,
    )
    assert "## session_search (scroll)" in out
    assert "sess-a" in out
    assert f"#{anchor}" in out
    assert "← anchor" in out


def test_scroll_clamps_window(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, ids = seeded_store
    anchor = ids["docker_assistant"]
    out = run_session_search(
        store,
        session_id="sess-a",
        around_message_id=anchor,
        window=500,  # absurd → clamped to 20
    )
    assert "±20 window" in out


def test_scroll_inside_current_session_is_rejected(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, ids = seeded_store
    out = run_session_search(
        store,
        session_id="sess-a",
        around_message_id=ids["docker_assistant"],
        current_session_id="sess-a",
    )
    assert "Error" in out
    assert "rejected" in out.lower()


def test_scroll_rejects_anchor_in_lineage(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    """If the active session is sess-new, scrolling into sess-old (its
    parent) should also be rejected — same lineage, same context."""
    store, _ = seeded_store
    # Take an arbitrary message id in sess-old
    row = store._conn.execute("SELECT id FROM messages WHERE session_id = 'sess-old' LIMIT 1").fetchone()
    out = run_session_search(
        store,
        session_id="sess-old",
        around_message_id=row[0],
        current_session_id="sess-new",
    )
    assert "rejected" in out.lower()


def test_scroll_unknown_session_returns_error(
    store: SqliteFtsStore,
) -> None:
    out = run_session_search(
        store,
        session_id="not-a-real-session",
        around_message_id=999,
    )
    assert "Error" in out
    assert "not found" in out


def test_scroll_takes_precedence_over_query(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    """Spec: SCROLL beats DISCOVERY when both look set."""
    store, ids = seeded_store
    out = run_session_search(
        store,
        query="kubernetes",
        session_id="sess-a",
        around_message_id=ids["docker_assistant"],
    )
    assert "(scroll)" in out
    assert "(discover)" not in out


# ---------------------------------------------------------------------------
# BROWSE mode
# ---------------------------------------------------------------------------


def test_browse_lists_recent_root_sessions(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, _ = seeded_store
    out = run_session_search(store)
    assert "## session_search (browse)" in out
    # Root sessions should appear; descendants (sess-new) should not
    # show up as standalone entries (parent_session_id IS NULL filter).
    assert "### sess-a" in out
    assert "### sess-b" in out
    assert "### sess-old" in out
    assert "### sess-new" not in out


def test_browse_hides_tool_sources(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, _ = seeded_store
    out = run_session_search(store)
    assert "### sess-bg" not in out


def test_browse_skips_current_session(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, _ = seeded_store
    out = run_session_search(store, current_session_id="sess-a")
    assert "### sess-a" not in out


def test_browse_empty_store(store: SqliteFtsStore) -> None:
    out = run_session_search(store)
    assert "No prior sessions" in out


# ---------------------------------------------------------------------------
# tool factory
# ---------------------------------------------------------------------------


def test_make_session_search_tool_invocation(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, ids = seeded_store
    tool = make_session_search_tool(store)
    result = tool.invoke(
        {
            "query": "",
            "session_id": "sess-a",
            "around_message_id": ids["docker_assistant"],
        }
    )
    assert "(scroll)" in result
    assert f"#{ids['docker_assistant']}" in result


def test_tool_uses_current_session_id_getter(seeded_store: tuple[SqliteFtsStore, dict[str, int]]) -> None:
    store, _ = seeded_store
    tool = make_session_search_tool(store, current_session_id_getter=lambda: "sess-a")
    result = tool.invoke({"query": "docker"})
    assert "sess-a" not in result.split("###", 1)[-1]
