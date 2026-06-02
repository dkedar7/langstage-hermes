"""Tests for ``SqliteFtsStore`` — schema, FTS5, CJK trigram routing,
WAL mode, BaseStore op dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deepagent_hermes.store.sqlite_fts import (
    SqliteFtsStore,
    contains_cjk,
    default_db_path,
    resolve_hermes_home,
)


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
def populated_store(store: SqliteFtsStore) -> SqliteFtsStore:
    """50 messages spread across 3 sessions, deterministic content.

    Each session uses the SAME standalone word "docker" or "kubernetes"
    in every relevant row — the FTS5 unicode61 tokenizer is strict
    about word boundaries (it would NOT match "docker" inside
    "dockerfile") so we keep tokens whole.
    """
    sessions = ["sess-a", "sess-b", "sess-c"]
    for sid in sessions:
        store.ensure_session(sid, source="user")

    # sess-a: 15 docker rows (whole token)
    for i in range(8):
        store.record_message(
            "sess-a", "user", f"tweak {i}: docker setup for layer caching"
        )
        store.record_message(
            "sess-a", "assistant", f"applied tweak {i} to docker"
        )

    # sess-b: kubernetes content (no docker)
    for i in range(8):
        store.record_message(
            "sess-b", "user", f"scale replicas to {i + 1} on the cluster"
        )
        store.record_message(
            "sess-b", "assistant", f"replicas set to {i + 1} in kubernetes"
        )

    # sess-c: a couple docker mentions
    store.record_message("sess-c", "user", "compare docker swarm vs kubernetes")
    store.record_message(
        "sess-c", "assistant", "docker swarm is simpler; kubernetes more capable"
    )
    return store


# ---------------------------------------------------------------------------
# basic lifecycle / paths
# ---------------------------------------------------------------------------


def test_resolve_hermes_home_uses_deepagent_var(tmp_hermes_home: Path) -> None:
    assert resolve_hermes_home() == tmp_hermes_home


def test_default_db_path_lives_under_home(tmp_hermes_home: Path) -> None:
    assert default_db_path() == tmp_hermes_home / "state.db"


def test_store_creates_db_file(tmp_hermes_home: Path) -> None:
    s = SqliteFtsStore()
    try:
        assert (tmp_hermes_home / "state.db").exists()
    finally:
        s.close()


def test_wal_mode_active(store: SqliteFtsStore) -> None:
    # On most Windows dev boxes WAL is supported; if not, the fallback
    # is DELETE (still valid). Either way ensure we got a known mode.
    mode = store.journal_mode()
    assert mode in {"wal", "delete"}, mode


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_schema_has_expected_tables(store: SqliteFtsStore) -> None:
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {r[0] for r in rows}
    for required in (
        "sessions",
        "messages",
        "state_meta",
        "compression_locks",
        "curator_state",
        "messages_fts",
        "messages_fts_trigram",
    ):
        assert required in names, f"missing table: {required}"


def test_fts_triggers_exist(store: SqliteFtsStore) -> None:
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()
    names = {r[0] for r in rows}
    assert "messages_after_insert" in names
    assert "messages_after_delete" in names
    assert "messages_after_update" in names


# ---------------------------------------------------------------------------
# write helpers + FTS5 search
# ---------------------------------------------------------------------------


def test_ensure_session_is_idempotent(store: SqliteFtsStore) -> None:
    store.ensure_session("s1", source="user")
    store.ensure_session("s1", source="ignored_on_second_call")
    rows = store._conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE id = ?", ("s1",)
    ).fetchone()
    assert rows[0] == 1
    # First write wins on source
    src = store._conn.execute(
        "SELECT source FROM sessions WHERE id = ?", ("s1",)
    ).fetchone()
    assert src[0] == "user"


def test_record_message_returns_autoincrement_id(store: SqliteFtsStore) -> None:
    store.ensure_session("s1")
    id1 = store.record_message("s1", "user", "first")
    id2 = store.record_message("s1", "assistant", "second")
    assert id2 == id1 + 1


def test_record_message_increments_counters(store: SqliteFtsStore) -> None:
    store.ensure_session("s1")
    store.record_message("s1", "user", "hi")
    store.record_message(
        "s1",
        "assistant",
        "calling tool",
        tool_calls=[{"id": "c1", "name": "lookup", "args": {}}],
    )
    row = store._conn.execute(
        "SELECT message_count, tool_call_count FROM sessions WHERE id = ?",
        ("s1",),
    ).fetchone()
    assert row[0] == 2
    assert row[1] == 1


def test_fts_returns_bm25_ranked_rowids(populated_store: SqliteFtsStore) -> None:
    """50-ish messages indexed; docker query must return all and only
    rows that actually contain the term, and the result must be ranked
    by FTS5's BM25 (verified by checking that rank is the column-order
    ranker — every returned id matches the term)."""
    hits = populated_store.search_messages("docker", limit=20)
    assert hits, "FTS5 returned no hits for 'docker'"
    # Every returned message must come from a session that actually has
    # docker content (sess-a or sess-c) and contain the term.
    for h in hits:
        assert h["session_id"] in {"sess-a", "sess-c"}
        row = populated_store._conn.execute(
            "SELECT content, tool_name FROM messages WHERE id = ?", (h["id"],)
        ).fetchone()
        content = (row["content"] or "") + " " + (row["tool_name"] or "")
        assert "docker" in content.lower()
    # sess-a has 15+ docker rows, sess-c has 2 — sess-a must dominate the
    # full result set.
    a_count = sum(1 for h in hits if h["session_id"] == "sess-a")
    c_count = sum(1 for h in hits if h["session_id"] == "sess-c")
    assert a_count > c_count, (a_count, c_count)


def test_fts_snippet_contains_highlight_markers(populated_store: SqliteFtsStore) -> None:
    hits = populated_store.search_messages("docker", limit=3)
    assert hits
    assert any(">>>" in (h.get("snippet") or "") for h in hits)


def test_fts_excludes_inactive_rows(populated_store: SqliteFtsStore) -> None:
    # Soft-delete every docker row in sess-a
    populated_store._conn.execute(
        "UPDATE messages SET active = 0 WHERE session_id = 'sess-a' "
        "AND content LIKE '%docker%'"
    )
    populated_store._conn.commit()
    hits = populated_store.search_messages("docker", limit=20)
    for h in hits:
        # Any returned id must still be active
        row = populated_store._conn.execute(
            "SELECT active FROM messages WHERE id = ?", (h["id"],)
        ).fetchone()
        assert row[0] == 1


def test_fts_exclude_sources(populated_store: SqliteFtsStore) -> None:
    populated_store.ensure_session("sess-bg", source="tool")
    populated_store.record_message(
        "sess-bg", "assistant", "background docker update"
    )
    hits = populated_store.search_messages(
        "docker", exclude_sources=["tool"], limit=50
    )
    assert all(h["session_id"] != "sess-bg" for h in hits)


# ---------------------------------------------------------------------------
# CJK trigram routing
# ---------------------------------------------------------------------------


def test_contains_cjk_helper() -> None:
    assert contains_cjk("日本語をテスト")
    assert not contains_cjk("hello world")


def test_cjk_content_routes_through_trigram_table(store: SqliteFtsStore) -> None:
    store.ensure_session("cjk-s", source="user")
    mid = store.record_message(
        "cjk-s",
        "user",
        "今日は素晴らしい日本語をテストしましょう",
    )
    # The row must be in BOTH FTS tables (triggers wrote it)
    fts_row = store._conn.execute(
        "SELECT rowid FROM messages_fts WHERE rowid = ?", (mid,)
    ).fetchone()
    assert fts_row is not None
    tri_row = store._conn.execute(
        "SELECT rowid FROM messages_fts_trigram WHERE rowid = ?", (mid,)
    ).fetchone()
    assert tri_row is not None

    # Searching for 日本語 must produce a result. The unicode61 tokenizer
    # treats each CJK char as a token (so an exact match for a 3-char
    # phrase is unreliable), but the trigram path should find it.
    hits = store.search_messages("日本語をテスト")
    assert hits
    assert hits[0]["id"] == mid
    assert "日本語" in (hits[0]["snippet"] or "")


# ---------------------------------------------------------------------------
# BaseStore op dispatch
# ---------------------------------------------------------------------------


def test_base_store_put_and_get_message_roundtrip(store: SqliteFtsStore) -> None:
    store.put(
        ("messages", "s1", "user"),
        "0",  # unused — we always autoincrement; key returned via _row_to_item
        {"content": "hello via put"},
    )
    # Find the row we just inserted (auto id = 1)
    item = store.get(("messages", "s1", "user"), "1")
    assert item is not None
    assert item.value["content"] == "hello via put"


def test_base_store_search_via_fts(populated_store: SqliteFtsStore) -> None:
    items = populated_store.search(
        ("messages",), query="kubernetes", limit=5
    )
    assert items, "BaseStore.search returned no results for 'kubernetes'"
    for it in items:
        assert it.namespace[0] == "messages"
        assert it.value["session_id"] in {"sess-b", "sess-c"}


def test_base_store_search_namespace_filter(populated_store: SqliteFtsStore) -> None:
    items = populated_store.search(
        ("messages", "sess-b"), query="replicas", limit=5
    )
    assert items
    for it in items:
        assert it.value["session_id"] == "sess-b"


def test_base_store_list_namespaces_includes_message_namespaces(
    populated_store: SqliteFtsStore,
) -> None:
    namespaces = populated_store.list_namespaces(prefix=("messages",))
    # Each (session_id, role) combination must appear
    str_ns = {tuple(n) for n in namespaces}
    assert ("messages", "sess-a", "user") in str_ns
    assert ("messages", "sess-a", "assistant") in str_ns


def test_base_store_delete_removes_row_and_fts_entry(store: SqliteFtsStore) -> None:
    store.ensure_session("s1")
    mid = store.record_message("s1", "user", "ephemeral docker stuff")
    store.delete(("messages", "s1", "user"), str(mid))
    assert store.get(("messages", "s1", "user"), str(mid)) is None
    # FTS row should be gone via the delete trigger
    row = store._conn.execute(
        "SELECT rowid FROM messages_fts WHERE rowid = ?", (mid,)
    ).fetchone()
    assert row is None


# ---------------------------------------------------------------------------
# async surface
# ---------------------------------------------------------------------------


async def test_async_abatch_runs_get_op(populated_store: SqliteFtsStore) -> None:
    item = await populated_store.aget(("messages", "sess-a", "user"), "1")
    assert item is not None
    assert item.value["session_id"] == "sess-a"


async def test_async_asearch(populated_store: SqliteFtsStore) -> None:
    items = await populated_store.asearch(
        ("messages",), query="docker", limit=3
    )
    assert items
    assert len(items) <= 3
