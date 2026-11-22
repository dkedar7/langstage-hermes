"""gh #233: ordinary punctuation in a search query used to reach FTS5 MATCH raw.

SQLite raised a syntax error, ``search_messages`` swallowed it as zero hits, and
both the ``search`` CLI and the agent's ``session_search`` tool reported "no
match". Punctuation now acts as a separator (as the tokenizer treats it when
indexing), the documented operators keep working, and a query that truly can't
run is reported as such.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.search.session_search import make_session_search_tool
from langstage_hermes.store.sqlite_fts import (
    FtsQueryError,
    SqliteFtsStore,
    _sanitize_fts5_query,
)


@pytest.fixture
def store(tmp_hermes_home: Path):
    s = SqliteFtsStore()
    s.ensure_session("s-slow", source="user")
    s.record_message("s-slow", "user", "what's slow about python's startup? my python script is slow")
    s.record_message(
        "s-slow",
        "assistant",
        "The user's slow python script: run python3 -m cProfile to profile it. The dev server on "
        "localhost:8080 logged error: connection refused. Notes in ~/scripts and a/b tests; "
        "mail user@example.org.",
    )
    s.ensure_session("s-other", source="user")
    s.record_message("s-other", "user", "rollback staging after the deploy")
    s.record_message("s-other", "assistant", "kubernetes rollback done on staging")
    try:
        yield s
    finally:
        s.close()


def _sessions(store: SqliteFtsStore, query: str) -> set[str]:
    return {h["session_id"] for h in store.search_messages(query)}


@pytest.mark.parametrize(
    "query",
    [
        "slow python?",
        "python's",
        "user's",
        "what's slow",
        "slow, python",
        "python:",
        "slow; python",
        "python!",
        "error: connection refused",
        "localhost:8080",
        "python3 -m cProfile",
        "~/scripts",
        "user@example",
        "[python]",
        "<python>",
        "python = slow",
        "#python",
        "slow -python",
        "a/b python",
    ],
)
def test_punctuated_query_finds_the_session(store: SqliteFtsStore, query: str) -> None:
    assert "s-slow" in _sessions(store, query), _sanitize_fts5_query(query)


@pytest.mark.parametrize(
    "query",
    ["$HOME", "50%", "python's", "localhost:8080", "~/scripts", "?", "-", "x y,", "a:b:c", "it's -m"],
)
def test_sanitized_query_never_raises_in_fts5(store: SqliteFtsStore, query: str) -> None:
    cleaned = _sanitize_fts5_query(query)
    if not cleaned:
        return
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
    conn.execute("SELECT * FROM t WHERE t MATCH ?", (cleaned,)).fetchall()


def test_operators_still_work(store: SqliteFtsStore) -> None:
    assert _sessions(store, "profile OR kubernetes") == {"s-slow", "s-other"}
    assert _sessions(store, "rollback AND staging") == {"s-other"}
    assert _sessions(store, "rollback NOT kubernetes") == {"s-other"}  # the user row
    assert _sessions(store, '"connection refused"') == {"s-slow"}
    assert _sessions(store, "kube*") == {"s-other"}
    assert _sessions(store, "prof*?") == {"s-slow"}


def test_phrase_order_is_kept(store: SqliteFtsStore) -> None:
    # A quoted phrase matches only in order; reversed, no hit.
    assert _sessions(store, '"refused connection"') == set()


def test_stray_operators_dropped() -> None:
    assert _sanitize_fts5_query("OR python") == "python"
    assert _sanitize_fts5_query("python AND") == "python"
    assert _sanitize_fts5_query("python AND OR slow") == "python AND slow"
    assert _sanitize_fts5_query("? , -") == ""


def test_query_fts5_cannot_run_raises(store: SqliteFtsStore, monkeypatch) -> None:
    # Simulate a query FTS5 rejects in both forms: the sanitized one and the retry.
    import langstage_hermes.store.sqlite_fts as mod

    monkeypatch.setattr(mod, "_sanitize_fts5_query", lambda q: "NEAR(")
    monkeypatch.setattr(mod, "_plain_words_query", lambda q: "AND")
    with pytest.raises(FtsQueryError):
        store.search_messages("anything")


def test_retry_with_plain_words_when_first_form_fails(store: SqliteFtsStore, monkeypatch) -> None:
    import langstage_hermes.store.sqlite_fts as mod

    monkeypatch.setattr(mod, "_sanitize_fts5_query", lambda q: "NEAR(")
    assert {h["session_id"] for h in store.search_messages("slow python?")} == {"s-slow"}


def test_session_search_tool_finds_punctuated_query(store: SqliteFtsStore) -> None:
    tool = make_session_search_tool(store, current_session_id_getter=lambda: "x")
    out = tool.invoke({"query": "user's slow python?"})
    assert "No matching sessions found" not in out
    assert "s-slow" in out


def test_session_search_tool_reports_rejected_query(store: SqliteFtsStore, monkeypatch) -> None:
    import langstage_hermes.store.sqlite_fts as mod

    monkeypatch.setattr(mod, "_sanitize_fts5_query", lambda q: "NEAR(")
    monkeypatch.setattr(mod, "_plain_words_query", lambda q: "AND")
    tool = make_session_search_tool(store, current_session_id_getter=lambda: "x")
    out = tool.invoke({"query": "anything"})
    assert "could not be run" in out
    assert "No matching sessions found" not in out


def test_cli_search_punctuated_query(store: SqliteFtsStore) -> None:
    from langstage_hermes.cli import cli

    store.close()
    r = CliRunner().invoke(cli, ["search", "slow python?"])
    assert r.exit_code == 0, r.output
    assert "s-slow" in r.output
    assert "No sessions match" not in r.output


def test_cli_search_rejected_query_exits_1(store: SqliteFtsStore, monkeypatch) -> None:
    import langstage_hermes.store.sqlite_fts as mod
    from langstage_hermes.cli import cli

    store.close()
    monkeypatch.setattr(mod, "_sanitize_fts5_query", lambda q: "NEAR(")
    monkeypatch.setattr(mod, "_plain_words_query", lambda q: "AND")
    r = CliRunner().invoke(cli, ["search", "anything"])
    assert r.exit_code == 1, r.output
    assert "could not be run" in r.output
    assert "No sessions match" not in r.output

    r = CliRunner().invoke(cli, ["search", "anything", "--json"])
    assert r.exit_code == 1
    payload = json.loads(r.output)
    assert payload["error"] and payload["results"] == []
