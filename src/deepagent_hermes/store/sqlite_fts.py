"""SQLite + FTS5 BaseStore implementation for deepagent-hermes.

Persistent message store and session-search index backing
``session_search`` (SPEC §11.3, D1) and the recorder middleware
(SPEC §13.3). Schema is verbatim from Hermes ``hermes_state.py`` so
existing Hermes ``state.db`` files are forward-compatible.

This is *not* the LangGraph checkpointer — that lives in the same DB
file but uses different tables managed by ``langgraph-checkpoint-sqlite``.
Both coexist cleanly.

Mapping onto ``BaseStore`` (langgraph):

- ``namespace=("messages", session_id, role)`` + ``key=str(message_id)``
  → row in the ``messages`` table.
- ``search(namespace_prefix=("messages",), query="...")`` runs an FTS5
  BM25 query over ``messages_fts`` (or ``messages_fts_trigram`` when
  the query carries CJK chars).
- Convenience helpers ``record_message`` / ``ensure_session`` /
  ``get_messages_around`` / ``get_anchored_view`` / ``list_recent_sessions``
  / ``resolve_to_lineage_root`` are used by the recorder middleware and
  the ``session_search`` tool — they sit outside the BaseStore API
  because the BaseStore namespace/key model is too tight for the rich
  session metadata Hermes needs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.store.base import (
    BaseStore,
    GetOp,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema — verbatim from hermes_state.py (subset; we don't carry the migration
# chain because we're starting fresh).
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    cwd TEXT,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL DEFAULT 0,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    rewind_count INTEGER DEFAULT 0,
    archived INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    active INTEGER DEFAULT 1,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS compression_locks (
    session_id TEXT PRIMARY KEY,
    holder TEXT,
    acquired_at REAL,
    expires_at REAL
);

CREATE TABLE IF NOT EXISTS curator_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_session_active
    ON messages(session_id, active, timestamp);
"""

# FTS5 virtual tables + triggers. content is the indexed payload; we
# concatenate content + tool_name + tool_calls so a search for a tool
# name surfaces both the assistant call and the tool result.
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content);

CREATE TRIGGER IF NOT EXISTS messages_after_insert
AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' ||
        COALESCE(new.tool_name, '') || ' ' ||
        COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_after_delete
AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_after_update
AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' ||
        COALESCE(new.tool_name, '') || ' ' ||
        COALESCE(new.tool_calls, '')
    );
END;
"""

FTS_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content, tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_trigram_after_insert
AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' ||
        COALESCE(new.tool_name, '') || ' ' ||
        COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_trigram_after_delete
AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_trigram_after_update
AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' ||
        COALESCE(new.tool_name, '') || ' ' ||
        COALESCE(new.tool_calls, '')
    );
END;
"""


# ---------------------------------------------------------------------------
# Resolution: where on disk does state.db live?
# ---------------------------------------------------------------------------

def resolve_hermes_home() -> Path:
    """Resolve the deepagent-hermes home directory.

    Precedence: ``DEEPAGENT_HERMES_HOME`` > ``HERMES_HOME`` >
    ``~/.deepagent-hermes/``. Directory is created if missing.
    """
    raw = os.environ.get("DEEPAGENT_HERMES_HOME") or os.environ.get("HERMES_HOME")
    home = Path(raw) if raw else Path.home() / ".deepagent-hermes"
    home.mkdir(parents=True, exist_ok=True)
    return home


def default_db_path() -> Path:
    return resolve_hermes_home() / "state.db"


# ---------------------------------------------------------------------------
# CJK detection (verbatim ported from Hermes)
# ---------------------------------------------------------------------------

def _is_cjk_codepoint(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF        # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF     # CJK Extension A
        or 0x20000 <= cp <= 0x2A6DF   # CJK Extension B
        or 0x3000 <= cp <= 0x303F     # CJK Symbols and Punctuation
        or 0x3040 <= cp <= 0x309F     # Hiragana
        or 0x30A0 <= cp <= 0x30FF     # Katakana
        or 0xAC00 <= cp <= 0xD7AF     # Hangul Syllables
    )


def contains_cjk(text: str) -> bool:
    """True if any CJK character appears in ``text``."""
    return any(_is_cjk_codepoint(ord(ch)) for ch in text)


def _count_cjk(text: str) -> int:
    return sum(1 for ch in text if _is_cjk_codepoint(ord(ch)))


# ---------------------------------------------------------------------------
# FTS5 query sanitization (subset of Hermes's logic — enough for our tests
# and typical agent-generated queries).
# ---------------------------------------------------------------------------

def _sanitize_fts5_query(query: str) -> str:
    """Make a user-typed query safe for FTS5 MATCH.

    - Preserve balanced double-quoted phrases.
    - Strip unmatched FTS5-special characters.
    - Wrap dotted/hyphenated tokens in quotes so the unicode61 tokenizer
      doesn't split them into separate AND-joined terms.
    """
    if not query:
        return ""

    quoted_parts: list[str] = []

    def _preserve_quoted(match: re.Match[str]) -> str:
        quoted_parts.append(match.group(0))
        return f"\x00Q{len(quoted_parts) - 1}\x00"

    sanitized = re.sub(r'"[^"]*"', _preserve_quoted, query)

    # Drop unmatched FTS5-special chars
    sanitized = re.sub(r'[+{}()\"^]', " ", sanitized)

    # Collapse repeated * and drop leading *
    sanitized = re.sub(r"\*+", "*", sanitized)
    sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

    # Drop dangling boolean operators
    sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
    sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())

    # Quote dotted/hyphenated tokens
    sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)

    for i, quoted in enumerate(quoted_parts):
        sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

    return sanitized.strip()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class SqliteFtsStore(BaseStore):
    """SQLite-backed ``BaseStore`` with FTS5 search over messages.

    Thread-safe under WAL mode for the typical
    "multiple readers + one writer" agent pattern. Writes use
    ``BEGIN IMMEDIATE`` + application-level jitter retry (20-150 ms)
    on ``sqlite3.OperationalError`` to dodge convoy-style lock storms.

    Async surface (``aget`` / ``aput`` / ``asearch`` / ``adelete`` /
    ``alist_namespaces`` / ``abatch``) is implemented by running the
    sync method in the default executor. SQLite calls are short and the
    contention is on the WAL lock, not the GIL, so a real async driver
    would add complexity without throughput gain.
    """

    supports_ttl = False

    # Write-contention tuning — matches Hermes.
    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020
    _WRITE_RETRY_MAX_S = 0.150

    __slots__ = (
        "_conn",
        "_db_path",
        "_fts_enabled",
        "_lock",
        "_trigram_enabled",
    )

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=1.0,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            # Fall back to DELETE on NFS/SMB/etc; tests still pass.
            try:
                self._conn.execute("PRAGMA journal_mode=DELETE")
            except sqlite3.OperationalError:
                pass
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._fts_enabled = False
        self._trigram_enabled = False
        self._init_schema()

    # ── lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> SqliteFtsStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── schema bootstrap ────────────────────────────────────────────

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(SCHEMA_SQL)
        # Probe FTS5 availability — the bundled python.org sqlite ships it,
        # but a hand-rolled Python build might not.
        try:
            cur.execute(
                "CREATE VIRTUAL TABLE temp._fts5_probe USING fts5(x)"
            )
            cur.execute("DROP TABLE temp._fts5_probe")
            fts5_available = True
        except sqlite3.OperationalError as exc:
            logger.warning(
                "SQLite FTS5 unavailable; session search disabled (%s)", exc
            )
            fts5_available = False

        if fts5_available:
            try:
                cur.executescript(FTS_SQL)
                self._fts_enabled = True
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 (unicode61) init failed: %s", exc)
            # Trigram is optional; absence just disables CJK trigram routing.
            try:
                cur.executescript(FTS_TRIGRAM_SQL)
                self._trigram_enabled = True
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 trigram init failed: %s", exc)
        self._conn.commit()

    # ── write helper with jitter retry ──────────────────────────────

    def _execute_write(self, fn):
        last_err: Exception | None = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                return result
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if "locked" in msg or "busy" in msg:
                    last_err = exc
                    if attempt < self._WRITE_MAX_RETRIES - 1:
                        time.sleep(
                            random.uniform(
                                self._WRITE_RETRY_MIN_S,
                                self._WRITE_RETRY_MAX_S,
                            )
                        )
                        continue
                raise
        raise last_err or sqlite3.OperationalError(
            "database is locked after max retries"
        )

    # ── journal mode probe (for tests) ──────────────────────────────

    def journal_mode(self) -> str:
        with self._lock:
            row = self._conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower() if row else ""

    # ====================================================================
    # Hermes-shaped helpers — recorder + session_search consume these.
    # ====================================================================

    def ensure_session(
        self,
        session_id: str,
        *,
        source: str = "user",
        **fields: Any,
    ) -> str:
        """Upsert a row into ``sessions``. Returns ``session_id``.

        ``fields`` may include any column on ``sessions`` — anything
        unknown is silently ignored to keep callers loose-coupled with
        schema additions.
        """
        if not session_id:
            raise ValueError("session_id is required")

        # Whitelist columns we actually have, drop the rest.
        allowed = {
            "user_id",
            "model",
            "model_config",
            "system_prompt",
            "parent_session_id",
            "cwd",
            "title",
        }
        clean = {k: v for k, v in fields.items() if k in allowed}
        if isinstance(clean.get("model_config"), dict):
            clean["model_config"] = json.dumps(clean["model_config"])

        cols = ["id", "source", "started_at", *list(clean.keys())]
        placeholders = ",".join("?" for _ in cols)
        col_sql = ",".join(cols)
        values: list[Any] = [session_id, source, time.time()]
        values.extend(clean.values())

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"INSERT OR IGNORE INTO sessions ({col_sql}) "
                f"VALUES ({placeholders})",
                values,
            )

        self._execute_write(_do)
        return session_id

    def end_session(
        self,
        session_id: str,
        *,
        end_reason: str | None = None,
        message_count: int | None = None,
        tool_call_count: int | None = None,
    ) -> None:
        """Mark a session ended; updates totals when provided."""
        sets: list[str] = ["ended_at = ?"]
        params: list[Any] = [time.time()]
        if end_reason is not None:
            sets.append("end_reason = ?")
            params.append(end_reason)
        if message_count is not None:
            sets.append("message_count = ?")
            params.append(message_count)
        if tool_call_count is not None:
            sets.append("tool_call_count = ?")
            params.append(tool_call_count)
        params.append(session_id)

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"UPDATE sessions SET {', '.join(sets)} "
                f"WHERE id = ? AND ended_at IS NULL",
                params,
            )

        self._execute_write(_do)

    def record_message(
        self,
        session_id: str,
        role: str,
        content: str | None,
        *,
        tool_calls: Any = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        token_count: int | None = None,
        finish_reason: str | None = None,
        reasoning: str | None = None,
        platform_message_id: str | None = None,
    ) -> int:
        """Append a message row. Returns the AUTOINCREMENT id.

        Bumps ``sessions.message_count`` (and ``tool_call_count`` when
        ``tool_calls`` is non-empty) atomically.
        """
        tool_calls_json: str | None
        if tool_calls is None:
            tool_calls_json = None
            num_tool_calls = 0
        elif isinstance(tool_calls, list):
            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            num_tool_calls = len(tool_calls)
        else:
            tool_calls_json = json.dumps(tool_calls)
            num_tool_calls = 1

        stored_content = _encode_content(content)

        def _do(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                """INSERT INTO messages
                       (session_id, role, content, tool_call_id, tool_calls,
                        tool_name, timestamp, token_count, finish_reason,
                        reasoning, platform_message_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    stored_content,
                    tool_call_id,
                    tool_calls_json,
                    tool_name,
                    time.time(),
                    token_count,
                    finish_reason,
                    reasoning,
                    platform_message_id,
                ),
            )
            msg_id = cur.lastrowid
            if num_tool_calls:
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1, "
                    "tool_call_count = tool_call_count + ? WHERE id = ?",
                    (num_tool_calls, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1 "
                    "WHERE id = ?",
                    (session_id,),
                )
            return int(msg_id)

        return self._execute_write(_do)

    # ── reads ────────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_recent_sessions(
        self,
        *,
        limit: int = 20,
        exclude_sources: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return most-recently-active sessions (root sessions only).

        Each row is enriched with ``last_active`` (max message timestamp,
        falling back to ``started_at``) and a 60-char ``preview`` taken
        from the first user message in the session.
        """
        where = ["s.parent_session_id IS NULL", "s.archived = 0"]
        params: list[Any] = []
        if exclude_sources:
            excl = list(exclude_sources)
            placeholders = ",".join("?" for _ in excl)
            where.append(f"COALESCE(s.source, '') NOT IN ({placeholders})")
            params.extend(excl)
        sql = f"""
            SELECT s.*,
                COALESCE(
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user'
                       AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2
                     WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            WHERE {' AND '.join(where)}
            ORDER BY last_active DESC, s.started_at DESC, s.id DESC
            LIMIT ?
        """
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            s = dict(row)
            raw = (s.pop("_preview_raw", "") or "").strip()
            if raw:
                s["preview"] = raw[:60] + ("..." if len(raw) > 60 else "")
            else:
                s["preview"] = ""
            out.append(s)
        return out

    def resolve_to_lineage_root(self, session_id: str) -> str:
        """Walk the ``parent_session_id`` chain to the topmost ancestor.

        Bounded loop with a visited set so a malformed cycle can't hang
        the agent.
        """
        if not session_id:
            return session_id
        visited: set[str] = set()
        cur = session_id
        while cur and cur not in visited:
            visited.add(cur)
            row = self.get_session(cur)
            if not row:
                return cur
            parent = row.get("parent_session_id")
            if not parent:
                return cur
            cur = parent
        return cur

    def lineage_ids(self, session_id: str) -> set[str]:
        """Return the full set of session ids in ``session_id``'s lineage
        (root → all descendants reachable via parent_session_id edges).
        """
        if not session_id:
            return set()
        root = self.resolve_to_lineage_root(session_id)
        seen: set[str] = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id FROM sessions WHERE parent_session_id = ?",
                    (current,),
                ).fetchall()
            for r in rows:
                child = r["id"]
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
        return seen

    def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        *,
        window: int = 5,
    ) -> dict[str, Any]:
        """Return ±``window`` messages centred on the anchor.

        ``messages_before`` / ``messages_after`` report how many rows
        we returned on each side — when one is < ``window`` the caller
        has reached a session boundary.
        """
        window = max(window, 0)
        with self._lock:
            anchor = self._conn.execute(
                "SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                (around_message_id, session_id),
            ).fetchone()
            if not anchor:
                return {"window": [], "messages_before": 0, "messages_after": 0}
            before_rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id <= ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window + 1),
            ).fetchall()
            after_rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window),
            ).fetchall()
        rows = list(reversed(before_rows)) + list(after_rows)
        out = [_hydrate_message(r) for r in rows]
        return {
            "window": out,
            "messages_before": max(0, len(before_rows) - 1),
            "messages_after": len(after_rows),
        }

    def get_anchored_view(
        self,
        session_id: str,
        around_message_id: int,
        *,
        window: int = 5,
        bookend: int = 3,
    ) -> dict[str, Any]:
        """Window + ``bookend_start`` + ``bookend_end`` (first / last
        ``bookend`` user+assistant messages of the session).
        """
        primitive = self.get_messages_around(
            session_id, around_message_id, window=window
        )
        win_rows = primitive["window"]
        empty = {
            "window": [],
            "messages_before": 0,
            "messages_after": 0,
            "bookend_start": [],
            "bookend_end": [],
        }
        if not win_rows:
            return empty

        keep = {"user", "assistant"}
        filtered_window = [
            m for m in win_rows
            if m.get("id") == around_message_id or m.get("role") in keep
        ]
        min_id = win_rows[0]["id"]
        max_id = win_rows[-1]["id"]

        bookend_start_rows: list[Any] = []
        bookend_end_rows: list[Any] = []
        if bookend > 0:
            with self._lock:
                bookend_start_rows = self._conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? AND id < ? "
                    "AND role IN ('user','assistant') "
                    "AND length(content) > 0 "
                    "ORDER BY id ASC LIMIT ?",
                    (session_id, min_id, bookend),
                ).fetchall()
                bookend_end_rows = self._conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? AND id > ? "
                    "AND role IN ('user','assistant') "
                    "AND length(content) > 0 "
                    "ORDER BY id DESC LIMIT ?",
                    (session_id, max_id, bookend),
                ).fetchall()
                bookend_end_rows = list(reversed(bookend_end_rows))

        return {
            "window": filtered_window,
            "messages_before": primitive["messages_before"],
            "messages_after": primitive["messages_after"],
            "bookend_start": [_hydrate_message(r) for r in bookend_start_rows],
            "bookend_end": [_hydrate_message(r) for r in bookend_end_rows],
        }

    # ── FTS5 search ──────────────────────────────────────────────────

    def search_messages(
        self,
        query: str,
        *,
        limit: int = 50,
        exclude_sources: Iterable[str] | None = None,
        role_filter: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 search across messages.

        Picks ``messages_fts_trigram`` when the query contains CJK so
        we don't tokenize 大别山 into ``大 AND 别 AND 山``.
        """
        if not self._fts_enabled or not query.strip():
            return []
        cleaned = _sanitize_fts5_query(query)
        if not cleaned:
            return []

        excl = list(exclude_sources) if exclude_sources else []
        roles = list(role_filter) if role_filter else []

        use_trigram = (
            self._trigram_enabled
            and contains_cjk(cleaned)
            and _count_cjk(cleaned) >= 3
        )
        if use_trigram:
            # Quote each non-operator token so % and * are literal
            tokens = cleaned.split()
            parts: list[str] = []
            for tok in tokens:
                if tok.upper() in {"AND", "OR", "NOT"}:
                    parts.append(tok)
                else:
                    parts.append('"' + tok.replace('"', '""') + '"')
            match_query = " ".join(parts)
            fts_table = "messages_fts_trigram"
            snippet_table = "messages_fts_trigram"
        else:
            match_query = cleaned
            fts_table = "messages_fts"
            snippet_table = "messages_fts"

        where = [f"{fts_table} MATCH ?", "m.active = 1"]
        params: list[Any] = [match_query]
        if excl:
            placeholders = ",".join("?" for _ in excl)
            where.append(f"COALESCE(s.source,'') NOT IN ({placeholders})")
            params.extend(excl)
        if roles:
            placeholders = ",".join("?" for _ in roles)
            where.append(f"m.role IN ({placeholders})")
            params.extend(roles)

        sql = f"""
            SELECT
                m.id, m.session_id, m.role, m.timestamp, m.tool_name,
                snippet({snippet_table}, 0, '>>>', '<<<', '...', 40) AS snippet,
                s.source, s.model, s.title, s.parent_session_id,
                s.started_at AS session_started
            FROM {fts_table}
            JOIN messages m ON m.id = {fts_table}.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            ORDER BY rank
            LIMIT ?
        """
        params.append(limit)
        with self._lock:
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                logger.debug("FTS5 search failed: %s", exc)
                return []
        return [dict(r) for r in rows]

    # ====================================================================
    # BaseStore — Op dispatch
    # ====================================================================

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._do_get(op))
            elif isinstance(op, PutOp):
                self._do_put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._do_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._do_list_namespaces(op))
            else:  # pragma: no cover — defensive
                raise TypeError(f"Unsupported op: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # SQLite is blocking. Run on the default executor so a long
        # FTS5 query doesn't peg the event loop.
        ops_list = list(ops)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.batch, ops_list)

    # ── op handlers ──────────────────────────────────────────────────

    def _do_get(self, op: GetOp) -> Item | None:
        namespace = op.namespace
        # ("messages", session_id, role) maps to a single row.
        if (
            len(namespace) >= 2
            and namespace[0] == "messages"
            and op.key.isdigit()
        ):
            session_id = namespace[1]
            role_filter = namespace[2] if len(namespace) >= 3 else None
            with self._lock:
                if role_filter is None:
                    row = self._conn.execute(
                        "SELECT * FROM messages WHERE id = ? AND session_id = ?",
                        (int(op.key), session_id),
                    ).fetchone()
                else:
                    row = self._conn.execute(
                        "SELECT * FROM messages WHERE id = ? AND session_id = ? "
                        "AND role = ?",
                        (int(op.key), session_id, role_filter),
                    ).fetchone()
            if not row:
                return None
            return _row_to_item(row, namespace)

        # ("sessions",) namespace → session metadata row.
        if namespace and namespace[0] == "sessions":
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM sessions WHERE id = ?", (op.key,)
                ).fetchone()
            if not row:
                return None
            now = datetime.fromtimestamp(
                row["started_at"] or time.time(), tz=UTC
            )
            return Item(
                value=dict(row),
                key=op.key,
                namespace=namespace,
                created_at=now,
                updated_at=now,
            )
        return None

    def _do_put(self, op: PutOp) -> None:
        namespace = op.namespace
        if op.value is None:
            # Delete
            if (
                len(namespace) >= 2
                and namespace[0] == "messages"
                and op.key.isdigit()
            ):
                session_id = namespace[1]

                def _do(conn: sqlite3.Connection) -> None:
                    conn.execute(
                        "DELETE FROM messages WHERE id = ? AND session_id = ?",
                        (int(op.key), session_id),
                    )

                self._execute_write(_do)
                return
            if namespace and namespace[0] == "sessions":

                def _do_s(conn: sqlite3.Connection) -> None:
                    conn.execute("DELETE FROM sessions WHERE id = ?", (op.key,))

                self._execute_write(_do_s)
                return
            return

        # Write
        if (
            len(namespace) >= 2
            and namespace[0] == "messages"
        ):
            session_id = namespace[1]
            role = namespace[2] if len(namespace) >= 3 else op.value.get("role")
            self.ensure_session(session_id)
            self.record_message(
                session_id,
                role or "user",
                op.value.get("content"),
                tool_calls=op.value.get("tool_calls"),
                tool_name=op.value.get("tool_name"),
                tool_call_id=op.value.get("tool_call_id"),
                token_count=op.value.get("token_count"),
                finish_reason=op.value.get("finish_reason"),
                reasoning=op.value.get("reasoning"),
                platform_message_id=op.value.get("platform_message_id"),
            )
            return

        if namespace and namespace[0] == "sessions":
            self.ensure_session(op.key, **op.value)
            return

        # Unknown namespace — refuse silently rather than raise.
        # The recorder middleware controls namespace shape; agent code
        # writing to an unknown namespace is a bug, but we don't want
        # to crash a long-running session for it.
        logger.debug("Ignoring put to unsupported namespace %r", namespace)

    def _do_search(self, op: SearchOp) -> list[SearchItem]:
        namespace_prefix = op.namespace_prefix
        if not namespace_prefix or namespace_prefix[0] != "messages":
            return []
        if not op.query:
            # No FTS5 query → return most recent messages in the namespace.
            session_filter = (
                namespace_prefix[1] if len(namespace_prefix) >= 2 else None
            )
            role_filter = (
                namespace_prefix[2] if len(namespace_prefix) >= 3 else None
            )
            where = ["m.active = 1"]
            params: list[Any] = []
            if session_filter:
                where.append("m.session_id = ?")
                params.append(session_filter)
            if role_filter:
                where.append("m.role = ?")
                params.append(role_filter)
            sql = (
                "SELECT * FROM messages m WHERE "
                + " AND ".join(where)
                + " ORDER BY m.id DESC LIMIT ? OFFSET ?"
            )
            params.extend([op.limit, op.offset])
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
            return [
                _row_to_search_item(row, namespace_prefix, score=None)
                for row in rows
            ]

        # FTS5 query path
        matches = self.search_messages(
            op.query,
            limit=op.limit + op.offset,
            role_filter=(
                [namespace_prefix[2]] if len(namespace_prefix) >= 3 else None
            ),
        )
        # Filter by session_id if present in the prefix.
        if len(namespace_prefix) >= 2:
            session_filter = namespace_prefix[1]
            matches = [m for m in matches if m["session_id"] == session_filter]
        matches = matches[op.offset : op.offset + op.limit]

        results: list[SearchItem] = []
        for i, match in enumerate(matches):
            ns = ("messages", match["session_id"], match.get("role") or "")
            ts = match.get("timestamp") or time.time()
            created = datetime.fromtimestamp(ts, tz=UTC)
            value = {
                "id": match["id"],
                "session_id": match["session_id"],
                "role": match.get("role"),
                "snippet": match.get("snippet"),
                "tool_name": match.get("tool_name"),
                "source": match.get("source"),
                "model": match.get("model"),
                "title": match.get("title"),
                "parent_session_id": match.get("parent_session_id"),
            }
            # Synthetic descending score so rank order is preserved when
            # callers sort on it.
            results.append(
                SearchItem(
                    namespace=ns,
                    key=str(match["id"]),
                    value=value,
                    created_at=created,
                    updated_at=created,
                    score=float(len(matches) - i),
                )
            )
        return results

    def _do_list_namespaces(
        self, op: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        match_conds = list(op.match_conditions or ())
        with self._lock:
            session_rows = self._conn.execute(
                "SELECT id FROM sessions ORDER BY started_at DESC"
            ).fetchall()
            role_rows = self._conn.execute(
                "SELECT DISTINCT session_id, role FROM messages"
            ).fetchall()
        ns_set: set[tuple[str, ...]] = set()
        for r in session_rows:
            ns_set.add(("messages", r["id"]))
            ns_set.add(("sessions",))
        for r in role_rows:
            ns_set.add(("messages", r["session_id"], r["role"] or ""))

        def _matches(ns: tuple[str, ...]) -> bool:
            for cond in match_conds:
                if not _match_namespace(ns, cond):
                    return False
            return True

        filtered = [ns for ns in ns_set if _matches(ns)]
        if op.max_depth is not None:
            filtered = [ns[: op.max_depth] for ns in filtered]
            filtered = list({ns for ns in filtered})
        filtered.sort()
        return filtered[op.offset : op.offset + op.limit]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_CONTENT_JSON_PREFIX = "\x00json:"


def _encode_content(content: Any) -> Any:
    """Serialise structured (list/dict) message content for sqlite.

    sqlite can't bind list / dict — multimodal AIMessage content blocks
    arrive as lists of part dicts. Sentinel-prefix the JSON so we can
    recognise it on read.
    """
    if content is None or isinstance(content, (str, bytes, int, float)):
        return content
    try:
        return _CONTENT_JSON_PREFIX + json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _decode_content(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_CONTENT_JSON_PREFIX):
        try:
            return json.loads(value[len(_CONTENT_JSON_PREFIX) :])
        except json.JSONDecodeError:
            return value
    return value


def _hydrate_message(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    msg = dict(row)
    if "content" in msg:
        msg["content"] = _decode_content(msg["content"])
    if msg.get("tool_calls"):
        try:
            msg["tool_calls"] = json.loads(msg["tool_calls"])
        except (json.JSONDecodeError, TypeError):
            msg["tool_calls"] = []
    return msg


def _row_to_item(
    row: sqlite3.Row, namespace: tuple[str, ...]
) -> Item:
    ts = row["timestamp"] or time.time()
    when = datetime.fromtimestamp(ts, tz=UTC)
    value = _hydrate_message(row)
    return Item(
        value=value,
        key=str(row["id"]),
        namespace=namespace,
        created_at=when,
        updated_at=when,
    )


def _row_to_search_item(
    row: sqlite3.Row,
    namespace_prefix: tuple[str, ...],
    *,
    score: float | None,
) -> SearchItem:
    ts = row["timestamp"] or time.time()
    when = datetime.fromtimestamp(ts, tz=UTC)
    value = _hydrate_message(row)
    role = value.get("role") or ""
    session_id = value.get("session_id") or (
        namespace_prefix[1] if len(namespace_prefix) >= 2 else ""
    )
    ns: tuple[str, ...] = ("messages", session_id, role)
    return SearchItem(
        namespace=ns,
        key=str(row["id"]),
        value=value,
        created_at=when,
        updated_at=when,
        score=score,
    )


def _match_namespace(ns: tuple[str, ...], cond: MatchCondition) -> bool:
    path = cond.path
    if cond.match_type == "prefix":
        if len(path) > len(ns):
            return False
        for i, p in enumerate(path):
            if p == "*":
                continue
            if ns[i] != p:
                return False
        return True
    if cond.match_type == "suffix":
        if len(path) > len(ns):
            return False
        offset = len(ns) - len(path)
        for i, p in enumerate(path):
            if p == "*":
                continue
            if ns[offset + i] != p:
                return False
        return True
    return False


__all__ = [
    "InvalidNamespaceError",  # re-export for callers
    "SqliteFtsStore",
    "contains_cjk",
    "default_db_path",
    "resolve_hermes_home",
]
