"""Skill mutation log + rollback.

The agent and the CLI both mutate skills (``skill_manage`` tool actions,
``skills install`` / ``audit rollback`` commands). Without a log, a
regression in the agent's curation logic can quietly corrupt a skill that
took weeks to refine — by the time it's noticed, the only recovery path
is whatever the user happens to have backed up elsewhere.

``SkillAuditLog`` lives in the same SQLite DB as the FTS store (under
``<HERMES_HOME>/state.db``) and records full before/after content for
every mutation, keyed by skill name + timestamp. Every entry carries
enough provenance (``source``, ``session_id``, ``tool_call_id``) to
trace a change back to the turn that produced it.

Rollback is "copy the historical ``before`` blob back to the SKILL.md
file" — no patch replay. That keeps the data model trivial and the
rollback semantics obvious: rolling back to mutation #42 means "the file
looked like this immediately before mutation #42 ran." The rollback
itself appends a new mutation row (action=``rollback``) so the log
remains append-only.

This module is intentionally pure: it owns the SQL and the rollback
file-write. The library and CLI compose against it.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["MutationRow", "RollbackError", "SkillAuditLog"]


VALID_ACTIONS: frozenset[str] = frozenset({"create", "patch", "write_file", "delete", "pin", "unpin", "rollback"})


class RollbackError(RuntimeError):
    """Raised when a rollback cannot be performed (missing target, bad path, …)."""


@dataclass
class MutationRow:
    """One row from the ``skill_mutations`` table.

    The ``before_content`` and ``after_content`` fields are the full
    SKILL.md payload (frontmatter + body) as bytes. For ``create``
    actions ``before_content`` is ``None``; for ``delete`` actions
    ``after_content`` is ``None``.
    """

    id: int
    timestamp: float
    skill_name: str
    action: str
    source: str | None
    session_id: str | None
    tool_call_id: str | None
    skill_path: str | None
    before_hash: str | None
    after_hash: str | None
    before_content: bytes | None
    after_content: bytes | None

    @classmethod
    def from_row(cls, row: sqlite3.Row | tuple) -> MutationRow:
        # sqlite3.Row supports indexing by name; tuples don't. We accept
        # both because tests sometimes hand us raw tuples.
        if isinstance(row, sqlite3.Row):
            d = dict(row)
        else:
            cols = (
                "id",
                "timestamp",
                "skill_name",
                "action",
                "source",
                "session_id",
                "tool_call_id",
                "skill_path",
                "before_hash",
                "after_hash",
                "before_content",
                "after_content",
            )
            d = dict(zip(cols, row, strict=False))
        return cls(
            id=int(d["id"]),
            timestamp=float(d["timestamp"]),
            skill_name=str(d["skill_name"]),
            action=str(d["action"]),
            source=d.get("source"),
            session_id=d.get("session_id"),
            tool_call_id=d.get("tool_call_id"),
            skill_path=d.get("skill_path"),
            before_hash=d.get("before_hash"),
            after_hash=d.get("after_hash"),
            before_content=d.get("before_content"),
            after_content=d.get("after_content"),
        )


def _sha256(data: bytes | None) -> str | None:
    if data is None:
        return None
    return hashlib.sha256(data).hexdigest()


class SkillAuditLog:
    """Append-only log of skill mutations, backed by a SQLite connection.

    Construct with the same DB path the store uses
    (``<HERMES_HOME>/state.db``). The schema lives in
    ``SqliteFtsStore.SCHEMA_SQL`` so the table is created the first time
    a store is opened — we just need the connection. We open our own
    connection (with ``check_same_thread=False``) instead of sharing the
    store's because the audit log can be called from sync tool handlers
    that run on worker threads.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        # check_same_thread=False so sync tool handlers on worker threads
        # can call record(). SQLite serialises writes internally; we don't
        # need our own lock as long as every write is a single statement.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Make sure the schema is there even if the audit log is the
        # *first* thing to touch the DB (some CLI paths open the audit
        # log before the store). The schema is idempotent.
        self._ensure_table()

    def _ensure_table(self) -> None:
        # Local copy of the DDL — keeps audit.py independent of the store
        # module's internal SQL constants. If the store's schema changes,
        # whichever side initialises first wins; both produce the same
        # table (CREATE IF NOT EXISTS).
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS skill_mutations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                skill_name TEXT NOT NULL,
                action TEXT NOT NULL,
                source TEXT,
                session_id TEXT,
                tool_call_id TEXT,
                skill_path TEXT,
                before_hash TEXT,
                after_hash TEXT,
                before_content BLOB,
                after_content BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_skill_mutations_name_ts
                ON skill_mutations(skill_name, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_skill_mutations_session
                ON skill_mutations(session_id);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:  # pragma: no cover — defensive
            logger.debug("SkillAuditLog: close raised", exc_info=True)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        skill_name: str,
        action: str,
        before_content: bytes | None,
        after_content: bytes | None,
        source: str | None = None,
        session_id: str | None = None,
        tool_call_id: str | None = None,
        skill_path: str | Path | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Append a single mutation row. Returns the inserted row id.

        Both content blobs are optional — ``create`` has no before,
        ``delete`` has no after, ``pin`` has both. Hashes are computed
        here so callers don't have to.
        """
        if action not in VALID_ACTIONS:
            raise ValueError(f"unknown audit action {action!r}; valid: {sorted(VALID_ACTIONS)}")
        ts = timestamp if timestamp is not None else time.time()
        before_hash = _sha256(before_content)
        after_hash = _sha256(after_content)
        cur = self._conn.execute(
            """
            INSERT INTO skill_mutations
                (timestamp, skill_name, action, source, session_id, tool_call_id,
                 skill_path, before_hash, after_hash, before_content, after_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                skill_name,
                action,
                source,
                session_id,
                tool_call_id,
                str(skill_path) if skill_path is not None else None,
                before_hash,
                after_hash,
                before_content,
                after_content,
            ),
        )
        self._conn.commit()
        row_id = cur.lastrowid
        assert row_id is not None
        return int(row_id)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list(
        self,
        *,
        skill_name: str | None = None,
        limit: int = 50,
        since: float | None = None,
    ) -> list[MutationRow]:
        """Most-recent-first list of mutations.

        Filter by ``skill_name`` to scope to one skill. ``since`` is a
        unix timestamp; only rows with ``timestamp >= since`` are
        returned.
        """
        sql = "SELECT * FROM skill_mutations"
        clauses: list[str] = []
        params: list[Any] = []
        if skill_name is not None:
            clauses.append("skill_name = ?")
            params.append(skill_name)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Tiebreak on id so two mutations recorded in the same time.time()
        # tick (millisecond-precision on Windows) come back in insertion
        # order. Without this, "most recent" is non-deterministic on ties.
        sql += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(int(limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [MutationRow.from_row(r) for r in rows]

    def get(self, mutation_id: int) -> MutationRow | None:
        """Fetch a single mutation by id."""
        row = self._conn.execute("SELECT * FROM skill_mutations WHERE id = ?", (mutation_id,)).fetchone()
        return MutationRow.from_row(row) if row else None

    def latest_for(self, skill_name: str) -> MutationRow | None:
        """Most recent mutation row for *skill_name*, or ``None`` if untouched."""
        rows = self.list(skill_name=skill_name, limit=1)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Diff + rollback
    # ------------------------------------------------------------------

    def diff_against_disk(self, skill_name: str, mutation_id: int) -> str:
        """Unified-diff string comparing the on-disk SKILL.md to the
        ``after_content`` of *mutation_id*. Empty string if identical.

        Use this to answer "what has changed since this revision?"
        """
        import difflib

        row = self.get(mutation_id)
        if row is None or row.skill_name != skill_name:
            raise RollbackError(f"mutation {mutation_id} not found for skill {skill_name!r}")
        if not row.skill_path:
            raise RollbackError(f"mutation {mutation_id} has no skill_path")
        current_path = Path(row.skill_path)
        current = current_path.read_bytes() if current_path.exists() else b""
        target = row.after_content or b""
        if current == target:
            return ""
        current_text = current.decode("utf-8", errors="replace").splitlines(keepends=True)
        target_text = target.decode("utf-8", errors="replace").splitlines(keepends=True)
        diff = difflib.unified_diff(
            target_text,
            current_text,
            fromfile=f"{skill_name}@mutation-{mutation_id}",
            tofile=f"{skill_name}@disk",
        )
        return "".join(diff)

    def rollback_to(
        self,
        skill_name: str,
        mutation_id: int,
        *,
        source: str = "rollback",
        session_id: str | None = None,
    ) -> Path:
        """Restore *skill_name* to the ``before_content`` of *mutation_id*.

        Semantics: "after this call, the SKILL.md on disk looks like it
        did immediately *before* mutation *mutation_id* ran." This is
        the inverse of replaying — it undoes mutations starting from
        *mutation_id* up to and including the latest.

        The rollback itself is logged as a new mutation (action=``rollback``)
        so the audit history remains a complete record.

        Raises:
            RollbackError: if the mutation isn't found, doesn't belong
                to *skill_name*, or has no recorded ``before_content``
                (i.e. it was a ``create`` — there's no pre-state to
                restore to; use ``delete`` instead).
        """
        target = self.get(mutation_id)
        if target is None:
            raise RollbackError(f"mutation #{mutation_id} not found")
        if target.skill_name != skill_name:
            raise RollbackError(f"mutation #{mutation_id} is for skill {target.skill_name!r}, not {skill_name!r}")
        if not target.skill_path:
            raise RollbackError(f"mutation #{mutation_id} has no skill_path recorded")
        if target.before_content is None:
            raise RollbackError(
                f"mutation #{mutation_id} was a {target.action} with no before-state — "
                f"there's nothing to roll back to. To remove the skill, use `skill_manage delete`."
            )

        path = Path(target.skill_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        before_disk = path.read_bytes() if path.exists() else None
        path.write_bytes(target.before_content)

        # Append the rollback itself as a new mutation row so the log
        # remains an honest history.
        self.record(
            skill_name=skill_name,
            action="rollback",
            before_content=before_disk,
            after_content=target.before_content,
            source=source,
            session_id=session_id,
            tool_call_id=None,
            skill_path=path,
        )
        return path
