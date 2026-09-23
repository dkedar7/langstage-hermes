"""gh #114: ``demo`` with HERMES_HOME set must not write into the real home.

The README's keyless "try it" flow points ``HERMES_HOME`` at the user's real store
and runs ``demo``. The docs promise only that the demo *session* is recorded into
``<HERMES_HOME>/state.db`` (so ``search`` reads it back). The demo used to run the
whole loop against that real home, leaving a ``profile-slow-python`` skill in the
user's library and a fabricated preference in their ``USER.md`` — which the frozen
memory snapshot then fed to every future real session.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from langstage_hermes.cli import cli


def _session_message_count(db: Path, session_id: str = "demo-001") -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
    finally:
        conn.close()


def test_demo_leaves_no_skill_or_memory_in_a_real_home(tmp_hermes_home: Path):
    result = CliRunner().invoke(cli, ["demo"])

    assert result.exit_code == 0, result.output
    assert "DEMO: PASS" in result.output
    # No demo skill in the user's library, no fabricated user-memory note.
    assert list((tmp_hermes_home / "skills").rglob("SKILL.md")) == []
    assert not (tmp_hermes_home / "memories" / "USER.md").exists()
    assert not (tmp_hermes_home / "memories" / "MEMORY.md").exists()
    skills = CliRunner().invoke(cli, ["skills", "list", "--json"])
    assert "profile-slow-python" not in skills.output
    # No audit rows for a skill that doesn't exist in this home.
    conn = sqlite3.connect(str(tmp_hermes_home / "state.db"))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "skill_mutations" in tables:
            assert conn.execute("SELECT COUNT(*) FROM skill_mutations").fetchone()[0] == 0
    finally:
        conn.close()


def test_demo_still_records_its_session_for_search(tmp_hermes_home: Path):
    """The documented demo -> search loop (gh #88) keeps working."""
    runner = CliRunner()
    assert runner.invoke(cli, ["demo"]).exit_code == 0

    data = json.loads(runner.invoke(cli, ["search", "python", "--json"]).output)
    assert any(r["session_id"] == "demo-001" for r in data["results"]), data


def test_rerunning_demo_does_not_duplicate_the_session(tmp_hermes_home: Path):
    runner = CliRunner()
    assert runner.invoke(cli, ["demo"]).exit_code == 0
    first = _session_message_count(tmp_hermes_home / "state.db")
    assert first > 0

    assert runner.invoke(cli, ["demo"]).exit_code == 0
    assert _session_message_count(tmp_hermes_home / "state.db") == first


def test_demo_does_not_touch_other_sessions_in_the_real_store(tmp_hermes_home: Path):
    from langstage_hermes.store.sqlite_fts import SqliteFtsStore

    store = SqliteFtsStore(db_path=tmp_hermes_home / "state.db")
    try:
        store.ensure_session("real-session")
        store.record_message("real-session", "user", "my real conversation about kubernetes")
    finally:
        store.close()

    assert CliRunner().invoke(cli, ["demo"]).exit_code == 0
    assert _session_message_count(tmp_hermes_home / "state.db", "real-session") == 1
