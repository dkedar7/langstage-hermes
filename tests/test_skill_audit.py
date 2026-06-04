"""Tests for the skill mutation log + rollback.

Covers the four entry points where mutations must be captured (write,
edit-via-patch, delete, pin/unpin) and the rollback path's contract:
restoring to mutation #N puts the SKILL.md on disk in the state it had
*immediately before* mutation #N ran, and the rollback itself appears
as a new audit row so the log stays append-only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from deepagent_hermes.skills.audit import RollbackError, SkillAuditLog
from deepagent_hermes.skills.library import SkillLibrary
from deepagent_hermes.skills.tools import _skill_manage_impl


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A fresh hermes-home with skills/ and an empty state.db beneath it."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    return tmp_path


@pytest.fixture
def audit_log(home: Path) -> SkillAuditLog:
    log = SkillAuditLog(db_path=str(home / "state.db"))
    yield log
    log.close()


@pytest.fixture
def library(home: Path, audit_log: SkillAuditLog) -> SkillLibrary:
    return SkillLibrary(dirs=[home / "skills"], audit_log=audit_log)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_create_records_create_mutation(library: SkillLibrary, audit_log: SkillAuditLog):
    """library.write() on a new path records action='create' with no before-state."""
    library.write("alpha", {"name": "alpha", "description": "alpha skill"}, "Body v1\n")
    rows = audit_log.list()
    assert len(rows) == 1
    assert rows[0].action == "create"
    assert rows[0].skill_name == "alpha"
    assert rows[0].before_content is None
    assert rows[0].after_content is not None
    assert b"Body v1" in rows[0].after_content


def test_overwrite_records_write_file_with_before_and_after(library: SkillLibrary, audit_log: SkillAuditLog):
    """Second write to the same name records action='write_file' with both blobs."""
    library.write("alpha", {"name": "alpha", "description": "v1"}, "Body v1\n")
    library.write("alpha", {"name": "alpha", "description": "v2"}, "Body v2\n")
    rows = audit_log.list()
    assert len(rows) == 2
    # rows are DESC by timestamp so the second write is rows[0]
    assert rows[0].action == "write_file"
    assert rows[0].before_content is not None
    assert b"Body v1" in rows[0].before_content
    assert b"Body v2" in rows[0].after_content


def test_delete_records_delete_mutation(library: SkillLibrary, audit_log: SkillAuditLog):
    """delete() captures the pre-delete content so rollback can restore it."""
    library.write("alpha", {"name": "alpha", "description": "alpha"}, "Body v1\n")
    archived = library.delete("alpha")
    assert archived is True
    rows = audit_log.list()
    assert rows[0].action == "delete"
    assert rows[0].before_content is not None
    assert rows[0].after_content is None


def test_patch_records_via_skill_manage(library: SkillLibrary, audit_log: SkillAuditLog):
    """The agent's skill_manage(patch) hits the audit log via _action_patch."""
    library.write("beta", {"name": "beta", "description": "beta"}, "Line one\nLine two\n")
    _skill_manage_impl(
        library,
        action="patch",
        name="beta",
        description="",
        body="",
        category="",
        old_str="Line one",
        new_str="Line ONE",
        frontmatter_data=None,
        tool_call_id="tcid-test",
    )
    rows = audit_log.list(skill_name="beta")
    # 2 rows: the initial create, then the patch
    assert [r.action for r in rows] == ["patch", "create"]
    assert rows[0].before_content is not None
    assert b"Line one" in rows[0].before_content
    assert b"Line ONE" in rows[0].after_content


def test_pin_records_pin_and_unpin(library: SkillLibrary, audit_log: SkillAuditLog):
    library.write("gamma", {"name": "gamma", "description": "g"}, "body\n")
    _skill_manage_impl(
        library,
        action="pin",
        name="gamma",
        description="",
        body="",
        category="",
        old_str="",
        new_str="",
        frontmatter_data=None,
        tool_call_id="tc-pin",
    )
    _skill_manage_impl(
        library,
        action="unpin",
        name="gamma",
        description="",
        body="",
        category="",
        old_str="",
        new_str="",
        frontmatter_data=None,
        tool_call_id="tc-unpin",
    )
    rows = audit_log.list(skill_name="gamma")
    assert [r.action for r in rows] == ["unpin", "pin", "create"]
    # The pin frontmatter mutation should change at least one byte.
    assert rows[1].before_content != rows[1].after_content


def test_provenance_is_recorded(library: SkillLibrary, audit_log: SkillAuditLog):
    """set_mutation_context() values land on the recorded row."""
    library.set_mutation_context(source="cli", session_id="sess-xyz", tool_call_id="tc-abc")
    library.write("delta", {"name": "delta", "description": "d"}, "body\n")
    row = audit_log.list(skill_name="delta")[0]
    assert row.source == "cli"
    assert row.session_id == "sess-xyz"
    assert row.tool_call_id == "tc-abc"


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def test_list_filters_by_skill_name(library: SkillLibrary, audit_log: SkillAuditLog):
    library.write("a", {"name": "a", "description": "a"}, "x\n")
    library.write("b", {"name": "b", "description": "b"}, "y\n")
    library.write("a", {"name": "a", "description": "a2"}, "x2\n")
    only_a = audit_log.list(skill_name="a")
    assert all(r.skill_name == "a" for r in only_a)
    assert len(only_a) == 2


def test_latest_for_returns_most_recent(library: SkillLibrary, audit_log: SkillAuditLog):
    library.write("eps", {"name": "eps", "description": "e"}, "v1\n")
    library.write("eps", {"name": "eps", "description": "e"}, "v2\n")
    latest = audit_log.latest_for("eps")
    assert latest is not None
    assert b"v2" in (latest.after_content or b"")


def test_latest_for_returns_none_for_unknown_skill(audit_log: SkillAuditLog):
    assert audit_log.latest_for("never-existed") is None


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_restores_pre_mutation_state(library: SkillLibrary, audit_log: SkillAuditLog):
    """Rolling back mutation #N puts disk in the state it had immediately *before* N."""
    library.write("zeta", {"name": "zeta", "description": "v1"}, "Body v1\n")
    library.write("zeta", {"name": "zeta", "description": "v2"}, "Body v2\n")
    rows = audit_log.list(skill_name="zeta")
    # rows[0] is the v2 write — rolling it back should restore v1.
    mutation_id = rows[0].id
    path = audit_log.rollback_to("zeta", mutation_id)
    restored = path.read_bytes()
    assert b"Body v1" in restored
    assert b"Body v2" not in restored


def test_rollback_appends_rollback_row(library: SkillLibrary, audit_log: SkillAuditLog):
    """The rollback itself shows up as a new mutation row of action='rollback'."""
    library.write("eta", {"name": "eta", "description": "v1"}, "v1\n")
    library.write("eta", {"name": "eta", "description": "v2"}, "v2\n")
    second_write = audit_log.list(skill_name="eta")[0]
    audit_log.rollback_to("eta", second_write.id)
    rows = audit_log.list(skill_name="eta")
    assert rows[0].action == "rollback"
    # The rollback row's after_content is the historical before_content.
    assert rows[0].after_content == second_write.before_content


def test_rollback_of_create_rejected(library: SkillLibrary, audit_log: SkillAuditLog):
    """Can't roll back a create — there's no pre-state to restore to."""
    library.write("theta", {"name": "theta", "description": "t"}, "body\n")
    create_row = audit_log.list(skill_name="theta")[0]
    assert create_row.action == "create"
    with pytest.raises(RollbackError) as excinfo:
        audit_log.rollback_to("theta", create_row.id)
    assert "nothing to roll back" in str(excinfo.value).lower()


def test_rollback_unknown_mutation_id_raises(audit_log: SkillAuditLog):
    with pytest.raises(RollbackError):
        audit_log.rollback_to("whatever", 999_999)


def test_rollback_wrong_skill_name_raises(library: SkillLibrary, audit_log: SkillAuditLog):
    library.write("a", {"name": "a", "description": "a"}, "x\n")
    row = audit_log.list(skill_name="a")[0]
    with pytest.raises(RollbackError):
        audit_log.rollback_to("b", row.id)


def test_rollback_after_delete_recreates_skill(library: SkillLibrary, audit_log: SkillAuditLog):
    """Rolling back a delete writes the skill back to its original path."""
    library.write("iota", {"name": "iota", "description": "i"}, "Body iota\n")
    original_path = library.get("iota").path
    library.delete("iota")
    assert not original_path.exists()
    delete_row = audit_log.list(skill_name="iota")[0]
    audit_log.rollback_to("iota", delete_row.id)
    assert original_path.exists()
    assert b"Body iota" in original_path.read_bytes()


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def test_diff_against_disk_empty_when_no_drift(library: SkillLibrary, audit_log: SkillAuditLog):
    library.write("kappa", {"name": "kappa", "description": "k"}, "stable\n")
    row = audit_log.list(skill_name="kappa")[0]
    assert audit_log.diff_against_disk("kappa", row.id) == ""


def test_diff_against_disk_shows_drift(library: SkillLibrary, audit_log: SkillAuditLog):
    library.write("lambda", {"name": "lambda", "description": "l"}, "Body v1\n")
    row = audit_log.list(skill_name="lambda")[0]
    # Hand-edit the file outside the library.
    skill_path = library.get("lambda").path
    skill_path.write_bytes(b"---\nname: lambda\ndescription: l\n---\nBody MANUAL\n")
    diff_text = audit_log.diff_against_disk("lambda", row.id)
    assert "MANUAL" in diff_text
    assert "Body v1" in diff_text


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_audit_log_creates_table_idempotently(home: Path):
    """Opening the audit log twice on the same DB doesn't crash."""
    SkillAuditLog(db_path=str(home / "state.db")).close()
    SkillAuditLog(db_path=str(home / "state.db")).close()
    # The table must exist after either init.
    with sqlite3.connect(str(home / "state.db")) as conn:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='skill_mutations'")
        assert cur.fetchone() is not None


def test_no_audit_log_means_no_recording(home: Path):
    """A library without an audit_log doesn't try to write to one."""
    lib = SkillLibrary(dirs=[home / "skills"], audit_log=None)
    lib.write("mu", {"name": "mu", "description": "m"}, "body\n")
    # No SQLite file should be created by the library when no audit log is attached.
    assert not (home / "state.db").exists()
