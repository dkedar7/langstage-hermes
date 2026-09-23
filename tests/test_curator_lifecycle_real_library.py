"""Curator lifecycle against the REAL ``SkillLibrary`` + ``SqliteFtsStore``.

``test_curator_lifecycle.py`` drives ``mark_stale_and_archive`` through a fake
library whose API (``write(skill)``, top-level ``hermes.*`` frontmatter) had
drifted from the real one, which is how three defects shipped unnoticed:

- gh #141 — ``skill_last_used:<name>`` had a reader but no writer (and the store
  silently dropped the ``state_meta`` namespace), so the lifecycle aged skills by
  SKILL.md mtime and archived skills the agent used every day.
- gh #119 — the curator read ``pinned`` from top-level ``hermes`` while the agent's
  ``skill_manage(pin)`` (and SPEC §9) write ``metadata.hermes.pinned``, so
  agent-pinned skills were archived.
- gh #120 — the stale transition called ``library.write(skill)`` (wrong signature;
  TypeError swallowed), so nothing was ever marked ``stale``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import frontmatter
import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.curator import (
    CuratorMiddleware,
    mark_stale_and_archive,
    read_skill_last_used,
    record_skill_use,
)
from langstage_hermes.skills.library import SkillLibrary
from langstage_hermes.skills.tools import _skill_manage_impl, _skill_view_impl, make_skill_tools
from langstage_hermes.store.sqlite_fts import SqliteFtsStore

DAY = 86400


def _write_skill(home: Path, name: str, *, age_days: float = 0, **extra) -> Path:
    root = home / "skills" / name
    root.mkdir(parents=True, exist_ok=True)
    skill_md = root / "SKILL.md"
    post = frontmatter.Post(f"# {name}\n\nBody of {name}.\n", name=name, description=f"{name} fixture", **extra)
    skill_md.write_text(frontmatter.dumps(post), encoding="utf-8")
    if age_days:
        t = time.time() - age_days * DAY
        os.utime(skill_md, (t, t))
    return skill_md


@pytest.fixture
def user_lib(tmp_hermes_home: Path) -> SkillLibrary:
    return SkillLibrary(dirs=[tmp_hermes_home / "skills"])


@pytest.fixture
def store(tmp_hermes_home: Path):
    s = SqliteFtsStore(db_path=tmp_hermes_home / "state.db")
    yield s
    s.close()


# ── gh #120: the stale transition actually writes ─────────────────────


def test_stale_is_written_to_nested_metadata_and_body_preserved(tmp_hermes_home: Path, user_lib: SkillLibrary):
    path = _write_skill(tmp_hermes_home, "stale-candidate", age_days=45)

    result = mark_stale_and_archive(user_lib, stale_days=30, archive_days=90)

    assert result["marked_stale"] == ["stale-candidate"]
    post = frontmatter.load(path)
    assert post.metadata["metadata"]["hermes"]["lifecycle"] == "stale"
    assert "Body of stale-candidate." in post.content
    # Idempotent: a second pass doesn't re-mark.
    assert mark_stale_and_archive(user_lib, stale_days=30, archive_days=90)["marked_stale"] == []


def test_curator_run_cli_marks_stale(tmp_hermes_home: Path):
    path = _write_skill(tmp_hermes_home, "stale-candidate", age_days=45)

    result = CliRunner().invoke(cli, ["curator", "run"])

    assert result.exit_code == 0, result.output
    assert "marked stale (1)" in result.output
    assert frontmatter.load(path).metadata["metadata"]["hermes"]["lifecycle"] == "stale"


# ── gh #119: pin read and write agree ─────────────────────────────────


def test_agent_pinned_skill_is_not_archived(tmp_hermes_home: Path, user_lib: SkillLibrary):
    _write_skill(tmp_hermes_home, "agent-pinned", age_days=100, metadata={"hermes": {"pinned": True}})

    result = mark_stale_and_archive(user_lib, stale_days=30, archive_days=90)

    assert result["skipped_pinned"] == ["agent-pinned"]
    assert result["archived"] == []
    assert (tmp_hermes_home / "skills" / "agent-pinned" / "SKILL.md").is_file()


def test_legacy_top_level_pin_is_still_honored(tmp_hermes_home: Path, user_lib: SkillLibrary):
    """Pins written by older ``curator pin`` (top-level ``hermes.pinned``) keep protecting."""
    _write_skill(tmp_hermes_home, "cli-pinned", age_days=100, hermes={"pinned": True})

    result = mark_stale_and_archive(user_lib, stale_days=30, archive_days=90)

    assert result["skipped_pinned"] == ["cli-pinned"]
    assert result["archived"] == []


def test_skill_manage_pin_protects_from_curator_run(tmp_hermes_home: Path, user_lib: SkillLibrary):
    """End to end: the agent pins via the real tool; `curator status`/`run` honor it."""
    _write_skill(tmp_hermes_home, "agent-pinned", age_days=100)
    _skill_manage_impl(
        user_lib,
        action="pin",
        name="agent-pinned",
        description="",
        body="",
        category="",
        old_str="",
        new_str="",
        frontmatter_data=None,
        tool_call_id="",
    )
    path = tmp_hermes_home / "skills" / "agent-pinned" / "SKILL.md"
    t = time.time() - 100 * DAY
    os.utime(path, (t, t))

    runner = CliRunner()
    status = runner.invoke(cli, ["curator", "status"])
    assert "agent-pinned" in status.output, status.output
    run = runner.invoke(cli, ["curator", "run"])
    assert run.exit_code == 0, run.output
    assert "skipped pinned (1)" in run.output
    assert path.is_file()


def test_cli_pin_writes_the_nested_key_and_unpin_clears_legacy(tmp_hermes_home: Path):
    path = _write_skill(tmp_hermes_home, "pinnable", hermes={"pinned": True})
    runner = CliRunner()

    assert runner.invoke(cli, ["curator", "pin", "pinnable"]).exit_code == 0
    meta = frontmatter.load(path).metadata
    assert meta["metadata"]["hermes"]["pinned"] is True
    assert "hermes" not in meta  # legacy top-level key migrated away
    assert SkillLibrary(dirs=[tmp_hermes_home / "skills"]).get("pinnable").pinned

    assert runner.invoke(cli, ["curator", "unpin", "pinnable"]).exit_code == 0
    assert not SkillLibrary(dirs=[tmp_hermes_home / "skills"]).get("pinnable").pinned


# ── gh #141: usage is recorded and drives the lifecycle ───────────────


def test_state_meta_round_trips_through_the_sqlite_store(store: SqliteFtsStore):
    record_skill_use(store, "daily-driver", now=1234.5)
    assert read_skill_last_used(store, "daily-driver") == 1234.5
    assert read_skill_last_used(store, "never-used") is None


def test_skill_view_records_last_used(tmp_hermes_home: Path, user_lib: SkillLibrary, store: SqliteFtsStore):
    _write_skill(tmp_hermes_home, "daily-driver", age_days=120)
    before = time.time()

    _skill_view_impl(user_lib, name="daily-driver", tool_call_id="", store=store)

    last = read_skill_last_used(store, "daily-driver")
    assert last is not None and last >= before


def test_make_skill_tools_skill_view_records_last_used(tmp_hermes_home: Path, user_lib: SkillLibrary, store: SqliteFtsStore):
    _write_skill(tmp_hermes_home, "daily-driver")
    tools = {t.name: t for t in make_skill_tools(user_lib, store=store)}

    tools["skill_view"].invoke({"type": "tool_call", "name": "skill_view", "args": {"name": "daily-driver"}, "id": "tc-1"})

    assert read_skill_last_used(store, "daily-driver") is not None


def test_used_skill_with_old_file_is_not_archived_by_curator_run(tmp_hermes_home: Path, user_lib: SkillLibrary):
    """Issue #141 repro B: a constantly-used skill whose FILE is 120 days old survives."""
    _write_skill(tmp_hermes_home, "daily-driver", age_days=120)
    _write_skill(tmp_hermes_home, "unused", age_days=120)
    s = SqliteFtsStore(db_path=tmp_hermes_home / "state.db")
    try:
        _skill_view_impl(user_lib, name="daily-driver", tool_call_id="", store=s)
    finally:
        s.close()

    result = CliRunner().invoke(cli, ["curator", "run"])

    assert result.exit_code == 0, result.output
    assert (tmp_hermes_home / "skills" / "daily-driver" / "SKILL.md").is_file()
    assert not (tmp_hermes_home / "skills" / "unused").exists()  # mtime fallback still applies


def test_curator_middleware_reads_usage_from_the_real_store(tmp_hermes_home: Path, user_lib: SkillLibrary, store: SqliteFtsStore):
    _write_skill(tmp_hermes_home, "daily-driver", age_days=120)
    record_skill_use(store, "daily-driver")
    now = time.time()
    store.put(("curator_state",), "state", {"last_run_at": now - 999 * 3600, "last_user_activity": now - 999 * 3600})

    CuratorMiddleware(user_lib, store).before_agent(state={"messages": []})

    assert (tmp_hermes_home / "skills" / "daily-driver" / "SKILL.md").is_file()
    report = sorted((tmp_hermes_home / "logs" / "curator").glob("*/run.json"))[-1]
    assert json.loads(report.read_text(encoding="utf-8"))["lifecycle"]["archived"] == []
