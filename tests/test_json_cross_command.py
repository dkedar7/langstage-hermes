"""Regression tests for the cross-command ``--json`` surface (gh #90).

The keyless ``search`` CLI (gh #79) shipped a ``--json`` mode, and its README
section advertised the flag as a *cross-command* scripting/CI surface —
"``--json`` emits structured output for scripting/CI, mirroring ``audit``/
``skills``". But ``--json`` was wired onto ``search`` alone: ``skills list
--json``, ``skills audit --json``, and ``audit log --json`` all crashed with
click's ``No such option: '--json'`` and exit code 2 — a crash-out, not a
graceful "unsupported".

These tests pin the fix: the three commands the reporter reached for now emit
one valid JSON object on stdout (matching ``search --json``'s conventions), the
human (non-json) output of each is unchanged, and ``--json`` changes only the
rendering — never the exit contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import frontmatter
import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli


def _write_skill(home: Path, name: str, description: str, body: str = "test body", category: str | None = None) -> Path:
    """Drop a valid SKILL.md under <home>/skills/[<category>/]<name>/SKILL.md."""
    parent = home / "skills"
    if category:
        parent = parent / category
    parent = parent / name
    parent.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(body, **{"name": name, "description": description})
    (parent / "SKILL.md").write_text(frontmatter.dumps(post), encoding="utf-8")
    return parent / "SKILL.md"


# ── the flag exists now (the crash the issue reported is gone) ──────────


@pytest.mark.parametrize(
    "argv",
    [
        ["skills", "list", "--json"],
        ["skills", "audit", "--json"],
        ["audit", "log", "--json"],
    ],
)
def test_json_flag_is_accepted_not_a_click_error(tmp_hermes_home: Path, argv: list[str]) -> None:
    """Before the fix each of these exited 2 with click's ``No such option``.

    After it, ``--json`` is a real option: the command runs and never emits the
    hard usage error. (``skills audit`` may still exit 1 if a bundled skill fails
    validation — that is a *validation* signal, not a parse error.)
    """
    res = CliRunner().invoke(cli, argv)
    assert res.exit_code != 2, res.output
    assert "No such option" not in res.output


# ── skills list --json ─────────────────────────────────────────────────


def test_skills_list_json_is_valid_and_actionable(tmp_hermes_home: Path) -> None:
    _write_skill(tmp_hermes_home, "json-probe", "a probe skill for the json surface")
    res = CliRunner().invoke(cli, ["skills", "list", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert isinstance(data["skills"], list)
    assert data["count"] == len(data["skills"])
    assert "load_errors" in data
    probe = next(s for s in data["skills"] if s["name"] == "json-probe")
    assert probe["description"] == "a probe skill for the json surface"
    assert set(probe) == {"name", "category", "description", "version", "path"}
    assert probe["path"].endswith("SKILL.md")


def test_skills_list_json_carries_full_untruncated_description(tmp_hermes_home: Path) -> None:
    """JSON is for machines: it carries the full description, not the human
    table's 80-char clip."""
    long_desc = "D" * 200
    _write_skill(tmp_hermes_home, "long-desc", long_desc)
    res = CliRunner().invoke(cli, ["skills", "list", "--json"])
    assert res.exit_code == 0, res.output
    probe = next(s for s in json.loads(res.output)["skills"] if s["name"] == "long-desc")
    assert probe["description"] == long_desc

    # ...and the human listing DOES clip it — a parity guard both ways.
    human = CliRunner().invoke(cli, ["skills", "list"])
    assert long_desc not in human.output
    assert "..." in human.output


def test_skills_list_human_output_unchanged(tmp_hermes_home: Path) -> None:
    _write_skill(tmp_hermes_home, "human-probe", "human listing description")
    res = CliRunner().invoke(cli, ["skills", "list"])
    assert res.exit_code == 0, res.output
    assert "human-probe" in res.output
    assert "skill(s)." in res.output
    # Human mode is the table, never JSON.
    with pytest.raises(json.JSONDecodeError):
        json.loads(res.output)


# ── skills audit --json ────────────────────────────────────────────────


def test_skills_audit_json_reports_structured_validation(tmp_hermes_home: Path) -> None:
    _write_skill(tmp_hermes_home, "clean-json-skill", "a valid description")
    res = CliRunner().invoke(cli, ["skills", "audit", "--json"])
    # Exit is identical to the human command: 0 if everything passes, 1 if some
    # (possibly bundled) skill fails. Either way the JSON is emitted and valid.
    assert res.exit_code in (0, 1), res.output
    data = json.loads(res.output)
    assert set(data) == {"ok", "skill_count", "failed_count", "results"}
    assert data["ok"] is (data["failed_count"] == 0)
    assert data["ok"] is (res.exit_code == 0)
    assert data["skill_count"] == len(data["results"])
    clean = next(r for r in data["results"] if r["name"] == "clean-json-skill")
    assert clean == {"name": "clean-json-skill", "ok": True, "errors": []}


def test_skills_audit_human_output_unchanged(tmp_hermes_home: Path) -> None:
    _write_skill(tmp_hermes_home, "clean-human-skill", "a valid description")
    res = CliRunner().invoke(cli, ["skills", "audit"])
    assert "pass validation" in res.output or "failed validation" in res.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(res.output)


# ── audit log --json ───────────────────────────────────────────────────


def test_audit_log_json_lists_mutations(tmp_hermes_home: Path) -> None:
    # Generate a real mutation through the CLI: removing a skill records a
    # rollback-able ``delete`` row (gh #39).
    _write_skill(tmp_hermes_home, "audited", "a skill that will be removed")
    runner = CliRunner()
    assert runner.invoke(cli, ["skills", "remove", "audited"]).exit_code == 0

    res = runner.invoke(cli, ["audit", "log", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["count"] == len(data["mutations"])
    assert data["count"] >= 1
    row = next(m for m in data["mutations"] if m["skill_name"] == "audited")
    assert row["action"] == "delete"
    assert isinstance(row["id"], int)
    assert isinstance(row["timestamp"], (int, float))
    # Scalar fields present; SKILL.md blobs are NOT dumped here (they live in
    # ``audit show``).
    assert set(row) == {
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
    }


def test_audit_log_json_empty_store_is_clean(tmp_hermes_home: Path) -> None:
    res = CliRunner().invoke(cli, ["audit", "log", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output) == {"mutations": [], "count": 0}


def test_audit_log_human_output_unchanged(tmp_hermes_home: Path) -> None:
    res = CliRunner().invoke(cli, ["audit", "log"])
    assert res.exit_code == 0, res.output
    assert "No skill mutations recorded yet." in res.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(res.output)
