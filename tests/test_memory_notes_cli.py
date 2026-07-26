"""Tests for the keyless ``langstage-hermes memory notes`` CLI (gh #94).

The bundled ``MarkdownProvider`` recalls sections from
``<HERMES_HOME>/memories/notes/*.md`` but its only reader was the live agent
(needs a model + key) — exactly the gap the keyless ``search`` CLI (gh #79)
closed for the FTS5 store. These tests pin the offline preview: the documented
repro (author a note, surface its section), the ``--json`` surface, that hits
carry the source file, ``--limit``, and graceful handling of a missing/empty
notes dir and a no-match query.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli


def _write_note(home: Path, name: str, body: str) -> Path:
    notes = home / "memories" / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    f = notes / name
    f.write_text(body, encoding="utf-8")
    return f


@pytest.fixture
def home_with_deploy_note(tmp_hermes_home: Path) -> Path:
    """The issue's repro: a hand-authored note with a Rollback section."""
    _write_note(
        tmp_hermes_home,
        "deploy.md",
        "# Deploy runbook\n\n"
        "## Rollback\n"
        "To roll back a bad deploy, run `kubectl rollout undo deploy/api` and page on-call.\n\n"
        "## Canary\n"
        "Ship 5% first, watch the error rate for 10 minutes.\n",
    )
    return tmp_hermes_home


# ── the keyless repro (human) ──────────────────────────────────────


def test_notes_surfaces_matching_section(home_with_deploy_note: Path):
    res = CliRunner().invoke(cli, ["memory", "notes", "rollback"])
    assert res.exit_code == 0, res.output
    # The Rollback section is surfaced, offline, no model.
    assert "Rollback" in res.output
    assert "kubectl rollout undo" in res.output
    # The source file is printed so the hit is actionable.
    assert "deploy.md" in res.output
    # A non-matching section from the same file stays out.
    assert "Canary" not in res.output


def test_notes_multiword_query(home_with_deploy_note: Path):
    res = CliRunner().invoke(cli, ["memory", "notes", "bad deploy rollback"])
    assert res.exit_code == 0, res.output
    assert "deploy.md" in res.output
    assert "Rollback" in res.output


# ── --json surface (mirrors search --json) ─────────────────────────


def test_notes_json_shape(home_with_deploy_note: Path):
    res = CliRunner().invoke(cli, ["memory", "notes", "rollback", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert set(data) == {"query", "count", "results"}
    assert data["query"] == "rollback"
    assert data["count"] == len(data["results"]) == 1
    hit = data["results"][0]
    # File is cleanly parsed out of the `_From <file>:_` prefix; section + full
    # snippet are both present.
    assert hit["file"] == "deploy.md"
    assert hit["section"].startswith("## Rollback")
    assert hit["snippet"].startswith("_From deploy.md:_")
    assert "kubectl rollout undo" in hit["snippet"]


def test_notes_human_output_is_not_json(home_with_deploy_note: Path):
    res = CliRunner().invoke(cli, ["memory", "notes", "rollback"])
    with pytest.raises(json.JSONDecodeError):
        json.loads(res.output)


# ── --limit ────────────────────────────────────────────────────────


def test_notes_limit_caps_results(tmp_hermes_home: Path):
    for i in range(6):
        _write_note(tmp_hermes_home, f"n{i}.md", f"## Section {i}\nglasswing reference {i}\n")
    res = CliRunner().invoke(cli, ["memory", "notes", "glasswing", "--limit", "2", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["count"] == 2


# ── no-match (notes exist, query misses) ───────────────────────────


def test_notes_no_match_is_graceful(home_with_deploy_note: Path):
    res = CliRunner().invoke(cli, ["memory", "notes", "kubernetes-spaghetti-zzz"])
    assert res.exit_code == 0, res.output
    assert "No notes match" in res.output


def test_notes_no_match_json_is_empty(home_with_deploy_note: Path):
    res = CliRunner().invoke(cli, ["memory", "notes", "kubernetes-spaghetti-zzz", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["count"] == 0
    assert data["results"] == []


# ── absent / empty notes dir ───────────────────────────────────────


def test_notes_absent_dir_is_graceful_and_creates_nothing(tmp_hermes_home: Path):
    """No memories/notes yet → a clear setup hint, exit 0, no traceback, and the
    read command must not create the dir as a side effect."""
    notes_dir = tmp_hermes_home / "memories" / "notes"
    assert not notes_dir.exists()
    res = CliRunner().invoke(cli, ["memory", "notes", "rollback"])
    assert res.exit_code == 0, res.output
    assert "No notes yet" in res.output
    assert not notes_dir.exists()


def test_notes_absent_dir_json_is_empty(tmp_hermes_home: Path):
    res = CliRunner().invoke(cli, ["memory", "notes", "rollback", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output) == {"query": "rollback", "count": 0, "results": []}


# ── empty query ────────────────────────────────────────────────────


def test_notes_empty_query_names_the_requirement(home_with_deploy_note: Path):
    res = CliRunner().invoke(cli, ["memory", "notes"])
    assert res.exit_code == 0, res.output
    assert "Provide a query" in res.output


def test_notes_empty_query_json_is_empty(home_with_deploy_note: Path):
    res = CliRunner().invoke(cli, ["memory", "notes", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output) == {"query": "", "count": 0, "results": []}
