"""Tests for the keyless ``langstage-hermes memory show`` CLI (gh #101).

The frozen-snapshot memory (``MEMORY.md`` + ``USER.md``) is the README's *first*
headline bullet, but its only reader was the ``/memory`` slash command inside a
keyed ``chat`` — offline there was no way to see "what has the agent learned about
me?" (USER.md) or "what's in the session snapshot?" (MEMORY.md) short of ``cat``.
This is the same agent-only-data gap the keyless ``search`` (#79) and
``memory notes`` (#94) CLIs closed for the other layers. These tests pin the
offline reader: the documented repro (populate a note, read it back), the
``--json`` surface, the ``--user`` / ``--session`` filters, the char-count-vs-limit
feedback, the ``dump`` alias, and graceful handling of a missing/empty layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli


def _write_memory(home: Path, fname: str, body: str) -> Path:
    mem = home / "memories"
    mem.mkdir(parents=True, exist_ok=True)
    f = mem / fname
    f.write_text(body, encoding="utf-8")
    return f


@pytest.fixture
def home_with_memory(tmp_hermes_home: Path) -> Path:
    """A home with both frozen-snapshot layers populated."""
    _write_memory(tmp_hermes_home, "USER.md", "Prefers a profiling-driven investigation procedure over guesswork.")
    _write_memory(tmp_hermes_home, "MEMORY.md", "Currently shipping the langstage-hermes release.")
    return tmp_hermes_home


# ── the keyless repro (human) ──────────────────────────────────────


def test_show_prints_both_layers(home_with_memory: Path):
    res = CliRunner().invoke(cli, ["memory", "show"])
    assert res.exit_code == 0, res.output
    # Both layers surfaced, offline, no model.
    assert "USER.md" in res.output
    assert "Prefers a profiling-driven investigation procedure" in res.output
    assert "MEMORY.md" in res.output
    assert "Currently shipping the langstage-hermes release." in res.output


def test_show_prints_char_count_vs_limit(home_with_memory: Path):
    """The near/over-budget feedback loop `/memory` in chat never gave — each layer
    prints its char count against the configured truncation budget (2200 / 1375)."""
    res = CliRunner().invoke(cli, ["memory", "show"])
    assert res.exit_code == 0, res.output
    assert "1,375 chars" in res.output  # USER.md budget
    assert "2,200 chars" in res.output  # MEMORY.md budget


# ── --user / --session filters ─────────────────────────────────────


def test_show_user_only(home_with_memory: Path):
    res = CliRunner().invoke(cli, ["memory", "show", "--user"])
    assert res.exit_code == 0, res.output
    assert "USER.md" in res.output
    assert "MEMORY.md" not in res.output


def test_show_session_only(home_with_memory: Path):
    res = CliRunner().invoke(cli, ["memory", "show", "--session"])
    assert res.exit_code == 0, res.output
    assert "MEMORY.md" in res.output
    assert "USER.md" not in res.output


# ── --json surface (stable keys for CI) ────────────────────────────


def test_show_json_shape(home_with_memory: Path):
    res = CliRunner().invoke(cli, ["memory", "show", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert set(data) == {"user", "memory"}
    user = data["user"]
    assert set(user) == {"path", "exists", "chars", "limit", "over_limit", "content"}
    assert user["exists"] is True
    assert user["chars"] == len("Prefers a profiling-driven investigation procedure over guesswork.")
    assert user["limit"] == 1375
    assert user["over_limit"] is False
    assert user["content"].startswith("Prefers a profiling-driven")
    assert data["memory"]["limit"] == 2200


def test_show_json_filter_narrows_keys(home_with_memory: Path):
    res = CliRunner().invoke(cli, ["memory", "show", "--session", "--json"])
    assert res.exit_code == 0, res.output
    assert set(json.loads(res.output)) == {"memory"}


def test_show_human_output_is_not_json(home_with_memory: Path):
    res = CliRunner().invoke(cli, ["memory", "show"])
    with pytest.raises(json.JSONDecodeError):
        json.loads(res.output)


# ── over-budget feedback ───────────────────────────────────────────


def test_show_flags_over_budget(tmp_hermes_home: Path):
    _write_memory(tmp_hermes_home, "USER.md", "x" * 1400)  # > 1375
    res = CliRunner().invoke(cli, ["memory", "show", "--user"])
    assert res.exit_code == 0, res.output
    assert "over budget" in res.output

    j = CliRunner().invoke(cli, ["memory", "show", "--user", "--json"])
    assert json.loads(j.output)["user"]["over_limit"] is True


# ── dump alias ─────────────────────────────────────────────────────


def test_dump_is_an_alias_for_show(home_with_memory: Path):
    show = CliRunner().invoke(cli, ["memory", "show", "--json"])
    dump = CliRunner().invoke(cli, ["memory", "dump", "--json"])
    assert dump.exit_code == 0, dump.output
    assert json.loads(dump.output) == json.loads(show.output)


# ── missing / empty layers (graceful, no traceback) ────────────────


def test_show_absent_memory_is_graceful_and_creates_nothing(tmp_hermes_home: Path):
    """No memories yet → a clear one-line message per layer, exit 0, no traceback,
    and the read command must not create the files as a side effect."""
    user_md = tmp_hermes_home / "memories" / "USER.md"
    memory_md = tmp_hermes_home / "memories" / "MEMORY.md"
    assert not user_md.exists() and not memory_md.exists()

    res = CliRunner().invoke(cli, ["memory", "show"])
    assert res.exit_code == 0, res.output
    assert "no user memory yet" in res.output
    assert "no session memory yet" in res.output
    # Read side must not materialise the files.
    assert not user_md.exists() and not memory_md.exists()


def test_show_absent_memory_json_is_empty_layers(tmp_hermes_home: Path):
    res = CliRunner().invoke(cli, ["memory", "show", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["user"]["exists"] is False
    assert data["user"]["chars"] == 0
    assert data["user"]["content"] == ""
    assert data["memory"]["exists"] is False


def test_show_empty_file_reads_as_no_memory(tmp_hermes_home: Path):
    """An empty (0-byte / whitespace-only) file is treated as no memory, not shown."""
    _write_memory(tmp_hermes_home, "USER.md", "   \n")
    res = CliRunner().invoke(cli, ["memory", "show", "--user"])
    assert res.exit_code == 0, res.output
    assert "no user memory yet" in res.output


# ── one populated, one empty (mixed) ───────────────────────────────


def test_show_mixed_one_layer_present(tmp_hermes_home: Path):
    _write_memory(tmp_hermes_home, "USER.md", "Likes concise reasoning over hedged language.")
    res = CliRunner().invoke(cli, ["memory", "show"])
    assert res.exit_code == 0, res.output
    assert "Likes concise reasoning" in res.output  # USER.md shown
    assert "no session memory yet" in res.output  # MEMORY.md absent, cleanly noted
