"""Tests for the bundled ``MarkdownProvider`` (zero-dep notes recall)."""

from __future__ import annotations

from pathlib import Path

import pytest

from langstage_hermes.memory.provider import get_provider
from langstage_hermes.plugins.builtin.markdown_provider import (
    MarkdownProvider,
    _split_into_sections,
    _tokenize_query,
    search_notes,
)


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    d = tmp_path / "memories" / "notes"
    d.mkdir(parents=True)
    return d


def _write(p: Path, name: str, body: str) -> Path:
    f = p / name
    f.write_text(body, encoding="utf-8")
    return f


# ── helpers ───────────────────────────────────────────────────────────


def test_tokenize_drops_short_tokens():
    assert _tokenize_query("a is in the database") == ["the", "database"]


def test_tokenize_lowercases_and_strips_punctuation():
    assert _tokenize_query("Hello, World! 2026") == ["hello", "world", "2026"]


def test_split_no_headings_returns_single_section():
    assert _split_into_sections("just one block of text") == ["just one block of text"]


def test_split_preserves_heading_with_section():
    body = "preamble line\n\n## First\nalpha\n\n## Second\nbeta\n"
    sections = _split_into_sections(body)
    assert len(sections) == 3
    assert sections[0].startswith("preamble")
    assert sections[1].startswith("## First")
    assert sections[2].startswith("## Second")


def test_split_handles_h1_and_h3():
    body = "# Top\nalpha\n## Sub\nbeta\n### Sub-sub\ngamma"
    sections = _split_into_sections(body)
    # All three headings produce their own section.
    assert len(sections) == 3


def test_split_empty_input():
    assert _split_into_sections("") == []
    assert _split_into_sections("   \n   ") == []


# ── search_notes (the pure recall function) ────────────────────────────


def test_search_returns_matching_section(notes_dir: Path):
    _write(notes_dir, "project.md", "## Glasswing\nIt's a Python library for etch telemetry.\n")
    results = search_notes("glasswing", notes_dir)
    assert len(results) == 1
    assert "Glasswing" in results[0]
    assert "_From project.md:_" in results[0]


def test_search_returns_empty_on_no_match(notes_dir: Path):
    _write(notes_dir, "project.md", "## Glasswing\nIt's a Python library.\n")
    assert search_notes("kubernetes", notes_dir) == []


def test_search_returns_empty_on_short_query(notes_dir: Path):
    _write(notes_dir, "project.md", "## A\nfoo bar baz\n")
    # All query tokens shorter than 3 chars get dropped → no recall.
    assert search_notes("a is", notes_dir) == []


def test_search_ranks_higher_token_overlap_first(notes_dir: Path):
    _write(
        notes_dir,
        "a.md",
        "## A\nThis mentions glasswing but nothing else relevant.\n",
    )
    _write(
        notes_dir,
        "b.md",
        "## B\nGlasswing and etch telemetry and polars — multiple hits.\n",
    )
    results = search_notes("glasswing etch polars", notes_dir, limit=5)
    assert len(results) == 2
    # b.md should rank first — 3 hits vs 1.
    assert "_From b.md:_" in results[0]
    assert "_From a.md:_" in results[1]


def test_search_breaks_ties_by_shorter_section(notes_dir: Path):
    _write(notes_dir, "long.md", "## L\nglasswing " + ("x " * 200) + "\n")
    _write(notes_dir, "short.md", "## S\nglasswing rocks\n")
    results = search_notes("glasswing", notes_dir, limit=5)
    # Both have 1 hit; shorter should win.
    assert "_From short.md:_" in results[0]


def test_search_respects_limit(notes_dir: Path):
    for i in range(10):
        _write(notes_dir, f"f{i}.md", f"## Section {i}\nglasswing reference {i}\n")
    results = search_notes("glasswing", notes_dir, limit=3)
    assert len(results) == 3


def test_search_handles_missing_dir(tmp_path: Path):
    nonexistent = tmp_path / "no-such-dir"
    assert search_notes("anything", nonexistent) == []


def test_search_skips_unreadable_files(notes_dir: Path, monkeypatch):
    _write(notes_dir, "good.md", "## Good\nglasswing here\n")
    _write(notes_dir, "bad.md", "## Bad\nglasswing too\n")
    real_read = Path.read_text

    def maybe_fail(self, *args, **kwargs):
        if self.name == "bad.md":
            raise OSError("simulated I/O error")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", maybe_fail)
    results = search_notes("glasswing", notes_dir)
    # Good file's section comes through; bad file is skipped quietly.
    assert any("_From good.md:_" in r for r in results)
    assert not any("_From bad.md:_" in r for r in results)


# ── MarkdownProvider (ABC integration) ─────────────────────────────────


def test_provider_recall_delegates_to_search_notes(notes_dir: Path):
    _write(notes_dir, "n.md", "## X\nglasswing telemetry\n")
    p = MarkdownProvider(notes_dir=notes_dir)
    p.setup_session("sess-1", user_id="kedar")
    results = p.recall("glasswing")
    assert len(results) == 1


def test_provider_recall_all_modes_behave_identically(notes_dir: Path):
    _write(notes_dir, "n.md", "## X\nglasswing telemetry\n")
    p = MarkdownProvider(notes_dir=notes_dir)
    hybrid = p.recall("glasswing", mode="hybrid")
    context = p.recall("glasswing", mode="context")
    tools = p.recall("glasswing", mode="tools")
    assert hybrid == context == tools


def test_provider_record_turn_is_noop(notes_dir: Path):
    p = MarkdownProvider(notes_dir=notes_dir)
    # Should not raise, should not create any file.
    p.record_turn("user", "anything")
    p.record_turn("assistant", "anything")
    assert list(notes_dir.iterdir()) == []


def test_provider_teardown_is_noop(notes_dir: Path):
    p = MarkdownProvider(notes_dir=notes_dir)
    p.setup_session("sess-1")
    p.teardown()  # should not raise
    # And recall still works after teardown — there are no handles to release.
    _write(notes_dir, "n.md", "## X\ntest content\n")
    assert p.recall("test") != []


def test_provider_uses_hermes_home_when_notes_dir_unset(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(tmp_path))
    notes = tmp_path / "memories" / "notes"
    notes.mkdir(parents=True)
    _write(notes, "x.md", "## X\nglasswing data\n")
    p = MarkdownProvider()  # no explicit notes_dir
    results = p.recall("glasswing")
    assert results
    assert "glasswing" in results[0].lower()


def test_provider_is_registered_as_markdown():
    cls = get_provider("markdown")
    assert cls is MarkdownProvider
