"""gh #154: a per-home command must never mutate the installed package.

``skills remove <bundled-skill>`` used to ``shutil.move`` the bundled skill
directory out of ``site-packages`` (into ``<wheel>/_bundled_skills/_archived/``),
deleting it for every ``HERMES_HOME`` and every user on that interpreter. The
curator's lifecycle pass walks the same library — bundled skills included — so an
old enough install could have its bundled skills archived (or stale-marked, i.e.
rewritten) by ``curator run`` / the weekly ``CuratorMiddleware`` too.

These tests never touch the real package: ``_bundled_skills_dir`` is pointed at a
throwaway copy so a regression moves a temp dir, not the source tree.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import frontmatter
import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.curator import mark_stale_and_archive
from langstage_hermes.skills import library as library_mod
from langstage_hermes.skills.library import BundledSkillError, SkillLibrary


def _write_skill(base: Path, name: str, *, category: str | None = None, age_days: float = 0) -> Path:
    root = base / category / name if category else base / name
    root.mkdir(parents=True, exist_ok=True)
    skill_md = root / "SKILL.md"
    post = frontmatter.Post(f"# {name}\n", name=name, description=f"{name} fixture")
    skill_md.write_text(frontmatter.dumps(post), encoding="utf-8")
    if age_days:
        t = time.time() - age_days * 86400
        os.utime(skill_md, (t, t))
    return skill_md


@pytest.fixture
def fake_bundled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in for the packaged ``_bundled_skills/`` tree."""
    bundled = tmp_path / "site-packages" / "langstage_hermes" / "_bundled_skills"
    _write_skill(bundled, "obsidian", category="note-taking", age_days=200)
    monkeypatch.setattr(library_mod, "_bundled_skills_dir", lambda: bundled)
    return bundled


def test_delete_refuses_a_bundled_skill(tmp_hermes_home: Path, fake_bundled: Path):
    lib = SkillLibrary()
    assert lib.get("obsidian") is not None and lib.get("obsidian").bundled

    with pytest.raises(BundledSkillError, match=r"skills.disabled"):
        lib.delete("obsidian")

    assert (fake_bundled / "note-taking" / "obsidian" / "SKILL.md").is_file()
    assert not (fake_bundled / "_archived").exists()


def test_skills_remove_bundled_is_refused_and_package_untouched(tmp_hermes_home: Path, fake_bundled: Path):
    result = CliRunner().invoke(cli, ["skills", "remove", "obsidian"])

    assert result.exit_code == 1, result.output
    assert "bundled" in result.output
    assert "skills.disabled" in result.output
    assert "Removed" not in result.output
    # The installed package is untouched and nothing was archived inside it.
    assert (fake_bundled / "note-taking" / "obsidian" / "SKILL.md").is_file()
    assert not (fake_bundled / "_archived").exists()


def test_skills_remove_user_shadow_still_archives_under_home(tmp_hermes_home: Path, fake_bundled: Path):
    """Removing a user skill that shadows a bundled one still works (gh #39) —
    the archive lands under HERMES_HOME and the bundled copy is re-exposed."""
    _write_skill(tmp_hermes_home / "skills", "obsidian")

    result = CliRunner().invoke(cli, ["skills", "remove", "obsidian"])

    assert result.exit_code == 0, result.output
    assert list((tmp_hermes_home / "skills" / "_archived").glob("obsidian-*"))
    assert (fake_bundled / "note-taking" / "obsidian" / "SKILL.md").is_file()
    assert SkillLibrary().get("obsidian").bundled


def test_curator_never_archives_or_rewrites_bundled_skills(tmp_hermes_home: Path, fake_bundled: Path):
    """An install older than archive_after_days must not lose its bundled skills."""
    bundled_md = fake_bundled / "note-taking" / "obsidian" / "SKILL.md"
    before = bundled_md.read_bytes()
    _write_skill(tmp_hermes_home / "skills", "old-user-skill", age_days=200)

    result = mark_stale_and_archive(SkillLibrary(), stale_days=30, archive_days=90)

    assert "obsidian" not in result["archived"]
    assert "obsidian" not in result["marked_stale"]
    assert result["archived"] == ["old-user-skill"]  # user skills still age out
    assert bundled_md.read_bytes() == before
    assert not (fake_bundled / "_archived").exists()


def test_curator_run_cli_leaves_bundled_skills_alone(tmp_hermes_home: Path, fake_bundled: Path):
    result = CliRunner().invoke(cli, ["curator", "run"])

    assert result.exit_code == 0, result.output
    assert "obsidian" not in result.output
    assert (fake_bundled / "note-taking" / "obsidian" / "SKILL.md").is_file()
    assert not (fake_bundled / "_archived").exists()
