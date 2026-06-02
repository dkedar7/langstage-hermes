"""Tests for ``deepagent_hermes.skills.library.SkillLibrary``."""

from __future__ import annotations

import sys
from pathlib import Path

import frontmatter
import pytest

from deepagent_hermes.skills.library import Skill, SkillLibrary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill(
    base: Path,
    *,
    name: str,
    description: str = "default description",
    category: str | None = None,
    body: str = "Body content.",
    extra: dict | None = None,
) -> Path:
    """Write a SKILL.md under ``base/[category/]name/SKILL.md``."""
    root = base / category / name if category else base / name
    root.mkdir(parents=True, exist_ok=True)
    skill_md = root / "SKILL.md"
    fm = {"name": name, "description": description}
    if extra:
        fm.update(extra)
    post = frontmatter.Post(body, **fm)
    skill_md.write_bytes(frontmatter.dumps(post).encode("utf-8"))
    return skill_md


@pytest.fixture
def lib(tmp_hermes_home) -> SkillLibrary:
    """SkillLibrary backed only by the tmp_hermes_home/skills dir."""
    skills_dir = tmp_hermes_home / "skills"
    return SkillLibrary(dirs=[skills_dir])


# ---------------------------------------------------------------------------
# Listing / get
# ---------------------------------------------------------------------------


def test_list_returns_all_three_skills(tmp_hermes_home, lib):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="alpha", description="alpha desc")
    _write_skill(skills_dir, name="beta", description="beta desc", category="cat1")
    _write_skill(skills_dir, name="gamma", description="gamma desc", category="cat2")

    names = [s.name for s in lib.list()]
    assert sorted(names) == ["alpha", "beta", "gamma"]


def test_get_returns_specific_skill(tmp_hermes_home, lib):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="alpha", description="alpha desc", body="Alpha body")
    _write_skill(skills_dir, name="beta", description="beta desc")

    skill = lib.get("alpha")
    assert skill is not None
    assert skill.name == "alpha"
    assert skill.description == "alpha desc"
    assert "Alpha body" in skill.body


def test_get_missing_returns_none(lib):
    assert lib.get("does-not-exist") is None


def test_category_extracted_from_path(tmp_hermes_home, lib):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="git-flow", category="software-development")
    _write_skill(skills_dir, name="solo")

    skills = {s.name: s for s in lib.list()}
    assert skills["git-flow"].category == "software-development"
    assert skills["solo"].category is None


def test_archived_skills_are_skipped(tmp_hermes_home, lib):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="active")
    _write_skill(skills_dir / "_archived", name="old-skill")
    names = [s.name for s in lib.list()]
    assert names == ["active"]


# ---------------------------------------------------------------------------
# Collision precedence (later dir wins)
# ---------------------------------------------------------------------------


def test_later_dir_wins_on_collision(tmp_hermes_home, tmp_path):
    earlier = tmp_path / "bundled"
    later = tmp_path / "project"
    earlier.mkdir()
    later.mkdir()
    _write_skill(earlier, name="shared", description="bundled version")
    _write_skill(later, name="shared", description="project override")

    lib = SkillLibrary(dirs=[earlier, later])
    skill = lib.get("shared")
    assert skill is not None
    assert skill.description == "project override"


# ---------------------------------------------------------------------------
# write() round-trip
# ---------------------------------------------------------------------------


def test_write_roundtrip(tmp_hermes_home, lib):
    skills_dir = tmp_hermes_home / "skills"
    fm = {"name": "round-trip", "description": "a round-trip test"}
    path = lib.write("round-trip", fm, "# Title\n\nBody.\n")
    assert path.exists()
    assert path.parent == skills_dir / "round-trip"

    skill = lib.get("round-trip")
    assert skill is not None
    assert skill.name == "round-trip"
    assert "# Title" in skill.body


def test_write_with_category(tmp_hermes_home, lib):
    skills_dir = tmp_hermes_home / "skills"
    fm = {"name": "categorized", "description": "x"}
    path = lib.write("categorized", fm, "Body", category="my-cat")
    assert path == skills_dir / "my-cat" / "categorized" / "SKILL.md"
    assert lib.get("categorized").category == "my-cat"


def test_write_rejects_invalid_frontmatter(lib):
    with pytest.raises(ValueError, match="invalid"):
        lib.write("BAD-NAME", {"name": "BAD-NAME", "description": "x"}, "body")


def test_write_rejects_name_mismatch(lib):
    with pytest.raises(ValueError, match="does not match"):
        lib.write("foo", {"name": "bar", "description": "x"}, "body")


# ---------------------------------------------------------------------------
# delete() archives
# ---------------------------------------------------------------------------


def test_delete_archives(tmp_hermes_home, lib):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="goner", description="bye")
    assert lib.get("goner") is not None

    assert lib.delete("goner") is True
    assert lib.get("goner") is None

    archive_root = skills_dir / "_archived"
    assert archive_root.exists()
    archived = list(archive_root.iterdir())
    assert len(archived) == 1
    assert archived[0].name.startswith("goner-")
    assert (archived[0] / "SKILL.md").exists()


def test_delete_missing_returns_false(lib):
    assert lib.delete("never-existed") is False


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_disabled_skill_filtered(tmp_hermes_home):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="alpha")
    _write_skill(skills_dir, name="beta")

    lib = SkillLibrary(dirs=[skills_dir], config={"disabled": ["beta"]})
    names = [s.name for s in lib.list()]
    assert names == ["alpha"]


def test_platform_disabled_filter(tmp_hermes_home, monkeypatch):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="alpha")
    _write_skill(skills_dir, name="beta")

    monkeypatch.setenv("HERMES_PLATFORM", "telegram")
    lib = SkillLibrary(
        dirs=[skills_dir],
        config={"platform_disabled": {"telegram": ["alpha"]}},
    )
    names = [s.name for s in lib.list()]
    assert names == ["beta"]


def test_platforms_field_filters_by_os(tmp_hermes_home, monkeypatch):
    skills_dir = tmp_hermes_home / "skills"
    # Skill restricted to a platform that does NOT match the current OS.
    other_platform = "linux" if sys.platform.startswith("win") else "windows"
    _write_skill(
        skills_dir, name="alpha", extra={"platforms": [other_platform]},
    )
    # Skill restricted to whatever the current OS is.
    if sys.platform.startswith("darwin"):
        current = "macos"
    elif sys.platform.startswith("win"):
        current = "windows"
    else:
        current = "linux"
    _write_skill(skills_dir, name="beta", extra={"platforms": [current]})

    lib = SkillLibrary(dirs=[skills_dir])
    names = [s.name for s in lib.list()]
    assert "beta" in names
    assert "alpha" not in names


# ---------------------------------------------------------------------------
# validate_all
# ---------------------------------------------------------------------------


def test_validate_all_reports_errors(tmp_hermes_home, lib):
    skills_dir = tmp_hermes_home / "skills"
    # Valid
    _write_skill(skills_dir, name="alpha", description="ok")
    # Invalid: name doesn't match parent dir
    bad = skills_dir / "bad"
    bad.mkdir()
    post = frontmatter.Post("body", name="not-bad", description="d")
    (bad / "SKILL.md").write_bytes(frontmatter.dumps(post).encode("utf-8"))

    results = lib.validate_all()
    assert results["alpha"] == []
    # The bad skill keys under its frontmatter name (not-bad) with parent_dir errors
    assert any(
        "parent directory" in err
        for errs in results.values()
        for err in errs
    )


# ---------------------------------------------------------------------------
# Empty / missing dirs are tolerated
# ---------------------------------------------------------------------------


def test_empty_dir_returns_no_skills(tmp_hermes_home, lib):
    assert lib.list() == []


def test_missing_dir_tolerated(tmp_path):
    lib = SkillLibrary(dirs=[tmp_path / "does-not-exist"])
    assert lib.list() == []
