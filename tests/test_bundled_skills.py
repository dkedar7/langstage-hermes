"""Validate every bundled SKILL.md against the agentskills.io spec.

If this test breaks, a borrowed skill has frontmatter that won't load — fix
the skill (don't loosen the validator).
"""

from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest

from langstage_hermes.skills.validator import validate

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_DIR = REPO_ROOT / "src" / "langstage_hermes" / "_bundled_skills"


def _all_skill_files() -> list[Path]:
    return sorted(BUNDLED_DIR.rglob("SKILL.md"))


def test_some_skills_bundled():
    skills = _all_skill_files()
    # We promised at least the v0.1.0a1 curated set.
    assert len(skills) >= 20, f"Expected >=20 bundled skills, found {len(skills)}"


@pytest.mark.parametrize("skill_path", _all_skill_files(), ids=lambda p: p.parent.name)
def test_bundled_skill_validates(skill_path: Path):
    post = frontmatter.load(skill_path)
    fm = dict(post.metadata)
    errs = validate(fm, parent_dir_name=skill_path.parent.name)
    assert not errs, f"{skill_path.relative_to(REPO_ROOT)}: {errs}"


def test_no_duplicate_skill_names():
    seen: dict[str, Path] = {}
    for p in _all_skill_files():
        post = frontmatter.load(p)
        name = post.metadata.get("name")
        if name in seen:
            pytest.fail(f"duplicate name {name!r} at {p} and {seen[name]}")
        seen[name] = p


# ── Runtime loading path (gh #-dogfood) ──────────────────────────────
#
# The tests above validate the SKILL.md files on disk via a hard-coded path —
# they never exercise `_bundled_skills_dir()` / `SkillLibrary`, so a bug there
# (it resolved `parents[3] / "skills"`, a nonexistent repo-root dir, instead of
# the in-package `_bundled_skills/`) loaded ZERO skills at runtime while these
# file tests stayed green. These tests go through the real runtime path.


def test_bundled_skills_dir_resolves_to_existing_dir():
    from langstage_hermes.skills.library import _bundled_skills_dir

    d = _bundled_skills_dir()
    assert d.is_dir(), f"_bundled_skills_dir() -> {d} does not exist"
    assert list(d.rglob("SKILL.md")), f"no SKILL.md under {d}"


def test_skilllibrary_loads_bundled_skills():
    """The library must actually LOAD bundled skills, not just ship the files."""
    from langstage_hermes.skills.library import SkillLibrary, _bundled_skills_dir

    loaded = SkillLibrary(dirs=[_bundled_skills_dir()]).list()
    assert len(loaded) >= 20, f"bundled SkillLibrary loaded only {len(loaded)}"


def test_default_skilllibrary_includes_bundled(tmp_path, monkeypatch):
    """The public-API default SkillLibrary() (bundled+user+project) loads bundled
    even with an empty HERMES_HOME — the path the dogfood repro hit."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.delenv("LANGSTAGE_HERMES_HOME", raising=False)
    from langstage_hermes.skills.library import SkillLibrary

    assert len(SkillLibrary().list()) >= 20


# ── exclusions are relative to the search dir, not the install prefix ──
#
# gh #-dogfood: the README's `uv venv .venv` puts the package under a `.venv`
# path; `_EXCLUDED_DIR_NAMES` was matched against the ABSOLUTE path, so every
# bundled skill's path contained `.venv` and ZERO skills loaded (verify -> exit 2).


def test_iter_skill_md_ignores_excluded_name_in_install_prefix(tmp_path):
    """A `.venv` in the install prefix (above the search dir) must NOT exclude skills."""
    from langstage_hermes.skills.library import SkillLibrary

    root = tmp_path / ".venv" / "Lib" / "site-packages" / "pkg" / "_bundled_skills"
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "SKILL.md").write_text("placeholder", encoding="utf-8")
    found = list(SkillLibrary._iter_skill_md_files(root))
    assert len(found) == 1, found


def test_iter_skill_md_still_skips_excluded_dirs_inside_tree(tmp_path):
    """Junk dirs *inside* the skill tree are still skipped (relative match)."""
    from langstage_hermes.skills.library import SkillLibrary

    root = tmp_path / "skills"
    (root / "real").mkdir(parents=True)
    (root / "real" / "SKILL.md").write_text("placeholder", encoding="utf-8")
    (root / ".venv" / "junk").mkdir(parents=True)
    (root / ".venv" / "junk" / "SKILL.md").write_text("placeholder", encoding="utf-8")
    found = {p.parent.name for p in SkillLibrary._iter_skill_md_files(root)}
    assert found == {"real"}, found


def test_skilllibrary_loads_from_venv_named_install_path(tmp_path):
    """End-to-end: a valid skill under a `.venv` path loads (the real failure)."""
    from langstage_hermes.skills.library import SkillLibrary

    root = tmp_path / ".venv" / "Lib" / "site-packages" / "langstage_hermes" / "_bundled_skills"
    (root / "demo-skill").mkdir(parents=True)
    (root / "demo-skill" / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: A demo skill.\nversion: 1.0.0\nplatforms: [linux, macos, windows]\n---\nbody\n",
        encoding="utf-8",
    )
    names = [s.name for s in SkillLibrary(dirs=[root]).list()]
    assert "demo-skill" in names, names
