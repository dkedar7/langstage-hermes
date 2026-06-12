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
