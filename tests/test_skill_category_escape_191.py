"""gh #191: a model-supplied ``category`` must not write outside the skill library.

``skill_manage(create)`` passed ``category`` straight into ``SkillLibrary.write``,
which joined it into the path: ``../`` or an absolute path planted a SKILL.md
anywhere the process could write, reported as success and invisible to
``skills list``. The guard lives in ``write`` so every caller is covered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import frontmatter
import pytest

from langstage_hermes.skills.library import SkillLibrary
from langstage_hermes.skills.tools import make_skill_tools


@pytest.fixture
def lib(tmp_path: Path) -> SkillLibrary:
    return SkillLibrary(dirs=[tmp_path / "skills"])


def _create(lib: SkillLibrary, category: str) -> dict:
    tool = {t.name: t for t in make_skill_tools(lib)}["skill_manage"]
    out = tool.invoke(
        {
            "type": "tool_call",
            "id": "c1",
            "name": "skill_manage",
            "args": {"action": "create", "name": "planted", "description": "x", "body": "b", "category": category},
        }
    )
    return json.loads(out.update["messages"][0].content)


def _planted(root: Path) -> list[Path]:
    return sorted(root.rglob("SKILL.md"))


@pytest.mark.parametrize(
    "category",
    [
        "../../victim",
        "..",
        ".",
        "a/b",
        "..\\..\\victim",
        "a\\b",
        "C:victim",
        "C:\\victim",
        "\\\\server\\share",
        "x\0y",
        "_archived",
        ".git",
        "node_modules",
        "_private",
        ".hidden",
        "trailing.",
        "trailing ",
        "CON",
        "nul.txt",
        "  ",
    ],
)
def test_create_rejects_unsafe_category(lib: SkillLibrary, tmp_path: Path, category: str) -> None:
    result = _create(lib, category)
    assert result["success"] is False
    assert "category" in result["error"]
    assert _planted(tmp_path) == []


def test_create_rejects_absolute_category(lib: SkillLibrary, tmp_path: Path) -> None:
    result = _create(lib, str(tmp_path / "victim"))
    assert result["success"] is False
    assert _planted(tmp_path) == []


def test_plain_category_still_works(lib: SkillLibrary, tmp_path: Path) -> None:
    result = _create(lib, "data-science")
    assert result["success"] is True
    assert Path(result["path"]) == tmp_path / "skills" / "data-science" / "planted" / "SKILL.md"
    assert [s.category for s in lib.list()] == ["data-science"]


def test_existing_category_keeps_its_name(lib: SkillLibrary, tmp_path: Path) -> None:
    # A user-made category dir that a *new* one couldn't be named is still writable.
    (tmp_path / "skills" / "_drafts").mkdir(parents=True)
    path = lib.write("planted", {"description": "x"}, "b", category="_drafts")
    assert path == tmp_path / "skills" / "_drafts" / "planted" / "SKILL.md"


def test_write_rejects_unsafe_category_directly(lib: SkillLibrary, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="category"):
        lib.write("planted", {"description": "x"}, "b", category="../escape")
    assert _planted(tmp_path) == []


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")


def test_symlinked_category_cannot_escape(lib: SkillLibrary, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "skills").mkdir()
    _symlink_or_skip(tmp_path / "skills" / "linked", outside)
    result = _create(lib, "linked")
    assert result["success"] is False
    assert list(outside.iterdir()) == []


def test_symlinked_skill_dir_cannot_escape(lib: SkillLibrary, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "skills").mkdir()
    _symlink_or_skip(tmp_path / "skills" / "planted", outside)
    with pytest.raises(ValueError, match="outside the skill library"):
        lib.write("planted", {"description": "x"}, "b")
    assert list(outside.iterdir()) == []


def test_write_file_keeps_an_existing_skills_category(lib: SkillLibrary, tmp_path: Path) -> None:
    lib.write("planted", {"description": "x"}, "b", category="research")
    tool = {t.name: t for t in make_skill_tools(lib)}["skill_manage"]
    out = tool.invoke(
        {
            "type": "tool_call",
            "id": "c2",
            "name": "skill_manage",
            "args": {"action": "write_file", "name": "planted", "frontmatter_data": {"description": "y"}, "body": "c"},
        }
    )
    result = json.loads(out.update["messages"][0].content)
    assert result["success"] is True
    md = tmp_path / "skills" / "research" / "planted" / "SKILL.md"
    assert frontmatter.load(md).metadata["description"] == "y"
