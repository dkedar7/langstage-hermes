"""gh #241: ``skills.external_dirs`` are discovery-only.

They came last in the search order (they win a name collision), and the library
wrote new skills to "the last dir in the search list", so every agent-authored
skill landed in the user's team checkout or read-only mount, and ``demo`` failed
after leaving its skill there. Now new skills land in ``<HERMES_HOME>/skills``
(or the project shadow), and the agent refuses to patch, overwrite, pin or
archive a skill that lives in an external dir.
"""

from __future__ import annotations

import json
from pathlib import Path

import frontmatter
import pytest

from langstage_hermes.config import HermesConfig
from langstage_hermes.skills.library import ExternalSkillError, ReadOnlySkillError, SkillLibrary, skill_search_dirs


def _write_skill(base: Path, *, name: str, category: str | None = None, body: str = "Body content.") -> Path:
    root = base / category / name if category else base / name
    root.mkdir(parents=True, exist_ok=True)
    skill_md = root / "SKILL.md"
    skill_md.write_bytes(frontmatter.dumps(frontmatter.Post(body, name=name, description="A team skill.")).encode())
    return skill_md


@pytest.fixture
def team(tmp_hermes_home, tmp_path, monkeypatch) -> Path:
    """An external skills dir set through the env, with one team skill in it."""
    monkeypatch.delenv("DEEPAGENT_HERMES_SKILLS_EXTERNAL_DIRS", raising=False)
    monkeypatch.chdir(tmp_path)
    ext = tmp_path / "team-skills"
    _write_skill(ext, name="team-skill", category="shared")
    monkeypatch.setenv("LANGSTAGE_HERMES_SKILLS_EXTERNAL_DIRS", str(ext))
    return ext


def _agent_library() -> SkillLibrary:
    dirs, external = skill_search_dirs(HermesConfig.resolve())
    return SkillLibrary(dirs, external_dirs=external)


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_new_skill_lands_in_hermes_home_not_external(tmp_hermes_home, team):
    """The issue's runtime repro: ``lib.write`` went to the team checkout."""
    before = _files(team)
    path = _agent_library().write("review-made", {"name": "review-made", "description": "x"}, "# x\nbody\n")
    assert path == tmp_hermes_home / "skills" / "review-made" / "SKILL.md"
    assert _files(team) == before


def test_project_shadow_still_receives_new_skills(tmp_hermes_home, team, tmp_path):
    """The write target is unchanged when there are no external dirs in the way."""
    shadow = tmp_path / ".langstage-hermes" / "skills"
    shadow.mkdir(parents=True)
    path = _agent_library().write("proj-made", {"name": "proj-made", "description": "x"}, "body\n")
    assert path == shadow / "proj-made" / "SKILL.md"


def test_external_skill_is_still_discovered_and_flagged(team):
    skill = _agent_library().get("team-skill")
    assert skill is not None
    assert skill.external and skill.external_dir == team


def test_external_dir_pointing_at_hermes_home_stays_writable(tmp_hermes_home, monkeypatch, tmp_path):
    """An external_dirs entry that is really the user dir is not read-only."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_SKILLS_EXTERNAL_DIRS", str(tmp_hermes_home / "skills"))
    lib = _agent_library()
    assert lib.external_dirs == []
    path = lib.write("mine", {"name": "mine", "description": "x"}, "body\n")
    assert path == tmp_hermes_home / "skills" / "mine" / "SKILL.md"


@pytest.mark.parametrize(
    ("action", "kwargs", "verb"),
    [
        ("create", {"description": "d", "body": "b", "category": ""}, "overwrite"),
        ("patch", {"old_str": "Body content.", "new_str": "Changed."}, "patch"),
        ("write_file", {"frontmatter_data": {"description": "d"}, "body": "b"}, "overwrite"),
        ("pin", {}, "modify"),
        ("delete", {}, "archive"),
    ],
)
def test_skill_manage_refuses_to_change_an_external_skill(team, action, kwargs, verb):
    from langstage_hermes.skills.tools import _skill_manage_impl

    before = {p: p.read_bytes() for p in _files(team)}
    args = {"description": "", "body": "", "category": "", "old_str": "", "new_str": "", "frontmatter_data": None}
    args.update(kwargs)
    out = _skill_manage_impl(_agent_library(), action=action, name="team-skill", tool_call_id="", **args)
    payload = json.loads(out.update["__content__"])
    assert payload["success"] is False
    assert "skills.external_dirs" in payload["error"] and f"refusing to {verb} it" in payload["error"]
    assert {p: p.read_bytes() for p in _files(team)} == before


def test_library_methods_raise_external_skill_error(team):
    lib = _agent_library()
    with pytest.raises(ExternalSkillError):
        lib.delete("team-skill")
    with pytest.raises(ExternalSkillError):
        lib.update_frontmatter("team-skill", {"name": "team-skill", "description": "y"}, audit_action="pin")
    with pytest.raises(ReadOnlySkillError):  # the CLI's catch-all for bundled + external
        lib.write("team-skill", {"description": "y"}, "body\n")


def test_curator_skips_external_skills(team):
    from langstage_hermes.curator import mark_stale_and_archive

    before = {p: p.read_bytes() for p in _files(team)}
    report = mark_stale_and_archive(_agent_library(), stale_days=0, archive_days=0, state_meta_get=lambda name: 0.0, now=10**10)
    assert "team-skill" not in report.get("archived", []) + report.get("marked_stale", [])
    assert {p: p.read_bytes() for p in _files(team)} == before


def test_cli_skills_remove_refuses_external_skill(team, capsys):
    from langstage_hermes.cli import _skills_remove

    with pytest.raises(SystemExit) as exc:
        _skills_remove("team-skill")
    assert exc.value.code == 1
    assert "skills.external_dirs" in capsys.readouterr().out
    assert (team / "shared" / "team-skill" / "SKILL.md").exists()


def test_demo_passes_with_external_dirs_and_project_shadow(tmp_path, monkeypatch):
    """The issue's headline repro: ``demo`` exited 1 and left its skill in the team dir."""
    from langstage_hermes.demo import run_demo

    monkeypatch.chdir(tmp_path)
    ext = tmp_path / "team-skills"
    ext.mkdir()
    shadow = tmp_path / ".langstage-hermes" / "skills"
    shadow.mkdir(parents=True)
    monkeypatch.setenv("LANGSTAGE_HERMES_SKILLS_EXTERNAL_DIRS", str(ext))
    result = run_demo(home=tmp_path / "demo-home")
    assert result.skill_created
    assert result.skill_path is not None and (tmp_path / "demo-home") in result.skill_path.parents
    assert _files(ext) == [] and _files(shadow) == []
    assert Path.cwd() == tmp_path
