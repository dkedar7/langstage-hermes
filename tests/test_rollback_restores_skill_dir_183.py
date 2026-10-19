"""gh #183: rolling back a `skills remove` restores the whole skill directory.

`delete` archives the directory (SKILL.md plus scripts/, references/, assets/), but
the rollback used to write back only SKILL.md and leave the supporting files in
`_archived/`, so the "restored" skill was silently broken.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.skills.library import SkillLibrary


def _make_tool_skill(skills: Path) -> Path:
    lib = SkillLibrary(dirs=[skills])
    lib.write(
        "tool-skill", {"name": "tool-skill", "description": "Run the helper script."}, "# Tool skill\n\nRun scripts/helper.py.\n"
    )
    d = skills / "tool-skill"
    (d / "scripts").mkdir()
    (d / "scripts" / "helper.py").write_text("print(42)\n", encoding="utf-8")
    (d / "references").mkdir()
    (d / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
    return d


def _remove(runner: CliRunner, name: str) -> str:
    r = runner.invoke(cli, ["skills", "remove", name])
    assert r.exit_code == 0, r.output
    marker = f"langstage-hermes audit rollback {name} "
    return r.output.split(marker, 1)[1].split("`", 1)[0].strip()


def test_rollback_of_remove_restores_supporting_files(tmp_hermes_home: Path):
    skills = tmp_hermes_home / "skills"
    d = _make_tool_skill(skills)
    runner = CliRunner()
    mutation_id = _remove(runner, "tool-skill")
    assert not d.exists()

    rb = runner.invoke(cli, ["audit", "rollback", "tool-skill", mutation_id, "--yes"])
    assert rb.exit_code == 0, rb.output
    assert (d / "SKILL.md").is_file()
    assert (d / "scripts" / "helper.py").read_text(encoding="utf-8") == "print(42)\n"
    assert (d / "references" / "notes.md").is_file()
    # The archive moved back: no duplicate copy left in _archived/.
    archived = skills / "_archived"
    assert not [p for p in archived.iterdir() if p.name.startswith("tool-skill-")]
    assert SkillLibrary(dirs=[skills]).get("tool-skill") is not None


def test_rollback_into_a_recreated_dir_fills_in_without_overwriting(tmp_hermes_home: Path):
    skills = tmp_hermes_home / "skills"
    d = _make_tool_skill(skills)
    runner = CliRunner()
    mutation_id = _remove(runner, "tool-skill")
    # Something recreated the directory with a different helper in the meantime.
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "helper.py").write_text("print('new')\n", encoding="utf-8")

    rb = runner.invoke(cli, ["audit", "rollback", "tool-skill", mutation_id, "--yes"])
    assert rb.exit_code == 0, rb.output
    assert (d / "SKILL.md").is_file()
    assert (d / "scripts" / "helper.py").read_text(encoding="utf-8") == "print('new')\n"
    assert (d / "references" / "notes.md").is_file()


def test_rollback_without_the_archive_still_restores_skill_md(tmp_hermes_home: Path):
    import shutil

    skills = tmp_hermes_home / "skills"
    d = _make_tool_skill(skills)
    runner = CliRunner()
    mutation_id = _remove(runner, "tool-skill")
    shutil.rmtree(skills / "_archived")  # cleaned up by hand

    rb = runner.invoke(cli, ["audit", "rollback", "tool-skill", mutation_id, "--yes"])
    assert rb.exit_code == 0, rb.output
    assert (d / "SKILL.md").is_file()
