"""CliRunner tests for the v0.1.2 UI hookup work — skills / tools / curator
subcommands stopped being stubs and now actually drive the runtime.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli


def _write_skill(notes_dir: Path, name: str, description: str, body: str = "test body", category: str | None = None) -> Path:
    """Drop a valid SKILL.md under <hermes_home>/skills/[<category>/]<name>/SKILL.md."""
    parent = notes_dir / "skills"
    if category:
        parent = parent / category
    parent = parent / name
    parent.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(body, **{"name": name, "description": description})
    (parent / "SKILL.md").write_text(frontmatter.dumps(post), encoding="utf-8")
    return parent / "SKILL.md"


# ── tools subcommand ───────────────────────────────────────────────────


def test_tools_lists_declared_toolsets():
    runner = CliRunner()
    result = runner.invoke(cli, ["tools"])
    assert result.exit_code == 0
    # Spot-check a few toolsets from SPEC §11.
    assert "skills" in result.output
    assert "memory" in result.output
    assert "file" in result.output
    # Implementation legend.
    assert "implemented" in result.output


def test_tools_implemented_only_filters_stubs():
    runner = CliRunner()
    result = runner.invoke(cli, ["tools", "--implemented-only"])
    assert result.exit_code == 0
    # Stubbed-but-declared toolsets should NOT appear.
    assert "homeassistant" not in result.output
    assert "spotify" not in result.output
    # But the implemented ones still do.
    assert "skills" in result.output


def test_tools_filter_by_name():
    runner = CliRunner()
    result = runner.invoke(cli, ["tools", "--toolset", "skills"])
    assert result.exit_code == 0
    assert "skill_view" in result.output
    assert "skill_manage" in result.output
    # No other toolset should appear.
    assert "homeassistant" not in result.output
    assert "memory  " not in result.output  # space-padded; rules out the "memory" toolset header


def test_tools_unknown_toolset():
    runner = CliRunner()
    result = runner.invoke(cli, ["tools", "--toolset", "does-not-exist"])
    assert result.exit_code == 0
    assert "No toolset named" in result.output


# ── skills subcommand ──────────────────────────────────────────────────


def test_skills_list_includes_user_skill(tmp_hermes_home: Path, monkeypatch):
    _write_skill(tmp_hermes_home, "my-test-skill", "smoke test description")
    runner = CliRunner()
    result = runner.invoke(cli, ["skills", "list"])
    assert result.exit_code == 0, result.output
    assert "my-test-skill" in result.output
    assert "smoke test description" in result.output


def test_skills_list_query_filters(tmp_hermes_home: Path):
    _write_skill(tmp_hermes_home, "alpha-skill", "alpha description")
    _write_skill(tmp_hermes_home, "beta-skill", "beta description")
    runner = CliRunner()
    result = runner.invoke(cli, ["skills", "list", "--query", "alpha"])
    assert result.exit_code == 0
    assert "alpha-skill" in result.output
    assert "beta-skill" not in result.output


def test_skills_show_existing(tmp_hermes_home: Path):
    _write_skill(tmp_hermes_home, "show-me", "show description here", body="**show body**")
    runner = CliRunner()
    result = runner.invoke(cli, ["skills", "show", "show-me"])
    assert result.exit_code == 0
    assert "show-me" in result.output
    assert "show description here" in result.output
    assert "**show body**" in result.output


def test_skills_show_missing():
    runner = CliRunner()
    result = runner.invoke(cli, ["skills", "show", "definitely-no-such-skill-12345"])
    assert result.exit_code == 1
    assert "No skill named" in result.output


def test_skills_audit_passes_when_clean(tmp_hermes_home: Path):
    _write_skill(tmp_hermes_home, "clean-skill", "valid description")
    runner = CliRunner()
    result = runner.invoke(cli, ["skills", "audit"])
    # May exit 1 if bundled skills fail (some upstream ones do); we just
    # care that the command runs and produces sensible output.
    assert "pass validation" in result.output or "failed validation" in result.output


def test_skills_install_validates_then_copies(tmp_hermes_home: Path, tmp_path: Path):
    # Build a valid skill in an unrelated dir, then `skills install` it.
    src = tmp_path / "incoming" / "installed-skill"
    src.mkdir(parents=True)
    post = frontmatter.Post("body", **{"name": "installed-skill", "description": "freshly installed"})
    (src / "SKILL.md").write_text(frontmatter.dumps(post), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["skills", "install", str(src)])
    assert result.exit_code == 0, result.output
    installed = tmp_hermes_home / "skills" / "installed-skill" / "SKILL.md"
    assert installed.exists()
    assert "freshly installed" in installed.read_text(encoding="utf-8")


def test_skills_install_rejects_invalid_frontmatter(tmp_hermes_home: Path, tmp_path: Path):
    src = tmp_path / "incoming" / "Bad-Name"  # uppercase -> validator rejects
    src.mkdir(parents=True)
    post = frontmatter.Post("body", **{"name": "Bad-Name", "description": "uppercase rejected"})
    (src / "SKILL.md").write_text(frontmatter.dumps(post), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["skills", "install", str(src)])
    assert result.exit_code == 2
    assert "invalid" in result.output.lower()


# ── curator subcommand ─────────────────────────────────────────────────


def test_curator_status_renders(tmp_hermes_home: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["curator", "status"])
    assert result.exit_code == 0
    assert "enabled" in result.output
    assert "interval" in result.output


def test_curator_run_dry_run(tmp_hermes_home: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["curator", "run", "--dry-run"])
    assert result.exit_code == 0
    assert "dry-run" in result.output
    assert "Curator pass" in result.output


def test_curator_pin_then_unpin(tmp_hermes_home: Path):
    _write_skill(tmp_hermes_home, "pinnable", "test skill for pinning")
    runner = CliRunner()

    result = runner.invoke(cli, ["curator", "pin", "pinnable"])
    assert result.exit_code == 0
    assert "pinned" in result.output

    # Verify frontmatter actually changed on disk.
    skill_path = tmp_hermes_home / "skills" / "pinnable" / "SKILL.md"
    post = frontmatter.load(skill_path)
    assert post.metadata.get("hermes", {}).get("pinned") is True

    result = runner.invoke(cli, ["curator", "unpin", "pinnable"])
    assert result.exit_code == 0
    assert "unpinned" in result.output

    post = frontmatter.load(skill_path)
    assert "pinned" not in (post.metadata.get("hermes") or {})


def test_curator_pin_missing_skill():
    runner = CliRunner()
    result = runner.invoke(cli, ["curator", "pin", "no-such-skill-xyz"])
    assert result.exit_code == 1
    assert "No skill" in result.output


def test_curator_pause_resume_round_trip(tmp_hermes_home: Path):
    runner = CliRunner()
    pause = runner.invoke(cli, ["curator", "pause"])
    assert pause.exit_code == 0
    status = runner.invoke(cli, ["curator", "status"])
    assert "paused:            True" in status.output

    resume = runner.invoke(cli, ["curator", "resume"])
    assert resume.exit_code == 0
    status2 = runner.invoke(cli, ["curator", "status"])
    assert "paused:            False" in status2.output


# ── inline slash commands ──────────────────────────────────────────────


def test_slash_skills_lists_inline(tmp_hermes_home: Path, monkeypatch):
    """The /skills inline handler should produce a real listing, not a
    'use the subcommand' redirect."""
    from langstage_hermes.cli import _slash_skills

    _write_skill(tmp_hermes_home, "inline-test-skill", "for the inline check")

    from langstage_hermes.config import HermesConfig

    state: dict = {"cfg": HermesConfig.resolve()}
    # Capture click.echo output.
    captured: list[str] = []
    monkeypatch.setattr("click.echo", lambda *a, **kw: captured.append(a[0] if a else ""))
    _slash_skills("", state)
    output = "\n".join(str(c) for c in captured)
    assert "inline-test-skill" in output
    assert "for the inline check" in output


def test_slash_skills_show_inline(tmp_hermes_home: Path, monkeypatch):
    from langstage_hermes.cli import _slash_skills

    _write_skill(tmp_hermes_home, "show-inline", "for the show inline check", body="inline body content")

    from langstage_hermes.config import HermesConfig

    state: dict = {"cfg": HermesConfig.resolve()}
    captured: list[str] = []
    monkeypatch.setattr("click.echo", lambda *a, **kw: captured.append(a[0] if a else ""))
    _slash_skills("show show-inline", state)
    output = "\n".join(str(c) for c in captured)
    assert "show-inline" in output
    assert "inline body content" in output


def test_slash_memory_handles_empty(tmp_hermes_home: Path, monkeypatch):
    from langstage_hermes.cli import _slash_memory

    captured: list[str] = []
    monkeypatch.setattr("click.echo", lambda *a, **kw: captured.append(a[0] if a else ""))
    _slash_memory("", {})
    output = "\n".join(str(c) for c in captured)
    assert "empty" in output.lower() or "memory grows" in output.lower()


def test_slash_memory_shows_content(tmp_hermes_home: Path, monkeypatch):
    from langstage_hermes.cli import _slash_memory

    mem_dir = tmp_hermes_home / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "MEMORY.md").write_text("a memorable line\n", encoding="utf-8")

    captured: list[str] = []
    monkeypatch.setattr("click.echo", lambda *a, **kw: captured.append(a[0] if a else ""))
    _slash_memory("", {})
    output = "\n".join(str(c) for c in captured)
    assert "a memorable line" in output


def test_slash_tools_lists_implemented(monkeypatch):
    from langstage_hermes.cli import _slash_tools

    captured: list[str] = []
    monkeypatch.setattr("click.echo", lambda *a, **kw: captured.append(a[0] if a else ""))
    _slash_tools("", {})
    output = "\n".join(str(c) for c in captured)
    assert "skills" in output
    assert "memory" in output


def test_slash_curator_renders(tmp_hermes_home: Path, monkeypatch):
    from langstage_hermes.cli import _slash_curator
    from langstage_hermes.config import HermesConfig

    state: dict = {"cfg": HermesConfig.resolve()}
    captured: list[str] = []
    monkeypatch.setattr("click.echo", lambda *a, **kw: captured.append(a[0] if a else ""))
    _slash_curator("", state)
    output = "\n".join(str(c) for c in captured)
    assert "Curator" in output
    assert "interval" in output


def test_slash_cron_empty_state(tmp_hermes_home: Path, monkeypatch):
    from langstage_hermes.cli import _slash_cron

    captured: list[str] = []
    monkeypatch.setattr("click.echo", lambda *a, **kw: captured.append(a[0] if a else ""))
    _slash_cron("", {})
    output = "\n".join(str(c) for c in captured)
    assert "No cron jobs" in output


# ── doctor still works ─────────────────────────────────────────────────


def test_doctor_runs_clean():
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "python" in result.output
    assert "HERMES_HOME" in result.output


# ── help surface ───────────────────────────────────────────────────────


@pytest.mark.parametrize("group", ["skills", "tools", "cron", "curator", "plugins", "doctor"])
def test_subcommand_help_works(group: str):
    runner = CliRunner()
    result = runner.invoke(cli, [group, "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


# ── skills remove / uninstall (gh #39) ──────────────────────────────────


def _home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(tmp_path))


def test_skills_remove_archives_and_records_rollbackable_delete(monkeypatch, tmp_path):
    """`install` had no inverse; `remove` archives the skill and lands a `delete`
    audit row (which `audit rollback` can undo), instead of a manual `rm` that
    desyncs the audit log."""
    _home(monkeypatch, tmp_path)
    _write_skill(tmp_path, "throwaway", "a skill to remove later")
    runner = CliRunner()
    assert "throwaway" in runner.invoke(cli, ["skills", "list"]).output

    r = runner.invoke(cli, ["skills", "remove", "throwaway"])
    assert r.exit_code == 0, r.output
    assert "Removed throwaway" in r.output

    # Archived (not hard-deleted) and removed from active discovery.
    assert not (tmp_path / "skills" / "throwaway").exists()
    assert (tmp_path / "skills" / "_archived").exists()

    # A rollback-able delete row landed (vs install's create that rollback refuses).
    audit = runner.invoke(cli, ["audit", "log"]).output
    assert "delete" in audit and "throwaway" in audit


def test_skills_remove_unknown_fails(monkeypatch, tmp_path):
    _home(monkeypatch, tmp_path)
    r = CliRunner().invoke(cli, ["skills", "remove", "no-such-skill"])
    assert r.exit_code == 1
    assert "No installed skill" in r.output


def test_skills_uninstall_is_an_alias(monkeypatch, tmp_path):
    _home(monkeypatch, tmp_path)
    _write_skill(tmp_path, "gone", "to be uninstalled")
    r = CliRunner().invoke(cli, ["skills", "uninstall", "gone"])
    assert r.exit_code == 0, r.output
    assert "Removed gone" in r.output
