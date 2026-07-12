"""Tests for the keyless / offline reflection→skill-creation demo (gh #69).

The demo drives the REAL shipped machinery — ``create_hermes_agent`` with the
genuine ``ReflectionMiddleware``, the ``task`` review dispatch, the real
``skill_manage`` / ``memory`` tools, ``SkillLibrary.write()``, the audit log and
the FTS5 store — against a scripted fake model instead of a live provider. These
tests assert the loop closes with NO API key and writes the real side effects,
and that the ``demo`` CLI command surfaces them and cleans up after itself.

Deliberately exercising the genuine loader + middleware (not a file-glob
stand-in) is the point per the issue: a scripted offline run catches the same
class of bug a real user would hit.
"""

from __future__ import annotations

import os
import tempfile as tempfile_mod
from pathlib import Path

import frontmatter
import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.demo import DEMO_SKILL_NAME, run_demo


def test_run_demo_writes_real_skill_and_memory_offline(tmp_path: Path):
    """The genuine loop closes: a real, valid SKILL.md + memory note land on disk."""
    home = tmp_path / "home"
    res = run_demo(home=home, nudge_interval=3)

    # The review subagent wrote a real SKILL.md via the genuine skill_manage path.
    assert res.skill_created
    assert res.skill_path is not None and res.skill_path.is_file()
    assert res.skill_path.is_relative_to(home)  # under the demo's HERMES_HOME
    assert res.skill_name == DEMO_SKILL_NAME

    # It's a valid agentskills.io SKILL.md — frontmatter parses with name + description.
    post = frontmatter.load(str(res.skill_path))
    assert post.metadata.get("name") == DEMO_SKILL_NAME
    assert post.metadata.get("description")
    assert post.content.strip()

    # The review subagent also persisted a durable memory note.
    assert res.memory_path is not None and res.memory_path.is_file()
    assert res.memory_entries

    # Genuine side effects were recorded — the audit log and FTS5 store, not a stub.
    assert "create" in res.audit_actions
    assert res.sessions_recorded >= 1


def test_run_demo_needs_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Prove it's truly keyless by stripping every provider credential first."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    res = run_demo(home=tmp_path / "home", nudge_interval=2)
    assert res.skill_created
    assert res.tool_iterations == 2


def test_run_demo_restores_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """run_demo points HERMES_HOME at its throwaway home only for the run, then
    restores the caller's environment (it's library-callable, not just a CLI)."""
    monkeypatch.setenv("HERMES_HOME", "/some/preexisting/home")
    monkeypatch.delenv("DEEPAGENT_HERMES_HOME", raising=False)

    run_demo(home=tmp_path / "home", nudge_interval=2)

    assert os.environ["HERMES_HOME"] == "/some/preexisting/home"
    assert "DEEPAGENT_HERMES_HOME" not in os.environ


def test_demo_command_closes_loop_and_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The ``demo`` command reports the loop closing and removes its throwaway home."""
    scratch = tmp_path / "tmproot"
    scratch.mkdir()
    monkeypatch.setattr(tempfile_mod, "tempdir", str(scratch))

    result = CliRunner().invoke(cli, ["demo"])

    assert result.exit_code == 0, result.output
    assert "DEMO: PASS" in result.output
    assert DEMO_SKILL_NAME in result.output
    # The whole throwaway HERMES_HOME (state.db included) is fully removed.
    assert list(scratch.glob("dah-demo-*")) == [], "demo leaked its throwaway HERMES_HOME"


def test_demo_command_keep_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``--keep-workspace`` preserves the home so the user can inspect the skill."""
    scratch = tmp_path / "tmproot"
    scratch.mkdir()
    monkeypatch.setattr(tempfile_mod, "tempdir", str(scratch))

    result = CliRunner().invoke(cli, ["demo", "--keep-workspace"])

    assert result.exit_code == 0, result.output
    kept = list(scratch.glob("dah-demo-*"))
    assert len(kept) == 1, f"expected the home to be kept, found: {kept}"
    assert list(kept[0].rglob("SKILL.md")), "kept home should contain the generated SKILL.md"
