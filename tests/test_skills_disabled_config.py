"""Regression tests for gh #74 — ``skills.disabled`` / ``skills.platform_disabled``
must actually exclude a skill from the agent's loaded toolset and from
``skills list``, not just resolve through config and get ignored.

The filter logic in ``SkillLibrary.list()`` was already correct and unit-tested
(see ``test_skill_library.py``); the bug was that no *production* caller threaded
the config in, so ``SkillLibrary.config`` was always ``{}`` and the filter was
dead code. These tests lock in the wiring at each construction site: the agent
runtime, the CLI's ``_skill_library`` (``skills list``/``show``), and the
module-level ``skills_list``/``skill_view`` tool library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import frontmatter
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langstage_hermes.agent import create_hermes_agent
from langstage_hermes.config import HermesConfig


def _write_skill(base: Path, *, name: str, description: str = "d", body: str = "Body.") -> None:
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(body, name=name, description=description)
    (root / "SKILL.md").write_bytes(frontmatter.dumps(post).encode("utf-8"))


def _stub_model() -> Any:
    """A `BaseChatModel` that builds the graph without a real API call."""
    return FakeListChatModel(responses=["stub"])


# ---------------------------------------------------------------------------
# The helper that shapes the config dict
# ---------------------------------------------------------------------------


def test_skills_filter_config_shape(tmp_hermes_home):
    cfg = HermesConfig.resolve()
    cfg.skills_disabled = ["arxiv"]
    cfg.skills_platform_disabled = {"telegram": ["obsidian"]}
    assert cfg.skills_filter_config() == {
        "disabled": ["arxiv"],
        "platform_disabled": {"telegram": ["obsidian"]},
    }


# ---------------------------------------------------------------------------
# Agent runtime (agent.py) — the path the issue's repro exercises
# ---------------------------------------------------------------------------


def test_agent_library_excludes_disabled_skill(tmp_hermes_home):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="alpha")
    _write_skill(skills_dir, name="beta")

    cfg = HermesConfig.resolve()
    cfg.skills_disabled = ["beta"]

    graph = create_hermes_agent(cfg, model=_stub_model())
    names = [s.name for s in graph.langstage_hermes_library.list()]
    assert "alpha" in names  # a non-disabled skill is unaffected
    assert "beta" not in names  # the disabled skill is not loaded into the agent


def test_agent_library_excludes_platform_disabled_skill(tmp_hermes_home, monkeypatch):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="alpha")
    _write_skill(skills_dir, name="beta")
    monkeypatch.setenv("HERMES_PLATFORM", "telegram")

    cfg = HermesConfig.resolve()
    cfg.skills_platform_disabled = {"telegram": ["alpha"]}

    graph = create_hermes_agent(cfg, model=_stub_model())
    names = [s.name for s in graph.langstage_hermes_library.list()]
    assert "alpha" not in names  # disabled for the active session platform
    assert "beta" in names


# ---------------------------------------------------------------------------
# CLI `_skill_library` (cli.py) — backs `skills list` / `skills show`
# ---------------------------------------------------------------------------


def test_cli_skill_library_excludes_disabled(tmp_hermes_home, monkeypatch):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="alpha")
    _write_skill(skills_dir, name="beta")
    monkeypatch.setenv("LANGSTAGE_HERMES_SKILLS_DISABLED", "beta")

    from langstage_hermes.cli import _skill_library

    lib = _skill_library(with_audit=False)
    names = [s.name for s in lib.list()]
    assert "alpha" in names
    assert "beta" not in names
    # `skills show <disabled>` (get→list) also stops resolving it.
    assert lib.get("beta") is None


# ---------------------------------------------------------------------------
# Module-level default library (skills/tools.py) — the `skills_list` tool
# ---------------------------------------------------------------------------


def test_default_library_tool_excludes_disabled(tmp_hermes_home, monkeypatch):
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="alpha")
    _write_skill(skills_dir, name="beta")
    monkeypatch.setenv("LANGSTAGE_HERMES_SKILLS_DISABLED", "beta")

    from langstage_hermes.skills import tools as skill_tools

    names = [s.name for s in skill_tools._default_library().list()]
    assert "alpha" in names
    assert "beta" not in names
