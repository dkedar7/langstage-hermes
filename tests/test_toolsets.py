"""Tests for the 33-toolset enumeration (SPEC §11)."""

from __future__ import annotations

from dataclasses import dataclass

from deepagent_hermes.tools.registry import HermesToolRegistry
from deepagent_hermes.tools.toolsets import (
    IMPLEMENTED_TOOLSETS,
    TOOLSETS,
    all_toolset_names,
    is_implemented,
    register_implemented_tools,
    resolve_enabled,
    tools_for,
)


@dataclass
class _FakeTool:
    name: str


# ── Enumeration ──────────────────────────────────────────────────────


def test_toolsets_has_exactly_33_entries() -> None:
    """SPEC §11 mandates 33 toolsets — guard against accidental drift."""
    assert len(TOOLSETS) == 33


def test_implemented_toolsets_is_a_subset() -> None:
    """Every name in ``IMPLEMENTED_TOOLSETS`` must exist in ``TOOLSETS``."""
    assert IMPLEMENTED_TOOLSETS.issubset(TOOLSETS.keys())


def test_implemented_set_matches_spec() -> None:
    """v0.1.0 ships the seven toolsets the spec lists; no more, no less."""
    expected = {"file", "todo", "clarify", "skills", "memory", "session_search", "terminal"}
    assert IMPLEMENTED_TOOLSETS == expected


def test_known_leaf_toolsets_are_present() -> None:
    """Spot-check that the Hermes leaf names port through."""
    for name in [
        "web",
        "search",
        "x_search",
        "vision",
        "video",
        "image_gen",
        "video_gen",
        "computer_use",
        "terminal",
        "file",
        "moa",
        "todo",
        "skills",
        "memory",
        "context_engine",
        "session_search",
        "browser",
        "cronjob",
        "messaging",
        "tts",
        "clarify",
        "code_execution",
        "delegation",
        "homeassistant",
        "kanban",
        "discord",
        "discord_admin",
        "yuanbao",
        "feishu_doc",
        "feishu_drive",
        "spotify",
        "debugging",
        "safe",
    ]:
        assert name in TOOLSETS, f"{name} missing from TOOLSETS"


def test_file_toolset_has_hermes_names() -> None:
    """File toolset must expose the Hermes-name surface, not deepagents's."""
    assert tools_for("file") == ["read_file", "write_file", "patch", "search_files"]


def test_terminal_toolset_has_terminal_and_process() -> None:
    assert tools_for("terminal") == ["terminal", "process"]


def test_skills_toolset_has_three_tools() -> None:
    assert sorted(tools_for("skills")) == sorted(["skills_list", "skill_view", "skill_manage"])


def test_all_toolset_names_sorted() -> None:
    names = all_toolset_names()
    assert names == sorted(names)
    assert len(names) == 33


# ── Helpers ──────────────────────────────────────────────────────────


def test_is_implemented_true_for_v1_set() -> None:
    for name in IMPLEMENTED_TOOLSETS:
        assert is_implemented(name) is True


def test_is_implemented_false_for_deferred() -> None:
    # Pick a few that are explicitly out-of-scope for v0.1.0.
    for name in ("discord", "spotify", "yuanbao", "homeassistant"):
        assert is_implemented(name) is False


def test_resolve_enabled_drops_disabled() -> None:
    enabled = resolve_enabled(disabled_toolsets=["web", "terminal"])
    assert "web" not in enabled
    assert "terminal" not in enabled
    assert "file" in enabled
    assert len(enabled) == 33 - 2


def test_resolve_enabled_default_is_full_set() -> None:
    assert resolve_enabled() == set(TOOLSETS.keys())


def test_tools_for_unknown_returns_empty() -> None:
    assert tools_for("does-not-exist") == []


# ── register_implemented_tools wiring ────────────────────────────────


def test_register_implemented_tools_fans_out_by_toolset() -> None:
    """``register_implemented_tools`` must place each callable under its canonical toolset."""
    reg = HermesToolRegistry()

    file_tools = [_FakeTool("read_file"), _FakeTool("write_file")]
    todo_tool = _FakeTool("todo")
    clarify_tool = _FakeTool("clarify")
    skills_tools = [_FakeTool("skill_view"), _FakeTool("skill_manage")]
    memory_tool = _FakeTool("memory")
    session_search_tool = _FakeTool("session_search")
    terminal_tools = [_FakeTool("terminal"), _FakeTool("process")]

    register_implemented_tools(
        reg,
        file_tools=file_tools,
        todo_tool=todo_tool,
        clarify_tool=clarify_tool,
        skills_tools=skills_tools,
        memory_tool=memory_tool,
        session_search_tool=session_search_tool,
        terminal_tools=terminal_tools,
    )

    grouped = reg.list_toolsets()
    assert sorted(grouped["file"]) == ["read_file", "write_file"]
    assert grouped["todo"] == ["todo"]
    assert grouped["clarify"] == ["clarify"]
    assert sorted(grouped["skills"]) == ["skill_manage", "skill_view"]
    assert grouped["memory"] == ["memory"]
    assert grouped["session_search"] == ["session_search"]
    assert sorted(grouped["terminal"]) == ["process", "terminal"]


def test_register_implemented_tools_partial() -> None:
    """All args optional — caller can wire subsystems incrementally."""
    reg = HermesToolRegistry()
    register_implemented_tools(reg, todo_tool=_FakeTool("todo"))
    grouped = reg.list_toolsets()
    assert grouped == {"todo": ["todo"]}
