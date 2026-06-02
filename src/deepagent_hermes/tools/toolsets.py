"""33-toolset enumeration for ``deepagent-hermes`` (SPEC §11).

Mirrors the toolset taxonomy from Hermes's ``toolsets.py`` so user-facing
``hermes tools`` / config behavior is identical. In v1 only a subset of these
toolsets has concrete tool implementations — the rest are entry points listed
in :data:`TOOLSETS` so the enum exists, slash commands can reference them,
and the gating wiring (``[agent.disabled_toolsets]``) round-trips correctly
through ``HermesConfig``.

The 33 toolsets here are the LEAF set Hermes ships — the additional
``hermes-<platform>`` composite toolsets in the upstream file (``hermes-cli``,
``hermes-telegram``, …) belong to the messaging gateway, which is explicitly
out of scope per SPEC §0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagent_hermes.tools.registry import HermesToolRegistry


# ── 33-toolset enumeration ────────────────────────────────────────────
#
# Each key is the canonical toolset name; each value is the list of tool
# names that compose it. Tool names match Hermes's wire shape so skill
# definitions (`requires_tools`, `fallback_for_tools`) port verbatim.

TOOLSETS: dict[str, list[str]] = {
    # ── Web / search ──────────────────────────────────────────────────
    "web": ["web_search", "web_extract"],
    "search": ["web_search"],
    "x_search": ["x_search"],
    # ── Vision / generative media ────────────────────────────────────
    "vision": ["vision_analyze"],
    "video": ["video_analyze"],
    "image_gen": ["image_generate"],
    "video_gen": ["video_generate"],
    # ── OS / process / files ─────────────────────────────────────────
    "computer_use": ["computer_use"],
    "terminal": ["terminal", "process"],
    "file": ["read_file", "write_file", "patch", "search_files"],
    # ── Reasoning / planning ─────────────────────────────────────────
    "moa": ["mixture_of_agents"],
    "todo": ["todo"],
    # ── Skills / memory / search ─────────────────────────────────────
    "skills": ["skills_list", "skill_view", "skill_manage"],
    "memory": ["memory"],
    "context_engine": [],
    "session_search": ["session_search"],
    # ── Browser automation ───────────────────────────────────────────
    "browser": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back",
        "browser_press", "browser_get_images",
        "browser_vision", "browser_console", "browser_cdp", "browser_dialog",
    ],
    # ── Scheduling / interaction ─────────────────────────────────────
    "cronjob": ["cronjob"],
    "messaging": ["send_message"],
    "tts": ["text_to_speech"],
    "clarify": ["clarify"],
    "code_execution": ["execute_code"],
    "delegation": ["delegate_task"],
    # ── External integrations ────────────────────────────────────────
    "homeassistant": [
        "ha_list_entities", "ha_get_state",
        "ha_list_services", "ha_call_service",
    ],
    "kanban": [
        "kanban_show", "kanban_list", "kanban_complete", "kanban_block",
        "kanban_heartbeat", "kanban_comment",
        "kanban_create", "kanban_link", "kanban_unblock",
    ],
    "discord": ["discord"],
    "discord_admin": ["discord_admin"],
    "yuanbao": [
        "yb_query_group_info", "yb_query_group_members",
        "yb_send_dm", "yb_search_sticker", "yb_send_sticker",
    ],
    "feishu_doc": ["feishu_doc_read"],
    "feishu_drive": [
        "feishu_drive_list_comments", "feishu_drive_list_comment_replies",
        "feishu_drive_reply_comment", "feishu_drive_add_comment",
    ],
    "spotify": [
        "spotify_playback", "spotify_devices", "spotify_queue",
        "spotify_search", "spotify_playlists", "spotify_albums",
        "spotify_library",
    ],
    # ── Scenario composites (Hermes parity) ──────────────────────────
    # These don't add new tool names of their own beyond what they
    # compose, but Hermes ships them as named toolsets and config
    # references rely on them, so they count toward the 33.
    "debugging": ["terminal", "process"],
    "safe": [],
}

# Sanity check at import time — protects against accidentally drifting from
# the SPEC count when editing this file.
assert len(TOOLSETS) == 33, (
    f"TOOLSETS must contain exactly 33 entries (per SPEC §11); got {len(TOOLSETS)}"
)


# Toolsets with at least one concrete implementation in v0.1.0. The rest are
# enum-only — they exist so config (``[agent].disabled_toolsets``) and slash
# commands can reference them but won't actually surface any tools until a
# v0.2+ release (or a plugin) registers their handlers.
IMPLEMENTED_TOOLSETS: set[str] = {
    "file",
    "todo",
    "clarify",
    "skills",
    "memory",
    "session_search",
    "terminal",
}


# ── Helpers ───────────────────────────────────────────────────────────


def all_toolset_names() -> list[str]:
    """Return every toolset name, sorted."""
    return sorted(TOOLSETS.keys())


def tools_for(toolset: str) -> list[str]:
    """Return the tool names declared for ``toolset``, or ``[]`` if unknown."""
    return list(TOOLSETS.get(toolset, []))


def is_implemented(toolset: str) -> bool:
    """Return whether v0.1.0 ships a concrete implementation for ``toolset``."""
    return toolset in IMPLEMENTED_TOOLSETS


def resolve_enabled(
    *,
    disabled_toolsets: list[str] | tuple[str, ...] | set[str] | None = None,
    platform: str = "cli",
) -> set[str]:
    """Compute the set of toolset names that should be active for a session.

    Implements the v1 slice of the SPEC §11 filtering: start from every
    toolset, drop anything in ``disabled_toolsets``. The ``platform`` argument
    is accepted for parity with Hermes's per-platform overrides (e.g. cron
    strips ``clarify`` / ``cronjob`` / ``messaging``); platform gating proper
    lives in the cron/gateway subsystems rather than here.
    """
    disabled = set(disabled_toolsets or ())
    return {name for name in TOOLSETS if name not in disabled}


def register_implemented_tools(
    registry: "HermesToolRegistry",
    *,
    file_tools: list | None = None,
    todo_tool=None,
    clarify_tool=None,
    skills_tools: list | None = None,
    memory_tool=None,
    session_search_tool=None,
    terminal_tools: list | None = None,
) -> None:
    """Register the v0.1.0 implemented tools into ``registry``.

    All arguments are optional so callers can wire up subsystems incrementally
    (the file toolset depends on a ``FilesystemBackend`` instance the agent
    factory provides; the session_search tool depends on the SQLite store;
    etc.). Each argument is the already-constructed tool callable / list of
    callables — this function only fans them out into the registry under the
    right toolset names, it does not build them.

    Keeping the wiring centralized here means the toolset → tool mapping has
    exactly one source of truth (the ``TOOLSETS`` dict above), and callers
    don't have to memorize which toolset string each tool belongs to.
    """
    if file_tools:
        for t in file_tools:
            registry.register(t, toolset="file")
    if todo_tool is not None:
        registry.register(todo_tool, toolset="todo")
    if clarify_tool is not None:
        registry.register(clarify_tool, toolset="clarify")
    if skills_tools:
        for t in skills_tools:
            registry.register(t, toolset="skills")
    if memory_tool is not None:
        registry.register(memory_tool, toolset="memory")
    if session_search_tool is not None:
        registry.register(session_search_tool, toolset="session_search")
    if terminal_tools:
        for t in terminal_tools:
            registry.register(t, toolset="terminal")


__all__ = [
    "TOOLSETS",
    "IMPLEMENTED_TOOLSETS",
    "all_toolset_names",
    "tools_for",
    "is_implemented",
    "resolve_enabled",
    "register_implemented_tools",
]
