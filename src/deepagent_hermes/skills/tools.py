"""``skills_list``, ``skill_view``, ``skill_manage`` tool implementations.

These are exposed as a small factory because each tool needs to close over
a :class:`SkillLibrary` instance — building them at module-import time would
hard-wire a particular library.

The tools return :class:`Command` instances when they need to update agent
state (notably ``skill_view`` which records the loaded body and
``skill_manage`` which resets ``iters_since_skill`` per SPEC §10.5).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any, Literal

import frontmatter
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command

from deepagent_hermes.skills.library import SkillLibrary
from deepagent_hermes.skills.prompt import clear_prompt_cache
from deepagent_hermes.skills.validator import validate as validate_frontmatter

logger = logging.getLogger(__name__)

__all__ = ["make_skill_tools", "skill_manage", "skill_view", "skills_list"]


# ---------------------------------------------------------------------------
# Module-level default-library tools (importable directly for tests/docs)
# ---------------------------------------------------------------------------


def _default_library() -> SkillLibrary:
    """Lazily build a default library so importing this module is cheap."""
    return SkillLibrary()


@tool
def skills_list(query: str = "", category: str = "") -> str:
    """List available skills as a markdown table.

    Args:
        query: Substring filter — matches against name OR description (case-insensitive).
        category: Restrict to a single category.

    Returns the table as markdown; the agent can scan it and pick a skill to
    ``skill_view``.
    """
    return _render_list(_default_library(), query=query, category=category)


@tool
def skill_view(
    name: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Load the full SKILL.md body for ``name`` and pin it into session context.

    The body is appended to ``state.loaded_skill_bodies[name]`` so the next
    ``wrap_model_call`` injects it into the system prompt (see
    ``SkillLoaderMiddleware``). Returns a ``Command`` so the state update is
    applied atomically with the tool result.
    """
    return _skill_view_impl(_default_library(), name=name, tool_call_id=tool_call_id)


@tool
def skill_manage(
    action: Literal["create", "patch", "write_file", "delete", "pin", "unpin"],
    name: str,
    description: str = "",
    body: str = "",
    category: str = "",
    old_str: str = "",
    new_str: str = "",
    frontmatter_data: dict[str, Any] | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """Curate the skill library.

    Actions:
      - ``create``: create a new skill. Requires ``description`` + ``body``;
        optional ``category``.
      - ``patch``: in-place edit. Requires ``old_str`` + ``new_str``.
      - ``write_file``: overwrite the SKILL.md whole. Requires ``frontmatter_data``
        + ``body``.
      - ``delete``: archive (not rm) the skill.
      - ``pin`` / ``unpin``: toggle ``metadata.hermes.pinned``.

    On any successful action, resets ``state["iters_since_skill"]`` to 0
    (SPEC §10.5).
    """
    return _skill_manage_impl(
        _default_library(),
        action=action,
        name=name,
        description=description,
        body=body,
        category=category,
        old_str=old_str,
        new_str=new_str,
        frontmatter_data=frontmatter_data,
        tool_call_id=tool_call_id,
    )


# ---------------------------------------------------------------------------
# Factory that closes over an explicit library
# ---------------------------------------------------------------------------


def make_skill_tools(library: SkillLibrary) -> list[BaseTool]:
    """Build the three tools bound to a specific library.

    Use this in the agent factory so the same library instance backs both the
    middleware (which renders the index) and the tools (which mutate it).
    """

    @tool("skills_list")
    def _skills_list(query: str = "", category: str = "") -> str:
        """List available skills as a markdown table.

        Args:
            query: Substring filter — matches against name OR description.
            category: Restrict to a single category.
        """
        return _render_list(library, query=query, category=category)

    @tool("skill_view")
    def _skill_view(
        name: str,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Load the full SKILL.md body for ``name`` and pin it into session context.

        Args:
            name: The skill name (as reported by ``skills_list``).
        """
        return _skill_view_impl(library, name=name, tool_call_id=tool_call_id)

    @tool("skill_manage")
    def _skill_manage(
        action: Literal["create", "patch", "write_file", "delete", "pin", "unpin"],
        name: str,
        description: str = "",
        body: str = "",
        category: str = "",
        old_str: str = "",
        new_str: str = "",
        frontmatter_data: dict[str, Any] | None = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Curate the skill library (create/patch/write_file/delete/pin/unpin)."""
        return _skill_manage_impl(
            library,
            action=action,
            name=name,
            description=description,
            body=body,
            category=category,
            old_str=old_str,
            new_str=new_str,
            frontmatter_data=frontmatter_data,
            tool_call_id=tool_call_id,
        )

    return [_skills_list, _skill_view, _skill_manage]


# ---------------------------------------------------------------------------
# Implementations (no decorator coupling)
# ---------------------------------------------------------------------------


def _render_list(library: SkillLibrary, *, query: str, category: str) -> str:
    """Filter + render skills as a markdown table."""
    skills = library.list()

    if category:
        skills = [s for s in skills if (s.category or "") == category]

    if query:
        q = query.lower()
        skills = [s for s in skills if q in s.name.lower() or q in s.description.lower()]

    if not skills:
        return "_No skills match._"

    lines = [
        "| name | category | description |",
        "| --- | --- | --- |",
    ]
    for s in skills:
        desc = s.description.replace("|", "\\|").replace("\n", " ").strip()
        lines.append(f"| {s.name} | {s.category or ''} | {desc} |")
    return "\n".join(lines)


def _skill_view_impl(library: SkillLibrary, *, name: str, tool_call_id: str) -> Command:
    """Return a Command that loads the named skill body into state."""
    skill = library.get(name)
    if skill is None:
        payload = json.dumps({"success": False, "error": f"skill {name!r} not found"})
        return _command_with_tool_message(payload, tool_call_id=tool_call_id)

    body = skill.body
    payload = json.dumps(
        {
            "success": True,
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "path": str(skill.path),
            "content": body,
        },
        ensure_ascii=False,
    )

    # Append to loaded_skill_bodies; track active_skills append.
    update: dict[str, Any] = {
        "loaded_skill_bodies": {skill.name: body},
        "active_skills": [skill.name],
    }
    return _command_with_tool_message(payload, tool_call_id=tool_call_id, update=update)


def _skill_manage_impl(
    library: SkillLibrary,
    *,
    action: str,
    name: str,
    description: str,
    body: str,
    category: str,
    old_str: str,
    new_str: str,
    frontmatter_data: dict[str, Any] | None,
    tool_call_id: str,
) -> Command:
    """Implement the five mutation actions and return a state-resetting Command."""
    try:
        if action == "create":
            path = _action_create(library, name=name, description=description, body=body, category=category)
            result = {"success": True, "action": action, "name": name, "path": str(path)}
        elif action == "patch":
            path = _action_patch(library, name=name, old_str=old_str, new_str=new_str)
            result = {"success": True, "action": action, "name": name, "path": str(path)}
        elif action == "write_file":
            path = _action_write_file(library, name=name, frontmatter_data=frontmatter_data or {}, body=body)
            result = {"success": True, "action": action, "name": name, "path": str(path)}
        elif action == "delete":
            archived = library.delete(name)
            if not archived:
                return _command_with_tool_message(
                    json.dumps({"success": False, "error": f"skill {name!r} not found"}),
                    tool_call_id=tool_call_id,
                )
            result = {"success": True, "action": action, "name": name, "archived": True}
        elif action in ("pin", "unpin"):
            path = _action_pin(library, name=name, pinned=(action == "pin"))
            result = {"success": True, "action": action, "name": name, "path": str(path)}
        else:
            return _command_with_tool_message(
                json.dumps({"success": False, "error": f"unknown action {action!r}"}),
                tool_call_id=tool_call_id,
            )
    except Exception as exc:
        logger.exception("skill_manage action %s failed for %s", action, name)
        return _command_with_tool_message(
            json.dumps({"success": False, "action": action, "error": str(exc)}),
            tool_call_id=tool_call_id,
        )

    # Successful mutation -> invalidate prompt cache + reset reflection counter.
    clear_prompt_cache()
    update: dict[str, Any] = {"iters_since_skill": 0}
    return _command_with_tool_message(
        json.dumps(result, ensure_ascii=False),
        tool_call_id=tool_call_id,
        update=update,
    )


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def _action_create(
    library: SkillLibrary,
    *,
    name: str,
    description: str,
    body: str,
    category: str,
) -> Path:
    if not description:
        raise ValueError("create: 'description' is required")
    if not body:
        raise ValueError("create: 'body' is required")
    fm = {"name": name, "description": description}
    return library.write(name, fm, body, category=category or None)


def _action_patch(library: SkillLibrary, *, name: str, old_str: str, new_str: str) -> Path:
    if not old_str:
        raise ValueError("patch: 'old_str' is required")
    skill = library.get(name)
    if skill is None:
        raise ValueError(f"skill {name!r} not found")
    full = skill.path.read_text(encoding="utf-8")
    if old_str not in full:
        raise ValueError("patch: 'old_str' not found in SKILL.md")
    count = full.count(old_str)
    if count > 1:
        raise ValueError(f"patch: 'old_str' is ambiguous (matched {count} times) — supply more context")
    full = full.replace(old_str, new_str, 1)
    skill.path.write_text(full, encoding="utf-8")
    return skill.path


def _action_write_file(
    library: SkillLibrary,
    *,
    name: str,
    frontmatter_data: dict[str, Any],
    body: str,
) -> Path:
    if not frontmatter_data:
        raise ValueError("write_file: 'frontmatter_data' is required")
    # Use library.write — it validates + handles missing dirs.
    fm = dict(frontmatter_data)
    fm.setdefault("name", name)
    # Preserve the originating dir / category if the skill exists; else default.
    existing = library.get(name)
    category: str | None = None
    target_dir: Path | None = None
    if existing is not None:
        category = existing.category
        # Place under the same search dir
        search_dir = library._find_search_dir_for(existing.path)
        target_dir = search_dir
    return library.write(name, fm, body, category=category, target_dir=target_dir)


def _action_pin(library: SkillLibrary, *, name: str, pinned: bool) -> Path:
    skill = library.get(name)
    if skill is None:
        raise ValueError(f"skill {name!r} not found")
    meta = dict(skill.metadata)
    nested = dict(meta.get("metadata") or {})
    hermes = dict(nested.get("hermes") or {})
    if pinned:
        hermes["pinned"] = True
    else:
        hermes.pop("pinned", None)
    if hermes:
        nested["hermes"] = hermes
    elif "hermes" in nested:
        del nested["hermes"]
    if nested:
        meta["metadata"] = nested
    elif "metadata" in meta:
        del meta["metadata"]
    # Re-validate + rewrite via library.write (preserves dir/category).
    errors = validate_frontmatter(meta, parent_dir_name=skill.name)
    if errors:
        raise ValueError("pin/unpin would produce invalid frontmatter:\n- " + "\n- ".join(errors))
    post = frontmatter.Post(skill.body, **meta)
    skill.path.write_bytes(frontmatter.dumps(post).encode("utf-8"))
    return skill.path


# ---------------------------------------------------------------------------
# Command helpers
# ---------------------------------------------------------------------------


def _command_with_tool_message(
    content: str,
    *,
    tool_call_id: str,
    update: dict[str, Any] | None = None,
) -> Command:
    """Wrap a tool response in a ``Command`` so we can also patch state.

    LangGraph tool nodes resolve the ``tool_call_id`` automatically when the
    decorator-wrapped function returns a string instead of a ``Command``,
    but as soon as we want to mutate state we need to return a Command and
    attach the ToolMessage ourselves. Tests sometimes invoke these helpers
    without a real tool_call_id; in that case we omit the message and just
    return the state update (so tests can inspect it).
    """
    if not tool_call_id:
        # Tests path: no real tool_call_id wired up.
        payload: dict[str, Any] = {}
        if update:
            payload.update(update)
        payload["__content__"] = content
        return Command(update=payload)

    msg = ToolMessage(content=content, tool_call_id=tool_call_id)
    payload = {"messages": [msg]}
    if update:
        payload.update(update)
    return Command(update=payload)
