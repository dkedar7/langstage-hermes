"""``todo`` toolset — alias of deepagents' ``write_todos`` tool.

Hermes's planning tool is named ``todo``; ``deepagents`` ships the same
capability under ``write_todos`` (with the matching ``TodoListMiddleware``
managing state). This module thinly re-exports both under the Hermes name so
prompts that say "use the todo tool" land on the right callable.

The middleware itself owns the state — this module just produces the tool
object the agent factory hands to ``create_agent``. Import is lazy so the
toolset enumeration is usable without the full deepagents stack installed.
"""

from __future__ import annotations

from typing import Any


def make_todo_tool() -> Any:
    """Return deepagents' ``write_todos`` tool, renamed to ``todo``.

    The renamed copy preserves the original handler + arg schema; only the
    user-facing ``.name`` differs. See :func:`make_todo_middleware` if you
    also need the middleware that holds the todo list across turns.
    """
    try:
        from deepagents.middleware.todo import write_todos  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - guarded for headless test envs
        raise RuntimeError(
            "make_todo_tool() requires deepagents. Install `deepagents>=0.6` "
            "or add it to your project dependencies."
        ) from exc

    return _rename_tool(write_todos, "todo")


def make_todo_middleware() -> Any:
    """Return a fresh ``TodoListMiddleware`` instance.

    Exposed as a factory rather than a module-level singleton so each
    compiled graph gets its own middleware (avoids state bleed between
    parallel agents in the same process).
    """
    try:
        from deepagents.middleware.todo import TodoListMiddleware  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - guarded for headless test envs
        raise RuntimeError(
            "make_todo_middleware() requires deepagents. Install "
            "`deepagents>=0.6` or add it to your project dependencies."
        ) from exc

    return TodoListMiddleware()


def _rename_tool(tool: Any, new_name: str) -> Any:
    """Return a copy of ``tool`` with ``.name`` set to ``new_name``.

    Mirrors the helper in :mod:`deepagent_hermes.tools.file` (kept duplicated
    rather than imported to avoid creating a cross-module import that would
    drag deepagents into the file module just because todo needs it).
    """
    if hasattr(tool, "model_copy"):
        return tool.model_copy(update={"name": new_name})
    if hasattr(tool, "copy"):
        return tool.copy(update={"name": new_name})  # type: ignore[attr-defined]
    tool.name = new_name
    return tool


__all__ = ["make_todo_middleware", "make_todo_tool"]
