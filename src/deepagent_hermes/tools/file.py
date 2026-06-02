"""``file`` toolset — Hermes-named aliases for deepagents' filesystem tools.

Hermes exposes four filesystem operations: ``read_file``, ``write_file``,
``patch`` (a fuzzy diff-like edit), and ``search_files`` (combined glob + grep).
``deepagents``'s ``FilesystemMiddleware`` already ships matching implementations
under slightly different names (``read_file``, ``write_file``, ``edit_file``,
``glob`` / ``grep``). This module re-binds those into the Hermes name set so
prompts / skills / docs that reference Hermes names port without change.

The middleware itself owns the actual ``FilesystemBackend``; this module just
returns a list of tool objects the agent factory can hand to ``create_agent``.
"""

from __future__ import annotations

from typing import Any


def make_file_tools(backend: Any) -> list[Any]:
    """Return Hermes-named filesystem tools bound to ``backend``.

    Args:
        backend: A ``deepagents.FilesystemBackend`` (or duck-typed equivalent).
            Constructed by the agent factory before middleware assembly.

    Returns:
        A list of LangChain ``BaseTool`` instances exposing ``read_file``,
        ``write_file``, ``patch``, and ``search_files``. The list ordering
        is stable for deterministic registry behavior.

    The import is lazy so importing this module doesn't drag in the full
    deepagents stack — necessary for the tool-registry / toolset tests to run
    in a thin venv.
    """
    try:
        from deepagents.middleware.filesystem import (  # type: ignore[import-not-found]
            FilesystemMiddleware,
        )
    except ImportError as exc:  # pragma: no cover - guarded for headless test envs
        raise RuntimeError(
            "make_file_tools() requires deepagents. Install `deepagents>=0.6` "
            "or add it to your project dependencies."
        ) from exc

    # deepagents exposes per-operation tool factories on the middleware class.
    # We instantiate the middleware with the supplied backend, then pull each
    # native tool out and (where the name differs) rebind under the Hermes name.
    mw = FilesystemMiddleware(backend=backend)
    native = {t.name: t for t in mw.tools}

    out: list[Any] = []

    # read_file: name matches — pass through directly.
    if "read_file" in native:
        out.append(native["read_file"])

    # write_file: name matches — pass through directly.
    if "write_file" in native:
        out.append(native["write_file"])

    # patch: deepagents calls this `edit_file`. Rebind via a thin wrapper so
    # the model sees the Hermes name + Hermes docstring.
    if "edit_file" in native:
        out.append(_rename_tool(native["edit_file"], "patch"))

    # search_files: deepagents splits this into `glob` and `grep`. For v1 we
    # alias `grep` (content search) under the Hermes name since that's the
    # 80% use; a fuller implementation would merge both into one tool with a
    # mode flag, but that costs us prompt-cache stability on every search
    # call. Defer to v0.2+.
    if "grep" in native:
        out.append(_rename_tool(native["grep"], "search_files"))

    return out


def _rename_tool(tool: Any, new_name: str) -> Any:
    """Return a copy of ``tool`` with ``.name`` set to ``new_name``.

    LangChain ``BaseTool`` instances are pydantic models; using
    ``model_copy`` keeps the original handler / arg schema / coroutine
    bindings intact while only swapping the surface name.
    """
    if hasattr(tool, "model_copy"):
        # pydantic v2 path — preserves schema + handler.
        return tool.model_copy(update={"name": new_name})
    if hasattr(tool, "copy"):
        # pydantic v1 fallback.
        return tool.copy(update={"name": new_name})  # type: ignore[attr-defined]
    # Last-resort: mutate in place. Acceptable because the deepagents
    # middleware constructs a fresh tool list per instance.
    tool.name = new_name
    return tool


__all__ = ["make_file_tools"]
