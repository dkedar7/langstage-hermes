"""``WorkspacePathMiddleware``: host paths inside the workspace reach the right file (gh #177).

hermes roots the file tools at ``FilesystemBackend(root_dir=<workspace>,
virtual_mode=True)``, so their ``/`` is the workspace. A model that passes the
workspace's host path instead (``/home/me/proj/answer.txt``, or
``C:\\Users\\me\\proj\\answer.txt`` on Windows) used to get it re-rooted under
the workspace: the file landed in ``<ws>/home/me/proj/answer.txt`` and the reply
named a path where it didn't exist. On Windows, deepagents refuses a drive-letter
path before the backend sees it, which is why the mapping happens here, on the tool
call, rather than in the backend.

This middleware rewrites the ``file_path`` / ``path`` argument of a file-tool call
when it is a host path inside the workspace, to the same file's ``/``-rooted path.
Anything else passes through untouched, so deepagents' own checks still refuse
``..`` and keep every path inside the workspace.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable
from pathlib import PurePath, PureWindowsPath
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

_PATH_ARGS: tuple[str, ...] = ("file_path", "path")
"""Argument names deepagents' file tools take a path under."""

_SLASH_BEFORE_DRIVE = re.compile(r"^[\\/]+(?=[A-Za-z]:)")


def to_virtual_path(path: str, root: PurePath) -> str:
    """Return ``path`` as the file tools' ``/``-rooted path if it is a host path inside ``root``.

    ``root`` is the workspace's absolute host path; its type picks the path rules
    (a ``PureWindowsPath`` root accepts ``C:\\...``, ``C:/...``, ``/C:/...`` and, on
    the workspace's drive, ``\\Users\\...``). A path that is not absolute, or not
    inside ``root``, is returned unchanged. ``..`` is never collapsed, so a host path
    that climbs out keeps its ``..`` and is still refused downstream.
    """
    if isinstance(root, PureWindowsPath):
        candidate = type(root)(_SLASH_BEFORE_DRIVE.sub("", path))
        if candidate.root and not candidate.drive:
            # Rooted but driveless ("\Users\me\proj"): Windows reads it on the
            # current drive, which for these tools is the workspace's.
            candidate = type(root)(root.drive + str(candidate))
    else:
        candidate = type(root)(path)
    if not candidate.is_absolute():
        return path
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return path
    rel = relative.as_posix()
    return "/" if rel == "." else f"/{rel}"


class WorkspacePathMiddleware(AgentMiddleware):
    """Rewrite host paths inside the workspace to ``/``-rooted paths on file-tool calls.

    Args:
        workspace_root: The workspace's absolute host path, the backend's ``root_dir``.
        tool_names: The file tools to normalize (the names ``FilesystemMiddleware``
            registers). Other tools' calls pass through unchanged.
    """

    def __init__(self, *, workspace_root: PurePath, tool_names: Iterable[str]) -> None:
        self.workspace_root = workspace_root
        self.tool_names = frozenset(tool_names)

    def _rewrite(self, request: Any) -> Any:
        call = request.tool_call
        args = call.get("args")
        if call.get("name") not in self.tool_names or not isinstance(args, dict):
            return request
        mapped = {
            key: to_virtual_path(value, self.workspace_root)
            for key, value in args.items()
            if key in _PATH_ARGS and isinstance(value, str) and value
        }
        if all(args[key] == value for key, value in mapped.items()):
            return request
        return request.override(tool_call={**call, "args": {**args, **mapped}})

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._rewrite(request))

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await handler(self._rewrite(request))


__all__ = ["WorkspacePathMiddleware", "to_virtual_path"]
