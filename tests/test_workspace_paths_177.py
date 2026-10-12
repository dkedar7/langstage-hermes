"""Files the agent writes land where the user expects (gh #177).

``create_hermes_agent`` roots the file tools at ``FilesystemBackend(root_dir=ws,
virtual_mode=True)``, where ``/`` is the workspace. The system prompt only gave the
model ``Working directory: <host path>``, so a model that wrote to
``<host path>/answer.txt`` got the whole host path re-rooted under the workspace:
the file landed in ``<ws>/<ws-host-path>/answer.txt`` and the reply named a path
that didn't exist. On Windows, deepagents refused the drive-letter path outright.

These tests drive the real file tools through ``create_hermes_agent`` with a fake
model that makes one ``write_file`` call, so no API key is needed.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from langstage_hermes.agent import create_hermes_agent
from langstage_hermes.workspace_paths import to_virtual_path


class _FakeWriter(BaseChatModel):
    """Calls ``write_file(target, "42")`` once, then answers ``done``. Records each system prompt."""

    target: str
    system_prompts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:  # pragma: no cover - identity only
        return "fake-writer"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.system_prompts.extend(m.text for m in messages if isinstance(m, SystemMessage))
        if any(isinstance(m, ToolMessage) for m in messages):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])
        call = {"name": "write_file", "args": {"file_path": self.target, "content": "42"}, "id": "call-1", "type": "tool_call"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


_PROMPT = {"messages": [{"role": "user", "content": "Write the result of 17 + 25 to a file named answer.txt."}]}


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _tool_message(out: dict[str, Any]) -> ToolMessage:
    return next(m for m in out["messages"] if isinstance(m, ToolMessage))


@pytest.fixture
def ws(tmp_hermes_home: Path, tmp_workspace: Path) -> Path:
    return tmp_workspace.resolve()


# ── the issue's repro, through the agent's real file tools ───────────


@pytest.mark.parametrize("form", ["native", "forward-slashes"])
def test_host_path_write_lands_at_workspace_top_level(ws: Path, form: str) -> None:
    host_path = ws / "answer.txt"
    target = str(host_path) if form == "native" else host_path.as_posix()
    agent = create_hermes_agent(workspace=ws, model=_FakeWriter(target=target))

    tool = _tool_message(agent.invoke(_PROMPT))

    assert tool.status != "error", tool.content
    assert (ws / "answer.txt").read_text(encoding="utf-8") == "42"
    assert _files(ws) == ["answer.txt"], "the write must not land in a nested copy of the workspace path"


async def test_host_path_write_lands_at_workspace_top_level_async(ws: Path) -> None:
    """The chat path drives the graph asynchronously (AG-UI), so cover awrap_tool_call too."""
    agent = create_hermes_agent(workspace=ws, model=_FakeWriter(target=str(ws / "answer.txt")))

    tool = _tool_message(await agent.ainvoke(_PROMPT))

    assert tool.status != "error", tool.content
    assert _files(ws) == ["answer.txt"]


def test_virtual_path_still_works(ws: Path) -> None:
    agent = create_hermes_agent(workspace=ws, model=_FakeWriter(target="/answer.txt"))

    tool = _tool_message(agent.invoke(_PROMPT))

    assert tool.status != "error", tool.content
    assert _files(ws) == ["answer.txt"]


@pytest.mark.parametrize(
    "make_target",
    [lambda ws: "/../outside.txt", lambda ws: str(ws / ".." / "outside.txt")],
    ids=["virtual-traversal", "host-traversal"],
)
def test_path_outside_workspace_is_still_refused(ws: Path, make_target: Any) -> None:
    agent = create_hermes_agent(workspace=ws, model=_FakeWriter(target=make_target(ws)))

    tool = _tool_message(agent.invoke(_PROMPT))

    assert tool.status == "error", tool.content
    assert not (ws.parent / "outside.txt").exists()
    assert _files(ws) == []


def test_host_path_outside_workspace_never_leaves_it(ws: Path) -> None:
    """A sibling's host path is not mapped: it is refused (Windows) or kept inside the workspace."""
    outside = ws.parent / "outside.txt"
    agent = create_hermes_agent(workspace=ws, model=_FakeWriter(target=str(outside)))

    agent.invoke(_PROMPT)

    assert not outside.exists()


# ── what the prompt tells the model ──────────────────────────────────


def test_prompt_says_slash_is_the_workspace(ws: Path) -> None:
    model = _FakeWriter(target="/answer.txt")
    create_hermes_agent(workspace=ws, model=model).invoke(_PROMPT)

    prompt = model.system_prompts[0]
    assert f"Working directory: {ws}" in prompt
    assert f"their `/` is {ws}, so pass `/answer.txt` for {ws / 'answer.txt'}" in prompt
    assert f"give the full path on their machine ({ws / 'answer.txt'})" in prompt


def test_prompt_has_no_slash_note_for_a_caller_backend(ws: Path) -> None:
    """A caller-supplied backend owns its own path semantics; hermes must not describe them."""
    from deepagents.backends.filesystem import FilesystemBackend

    model = _FakeWriter(target="/../refused.txt")  # write nothing through a real-path backend
    backend = FilesystemBackend(root_dir=str(ws), virtual_mode=False)
    create_hermes_agent(workspace=ws, model=model, backend=backend).invoke(_PROMPT)

    prompt = model.system_prompts[0]
    assert f"Working directory: {ws}" in prompt
    assert "their `/` is" not in prompt


# ── the mapping itself, for both path flavours on any OS ─────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/home/me/proj/answer.txt", "/answer.txt"),
        ("/home/me/proj/src/app.py", "/src/app.py"),
        ("/home/me/proj", "/"),
        ("/home/me/proj/", "/"),
        ("/answer.txt", "/answer.txt"),  # already virtual
        ("answer.txt", "answer.txt"),  # relative
        ("/home/me/proj-2/answer.txt", "/home/me/proj-2/answer.txt"),  # sibling sharing a prefix
        ("/etc/passwd", "/etc/passwd"),  # outside: left for the sandbox to keep inside
        ("/home/me/proj/../secret", "/../secret"),  # `..` survives, so it is still refused
    ],
)
def test_to_virtual_path_posix(path: str, expected: str) -> None:
    assert to_virtual_path(path, PurePosixPath("/home/me/proj")) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"C:\Users\me\proj\answer.txt", "/answer.txt"),
        ("C:/Users/me/proj/answer.txt", "/answer.txt"),
        ("/C:/Users/me/proj/answer.txt", "/answer.txt"),
        (r"c:\users\ME\proj\sub\answer.txt", "/sub/answer.txt"),  # Windows paths ignore case
        ("/Users/me/proj/answer.txt", "/answer.txt"),  # driveless: on the workspace's drive
        (r"C:\Users\me\proj", "/"),
        ("/answer.txt", "/answer.txt"),
        (r"C:\Users\me\proj-2\answer.txt", r"C:\Users\me\proj-2\answer.txt"),
        (r"D:\Users\me\proj\answer.txt", r"D:\Users\me\proj\answer.txt"),  # other drive
        (r"C:\Users\me\proj\..\secret", "/../secret"),
    ],
)
def test_to_virtual_path_windows(path: str, expected: str) -> None:
    assert to_virtual_path(path, PureWindowsPath(r"C:\Users\me\proj")) == expected
