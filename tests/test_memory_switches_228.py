"""gh #228: ``memory.memory_enabled`` / ``memory.user_profile_enabled`` were ignored.

Both switches were resolved by ``HermesConfig`` and shown by ``--show-config``,
but nothing read them: the agent still bound the ``memory`` tool, put USER.md and
MEMORY.md in its system prompt and wrote to them. What they do now:

- ``memory_enabled = false``: MEMORY.md is not put in the prompt, and the
  ``memory`` tool refuses ``target="memory"`` (``read`` too).
- ``user_profile_enabled = false``: the same for USER.md and ``target="user"``.
- both off: no ``memory`` tool for the agent or the review fork, no memory
  guidance, no memory review, and ``tools`` marks the toolset ✗.

Files already on disk are never touched or deleted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langstage_hermes import HermesConfig, hermes_middleware
from langstage_hermes.memory.tool import MemoryToolMiddleware, build_snapshot, enabled_targets

USER_SENTINEL = "User's name is SENTINEL_USER."
MEM_SENTINEL = "The repo uses SENTINEL_MEM for builds."

COMBOS = [(True, True), (False, True), (True, False), (False, False)]


@pytest.fixture
def home(tmp_hermes_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    mem = tmp_hermes_home / "memories"
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "USER.md").write_text(USER_SENTINEL, encoding="utf-8")
    (mem / "MEMORY.md").write_text(MEM_SENTINEL, encoding="utf-8")
    return tmp_hermes_home


def _cfg(memory_enabled: bool, user_profile_enabled: bool, **extra: Any) -> HermesConfig:
    return HermesConfig.resolve(
        overrides={"memory_enabled": memory_enabled, "memory_user_profile_enabled": user_profile_enabled, **extra}
    )


def _files(home: Path) -> tuple[str, str]:
    mem = home / "memories"
    return (mem / "USER.md").read_text(encoding="utf-8"), (mem / "MEMORY.md").read_text(encoding="utf-8")


def _call(tool: Any, **args: Any) -> dict[str, Any]:
    cmd = tool.invoke({"args": args, "name": "memory", "type": "tool_call", "id": "c1"})
    return json.loads(cmd.update["messages"][0].content)


class _Recorder(GenericFakeChatModel):
    """Calls memory(add) on both layers, then answers. Records tools and prompt."""

    bound: ClassVar[list[str]] = []
    system: ClassVar[list[str]] = []

    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        type(self).bound = sorted(getattr(t, "name", None) or t["name"] for t in tools)
        return self

    def _generate(self, messages: list[Any], *a: Any, **kw: Any):
        if messages and getattr(messages[0], "type", "") == "system":
            type(self).system.append(str(messages[0].content))
        return super()._generate(messages, *a, **kw)


def _recorder() -> _Recorder:
    _Recorder.bound, _Recorder.system = [], []
    calls = [
        {"name": "memory", "id": "mu", "args": {"action": "add", "target": "user", "entry": "Likes tea."}},
        {"name": "memory", "id": "mm", "args": {"action": "add", "target": "memory", "entry": "CI is slow."}},
    ]
    return _Recorder(disable_streaming=True, messages=iter([AIMessage("", tool_calls=calls), *[AIMessage("done")] * 6]))


class _Aux(FakeListChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


# ── the config → layers mapping ──────────────────────────────────────


@pytest.mark.parametrize(("mem", "usr"), COMBOS)
def test_enabled_targets_follow_the_switches(home: Path, mem: bool, usr: bool) -> None:
    expected = tuple(t for t, on in (("memory", mem), ("user", usr)) if on)
    assert enabled_targets(_cfg(mem, usr)) == expected


def test_env_vars_turn_the_layers_off(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSTAGE_HERMES_MEMORY_ENABLED", "false")
    monkeypatch.setenv("LANGSTAGE_HERMES_MEMORY_USER_PROFILE_ENABLED", "false")
    assert enabled_targets(HermesConfig.resolve()) == ()


# ── snapshot ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(("mem", "usr"), COMBOS)
def test_snapshot_leaves_out_an_off_layer(home: Path, mem: bool, usr: bool) -> None:
    snap = build_snapshot(memory_enabled=mem, user_profile_enabled=usr)
    assert ("SENTINEL_MEM" in snap["memory_snapshot"]) is mem
    assert ("SENTINEL_USER" in snap["user_snapshot"]) is usr


# ── the tool ─────────────────────────────────────────────────────────


def test_both_off_binds_no_tool(home: Path) -> None:
    assert MemoryToolMiddleware.from_config(_cfg(False, False)).tools == []


@pytest.mark.parametrize(("mem", "usr"), [(False, True), (True, False)])
def test_one_off_refuses_that_target_for_every_action(home: Path, mem: bool, usr: bool) -> None:
    (tool,) = MemoryToolMiddleware.from_config(_cfg(mem, usr)).tools
    off, on = ("memory", "user") if not mem else ("user", "memory")
    before = _files(home)
    for args in (
        {"action": "read"},
        {"action": "add", "entry": "New fact."},
        {"action": "replace", "index": 0, "entry": "Other fact."},
        {"action": "remove", "index": 0},
    ):
        out = _call(tool, target=off, **args)
        assert out["success"] is False
        assert "turned off" in out["error"] and "Nothing was changed" in out["error"]
        # A refused read must not leak the layer's contents.
        assert "SENTINEL" not in json.dumps(out)
    assert _files(home) == before
    assert _call(tool, action="add", target=on, entry="Fine.")["success"] is True


def test_description_names_only_the_layers_that_are_on(home: Path) -> None:
    (tool,) = MemoryToolMiddleware.from_config(_cfg(True, False)).tools
    assert "user   — who the user is" not in tool.description
    assert "user_profile_enabled = false" in tool.description
    assert "must not be saved under 'memory'" in tool.description
    (tool,) = MemoryToolMiddleware.from_config(_cfg(False, True)).tools
    assert "memory — your own notes" not in tool.description
    assert "memory_enabled = false" in tool.description
    (tool,) = MemoryToolMiddleware.from_config(_cfg(True, True)).tools
    assert "turned off" not in tool.description


# ── the full agent (create_hermes_agent) ─────────────────────────────


@pytest.mark.parametrize(("mem", "usr"), COMBOS)
def test_full_agent_honors_the_switches(home: Path, mem: bool, usr: bool) -> None:
    from langstage_hermes.agent import create_hermes_agent
    from langstage_hermes.demo import _close_agent_resources

    graph = create_hermes_agent(
        _cfg(mem, usr, memory_nudge_interval=50),
        model=_recorder(),
        aux_model=_Aux(responses=["ok"] * 10),
        workspace=str(home),
    )
    try:
        graph.invoke({"messages": [{"role": "user", "content": "remember I like tea"}]}, {"configurable": {"thread_id": "t"}})
    finally:
        _close_agent_resources(graph)
    assert ("memory" in _Recorder.bound) is (mem or usr)
    prompt = _Recorder.system[0]
    assert ("SENTINEL_MEM" in prompt) is mem
    assert ("SENTINEL_USER" in prompt) is usr
    if not (mem or usr):
        # No memory guidance either: the prompt doesn't describe a tool that isn't there.
        assert "Use the `memory` tool" not in prompt
    user_md, mem_md = _files(home)
    assert ("Likes tea." in user_md) is usr
    assert ("CI is slow." in mem_md) is mem
    # Off layers keep what was on disk.
    assert USER_SENTINEL in user_md and MEM_SENTINEL in mem_md


@pytest.mark.parametrize(("mem", "usr"), [(True, False), (False, False)])
def test_review_fork_cannot_write_an_off_layer(home: Path, mem: bool, usr: bool) -> None:
    """The real review wiring (demo review model writes ``target="user"``)."""
    from langstage_hermes import create_hermes_agent
    from langstage_hermes.demo import DemoMainModel, DemoReviewModel, _close_agent_resources

    ws = home / "ws"
    ws.mkdir()
    cfg = _cfg(mem, usr, skills_creation_nudge_interval=3, memory_nudge_interval=1)
    agent = create_hermes_agent(cfg, workspace=ws, session_id="r228", model=DemoMainModel(), aux_model=DemoReviewModel())
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": "organise the notes"}], "session_id": "r228"},
            config={"configurable": {"thread_id": "r228"}},
        )
    finally:
        _close_agent_resources(agent)
    assert _files(home) == (USER_SENTINEL, MEM_SENTINEL)


def test_review_fork_still_writes_when_the_layer_is_on(home: Path) -> None:
    from langstage_hermes import create_hermes_agent
    from langstage_hermes.demo import DEMO_MEMORY_ENTRY, DemoMainModel, DemoReviewModel, _close_agent_resources

    ws = home / "ws"
    ws.mkdir()
    cfg = _cfg(False, True, skills_creation_nudge_interval=3, memory_nudge_interval=1)
    agent = create_hermes_agent(cfg, workspace=ws, session_id="r228b", model=DemoMainModel(), aux_model=DemoReviewModel())
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": "organise the notes"}], "session_id": "r228b"},
            config={"configurable": {"thread_id": "r228b"}},
        )
    finally:
        _close_agent_resources(agent)
    assert DEMO_MEMORY_ENTRY in _files(home)[0]


def test_no_memory_review_is_scheduled_without_a_memory_tool() -> None:
    from langstage_hermes.reflection import ReflectionMiddleware

    mw = ReflectionMiddleware(library=None, store=None, model=None, memory_nudge_interval=1, memory_toolset_enabled=False)
    state = {"messages": [AIMessage("done")], "turns_since_memory": 5, "iters_since_skill": 0}
    assert mw.after_model(state) is None


# ── embedded (hermes_middleware) ─────────────────────────────────────


@pytest.mark.parametrize(("mem", "usr"), COMBOS)
def test_embedded_middleware_honors_the_switches(home: Path, mem: bool, usr: bool) -> None:
    h = hermes_middleware(_cfg(mem, usr, memory_nudge_interval=50), aux_model=_Aux(responses=["ok"] * 10))
    try:
        agent = create_agent(_recorder(), tools=[], middleware=[*h], checkpointer=InMemorySaver())
        agent.invoke({"messages": [HumanMessage("remember I like tea")]}, {"configurable": {"thread_id": "e"}})
    finally:
        h.close()
    assert ("memory" in _Recorder.bound) is (mem or usr)
    prompt = _Recorder.system[0]
    assert ("SENTINEL_MEM" in prompt) is mem
    assert ("SENTINEL_USER" in prompt) is usr
    user_md, mem_md = _files(home)
    assert ("Likes tea." in user_md) is usr
    assert ("CI is slow." in mem_md) is mem


# ── the `tools` listing ──────────────────────────────────────────────


def test_tools_marks_memory_off_when_both_layers_are_off(home: Path, tmp_path: Path) -> None:
    from langstage_hermes.cli import cli

    (tmp_path / "langstage-hermes.toml").write_text(
        "[memory]\nmemory_enabled = false\nuser_profile_enabled = false\n", encoding="utf-8"
    )
    out = CliRunner().invoke(cli, ["tools", "--toolset", "memory"]).output
    assert "✗ memory" in out
    assert "memory.memory_enabled and memory.user_profile_enabled = false" in out


def test_tools_keeps_memory_on_with_one_layer_on(home: Path, tmp_path: Path) -> None:
    from langstage_hermes.cli import cli

    (tmp_path / "langstage-hermes.toml").write_text("[memory]\nuser_profile_enabled = false\n", encoding="utf-8")
    out = CliRunner().invoke(cli, ["tools", "--toolset", "memory"]).output
    assert "● memory" in out
