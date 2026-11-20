"""gh #228 follow-up: with one memory layer off, the review asks only for the other.

0.5.7 made the ``memory`` tool refuse an off layer, but the background memory
review still ran on prompts written for both layers. With the user profile off,
``memory_review.md`` asked it to save "things the user revealed about
themselves", and the tool's only remaining target was MEMORY.md. Now:

- the review's system prompt and its per-kind prompts name only the layer that
  is on, and say which kind of fact must not move into it;
- the ``memory`` tool's schema offers only the targets that are on, so an off
  layer isn't something the model can pick (a call that names it anyway is still
  refused, and nothing is written);
- the agent's own memory guidance carries the same note.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver

from langstage_hermes import HermesConfig, hermes_middleware
from langstage_hermes.memory.tool import MemoryToolMiddleware, layer_note
from langstage_hermes.prompts import PromptAssemblyMiddleware
from langstage_hermes.reflection import (
    ReflectionMiddleware,
    build_review_subagent,
    load_prompt,
    review_prompt,
    review_system_prompt,
)

USER_SENTINEL = "User's name is SENTINEL_USER."
MEM_SENTINEL = "The repo uses SENTINEL_MEM for builds."

ONE_OFF = [
    # (memory_enabled, user_profile_enabled, the layer that is off, the one that is on)
    pytest.param(True, False, "user", "memory", id="user-profile-off"),
    pytest.param(False, True, "memory", "user", id="notes-off"),
]


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


# ── the prompts ──────────────────────────────────────────────────────


def test_both_layers_on_keeps_the_shipped_prompts() -> None:
    assert review_system_prompt(("memory", "user")) == load_prompt("combined_review.md")
    assert review_prompt("memory", ("memory", "user")) == load_prompt("memory_review.md")
    assert review_prompt("combined", ("memory", "user")) == load_prompt("combined_review.md")
    assert layer_note(("memory", "user")) == "" and layer_note(()) == ""


def test_no_layer_on_gives_the_review_a_skills_only_system_prompt() -> None:
    text = review_system_prompt(())
    assert text == load_prompt("skill_review.md")
    assert "## Memory" not in text


def test_user_profile_off_review_prompts_ask_only_for_notes() -> None:
    for text in (review_system_prompt(("memory",)), review_prompt("memory", ("memory",))):
        assert "who the user is" not in text
        assert "revealed things about themselves" not in text
        assert "target='memory'" in text
        assert "user_profile_enabled = false" in text
        assert "Do not save personal facts about the user" in text
    combined = review_system_prompt(("memory",))
    # The skills half stays, minus the paragraph that files user preferences under memory.
    assert "## Skills — how to do this class of task" in combined
    assert "### User-preference embedding" not in combined
    assert "If you notice overlapping existing skills" in combined
    assert "## Protected skills" in combined


def test_notes_off_review_prompts_ask_only_for_the_user_profile() -> None:
    for text in (review_system_prompt(("user",)), review_prompt("memory", ("user",))):
        assert "target='user'" in text
        assert "memory_enabled = false" in text
        assert "Do not put environment details, conventions or tool quirks in the user profile" in text
    assert "## Memory — who the user is" in review_system_prompt(("user",))


def test_review_subagent_and_middleware_use_the_adapted_prompts() -> None:
    spec = build_review_subagent(library=None, store=None, aux_model=None, memory_targets=("memory",))
    assert spec["system_prompt"] == review_system_prompt(("memory",))
    mw = ReflectionMiddleware(library=None, store=None, model=None, memory_targets=("user",))
    for kind in ("memory", "combined"):
        msg = mw._review_input({"messages": []}, kind)["messages"][-1]
        assert isinstance(msg, HumanMessage)
        assert msg.content == review_prompt(kind, ("user",))
    assert mw._review_input({"messages": []}, "skills")["messages"][-1].content == load_prompt("skill_review.md")


@pytest.mark.parametrize(("mem", "usr", "off", "on"), ONE_OFF)
def test_agent_memory_guidance_names_the_kept_layer(mem: bool, usr: bool, off: str, on: str) -> None:
    prompt = PromptAssemblyMiddleware(enabled_toolsets=("memory",), memory_targets=(on,)).assemble()
    assert layer_note((on,)) in prompt
    assert "This overrides the priorities above." in prompt
    embedded = PromptAssemblyMiddleware(enabled_toolsets=("memory",), embedded=True, memory_targets=(on,))
    assert layer_note((on,)) in embedded.assemble_embedded()
    both = PromptAssemblyMiddleware(enabled_toolsets=("memory",)).assemble()
    assert "turned off" not in both


# ── the tool offers only the layers that are on ──────────────────────


@pytest.mark.parametrize(("mem", "usr", "off", "on"), ONE_OFF)
def test_tool_schema_offers_only_the_layer_that_is_on(home: Path, mem: bool, usr: bool, off: str, on: str) -> None:
    (tool,) = MemoryToolMiddleware.from_config(_cfg(mem, usr)).tools
    target = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]["target"]
    assert target["enum"] == [on]
    assert "const" not in target
    (both,) = MemoryToolMiddleware.from_config(_cfg(True, True)).tools
    assert convert_to_openai_tool(both)["function"]["parameters"]["properties"]["target"]["enum"] == ["memory", "user"]


@pytest.mark.parametrize(("mem", "usr", "off", "on"), ONE_OFF)
def test_a_call_naming_the_off_layer_is_still_refused(home: Path, mem: bool, usr: bool, off: str, on: str) -> None:
    (tool,) = MemoryToolMiddleware.from_config(_cfg(mem, usr)).tools
    before = _files(home)
    msg = tool.invoke({"args": {"action": "add", "target": off, "entry": "x"}, "name": "memory", "type": "tool_call", "id": "c"})
    out = json.loads(msg.content)
    assert out["success"] is False
    assert "turned off" in out["error"] and "Nothing was changed" in out["error"]
    assert _files(home) == before


# ── the review fork end to end ───────────────────────────────────────


class _ReviewModel(GenericFakeChatModel):
    """A scripted review: writes to the layer that is off, then the one that is on.

    Records the tool schemas it was bound with and the system prompt it got.
    """

    bound: ClassVar[list[dict[str, Any]]] = []
    system: ClassVar[list[str]] = []

    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        type(self).bound = [convert_to_openai_tool(t) for t in tools]
        return self

    def _generate(self, messages: list[Any], *a: Any, **kw: Any):
        if messages and getattr(messages[0], "type", "") == "system":
            type(self).system.append(str(messages[0].content))
        return super()._generate(messages, *a, **kw)


def _review_model(off: str, on: str) -> _ReviewModel:
    _ReviewModel.bound, _ReviewModel.system = [], []
    calls = [
        {"name": "memory", "id": "w-off", "args": {"action": "add", "target": off, "entry": "OFF_LAYER_WRITE"}},
        {"name": "memory", "id": "w-on", "args": {"action": "add", "target": on, "entry": "ON_LAYER_WRITE"}},
    ]
    return _ReviewModel(disable_streaming=True, messages=iter([AIMessage("", tool_calls=calls), *[AIMessage("Done.")] * 6]))


class _Main(FakeListChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


@pytest.mark.parametrize(("mem", "usr", "off", "on"), ONE_OFF)
def test_full_agent_review_is_offered_only_the_on_layer(home: Path, mem: bool, usr: bool, off: str, on: str) -> None:
    from langstage_hermes.agent import create_hermes_agent
    from langstage_hermes.demo import _close_agent_resources

    ws = home / "ws"
    ws.mkdir()
    review = _review_model(off, on)
    agent = create_hermes_agent(
        _cfg(mem, usr, memory_nudge_interval=1, skills_creation_nudge_interval=50),
        workspace=ws,
        session_id="rv",
        model=_Main(responses=["All done."] * 10),
        aux_model=review,
    )
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": "hello"}], "session_id": "rv"},
            config={"configurable": {"thread_id": "rv"}},
        )
    finally:
        _close_agent_resources(agent)
    # The review ran, with a memory tool that offers only the layer that is on.
    (memory_schema,) = [t for t in _ReviewModel.bound if t["function"]["name"] == "memory"]
    assert memory_schema["function"]["parameters"]["properties"]["target"]["enum"] == [on]
    assert layer_note((on,)) in _ReviewModel.system[0]
    user_md, mem_md = _files(home)
    off_text, on_text = (user_md, mem_md) if off == "user" else (mem_md, user_md)
    # The scripted write to the off layer was refused; the on layer got its entry.
    assert "OFF_LAYER_WRITE" not in user_md + mem_md
    assert "ON_LAYER_WRITE" in on_text
    assert off_text == (USER_SENTINEL if off == "user" else MEM_SENTINEL)


@pytest.mark.parametrize(("mem", "usr", "off", "on"), ONE_OFF)
def test_embedded_review_is_offered_only_the_on_layer(home: Path, mem: bool, usr: bool, off: str, on: str) -> None:
    review = _review_model(off, on)
    h = hermes_middleware(_cfg(mem, usr, memory_nudge_interval=1, skills_creation_nudge_interval=50), aux_model=review)
    try:
        agent = create_agent(_Main(responses=["All done."] * 10), tools=[], middleware=[*h], checkpointer=InMemorySaver())
        agent.invoke({"messages": [HumanMessage("hello")]}, {"configurable": {"thread_id": "e"}})
    finally:
        h.close()
    (memory_schema,) = [t for t in _ReviewModel.bound if t["function"]["name"] == "memory"]
    assert memory_schema["function"]["parameters"]["properties"]["target"]["enum"] == [on]
    assert layer_note((on,)) in _ReviewModel.system[0]
    user_md, mem_md = _files(home)
    assert "OFF_LAYER_WRITE" not in user_md + mem_md
    assert "ON_LAYER_WRITE" in (mem_md if on == "memory" else user_md)
