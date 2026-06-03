"""Tests for ``ReflectionMiddleware`` — the closed-loop trigger logic.

These tests are deliberately driver-free: we don't compile a graph or run
``langchain.agents.create_agent``. Instead we exercise the middleware's hooks
directly with hand-built ``ToolCallRequest`` / state dicts. That keeps the
tests fast, lets us assert on intermediate state updates, and avoids needing
real model credentials or a checkpointer.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as tool_decorator
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deepagent_hermes.reflection import (
    ReflectionMiddleware,
    build_review_subagent,
    load_prompt,
)

# ── helpers ──────────────────────────────────────────────────────────


@tool_decorator
def fake_tool(query: str) -> str:
    """No-op tool for exercising the wrap path."""
    return f"ran: {query}"


def make_tool_request(name: str, *, state: dict[str, Any]) -> ToolCallRequest:
    """Build a minimal ``ToolCallRequest`` we can hand to ``wrap_tool_call``."""
    tool_call = {"name": name, "args": {"query": "hi"}, "id": f"call_{name}"}
    # `runtime` may be None per the dataclass docs; the middleware shouldn't touch it.
    return ToolCallRequest(
        tool_call=tool_call,
        tool=fake_tool,
        state=state,
        runtime=None,  # type: ignore[arg-type]
    )


def make_handler(reply: str = "ok") -> Any:
    """Return a sync handler that ignores the request and yields a `ToolMessage`."""
    def _handler(req: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content=reply, tool_call_id=req.tool_call["id"])
    return _handler


def make_middleware(
    *,
    skill_nudge_interval: int = 3,
    memory_nudge_interval: int = 3,
    review_graph: Any = None,
) -> ReflectionMiddleware:
    return ReflectionMiddleware(
        skill_nudge_interval=skill_nudge_interval,
        memory_nudge_interval=memory_nudge_interval,
        library=MagicMock(name="library"),
        store=MagicMock(name="store"),
        model=MagicMock(name="model"),
        aux_model=MagicMock(name="aux_model"),
        review_graph=review_graph,
    )


def merge_state(state: dict[str, Any], update: Any) -> dict[str, Any]:
    """Apply a ``Command``-style or dict-style update to a state dict."""
    if update is None:
        return state
    if isinstance(update, Command):
        return {**state, **(update.update or {})}
    if isinstance(update, dict):
        return {**state, **update}
    return state


# ── tests ────────────────────────────────────────────────────────────


def test_skill_counter_increments_on_non_skill_tool_calls():
    """Three fake-tool calls should set ``iters_since_skill`` to 3."""
    mw = make_middleware(skill_nudge_interval=3)
    state: dict[str, Any] = {"messages": [HumanMessage(content="go")], "iters_since_skill": 0}

    handler = make_handler()
    for _ in range(3):
        req = make_tool_request("fake_tool", state=state)
        result = mw.wrap_tool_call(req, handler)
        state = merge_state(state, result)

    assert state["iters_since_skill"] == 3


def test_skill_manage_call_resets_counter():
    """A ``skill_manage`` call resets ``iters_since_skill`` to 0."""
    mw = make_middleware(skill_nudge_interval=10)
    state: dict[str, Any] = {"messages": [], "iters_since_skill": 7}
    req = make_tool_request("skill_manage", state=state)
    result = mw.wrap_tool_call(req, make_handler())
    state = merge_state(state, result)
    assert state["iters_since_skill"] == 0


def test_memory_call_resets_memory_counter():
    """A ``memory`` call resets ``turns_since_memory`` to 0."""
    mw = make_middleware()
    state: dict[str, Any] = {"messages": [], "turns_since_memory": 5}
    req = make_tool_request("memory", state=state)
    result = mw.wrap_tool_call(req, make_handler())
    state = merge_state(state, result)
    assert state["turns_since_memory"] == 0


def test_pending_review_kind_fires_at_threshold():
    """When ``iters_since_skill`` reaches the threshold AND the model produced a
    final response (no pending tool calls), ``after_model`` flags
    ``pending_review_kind = "skills"``."""
    mw = make_middleware(skill_nudge_interval=3, memory_nudge_interval=99)

    state: dict[str, Any] = {"messages": [HumanMessage(content="go")], "iters_since_skill": 0}
    handler = make_handler()

    # Three tool-using turns of a non-skill tool.
    for _ in range(3):
        req = make_tool_request("fake_tool", state=state)
        result = mw.wrap_tool_call(req, handler)
        state = merge_state(state, result)
    assert state["iters_since_skill"] == 3

    # Simulate final-response AIMessage on the wire.
    state["messages"] = [
        HumanMessage(content="go"),
        AIMessage(content="all done"),
    ]
    update = mw.after_model(state)
    assert update is not None
    assert update["pending_review_kind"] == "skills"
    assert "last_review_started_at" in update


def test_pending_review_does_not_fire_with_pending_tool_calls():
    """Tool calls are still in flight → no review trigger yet."""
    mw = make_middleware(skill_nudge_interval=2, memory_nudge_interval=99)
    state: dict[str, Any] = {
        "messages": [
            HumanMessage(content="go"),
            AIMessage(
                content="",
                tool_calls=[{"name": "fake_tool", "args": {"query": "x"}, "id": "t1"}],
            ),
        ],
        "iters_since_skill": 5,  # well over threshold
        "turns_since_memory": 0,
    }
    assert mw.after_model(state) is None


def test_pending_review_does_not_fire_below_threshold():
    mw = make_middleware(skill_nudge_interval=10, memory_nudge_interval=10)
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="go"), AIMessage(content="done")],
        "iters_since_skill": 3,
        "turns_since_memory": 3,
    }
    assert mw.after_model(state) is None


def test_combined_review_when_both_counters_fire():
    mw = make_middleware(skill_nudge_interval=2, memory_nudge_interval=2)
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="go"), AIMessage(content="done")],
        "iters_since_skill": 5,
        "turns_since_memory": 5,
    }
    update = mw.after_model(state)
    assert update is not None
    assert update["pending_review_kind"] == "combined"


def test_user_turn_bumps_memory_counter():
    """A fresh user prompt (HumanMessage after a final AIMessage) bumps
    ``turns_since_memory``."""
    mw = make_middleware()
    state: dict[str, Any] = {
        "messages": [
            HumanMessage(content="prev"),
            AIMessage(content="prev answer"),
            HumanMessage(content="next"),
        ],
        "turns_since_memory": 4,
    }
    update = mw.before_model(state)
    assert update == {"turns_since_memory": 5}


def test_tool_result_resume_does_not_bump_memory_counter():
    """A ToolMessage before the latest HumanMessage isn't a real user turn —
    don't bump the counter on resume."""
    mw = make_middleware()
    state: dict[str, Any] = {
        "messages": [
            HumanMessage(content="ask"),
            ToolMessage(content="result", tool_call_id="t1"),
        ],
        "turns_since_memory": 4,
    }
    # Latest is ToolMessage, not HumanMessage — should be a no-op.
    assert mw.before_model(state) is None


def test_after_agent_invokes_review_graph_and_resets():
    """When ``pending_review_kind`` is set, ``after_agent`` should invoke the
    review graph and clear the relevant counters."""
    review_graph = MagicMock(name="review_graph")
    review_graph.invoke.return_value = {"messages": [AIMessage(content="reviewed")]}

    mw = make_middleware(review_graph=review_graph)
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="hi"), AIMessage(content="ok")],
        "iters_since_skill": 12,
        "turns_since_memory": 12,
        "pending_review_kind": "combined",
    }
    update = mw.after_agent(state)
    review_graph.invoke.assert_called_once()
    assert update == {
        "pending_review_kind": None,
        "iters_since_skill": 0,
        "turns_since_memory": 0,
    }


def test_after_agent_noop_without_pending_review():
    mw = make_middleware()
    state: dict[str, Any] = {
        "messages": [AIMessage(content="ok")],
        "pending_review_kind": None,
    }
    assert mw.after_agent(state) is None


def test_review_subagent_factory_returns_well_formed_dict():
    """The factory should produce a SubAgent-shaped dict pointing at the
    combined-review prompt."""
    spec = build_review_subagent(
        library=MagicMock(),
        store=MagicMock(),
        aux_model=MagicMock(),
        tools=[],
    )
    assert spec["name"] == "review"
    assert "system_prompt" in spec
    assert "Review the conversation" in spec["system_prompt"]
    assert "memory" in spec["system_prompt"].lower()
    assert "skill" in spec["system_prompt"].lower()
    assert spec["tools"] == []


def test_load_prompt_finds_known_prompts():
    """The dev-mode fallback should locate every required prompt file."""
    for name in (
        "memory_review.md",
        "skill_review.md",
        "combined_review.md",
        "curator_review.md",
        "default_identity.md",
    ):
        body = load_prompt(name)
        assert body.strip(), f"{name} is empty"
        # Comment header is the attribution marker.
        assert "Adapted from hermes-agent" in body
