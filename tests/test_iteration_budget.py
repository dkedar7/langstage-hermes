"""Tests for ``IterationBudgetMiddleware``.

Verifies (SPEC §8):

* ``before_agent`` seeds the counter to ``max_iterations`` when missing.
* Each non-refund tool call decrements by 1.
* ``execute_code`` (and other refund tools) does NOT decrement.
* When budget hits 0, ``before_model`` returns ``{"jump_to": "end"}`` + a
  ``[budget_exhausted: ...]`` ``AIMessage``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from deepagent_hermes.budget import IterationBudgetMiddleware

# ── helpers ──────────────────────────────────────────────────────────


def _make_tool_request(*, name: str, state: dict, tool_call_id: str = "call-1"):
    """Build a minimal stand-in for ``ToolCallRequest``.

    The middleware only reads ``request.tool_call["name"]`` and ``request.state``,
    so a small SimpleNamespace-like object is enough — avoids the real
    dataclass which wants a fully-formed ``ToolRuntime``.
    """

    class _Req:
        pass

    req = _Req()
    req.tool_call = {"name": name, "args": {}, "id": tool_call_id}
    req.state = state
    return req


def _tool_result(content: str, tool_call_id: str = "call-1") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id)


# ── tests ────────────────────────────────────────────────────────────


def test_before_agent_seeds_counter_when_missing():
    mw = IterationBudgetMiddleware(max_iterations=90)
    update = mw.before_agent({})
    assert update == {"iteration_budget_remaining": 90}


def test_before_agent_leaves_existing_value_alone():
    mw = IterationBudgetMiddleware(max_iterations=90)
    assert mw.before_agent({"iteration_budget_remaining": 7}) is None


def test_five_tool_calls_drain_budget_of_five():
    mw = IterationBudgetMiddleware(max_iterations=5)
    state = {"iteration_budget_remaining": 5}

    for i in range(5):
        req = _make_tool_request(name="some_tool", state=state, tool_call_id=f"c{i}")
        handler = MagicMock(return_value=_tool_result(f"result-{i}", tool_call_id=f"c{i}"))
        result = mw.wrap_tool_call(req, handler)
        # Result is a Command carrying the message + decremented budget
        assert isinstance(result, Command)
        update = result.update
        assert isinstance(update, dict)
        assert "iteration_budget_remaining" in update
        # Apply the update to our local state mirror
        state["iteration_budget_remaining"] = update["iteration_budget_remaining"]

    assert state["iteration_budget_remaining"] == 0


def test_sixth_before_model_jumps_to_end():
    mw = IterationBudgetMiddleware(max_iterations=5)
    # Simulate state after 5 consumes
    update = mw.before_model({"iteration_budget_remaining": 0})
    assert update is not None
    assert update.get("jump_to") == "end"
    msgs = update.get("messages") or []
    assert msgs and isinstance(msgs[0], AIMessage)
    assert "budget_exhausted" in msgs[0].content
    assert "max_iterations=5" in msgs[0].content


def test_before_model_passes_through_when_budget_remains():
    mw = IterationBudgetMiddleware(max_iterations=5)
    assert mw.before_model({"iteration_budget_remaining": 3}) is None
    assert mw.before_model({"iteration_budget_remaining": 1}) is None


def test_refund_tools_do_not_decrement():
    mw = IterationBudgetMiddleware(max_iterations=5, refund_tools=("execute_code",))
    state = {"iteration_budget_remaining": 5}
    req = _make_tool_request(name="execute_code", state=state)
    handler = MagicMock(return_value=_tool_result("ok"))
    result = mw.wrap_tool_call(req, handler)
    # Refunded tools: result is passed through unmodified — no Command wrap
    assert result is handler.return_value  # untouched ToolMessage


def test_custom_refund_tools_list():
    mw = IterationBudgetMiddleware(
        max_iterations=5, refund_tools=("execute_code", "internal_eval")
    )
    state = {"iteration_budget_remaining": 5}

    # internal_eval: refunded
    req_internal = _make_tool_request(name="internal_eval", state=state)
    handler = MagicMock(return_value=_tool_result("ok"))
    result = mw.wrap_tool_call(req_internal, handler)
    assert result is handler.return_value

    # other tool: decrements
    req_other = _make_tool_request(name="terminal", state=state, tool_call_id="c2")
    handler2 = MagicMock(return_value=_tool_result("done", tool_call_id="c2"))
    result2 = mw.wrap_tool_call(req_other, handler2)
    assert isinstance(result2, Command)
    assert result2.update["iteration_budget_remaining"] == 4


def test_wrap_tool_call_preserves_existing_command_updates():
    """If the handler itself returns a Command, our decrement is merged in."""
    mw = IterationBudgetMiddleware(max_iterations=5)
    state = {"iteration_budget_remaining": 5}
    req = _make_tool_request(name="foo", state=state)

    inner_cmd = Command(update={"some_other_key": "preserved"})
    handler = MagicMock(return_value=inner_cmd)
    result = mw.wrap_tool_call(req, handler)
    assert isinstance(result, Command)
    assert result.update["some_other_key"] == "preserved"
    assert result.update["iteration_budget_remaining"] == 4


def test_before_model_handles_missing_remaining_field():
    """If a thread somehow lost the field, default back to ``max_iterations``
    so we don't lock the agent out."""
    mw = IterationBudgetMiddleware(max_iterations=10)
    # No 'iteration_budget_remaining' key set
    assert mw.before_model({}) is None
