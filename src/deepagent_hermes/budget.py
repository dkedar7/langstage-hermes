"""``IterationBudgetMiddleware`` — per-thread iteration cap (SPEC §8).

Hermes tracks budget as ``IterationBudget`` instance attrs on each ``AIAgent``
(parent = 90, subagent = 50). In ``deepagents``, middleware is stateless, so
the counter lives in ``HermesState["iteration_budget_remaining"]`` — that field
is the per-thread persistence boundary.

Hooks:

* ``before_agent`` — seed the counter if missing (idempotent).
* ``before_model`` — gated by ``@hook_config(can_jump_to=["end"])``: when the
  remaining budget is ``<= 0`` we append a final ``AIMessage`` describing the
  exhaustion and return ``{"jump_to": "end"}``.
* ``wrap_tool_call`` — runs the tool first, then decrements the counter via
  a ``Command(update=...)`` unless the tool name is in ``refund_tools``
  (``execute_code`` by default — programmatic calls are refunded so they
  don't eat the agent's budget).

The decrement happens AFTER the tool returns so a failing tool also costs a
budget unit (matches Hermes's ``IterationBudget.consume()`` semantics —
consumption is unconditional, refund is an explicit opt-in for known
programmatic tools).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import hook_config
from langchain.agents.middleware.types import (
    AgentMiddleware,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

_DEFAULT_REFUND_TOOLS: tuple[str, ...] = ("execute_code",)


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    if state is None:
        return default
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


class IterationBudgetMiddleware(AgentMiddleware):
    """Decrement-on-tool-call iteration budget with end-jump on exhaustion.

    Args:
        max_iterations: Initial budget seeded on the first agent invocation.
            Default 90 (Hermes parent). For subagents pass ``50``.
        refund_tools: Tool names that DON'T consume the budget. Default
            ``("execute_code",)`` — programmatic loops shouldn't eat the
            outer agent's per-turn cap.
    """

    def __init__(
        self,
        max_iterations: int = 90,
        *,
        refund_tools: tuple[str, ...] = _DEFAULT_REFUND_TOOLS,
    ) -> None:
        self.max_iterations = max_iterations
        self.refund_tools = tuple(refund_tools)

    # ── before_agent: seed counter ───────────────────────────────────

    def before_agent(
        self, state: Any, runtime: Runtime[Any] | None = None
    ) -> dict[str, Any] | None:
        """Seed ``iteration_budget_remaining`` to ``max_iterations`` if unset.

        Idempotent: if a caller (e.g. a resumed thread) already has a non-None
        remaining value we leave it alone.
        """
        current = _state_get(state, "iteration_budget_remaining", None)
        if current is None:
            return {"iteration_budget_remaining": self.max_iterations}
        return None

    async def abefore_agent(
        self, state: Any, runtime: Runtime[Any] | None = None
    ) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    # ── before_model: check + jump-to-end on exhaustion ──────────────

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: Any, runtime: Runtime[Any] | None = None
    ) -> dict[str, Any] | None:
        """If budget is exhausted, append a final ``AIMessage`` and jump to end."""
        remaining = _state_get(state, "iteration_budget_remaining", self.max_iterations)
        if remaining is None:
            remaining = self.max_iterations
        if remaining > 0:
            return None

        final = AIMessage(
            content=f"[budget_exhausted: max_iterations={self.max_iterations} reached]"
        )
        return {"messages": [final], "jump_to": "end"}

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: Any, runtime: Runtime[Any] | None = None
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    # ── wrap_tool_call: decrement after the tool runs ────────────────

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run the tool, then decrement the budget unless the tool is refunded."""
        result = handler(request)
        return self._maybe_decrement(request, result)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        return self._maybe_decrement(request, result)

    # ── private ──────────────────────────────────────────────────────

    def _maybe_decrement(
        self,
        request: Any,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        """Apply the decrement to the result if this tool isn't refunded.

        We attach the decrement as a state update on the returned ``Command``
        (or wrap a plain ``ToolMessage`` in one). ``langgraph`` merges the
        update into the running state, so the next ``before_model`` reads the
        new value.
        """
        tool_name = self._tool_name(request)
        if tool_name in self.refund_tools:
            return result

        # Read the live remaining from the request's state snapshot.
        state = getattr(request, "state", None)
        remaining = _state_get(state, "iteration_budget_remaining", self.max_iterations)
        if remaining is None:
            remaining = self.max_iterations
        new_remaining = max(0, int(remaining) - 1)

        # If the handler returned a Command, fold our update into it.
        if isinstance(result, Command):
            existing_update = result.update or {}
            if isinstance(existing_update, dict):
                merged = {**existing_update, "iteration_budget_remaining": new_remaining}
                # ``Command`` is a dataclass-ish wrapper — easiest to rebuild it.
                return Command(
                    update=merged,
                    goto=result.goto,
                    graph=result.graph,
                    resume=result.resume,
                )
            # Non-dict update — leave as-is (shouldn't happen in practice).
            return result

        # Plain ToolMessage: wrap in a Command carrying both the message and
        # the decrement so the langgraph state merge picks up both.
        return Command(
            update={
                "messages": [result],
                "iteration_budget_remaining": new_remaining,
            }
        )

    @staticmethod
    def _tool_name(request: Any) -> str:
        tc = getattr(request, "tool_call", None) or {}
        if isinstance(tc, dict):
            return str(tc.get("name") or "")
        return str(getattr(tc, "name", "") or "")


__all__ = ["IterationBudgetMiddleware"]
