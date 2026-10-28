"""``ModelOverrideMiddleware`` — honor ``/model`` for the rest of the session (gh #193).

``chat``'s ``/model provider:name`` stores the id in the REPL state, and every turn
forwards it into graph state as ``model_override``. Before this middleware nothing
read it, so the switch was reported but every turn still ran on the original model.

``wrap_model_call`` swaps the request's model for the override (built once per id
with the same ``init_chat_model`` path as the main model). ``create_agent`` binds the
tools to whatever model the request carries, so tools keep working. The override
applies only to the main model; the aux model (reflection, compression summaries)
is unchanged.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse


class _ModelOverrideState(AgentState):
    """Declare ``model_override`` on the graph state so the turn input reaches the hooks."""

    model_override: NotRequired[str | None]


class ModelOverrideMiddleware(AgentMiddleware):
    """Run the model call on ``state["model_override"]`` when it names another model."""

    state_schema = _ModelOverrideState

    def __init__(self, default_model_id: str | None, init_model: Callable[[str], Any]) -> None:
        super().__init__()
        self._default = (default_model_id or "").strip() or None
        self._init = init_model
        self._models: dict[str, Any] = {}

    def _override(self, request: ModelRequest) -> ModelRequest:
        state = request.state or {}
        wanted = (state.get("model_override") or "").strip()
        if not wanted or wanted == self._default:
            return request
        model = self._models.get(wanted)
        if model is None:
            model = self._models[wanted] = self._init(wanted)
        return request.override(model=model)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(self._override(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._override(request))


__all__ = ["ModelOverrideMiddleware"]
