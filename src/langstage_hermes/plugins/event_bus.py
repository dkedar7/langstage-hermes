"""``PluginEventBus`` — translate middleware events into plugin hook fires.

SPEC §15.3 declares 17 plugin lifecycle hook names. The v0.1 system wired
five of them implicitly through other middleware (``pre_tool_call`` and
``post_tool_call`` via ``wrap_tool_call``; ``pre_llm_call`` and
``post_llm_call`` via ``wrap_model_call``; ``on_session_start`` /
``on_session_end`` via ``before_agent`` / ``after_agent``). The other 12
were accepted by ``PluginContext.register_hook`` but never invoked — plugin
authors registered transforms or finalizers and watched them silently
no-op.

``PluginEventBus`` closes that gap. It is a single ``AgentMiddleware`` that
sits as the outermost layer in the stack and, on each middleware event,
walks the global hook registry (``context.get_global_hook_registry()``)
and invokes every registered plugin callback for the corresponding hook
name.

Hooks wired by this bus (8 net-new fires in v0.2, plus consolidated firing
for the 7 already wired by other middleware):

  Tool surface
    - ``pre_tool_call``               — before ``handler(request)``
    - ``post_tool_call``              — after ``handler(request)``
    - ``transform_terminal_output``   — on result content when tool name in
                                        ``{"terminal","process","execute_code","bash"}``
    - ``transform_tool_result``       — on every tool's result content
    - ``pre_approval_request``        — when the tool is in the configured
                                        interrupt set, before raising
    - ``post_approval_response``      — after a tool returns following a
                                        resume from interrupt

  LLM surface
    - ``pre_llm_call``                — before ``handler(request)``
    - ``post_llm_call``               — after ``handler(request)``
    - ``transform_llm_output``        — on ``ModelResponse.result`` before
                                        returning
    - ``pre_api_request``             — conflated with ``pre_llm_call`` in
                                        v0.2 (no separate API-request layer
                                        exists in the current middleware
                                        shape — would need a hand-written
                                        ``BaseChatModel`` wrapper)
    - ``post_api_request``            — conflated with ``post_llm_call``

  Session lifecycle
    - ``on_session_start``            — ``before_agent``
    - ``on_session_end``              — ``after_agent``
    - ``on_session_finalize``         — ``after_agent`` AFTER ``on_session_end``

Hooks intentionally NOT wired by this bus (would need machinery we don't
have here in v0.2; documented so plugin authors don't file bugs):

    - ``on_session_reset``    — fired only by the CLI's ``/reset`` command,
                                not by any middleware event. The CLI is
                                responsible.
    - ``subagent_stop``       — would need hooks inside ``SubAgentMiddleware``
                                we don't have access to. Skip.
    - ``pre_gateway_dispatch``— gateway / messaging surface is out of scope
                                per SPEC §0. Skip.

Placement: register this middleware as the OUTERMOST layer in the agent
stack so it sees the unmodified request and the final response. Inner
middleware (caching, compression, prompt assembly, recorder) can still
mutate the request/response between the outer ``pre_*`` and ``post_*``
fires; that's by design — plugin transforms apply on top of the inner
middleware pipeline, not in lieu of it.

Error handling: every plugin callback runs under ``try/except``. A raised
exception is logged at WARNING with full traceback and swallowed — one
bad plugin must NOT kill the agent.

Transform hooks: callbacks return either ``None`` (pass-through, value
unchanged) or a replacement value. Multiple plugins chain in registration
order; each one sees the previous plugin's output.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from langstage_hermes.plugins.context import get_global_hook_registry

logger = logging.getLogger(__name__)


# Tool names that should additionally fire ``transform_terminal_output``.
# Matches the canonical Hermes terminal toolset names; extensible via the
# ``terminal_tool_names`` constructor arg.
_DEFAULT_TERMINAL_TOOL_NAMES: frozenset[str] = frozenset({"terminal", "process", "execute_code", "bash"})


def _tool_name_of(request: ToolCallRequest) -> str | None:
    """Best-effort extraction of the tool name from a ``ToolCallRequest``."""
    call = getattr(request, "tool_call", None) or {}
    if isinstance(call, dict):
        return call.get("name")
    return getattr(call, "name", None)


def _tool_args_of(request: ToolCallRequest) -> Any:
    """Best-effort extraction of the tool args from a ``ToolCallRequest``."""
    call = getattr(request, "tool_call", None) or {}
    if isinstance(call, dict):
        return call.get("args")
    return getattr(call, "args", None)


def _extract_result_content(result: Any) -> tuple[Any, ToolMessage | None]:
    """Return ``(content, tool_message)`` for the user-visible tool output.

    Handles three shapes:
      - ``ToolMessage`` — content is ``message.content``.
      - ``Command`` with ``update["messages"]`` containing a ``ToolMessage``
        — content is the last ToolMessage's content.
      - anything else — content is the raw value; ``tool_message`` is None.
    """
    if isinstance(result, ToolMessage):
        return result.content, result
    if isinstance(result, Command):
        update = result.update or {}
        if isinstance(update, dict):
            msgs = update.get("messages") or []
            for m in reversed(msgs):
                if isinstance(m, ToolMessage):
                    return m.content, m
    return result, None


def _replace_result_content(result: Any, tool_msg: ToolMessage | None, new_content: Any) -> Any:
    """Return a new result with ``new_content`` substituted in the right slot.

    Mirrors :func:`_extract_result_content`. When the original carrier is a
    ``ToolMessage`` we ``model_copy`` it; when it's wrapped in a
    ``Command``, we replace the ToolMessage inside ``update["messages"]``
    and return a new ``Command``. Anything else short-circuits to the
    replacement value.
    """
    if tool_msg is not None and isinstance(result, ToolMessage):
        return result.model_copy(update={"content": new_content})
    if isinstance(result, Command) and tool_msg is not None:
        update = dict(result.update or {})
        msgs = list(update.get("messages") or [])
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], ToolMessage) and msgs[i] is tool_msg:
                msgs[i] = tool_msg.model_copy(update={"content": new_content})
                break
        update["messages"] = msgs
        return Command(update=update, goto=result.goto)
    # No carrier — caller is responsible for using the raw value.
    return new_content


class PluginEventBus(AgentMiddleware):
    """Fire plugin lifecycle hooks at the corresponding LangChain middleware events.

    Register as the OUTERMOST middleware so this sees the unmodified request
    and the final response. Plugin functions registered via
    :meth:`PluginContext.register_hook` are invoked synchronously; any
    exception they raise is logged at WARNING and swallowed so one bad
    plugin can't kill the agent.

    Args:
        hook_registry: Optional explicit hook store. If provided, the bus
            reads from it instead of the module-level
            ``get_global_hook_registry()`` mapping. Useful in tests.
        terminal_tool_names: Optional override of the tool-name set that
            triggers ``transform_terminal_output``. Defaults to the canonical
            Hermes set (``terminal``, ``process``, ``execute_code``, ``bash``).
        interrupt_tool_names: Optional set of tool names that should fire
            ``pre_approval_request`` / ``post_approval_response``. Defaults
            to the names declared in the ``DEEPAGENT_HERMES_INTERRUPT_ON``
            environment variable (comma-separated) — matching the
            ``HumanInTheLoopMiddleware`` configuration in
            ``langstage_hermes.agent``.
    """

    def __init__(
        self,
        *,
        hook_registry: dict[str, list[Callable[..., Any]]] | None = None,
        terminal_tool_names: frozenset[str] | set[str] | None = None,
        interrupt_tool_names: frozenset[str] | set[str] | None = None,
    ) -> None:
        super().__init__()
        # Tools attribute is required by the AgentMiddleware contract.
        self.tools: list[Any] = []
        self._explicit_registry = hook_registry
        self._terminal_tool_names = frozenset(
            terminal_tool_names if terminal_tool_names is not None else _DEFAULT_TERMINAL_TOOL_NAMES
        )
        if interrupt_tool_names is None:
            csv = os.getenv("LANGSTAGE_HERMES_INTERRUPT_ON") or os.getenv("DEEPAGENT_HERMES_INTERRUPT_ON", "")
            self._interrupt_tool_names = frozenset(n.strip() for n in csv.split(",") if n.strip())
        else:
            self._interrupt_tool_names = frozenset(interrupt_tool_names)

    # ── registry access ─────────────────────────────────────────────

    def _registry(self) -> dict[str, list[Callable[..., Any]]]:
        """Return the live hook registry — module-level by default."""
        if self._explicit_registry is not None:
            return self._explicit_registry
        return get_global_hook_registry()

    # ── fire helpers ────────────────────────────────────────────────

    def _fire(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Invoke every callback registered under ``hook_name``.

        Side-effect-only — return values are ignored. For transform hooks
        (which mutate a pipeline value) use :meth:`_fire_transform`.
        """
        for fn in list(self._registry().get(hook_name, [])):
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                logger.warning("plugin hook %s raised: %s", hook_name, exc, exc_info=True)

    def _fire_transform(self, hook_name: str, value: Any, *extra_args: Any, **kwargs: Any) -> Any:
        """Run each callback for ``hook_name``, chaining the output as the new value.

        Callback signature: ``fn(value, *extra_args, **kwargs) -> Any | None``.
        Returning ``None`` is a pass-through (value unchanged). Returning
        anything else replaces ``value`` for subsequent callbacks.
        """
        current = value
        for fn in list(self._registry().get(hook_name, [])):
            try:
                result = fn(current, *extra_args, **kwargs)
            except Exception as exc:
                logger.warning("plugin hook %s raised: %s", hook_name, exc, exc_info=True)
                continue
            if result is not None:
                current = result
        return current

    # ── before_agent / after_agent (session lifecycle) ──────────────

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Fire ``on_session_start``."""
        del runtime
        self._fire("on_session_start", state)
        return None

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Fire ``on_session_end`` then ``on_session_finalize`` in that order."""
        del runtime
        self._fire("on_session_end", state)
        self._fire("on_session_finalize", state)
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)

    # ── wrap_model_call (LLM surface) ───────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Fire pre/post LLM hooks around the inner handler.

        ``pre_llm_call`` callbacks may return a replacement ``ModelRequest``
        (anything else, including ``None``, leaves the request unchanged).
        ``post_llm_call`` callbacks may return a replacement ``ModelResponse``.
        ``transform_llm_output`` is fired between the inner handler and the
        ``post_llm_call`` notification, and it operates on the
        ``response.result`` list of messages.

        ``pre_api_request`` and ``post_api_request`` are conflated with
        ``pre_llm_call`` and ``post_llm_call`` in v0.2 — see module docstring.
        """
        # ── pre_llm_call / pre_api_request ──
        request = self._fire_request_transform("pre_llm_call", request)
        request = self._fire_request_transform("pre_api_request", request)

        response = handler(request)

        # ── transform_llm_output ──
        new_messages = self._fire_transform("transform_llm_output", response.result)
        if new_messages is not response.result:
            response = self._replace_response_messages(response, new_messages)

        # ── post_llm_call / post_api_request ──
        response = self._fire_response_transform("post_llm_call", request, response)
        response = self._fire_response_transform("post_api_request", request, response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        request = self._fire_request_transform("pre_llm_call", request)
        request = self._fire_request_transform("pre_api_request", request)
        response = await handler(request)
        new_messages = self._fire_transform("transform_llm_output", response.result)
        if new_messages is not response.result:
            response = self._replace_response_messages(response, new_messages)
        response = self._fire_response_transform("post_llm_call", request, response)
        response = self._fire_response_transform("post_api_request", request, response)
        return response

    def _fire_request_transform(self, hook_name: str, request: ModelRequest) -> ModelRequest:
        """Fire a ``pre_llm_call``-style hook; allow returning a new request."""
        current = request
        for fn in list(self._registry().get(hook_name, [])):
            try:
                result = fn(current)
            except Exception as exc:
                logger.warning("plugin hook %s raised: %s", hook_name, exc, exc_info=True)
                continue
            if result is not None:
                current = result
        return current

    def _fire_response_transform(
        self,
        hook_name: str,
        request: ModelRequest,
        response: ModelResponse,
    ) -> ModelResponse:
        """Fire a ``post_llm_call``-style hook; allow returning a new response."""
        current = response
        for fn in list(self._registry().get(hook_name, [])):
            try:
                result = fn(request, current)
            except Exception as exc:
                logger.warning("plugin hook %s raised: %s", hook_name, exc, exc_info=True)
                continue
            if result is not None:
                current = result
        return current

    @staticmethod
    def _replace_response_messages(response: ModelResponse, new_messages: Any) -> ModelResponse:
        """Return a ModelResponse copy with ``result`` swapped to ``new_messages``."""
        return ModelResponse(
            result=list(new_messages) if new_messages is not None else [],
            structured_response=response.structured_response,
        )

    # ── wrap_tool_call (tool surface) ───────────────────────────────

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Fire pre/post tool hooks + transforms around the inner handler.

        Order:
          1. ``pre_tool_call(request)``
          2. ``pre_approval_request(request)`` if tool is in interrupt set
          3. ``handler(request)``
          4. ``post_approval_response(request, result)`` if tool was in interrupt set
          5. ``transform_terminal_output(content, tool_name, args)`` for terminal tools
          6. ``transform_tool_result(content, tool_name, args)`` for every tool
          7. ``post_tool_call(request, result)``
        """
        tool_name = _tool_name_of(request)
        is_interrupt = tool_name is not None and tool_name in self._interrupt_tool_names

        self._fire("pre_tool_call", request)
        if is_interrupt:
            self._fire("pre_approval_request", request)

        result = handler(request)

        if is_interrupt:
            self._fire("post_approval_response", request, result)

        result = self._apply_tool_result_transforms(request, result, tool_name)

        self._fire("post_tool_call", request, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        tool_name = _tool_name_of(request)
        is_interrupt = tool_name is not None and tool_name in self._interrupt_tool_names
        self._fire("pre_tool_call", request)
        if is_interrupt:
            self._fire("pre_approval_request", request)
        result = await handler(request)
        if is_interrupt:
            self._fire("post_approval_response", request, result)
        result = self._apply_tool_result_transforms(request, result, tool_name)
        self._fire("post_tool_call", request, result)
        return result

    def _apply_tool_result_transforms(
        self,
        request: ToolCallRequest,
        result: Any,
        tool_name: str | None,
    ) -> Any:
        """Run terminal-output + tool-result transforms on ``result``'s content."""
        registry = self._registry()
        has_terminal = bool(registry.get("transform_terminal_output"))
        has_tool = bool(registry.get("transform_tool_result"))
        if not has_terminal and not has_tool:
            return result

        content, tool_msg = _extract_result_content(result)
        original_content = content
        args = _tool_args_of(request)

        if has_terminal and tool_name in self._terminal_tool_names:
            content = self._fire_transform("transform_terminal_output", content, tool_name, args)

        if has_tool:
            content = self._fire_transform("transform_tool_result", content, tool_name, args)

        if content is original_content:
            return result
        return _replace_result_content(result, tool_msg, content)


__all__ = ["PluginEventBus"]
