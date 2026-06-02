"""``AnthropicCachingS3Middleware`` — Hermes ``system_and_3`` cache strategy.

SPEC §6 / Hermes ``agent/prompt_caching.py``: place 4 ``cache_control``
breakpoints per request — the system message + the last 3 non-system
messages — at a uniform TTL (``"5m"`` default, ``"1h"`` opt-in). Anthropic
caps explicit breakpoints at 4 per request, so this strategy is the maximum
cache discipline you can get without breaking the limit.

Implementation note (langchain-anthropic interop): we subclass
``AnthropicPromptCachingMiddleware`` from ``langchain_anthropic.middleware``
because:

1. It already handles the system-message tagging via ``_tag_system_message``
   (which deals with all the format variations: ``str`` content vs. list of
   content blocks).
2. It already gates on ``isinstance(request.model, ChatAnthropic)`` and respects
   ``unsupported_model_behavior``.
3. It already sets ``model_settings["cache_control"]`` so the Bedrock
   transport variant routes correctly.

We extend it by overriding ``_apply_caching`` to add cache_control to the
last 3 non-system messages on top of the parent's system+tools breakpoints.
Tool tagging is dropped (Hermes never tags the tools block separately — the
4 breakpoint budget goes entirely to system + 3 messages).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
)
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import AnyMessage, SystemMessage


def _tag_message_last_block(message: AnyMessage, cache_control: dict[str, str]) -> AnyMessage:
    """Return a copy of ``message`` with ``cache_control`` on its last content block.

    Mirrors the format-handling in ``langchain_anthropic._tag_system_message``:

    - ``str`` content -> wrap in a single ``{"type": "text", ...}`` block with
      ``cache_control`` attached.
    - ``list`` content -> copy the list, replace the last dict block with one
      carrying ``cache_control``.
    - Anything else (e.g. ``None``, non-string scalar) -> message returned
      unchanged so the cache budget is not consumed on an unaddressable slot.
    """
    content = message.content
    if isinstance(content, str):
        if not content:
            return message
        new_content: list[Any] = [
            {"type": "text", "text": content, "cache_control": cache_control}
        ]
        # ``model_copy`` preserves message subclass (Human / AI / Tool / System)
        return message.model_copy(update={"content": new_content})
    if isinstance(content, list):
        if not content:
            return message
        new_content = list(content)
        last = new_content[-1]
        if isinstance(last, dict):
            new_content[-1] = {**last, "cache_control": cache_control}
        elif isinstance(last, str):
            # Convert a trailing raw string into a text block carrying cache_control.
            new_content[-1] = {"type": "text", "text": last, "cache_control": cache_control}
        else:
            return message
        return message.model_copy(update={"content": new_content})
    return message


class AnthropicCachingS3Middleware(AnthropicPromptCachingMiddleware):
    """``system_and_3`` cache strategy — system message + last 3 non-system messages.

    Inherits the parent's no-op behavior for non-Anthropic models — the call
    flows straight through ``handler(request)`` if the model isn't ChatAnthropic.

    Args:
        ttl: ``"5m"`` (default) or ``"1h"``. Same wire format as the upstream
            middleware. ``"1h"`` is opt-in because Anthropic bills it per the
            ``ttl`` field; for short interactive sessions ``"5m"`` is enough.
        min_messages_to_cache: Skip caching entirely if the conversation has
            fewer messages than this. Default 0 (always cache when Anthropic).
    """

    def __init__(
        self,
        ttl: Literal["5m", "1h"] = "5m",
        min_messages_to_cache: int = 0,
        *,
        unsupported_model_behavior: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> None:
        # ``type`` is fixed to ``"ephemeral"`` upstream; we mirror that.
        super().__init__(
            type="ephemeral",
            ttl=ttl,
            min_messages_to_cache=min_messages_to_cache,
            unsupported_model_behavior=unsupported_model_behavior,
        )

    # ── strategy override ────────────────────────────────────────────

    def _apply_caching(self, request: ModelRequest) -> ModelRequest:
        """Apply ``system_and_3``: system breakpoint + last 3 non-system messages.

        Total breakpoints: up to 4, matching the Anthropic per-request cap.
        If the conversation has fewer than 3 non-system messages, we tag as
        many as exist (e.g. 1 message + system => 2 breakpoints).

        Tools are NOT tagged in ``system_and_3`` — Hermes spends the entire
        4-breakpoint budget on conversation, not tool schemas.
        """
        cache_control = self._cache_control
        overrides: dict[str, Any] = {}

        # Keep the parent's model-settings nudge — Bedrock transport needs the
        # top-level kwarg to expand into block-level breakpoints correctly.
        overrides["model_settings"] = {
            **request.model_settings,
            "cache_control": cache_control,
        }

        # System message (1 breakpoint when present, content non-empty).
        if request.system_message is not None:
            new_sys = _tag_message_last_block(request.system_message, cache_control)
            if new_sys is not request.system_message:
                overrides["system_message"] = new_sys

        # Last 3 non-system messages (up to 3 breakpoints).
        if request.messages:
            non_system_indices = [
                i for i, m in enumerate(request.messages) if not isinstance(m, SystemMessage)
            ]
            target_indices = set(non_system_indices[-3:])
            if target_indices:
                new_messages: list[AnyMessage] = []
                for i, msg in enumerate(request.messages):
                    if i in target_indices:
                        new_messages.append(_tag_message_last_block(msg, cache_control))
                    else:
                        new_messages.append(msg)
                overrides["messages"] = new_messages

        return request.override(**overrides)

    # ``wrap_model_call`` and ``awrap_model_call`` inherit from the parent —
    # they already call ``_should_apply_caching`` and route to our overridden
    # ``_apply_caching`` correctly.

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if not self._should_apply_caching(request):
            return handler(request)
        return handler(self._apply_caching(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if not self._should_apply_caching(request):
            return await handler(request)
        return await handler(self._apply_caching(request))


__all__ = ["AnthropicCachingS3Middleware"]
