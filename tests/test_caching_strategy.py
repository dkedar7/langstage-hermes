"""Tests for ``AnthropicCachingS3Middleware`` — Hermes ``system_and_3`` strategy.

Verifies (SPEC §6):

* No-op when the model isn't Anthropic.
* 4 ``cache_control`` markers placed on a long conversation (system + last 3).
* 1 marker on a single-message conversation.
* TTL is reflected in the marker.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelRequest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepagent_hermes.caching import AnthropicCachingS3Middleware

# ── helpers ──────────────────────────────────────────────────────────


class _StubChatAnthropic(ChatAnthropic):
    """ChatAnthropic stub that skips actual SDK init (no API key needed)."""

    def __init__(self):
        # Pydantic init via parent — pass minimal required fields.
        super().__init__(model="claude-sonnet-4-5", anthropic_api_key="sk-stub-not-used")


def _make_request(messages: list, *, model=None, system_message: SystemMessage | None = None):
    return ModelRequest(
        model=model if model is not None else _StubChatAnthropic(),
        messages=messages,
        system_message=system_message,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": messages},
        runtime=None,
        model_settings={},
    )


def _count_cache_markers(messages, system_message) -> tuple[int, list[dict]]:
    """Return (count_of_cache_control_blocks, list_of_marker_dicts).

    Walks the (possibly tagged) system message + each message's content.
    """
    found: list[dict] = []
    for msg in [system_message, *messages] if system_message is not None else messages:
        if msg is None:
            continue
        content = msg.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    found.append(block["cache_control"])
    return len(found), found


# ── tests ────────────────────────────────────────────────────────────


def test_six_message_conversation_gets_four_markers():
    """6 non-system messages + a system prompt -> 4 markers (system + last 3)."""
    mw = AnthropicCachingS3Middleware(ttl="5m")
    messages = [
        HumanMessage(content="msg 1"),
        AIMessage(content="msg 2"),
        HumanMessage(content="msg 3"),
        AIMessage(content="msg 4"),
        HumanMessage(content="msg 5"),
        AIMessage(content="msg 6"),
    ]
    system = SystemMessage(content="you are a helpful agent")
    request = _make_request(messages, system_message=system)

    new_request = mw._apply_caching(request)
    count, markers = _count_cache_markers(new_request.messages, new_request.system_message)
    assert count == 4
    # All markers ephemeral, no TTL override on 5m default
    assert all(m["type"] == "ephemeral" for m in markers)
    assert all("ttl" not in m or m["ttl"] == "5m" for m in markers)

    # Only the LAST 3 user/assistant messages should be tagged
    tagged_indices = [
        i
        for i, msg in enumerate(new_request.messages)
        if isinstance(msg.content, list)
        and any(isinstance(b, dict) and "cache_control" in b for b in msg.content)
    ]
    assert tagged_indices == [3, 4, 5]


def test_single_message_conversation_gets_two_markers():
    """1 message + system -> system + 1 message = 2 markers (the spec said
    'with 1 message, assert only 1 marker' — but a system prompt is also a
    cache target, so we hit 2 when system is present; with no system, 1)."""
    mw = AnthropicCachingS3Middleware(ttl="5m")
    request_no_sys = _make_request([HumanMessage(content="hello")])
    new_req = mw._apply_caching(request_no_sys)
    count, _ = _count_cache_markers(new_req.messages, new_req.system_message)
    assert count == 1  # just the lone message


def test_one_message_with_system_gets_two_markers():
    mw = AnthropicCachingS3Middleware(ttl="5m")
    request = _make_request(
        [HumanMessage(content="hello")],
        system_message=SystemMessage(content="be brief"),
    )
    new_req = mw._apply_caching(request)
    count, _ = _count_cache_markers(new_req.messages, new_req.system_message)
    assert count == 2  # system + 1 message


def test_ttl_1h_propagates_to_marker():
    mw = AnthropicCachingS3Middleware(ttl="1h")
    messages = [HumanMessage(content="msg 1"), AIMessage(content="msg 2")]
    request = _make_request(messages, system_message=SystemMessage(content="sys"))
    new_req = mw._apply_caching(request)
    _, markers = _count_cache_markers(new_req.messages, new_req.system_message)
    assert markers, "expected at least one cache_control marker"
    assert all(m["ttl"] == "1h" for m in markers)


def test_no_op_on_non_anthropic_model():
    """If the model isn't ChatAnthropic, ``_should_apply_caching`` returns False
    and ``wrap_model_call`` calls the handler with the unmodified request."""
    mw = AnthropicCachingS3Middleware(ttl="5m", unsupported_model_behavior="ignore")
    fake_non_anthropic = MagicMock()
    fake_non_anthropic.__class__ = type("NotAnthropic", (), {})
    request = _make_request(
        [HumanMessage(content="hi")],
        model=fake_non_anthropic,
        system_message=SystemMessage(content="sys"),
    )

    captured: dict[str, Any] = {}

    def handler(req):
        captured["request"] = req
        return MagicMock(name="ModelResponse")

    mw.wrap_model_call(request, handler)
    captured_req = captured["request"]
    # No cache_control blocks should have been added.
    sys_content = captured_req.system_message.content
    assert isinstance(sys_content, str)  # untouched -- still a raw string


def test_short_circuits_when_below_min_messages_to_cache():
    mw = AnthropicCachingS3Middleware(ttl="5m", min_messages_to_cache=10)
    request = _make_request(
        [HumanMessage(content="lonely")],
        system_message=SystemMessage(content="sys"),
    )

    captured: dict[str, Any] = {}

    def handler(req):
        captured["request"] = req
        return MagicMock(name="ModelResponse")

    mw.wrap_model_call(request, handler)
    req = captured["request"]
    assert isinstance(req.system_message.content, str), (
        "system message should not have been tagged when below min_messages_to_cache"
    )
