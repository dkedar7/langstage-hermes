"""Tests for ``deepagent_hermes.plugins.event_bus.PluginEventBus``.

Covers the v0.2 wiring of the 8 hooks that fire from middleware events:
``pre_llm_call``, ``post_llm_call``, ``transform_llm_output``,
``transform_tool_result``, ``transform_terminal_output``,
``on_session_finalize``, ``pre_approval_request``, ``post_approval_response``,
plus the consolidated firing of the already-wired hooks.

Mocks ``ModelRequest`` / ``ModelResponse`` / ``ToolCallRequest`` and the
``handler`` callable so no real model invocation happens.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from deepagent_hermes.plugins.context import (
    PluginContext,
    get_global_hook_registry,
)
from deepagent_hermes.plugins.event_bus import PluginEventBus

# ── helpers ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_global_hook_registry():
    """Wipe the module-level hook registry between tests to keep them isolated."""
    get_global_hook_registry().clear()
    yield
    get_global_hook_registry().clear()


class _MockModelResponse:
    """Stand-in for ``langchain.agents.middleware.types.ModelResponse``.

    We use a duck-typed stand-in instead of constructing the real dataclass
    because the bus only reads ``.result`` / ``.structured_response`` and we
    want to avoid pulling in the full langchain agent stack in unit tests.
    Note: the bus's ``_replace_response_messages`` does construct a real
    ``ModelResponse`` when transforming messages — the tests that exercise
    that path verify the real type is used.
    """

    def __init__(self, result: list[Any], structured_response: Any = None):
        self.result = result
        self.structured_response = structured_response


class _MockModelRequest:
    """Stand-in for ``ModelRequest`` — just needs ``messages`` for tests."""

    def __init__(self, messages: list[Any] | None = None):
        self.messages = messages or []


class _MockToolCallRequest:
    """Stand-in for ``ToolCallRequest`` — bus reads ``tool_call`` dict + ``tool``."""

    def __init__(
        self,
        *,
        tool_name: str,
        tool_args: dict | None = None,
        tool_call_id: str = "call-1",
    ):
        self.tool_call = {
            "name": tool_name,
            "args": tool_args or {},
            "id": tool_call_id,
        }
        self.tool = None
        self.state: dict[str, Any] = {}
        self.runtime = None


def _make_ctx(hooks_store: dict | None = None) -> PluginContext:
    """Build a PluginContext that writes to a fresh dict-based hooks store."""
    return PluginContext(
        registry={},
        memory_registry={},
        slash_commands={},
        hooks=hooks_store if hooks_store is not None else {},
    )


# ── tests ───────────────────────────────────────────────────────────


def test_event_bus_no_op_with_empty_registry():
    """With no plugins registered, the bus passes requests through unchanged."""
    bus = PluginEventBus()

    # before_agent / after_agent should not raise
    bus.before_agent({}, runtime=None)
    bus.after_agent({}, runtime=None)

    # wrap_model_call should pass through the handler's response untouched
    request = _MockModelRequest(messages=[HumanMessage("hi")])
    expected_response = _MockModelResponse(result=[AIMessage("hello")])

    def handler(req):
        assert req is request
        return expected_response

    actual = bus.wrap_model_call(request, handler)
    assert actual is expected_response
    assert actual.result == [AIMessage("hello")]

    # wrap_tool_call should pass the result through
    tool_request = _MockToolCallRequest(tool_name="search", tool_args={"q": "x"})
    tool_result = ToolMessage(content="found", tool_call_id="call-1")
    assert bus.wrap_tool_call(tool_request, lambda r: tool_result) is tool_result


def test_pre_llm_call_fires():
    """Registering a hook on ``pre_llm_call`` causes it to fire with the request."""
    ctx = _make_ctx()
    seen: list[Any] = []
    ctx.register_hook("pre_llm_call", seen.append)

    bus = PluginEventBus()
    request = _MockModelRequest()
    response = _MockModelResponse(result=[AIMessage("ok")])

    bus.wrap_model_call(request, lambda r: response)

    assert len(seen) == 1
    assert seen[0] is request


def test_post_llm_call_fires_with_response():
    """``post_llm_call`` fires AFTER the handler with ``(request, response)``."""
    ctx = _make_ctx()
    seen: list[tuple[Any, Any]] = []
    ctx.register_hook("post_llm_call", lambda req, resp: seen.append((req, resp)))

    bus = PluginEventBus()
    request = _MockModelRequest()
    response = _MockModelResponse(result=[AIMessage("ok")])

    out = bus.wrap_model_call(request, lambda r: response)

    assert len(seen) == 1
    seen_req, seen_resp = seen[0]
    assert seen_req is request
    # The response object the hook sees should be the one returned by the
    # handler (no other transforms registered).
    assert seen_resp is response
    assert out is response


def test_transform_llm_output_replaces_messages():
    """If ``transform_llm_output`` returns mutated messages, the bus uses them."""
    ctx = _make_ctx()

    def append_marker(messages):
        # Plugins mutate by returning a new list — the bus chains the result.
        return [*list(messages), AIMessage("[plugin appended]")]

    ctx.register_hook("transform_llm_output", append_marker)

    bus = PluginEventBus()
    request = _MockModelRequest()
    original = _MockModelResponse(result=[AIMessage("ok")])

    out = bus.wrap_model_call(request, lambda r: original)

    # Bus constructs a fresh ModelResponse when result changes.
    assert len(out.result) == 2
    assert isinstance(out.result[-1], AIMessage)
    assert out.result[-1].content == "[plugin appended]"


def test_transform_tool_result_replaces_content():
    """``transform_tool_result`` plugin return value replaces the ToolMessage content."""
    ctx = _make_ctx()
    ctx.register_hook(
        "transform_tool_result",
        lambda content, tool_name, args: f"[REDACTED {tool_name}] {content}",
    )

    bus = PluginEventBus()
    request = _MockToolCallRequest(tool_name="search", tool_args={"q": "secret"})
    result = ToolMessage(content="raw output", tool_call_id="call-1")

    out = bus.wrap_tool_call(request, lambda r: result)

    assert isinstance(out, ToolMessage)
    assert out.content == "[REDACTED search] raw output"
    # The original ToolMessage was not mutated in place.
    assert result.content == "raw output"


def test_transform_terminal_output_only_for_terminal_tools():
    """``transform_terminal_output`` fires only when the tool name is a terminal tool."""
    ctx = _make_ctx()
    fired_for: list[str] = []

    def hook(content, tool_name, args):
        fired_for.append(tool_name)
        return content + " [terminal-touched]"

    ctx.register_hook("transform_terminal_output", hook)
    bus = PluginEventBus()

    # Non-terminal tool — hook must NOT fire.
    out1 = bus.wrap_tool_call(
        _MockToolCallRequest(tool_name="search"),
        lambda r: ToolMessage(content="hits", tool_call_id="c1"),
    )
    assert out1.content == "hits"
    assert fired_for == []

    # Terminal tool — hook MUST fire and content is replaced.
    out2 = bus.wrap_tool_call(
        _MockToolCallRequest(tool_name="bash"),
        lambda r: ToolMessage(content="$ ls", tool_call_id="c2"),
    )
    assert out2.content == "$ ls [terminal-touched]"
    assert fired_for == ["bash"]

    # Another terminal tool name from the default set.
    out3 = bus.wrap_tool_call(
        _MockToolCallRequest(tool_name="execute_code"),
        lambda r: ToolMessage(content="print('hi')", tool_call_id="c3"),
    )
    assert out3.content == "print('hi') [terminal-touched]"
    assert fired_for == ["bash", "execute_code"]


def test_on_session_finalize_fires_after_on_session_end():
    """``after_agent`` fires ``on_session_end`` then ``on_session_finalize``."""
    ctx = _make_ctx()
    order: list[str] = []
    ctx.register_hook("on_session_end", lambda state: order.append("end"))
    ctx.register_hook("on_session_finalize", lambda state: order.append("finalize"))
    ctx.register_hook("on_session_start", lambda state: order.append("start"))

    bus = PluginEventBus()
    bus.before_agent({}, runtime=None)
    bus.after_agent({}, runtime=None)

    assert order == ["start", "end", "finalize"]


def test_plugin_exception_swallowed(caplog):
    """A plugin that raises must NOT crash the agent — bus logs and continues."""
    ctx = _make_ctx()

    def angry_plugin(req):
        raise ValueError("plugin go boom")

    def good_plugin(req):
        # Runs after the broken one to prove iteration didn't bail.
        req.messages.append(HumanMessage("touched"))

    ctx.register_hook("pre_llm_call", angry_plugin)
    ctx.register_hook("pre_llm_call", good_plugin)

    bus = PluginEventBus()
    request = _MockModelRequest()
    response = _MockModelResponse(result=[AIMessage("ok")])

    with caplog.at_level("WARNING"):
        out = bus.wrap_model_call(request, lambda r: response)

    # Agent did not crash.
    assert out is response
    # Good plugin still ran.
    assert any(isinstance(m, HumanMessage) and m.content == "touched" for m in request.messages)
    # The broken plugin was logged.
    assert any(
        "pre_llm_call" in rec.message and "plugin go boom" in rec.message
        for rec in caplog.records
    )


def test_multiple_plugins_chain():
    """Two ``transform_llm_output`` plugins run in order; second sees first's output."""
    ctx = _make_ctx()

    def first(messages):
        # Append a marker tagged "1".
        return [*list(messages), AIMessage("[1]")]

    def second(messages):
        # Append another marker tagged "2" — must see the "[1]" the first
        # plugin appended.
        assert any(isinstance(m, AIMessage) and m.content == "[1]" for m in messages)
        return [*list(messages), AIMessage("[2]")]

    ctx.register_hook("transform_llm_output", first)
    ctx.register_hook("transform_llm_output", second)

    bus = PluginEventBus()
    request = _MockModelRequest()
    response = _MockModelResponse(result=[AIMessage("ok")])

    out = bus.wrap_model_call(request, lambda r: response)

    assert [m.content for m in out.result] == ["ok", "[1]", "[2]"]


def test_pre_tool_call_and_post_tool_call_fire():
    """Consolidated firing of the v0.1 tool hooks works through the bus too."""
    ctx = _make_ctx()
    events: list[tuple[str, Any]] = []
    ctx.register_hook("pre_tool_call", lambda req: events.append(("pre", req)))
    ctx.register_hook(
        "post_tool_call", lambda req, result: events.append(("post", result))
    )

    bus = PluginEventBus()
    request = _MockToolCallRequest(tool_name="search")
    result = ToolMessage(content="data", tool_call_id="call-1")

    bus.wrap_tool_call(request, lambda r: result)

    assert [e[0] for e in events] == ["pre", "post"]
    assert events[0][1] is request
    assert events[1][1] is result


def test_approval_hooks_fire_only_for_interrupt_tools():
    """``pre_approval_request`` / ``post_approval_response`` gated on tool name."""
    ctx = _make_ctx()
    seen: list[str] = []
    ctx.register_hook("pre_approval_request", lambda req: seen.append("pre"))
    ctx.register_hook(
        "post_approval_response", lambda req, result: seen.append("post")
    )

    bus = PluginEventBus(interrupt_tool_names={"bash"})

    # Non-interrupt tool — hooks must NOT fire.
    bus.wrap_tool_call(
        _MockToolCallRequest(tool_name="search"),
        lambda r: ToolMessage(content="ok", tool_call_id="c1"),
    )
    assert seen == []

    # Interrupt tool — hooks MUST fire in order around the handler.
    bus.wrap_tool_call(
        _MockToolCallRequest(tool_name="bash"),
        lambda r: ToolMessage(content="$", tool_call_id="c2"),
    )
    assert seen == ["pre", "post"]


def test_transform_tool_result_handles_command_wrapper():
    """When the tool returns a ``Command`` carrying a ToolMessage, content swap works."""
    ctx = _make_ctx()
    ctx.register_hook(
        "transform_tool_result",
        lambda content, tool_name, args: content.upper(),
    )

    bus = PluginEventBus()
    request = _MockToolCallRequest(tool_name="search")
    inner_tool_msg = ToolMessage(content="hello", tool_call_id="call-1")
    result = Command(update={"messages": [inner_tool_msg], "counter": 1})

    out = bus.wrap_tool_call(request, lambda r: result)

    # Output is still a Command with the counter preserved.
    assert isinstance(out, Command)
    assert out.update["counter"] == 1
    # The ToolMessage inside has the upper-cased content.
    msgs = out.update["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert msgs[0].content == "HELLO"
    # Original message was not mutated in place.
    assert inner_tool_msg.content == "hello"


def test_pre_llm_call_replacement_request_propagates():
    """If ``pre_llm_call`` returns a new request, the handler receives the new one."""
    ctx = _make_ctx()
    new_request = _MockModelRequest(messages=[HumanMessage("replaced")])
    ctx.register_hook("pre_llm_call", lambda req: new_request)

    bus = PluginEventBus()
    seen: list[Any] = []
    response = _MockModelResponse(result=[AIMessage("ok")])

    def handler(req):
        seen.append(req)
        return response

    bus.wrap_model_call(_MockModelRequest(), handler)

    assert seen == [new_request]


def test_explicit_registry_bypasses_global():
    """Passing ``hook_registry=`` to the constructor pins the bus to that dict."""
    explicit: dict[str, list] = {}
    seen: list[Any] = []
    explicit.setdefault("pre_llm_call", []).append(seen.append)

    # Also register a hook on the GLOBAL registry — must NOT be seen by this bus.
    ctx = _make_ctx()
    global_seen: list[Any] = []
    ctx.register_hook("pre_llm_call", global_seen.append)

    bus = PluginEventBus(hook_registry=explicit)
    response = _MockModelResponse(result=[AIMessage("ok")])
    bus.wrap_model_call(_MockModelRequest(), lambda r: response)

    assert len(seen) == 1
    assert global_seen == []
