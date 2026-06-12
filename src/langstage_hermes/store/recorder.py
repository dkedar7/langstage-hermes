"""HermesStateRecorderMiddleware — persist every message to ``state.db``.

Hooks into the agent lifecycle:

- ``before_agent``: upsert the session row (so FTS5 search and the
  session_search tool can find this conversation even if it never
  finishes).
- ``after_model``: write the latest ``AIMessage``'s content (plus a
  JSON snapshot of any tool calls it issued).
- ``wrap_tool_call``: after the tool returns, write a ``role="tool"``
  row tagged with ``tool_name`` + ``tool_call_id``.
- ``after_agent``: close the session row with ``ended_at``,
  ``message_count``, ``tool_call_count``.

The recorder is intentionally permissive — it logs and swallows
``sqlite3.OperationalError`` rather than crashing the agent if state.db
is briefly unavailable (NFS, antivirus lock-up, etc.). Losing one
turn of recall history is preferable to a hard agent failure.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from langstage_hermes.store.sqlite_fts import SqliteFtsStore


def _take_last_str(_a: str | None, b: str | None) -> str | None:
    """Last-write-wins reducer for session lineage fields. Tolerates parallel
    writes when parent + subagent fire ``before_agent`` in the same superstep.
    """
    return b


class _RecorderStateExt(AgentState):
    """Schema extension so ``session_id`` / ``parent_session_id`` returned from
    ``before_agent`` actually propagate to subsequent hooks. Without this,
    LangGraph rejects unknown state keys and the recorder's manufactured
    ``session_id`` is silently dropped, leading to FK violations in
    ``record_message``.

    Reducer-annotated for safe parallel writes from subagent dispatches.
    """

    session_id: NotRequired[Annotated[str, _take_last_str]]
    parent_session_id: NotRequired[Annotated[str | None, _take_last_str]]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_session_id(state: Any) -> str:
    """Pull session_id out of state, manufacturing one if needed.

    HermesState declares a ``session_id`` field; in tests / minimal
    agents the state may be a bare dict where the field is missing.
    Returning a fresh uuid keeps the recorder usable in both cases.
    """
    if isinstance(state, dict):
        sid = state.get("session_id")
        if sid:
            return str(sid)
    else:
        sid = getattr(state, "session_id", None)
        if sid:
            return str(sid)
    return f"sess-{uuid.uuid4().hex[:12]}"


def _serialize_content(content: Any) -> str | None:
    """Convert LangChain message ``content`` to something searchable.

    Strings pass through; lists of content parts (multimodal /
    Anthropic content blocks) are flattened to their text portions
    because the FTS index only indexes a string column.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # Anthropic-style: {"type": "text", "text": "..."}
                if p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif "text" in p and isinstance(p["text"], str):
                    parts.append(p["text"])
        return "\n".join(parts) if parts else json.dumps(content)
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _serialize_tool_calls(message: AIMessage) -> list[dict[str, Any]] | None:
    """Pull tool calls off an AIMessage as plain JSON-serialisable dicts."""
    calls = getattr(message, "tool_calls", None) or []
    if not calls:
        return None
    out: list[dict[str, Any]] = []
    for c in calls:
        if isinstance(c, dict):
            out.append(
                {
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "args": c.get("args"),
                }
            )
        else:
            out.append(
                {
                    "id": getattr(c, "id", None),
                    "name": getattr(c, "name", None),
                    "args": getattr(c, "args", None),
                }
            )
    return out


def _message_id_key(msg: BaseMessage) -> str:
    """Stable per-message key used for dedupe across turns."""
    mid = getattr(msg, "id", None)
    if mid:
        return f"id:{mid}"
    # Fallback: hash content+type — sufficient for in-process dedupe.
    content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, default=str)
    return f"{type(msg).__name__}:{hash((content, getattr(msg, 'name', None)))}"


# ---------------------------------------------------------------------------
# middleware
# ---------------------------------------------------------------------------


class HermesStateRecorderMiddleware(AgentMiddleware):
    """Write every agent message to a ``SqliteFtsStore`` for later search.

    Constructor takes a store explicitly so tests can hand in a
    temp-dir-backed instance and so the agent factory stays in charge
    of where state.db lives.
    """

    state_schema = _RecorderStateExt

    def __init__(self, store: SqliteFtsStore) -> None:
        super().__init__()
        self.store = store
        # tools attribute is required by the AgentMiddleware contract.
        self.tools = []
        # Per-(thread, session) dedupe set so re-streaming a graph
        # state doesn't insert the same AIMessage twice.
        self._seen_message_keys: set[tuple[str, str]] = set()
        # Per-session tool-call ids we've already persisted.
        self._seen_tool_call_ids: set[tuple[str, str]] = set()

    # ── lifecycle ────────────────────────────────────────────────────

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        del runtime
        session_id = _resolve_session_id(state)
        source = os.environ.get("LANGSTAGE_HERMES_SESSION_SOURCE") or os.environ.get("DEEPAGENT_HERMES_SESSION_SOURCE", "user")
        parent = state.get("parent_session_id") if isinstance(state, dict) else getattr(state, "parent_session_id", None)
        try:
            self.store.ensure_session(
                session_id,
                source=source,
                parent_session_id=parent,
            )
        except sqlite3.OperationalError as exc:
            logger.warning("recorder.before_agent ensure_session failed: %s", exc)
        # Surface the session_id we used so downstream middleware / the
        # session_search tool can read it consistently.
        return {"session_id": session_id}

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        del runtime
        session_id = _resolve_session_id(state)
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", [])
        if not messages:
            return None
        # The latest message after a model call is the AIMessage.
        latest = messages[-1]
        if not isinstance(latest, AIMessage):
            return None
        key = (session_id, _message_id_key(latest))
        if key in self._seen_message_keys:
            return None
        self._seen_message_keys.add(key)
        try:
            self.store.record_message(
                session_id,
                "assistant",
                _serialize_content(latest.content),
                tool_calls=_serialize_tool_calls(latest),
                finish_reason=getattr(latest, "response_metadata", {}).get("finish_reason"),
            )
        except sqlite3.OperationalError as exc:
            logger.warning("recorder.after_model record_message failed: %s", exc)
        return None

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        result = handler(request)
        self._record_tool_result(request, result)
        return result

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        result = await handler(request)
        self._record_tool_result(request, result)
        return result

    def _record_tool_result(self, request: Any, result: Any) -> None:
        # ToolCallRequest exposes .state, .tool_call, .tool
        state = getattr(request, "state", None)
        session_id = _resolve_session_id(state)
        call = getattr(request, "tool_call", None) or {}
        tool_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        tool_call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        if tool_call_id:
            key = (session_id, tool_call_id)
            if key in self._seen_tool_call_ids:
                return
            self._seen_tool_call_ids.add(key)
        content: str | None
        if isinstance(result, ToolMessage):
            content = _serialize_content(result.content)
        elif hasattr(result, "update"):
            # Command — pull last ToolMessage from update.messages if present
            update = result.update or {}
            tool_msg: ToolMessage | None = None
            msgs = update.get("messages") if isinstance(update, dict) else None
            if msgs:
                for m in reversed(msgs):
                    if isinstance(m, ToolMessage):
                        tool_msg = m
                        break
            content = _serialize_content(tool_msg.content) if tool_msg else None
        else:
            content = _serialize_content(result)
        try:
            self.store.record_message(
                session_id,
                "tool",
                content,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )
        except sqlite3.OperationalError as exc:
            logger.warning("recorder.wrap_tool_call record_message failed: %s", exc)

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        del runtime
        session_id = _resolve_session_id(state)
        messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", [])
        message_count = len(messages or [])
        tool_call_count = sum(len(getattr(m, "tool_calls", []) or []) for m in (messages or []) if isinstance(m, AIMessage))
        try:
            self.store.end_session(
                session_id,
                end_reason="completed",
                message_count=message_count,
                tool_call_count=tool_call_count,
            )
        except sqlite3.OperationalError as exc:
            logger.warning("recorder.after_agent end_session failed: %s", exc)
        return None

    async def aafter_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)


__all__ = ["HermesStateRecorderMiddleware"]
