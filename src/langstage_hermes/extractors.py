"""Tool-result extractors for ``langstage-core``'s AG-UI frame stream.

The four extractors in :data:`ALL_EXTRACTORS` (``SkillManageExtractor``,
``SkillViewExtractor``, ``CompressionExtractor``, ``MemoryExtractor``) turn
langstage-hermes tool results (skill writes and views, context compression,
memory updates) into typed ``{"type": "extraction", ...}`` frames that any host
UI built on ``langstage-core`` can render. They follow the
``langstage_core.extractors.base.ToolExtractor`` protocol (``tool_name`` /
``extracted_type`` / ``extract(content)``). ``langstage-core`` also ships
equivalents among its built-in extractors; hermes wires these copies so the
payloads stay in step with its own tools.

Register them by passing instances to ``iter_event_frames``, exactly as
``langstage_hermes.agui_stream`` does (``graph`` is a compiled hermes agent,
e.g. from ``create_hermes_agent``)::

    from langstage_core.agui import build_agent, iter_event_frames

    from langstage_hermes.extractors import ALL_EXTRACTORS

    extractors = [cls() for cls in ALL_EXTRACTORS]


    async def stream(graph, message, thread_id):
        agent = build_agent(graph)
        async for frame in iter_event_frames(agent, message, thread_id, extractors=extractors):
            if frame["type"] == "extraction":
                print(frame["extracted_type"], frame["data"])
"""

from __future__ import annotations

import json
from typing import Any, ClassVar


def _parse_json_content(content: Any) -> dict[str, Any] | None:
    """Best-effort JSON parse — accepts str, dict, or returns None."""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


class SkillManageExtractor:
    """Surface skill_manage / skill_view actions as inline events.

    Emits ``ToolExtractedEvent(tool_name="skill_manage", extracted_type="skill_created"
    | "skill_updated" | "skill_deleted" | "skill_viewed", data={"name", "action"})``.

    Hosts render these as "🧠 skill created: pdf-merging" inline in the chat
    timeline, separate from the raw tool-result text. The agent's compounding
    knowledge gets a visible feedback loop — which is the whole point of the
    Hermes reflection design.
    """

    tool_name = "skill_manage"
    extracted_type = "skill_event"

    _ACTION_TO_TYPE: ClassVar[dict[str, str]] = {
        "create": "skill_created",
        "patch": "skill_updated",
        "write_file": "skill_updated",
        "delete": "skill_deleted",
        "pin": "skill_updated",
        "unpin": "skill_updated",
    }

    def extract(self, content: Any) -> dict[str, Any] | None:
        # The skill_manage tool's Command may serialize as the message content;
        # we try multiple shapes that the parser may hand us.
        parsed = _parse_json_content(content)
        if parsed is None:
            # Fall back to scanning the tool's plain-text reply.
            if not isinstance(content, str):
                return None
            text = content.lower()
            for action, etype in self._ACTION_TO_TYPE.items():
                if action in text:
                    # Best-effort name extraction.
                    return {"action": action, "extracted_subtype": etype}
            return None

        action = parsed.get("action")
        name = parsed.get("name")
        if action is None or name is None:
            return None
        etype = self._ACTION_TO_TYPE.get(action, "skill_event")
        return {"action": action, "name": name, "extracted_subtype": etype}


class SkillViewExtractor:
    """Surface skill_view tool calls.

    Emits a separate ``skill_loaded`` event when the agent decides to load
    a skill body into its prompt. Hosts can render this differently from
    creation/update — it's a read, not a mutation.
    """

    tool_name = "skill_view"
    extracted_type = "skill_loaded"

    def extract(self, content: Any) -> dict[str, Any] | None:
        # skill_view returns the body as the tool result; we don't try to
        # parse it. We only care that it ran; the parser will emit our event
        # with whatever data we return (None means no event).
        if not content:
            return None
        return {"loaded": True, "body_chars": len(str(content))}


class CompressionExtractor:
    """Surface context-compression events.

    The compression middleware emits a ``__compression__`` synthetic tool message
    when it runs (see ``HermesCompressionMiddleware``). This extractor pulls out
    the compression ratio and section count so the UI can show a banner like
    "context compressed: 47k → 9k tokens (5x)".
    """

    tool_name = "__compression__"
    extracted_type = "compression_summary"

    def extract(self, content: Any) -> dict[str, Any] | None:
        parsed = _parse_json_content(content)
        if parsed is None:
            return None
        # Expected shape (best-effort — fields may be missing):
        # {"before_tokens": int, "after_tokens": int, "ratio": float,
        #  "section_count": int, "skipped": bool, "reason": str}
        keys = {"before_tokens", "after_tokens", "ratio", "section_count", "skipped", "reason"}
        out = {k: parsed[k] for k in keys if k in parsed}
        return out or None


class MemoryExtractor:
    """Surface memory tool actions.

    Emits ``memory_updated`` events when the agent writes to MEMORY.md or
    USER.md. Distinguishes target (``memory`` vs ``user``) so hosts can
    render the two streams separately.
    """

    tool_name = "memory"
    extracted_type = "memory_updated"

    _ACTION_TO_TYPE: ClassVar[dict[str, str]] = {
        "add": "memory_added",
        "replace": "memory_replaced",
        "remove": "memory_removed",
        "read": "memory_read",
    }

    def extract(self, content: Any) -> dict[str, Any] | None:
        parsed = _parse_json_content(content)
        if parsed is None:
            return None
        action = parsed.get("action")
        target = parsed.get("target")
        if action is None or target is None:
            return None
        etype = self._ACTION_TO_TYPE.get(action, "memory_updated")
        out = {"action": action, "target": target, "extracted_subtype": etype}
        if "index" in parsed:
            out["index"] = parsed["index"]
        return out


ALL_EXTRACTORS = (
    SkillManageExtractor,
    SkillViewExtractor,
    CompressionExtractor,
    MemoryExtractor,
)
