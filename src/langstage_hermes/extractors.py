"""Tool-result extractors for ``langstage-core``.

These three extractors surface langstage-hermes runtime events as typed
``ToolExtractedEvent``s in any host UI built on the parser. They follow the
``langstage_core.extractors.base.ToolExtractor`` protocol verbatim
so they can be upstreamed to the parser's built-in extractor set (target PR
to dkedar7/langstage-core).

Until upstreamed, hosts can register them manually::

    from langstage_core import StreamParser
    from langstage_hermes.extractors import (
        SkillManageExtractor, CompressionExtractor, MemoryExtractor,
    )

    parser = StreamParser(extractors=[
        SkillManageExtractor(),
        CompressionExtractor(),
        MemoryExtractor(),
    ])
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
