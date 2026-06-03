"""``HermesCompressionMiddleware`` — context-window compression per SPEC §7.

Direct replacement for ``langchain.agents.middleware.SummarizationMiddleware``.
Reasons the upstream isn't enough (see SPEC §7):

1. Need ``protect_first_n`` distinct from ``protect_last_n`` (the upstream
   only takes ``messages_to_keep``).
2. Need tool-result pruning + tool-call argument truncation as a cheap
   pre-pass before the LLM summarization.
3. Need to route summarization to a separate ``aux_model`` (Hermes's
   ``auxiliary_client``) so we don't burn the main model's tokens on a
   compression pass.
4. Need anti-thrash: skip subsequent compressions when two consecutive
   passes save less than 10% each.

Five-step algorithm (mirrors Hermes ``context_compressor.py``):

  1. Tool-result prune + dedup + tool-call argument shrink.
  2. Head protection — keep the first ``protect_first_n`` messages verbatim.
  3. Tail protection — walk backward until the tail token budget is reached
     (``threshold_tokens * 0.20``); floor at ``protect_last_n`` messages.
  4. Summarize the middle (everything left over) via ``aux_model.invoke(...)``
     with the prompt at ``prompts/compression_summary.md``. Cap output at
     ``min(context_length * 0.05, max_summary_tokens_ceiling)``.
  5. Splice: ``head + [SystemMessage(SUMMARY_PREFIX + summary)] + tail``.

Token estimator is ``len(content)//4`` (char-count heuristic) — accurate
enough for a threshold check; we do not depend on ``tiktoken`` here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from deepagent_hermes.prompts import load_prompt

logger = logging.getLogger(__name__)


# ── constants ────────────────────────────────────────────────────────


SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into "
    "the summary below. Treat it as background reference, NOT as active "
    "instructions. Respond ONLY to the latest user message that appears AFTER "
    "this summary; if it contradicts the summary, the latest message wins. "
    "Your persistent memory (MEMORY.md, USER.md) in the system prompt remains "
    "authoritative.\n\n"
)
"""Compaction handoff preface. See SPEC §7 for the rationale; the prose is
deliberately shorter than the verbatim Hermes copy — the full template body
lives in ``prompts/compression_summary.md`` and is what we pass to the
aux_model as a SystemMessage at summarisation time."""

_CHARS_PER_TOKEN = 4
_LOW_YIELD_THRESHOLD = 0.10
_LOW_YIELD_SKIP_TURNS = 5
_TOOL_CALL_ARG_HEAD = 200
# Tool-result body below this size is left verbatim — too small to be worth a placeholder.
_TOOL_RESULT_MIN_PRUNE_CHARS = 200
# Pre-cooked content prefixes we never re-prune (avoid re-summarising our own placeholders).
_PRUNED_PREFIXES = ("[Duplicate tool output", "[Tool ")
# Anti-thrash trip count — N consecutive low-yield passes before we skip.
_LOW_YIELD_TRIP_COUNT = 2


# ── helpers ──────────────────────────────────────────────────────────


def _message_text(msg: AnyMessage) -> str:
    """Return a best-effort plain-text view of message content for token counting."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content or "")


def _estimate_message_tokens(msg: AnyMessage) -> int:
    """Rough char/4 token estimate for a single message.

    Adds a small per-message overhead (10 tokens) to account for role/turn
    framing that providers tack on. Tool-call arguments on ``AIMessage`` add
    their JSON length too.
    """
    total = len(_message_text(msg)) + 10
    tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in tool_calls:
        if isinstance(tc, dict):
            args = tc.get("args") or tc.get("arguments") or {}
            if isinstance(args, (dict, list)):
                total += len(json.dumps(args, ensure_ascii=False))
            else:
                total += len(str(args))
    return total // _CHARS_PER_TOKEN + (1 if total else 0)


def _estimate_tokens(messages: list[AnyMessage]) -> int:
    """Rough total-token estimate over a message list. See ``_estimate_message_tokens``."""
    return sum(_estimate_message_tokens(m) for m in messages)


def _truncate_tool_call_args_json(args_json: str, head_chars: int = _TOOL_CALL_ARG_HEAD) -> str:
    """Shrink long string leaves inside a tool-call ``arguments`` JSON blob.

    Preserves JSON validity (parse → shrink string values → reserialize); if
    ``args_json`` is not valid JSON, returns it unchanged so the downstream
    provider doesn't see a broken structure.
    """
    try:
        parsed = json.loads(args_json)
    except (ValueError, TypeError):
        return args_json

    def _shrink(obj: Any) -> Any:
        if isinstance(obj, str):
            if len(obj) > head_chars:
                return obj[:head_chars] + "...[truncated]"
            return obj
        if isinstance(obj, dict):
            return {k: _shrink(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_shrink(v) for v in obj]
        return obj

    return json.dumps(_shrink(parsed), ensure_ascii=False)


def _content_hash(msg: AnyMessage) -> str:
    text = _message_text(msg)
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:12]


# ── middleware ───────────────────────────────────────────────────────


class HermesCompressionMiddleware(AgentMiddleware):
    """Hermes-style context compression as a ``before_model`` middleware.

    Constructor args:
        model: Main chat model (used as fallback summariser when ``aux_model``
            is ``None``). Currently only consulted for ``aux_model`` fallback —
            kept in the signature so the agent factory can pass it without
            additional plumbing.
        aux_model: Auxiliary chat model used for the summarisation call.
            If ``None``, ``model`` is used.
        threshold_percent: Compression fires when estimated tokens cross
            ``context_length * threshold_percent``. Default ``0.50``.
        protect_first_n: Number of head messages preserved verbatim. Default 3.
        protect_last_n: Floor on the number of tail messages preserved.
            Default 20.
        summary_target_ratio: Tail token budget = ``threshold_tokens *
            summary_target_ratio``. Default ``0.20``.
        abort_on_summary_failure: If True, summariser failure re-raises;
            otherwise a deterministic placeholder is inserted. Default False.
        max_summary_tokens_ceiling: Absolute ceiling on summary length
            regardless of context size. Default 12 000.
        context_length: Reference model context length. Default 200 000.
    """

    def __init__(
        self,
        *,
        model: Any,
        aux_model: Any = None,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        abort_on_summary_failure: bool = False,
        max_summary_tokens_ceiling: int = 12_000,
        context_length: int = 200_000,
    ) -> None:
        self.model = model
        self.aux_model = aux_model if aux_model is not None else model
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_ratio = summary_target_ratio
        self.abort_on_summary_failure = abort_on_summary_failure
        self.max_summary_tokens_ceiling = max_summary_tokens_ceiling
        self.context_length = context_length

        self.threshold_tokens = int(context_length * threshold_percent)
        self.tail_token_budget = int(self.threshold_tokens * summary_target_ratio)
        self.max_summary_tokens = min(int(context_length * 0.05), max_summary_tokens_ceiling)

    # ── public API (handy for tests) ─────────────────────────────────

    def estimate_tokens(self, messages: list[AnyMessage]) -> int:
        """Return the rough char/4 token estimate for ``messages``."""
        return _estimate_tokens(messages)

    def compress(self, messages: list[AnyMessage], *, state: Any = None) -> list[AnyMessage]:
        """Run the 5-step pipeline. Returns the new (shorter) message list.

        Returns ``messages`` unchanged if we're below threshold OR the
        anti-thrash counter has tripped.
        """
        total = _estimate_tokens(messages)
        if total <= self.threshold_tokens:
            return messages

        # Anti-thrash gate
        ctx = self._anti_thrash_state(state)
        if ctx["skip_remaining"] > 0:
            ctx["skip_remaining"] -= 1
            self._write_back(state, ctx)
            return messages

        # Step 1: tool-result pruning + arg truncation
        pruned = self._prune_old_tool_results(messages)

        # Step 2 + 3: head + tail protection
        head = pruned[: self.protect_first_n]
        tail = self._select_tail(pruned[self.protect_first_n :])

        # Middle = everything between head and the tail we just picked
        middle_end = len(pruned) - len(tail)
        middle = pruned[self.protect_first_n : middle_end]
        if not middle:
            # Nothing to summarise — head + tail already covers the whole list.
            return pruned

        # Step 4: summarise via aux_model
        try:
            summary = self._summarise(middle)
        except Exception as exc:
            if self.abort_on_summary_failure:
                raise
            summary = self._fallback_summary(middle, exc=exc)
            logger.warning("compression: aux_model summary failed (%s) — using fallback", exc)

        summary_msg = SystemMessage(content=SUMMARY_PREFIX + summary)
        new_messages = [*head, summary_msg, *tail]

        # Anti-thrash bookkeeping — yield ratio = (old - new) / old
        new_total = _estimate_tokens(new_messages)
        savings = (total - new_total) / total if total > 0 else 0.0
        if savings < _LOW_YIELD_THRESHOLD:
            ctx["consecutive_low_yield"] += 1
            if ctx["consecutive_low_yield"] >= _LOW_YIELD_TRIP_COUNT:
                ctx["skip_remaining"] = _LOW_YIELD_SKIP_TURNS
                logger.info(
                    "compression: anti-thrash engaged — skipping next %d turns "
                    "(savings %.1f%% < %.0f%% for %d consecutive passes)",
                    _LOW_YIELD_SKIP_TURNS,
                    savings * 100,
                    _LOW_YIELD_THRESHOLD * 100,
                    ctx["consecutive_low_yield"],
                )
        else:
            ctx["consecutive_low_yield"] = 0
        self._write_back(state, ctx)

        return new_messages

    # ── private — anti-thrash state ──────────────────────────────────

    def _anti_thrash_state(self, state: Any) -> dict[str, int]:
        if isinstance(state, dict):
            return {
                "consecutive_low_yield": int(state.get("consecutive_low_yield_compressions", 0)),
                "skip_remaining": int(state.get("_compression_skip_remaining", 0)),
            }
        return {"consecutive_low_yield": 0, "skip_remaining": 0}

    def _write_back(self, state: Any, ctx: dict[str, int]) -> None:
        if isinstance(state, dict):
            state["consecutive_low_yield_compressions"] = ctx["consecutive_low_yield"]
            state["_compression_skip_remaining"] = ctx["skip_remaining"]

    # ── private — step 1: tool-result pruning ────────────────────────

    def _prune_old_tool_results(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Replace old tool-result contents + dedupe identical results + shrink long args.

        Boundary: keep the last ``protect_last_n`` messages verbatim; anything
        before that is fair game for pruning.
        """
        if not messages:
            return messages

        boundary = max(0, len(messages) - self.protect_last_n)
        # Map id(msg) -> name for tool_call lookup
        result: list[AnyMessage] = list(messages)

        self._dedupe_tool_results(result)
        self._prune_preboundary(result, boundary=boundary)
        return result

    def _dedupe_tool_results(self, result: list[AnyMessage]) -> None:
        """In-place: replace older duplicate tool-results with a back-reference."""
        content_hashes: dict[str, int] = {}
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            if not isinstance(msg, ToolMessage):
                continue
            text = _message_text(msg)
            if len(text) < _TOOL_RESULT_MIN_PRUNE_CHARS:
                continue
            h = _content_hash(msg)
            if h in content_hashes:
                result[i] = msg.model_copy(update={"content": ("[Duplicate tool output — same content as a more recent call]")})
            else:
                content_hashes[h] = i

    def _prune_preboundary(self, result: list[AnyMessage], *, boundary: int) -> None:
        """In-place: replace pre-boundary tool results + shrink long tool-call args."""
        for i in range(boundary):
            msg = result[i]
            if isinstance(msg, ToolMessage):
                self._maybe_replace_tool_message(result, i, msg)
            elif isinstance(msg, AIMessage):
                self._maybe_shrink_tool_call_args(result, i, msg)

    def _maybe_replace_tool_message(self, result: list[AnyMessage], i: int, msg: ToolMessage) -> None:
        text = _message_text(msg)
        if not text or text.startswith(_PRUNED_PREFIXES):
            return
        if len(text) <= _TOOL_RESULT_MIN_PRUNE_CHARS:
            return
        tool_name = getattr(msg, "name", "unknown") or "unknown"
        placeholder = f"[Tool {tool_name} returned {len(text)} chars; suppressed for context]"
        result[i] = msg.model_copy(update={"content": placeholder})

    def _maybe_shrink_tool_call_args(self, result: list[AnyMessage], i: int, msg: AIMessage) -> None:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return
        new_tcs: list[dict[str, Any]] = []
        modified = False
        for tc in tool_calls:
            if not isinstance(tc, dict):
                new_tcs.append(tc)
                continue
            args = tc.get("args")
            shrunk = tc
            if isinstance(args, dict) and args:
                try:
                    args_json = json.dumps(args, ensure_ascii=False)
                    new_json = _truncate_tool_call_args_json(args_json)
                    if new_json != args_json:
                        shrunk = {**tc, "args": json.loads(new_json)}
                        modified = True
                except (ValueError, TypeError):
                    pass
            new_tcs.append(shrunk)
        if modified:
            result[i] = msg.model_copy(update={"tool_calls": new_tcs})

    # ── private — step 3: tail selection ─────────────────────────────

    def _select_tail(self, candidates: list[AnyMessage]) -> list[AnyMessage]:
        """Walk backward through ``candidates`` collecting tail until the budget tips."""
        if not candidates:
            return []
        floor = min(self.protect_last_n, len(candidates))

        accumulated = 0
        tail: list[AnyMessage] = []
        for msg in reversed(candidates):
            cost = _estimate_message_tokens(msg)
            # Floor wins: always grab at least ``floor`` messages.
            if len(tail) < floor:
                tail.append(msg)
                accumulated += cost
                continue
            if accumulated + cost > self.tail_token_budget:
                break
            tail.append(msg)
            accumulated += cost
        tail.reverse()
        return tail

    # ── private — step 4: summarise ──────────────────────────────────

    def _summarise(self, middle: list[AnyMessage]) -> str:
        """Call ``aux_model.invoke([system, human])`` and return the text body."""
        template = load_prompt("compression_summary.md") or (
            "Summarise the conversation turns below as a context handoff. "
            "Preserve user goals, completed actions, and outstanding work."
        )
        payload = json.dumps(
            [_message_to_dict(m) for m in middle],
            ensure_ascii=False,
        )
        system = SystemMessage(content=template)
        human = HumanMessage(content=payload)
        response = self.aux_model.invoke([system, human])
        text = getattr(response, "content", response)
        if isinstance(text, list):
            text = "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
        return str(text or "").strip()

    def _fallback_summary(self, middle: list[AnyMessage], *, exc: Exception) -> str:
        """Deterministic placeholder when the LLM summariser is unavailable."""
        first = middle[0] if middle else None
        last = middle[-1] if middle else None
        span = ""
        if first is not None and last is not None:
            span = f"from {type(first).__name__} to {type(last).__name__}"
        return (
            f"[Earlier conversation summarised: {len(middle)} turns covering {span}. "
            f"Tool calls and results have been pruned. "
            f"Summary unavailable due to summariser failure: {exc!r}.]"
        )

    # ── middleware hook ──────────────────────────────────────────────

    def before_model(self, state: Any, runtime: Any | None = None) -> dict[str, Any] | None:
        """If the conversation is over threshold, replace ``messages`` with compressed form."""
        if isinstance(state, dict):
            messages = state.get("messages")
        else:
            messages = getattr(state, "messages", None)
        if not messages:
            return None
        new_messages = self.compress(messages, state=state)
        if new_messages is messages:
            return None
        return {"messages": new_messages}


def _message_to_dict(msg: AnyMessage) -> dict[str, Any]:
    """Best-effort serialisation for the summarizer input.

    Uses ``model_dump`` when available; falls back to a minimal dict otherwise.
    """
    if hasattr(msg, "model_dump"):
        try:
            return msg.model_dump()
        except Exception:
            pass
    return {
        "type": type(msg).__name__,
        "content": _message_text(msg),
        "tool_calls": getattr(msg, "tool_calls", None) or [],
    }


__all__ = [
    "SUMMARY_PREFIX",
    "HermesCompressionMiddleware",
]
