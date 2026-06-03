"""``session_search`` — long-term conversation recall over SQLite + FTS5.

Single tool, three calling modes inferred from which args are set
(matches Hermes's session_search behaviour byte-for-byte except for
the markdown output shape — Hermes returns JSON, we return rich
markdown because LLMs read it more reliably than json strings):

* **DISCOVERY** — pass ``query``. Runs FTS5 (BM25) over the message
  index, dedupes hits by session lineage, and returns the top 10
  sessions each with: snippet, ±5-message window around the match,
  ``bookend_start`` (first 3 user/assistant messages of the session)
  and ``bookend_end`` (last 3). Detects CJK characters and routes to
  the trigram FTS table when present.
* **SCROLL** — pass ``session_id`` + ``around_message_id``. Returns
  a ±``window`` window centred on the anchor. ``window`` is clamped
  to ``[1, 20]``. We REJECT scrolling inside the active session
  lineage so the agent can't waste context re-reading messages it
  already has.
* **BROWSE** — no args. Lists the 20 most-recently-active root
  sessions (chronological, newest first).

``sources_exclude`` defaults to ``["tool"]`` so reflection-fork
sessions (tagged ``HERMES_SESSION_SOURCE=tool``) don't pollute the
user's recall surface. The lineage of ``current_session_id`` is also
ALWAYS filtered out — those messages are already in context.

This module exposes a factory ``make_session_search_tool(store)``
that returns a LangChain ``BaseTool``. The agent factory binds the
store at build time so the tool doesn't need to reach into runtime to
find it. The current session id can be passed explicitly per call
(via ``_current_session_id``) — that's how the agent threads it from
state into the tool call.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from langchain_core.tools import StructuredTool

from deepagent_hermes.store.sqlite_fts import SqliteFtsStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _fmt_ts(ts: float | int | None) -> str:
    if ts is None:
        return "unknown"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%B %d, %Y at %I:%M %p")
    except (ValueError, OSError, OverflowError):
        return str(ts)


def _content_excerpt(content: Any, *, max_chars: int = 200) -> str:
    """Best-effort one-line preview of a message body.

    Handles multimodal content blocks by concatenating their text
    parts so the agent sees something readable.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        text = " ".join(parts).strip() or "[non-text content]"
    elif isinstance(content, dict) and "text" in content:
        text = str(content.get("text") or "")
    else:
        text = str(content)
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _render_message(msg: dict[str, Any], *, anchor_id: int | None = None) -> str:
    marker = " ← anchor" if anchor_id is not None and msg.get("id") == anchor_id else ""
    role = msg.get("role") or "?"
    mid = msg.get("id")
    body = _content_excerpt(msg.get("content"))
    tool_name = msg.get("tool_name")
    tool_part = f" [{tool_name}]" if tool_name else ""
    return f"- **{role}** (#{mid}){tool_part}{marker}: {body}"


def _join_messages(messages: list[dict[str, Any]], *, anchor_id: int | None = None) -> str:
    return "\n".join(_render_message(m, anchor_id=anchor_id) for m in messages)


# ---------------------------------------------------------------------------
# tool implementation
# ---------------------------------------------------------------------------


def run_session_search(
    store: SqliteFtsStore,
    *,
    query: str = "",
    session_id: str = "",
    around_message_id: int | None = None,
    window: int = 5,
    sources_exclude: list[str] | None = None,
    current_session_id: str = "",
) -> str:
    """Pure-function implementation — public for testing.

    The ``StructuredTool`` returned by :func:`make_session_search_tool`
    is a thin wrapper that fills in ``store`` and
    ``current_session_id`` from its closure.
    """
    excl = list(sources_exclude) if sources_exclude is not None else ["tool"]

    # SCROLL takes precedence over DISCOVERY when both look set.
    if session_id and around_message_id is not None:
        return _scroll(
            store,
            session_id=session_id,
            around_message_id=int(around_message_id),
            window=window,
            current_session_id=current_session_id,
        )

    if not query or not query.strip():
        return _browse(
            store,
            current_session_id=current_session_id,
            exclude_sources=excl,
        )

    return _discover(
        store,
        query=query.strip(),
        current_session_id=current_session_id,
        exclude_sources=excl,
    )


# ── mode: BROWSE ──────────────────────────────────────────────────────


def _browse(
    store: SqliteFtsStore,
    *,
    current_session_id: str,
    exclude_sources: list[str],
) -> str:
    try:
        sessions = store.list_recent_sessions(limit=20, exclude_sources=exclude_sources)
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("session_search browse failed")
        return f"## session_search (browse)\n\nError loading sessions: {exc}"

    current_root = store.resolve_to_lineage_root(current_session_id) if current_session_id else ""
    surfaced: list[dict[str, Any]] = []
    for s in sessions:
        if current_root and s["id"] == current_root:
            continue
        surfaced.append(s)
        if len(surfaced) >= 20:
            break

    if not surfaced:
        return "## session_search (browse)\n\nNo prior sessions on disk yet.\n"

    lines = [
        "## session_search (browse)",
        "",
        f"Showing {len(surfaced)} most-recently-active sessions. Pass `query=` "
        "to search, or `session_id` + `around_message_id` to scroll.",
        "",
    ]
    for s in surfaced:
        title = s.get("title") or "(no title)"
        lines.append(
            f"### {s['id']}  —  {title}\n"
            f"- source: `{s.get('source') or 'unknown'}`  "
            f"started: {_fmt_ts(s.get('started_at'))}  "
            f"last active: {_fmt_ts(s.get('last_active'))}\n"
            f"- messages: {s.get('message_count') or 0}  "
            f"tool calls: {s.get('tool_call_count') or 0}\n"
            f"- preview: {s.get('preview') or '(empty)'}\n"
        )
    return "\n".join(lines)


# ── mode: SCROLL ──────────────────────────────────────────────────────


def _scroll(
    store: SqliteFtsStore,
    *,
    session_id: str,
    around_message_id: int,
    window: int,
    current_session_id: str,
) -> str:
    # Clamp window per spec.
    try:
        window = int(window)
    except (TypeError, ValueError):
        window = 5
    window = max(1, min(window, 20))

    # Refuse scrolling inside the current session lineage.
    if current_session_id:
        try:
            anchor_root = store.resolve_to_lineage_root(session_id)
            cur_root = store.resolve_to_lineage_root(current_session_id)
        except Exception:
            anchor_root, cur_root = "", ""
        if anchor_root and cur_root and anchor_root == cur_root:
            return (
                "## session_search (scroll)\n\n"
                "**Error**: scroll rejected — anchor lives in the current "
                "session lineage; those messages are already in your context.\n"
            )

    meta = store.get_session(session_id)
    if not meta:
        return f"## session_search (scroll)\n\n**Error**: session_id `{session_id}` not found.\n"

    view = store.get_messages_around(session_id, around_message_id, window=window)
    messages = view["window"]
    if not messages:
        return f"## session_search (scroll)\n\n**Error**: message #{around_message_id} not in session `{session_id}`.\n"

    title = meta.get("title") or "(no title)"
    lines = [
        "## session_search (scroll)",
        "",
        f"**Session**: `{session_id}` — {title}  ({_fmt_ts(meta.get('started_at'))})",
        f"**Anchor**: message #{around_message_id}  "
        f"(±{window} window; {view['messages_before']} before, "
        f"{view['messages_after']} after)",
        "",
        _join_messages(messages, anchor_id=around_message_id),
        "",
    ]
    if view["messages_before"] < window:
        lines.append("_(at session start — no earlier messages)_")
    if view["messages_after"] < window:
        lines.append("_(at session end — no later messages)_")
    lines.append("")
    lines.append("To page further: pass the first/last message id back as `around_message_id` to walk backward/forward.")
    return "\n".join(lines)


# ── mode: DISCOVERY ───────────────────────────────────────────────────


def _discover(
    store: SqliteFtsStore,
    *,
    query: str,
    current_session_id: str,
    exclude_sources: list[str],
) -> str:
    try:
        raw_hits = store.search_messages(
            query,
            limit=50,
            exclude_sources=exclude_sources,
            role_filter=["user", "assistant"],
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("session_search discover failed")
        return f"## session_search (discover)\n\nFTS5 query failed: {exc}\n"

    if not raw_hits:
        return f"## session_search (discover)\n\n**Query**: `{query}`\n\nNo matching sessions found.\n"

    current_root = store.resolve_to_lineage_root(current_session_id) if current_session_id else ""

    # Dedupe by lineage root; preserve the raw hit's owning session_id so
    # the anchored window pairs validly with the FTS match id.
    seen_lineages: dict[str, dict[str, Any]] = {}
    for hit in raw_hits:
        lineage_root = store.resolve_to_lineage_root(hit["session_id"])
        if current_root and lineage_root == current_root:
            continue
        if lineage_root not in seen_lineages:
            seen_lineages[lineage_root] = hit
        if len(seen_lineages) >= 10:
            break

    if not seen_lineages:
        return (
            f"## session_search (discover)\n\n"
            f"**Query**: `{query}`\n\n"
            "All matching sessions are in the current session's lineage "
            "(already in context).\n"
        )

    sections = [
        "## session_search (discover)",
        "",
        f"**Query**: `{query}`",
        f"**Results**: {len(seen_lineages)} session(s) ranked by BM25 relevance",
        "",
    ]
    for lineage_root, hit in seen_lineages.items():
        owning_sid = hit["session_id"]
        msg_id = hit["id"]
        try:
            view = store.get_anchored_view(owning_sid, msg_id, window=5, bookend=3)
        except Exception as exc:
            logger.warning("get_anchored_view failed for %s/%s: %s", owning_sid, msg_id, exc)
            continue
        session_meta = store.get_session(lineage_root) or {}
        title = session_meta.get("title") or hit.get("title") or "(no title)"
        when = _fmt_ts(session_meta.get("started_at") or hit.get("session_started"))
        source = session_meta.get("source") or hit.get("source") or "unknown"
        model = session_meta.get("model") or hit.get("model") or "unknown"
        snippet = hit.get("snippet") or ""

        sections.append(f"### Session `{owning_sid}` — {title}")
        if lineage_root != owning_sid:
            sections.append(f"*lineage root*: `{lineage_root}`")
        sections.append(f"- when: {when}  source: `{source}`  model: `{model}`")
        sections.append(f"- match: **{hit.get('role') or '?'}** (#{msg_id}) — {snippet}")

        if view.get("bookend_start"):
            sections.append("")
            sections.append("**bookend_start** (session opening):")
            sections.append(_join_messages(view["bookend_start"]))

        if view.get("window"):
            sections.append("")
            sections.append(f"**window** (±5 around match; {view['messages_before']} before, {view['messages_after']} after):")
            sections.append(_join_messages(view["window"], anchor_id=msg_id))

        if view.get("bookend_end"):
            sections.append("")
            sections.append("**bookend_end** (session resolution):")
            sections.append(_join_messages(view["bookend_end"]))

        sections.append("")
        sections.append(
            f'_To read more: call `session_search(session_id="{owning_sid}", around_message_id={msg_id}, window=10)`._'
        )
        sections.append("")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


_SESSION_SEARCH_DOC = (
    "Search past sessions stored on disk, or scroll inside one. "
    "FTS5 BM25 retrieval over the SQLite message store — no LLM calls.\n\n"
    "Three calling shapes (inferred from which args are set):\n"
    "  1) DISCOVERY — pass `query`: returns top sessions, each with a "
    "snippet, ±5 message window around the match, and 3-message bookends "
    "at session start + end.\n"
    "  2) SCROLL — pass `session_id` + `around_message_id`: returns ±"
    "`window` messages centred on the anchor (default 5, clamped 1-20). "
    "Scrolling inside the *current* session is REJECTED (already in "
    "context).\n"
    "  3) BROWSE — no args: lists recent sessions chronologically.\n\n"
    "FTS5 syntax: multi-word queries default to AND. Use OR explicitly, "
    'quoted phrases ("docker networking"), or prefix wildcards (deploy*).'
)


def make_session_search_tool(
    store: SqliteFtsStore,
    *,
    name: str = "session_search",
    current_session_id_getter: Any = None,
) -> StructuredTool:
    """Build a LangChain ``StructuredTool`` bound to ``store``.

    ``current_session_id_getter`` is an optional zero-arg callable
    returning the active session id (so the tool can exclude the
    current lineage from results). Tests pass it as a lambda;
    production wiring would set it to ``lambda: state.session_id``
    after binding state. When omitted, the agent must pass
    ``_current_session_id`` explicitly per call.
    """

    def _impl(
        query: str = "",
        session_id: str = "",
        around_message_id: int | None = None,
        window: int = 5,
        sources_exclude: list[str] | None = None,
        _current_session_id: str = "",
    ) -> str:
        # Prefer the explicit per-call override; fall back to the getter.
        cur = _current_session_id
        if not cur and current_session_id_getter is not None:
            try:
                cur = current_session_id_getter() or ""
            except Exception:
                cur = ""
        return run_session_search(
            store,
            query=query,
            session_id=session_id,
            around_message_id=around_message_id,
            window=window,
            sources_exclude=sources_exclude,
            current_session_id=cur or "",
        )

    return StructuredTool.from_function(
        func=_impl,
        name=name,
        description=_SESSION_SEARCH_DOC,
    )


__all__ = ["make_session_search_tool", "run_session_search"]
