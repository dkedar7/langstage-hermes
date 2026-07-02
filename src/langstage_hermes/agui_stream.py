"""Experimental in-process AG-UI render path for the hermes chat cli.

ADR 0002/0003: drive the agent through the official ``ag-ui-langgraph`` adapter
in-process (no web server) and map AG-UI events onto ``event_to_dict``-shaped
frames via the core's ``agui.iter_event_frames`` — the same wire the cli already
renders. hermes' four tool-result extractors ride the core's ``extractors=`` param,
so the skill / memory / compression callouts surface as ``extraction`` frames.

(Those extractors were previously defined but never registered — a bare
``StreamParser()`` ran the legacy path — so the callouts were dead code. Wiring
them here makes them fire for the first time.)

hermes' agent takes a richer input than plain messages (session_id, model_override,
iteration_budget_remaining); those ride ``iter_event_frames(state=...)``, which the
adapter forwards into the graph input.

Requires the ``agui`` extra::

    pip install "langstage-hermes[agui]"
"""

from __future__ import annotations

import asyncio
from typing import Any

_IMPORT_HINT = 'the AG-UI path needs the agui extra: pip install "langstage-hermes[agui]"'


def ensure_agui_available() -> None:
    """Raise a clean, actionable error if the AG-UI adapter isn't installed."""
    try:
        import ag_ui_langgraph  # noqa: F401
        from langgraph_stream_parser.agui import iter_event_frames  # noqa: F401
    except ImportError as e:  # pragma: no cover - only without the extra
        raise RuntimeError(_IMPORT_HINT) from e


def build_session_agent(graph: Any, *, name: str = "langstage-hermes") -> Any:
    """Wrap the graph once (checkpointer attached by the core bridge)."""
    ensure_agui_available()
    from langgraph_stream_parser.agui import build_agent

    return build_agent(graph, name=name)


def _extractors() -> list:
    from langstage_hermes.extractors import ALL_EXTRACTORS

    return [cls() for cls in ALL_EXTRACTORS]


def stream_frames_sync(agent, message: str, thread_id: str, *, state=None, resume=None):
    """Sync bridge: pump ``iter_event_frames`` (with hermes' extractors wired) one
    frame at a time. hermes' turn loop is a plain sync process, so a fresh event
    loop is safe and keeps rendering lazy."""
    from langgraph_stream_parser.agui import iter_event_frames

    loop = asyncio.new_event_loop()
    try:
        agen = iter_event_frames(agent, message, thread_id, extractors=_extractors(), state=state, resume=resume)
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()
