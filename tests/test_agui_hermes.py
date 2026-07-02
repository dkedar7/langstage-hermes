"""hermes' experimental AG-UI render path (ADR 0002/0003).

Validates that hermes' four extractors — dead code on the legacy StreamParser path
(never registered) — now fire via the core's iter_event_frames(extractors=...),
surfacing skill/memory callouts on the AG-UI stream.
"""

from collections.abc import Iterator

import pytest

pytest.importorskip("ag_ui_langgraph")
pytest.importorskip("fastapi")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool

from langstage_hermes.agui_stream import build_session_agent, stream_frames_sync


@tool
def skill_manage(action: str, name: str) -> str:
    """Create/update/delete a skill."""
    import json

    return json.dumps({"action": action, "name": name})


class _FakeToolModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        if any(isinstance(m, ToolMessage) for m in messages):
            yield ChatGenerationChunk(message=AIMessageChunk(content="Created."))
        else:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="", tool_call_chunks=[{"name": "skill_manage", "args": "", "id": "c1", "index": 0}]
                )
            )
            for seg in ('{"action": "create", ', '"name": "pdf-merging"}'):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", tool_call_chunks=[{"name": None, "args": seg, "id": None, "index": 0}])
                )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        chunks = list(self._stream(messages))
        msg = chunks[0].message
        for c in chunks[1:]:
            msg = msg + c.message
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=msg.content, tool_calls=getattr(msg, "tool_calls", [])))]
        )


def test_skill_extractor_fires_on_agui_path():
    from langgraph.prebuilt import create_react_agent

    agent = build_session_agent(create_react_agent(_FakeToolModel(), [skill_manage]))
    frames = list(stream_frames_sync(agent, "make a pdf-merging skill", "t1", state={}))

    extraction = [f for f in frames if f.get("type") == "extraction"]
    assert extraction, f"no extraction frame — extractor didn't fire: {[f.get('type') for f in frames]}"
    ex = extraction[0]
    assert ex["extracted_type"] == "skill_event"
    assert ex["data"]["name"] == "pdf-merging"
    assert ex["data"]["extracted_subtype"] == "skill_created"


def test_render_extraction_frame_draws_callout(capsys):
    from langstage_hermes.cli import _render_extraction_frame

    _render_extraction_frame(
        {
            "type": "extraction",
            "tool_name": "skill_manage",
            "extracted_type": "skill_event",
            "data": {"name": "pdf-merging", "extracted_subtype": "skill_created"},
        }
    )
    out = capsys.readouterr().out
    assert "skill created" in out and "pdf-merging" in out
