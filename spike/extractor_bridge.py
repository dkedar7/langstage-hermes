"""ADR 0003 Stage 1 prototype: run tool-result extractors over the in-process
AG-UI stream and surface their output as `extraction` frames — the mechanism
that would let hermes' skill/memory/compression callouts work on AG-UI.

Validated against hermes' REAL SkillManageExtractor. Spike, not for merge.
"""
import asyncio
import json
import uuid


async def iter_event_frames_with_extractors(agent, message, thread_id, *, extractors=(), max_result_len=50_000):
    """iter_event_frames + extractor bridge: on each tool result, run the matching
    extractor and emit an `extraction` frame (parity with event_to_dict(ToolExtractedEvent))."""
    from ag_ui.core.types import RunAgentInput, UserMessage

    by_tool = {e.tool_name: e for e in extractors}
    run_input = RunAgentInput(
        thread_id=thread_id, run_id=str(uuid.uuid4()), state={},
        messages=[UserMessage(id=str(uuid.uuid4()), role="user", content=message)],
        tools=[], context=[], forwarded_props={},
    )
    tool_names = {}
    async for ev in agent.run(run_input):
        t = type(ev).__name__
        if t == "TextMessageContentEvent":
            yield {"type": "content", "content": ev.delta, "role": "assistant", "node": "agent"}
        elif t == "ToolCallStartEvent":
            tool_names[ev.tool_call_id] = ev.tool_call_name
        elif t == "ToolCallResultEvent":
            name = tool_names.get(ev.tool_call_id, "tool")
            content = getattr(ev, "content", "")
            yield {"type": "tool_end", "id": ev.tool_call_id, "name": name,
                   "result": str(content)[:max_result_len], "status": "success",
                   "error_message": None, "duration_ms": None}
            # --- the bridge: run the matching extractor over the tool result ---
            extractor = by_tool.get(name)
            if extractor is not None:
                data = extractor.extract(content)
                if data is not None:
                    yield {"type": "extraction", "tool_name": name,
                           "extracted_type": extractor.extracted_type, "data": data}
        elif t == "RunErrorEvent":
            yield {"type": "error", "error": getattr(ev, "message", "err")}
            return
    yield {"type": "complete"}


def build_skill_agent():
    """A keyless react agent whose `skill_manage` tool returns a JSON result
    that hermes' SkillManageExtractor recognizes."""
    from typing import Iterator, List
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent

    @tool
    def skill_manage(action: str, name: str) -> str:
        """Create/update/delete a skill."""
        return json.dumps({"action": action, "name": name})

    class M(BaseChatModel):
        @property
        def _llm_type(self): return "m"
        def bind_tools(self, tools, **k): return self
        def _stream(self, messages: List[BaseMessage], stop=None, run_manager=None, **k) -> Iterator[ChatGenerationChunk]:
            if any(isinstance(m, ToolMessage) for m in messages):
                yield ChatGenerationChunk(message=AIMessageChunk(content="Done."))
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                    {"name": "skill_manage", "args": "", "id": "c1", "index": 0}]))
                for seg in ('{"action": "create", ', '"name": "pdf-merging"}'):
                    yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                        {"name": None, "args": seg, "id": None, "index": 0}]))
        def _generate(self, messages, stop=None, run_manager=None, **k) -> ChatResult:
            cs = list(self._stream(messages)); msg = cs[0].message
            for c in cs[1:]: msg = msg + c.message
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=msg.content, tool_calls=getattr(msg, "tool_calls", [])))])

    return create_react_agent(M(), [skill_manage])


async def main():
    from langgraph_stream_parser.agui import build_agent
    from langgraph_stream_parser.events import ToolExtractedEvent, event_to_dict
    from langstage_hermes.extractors import SkillManageExtractor  # hermes' REAL extractor

    agent = build_agent(build_skill_agent())
    frames = [f async for f in iter_event_frames_with_extractors(
        agent, "make a pdf-merging skill", "t1", extractors=[SkillManageExtractor()])]

    extraction = [f for f in frames if f["type"] == "extraction"]
    print("frame types:", [f["type"] for f in frames])
    print("extraction frame:", extraction[0] if extraction else None)

    # Parity: does the bridged frame equal event_to_dict(ToolExtractedEvent(...))?
    assert extraction, "no extraction frame emitted!"
    ex = extraction[0]
    expected = event_to_dict(ToolExtractedEvent(
        tool_name=ex["tool_name"], extracted_type=ex["extracted_type"], data=ex["data"]))
    print("event_to_dict parity:", ex == expected)
    print("hermes renderer would read: extracted_type=%r data=%r"
          % (ex["extracted_type"], ex["data"]))
    assert ex == expected
    print("\nPROTOTYPE PASSED: hermes' real SkillManageExtractor ran over the AG-UI"
          " tool result and produced an `extraction` frame at event_to_dict parity.")


asyncio.run(main())
