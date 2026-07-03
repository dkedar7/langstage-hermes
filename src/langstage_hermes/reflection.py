"""Reflection middleware — closed-loop skill & memory creation (SPEC §9).

The reflection loop is THE differentiator of this runtime. After a turn finishes,
``ReflectionMiddleware`` checks two counters tracked in ``HermesState``:

* ``iters_since_skill`` — incremented on every tool-using turn, reset when the
  agent calls ``skill_manage``.
* ``turns_since_memory`` — incremented on every user turn, reset when the agent
  calls ``memory``.

When either counter hits its threshold and the model has just produced a final
response (no pending tool calls), the middleware flags ``pending_review_kind``
and the ``after_agent`` hook spawns a **review subagent** narrowed to the
``memory`` + ``skill_manage`` tools. The subagent receives the conversation
snapshot plus a review prompt and decides whether anything is worth saving.

This is decision (B) in SPEC §9: subagent-native spawning rather than a
``threading.Thread``. The win is observability — events flow through the
``langstage-core`` event stream so hosts can surface
"skill updated: pdf-merging" inline. The cost is that the review runs
**synchronously** in v1: the user waits for it before the next turn. v2 can
move to a background thread once we have a sturdier event bus.

The factory ``build_review_subagent`` returns a ``SubAgent`` TypedDict per
``deepagents``; callers register it through ``SubAgentMiddleware`` at
agent-build time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command


def _take_last(_existing: Any, new: Any) -> Any:
    """Last-write-wins reducer for fields that may be updated by parallel
    middleware branches in the same LangGraph superstep (e.g. parent agent
    + subagent both touching the counter). Without this, LangGraph raises
    ``InvalidUpdateError: At key 'X': Can receive only one value per step``.

    Surfaced live during the 2026-06-02 dogfood run when the review subagent
    dispatch landed on the same step as a counter increment.
    """
    return new


class _ReflectionStateExt(AgentState):
    """Schema extension so the counter / coordination fields the reflection
    middleware emits actually persist across hooks. Without this, the field
    updates returned from ``wrap_tool_call`` / ``before_model`` / ``after_model``
    are silently dropped by LangGraph (same failure mode as the recorder's
    session_id bug from 2026-06-02).

    Reducer-annotated to tolerate parallel writes from the parent agent and
    spawned subagents in the same superstep.
    """

    iters_since_skill: NotRequired[Annotated[int, _take_last]]
    turns_since_memory: NotRequired[Annotated[int, _take_last]]
    pending_review_kind: NotRequired[Annotated[Literal["memory", "skills", "combined"] | None, _take_last]]
    last_review_started_at: NotRequired[Annotated[float, _take_last]]


if TYPE_CHECKING:
    from typing_extensions import TypedDict

    class SubAgent(TypedDict, total=False):
        name: str
        description: str
        system_prompt: str
        tools: list[Any]
        model: Any
        middleware: list[Any]


logger = logging.getLogger(__name__)


# ── prompt loading ───────────────────────────────────────────────────


_PROMPT_PACKAGE = "langstage_hermes._prompts"
_PROMPTS_DIR = Path(__file__).resolve().parent / "_prompts"


def load_prompt(name: str) -> str:
    """Read a prompt file by relative name (e.g. ``"combined_review.md"``).

    Prompts live inside the package at ``langstage_hermes/_prompts/`` (as of
    v0.1.2). We try ``importlib.resources`` first because it's the supported
    cross-installer path; on the rare loader that doesn't expose package data
    that way we fall back to a direct filesystem read next to ``__file__``.
    """
    try:
        return resources.files(_PROMPT_PACKAGE).joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        path = _PROMPTS_DIR.joinpath(name)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        msg = f"prompt not found: {name} (tried package {_PROMPT_PACKAGE!r} and {path})"
        raise FileNotFoundError(msg) from None


# ── review subagent factory ──────────────────────────────────────────


def build_review_subagent(
    *,
    library: Any,
    store: Any,
    aux_model: Any,
    tools: list[Any] | None = None,
) -> dict[str, Any]:
    """Return a ``SubAgent`` spec for the background review fork.

    Registered with ``SubAgentMiddleware`` at agent-build time per SPEC §9
    decision (B). The subagent's system prompt is the combined review prompt
    so it can act on either signal; the dispatching middleware decides whether
    skills, memory, or both are in play and passes the right ``description``
    string to the ``task`` tool.

    Args:
        library: The ``SkillLibrary`` instance (passed through for future use
            by skill tools — the subagent itself dispatches them by name).
        store: The Hermes store (memory & FTS backend). Reserved.
        aux_model: The auxiliary chat model — typically a cheaper / faster
            model than the main one (Hermes uses a separate aux client; we
            pass an ``init_chat_model(...)`` result).
        tools: Optional explicit tools list. If ``None``, the subagent is
            registered with no extra tools and inherits whatever the parent
            agent makes available via the ``task`` dispatch (Hermes narrows to
            ``memory`` + ``skill_manage`` — wiring that whitelist is the
            caller's responsibility at agent-build time).

    Returns:
        A dict matching ``deepagents.middleware.subagents.SubAgent`` shape so
        the caller can drop it directly into ``SubAgentMiddleware(subagents=[
        build_review_subagent(...), ...])``.
    """
    del library, store  # reserved for future schema use
    spec: dict[str, Any] = {
        "name": "review",
        "description": (
            "Reflects on the recent conversation and updates the skill library "
            "and/or memory store. Spawned by ReflectionMiddleware after every "
            "<nudge_interval> tool iterations or user turns. Do not call directly "
            "from user input — the middleware schedules it."
        ),
        "system_prompt": load_prompt("combined_review.md"),
        "tools": tools or [],
    }
    if aux_model is not None:
        spec["model"] = aux_model
    return spec


# ── ReflectionMiddleware ─────────────────────────────────────────────


_SKILL_TOOL_NAME = "skill_manage"
_MEMORY_TOOL_NAME = "memory"


class ReflectionMiddleware(AgentMiddleware):
    """Track counters, detect end-of-turn, and spawn the review subagent.

    Counters live in ``HermesState`` (not on ``self``) because middleware is
    stateless and state IS the per-thread persistence boundary in ``deepagents``.

    Hooks:

    * ``wrap_tool_call`` — runs the tool first, then on success increments or
      resets the appropriate counter.
    * ``before_model`` — detects a real user-turn boundary (the message before
      the current ``HumanMessage`` is an ``AIMessage`` with no pending tool
      calls — i.e. the last assistant turn was a final response, not a
      tool-result resume) and bumps ``turns_since_memory``.
    * ``after_model`` — if the just-produced ``AIMessage`` has no pending tool
      calls AND a counter is over threshold, flags ``pending_review_kind``.
    * ``after_agent`` — if ``pending_review_kind`` is set, invokes the review
      subagent inline and clears the counters that fired.

    The "inline invocation" path is implemented synchronously by compiling
    ``self._review_graph`` once at construction time and ``.invoke()``-ing it
    in ``after_agent``. This is the SPEC §9 decision-(B) trade-off: observable
    but adds latency; v2 can move the call onto a ``threading.Thread``.
    """

    state_schema = _ReflectionStateExt

    def __init__(
        self,
        skill_nudge_interval: int = 10,
        memory_nudge_interval: int = 10,
        *,
        library: Any,
        store: Any,
        model: Any,
        aux_model: Any = None,
        skills_toolset_enabled: bool = True,
        review_graph: Any | None = None,
    ) -> None:
        """Construct the middleware.

        Args:
            skill_nudge_interval: Tool iterations without a ``skill_manage``
                call before a skill review is scheduled. Default 10
                (matches Hermes ``_skill_nudge_interval``).
            memory_nudge_interval: User turns without a ``memory`` call before
                a memory review is scheduled. Default 10.
            library: The ``SkillLibrary`` — handed to the review subagent.
            store: The Hermes store — handed to the review subagent.
            model: The primary chat model. Reserved for future heuristics
                (e.g. skip review when the model is "tiny").
            aux_model: Cheaper model the review fork uses. Defaults to
                ``model`` if ``None``.
            skills_toolset_enabled: When ``False``, ``iters_since_skill`` is
                never bumped (no review will fire for skills).
            review_graph: Pre-compiled review subgraph. If ``None``, the
                middleware can't invoke synchronously; callers must wire the
                review subagent via ``SubAgentMiddleware`` instead and
                implement their own dispatch in ``after_agent``. Passing this
                lets tests inject a mock.
        """
        super().__init__()
        self.skill_nudge_interval = skill_nudge_interval
        self.memory_nudge_interval = memory_nudge_interval
        self.library = library
        self.store = store
        self.model = model
        self.aux_model = aux_model or model
        self.skills_toolset_enabled = skills_toolset_enabled
        self._review_graph = review_graph
        # No extra tools registered by this middleware itself — the `task`
        # tool that dispatches to the review subagent is owned by
        # `SubAgentMiddleware` per SPEC §9.
        self.tools: list[Any] = []

    # ── wrap_tool_call ───────────────────────────────────────────

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Execute the tool, then update counters based on which tool ran."""
        result = handler(request)

        # Only bookkeep on a clean result; let exceptions propagate and let
        # `wrap_tool_call`s further out handle them.
        tool_name = request.tool_call.get("name") if request.tool_call else None
        update = self._counter_update_for_tool(tool_name, request.state)
        if not update:
            return result

        # `result` may be a ToolMessage OR a Command from a subagent. Preserve
        # whichever it is by piggybacking the counter delta onto a Command.
        if isinstance(result, Command):
            base_update = dict(result.update or {})
            base_update.update(update)
            return Command(update=base_update, goto=result.goto)
        # Bare ToolMessage — wrap in a Command so we can ship state updates
        # alongside the message.
        return Command(update={**update, "messages": [result]})

    def _counter_update_for_tool(self, tool_name: str | None, state: Any) -> dict[str, Any]:
        """Decide which counter to bump/reset based on the tool that ran."""
        if tool_name == _SKILL_TOOL_NAME:
            # Successful skill_manage call resets the skills counter.
            return {"iters_since_skill": 0}
        if tool_name == _MEMORY_TOOL_NAME:
            # Successful memory call resets the memory counter.
            return {"turns_since_memory": 0}
        if not self.skills_toolset_enabled:
            return {}
        # Any other tool counts as a "tool-using iteration" — bump the
        # skill-review counter.
        current = self._read_int(state, "iters_since_skill")
        return {"iters_since_skill": current + 1}

    # ── before_model ─────────────────────────────────────────────

    def before_model(self, state: Any, runtime: Runtime[Any] | None = None) -> dict[str, Any] | None:
        """Detect a user-turn boundary and bump ``turns_since_memory``.

        A "real user turn" is one where the LATEST message is a
        ``HumanMessage`` AND the message before it (if any) is an
        ``AIMessage`` with no pending tool calls. That distinguishes a fresh
        user prompt from a tool-result loopback (where the last message would
        be a ``ToolMessage``).
        """
        messages = self._messages(state)
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, HumanMessage):
            return None
        if len(messages) >= 2:
            prior = messages[-2]
            if isinstance(prior, AIMessage) and prior.tool_calls:
                # AIMessage with pending tool calls means the human is somehow
                # interleaved — treat as not-a-user-turn to be safe.
                return None
            if isinstance(prior, ToolMessage):
                # Tool-result resume, not a user turn.
                return None
        current = self._read_int(state, "turns_since_memory")
        return {"turns_since_memory": current + 1}

    # ── after_model ──────────────────────────────────────────────

    def after_model(self, state: Any, runtime: Runtime[Any] | None = None) -> dict[str, Any] | None:
        """Flag a review when the model has finished and a counter is over."""
        messages = self._messages(state)
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None
        # Don't fire while tool calls are pending — wait for the final response.
        if last.tool_calls:
            return None
        # Don't fire while interrupted (jump_to set).
        if self._get(state, "jump_to"):
            return None

        iters_since_skill = self._read_int(state, "iters_since_skill")
        turns_since_memory = self._read_int(state, "turns_since_memory")
        skill_due = self.skills_toolset_enabled and (iters_since_skill >= self.skill_nudge_interval)
        memory_due = turns_since_memory >= self.memory_nudge_interval

        if not (skill_due or memory_due):
            return None

        kind: Literal["memory", "skills", "combined"]
        if skill_due and memory_due:
            kind = "combined"
        elif skill_due:
            kind = "skills"
        else:
            kind = "memory"

        return {
            "pending_review_kind": kind,
            "last_review_started_at": time.time(),
        }

    # ── after_agent ──────────────────────────────────────────────

    def after_agent(self, state: Any, runtime: Runtime[Any] | None = None) -> dict[str, Any] | None:
        """Invoke the review subagent synchronously and clear the counters."""
        kind = self._get(state, "pending_review_kind")
        if not kind:
            return None

        review_prompt = self._prompt_for(kind)
        reset: dict[str, Any] = {"pending_review_kind": None}
        if kind in ("skills", "combined"):
            reset["iters_since_skill"] = 0
        if kind in ("memory", "combined"):
            reset["turns_since_memory"] = 0

        if self._review_graph is None:
            # No compiled subgraph wired — log and return the counter reset
            # anyway so the trigger doesn't stay stuck on. Production wiring
            # via SubAgentMiddleware will replace this branch.
            logger.info(
                "Reflection: pending_review_kind=%s but no review_graph wired; skipping inline invocation (counters reset).",
                kind,
            )
            return reset

        try:
            history = list(self._messages(state))
            sub_state = {"messages": [*history, HumanMessage(content=review_prompt)]}
            self._review_graph.invoke(sub_state)
        except Exception as exc:
            logger.warning("Reflection: review subagent failed: %s", exc)

        return reset

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _messages(state: Any) -> list[Any]:
        msgs = ReflectionMiddleware._get(state, "messages")
        return list(msgs) if msgs else []

    @staticmethod
    def _get(state: Any, key: str) -> Any:
        if isinstance(state, dict):
            return state.get(key)
        return getattr(state, key, None)

    @staticmethod
    def _read_int(state: Any, key: str) -> int:
        value = ReflectionMiddleware._get(state, key)
        if isinstance(value, int):
            return value
        return 0

    @staticmethod
    def _prompt_for(kind: str) -> str:
        if kind == "memory":
            return load_prompt("memory_review.md")
        if kind == "skills":
            return load_prompt("skill_review.md")
        return load_prompt("combined_review.md")

    # ── async parity ─────────────────────────────────────────────

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        tool_name = request.tool_call.get("name") if request.tool_call else None
        update = self._counter_update_for_tool(tool_name, request.state)
        if not update:
            return result
        if isinstance(result, Command):
            base_update = dict(result.update or {})
            base_update.update(update)
            return Command(update=base_update, goto=result.goto)
        return Command(update={**update, "messages": [result]})

    async def abefore_model(self, state: Any, runtime: Runtime[Any] | None = None) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    async def aafter_model(self, state: Any, runtime: Runtime[Any] | None = None) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    async def aafter_agent(self, state: Any, runtime: Runtime[Any] | None = None) -> dict[str, Any] | None:
        # Synchronous path is fine — review_graph.invoke is sync.
        return self.after_agent(state, runtime)


__all__ = [
    "ReflectionMiddleware",
    "build_review_subagent",
    "load_prompt",
]
