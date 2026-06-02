"""`HermesState` — typed agent state for ``deepagent-hermes``.

Extends ``langchain.agents.middleware.types.AgentState`` (which already carries
``messages``, ``jump_to``, ``structured_response``) with every Hermes-specific
field Hermes tracks as ``agent._*`` instance attrs (SPEC §3).

Why state, not instance attrs? In ``deepagents`` / ``langgraph``, middleware is
**stateless**: the same compiled graph services every thread, so per-conversation
data MUST live in the state dict (it's the per-thread persistence boundary).

Counter and snapshot fields are annotated ``PrivateStateAttr`` so they don't
leak into the public input/output JSON schema — they're middleware bookkeeping,
not part of the agent's I/O contract.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired

from langchain.agents.middleware.types import AgentState, PrivateStateAttr

# ── HermesState ──────────────────────────────────────────────────────


class HermesState(AgentState):
    """Per-thread state for a Hermes agent run.

    Inherits ``messages`` / ``jump_to`` / ``structured_response`` from
    ``AgentState``. All Hermes-added fields are ``NotRequired`` so a fresh
    invocation can start with just ``messages`` and let middleware populate the
    rest via ``initial_hermes_state`` (or via the first ``before_agent`` pass).
    """

    # ── iteration tracking — drives reflection triggers (SPEC §3) ──
    iters_since_skill: NotRequired[Annotated[int, PrivateStateAttr]]
    """Tool-using turns since last ``skill_manage``. Reset to 0 on call."""

    turns_since_memory: NotRequired[Annotated[int, PrivateStateAttr]]
    """User turns since last ``memory`` tool call. Reset to 0 on call."""

    iteration_budget_remaining: NotRequired[Annotated[int, PrivateStateAttr]]
    """Iterations left before the budget middleware jumps to ``end``."""

    # ── skill state ──
    active_skills: NotRequired[Annotated[list[str], PrivateStateAttr]]
    """Names of skills currently loaded via ``skill_view``."""

    loaded_skill_bodies: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]
    """Cached SKILL.md bodies, keyed by skill name (cost-amortized)."""

    # ── compression state ──
    last_compression_at: NotRequired[Annotated[int, PrivateStateAttr]]
    """Message index at last compression — anti-thrash guard."""

    consecutive_low_yield_compressions: NotRequired[Annotated[int, PrivateStateAttr]]
    """Number of recent compressions yielding <10% reduction (≥2 → skip)."""

    # ── background-review coordination ──
    pending_review_kind: NotRequired[
        Annotated[Literal["memory", "skills", "combined"] | None, PrivateStateAttr]
    ]
    """Which review the next ``after_agent`` should spawn (or ``None``)."""

    last_review_started_at: NotRequired[Annotated[float, PrivateStateAttr]]
    """UNIX timestamp of the last review spawn — sentinel against re-entry."""

    # ── cost / budget ──
    estimated_cost_usd: NotRequired[Annotated[float, PrivateStateAttr]]
    """Running estimate (pre-billing); updated per turn."""

    actual_cost_usd: NotRequired[Annotated[float | None, PrivateStateAttr]]
    """Settled cost from provider headers, when available."""

    # ── session lineage ──
    session_id: NotRequired[str]
    """Stable per-thread ID; same as the langgraph thread_id by convention."""

    parent_session_id: NotRequired[str | None]
    """Parent session if this is a delegated subagent — else ``None``."""

    rewind_count: NotRequired[Annotated[int, PrivateStateAttr]]
    """How many times this session has been rewound via ``/rollback``."""

    # ── frozen prompt snapshots (SPEC §13.1 prefix-cache discipline) ──
    memory_snapshot: NotRequired[Annotated[str, PrivateStateAttr]]
    """MEMORY.md content frozen at session start — never mutated mid-session."""

    user_snapshot: NotRequired[Annotated[str, PrivateStateAttr]]
    """USER.md content frozen at session start — never mutated mid-session."""

    # ── runtime overrides ──
    model_override: NotRequired[Annotated[str | None, PrivateStateAttr]]
    """Provider:model string set by ``/model`` slash command; ``None`` = default."""


# ── factory ──────────────────────────────────────────────────────────


def initial_hermes_state(
    session_id: str,
    *,
    max_iterations: int = 90,
    parent_session_id: str | None = None,
) -> HermesState:
    """Return a fresh ``HermesState`` with every Hermes-tracked field at its default.

    ``messages`` is left out — ``langgraph`` populates it from the user's
    invocation. Callers that need to seed messages should merge them in.
    """
    return HermesState(  # type: ignore[typeddict-item]
        iters_since_skill=0,
        turns_since_memory=0,
        iteration_budget_remaining=max_iterations,
        active_skills=[],
        loaded_skill_bodies={},
        last_compression_at=0,
        consecutive_low_yield_compressions=0,
        pending_review_kind=None,
        last_review_started_at=0.0,
        estimated_cost_usd=0.0,
        actual_cost_usd=None,
        session_id=session_id,
        parent_session_id=parent_session_id,
        rewind_count=0,
        memory_snapshot="",
        user_snapshot="",
        model_override=None,
    )


__all__ = ["HermesState", "initial_hermes_state"]
