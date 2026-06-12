"""Tests for ``langstage_hermes.state.HermesState`` + ``initial_hermes_state``."""

from __future__ import annotations

from typing import get_args, get_type_hints

from langchain.agents.middleware.types import AgentState, PrivateStateAttr

from langstage_hermes.state import HermesState, initial_hermes_state

# ── factory: every documented field is present with the right default ──


EXPECTED_DEFAULTS: dict[str, object] = {
    "iters_since_skill": 0,
    "turns_since_memory": 0,
    "iteration_budget_remaining": 90,
    "active_skills": [],
    "loaded_skill_bodies": {},
    "last_compression_at": 0,
    "consecutive_low_yield_compressions": 0,
    "pending_review_kind": None,
    "last_review_started_at": 0.0,
    "estimated_cost_usd": 0.0,
    "actual_cost_usd": None,
    "session_id": "sess-123",
    "parent_session_id": None,
    "rewind_count": 0,
    "memory_snapshot": "",
    "user_snapshot": "",
    "model_override": None,
}


def test_initial_state_has_all_expected_keys():
    state = initial_hermes_state("sess-123")
    assert set(state.keys()) == set(EXPECTED_DEFAULTS.keys())


def test_initial_state_default_values():
    state = initial_hermes_state("sess-123")
    for key, expected in EXPECTED_DEFAULTS.items():
        assert state[key] == expected, f"field {key!r}: got {state[key]!r}, want {expected!r}"


def test_initial_state_respects_max_iterations():
    state = initial_hermes_state("sess-abc", max_iterations=50)
    assert state["iteration_budget_remaining"] == 50


def test_initial_state_subagent_lineage():
    state = initial_hermes_state("sub-001", max_iterations=50, parent_session_id="root-007")
    assert state["session_id"] == "sub-001"
    assert state["parent_session_id"] == "root-007"
    assert state["iteration_budget_remaining"] == 50


# ── TypedDict shape ──────────────────────────────────────────────────


def test_hermes_state_extends_agent_state():
    """``HermesState`` must carry the base AgentState contract.

    TypedDict inheritance doesn't show through ``__mro__`` (TypedDicts subclass
    ``dict`` directly), so we verify by checking that base fields are merged
    into ``__annotations__`` per PEP 589.
    """
    base_fields = set(AgentState.__annotations__.keys())
    child_fields = set(HermesState.__annotations__.keys())
    assert base_fields.issubset(child_fields), f"missing AgentState fields: {base_fields - child_fields}"


def test_hermes_state_annotation_keys_cover_spec_fields():
    """SPEC §3 fields must appear in the TypedDict annotations."""
    spec_keys = {
        "iters_since_skill",
        "turns_since_memory",
        "iteration_budget_remaining",
        "active_skills",
        "loaded_skill_bodies",
        "last_compression_at",
        "consecutive_low_yield_compressions",
        "pending_review_kind",
        "last_review_started_at",
        "estimated_cost_usd",
        "actual_cost_usd",
        "session_id",
        "parent_session_id",
        "rewind_count",
        "memory_snapshot",
        "user_snapshot",
        "model_override",
    }
    assert spec_keys.issubset(set(HermesState.__annotations__.keys()))


def test_private_state_attrs_annotated():
    """Counters carry ``PrivateStateAttr`` so they don't leak to public schema."""
    hints = get_type_hints(HermesState, include_extras=True)
    for fname in (
        "iters_since_skill",
        "turns_since_memory",
        "iteration_budget_remaining",
        "active_skills",
        "loaded_skill_bodies",
        "last_compression_at",
        "consecutive_low_yield_compressions",
        "pending_review_kind",
        "rewind_count",
        "memory_snapshot",
        "user_snapshot",
    ):
        ann = hints[fname]
        # NotRequired[Annotated[T, PrivateStateAttr]] → unwrap NotRequired then check metadata.
        # get_args(NotRequired[X])[0] is Annotated[T, metadata...]
        inner = get_args(ann)
        # ``Annotated`` metadata sits in get_args(inner[0])[1:] when wrapped in NotRequired,
        # but ``get_type_hints(include_extras=True)`` on TypedDict already strips NotRequired
        # in some Python versions. Handle both shapes.
        metadata: tuple = ()
        if inner:
            # NotRequired[Annotated[...]] case
            metadata = get_args(inner[0])[1:] if get_args(inner[0]) else ()
        if not metadata:
            # Already an Annotated (NotRequired stripped)
            metadata = get_args(ann)[1:] if get_args(ann) else ()
        assert any(m is PrivateStateAttr for m in metadata), f"field {fname!r} missing PrivateStateAttr; metadata={metadata!r}"
