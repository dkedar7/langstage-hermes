"""The shipped agent runs the reflection review itself (gh #179).

``ReflectionMiddleware.after_model`` flags ``pending_review_kind`` when a counter
crosses its threshold. Before the fix ``create_hermes_agent`` never passed a
``review_graph``, so ``after_agent`` logged at INFO, reset the counters and
dropped the review. The loop closed only when the main model happened to call
``task(subagent_type="review")`` on its own, which a real model is never told to
do. These tests drive the real ``create_hermes_agent`` wiring (no mocked review
graph) with keyless scripted models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langstage_hermes import HermesConfig, create_hermes_agent
from langstage_hermes.demo import DEMO_SKILL_NAME, DemoMainModel, DemoReviewModel, _close_agent_resources, _ScriptedModel
from langstage_hermes.reflection import ReflectionMiddleware


class _NeverReviewsModel(_ScriptedModel):
    """A main model that behaves like a real one: some tool calls, then an answer.

    It never calls ``task``, so the review can only run if the middleware runs it.
    """

    tool_turns: int = 3

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any):
        n = sum(isinstance(m, AIMessage) for m in messages)
        if n < self.tool_turns:
            return self._result(AIMessage(content="", tool_calls=[{"name": "ls", "id": f"ls-{n}", "args": {"path": "."}}]))
        return self._result(AIMessage(content="Done."))


_REVIEW_CALLS: list[int] = []


class _CountingReviewModel(DemoReviewModel):
    """The demo's review model, counting each review run (its first call)."""

    def _generate(self, messages: list[Any], *args: Any, **kwargs: Any):
        if not any(getattr(m, "type", None) == "tool" and getattr(m, "tool_call_id", "") == "skill-create" for m in messages):
            _REVIEW_CALLS.append(1)
        return super()._generate(messages, *args, **kwargs)


@pytest.fixture
def home(tmp_hermes_home: Path) -> Path:
    _REVIEW_CALLS.clear()
    return tmp_hermes_home


def _run(home: Path, main: Any, *, skill_interval: int = 3, memory_interval: int = 1, sid: str = "r179") -> dict:
    ws = home / "ws"
    ws.mkdir(exist_ok=True)
    cfg = HermesConfig.resolve(
        overrides={"skills_creation_nudge_interval": skill_interval, "memory_nudge_interval": memory_interval}
    )
    agent = create_hermes_agent(cfg, workspace=ws, session_id=sid, model=main, aux_model=_CountingReviewModel())
    try:
        return agent.invoke(
            {
                "messages": [{"role": "user", "content": "list files a few times"}],
                "session_id": sid,
                "iteration_budget_remaining": 30,
            },
            config={"configurable": {"thread_id": sid}},
        )
    finally:
        _close_agent_resources(agent)


def _skills(home: Path) -> list[Path]:
    return sorted((home / "skills").rglob("SKILL.md")) if (home / "skills").is_dir() else []


def test_due_review_runs_when_main_model_never_calls_task(home: Path):
    """The issue's repro: the review runs and writes a skill + memory anyway."""
    state = _run(home, _NeverReviewsModel())

    tool_calls = [tc["name"] for m in state["messages"] if isinstance(m, AIMessage) for tc in m.tool_calls]
    assert "task" not in tool_calls  # the main model never asked for it
    assert len(_REVIEW_CALLS) == 1  # the review ran exactly once, on the aux model
    skills = _skills(home)
    assert [p.parent.name for p in skills] == [DEMO_SKILL_NAME]
    assert (home / "memories" / "USER.md").read_text(encoding="utf-8").strip()
    # Counters reset, nothing left pending.
    assert state.get("pending_review_kind") is None
    assert state.get("iters_since_skill") == 0
    assert state.get("turns_since_memory") == 0
    # The review's own messages stay out of the main conversation.
    assert state["messages"][-1].content == "Done."


def test_no_review_below_threshold(home: Path):
    _run(home, _NeverReviewsModel(), skill_interval=10, memory_interval=10)
    assert _REVIEW_CALLS == []
    assert _skills(home) == []


def test_no_double_review_when_model_calls_task_review_itself(home: Path):
    """A model that delegates to the review subagent itself already closed the loop.

    The ``task(review)`` call resets the counters, so ``after_agent`` doesn't run a
    second review on the same turn.
    """
    state = _run(home, DemoMainModel(nudge_interval=3, delegate_review=True), skill_interval=3, memory_interval=3)
    tool_calls = [tc["name"] for m in state["messages"] if isinstance(m, AIMessage) for tc in m.tool_calls]
    assert "task" in tool_calls
    assert len(_REVIEW_CALLS) == 1
    assert [p.parent.name for p in _skills(home)] == [DEMO_SKILL_NAME]
    assert state.get("pending_review_kind") is None
    assert state.get("iters_since_skill") == 0


async def test_due_review_runs_on_the_async_path(home: Path):
    """chat drives the graph asynchronously (AG-UI); the review runs there too."""
    ws = home / "ws"
    ws.mkdir(exist_ok=True)
    cfg = HermesConfig.resolve(overrides={"skills_creation_nudge_interval": 3, "memory_nudge_interval": 1})
    agent = create_hermes_agent(
        cfg, workspace=ws, session_id="a179", model=_NeverReviewsModel(), aux_model=_CountingReviewModel()
    )
    try:
        await agent.ainvoke(
            {"messages": [{"role": "user", "content": "go"}], "session_id": "a179", "iteration_budget_remaining": 30},
            config={"configurable": {"thread_id": "a179"}},
        )
    finally:
        _close_agent_resources(agent)
    assert len(_REVIEW_CALLS) == 1
    assert [p.parent.name for p in _skills(home)] == [DEMO_SKILL_NAME]


def test_unwired_review_logs_a_warning(caplog: pytest.LogCaptureFixture):
    """A host that builds ReflectionMiddleware without a review graph is told so."""
    mw = ReflectionMiddleware(library=None, store=None, model=None)
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content="ok")], "pending_review_kind": "skills"}
    with caplog.at_level(logging.WARNING, logger="langstage_hermes.reflection"):
        update = mw.after_agent(state)
    assert update == {"pending_review_kind": None, "iters_since_skill": 0}
    assert any(r.levelno == logging.WARNING and "review" in r.getMessage() for r in caplog.records)
