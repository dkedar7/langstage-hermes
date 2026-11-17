"""gh #224: a skill the reflection review writes is audited as ``review``.

``skill_manage`` stamped ``agent`` on every mutation, whoever was behind the
call, so ``audit log`` filed a skill the background review wrote on its own next
to the ones the user asked for. The same review's memory writes already said
``review`` (gh #222). ``audit log`` also printed the ``session:`` line for memory
rows only.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langstage_hermes import hermes_middleware
from langstage_hermes.cli import cli
from langstage_hermes.curator import CuratorMiddleware, _save_curator_state
from langstage_hermes.reflection import ReflectionMiddleware
from langstage_hermes.skills.audit import SkillAuditLog
from langstage_hermes.skills.tools import _skill_manage_impl

BODY = "# Order lookup\n\n1. Call lookup_order.\n2. Report the status.\n"


class _Scripted(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


class _Aux(FakeListChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


@pytest.fixture
def home(tmp_hermes_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return tmp_hermes_home


@pytest.fixture
def mws(home: Path) -> Any:
    stack = hermes_middleware(aux_model=_Aux(responses=["ok"] * 10), session_id="default-session")
    try:
        yield stack
    finally:
        stack.close()


def _manage(library: Any, action: str, name: str, session_id: str | None = None, **kw: Any) -> None:
    args = {
        "description": "Look up an order status by id.",
        "body": BODY,
        "category": "",
        "old_str": "",
        "new_str": "",
        "frontmatter_data": None,
    }
    args.update(kw)
    _skill_manage_impl(library, action=action, name=name, tool_call_id=f"c-{action}-{name}", session_id=session_id, **args)


def _create(library: Any, name: str, session_id: str | None = None) -> None:
    _manage(library, "create", name, session_id)


def _row(mws: Any, name: str) -> Any:
    (row,) = mws.library.audit_log.list(skill_name=name)
    return row


class _ReviewGraph:
    """Stands in for the review fork: it edits the shared library through ``skill_manage``."""

    def __init__(self, library: Any, name: str = "reviewed-skill", before_write: Any = None) -> None:
        self.library = library
        self.name = name
        self.before_write = before_write

    def invoke(self, _input: Any) -> dict[str, Any]:
        if self.before_write is not None:
            self.before_write()
        _create(self.library, self.name)
        return {"messages": []}

    async def ainvoke(self, input: Any) -> dict[str, Any]:
        return self.invoke(input)


def _review(mws: Any, graph: Any, *, use_async: bool = False, session_id: str = "chat-7") -> None:
    mw = ReflectionMiddleware(library=mws.library, store=mws.store, model=None, review_graph=graph)
    state = {"messages": [], "pending_review_kind": "skills", "session_id": session_id}
    if use_async:
        asyncio.run(mw.aafter_agent(state))
    else:
        mw.after_agent(state)


# ── who wrote it ─────────────────────────────────────────────────────


@pytest.mark.parametrize("use_async", [False, True])
def test_a_skill_the_review_writes_is_labelled_review_under_the_run_session(mws: Any, use_async: bool) -> None:
    _review(mws, _ReviewGraph(mws.library), use_async=use_async)
    row = _row(mws, "reviewed-skill")
    assert (row.action, row.source, row.session_id) == ("create", "review", "chat-7")


def test_a_patch_by_the_review_is_labelled_review_and_the_original_row_is_left_alone(mws: Any) -> None:
    _create(mws.library, "order-lookup", "chat-1")

    class _Patcher:
        def invoke(self, _input: Any) -> dict[str, Any]:
            _manage(mws.library, "patch", "order-lookup", old_str="Report the status.", new_str="Report the status and ETA.")
            return {"messages": []}

    _review(mws, _Patcher(), session_id="chat-2")
    patch, create = mws.library.audit_log.list(skill_name="order-lookup")
    assert (patch.action, patch.source, patch.session_id) == ("patch", "review", "chat-2")
    assert (create.action, create.source, create.session_id) == ("create", "agent", "chat-1")  # not rewritten


def test_the_agent_stays_agent_after_a_review_in_the_same_process(mws: Any) -> None:
    """The label ends with the review: the next skill the agent saves is its own."""
    _review(mws, _ReviewGraph(mws.library))
    assert _row(mws, "reviewed-skill").source == "review"

    args = {"action": "create", "name": "asked-for", "description": "Look up an order status by id.", "body": BODY}
    call = AIMessage("", tool_calls=[{"name": "skill_manage", "id": "c1", "args": args}])
    model = _Scripted(disable_streaming=True, messages=iter([call, *[AIMessage("done")] * 6]))
    agent = create_agent(model, tools=[], middleware=[*mws], checkpointer=InMemorySaver())
    agent.invoke(
        {"messages": [{"role": "user", "content": "save that as a skill"}], "session_id": "chat-8"},
        {"configurable": {"thread_id": "t8"}},
    )
    row = _row(mws, "asked-for")
    assert (row.source, row.session_id) == ("agent", "chat-8")


def test_a_review_that_raises_does_not_leave_the_label_behind(mws: Any) -> None:
    class _Boom:
        def invoke(self, _input: Any) -> dict[str, Any]:
            raise RuntimeError("review model is down")

    try:
        _review(mws, _Boom())
    except RuntimeError:
        pass  # whether the middleware swallows it is not this test's business
    _create(mws.library, "after-failure")
    assert _row(mws, "after-failure").source == "agent"


def test_agent_write_while_a_review_is_running_in_another_thread_stays_agent(mws: Any) -> None:
    """The label is context-scoped, not a field on the library every run shares."""
    in_review, agent_done = threading.Event(), threading.Event()

    def before_write() -> None:
        in_review.set()  # the review block is open ...
        assert agent_done.wait(10)  # ... and stays open while the agent writes

    review = threading.Thread(target=_review, args=(mws, _ReviewGraph(mws.library, before_write=before_write)))
    review.start()
    try:
        assert in_review.wait(10)
        _create(mws.library, "agent-during-review", "chat-agent")
    finally:
        agent_done.set()
        review.join(10)
    during = _row(mws, "agent-during-review")
    assert (during.source, during.session_id) == ("agent", "chat-agent")
    reviewed = _row(mws, "reviewed-skill")
    assert (reviewed.source, reviewed.session_id) == ("review", "chat-7")


def test_concurrent_review_and_agent_writes_keep_their_own_label(mws: Any) -> None:
    barrier = threading.Barrier(2)

    def as_review() -> None:
        with mws.library.mutation_source("review"):
            barrier.wait(10)
            _create(mws.library, "skill-review")

    def as_agent() -> None:
        barrier.wait(10)
        _create(mws.library, "skill-agent")

    threads = [threading.Thread(target=fn) for fn in (as_review, as_agent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert _row(mws, "skill-review").source == "review"
    assert _row(mws, "skill-agent").source == "agent"


def test_interleaved_async_tasks_keep_their_own_label(mws: Any) -> None:
    async def main() -> None:
        review_open, agent_done = asyncio.Event(), asyncio.Event()

        async def as_review() -> None:
            with mws.library.mutation_source("review"):
                review_open.set()
                await agent_done.wait()
                _create(mws.library, "task-review")

        async def as_agent() -> None:
            await review_open.wait()
            _create(mws.library, "task-agent")
            agent_done.set()

        await asyncio.gather(as_review(), as_agent())

    asyncio.run(main())
    assert _row(mws, "task-review").source == "review"
    assert _row(mws, "task-agent").source == "agent"


def test_mutation_source_none_keeps_the_current_label(mws: Any) -> None:
    with mws.library.mutation_source("review"), mws.library.mutation_source(None):
        _create(mws.library, "nested")
    assert _row(mws, "nested").source == "review"
    with mws.library.mutation_source(""):
        _create(mws.library, "blank")
    assert _row(mws, "blank").source == "agent"


def test_a_stand_in_library_without_the_scope_still_gets_its_review(home: Path) -> None:
    ran = []

    class _Graph:
        def invoke(self, _input: Any) -> dict[str, Any]:
            ran.append(True)
            return {"messages": []}

    mw = ReflectionMiddleware(library=object(), store=None, model=None, review_graph=_Graph())
    mw.after_agent({"messages": [], "pending_review_kind": "skills", "session_id": "chat-7"})
    assert ran == [True]


def test_curator_changes_keep_the_label_they_had(mws: Any) -> None:
    """Only the reflection review is relabelled; the curator pass is untouched by this change."""
    import time

    class _CuratorGraph:
        def invoke(self, _input: Any) -> dict[str, Any]:
            _create(mws.library, "curated-skill")
            return {"messages": [AIMessage("consolidated")]}

    long_ago = time.time() - 400 * 3600
    _save_curator_state(mws.store, {"last_run_at": long_ago, "last_user_activity": long_ago})
    CuratorMiddleware(mws.library, mws.store, curator_graph=_CuratorGraph()).before_agent(
        {"messages": [], "session_id": "chat-7"}
    )
    row = _row(mws, "curated-skill")
    assert (row.source, row.session_id) == ("agent", "chat-7")


def test_shipped_review_wiring_labels_its_skill_review(home: Path) -> None:
    """The real ``create_hermes_agent`` review fork and the real ``skill_manage`` tool (demo models, no mocks)."""
    from langstage_hermes import HermesConfig, create_hermes_agent
    from langstage_hermes.demo import DEMO_SKILL_NAME, DemoMainModel, DemoReviewModel, _close_agent_resources

    ws = home / "ws"
    ws.mkdir()
    cfg = HermesConfig.resolve(overrides={"skills_creation_nudge_interval": 3, "memory_nudge_interval": 1})
    agent = create_hermes_agent(cfg, workspace=ws, session_id="r224", model=DemoMainModel(), aux_model=DemoReviewModel())
    try:
        agent.invoke(
            {
                "messages": [{"role": "user", "content": "organise the notes"}],
                "session_id": "r224",
                "iteration_budget_remaining": 30,
            },
            config={"configurable": {"thread_id": "r224"}},
        )
    finally:
        _close_agent_resources(agent)
    log = SkillAuditLog(home / "state.db")
    try:
        rows = log.list(skill_name=DEMO_SKILL_NAME)
    finally:
        log.close()
    assert rows, "the review fork wrote no skill"
    assert {(r.source, r.session_id) for r in rows} == {("review", "r224")}


# ── the CLI ──────────────────────────────────────────────────────────


def _record(home: Path, name: str, *, source: str | None, session_id: str | None, action: str = "create") -> int:
    log = SkillAuditLog(home / "state.db")
    try:
        return log.record(
            skill_name=name,
            action=action,
            before_content=None,
            after_content=b"---\nname: x\n---\nbody\n",
            source=source,
            session_id=session_id,
            tool_call_id=None,
            skill_path=None,
        )
    finally:
        log.close()


def test_issue_repro_demo_then_audit_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The issue's steps: ``demo --keep-workspace``, then ``audit log`` on the kept home."""
    import tempfile

    for var in ("HERMES_HOME", "LANGSTAGE_HERMES_HOME", "DEEPAGENT_HERMES_HOME"):
        monkeypatch.delenv(var, raising=False)
    scratch = tmp_path / "tmproot"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    runner = CliRunner()
    assert runner.invoke(cli, ["demo", "--keep-workspace"]).exit_code == 0
    (kept,) = scratch.glob("dah-demo-*")
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(kept))

    data = json.loads(runner.invoke(cli, ["audit", "log", "--json"]).stdout)
    by_kind = {("memory" if m["skill_name"].startswith("memory:") else "skill"): m for m in data["mutations"]}
    assert set(by_kind) == {"memory", "skill"}
    assert {(m["source"], m["session_id"]) for m in by_kind.values()} == {("review", "demo-001")}

    text = runner.invoke(cli, ["audit", "log", "--skills"]).output.splitlines()
    assert text[0].endswith("(review)") and "create" in text[0]
    assert text[1] == "    session: demo-001"


def test_log_prints_the_session_line_for_skill_rows(home: Path) -> None:
    _record(home, "order-lookup", source="review", session_id="chat-7")
    out = CliRunner().invoke(cli, ["audit", "log"]).output
    lines = out.splitlines()
    assert lines[0].endswith("order-lookup  (review)")
    assert lines[1:] == ["    session: chat-7"]


def test_log_prints_no_session_line_and_no_blank_line_for_a_row_without_one(home: Path) -> None:
    """Rows from before sessions were recorded, and CLI changes, have none."""
    _record(home, "old-skill", source="agent", session_id=None)
    _record(home, "no-id", source="cli", session_id="")
    out = CliRunner().invoke(cli, ["audit", "log"]).output
    lines = out.splitlines()
    assert len(lines) == 2, out
    assert "session:" not in out
    assert all(line.strip() for line in lines)


def test_log_full_prints_the_session_once(home: Path) -> None:
    _record(home, "order-lookup", source="review", session_id="chat-7")
    out = CliRunner().invoke(cli, ["audit", "log", "--full"]).output
    assert out.count("session: chat-7") == 1


def test_log_json_keys_are_unchanged_and_only_the_source_value_is_new(home: Path) -> None:
    _record(home, "order-lookup", source="review", session_id="chat-7")
    _record(home, "older", source="agent", session_id="chat-1")
    data = json.loads(CliRunner().invoke(cli, ["audit", "log", "--json"]).stdout)
    assert set(data) == {"mutations", "count"}
    keys = {
        "id",
        "timestamp",
        "skill_name",
        "action",
        "source",
        "session_id",
        "tool_call_id",
        "skill_path",
        "before_hash",
        "after_hash",
    }
    assert all(set(m) == keys for m in data["mutations"])
    assert [(m["skill_name"], m["source"], m["session_id"]) for m in data["mutations"]] == [
        ("older", "agent", "chat-1"),
        ("order-lookup", "review", "chat-7"),
    ]


def test_rollback_and_diff_work_on_a_review_row(mws: Any, home: Path) -> None:
    """Undoing what the review wrote is the point of labelling it."""
    _create(mws.library, "order-lookup", "chat-1")

    class _Patcher:
        def invoke(self, _input: Any) -> dict[str, Any]:
            _manage(mws.library, "patch", "order-lookup", old_str="Report the status.", new_str="Guess the status.")
            return {"messages": []}

    _review(mws, _Patcher(), session_id="chat-2")
    patch = mws.library.audit_log.list(skill_name="order-lookup")[0]
    assert patch.source == "review"
    skill_md = next((home / "skills").rglob("SKILL.md"))
    assert "Guess the status." in skill_md.read_text(encoding="utf-8")

    runner = CliRunner()
    diff = runner.invoke(cli, ["audit", "diff", "order-lookup", str(patch.id)])
    assert diff.exit_code == 0, diff.output
    res = runner.invoke(cli, ["audit", "rollback", "order-lookup", str(patch.id), "--yes"])
    assert res.exit_code == 0, res.output
    text = skill_md.read_text(encoding="utf-8")
    assert "Report the status." in text and "Guess the status." not in text

    log = SkillAuditLog(home / "state.db")
    try:
        rollback, patched, created = log.list(skill_name="order-lookup")
    finally:
        log.close()
    assert (rollback.action, rollback.source) == ("rollback", "cli")  # the CLI keeps its own label
    assert (patched.source, created.source) == ("review", "agent")
