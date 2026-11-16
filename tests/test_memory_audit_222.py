"""gh #222: memory changes are recorded in the audit log and can be undone.

``USER.md`` / ``MEMORY.md`` reach every later session's system prompt, and the
agent rewrites them on its own. Before the fix a ``replace`` or ``remove``
overwrote the file in place: the earlier text was gone, ``audit log`` said "No
skill mutations recorded yet", and nothing said when or in which conversation
the change happened. Skills already had a log and a rollback; memory had
neither.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage

from langstage_hermes.cli import cli
from langstage_hermes.memory import audit as memory_audit
from langstage_hermes.memory.tool import ENTRY_DELIMITER, MemoryToolMiddleware, _make_memory_tool, build_snapshot
from langstage_hermes.reflection import ReflectionMiddleware
from langstage_hermes.skills.audit import RollbackError, SkillAuditLog

FAILOVER = "Prod DB failover: page the on-call DBA; never run migrations on Fridays."
CONCISE = "Prefers concise answers."
ACME = "Works at Acme on the billing team."

FIXTURE_053 = Path(__file__).parent / "fixtures" / "state_0_5_3.db"


@pytest.fixture
def home(tmp_hermes_home: Path) -> Path:
    return tmp_hermes_home


def _tool(memory_char_limit: int = 2200, user_char_limit: int = 1375) -> Any:
    return _make_memory_tool(memory_char_limit=memory_char_limit, user_char_limit=user_char_limit)


_CALLS = iter(range(1, 10_000))


def _call(tool: Any, **args: Any) -> dict[str, Any]:
    """Invoke the memory tool the way LangChain does and return its JSON payload."""
    cmd = tool.invoke({"args": args, "name": "memory", "type": "tool_call", "id": f"c{next(_CALLS)}"})
    return json.loads(cmd.update["messages"][0].content)


def _log(home: Path) -> SkillAuditLog:
    return SkillAuditLog(home / "state.db")


def _rows(home: Path, **kw: Any) -> list[Any]:
    log = _log(home)
    try:
        return log.list(**kw)
    finally:
        log.close()


def _entries(home: Path, target: str = "user") -> list[str]:
    path = home / "memories" / ("USER.md" if target == "user" else "MEMORY.md")
    if not path.is_file():
        return []
    return [e.strip() for e in path.read_text(encoding="utf-8").split(ENTRY_DELIMITER) if e.strip()]


def _issue_repro() -> Any:
    """The issue's evidence script: a note is added, then the agent replaces it."""
    t = _tool()
    _call(t, action="add", target="user", entry=FAILOVER)
    _call(t, action="replace", target="user", index=0, entry=CONCISE)
    return t


# ── recording ────────────────────────────────────────────────────────


def test_issue_repro_replaced_text_is_recorded_and_recoverable(home: Path) -> None:
    _issue_repro()
    assert _entries(home) == [CONCISE]  # the note is gone from USER.md, as before

    replace, add = _rows(home)
    assert (add.action, replace.action) == ("add", "replace")
    assert replace.skill_name == add.skill_name == "memory:user"
    assert (replace.kind, replace.memory_target) == ("memory", "user")
    # ... but the log still has it, and knows what replaced it.
    assert memory_audit.entries_of(replace.before_content) == [FAILOVER]
    assert memory_audit.entries_of(replace.after_content) == [CONCISE]
    assert memory_audit.describe_change(replace) == ([FAILOVER], [CONCISE])
    assert add.before_content is None  # the file did not exist yet
    assert replace.source == "agent"
    assert replace.tool_call_id and replace.tool_call_id.startswith("c")
    assert Path(replace.skill_path) == home / "memories" / "USER.md"

    res = CliRunner().invoke(cli, ["audit", "log"])
    assert res.exit_code == 0, res.output
    assert "No skill mutations recorded yet" not in res.output
    assert "memory:user" in res.output
    assert f"- {FAILOVER[:40]}" in res.output
    assert f"+ {CONCISE}" in res.output

    res = CliRunner().invoke(cli, ["audit", "rollback", "memory:user", str(replace.id), "--yes"])
    assert res.exit_code == 0, res.output
    assert _entries(home) == [FAILOVER]


def test_remove_and_memory_target_are_recorded(home: Path) -> None:
    t = _tool()
    _call(t, action="add", target="memory", entry="CI runs on ubuntu-latest.")
    _call(t, action="add", target="memory", entry="Docs live at docs.example.org.")
    _call(t, action="remove", target="memory", index=0)

    remove, _add2, _add1 = _rows(home, skill_name="memory:memory")
    assert remove.action == "remove"
    assert memory_audit.describe_change(remove) == (["CI runs on ubuntu-latest."], [])
    assert memory_audit.entries_of(remove.after_content) == ["Docs live at docs.example.org."]
    assert _rows(home, skill_name="memory:user") == []


def test_calls_that_change_nothing_record_nothing(home: Path) -> None:
    t = _tool(user_char_limit=40)
    _call(t, action="add", target="user", entry=CONCISE)
    assert len(_rows(home)) == 1
    assert _call(t, action="add", target="user", entry=CONCISE)["message"].startswith("Entry already exists")
    assert _call(t, action="add", target="user", entry=FAILOVER)["success"] is False  # over the limit
    assert _call(t, action="replace", target="user", index=5, entry="x")["success"] is False
    assert _call(t, action="remove", target="user", index=5)["success"] is False
    assert _call(t, action="replace", target="user", index=0, entry=CONCISE)["success"] is True  # same text
    assert _call(t, action="read", target="user")["entries"] == [CONCISE]
    assert len(_rows(home)) == 1


def test_audit_failure_never_fails_the_write(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> Any:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(memory_audit, "_open_log", _boom)
    payload = _call(_tool(), action="add", target="user", entry=CONCISE)
    assert payload["success"] is True
    assert _entries(home) == [CONCISE]
    assert "not recorded in the audit log" in payload["audit_note"]
    assert "database is locked" in payload["audit_note"]


def test_unusable_state_db_never_fails_the_write(home: Path) -> None:
    (home / "state.db").mkdir()  # sqlite cannot open a directory
    payload = _call(_tool(), action="add", target="user", entry=CONCISE)
    assert payload["success"] is True
    assert _entries(home) == [CONCISE]
    assert "audit_note" in payload


def test_a_recorded_write_carries_no_audit_note(home: Path) -> None:
    assert "audit_note" not in _call(_tool(), action="add", target="user", entry=CONCISE)


def test_recording_leaves_the_frozen_snapshot_alone(home: Path) -> None:
    mw = MemoryToolMiddleware()
    _call(mw.tools[0], action="add", target="user", entry=CONCISE)
    state: dict[str, Any] = {"messages": [], **(mw.before_agent({"messages": []}, None) or {})}
    frozen = (state["memory_snapshot"], state["user_snapshot"])

    cmd = mw.tools[0].invoke(
        {
            "args": {"action": "replace", "target": "user", "index": 0, "entry": ACME, "state": state},
            "name": "memory",
            "type": "tool_call",
            "id": "c-snap",
        }
    )
    # The tool's own update never carries a snapshot key, and the state it was
    # handed is untouched; only a later session sees the new text.
    assert set(cmd.update) == {"messages", "turns_since_memory"}
    assert (state["memory_snapshot"], state["user_snapshot"]) == frozen
    assert mw.before_agent({**state, "memory_snapshot_loaded": True}, None) is None
    assert ACME in build_snapshot()["user_snapshot"]
    assert len(_rows(home)) == 2


def test_rows_follow_a_swapped_home(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    t = _tool()
    _call(t, action="add", target="user", entry=CONCISE)
    other = tmp_path / "other_home"
    other.mkdir()
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(other))
    monkeypatch.setenv("HERMES_HOME", str(other))
    _call(t, action="add", target="user", entry=ACME)
    assert [memory_audit.describe_change(r)[1] for r in _rows(home)] == [[CONCISE]]
    assert [memory_audit.describe_change(r)[1] for r in _rows(other)] == [[ACME]]


# ── provenance: session and source ───────────────────────────────────


class _Scripted(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


class _Aux(FakeListChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


def test_full_agent_records_the_run_session_and_agent_source(home: Path) -> None:
    from langstage_hermes import HermesConfig
    from langstage_hermes.agent import create_hermes_agent
    from langstage_hermes.demo import _close_agent_resources

    args = {"action": "add", "target": "user", "entry": CONCISE}
    call = AIMessage("", tool_calls=[{"name": "memory", "id": "m1", "args": args}])
    model = _Scripted(disable_streaming=True, messages=iter([call, *[AIMessage("done")] * 6]))
    graph = create_hermes_agent(HermesConfig.resolve(), model=model, aux_model=_Aux(responses=["ok"] * 10), workspace=str(home))
    try:
        graph.invoke(
            {"messages": [{"role": "user", "content": "remember I like short answers"}], "session_id": "chat-9"},
            {"configurable": {"thread_id": "t1"}},
        )
    finally:
        _close_agent_resources(graph)
    (row,) = _rows(home, kind="memory")
    assert (row.action, row.source, row.session_id, row.tool_call_id) == ("add", "agent", "chat-9", "m1")


class _ReviewGraph:
    """Stands in for the review fork: one memory write, no session in its own state."""

    def __init__(self) -> None:
        self.tool = _tool()

    def invoke(self, _input: Any) -> dict[str, Any]:
        _call(self.tool, action="add", target="user", entry=CONCISE)
        return {"messages": []}

    async def ainvoke(self, input: Any) -> dict[str, Any]:
        return self.invoke(input)


@pytest.mark.parametrize("use_async", [False, True])
def test_review_fork_writes_are_labelled_review_under_the_run_session(home: Path, use_async: bool) -> None:
    mw = ReflectionMiddleware(library=None, store=None, model=None, review_graph=_ReviewGraph())
    state = {"messages": [], "pending_review_kind": "memory", "session_id": "chat-7"}
    if use_async:
        asyncio.run(mw.aafter_agent(state))
    else:
        mw.after_agent(state)
    (row,) = _rows(home)
    assert (row.source, row.session_id) == ("review", "chat-7")
    # The label is scoped to the review: a later write by the agent is its own.
    _call(_tool(), action="add", target="user", entry=ACME)
    assert _rows(home)[0].source == "agent"
    assert _rows(home)[0].session_id is None


def test_shipped_review_wiring_records_review_source(home: Path) -> None:
    """The real ``create_hermes_agent`` review fork (demo models, no mocks)."""
    from langstage_hermes import HermesConfig, create_hermes_agent
    from langstage_hermes.demo import DemoMainModel, DemoReviewModel, _close_agent_resources

    ws = home / "ws"
    ws.mkdir()
    cfg = HermesConfig.resolve(overrides={"skills_creation_nudge_interval": 3, "memory_nudge_interval": 1})
    agent = create_hermes_agent(cfg, workspace=ws, session_id="r222", model=DemoMainModel(), aux_model=DemoReviewModel())
    try:
        agent.invoke(
            {
                "messages": [{"role": "user", "content": "organise the notes"}],
                "session_id": "r222",
                "iteration_budget_remaining": 30,
            },
            config={"configurable": {"thread_id": "r222"}},
        )
    finally:
        _close_agent_resources(agent)
    rows = _rows(home, kind="memory")
    assert rows, "the review fork wrote no memory"
    assert {(r.source, r.session_id) for r in rows} == {("review", "r222")}


# ── audit log listing ────────────────────────────────────────────────


def _write_skill(home: Path, name: str) -> None:
    d = home / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Look up an order status by id for Acme support.\n---\n# {name}\n\nSteps.\n",
        encoding="utf-8",
    )


def _mixed(home: Path) -> None:
    _write_skill(home, "order-lookup")
    assert CliRunner().invoke(cli, ["skills", "remove", "order-lookup"]).exit_code == 0
    _issue_repro()


def test_log_lists_memory_next_to_skills_and_filters(home: Path) -> None:
    _mixed(home)
    runner = CliRunner()

    both = runner.invoke(cli, ["audit", "log"]).output
    assert "order-lookup" in both and "memory:user" in both

    mem = runner.invoke(cli, ["audit", "log", "--memory"]).output
    assert "memory:user" in mem and "order-lookup" not in mem

    skills = runner.invoke(cli, ["audit", "log", "--skills"]).output
    assert "order-lookup" in skills and "memory:user" not in skills

    one = runner.invoke(cli, ["audit", "log", "--skill", "memory:user"]).output
    assert "memory:user" in one and "order-lookup" not in one

    res = runner.invoke(cli, ["audit", "log", "--memory", "--skills"])
    assert res.exit_code == 64  # the CLI's usage-error code
    assert "mutually exclusive" in res.output


def test_log_json_keeps_skill_rows_as_they_were_and_tags_memory_rows(home: Path) -> None:
    _mixed(home)
    res = CliRunner().invoke(cli, ["audit", "log", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert set(data) == {"mutations", "count"}
    assert data["count"] == len(data["mutations"]) == 3
    skill_keys = {
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
    skill_row = next(m for m in data["mutations"] if m["skill_name"] == "order-lookup")
    assert set(skill_row) == skill_keys  # unchanged for existing consumers
    mem_rows = [m for m in data["mutations"] if m["skill_name"] == "memory:user"]
    assert [m["action"] for m in mem_rows] == ["replace", "add"]
    for m in mem_rows:
        assert set(m) == skill_keys | {"kind", "target"}
        assert (m["kind"], m["target"]) == ("memory", "user")
    # Like SKILL.md bodies, memory text stays out of the listing.
    assert FAILOVER not in res.stdout and CONCISE not in res.stdout

    only = json.loads(CliRunner().invoke(cli, ["audit", "log", "--json", "--memory"]).stdout)
    assert {m["skill_name"] for m in only["mutations"]} == {"memory:user"}


def test_log_messages_when_empty(home: Path) -> None:
    runner = CliRunner()
    assert "No skill mutations recorded yet." in runner.invoke(cli, ["audit", "log"]).output
    assert "No memory mutations recorded yet." in runner.invoke(cli, ["audit", "log", "--memory"]).output
    assert json.loads(runner.invoke(cli, ["audit", "log", "--json", "--memory"]).stdout) == {"mutations": [], "count": 0}


def test_log_shows_the_session_that_made_the_change(home: Path) -> None:
    with memory_audit.memory_mutation_context(session_id="chat-7"):
        _issue_repro()
    out = CliRunner().invoke(cli, ["audit", "log", "--memory"]).output
    assert "session: chat-7" in out


def test_memory_show_history_is_the_memory_log(home: Path) -> None:
    _mixed(home)
    _call(_tool(), action="add", target="memory", entry="CI runs on ubuntu-latest.")
    runner = CliRunner()
    out = runner.invoke(cli, ["memory", "show", "--history"]).output
    assert "memory:user" in out and "memory:memory" in out and "order-lookup" not in out
    out = runner.invoke(cli, ["memory", "show", "--history", "--user"]).output
    assert "memory:user" in out and "memory:memory" not in out
    data = json.loads(runner.invoke(cli, ["memory", "show", "--history", "--session", "--json"]).stdout)
    assert [m["skill_name"] for m in data["mutations"]] == ["memory:memory"]
    # Without the flag `memory show` is what it was.
    assert CONCISE in runner.invoke(cli, ["memory", "show"]).output


def test_audit_show_prints_the_layer_before_and_after(home: Path) -> None:
    _issue_repro()
    replace = _rows(home)[0]
    out = CliRunner().invoke(cli, ["audit", "show", str(replace.id)]).output
    assert "--- before ---" in out and FAILOVER in out
    assert "--- after ---" in out and CONCISE in out
    diff = CliRunner().invoke(cli, ["audit", "show", str(replace.id), "--diff"]).output
    assert f"-{FAILOVER}" in diff and f"+{CONCISE}" in diff


# ── upgrade: a state.db written by 0.5.3 ─────────────────────────────


def test_db_created_by_0_5_3_keeps_its_rows_and_takes_memory_rows(home: Path) -> None:
    """``tests/fixtures/state_0_5_3.db`` was written by langstage-hermes 0.5.3 from PyPI
    (one ``skills remove``). Memory rows share the existing table, so nothing migrates."""
    shutil.copyfile(FIXTURE_053, home / "state.db")
    conn = sqlite3.connect(home / "state.db")
    try:
        old = conn.execute("SELECT * FROM skill_mutations ORDER BY id").fetchall()
        columns = [r[1] for r in conn.execute("PRAGMA table_info(skill_mutations)")]
    finally:
        conn.close()
    assert len(old) == 1

    _issue_repro()

    conn = sqlite3.connect(home / "state.db")
    try:
        assert conn.execute("SELECT * FROM skill_mutations WHERE id = ?", (old[0][0],)).fetchall() == old
        assert [r[1] for r in conn.execute("PRAGMA table_info(skill_mutations)")] == columns
    finally:
        conn.close()

    data = json.loads(CliRunner().invoke(cli, ["audit", "log", "--json"]).stdout)
    assert [(m["skill_name"], m["action"]) for m in data["mutations"]] == [
        ("memory:user", "replace"),
        ("memory:user", "add"),
        ("order-lookup", "delete"),
    ]
    # Ids continue the existing sequence, so an old id never changes meaning.
    assert [m["id"] for m in data["mutations"]] == [3, 2, 1]
    res = CliRunner().invoke(cli, ["audit", "rollback", "memory:user", "3", "--yes"])
    assert res.exit_code == 0, res.output
    assert _entries(home) == [FAILOVER]


# ── rollback ─────────────────────────────────────────────────────────


def _rollback(mutation_id: int, name: str = "memory:user", *args: str, input: str | None = None) -> Any:
    return CliRunner().invoke(cli, ["audit", "rollback", name, str(mutation_id), *args], input=input)


def test_rollback_reverts_one_entry_and_keeps_later_ones(home: Path) -> None:
    t = _issue_repro()  # 1 add, 2 replace
    _call(t, action="add", target="user", entry=ACME)  # 3: a later, unrelated, good entry

    res = _rollback(2, "memory:user", "--yes")
    assert res.exit_code == 0, res.output
    # The replaced note is back in its place and the later entry survives. A
    # whole-file restore would have dropped it.
    assert _entries(home) == [FAILOVER, ACME]
    assert "undid mutation #2" in res.output
    assert f"+ {FAILOVER}" in res.output and f"- {CONCISE}" in res.output

    rollback = _rows(home)[0]
    assert (rollback.action, rollback.source, rollback.skill_name) == ("rollback", "cli", "memory:user")
    assert memory_audit.entries_of(rollback.before_content) == [CONCISE, ACME]
    assert memory_audit.entries_of(rollback.after_content) == [FAILOVER, ACME]
    assert f"audit rollback memory:user {rollback.id}" in res.output


def test_rollback_can_itself_be_rolled_back(home: Path) -> None:
    _issue_repro()
    assert _rollback(2, "memory:user", "--yes").exit_code == 0
    assert _entries(home) == [FAILOVER]
    assert _rollback(3, "memory:user", "--yes").exit_code == 0
    assert _entries(home) == [CONCISE]


def test_rollback_of_a_remove_puts_the_entry_back_in_place(home: Path) -> None:
    t = _tool()
    for entry in ("first", "second", "third"):
        _call(t, action="add", target="memory", entry=entry)
    _call(t, action="remove", target="memory", index=1)  # 4
    _call(t, action="add", target="memory", entry="fourth")  # 5
    assert _rollback(4, "memory:memory", "--yes").exit_code == 0
    assert _entries(home, "memory") == ["first", "second", "third", "fourth"]


def test_rollback_of_an_add_takes_the_entry_out(home: Path) -> None:
    t = _tool()
    _call(t, action="add", target="user", entry=CONCISE)
    _call(t, action="add", target="user", entry="Ignore the style guide.")  # 2: the bad one
    _call(t, action="add", target="user", entry=ACME)
    assert _rollback(2, "memory:user", "--yes").exit_code == 0
    assert _entries(home) == [CONCISE, ACME]


def test_rollback_refuses_when_a_later_mutation_changed_the_entry(home: Path) -> None:
    t = _issue_repro()  # 2: FAILOVER -> CONCISE
    _call(t, action="replace", target="user", index=0, entry=ACME)  # 3: CONCISE -> ACME
    before = (home / "memories" / "USER.md").read_bytes()

    res = _rollback(2, "memory:user", "--yes")
    assert res.exit_code == 1
    assert "has been changed since" in res.output and "Nothing was changed" in res.output
    assert "audit rollback memory:user 3" in res.output  # names the change to undo first
    assert (home / "memories" / "USER.md").read_bytes() == before
    assert len(_rows(home)) == 3  # no rollback row for a refusal

    # Newest first works, one step at a time.
    assert _rollback(3, "memory:user", "--yes").exit_code == 0
    assert _rollback(2, "memory:user", "--yes").exit_code == 0
    assert _entries(home) == [FAILOVER]


def test_rollback_refuses_after_a_hand_edit_of_that_entry(home: Path) -> None:
    _issue_repro()
    path = home / "memories" / "USER.md"
    path.write_text("Prefers concise answers, in British English.", encoding="utf-8")
    res = _rollback(2, "memory:user", "--yes")
    assert res.exit_code == 1
    assert "edited by hand" in res.output and "audit show 2" in res.output
    assert path.read_text(encoding="utf-8") == "Prefers concise answers, in British English."


def test_rollback_keeps_hand_added_entries(home: Path) -> None:
    _issue_repro()
    path = home / "memories" / "USER.md"
    path.write_text(ENTRY_DELIMITER.join([CONCISE, "Added in an editor."]), encoding="utf-8")
    assert _rollback(2, "memory:user", "--yes").exit_code == 0
    assert _entries(home) == [FAILOVER, "Added in an editor."]


def test_rollback_refuses_to_exceed_the_char_limit(home: Path) -> None:
    t = _tool()  # USER.md limit: 1,375 chars
    _call(t, action="add", target="user", entry="x" * 1300)
    _call(t, action="replace", target="user", index=0, entry="short")  # 2
    assert _call(t, action="add", target="user", entry="y" * 1300)["success"] is True  # 3
    before = (home / "memories" / "USER.md").read_bytes()

    res = _rollback(2, "memory:user", "--yes")
    assert res.exit_code == 1
    assert "over its limit" in res.output and "Nothing was changed" in res.output
    assert "/1,375 chars" in res.output
    assert (home / "memories" / "USER.md").read_bytes() == before
    assert len(_rows(home)) == 3


def test_rollback_with_nothing_left_to_undo(home: Path) -> None:
    _issue_repro()
    path = home / "memories" / "USER.md"
    path.write_text(FAILOVER, encoding="utf-8")  # someone already put it back by hand
    res = _rollback(2, "memory:user", "--yes")
    assert res.exit_code == 1
    assert "has been changed since" in res.output or "nothing to undo" in res.output
    assert path.read_text(encoding="utf-8") == FAILOVER


def test_rollback_asks_first_and_cancel_changes_nothing(home: Path) -> None:
    _issue_repro()
    res = _rollback(2, "memory:user", input="n\n")
    assert res.exit_code == 0
    assert "This will change" in res.output and f"+ {FAILOVER}" in res.output
    assert "Cancelled." in res.output
    assert _entries(home) == [CONCISE]
    res = _rollback(2, "memory:user", input="y\n")
    assert res.exit_code == 0
    assert _entries(home) == [FAILOVER]


def test_rollback_rejects_wrong_ids_and_names(home: Path) -> None:
    _mixed(home)  # 1: skill delete, 2: memory add, 3: memory replace
    assert "not found" in _rollback(99, "memory:user", "--yes").output
    res = _rollback(1, "memory:user", "--yes")
    assert res.exit_code == 1 and "'order-lookup', not 'memory:user'" in res.output
    res = _rollback(3, "memory:memory", "--yes")
    assert res.exit_code == 1 and "'memory:user', not 'memory:memory'" in res.output
    res = _rollback(3, "memory:bogus", "--yes")
    assert res.exit_code == 1
    # A memory id under a skill name is refused by the skill path, as before.
    res = _rollback(3, "order-lookup", "--yes")
    assert res.exit_code == 1 and "memory:user" in res.output
    assert _entries(home) == [CONCISE]


def test_rollback_to_api_is_per_entry_for_memory(home: Path) -> None:
    t = _issue_repro()
    _call(t, action="add", target="user", entry=ACME)
    log = _log(home)
    try:
        path = log.rollback_to("memory:user", 2, source="cli")
        assert path == home / "memories" / "USER.md"
        assert _entries(home) == [FAILOVER, ACME]
        with pytest.raises(RollbackError, match="has been changed since"):
            log.rollback_to("memory:user", 2)
    finally:
        log.close()


def test_rollback_writes_the_current_home_not_a_recorded_path(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state.db moved to another home must not write back into the old one."""
    _issue_repro()
    other = tmp_path / "moved_home"
    shutil.copytree(home, other)
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(other))
    monkeypatch.setenv("HERMES_HOME", str(other))
    assert _rollback(2, "memory:user", "--yes").exit_code == 0
    assert _entries(other) == [FAILOVER]
    assert _entries(home) == [CONCISE]


# ── hand edits ───────────────────────────────────────────────────────


def test_hand_edit_is_flagged_and_diffable(home: Path) -> None:
    _issue_repro()
    runner = CliRunner()
    clean = runner.invoke(cli, ["audit", "log"])
    assert "differs from its last recorded change" not in clean.output
    assert "No drift" in runner.invoke(cli, ["audit", "diff", "memory:user", "2"]).output

    (home / "memories" / "USER.md").write_text(ENTRY_DELIMITER.join([CONCISE, "Added in an editor."]), encoding="utf-8")

    res = runner.invoke(cli, ["audit", "log"])
    assert "USER.md on disk differs from its last recorded change (#2)" in res.stderr
    assert "audit diff memory:user 2" in res.stderr
    assert "differs" not in runner.invoke(cli, ["audit", "log", "--skills"]).output

    data = json.loads(runner.invoke(cli, ["audit", "log", "--json"]).stdout)
    assert data["memory_drift"] == [{"target": "user", "last_mutation_id": 2}]

    diff = runner.invoke(cli, ["audit", "diff", "memory:user", "2"]).output
    assert "+Added in an editor." in diff
    assert f"-{CONCISE}" not in diff


def test_plan_revert_is_pure(home: Path) -> None:
    _issue_repro()
    log = _log(home)
    try:
        plan = memory_audit.preview_revert(log, "memory:user", 2)
    finally:
        log.close()
    assert (plan.put_back, plan.taken_out) == ([FAILOVER], [CONCISE])
    assert _entries(home) == [CONCISE]
    assert len(_rows(home)) == 2
