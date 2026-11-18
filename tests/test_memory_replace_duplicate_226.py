"""gh #226: ``replace`` could store an entry twice, and ``audit rollback`` then reordered the layer.

``add`` refuses text the layer already holds ("Entry already exists"), but
``replace`` did not check: replacing entry *i* with the text of entry *j* left
the same entry in ``USER.md`` / ``MEMORY.md`` twice, counted twice against the
char budget. ``audit rollback`` of that change then took out the *first* entry
with that text instead of the one the mutation wrote, so the layer came back
with the right entries in the wrong order.

Two fixes are pinned here. ``replace`` refuses a repeat and says what to do
instead. The rollback works on positions: with nothing changed since, it gives
back the recorded "before" exactly, whether or not the layer repeats an entry
(layers written by 0.5.4 / 0.5.5, or by hand, still can).
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.memory import audit as memory_audit
from langstage_hermes.memory.tool import ENTRY_DELIMITER, _make_memory_tool
from langstage_hermes.skills.audit import MutationRow, RollbackError, SkillAuditLog

CONCISE = "Prefers concise answers."
VIM = "Uses vim."
ACME = "Works at Acme on the billing team."

FIXTURE_055 = Path(__file__).parent / "fixtures" / "state_0_5_5_duplicate.db"


@pytest.fixture
def home(tmp_hermes_home: Path) -> Path:
    return tmp_hermes_home


def _tool(memory_char_limit: int = 2200, user_char_limit: int = 1375) -> Any:
    return _make_memory_tool(memory_char_limit=memory_char_limit, user_char_limit=user_char_limit)


_CALLS = iter(range(1, 1_000_000))


def _call(tool: Any, **args: Any) -> dict[str, Any]:
    cmd = tool.invoke({"args": args, "name": "memory", "type": "tool_call", "id": f"c{next(_CALLS)}"})
    return json.loads(cmd.update["messages"][0].content)


def _path(home: Path, target: str = "user") -> Path:
    return home / "memories" / ("USER.md" if target == "user" else "MEMORY.md")


def _entries(home: Path, target: str = "user") -> list[str]:
    path = _path(home, target)
    if not path.is_file():
        return []
    return [e.strip() for e in path.read_text(encoding="utf-8").split(ENTRY_DELIMITER) if e.strip()]


def _write(home: Path, entries: list[str], target: str = "user") -> None:
    """Write a layer the way an editor would: no audit row."""
    path = _path(home, target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


def _rows(home: Path) -> list[Any]:
    log = SkillAuditLog(home / "state.db")
    try:
        return log.list(limit=1000)
    finally:
        log.close()


def _rollback(mutation_id: int, name: str = "memory:user") -> Any:
    return CliRunner().invoke(cli, ["audit", "rollback", name, str(mutation_id), "--yes"])


def _row(before: list[str], after: list[str], *, id: int = 7, action: str = "replace") -> MutationRow:
    """An audit row as the memory tool records one: the whole layer before and after."""

    def enc(entries: list[str]) -> bytes | None:
        return ENTRY_DELIMITER.join(entries).encode("utf-8") if entries else None

    return MutationRow(
        id=id,
        timestamp=0.0,
        skill_name="memory:user",
        action=action,
        source="agent",
        session_id=None,
        tool_call_id=None,
        skill_path=None,
        before_hash=None,
        after_hash=None,
        before_content=enc(before),
        after_content=enc(after),
    )


def _revert(before: list[str], after: list[str], current: list[str], **kw: Any) -> memory_audit.RevertPlan:
    return memory_audit.plan_revert(_row(before, after), current, limit=kw.pop("limit", 10**9), **kw)


# ── replace refuses a repeat ─────────────────────────────────────────


def test_issue_repro_replace_refuses_text_another_entry_already_holds(home: Path) -> None:
    t = _tool()
    _call(t, action="add", target="user", entry=CONCISE)
    _call(t, action="add", target="user", entry=VIM)
    before = _path(home).read_bytes()

    res = _call(t, action="replace", target="user", index=1, entry=CONCISE)

    assert res["success"] is False
    assert "Entry 0 already says that" in res["error"] and "Nothing was changed" in res["error"]
    # It says how to get the merge the caller was after, with the real index.
    assert "memory(action='remove', target='user', index=1)" in res["error"]
    assert res["entries"] == [CONCISE, VIM]
    assert _path(home).read_bytes() == before
    assert [r.action for r in _rows(home)] == ["add", "add"]  # a refusal records nothing

    # ``add`` and ``replace`` now agree, and the suggested call does the merge.
    assert _call(t, action="add", target="user", entry=CONCISE)["message"].startswith("Entry already exists")
    assert _call(t, action="remove", target="user", index=1)["entries"] == [CONCISE]


def test_replace_uses_the_same_notion_of_same_entry_as_add(home: Path) -> None:
    """``add`` compares the stripped text exactly: case and inner spacing count."""
    t = _tool()
    _call(t, action="add", target="user", entry=CONCISE)
    _call(t, action="add", target="user", entry=VIM)
    # Surrounding whitespace is stripped by both, so this is a repeat for both.
    assert _call(t, action="add", target="user", entry=f"  {CONCISE}\n")["entry_count"] == 2
    assert _call(t, action="replace", target="user", index=1, entry=f"  {CONCISE}\n")["success"] is False
    # A different case is a different entry for ``add``, so ``replace`` allows it too.
    assert _call(t, action="add", target="user", entry=CONCISE.upper())["entry_count"] == 3
    res = _call(t, action="replace", target="user", index=1, entry=CONCISE.lower())
    assert res["success"] is True and res["entries"] == [CONCISE, CONCISE.lower(), CONCISE.upper()]


def test_replace_with_its_own_text_is_still_a_quiet_no_op(home: Path) -> None:
    t = _tool()
    _call(t, action="add", target="memory", entry=CONCISE)
    res = _call(t, action="replace", target="memory", index=0, entry=CONCISE)
    assert res["success"] is True and "already says that" in res["message"]
    assert len(_rows(home)) == 1


def test_replace_refusal_applies_to_both_layers_and_names_the_target(home: Path) -> None:
    t = _tool()
    for entry in (ACME, VIM, CONCISE):
        _call(t, action="add", target="memory", entry=entry)
    res = _call(t, action="replace", target="memory", index=0, entry=CONCISE)
    assert res["success"] is False
    assert "Entry 2 already says that" in res["error"]
    assert "memory(action='remove', target='memory', index=0)" in res["error"]
    assert _entries(home, "memory") == [ACME, VIM, CONCISE]


def test_a_layer_that_already_repeats_an_entry_can_be_cleaned_up(home: Path) -> None:
    """0.5.4 / 0.5.5 (or an editor) may have left a repeat. The tool refuses to add
    another, lets the repeat be removed, and never rewrites the layer on its own."""
    _write(home, [CONCISE, VIM, CONCISE])
    t = _tool()
    assert _call(t, action="read", target="user")["entries"] == [CONCISE, VIM, CONCISE]
    assert _call(t, action="replace", target="user", index=1, entry=CONCISE)["success"] is False
    # Replacing one copy with its own text is the no-op, not a refusal.
    assert "already says that" in _call(t, action="replace", target="user", index=2, entry=CONCISE)["message"]
    assert _entries(home) == [CONCISE, VIM, CONCISE]
    assert _call(t, action="remove", target="user", index=2)["entries"] == [CONCISE, VIM]


def test_the_tool_description_tells_the_model_about_the_refusal() -> None:
    assert "refused if another entry already says exactly that" in _tool().description


def test_review_fork_gets_the_same_refusal(home: Path) -> None:
    """The reflection review writes through the same tool, under its own label."""
    t = _tool()
    _call(t, action="add", target="user", entry=CONCISE)
    _call(t, action="add", target="user", entry=VIM)
    with memory_audit.memory_mutation_context(source="review", session_id="s-1"):
        res = _call(t, action="replace", target="user", index=1, entry=CONCISE)
    assert res["success"] is False
    assert _entries(home) == [CONCISE, VIM]
    assert [r.source for r in _rows(home)] == ["agent", "agent"]


# ── rollback: a row written by 0.5.5 ─────────────────────────────────


def test_rollback_of_a_0_5_5_duplicate_replace_restores_the_order(home: Path) -> None:
    """``tests/fixtures/state_0_5_5_duplicate.db`` was written by langstage-hermes 0.5.5
    from PyPI running the issue's script: add, add, then the ``replace`` that stored
    "Prefers concise answers." twice. 0.5.5 rolled #3 back to [vim, concise]."""
    shutil.copyfile(FIXTURE_055, home / "state.db")
    _write(home, [CONCISE, CONCISE])  # what 0.5.5 left on disk

    res = _rollback(3)

    assert res.exit_code == 0, res.output
    assert _entries(home) == [CONCISE, VIM]
    assert f"- {CONCISE}" in res.output and f"+ {VIM}" in res.output
    # The rollback is recorded and undoing it gives back the 0.5.5 state, repeat included.
    assert _rollback(4).exit_code == 0
    assert _entries(home) == [CONCISE, CONCISE]
    assert _rollback(5).exit_code == 0
    assert _entries(home) == [CONCISE, VIM]


def test_rollback_of_older_0_5_5_rows_still_works(home: Path) -> None:
    shutil.copyfile(FIXTURE_055, home / "state.db")
    _write(home, [CONCISE, CONCISE])
    # #2 added "Uses vim.", which #3 replaced: refused, pointing at #3, as before.
    res = _rollback(2)
    assert res.exit_code == 1
    assert "has been changed since" in res.output and "audit rollback memory:user 3" in res.output
    assert _entries(home) == [CONCISE, CONCISE]
    assert _rollback(3).exit_code == 0
    assert _rollback(2).exit_code == 0
    assert _entries(home) == [CONCISE]


# ── rollback: position, not text ─────────────────────────────────────


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (["A", "B"], ["A", "A"]),  # the issue: replace index 1 with the text of entry 0
        (["B", "A"], ["A", "A"]),  # replace index 0 with the text of entry 1
        (["A", "B", "C"], ["A", "B", "A"]),
        (["A", "B", "C"], ["C", "B", "C"]),
        (["A", "A", "B"], ["A", "B"]),  # remove one of two copies
        (["A", "B", "A"], ["A", "B"]),
        (["A", "B", "A"], ["B", "A"]),
        (["A", "A"], ["A"]),
        (["A", "A", "A"], ["A", "B", "A"]),  # replace the middle copy
        (["A"], ["A", "A"]),  # a repeat appended (only a rollback row or a hand edit can)
        (["A", "B"], []),
        ([], ["A", "A"]),
    ],
)
def test_undo_with_nothing_changed_since_is_exactly_the_layer_before(before: list[str], after: list[str]) -> None:
    assert _revert(before, after, list(after)).result == before


def test_undo_takes_out_the_copy_the_mutation_wrote_not_the_first(home: Path) -> None:
    # #: replaced C (index 2) with a second A. Since then the ORIGINAL A was removed.
    plan = _revert(["A", "B", "C"], ["A", "B", "A"], ["B", "A"])
    assert plan.result == ["B", "C"]
    assert (plan.taken_out, plan.put_back) == (["A"], ["C"])


def test_undo_refuses_when_the_copy_the_mutation_wrote_is_the_one_that_went(home: Path) -> None:
    # Same mutation, but the copy it wrote (index 2) was removed since; the A that
    # is left is the original, which this mutation never touched.
    later = _row(["A", "B", "A"], ["A", "B"], id=8, action="remove")
    with pytest.raises(RollbackError, match="has been changed since") as exc:
        _revert(["A", "B", "C"], ["A", "B", "A"], ["A", "B"], later=[later])
    assert "audit rollback memory:user 8" in str(exc.value)
    assert "Nothing was changed" in str(exc.value)


def test_undo_refuses_clearly_when_copies_cannot_be_told_apart() -> None:
    # The mutation added A after X. The file was then reordered by hand and holds
    # A twice, neither where the mutation left it.
    with pytest.raises(RollbackError, match="more than once") as exc:
        _revert(["X"], ["X", "A"], ["A", "Y", "A", "X"])
    msg = str(exc.value)
    assert "Nothing was changed" in msg and "USER.md" in msg and "audit diff memory:user 7" in msg


def test_undo_of_a_removal_restores_a_repeat_the_layer_had(home: Path) -> None:
    """The layer held A twice before the removal; undoing the removal gives that back.
    The old rule ("already there, leave it") refused with "nothing to undo"."""
    _write(home, ["A", "B", "A"])
    t = _tool()
    _call(t, action="remove", target="user", index=2)  # 1
    assert _entries(home) == ["A", "B"]
    res = _rollback(1)
    assert res.exit_code == 0, res.output
    assert _entries(home) == ["A", "B", "A"]


def test_undo_does_not_add_a_repeat_the_layer_never_had(home: Path) -> None:
    t = _tool()
    _call(t, action="add", target="user", entry=CONCISE)  # 1
    _call(t, action="add", target="user", entry=VIM)  # 2
    _call(t, action="remove", target="user", index=0)  # 3
    _call(t, action="add", target="user", entry=CONCISE)  # 4: re-added, at the end
    res = _rollback(3)
    assert res.exit_code == 1 and "nothing to undo" in res.output
    assert _entries(home) == [VIM, CONCISE]


def test_undo_keeps_entries_added_since_around_a_repeat(home: Path) -> None:
    _write(home, ["A", "B", "A"])
    t = _tool()
    _call(t, action="replace", target="user", index=2, entry="C")  # 1: second A -> C
    _call(t, action="add", target="user", entry="D")  # 2
    assert _rollback(1).exit_code == 0
    assert _entries(home) == ["A", "B", "A", "D"]


def test_undo_still_finds_an_entry_moved_by_hand(home: Path) -> None:
    """With no repeats, an entry is found by its text wherever it now sits (0.5.4 behaviour)."""
    t = _tool()
    for entry in ("first", "second", "third"):
        _call(t, action="add", target="user", entry=entry)
    _call(t, action="replace", target="user", index=1, entry="2nd")  # 4
    _write(home, ["2nd", "third", "first"])  # reordered in an editor
    assert _rollback(4).exit_code == 0
    assert _entries(home) == ["second", "third", "first"]


def test_undo_refusals_from_0_5_4_are_kept(home: Path) -> None:
    t = _tool()
    _call(t, action="add", target="user", entry="x" * 1300)
    _call(t, action="replace", target="user", index=0, entry="short")  # 2
    _call(t, action="add", target="user", entry="y" * 1300)  # 3
    res = _rollback(2)
    assert res.exit_code == 1 and "over its limit" in res.output
    _write(home, ["short, edited by hand", "y" * 1300])
    res = _rollback(2)
    assert res.exit_code == 1 and "edited by hand" in res.output
    assert _entries(home) == ["short, edited by hand", "y" * 1300]


def test_plan_reports_what_it_takes_out_and_puts_back() -> None:
    plan = _revert(["A", "B"], ["A", "A"], ["A", "A"])
    assert (plan.taken_out, plan.put_back, plan.already_present) == (["A"], ["B"], [])
    assert plan.current == ["A", "A"]  # the input is not modified


# ── properties ───────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(40))
def test_property_undo_of_any_recorded_change_restores_the_exact_layer(seed: int) -> None:
    """For any before/after pair, repeats included, undoing with nothing changed
    since gives back ``before``: same entries, same order."""
    rng = random.Random(seed)
    for _ in range(250):
        alphabet = "ABCD"[: rng.randint(1, 4)]  # few letters, so repeats are common
        before = [rng.choice(alphabet) for _ in range(rng.randint(0, 7))]
        after = [rng.choice(alphabet) for _ in range(rng.randint(0, 7))]
        if before == after:
            with pytest.raises(RollbackError, match="nothing to undo"):
                _revert(before, after, list(after))
            continue
        plan = _revert(before, after, list(after))
        assert plan.result == before, (before, after)
        # And the undo of that undo is the layer the mutation left.
        assert _revert(after, before, plan.result).result == after, (before, after)


@pytest.mark.parametrize("seed", range(12))
def test_property_memory_tool_never_stores_a_repeat_and_last_change_undoes_exactly(home: Path, seed: int) -> None:
    """Random add / replace / remove calls from a small pool of texts (so the tool is
    asked for repeats all the time): the layer never holds an entry twice, and
    rolling back the last recorded change restores the previous layer exactly."""
    rng = random.Random(1000 + seed)
    pool = [f"note {c}" for c in "ABCDEF"]
    t = _tool()
    log = SkillAuditLog(home / "state.db")
    try:
        checked = 0
        for _ in range(45):
            previous = _entries(home)
            action = rng.choice(["add", "add", "replace", "replace", "remove"])
            args: dict[str, Any] = {"action": action, "target": "user"}
            if action in ("add", "replace"):
                args["entry"] = rng.choice(pool)
            if action in ("replace", "remove"):
                args["index"] = rng.randint(0, max(len(previous) - 1, 0))
            _call(t, **args)
            now = _entries(home)
            assert len(now) == len(set(now)), (args, now)
            if now == previous:
                continue  # refused or a no-op: nothing recorded
            last = log.latest_for("memory:user")
            assert memory_audit.entries_of(last.after_content) == now
            plan = memory_audit.preview_revert(log, "memory:user", last.id)
            assert plan.result == previous, (args, previous, now)
            if rng.random() < 0.3:
                # Carry it out, then undo the undo: both land exactly.
                _, undo_id = memory_audit.rollback_memory(log, "memory:user", last.id, source="cli")
                assert _entries(home) == previous
                memory_audit.rollback_memory(log, "memory:user", undo_id, source="cli")
                assert _entries(home) == now
            checked += 1
        assert checked > 10
    finally:
        log.close()


@pytest.mark.parametrize("seed", range(12))
def test_property_layers_that_already_repeat_entries_roll_back_exactly(home: Path, seed: int) -> None:
    """Start from a hand-written layer that repeats entries, then let the tool change it."""
    rng = random.Random(2000 + seed)
    pool = [f"note {c}" for c in "ABC"]
    _write(home, [rng.choice(pool) for _ in range(rng.randint(2, 6))])
    t = _tool()
    log = SkillAuditLog(home / "state.db")
    try:
        for _ in range(25):
            previous = _entries(home)
            action = rng.choice(["add", "replace", "remove"])
            args: dict[str, Any] = {"action": action, "target": "user"}
            if action in ("add", "replace"):
                args["entry"] = rng.choice([*pool, "note D", "note E"])
            if action in ("replace", "remove"):
                args["index"] = rng.randint(0, max(len(previous) - 1, 0))
            _call(t, **args)
            now = _entries(home)
            # The tool never makes a repeat worse.
            assert len(now) - len(set(now)) <= len(previous) - len(set(previous)), (args, previous, now)
            if now == previous:
                continue
            last = log.latest_for("memory:user")
            assert memory_audit.preview_revert(log, "memory:user", last.id).result == previous, (args, previous, now)
    finally:
        log.close()
