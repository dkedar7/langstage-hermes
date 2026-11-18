"""Memory mutation log + undo (gh #222).

``USER.md`` and ``MEMORY.md`` go into every later session's system prompt, and
the agent rewrites them on its own: the ``memory`` tool's ``add`` / ``replace``
/ ``remove``, called by the agent or by the reflection review fork. Before this
module a ``replace`` or ``remove`` overwrote the file in place and the earlier
text was gone, with nothing to say when it changed or which conversation
changed it.

Every memory write now appends a row to the same audit table skills use
(``skill_mutations`` in ``<HERMES_HOME>/state.db``), under the name
``memory:user`` or ``memory:memory``. A skill name can't contain a colon, so the
two never collide, and an existing ``state.db`` needs no migration. The row
holds the whole layer before and after the write, the source (``agent`` /
``review`` / ``cli``), the run's ``session_id`` and the tool call id.

Undo is per entry, not per file. A skill is one document, so rolling it back
restores the file. A memory layer is a list of independent entries, and
restoring the whole file to an old state would silently drop every entry added
since. :func:`rollback_memory` therefore reverts only what the chosen mutation
changed: it puts back the entry that mutation removed or replaced and takes out
the entry it wrote, leaving every other entry alone. When nothing has changed
since the mutation, that is exactly "the layer as it was before the mutation",
order included. The undo works on positions, not texts (gh #226): a layer can
hold the same text twice (0.5.4 and 0.5.5 let ``replace`` write a repeat, and a
hand edit still can), and "the first entry that says this" is then the wrong
one to take out.
When the entry the mutation wrote is no longer on disk (a later mutation or a
hand edit changed it), the rollback refuses and changes nothing rather than
guess. It also refuses a restore that would put the layer over its char limit.
The rollback is itself logged (``action="rollback"``), so it can be undone the
same way.

Recording is best-effort: the memory file is already written by the time the
row is appended, and a locked or unwritable ``state.db`` must never fail the
write. :func:`record_memory_mutation` returns a note instead of raising.

Nothing here touches the frozen snapshot in graph state.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from langstage_hermes.skills.audit import MutationRow, SkillAuditLog

logger = logging.getLogger(__name__)

__all__ = [
    "MEMORY_PREFIX",
    "RevertPlan",
    "audit_name",
    "describe_change",
    "disk_drift",
    "entries_of",
    "is_memory_name",
    "memory_mutation_context",
    "plan_revert",
    "record_memory_mutation",
    "rollback_memory",
    "target_of",
]

Target = Literal["memory", "user"]

MEMORY_PREFIX = "memory:"

# (source, session_id) for writes made inside a block (the reflection review).
# A ContextVar for the same reason the skill library uses one (gh #218): one
# compiled graph serves many conversations, and concurrent runs must not see
# each other's attribution.
_CONTEXT: ContextVar[tuple[str | None, str | None]] = ContextVar("hermes_memory_mutation_context", default=(None, None))


@contextmanager
def memory_mutation_context(*, source: str | None = None, session_id: str | None = None) -> Iterator[None]:
    """Attribute memory writes inside the block to ``source`` / ``session_id``.

    A value left ``None`` keeps whatever an enclosing block set, so the review
    fork's own tool call doesn't undo its parent's session.
    """
    cur_source, cur_session = _CONTEXT.get()
    token = _CONTEXT.set((source or cur_source, session_id or cur_session))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current_context() -> tuple[str | None, str | None]:
    """The ``(source, session_id)`` an enclosing :func:`memory_mutation_context` set."""
    return _CONTEXT.get()


# ── names ────────────────────────────────────────────────────────────


def audit_name(target: str) -> str:
    """The audit-log name for a memory layer: ``memory:user`` / ``memory:memory``."""
    return f"{MEMORY_PREFIX}{target}"


def is_memory_name(name: str | None) -> bool:
    return bool(name) and str(name).startswith(MEMORY_PREFIX)


def target_of(name: str | None) -> Target | None:
    """``"user"`` / ``"memory"`` for a memory audit name, else ``None``."""
    if not is_memory_name(name):
        return None
    target = str(name)[len(MEMORY_PREFIX) :]
    return target if target in ("user", "memory") else None  # type: ignore[return-value]


def filename_for(target: str) -> str:
    return "USER.md" if target == "user" else "MEMORY.md"


# ── content helpers ──────────────────────────────────────────────────


def entries_of(content: bytes | None) -> list[str]:
    """Split a recorded layer (or the file's bytes) into its entries.

    Line endings are normalised first: on Windows the memory file is written in
    text mode, so its bytes carry ``\\r\\n`` around the delimiter.
    """
    from langstage_hermes.memory.tool import ENTRY_DELIMITER

    if not content:
        return []
    text = content.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return [e for e in (part.strip() for part in text.split(ENTRY_DELIMITER)) if e]


def read_bytes(path: Path) -> bytes | None:
    """The file's bytes, or ``None`` when it doesn't exist or can't be read."""
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def _state_db() -> Path:
    from langstage_hermes.config import hermes_home

    return hermes_home() / "state.db"


def _open_log() -> SkillAuditLog:
    from langstage_hermes.skills.audit import SkillAuditLog

    db_path = _state_db()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SkillAuditLog(db_path=str(db_path))


# ── recording ────────────────────────────────────────────────────────


def record_memory_mutation(
    *,
    target: str,
    action: str,
    path: Path,
    before: bytes | None,
    after: bytes | None,
    source: str | None = None,
    session_id: str | None = None,
    tool_call_id: str | None = None,
) -> tuple[int | None, str | None]:
    """Append one memory mutation to ``<HERMES_HOME>/state.db``.

    Returns ``(mutation_id, None)``, or ``(None, note)`` when the row could not
    be written. Never raises: the memory file is already on disk, and failing
    the caller over a missing audit row would be worse than the missing row.

    The home is resolved per call (like the memory file itself), and the
    connection is opened and closed here, so a swapped ``HERMES_HOME`` is
    honoured and no handle outlives the write.
    """
    ctx_source, ctx_session = _CONTEXT.get()
    try:
        log = _open_log()
        try:
            mutation_id = log.record(
                skill_name=audit_name(target),
                action=action,
                before_content=before,
                after_content=after,
                source=ctx_source or source,
                session_id=session_id or ctx_session,
                tool_call_id=tool_call_id or None,
                skill_path=path,
            )
        finally:
            log.close()
    except Exception as exc:
        logger.warning("Failed to record memory %s on %s in the audit log", action, target, exc_info=True)
        return None, f"saved, but not recorded in the audit log ({type(exc).__name__}: {exc})"
    return mutation_id, None


# ── describing a change ──────────────────────────────────────────────


def _opcodes(before: list[str], after: list[str]) -> list[tuple[str, int, int, int, int]]:
    return [op for op in SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes() if op[0] != "equal"]


def describe_change(row: MutationRow) -> tuple[list[str], list[str]]:
    """``(entries the mutation took out, entries it wrote)`` for one audit row."""
    before = entries_of(row.before_content)
    after = entries_of(row.after_content)
    removed: list[str] = []
    added: list[str] = []
    for _tag, i1, i2, j1, j2 in _opcodes(before, after):
        removed.extend(before[i1:i2])
        added.extend(after[j1:j2])
    return removed, added


# ── undo ─────────────────────────────────────────────────────────────


@dataclass
class RevertPlan:
    """What undoing one memory mutation would do to the layer on disk."""

    target: Target
    mutation_id: int
    path: Path
    current: list[str]
    result: list[str]
    #: Entries the rollback puts back (the mutation had removed or replaced them).
    put_back: list[str] = field(default_factory=list)
    #: Entries the rollback takes out (the mutation had written them).
    taken_out: list[str] = field(default_factory=list)
    #: Entries the mutation removed that are already back on disk; left as they are.
    already_present: list[str] = field(default_factory=list)


def _short(entry: str, width: int = 70) -> str:
    flat = " ".join(entry.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _char_count(entries: list[str]) -> int:
    from langstage_hermes.memory.tool import ENTRY_DELIMITER

    return len(ENTRY_DELIMITER.join(entries)) if entries else 0


class _Locator:
    """Where each entry of the recorded ``after`` layer sits in the layer on disk.

    A memory layer may hold the same text more than once (0.5.4 and 0.5.5 let
    ``replace`` write a repeat, and a hand edit still can), so "the entry the
    mutation wrote" has to be a position, not a text (gh #226). The recorded
    ``after`` is lined up with the current entries; an entry that kept its place
    relative to its neighbours is found that way. One that did not (the file was
    reordered by hand) is found by its text, but only among the current entries
    nothing else lines up with, and only when exactly one of them says it.
    """

    def __init__(self, after: list[str], current: list[str]) -> None:
        self.after = after
        self.aligned: dict[int, int] = {}
        for a, b, size in SequenceMatcher(a=after, b=current, autojunk=False).get_matching_blocks():
            for k in range(size):
                self.aligned[a + k] = b + k
        lined_up = set(self.aligned.values())
        # text -> current positions that no recorded entry lines up with.
        self.loose: dict[str, list[int]] = {}
        for pos, entry in enumerate(current):
            if pos not in lined_up:
                self.loose.setdefault(entry, []).append(pos)

    def find(self, j: int, *, claim: bool = False) -> int | None:
        """The current position of ``after[j]``; ``None`` if gone, ``-1`` if it can't be told apart."""
        if j in self.aligned:
            return self.aligned[j]
        spots = self.loose.get(self.after[j], [])
        if not spots:
            return None
        if len(spots) > 1:
            return -1
        return spots.pop() if claim else spots[0]


def plan_revert(
    row: MutationRow,
    current: list[str],
    *,
    limit: int,
    later: list[MutationRow] | None = None,
) -> RevertPlan:
    """Work out the per-entry undo of *row* against the layer's *current* entries.

    The undo is by position (gh #226): the recorded before/after give the edit
    the mutation made, and each step is reversed where that entry sits now. With
    nothing changed since, the result is the recorded ``before``, entry for entry
    and in the same order, whether or not the layer repeats an entry.

    Raises:
        RollbackError: the entry the mutation wrote is no longer on disk (or
            can't be told from a copy of it), there is nothing left to undo, or
            the result would exceed *limit*. Nothing is written in any of those
            cases.
    """
    from langstage_hermes.skills.audit import RollbackError

    target = target_of(row.skill_name)
    if target is None:
        raise RollbackError(f"mutation #{row.id} is not a memory mutation")
    fname = filename_for(target)
    before = entries_of(row.before_content)
    after = entries_of(row.after_content)
    path = Path(row.skill_path) if row.skill_path else Path(fname)
    current = list(current)
    plan = RevertPlan(target=target, mutation_id=row.id, path=path, current=current, result=list(current))

    script = SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes()
    changes = [op for op in script if op[0] != "equal"]
    # before index -> after index, for the entries the mutation left alone.
    kept = {i1 + k: j1 + k for tag, i1, i2, j1, _j2 in script if tag == "equal" for k in range(i2 - i1)}
    where = _Locator(after, current)

    # 1. Find every entry the mutation wrote. All of them must still be there.
    taken: set[int] = set()  # current positions the rollback takes out
    written: dict[int, int] = {}  # after index -> current position
    for _tag, _i1, _i2, j1, j2 in changes:
        for j in range(j1, j2):
            pos = where.find(j, claim=True)
            if pos is None:
                # The newest later mutation that took this entry out is the one to undo first.
                culprit = next((r for r in later or [] if after[j] in describe_change(r)[0]), None)
                if culprit is not None:
                    hint = (
                        f" Mutation #{culprit.id} ({culprit.action}) changed it; undo that one first: "
                        f"`langstage-hermes audit rollback {row.skill_name} {culprit.id}`."
                    )
                else:
                    hint = f" No recorded change accounts for it, so {fname} was probably edited by hand."
                raise RollbackError(
                    f"mutation #{row.id} wrote the entry “{_short(after[j])}”, which is no longer in {fname}: "
                    f"it has been changed since. Nothing was changed.{hint} "
                    f"`langstage-hermes audit show {row.id}` prints the earlier text."
                )
            if pos < 0:
                raise RollbackError(
                    f"mutation #{row.id} wrote the entry “{_short(after[j])}”, and {fname} now holds that entry more "
                    f"than once, in a different order than the mutation left it, so it is not clear which copy to "
                    f"take out. Nothing was changed. Remove the extra copy from {fname} first, then retry. "
                    f"`langstage-hermes audit diff {row.skill_name} {row.id}` shows what changed since."
                )
            taken.add(pos)
            written[j] = pos
            plan.taken_out.append(after[j])

    # 2. Put back what the mutation took out, where it was. ``inserts[p]`` goes
    #    in just before current position ``p`` (``len(current)`` is the end).
    have = Counter(current)
    for pos in taken:
        have[current[pos]] -= 1
    had = Counter(before)
    inserts: dict[int, list[str]] = {}
    for _tag, i1, i2, j1, j2 in changes:
        if j2 > j1:
            # A replacement: the earlier text goes where the new text is.
            anchor = written[j1]
        else:
            # A pure removal: after its nearest surviving earlier neighbour, else
            # before its nearest later one, else at the end.
            anchor = len(current)
            for i in range(i1 - 1, -1, -1):
                pos = where.find(kept[i]) if i in kept else None
                if pos is not None and pos >= 0:
                    anchor = pos + 1
                    break
            else:
                for i in range(i2, len(before)):
                    pos = where.find(kept[i]) if i in kept else None
                    if pos is not None and pos >= 0:
                        anchor = pos
                        break
        for entry in before[i1:i2]:
            # Already back (a later change re-added it): the layer holds as many
            # copies as it did before the mutation. Putting another in would
            # leave it with a repeat it never had.
            if have[entry] >= had[entry]:
                plan.already_present.append(entry)
                continue
            inserts.setdefault(anchor, []).append(entry)
            have[entry] += 1
            plan.put_back.append(entry)

    result: list[str] = []
    for pos, entry in enumerate(current):
        result.extend(inserts.get(pos, ()))
        if pos not in taken:
            result.append(entry)
    result.extend(inserts.get(len(current), ()))
    plan.result = result

    if not plan.put_back and not plan.taken_out:
        raise RollbackError(
            f"nothing to undo: {fname} already holds what it held before mutation #{row.id}. Nothing was changed."
        )
    new_total = _char_count(result)
    if new_total > limit and new_total > _char_count(plan.current):
        raise RollbackError(
            f"undoing mutation #{row.id} would put {fname} at {new_total:,}/{limit:,} chars, over its limit. "
            f"Nothing was changed. Remove or shorten an entry first, then retry."
        )
    return plan


def char_limit_for(target: str) -> int:
    """The configured char limit for a layer, falling back to the documented default."""
    from langstage_hermes.memory.tool import DEFAULT_MEMORY_CHAR_LIMIT, DEFAULT_USER_CHAR_LIMIT

    try:
        from langstage_hermes.config import HermesConfig

        cfg = HermesConfig.resolve()
        return int(cfg.memory_user_char_limit if target == "user" else cfg.memory_char_limit)
    except Exception:
        # A malformed config must not turn an offline rollback into a traceback.
        return DEFAULT_USER_CHAR_LIMIT if target == "user" else DEFAULT_MEMORY_CHAR_LIMIT


def _later_rows(log: SkillAuditLog, name: str, mutation_id: int) -> list[MutationRow]:
    """Mutations of *name* recorded after *mutation_id*, newest first."""
    rows = [r for r in log.list(skill_name=name, limit=1000) if r.id > mutation_id]
    return sorted(rows, key=lambda r: r.id, reverse=True)


def preview_revert(log: SkillAuditLog, name: str, mutation_id: int, *, limit: int | None = None) -> RevertPlan:
    """The plan :func:`rollback_memory` would carry out, without writing anything."""
    from langstage_hermes.memory.tool import _file_for, _read_entries
    from langstage_hermes.skills.audit import RollbackError

    row = log.get(mutation_id)
    if row is None:
        raise RollbackError(f"mutation #{mutation_id} not found")
    if row.skill_name != name:
        raise RollbackError(f"mutation #{mutation_id} is for {row.skill_name!r}, not {name!r}")
    target = target_of(name)
    if target is None:
        raise RollbackError(f"{name!r} is not a memory layer; use memory:user or memory:memory")
    # Always the layer of the current home, never a path recorded under another one.
    path = _file_for(target)
    plan = plan_revert(
        row,
        _read_entries(path),
        limit=limit if limit is not None else char_limit_for(target),
        later=_later_rows(log, name, mutation_id),
    )
    plan.path = path
    return plan


def rollback_memory(
    log: SkillAuditLog,
    name: str,
    mutation_id: int,
    *,
    source: str = "rollback",
    session_id: str | None = None,
    limit: int | None = None,
) -> tuple[RevertPlan, int | None]:
    """Undo memory mutation *mutation_id* on layer *name* (``memory:user`` / ``memory:memory``).

    Per-entry (see the module docstring): the entry the mutation removed or
    replaced comes back, the entry it wrote goes, and every other entry stays.
    Returns the plan that was applied and the id of the ``rollback`` row.
    """
    from langstage_hermes.memory.tool import _write_entries_atomic

    plan = preview_revert(log, name, mutation_id, limit=limit)
    before_bytes = read_bytes(plan.path)
    _write_entries_atomic(plan.path, plan.result)
    new_id: int | None = None
    try:
        new_id = log.record(
            skill_name=name,
            action="rollback",
            before_content=before_bytes,
            after_content=read_bytes(plan.path),
            source=source,
            session_id=session_id,
            tool_call_id=None,
            skill_path=plan.path,
        )
    except Exception:
        logger.warning("Failed to record the rollback of memory mutation #%s", mutation_id, exc_info=True)
    return plan, new_id


# ── hand edits ───────────────────────────────────────────────────────


def disk_drift(log: SkillAuditLog, target: str) -> Any | None:
    """The latest row for *target* if the layer on disk no longer matches it, else ``None``.

    An edit made in an editor bypasses the log. Comparing the entries on disk
    with the ``after`` of the newest recorded change shows that one happened.
    A layer with no recorded change yet is not reported.
    """
    from langstage_hermes.memory.tool import _file_for, _read_entries

    latest = log.latest_for(audit_name(target))
    if latest is None:
        return None
    if _read_entries(_file_for(target)) == entries_of(latest.after_content):  # type: ignore[arg-type]
        return None
    return latest


def diff_against_disk(log: SkillAuditLog, name: str, mutation_id: int) -> str:
    """Unified diff of the layer as *mutation_id* left it vs. the layer on disk ("" if same)."""
    from langstage_hermes.memory.tool import ENTRY_DELIMITER, _file_for, _read_entries
    from langstage_hermes.skills.audit import RollbackError, render_unified_diff

    row = log.get(mutation_id)
    if row is None or row.skill_name != name:
        raise RollbackError(f"mutation {mutation_id} not found for {name!r}")
    target = target_of(name)
    if target is None:
        raise RollbackError(f"{name!r} is not a memory layer; use memory:user or memory:memory")
    recorded = entries_of(row.after_content)
    current = _read_entries(_file_for(target))
    if recorded == current:
        return ""
    return render_unified_diff(
        (ENTRY_DELIMITER.join(recorded) + "\n").encode("utf-8") if recorded else b"",
        (ENTRY_DELIMITER.join(current) + "\n").encode("utf-8") if current else b"",
        fromfile=f"{name}@mutation-{mutation_id}",
        tofile=f"{name}@disk",
    )
