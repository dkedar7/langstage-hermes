"""``MemoryToolMiddleware`` + the single ``memory`` tool (SPEC §13.1).

Frozen-snapshot pattern — the load-bearing invariant
----------------------------------------------------

At ``before_agent`` time we read ``$HERMES_HOME/memories/MEMORY.md`` and
``USER.md`` from disk, scan every entry for prompt-injection content, and
store the result as ``state["memory_snapshot"]`` and ``state["user_snapshot"]``.

The ``memory`` tool writes mutations to disk immediately (durable across
sessions and processes) but **does not** mutate the snapshot fields. The
volatile system-prompt layer reads only from the snapshot fields, so the
system prompt stays byte-identical for the entire session — which is the
whole point: the Anthropic / OpenAI prefix caches hit every turn.

The snapshot refreshes implicitly at the next session start.

Threat-pattern scanning
-----------------------

Each entry is scanned with ``threat_patterns.scan(entry, scope="memory")``
at snapshot-build time. On a hit, the snapshot contains
``"[BLOCKED: <reason>]"`` instead of the raw entry; the live disk file
keeps the raw text so the user can ``memory(action="read", ...)`` to see
the poisoned entry and ``memory(action="remove", index=N)`` to delete it.
Silently dropping would hide the attack.

Entry delimiter is ``\\n§\\n`` (section sign), mirroring Hermes verbatim.

Tool surface
------------

```
memory(
    action: Literal["add", "replace", "remove", "read"],
    target: Literal["memory", "user"],
    entry: str = "",
    index: int | None = None,
) -> str
```

- ``add``: append ``entry``; reject if the new total exceeds the char limit.
- ``replace``: overwrite the entry at ``index`` with ``entry``.
- ``remove``: drop the entry at ``index``.
- ``read``: return the **live** disk contents (not the snapshot) so the user
  can see what's actually there, including anything the snapshot blocked.

Char limits (defaults match SPEC §2): ``memory_char_limit=2200`` for MEMORY.md,
``user_char_limit=1375`` for USER.md. Limits apply to the joined byte length
of all entries — exceeding them returns a helpful error instead of silently
truncating.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

from deepagent_hermes.memory import threat_patterns

logger = logging.getLogger(__name__)

# Entry separator — matches Hermes exactly. Tests pin this value.
ENTRY_DELIMITER = "\n§\n"

# Default char limits — SPEC §2 / Hermes defaults. Overridable per-instance.
DEFAULT_MEMORY_CHAR_LIMIT = 2200
DEFAULT_USER_CHAR_LIMIT = 1375

Target = Literal["memory", "user"]
Action = Literal["add", "replace", "remove", "read"]


# ── State schema ─────────────────────────────────────────────────────


class MemoryStateExt(AgentState):
    """State fields owned by ``MemoryToolMiddleware``.

    These fields are also declared on the global ``HermesState`` (in
    ``deepagent_hermes/state.py``, owned by another agent). Declaring them
    here makes the middleware testable in isolation; ``langchain.agents``
    merges the schemas so duplicate-declared fields collapse to a single
    column at compile time.

    - ``memory_snapshot`` / ``user_snapshot``: frozen at ``before_agent``;
      consumed by the volatile-layer prompt builder. Never mutated by the
      ``memory`` tool — that's the invariant the prefix cache depends on.
    - ``turns_since_memory``: reset to 0 on every ``memory`` tool call so
      the reflection middleware can decide when to nudge.
    """

    memory_snapshot: NotRequired[Annotated[str, lambda _a, b: b]]
    user_snapshot: NotRequired[Annotated[str, lambda _a, b: b]]
    turns_since_memory: NotRequired[Annotated[int, lambda _a, b: b]]


# ── Disk helpers ─────────────────────────────────────────────────────


def _hermes_home() -> Path:
    """Resolve ``HERMES_HOME`` from env. Tests use the ``tmp_hermes_home`` fixture.

    Resolution order (matches ``config.hermes_home``): ``DEEPAGENT_HERMES_HOME``
    → ``HERMES_HOME`` → ``~/.deepagent-hermes``.
    """
    return Path(
        os.environ.get("DEEPAGENT_HERMES_HOME")
        or os.environ.get("HERMES_HOME")
        or (Path.home() / ".deepagent-hermes")
    )


def _memory_dir() -> Path:
    """Profile-scoped memory directory. Resolved per-call so HERMES_HOME swaps stick.

    Caching this at import time would break the ``tmp_hermes_home`` fixture
    (and real-world profile switches), so we re-resolve every call. Cheap.
    """
    return _hermes_home() / "memories"


def _file_for(target: Target) -> Path:
    return _memory_dir() / ("USER.md" if target == "user" else "MEMORY.md")


def _read_entries(path: Path) -> list[str]:
    """Read a memory file and split on the section delimiter.

    Returns ``[]`` on missing/empty file or any I/O error — callers should
    not see exceptions from a malformed disk file.
    """
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to read %s: %s", path, e)
        return []
    if not raw.strip():
        return []
    entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
    return [e for e in entries if e]


def _write_entries_atomic(path: Path, entries: list[str]) -> None:
    """Persist ``entries`` to ``path`` via temp-file + rename.

    Atomic rename avoids the read/empty-window race that ``open("w")`` would
    create — concurrent readers see either the old complete file or the new
    one, never an empty truncated file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ENTRY_DELIMITER.join(entries)
    fd, tmp_path_str = tempfile.mkstemp(
        dir=str(path.parent), prefix=".mem_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path_str, path)
    except BaseException:
        # Don't leave temp files lying around on any failure.
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


def _char_count(entries: list[str]) -> int:
    """Joined-byte length, accounting for the delimiters between entries."""
    if not entries:
        return 0
    return len(ENTRY_DELIMITER.join(entries))


# ── Snapshot rendering ───────────────────────────────────────────────


def _render_block(target: Target, entries: list[str], char_limit: int) -> str:
    """Format a memory block for system-prompt injection. Empty entries → ``""``."""
    if not entries:
        return ""

    content = ENTRY_DELIMITER.join(entries)
    current = len(content)
    pct = min(100, int((current / char_limit) * 100)) if char_limit > 0 else 0

    if target == "user":
        header = (
            f"USER PROFILE (who the user is) [{pct}% — {current:,}/{char_limit:,} chars]"
        )
    else:
        header = (
            f"MEMORY (your personal notes) [{pct}% — {current:,}/{char_limit:,} chars]"
        )
    sep = "═" * 46
    return f"{sep}\n{header}\n{sep}\n{content}"


def _sanitize_for_snapshot(entries: list[str], filename: str) -> list[str]:
    """Replace threat-matching entries with ``[BLOCKED: ...]`` placeholders.

    Run only against the snapshot — live disk state keeps raw text so the
    user can inspect and remove poisoned entries. See module docstring.
    """
    sanitized: list[str] = []
    for entry in entries:
        if not entry or entry.startswith("[BLOCKED:"):
            sanitized.append(entry)
            continue
        reason = threat_patterns.scan(entry, scope="memory")
        if reason:
            logger.warning("Memory entry from %s blocked at load: %s", filename, reason)
            sanitized.append(
                f"[BLOCKED: {filename} entry {reason}. "
                f"Use memory(action='read', target=...) to inspect and "
                f"memory(action='remove', ...) to delete.]"
            )
        else:
            sanitized.append(entry)
    return sanitized


def build_snapshot(
    *, memory_char_limit: int = DEFAULT_MEMORY_CHAR_LIMIT,
    user_char_limit: int = DEFAULT_USER_CHAR_LIMIT,
) -> dict[str, str]:
    """Return ``{"memory_snapshot": ..., "user_snapshot": ...}`` from disk.

    Pure function — useful for tests and for the prompt-assembly middleware to
    rebuild the snapshot on demand (e.g. ``/reload`` slash command).
    """
    _memory_dir().mkdir(parents=True, exist_ok=True)
    mem_entries = _read_entries(_file_for("memory"))
    usr_entries = _read_entries(_file_for("user"))

    # Dedupe before sanitizing — keeps placeholders deterministic in tests.
    mem_entries = list(dict.fromkeys(mem_entries))
    usr_entries = list(dict.fromkeys(usr_entries))

    sanitized_mem = _sanitize_for_snapshot(mem_entries, "MEMORY.md")
    sanitized_usr = _sanitize_for_snapshot(usr_entries, "USER.md")

    return {
        "memory_snapshot": _render_block("memory", sanitized_mem, memory_char_limit),
        "user_snapshot": _render_block("user", sanitized_usr, user_char_limit),
    }


# ── Tool factory ─────────────────────────────────────────────────────


def _make_memory_tool(
    *, memory_char_limit: int, user_char_limit: int
):
    """Build the ``memory`` tool bound to the configured char limits.

    The tool is built as a closure (not a module-level function) so the
    char limits travel with the middleware instance — letting two
    differently-configured middlewares coexist in tests.
    """

    def _limit_for(target: Target) -> int:
        return user_char_limit if target == "user" else memory_char_limit

    def _format_response(
        target: Target, entries: list[str], message: str
    ) -> str:
        current = _char_count(entries)
        limit = _limit_for(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        return json.dumps(
            {
                "success": True,
                "target": target,
                "message": message,
                "entries": entries,
                "entry_count": len(entries),
                "usage": f"{pct}% — {current:,}/{limit:,} chars",
            },
            ensure_ascii=False,
        )

    def _format_error(message: str, **extra: Any) -> str:
        return json.dumps({"success": False, "error": message, **extra}, ensure_ascii=False)

    @tool(
        "memory",
        description=(
            "Save durable information to persistent memory that survives across sessions. "
            "Memory is injected into future turns, so keep it compact and focused on facts "
            "that will still matter later.\n\n"
            "Actions:\n"
            "  add     — append a new entry (rejects if char-limit would overflow).\n"
            "  replace — overwrite the entry at `index` with `entry`.\n"
            "  remove  — drop the entry at `index`.\n"
            "  read    — return current on-disk entries (NOT the system-prompt snapshot, "
            "             which is frozen for the session).\n\n"
            "Targets:\n"
            "  memory — your own notes (environment facts, conventions, tool quirks).\n"
            "  user   — who the user is (preferences, role, communication style).\n\n"
            "IMPORTANT: writes update disk immediately and are visible to future sessions, "
            "but the CURRENT session's system prompt is FROZEN — that's intentional, it "
            "preserves the prefix cache. Don't write the same fact twice expecting the "
            "second turn to see it."
        ),
    )
    def memory(
        action: Action,
        target: Target,
        entry: str = "",
        index: int | None = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Single entry-point for memory mutations + reads."""
        # Validate target up front — anything else is a programming bug,
        # not a recoverable user error.
        if target not in ("memory", "user"):
            payload = _format_error(
                f"Invalid target {target!r}. Use 'memory' or 'user'."
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(content=payload, tool_call_id=tool_call_id)
                    ],
                }
            )

        path = _file_for(target)
        limit = _limit_for(target)
        entries = _read_entries(path)  # always read fresh — picks up sister-session writes

        # ── action dispatch ──
        if action == "read":
            payload = _format_response(target, entries, "Current entries on disk.")

        elif action == "add":
            clean = (entry or "").strip()
            if not clean:
                payload = _format_error("entry is required for action='add'.")
            else:
                # Threat-scan on write too: if it's poisoned, the user will see
                # the rejection immediately rather than discovering it next
                # session when the snapshot replaces it with [BLOCKED: ...].
                reason = threat_patterns.scan(clean, scope="memory")
                if reason:
                    payload = _format_error(
                        f"Rejected: entry {reason}. Memory enters the system "
                        f"prompt and must not contain injection payloads."
                    )
                elif clean in entries:
                    payload = _format_response(
                        target, entries, "Entry already exists (no duplicate added)."
                    )
                else:
                    new_entries = entries + [clean]
                    new_total = _char_count(new_entries)
                    if new_total > limit:
                        payload = _format_error(
                            f"Memory at {_char_count(entries):,}/{limit:,} chars. "
                            f"Adding this entry ({len(clean)} chars) would exceed the "
                            f"limit. Replace or remove existing entries first.",
                            current_entries=entries,
                            usage=f"{_char_count(entries):,}/{limit:,}",
                        )
                    else:
                        _write_entries_atomic(path, new_entries)
                        payload = _format_response(
                            target, new_entries, "Entry added."
                        )

        elif action == "replace":
            clean = (entry or "").strip()
            if index is None:
                payload = _format_error("index is required for action='replace'.")
            elif not clean:
                payload = _format_error(
                    "entry is required for action='replace'. Use 'remove' to delete."
                )
            elif index < 0 or index >= len(entries):
                payload = _format_error(
                    f"index {index} out of range (0..{len(entries) - 1}). "
                    f"Use action='read' first to see current entries.",
                    entries=entries,
                )
            else:
                reason = threat_patterns.scan(clean, scope="memory")
                if reason:
                    payload = _format_error(
                        f"Rejected: replacement entry {reason}."
                    )
                else:
                    test_entries = list(entries)
                    test_entries[index] = clean
                    new_total = _char_count(test_entries)
                    if new_total > limit:
                        payload = _format_error(
                            f"Replacement would put memory at {new_total:,}/{limit:,} "
                            f"chars. Shorten the new content or remove other entries first."
                        )
                    else:
                        _write_entries_atomic(path, test_entries)
                        payload = _format_response(
                            target, test_entries, f"Entry at index {index} replaced."
                        )

        elif action == "remove":
            if index is None:
                payload = _format_error("index is required for action='remove'.")
            elif index < 0 or index >= len(entries):
                payload = _format_error(
                    f"index {index} out of range (0..{len(entries) - 1}). "
                    f"Use action='read' first to see current entries.",
                    entries=entries,
                )
            else:
                new_entries = list(entries)
                removed = new_entries.pop(index)
                _write_entries_atomic(path, new_entries)
                payload = _format_response(
                    target,
                    new_entries,
                    f"Removed entry at index {index}: {removed[:60]}…"
                    if len(removed) > 60
                    else f"Removed entry at index {index}: {removed}",
                )

        else:
            payload = _format_error(
                f"Unknown action {action!r}. Use: add, replace, remove, read."
            )

        # On any successful write OR read, reset the nudge counter so the
        # reflection middleware doesn't immediately re-prompt. We can be a
        # bit liberal here — a no-op read is still "the agent thought about
        # memory this turn".
        return Command(
            update={
                "messages": [
                    ToolMessage(content=payload, tool_call_id=tool_call_id)
                ],
                "turns_since_memory": 0,
            }
        )

    return memory


# ── Middleware ───────────────────────────────────────────────────────


class MemoryToolMiddleware(AgentMiddleware):
    """Loads the memory snapshot at session start; exposes the ``memory`` tool.

    Mid-session ``memory`` tool calls hit disk immediately but do not mutate
    ``state["memory_snapshot"]`` / ``state["user_snapshot"]`` — that's the
    invariant the prefix-cache discipline (SPEC §6) relies on. The snapshot
    refreshes on the next session start.
    """

    state_schema = MemoryStateExt

    def __init__(
        self,
        *,
        memory_char_limit: int = DEFAULT_MEMORY_CHAR_LIMIT,
        user_char_limit: int = DEFAULT_USER_CHAR_LIMIT,
    ) -> None:
        """Initialize the middleware.

        Args:
            memory_char_limit: Joined-byte limit for MEMORY.md (default 2200,
                per SPEC §2 and Hermes verbatim).
            user_char_limit: Joined-byte limit for USER.md (default 1375).
        """
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Tools attribute is read by `create_agent` at compile time.
        self.tools = [
            _make_memory_tool(
                memory_char_limit=memory_char_limit,
                user_char_limit=user_char_limit,
            )
        ]

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Build the frozen snapshot from disk and seat it in state.

        Skips re-build if the snapshot is already populated — supports the
        ``deepagents``/langgraph thread-resume case where ``before_agent`` runs
        again on the same compiled graph.
        """
        # Idempotent — if a snapshot is already in state (resumed thread),
        # don't reload. The snapshot is frozen by definition.
        if "memory_snapshot" in state and "user_snapshot" in state:
            return None

        snapshot = build_snapshot(
            memory_char_limit=self.memory_char_limit,
            user_char_limit=self.user_char_limit,
        )
        # Seed turns_since_memory if absent — first turn after fresh start.
        return {
            **snapshot,
            "turns_since_memory": state.get("turns_since_memory", 0),
        }


__all__ = [
    "DEFAULT_MEMORY_CHAR_LIMIT",
    "DEFAULT_USER_CHAR_LIMIT",
    "ENTRY_DELIMITER",
    "MemoryStateExt",
    "MemoryToolMiddleware",
    "build_snapshot",
]
