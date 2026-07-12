"""Keyless / offline reflection→skill-creation demo (gh #69).

The headline value of this runtime is the **closed reflection→skill-creation
loop**: the agent works for a while, a review subagent reflects on what it just
did, and crystallises a reusable ``SKILL.md`` (and durable ``MEMORY.md`` notes)
into the skill library. Every *documented* way to watch that loop close needs a
paid API key and a live multi-turn session, so a brand-new adopter can't confirm
the one thing the README sells before committing a key and spending tokens.

This module closes that gap. It drives the **real** shipped machinery —
``create_hermes_agent`` with the genuine ``ReflectionMiddleware``,
``SubAgentMiddleware`` review dispatch, ``SkillLibrary.write()``, the real
``skill_manage`` / ``memory`` tools, the audit log and the FTS5 store — against
two *scripted* fake chat models instead of a live provider. No network, no API
key, fully deterministic. The side effects are real: a genuine ``SKILL.md`` and
``USER.md`` land under the demo's throwaway ``HERMES_HOME``.

The only thing faked is the model: the ``model=`` / ``aux_model=`` kwargs that
``create_hermes_agent`` already accepts (the bring-your-own-model path) let us
thread in a scripted ``BaseChatModel`` whose turns:

1. call a real tool a few times (crossing the ``skills_creation_nudge_interval``
   counter that ``ReflectionMiddleware`` tracks), then
2. spawn the ``review`` subagent via the genuine ``task`` tool — exactly how the
   live runtime closes the loop.

The review subagent (driven by the scripted *aux* model) then calls the real
``skill_manage`` + ``memory`` tools, writing the real files.

``run_demo`` is import-and-call friendly so both the CLI ``demo`` command and the
test suite exercise the identical path a real user takes.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# ── the canned skill + memory the review subagent "discovers" ────────────────
#
# A realistic, reusable *procedure* — the exact shape SKILL.md is meant for
# (mirrors examples/dogfood_procedural.py, which targets the same write path
# against a live model).

DEMO_SKILL_NAME = "profile-slow-python"
DEMO_SKILL_CATEGORY = "engineering"
DEMO_SKILL_DESCRIPTION = (
    "Investigate why a Python script is slow: read it, form a hotspot hypothesis, "
    "confirm with cProfile, then propose one concrete fix."
)
DEMO_SKILL_BODY = """\
# Profiling a slow Python script

A repeatable procedure for turning "this script feels slow" into one concrete fix.

## Steps

1. **Read the script** end to end; note loops, I/O, and anything quadratic.
2. **Form a hypothesis** about the hotspot from first principles before measuring.
3. **Confirm with `cProfile`**: `python -m cProfile -s cumtime script.py`.
4. **Analyse** the top cumulative-time frames — is the hotspot where you guessed?
5. **Propose ONE fix** (algorithmic first, micro-optimisation last) and re-profile.

## Notes

- Measure before optimising; a confirmed hotspot beats a guessed one.
- Prefer an O(n) rewrite over shaving constants off an O(n^2) loop.
"""

DEMO_MEMORY_TARGET = "user"
DEMO_MEMORY_ENTRY = "Prefers a profiling-driven investigation procedure over guesswork when debugging performance."

_LS_TOOL = "ls"
_TASK_TOOL = "task"
_REVIEW_SUBAGENT = "review"


# ── scripted fake models ─────────────────────────────────────────────────────


class _ScriptedModel(BaseChatModel):
    """Base for the two scripted demo models.

    ``bind_tools`` is a no-op returning ``self`` — the scripted turns already
    target real tool *names*, so we don't need the bound-tool schema. Every
    other langchain call path (``invoke`` / ``with_config`` / streaming via the
    default ``_generate`` bridge) works unchanged.
    """

    @property
    def _llm_type(self) -> str:  # pragma: no cover - identity only
        return "langstage-hermes-demo-scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    @staticmethod
    def _result(message: AIMessage) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=message)])


class DemoMainModel(_ScriptedModel):
    """The primary agent's scripted brain.

    Emits ``nudge_interval`` tool-using turns (a harmless ``ls``) so the genuine
    ``ReflectionMiddleware`` counter crosses ``skills_creation_nudge_interval``,
    then spawns the ``review`` subagent via the real ``task`` tool, then returns
    a final text answer.
    """

    nudge_interval: int = 3

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        n_ai = sum(1 for m in messages if isinstance(m, AIMessage))
        if n_ai < self.nudge_interval:
            # A real, side-effect-free tool call — bumps iters_since_skill.
            return self._result(AIMessage(content="", tool_calls=[{"name": _LS_TOOL, "id": f"ls-{n_ai}", "args": {}}]))
        if n_ai == self.nudge_interval:
            # Cross the threshold, then spawn the reflection review subagent the
            # same way the live agent does — via the genuine `task` tool.
            return self._result(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": _TASK_TOOL,
                            "id": "task-review",
                            "args": {
                                "subagent_type": _REVIEW_SUBAGENT,
                                "description": (
                                    "Reflect on this session. If a reusable procedure emerged, persist it as a "
                                    "skill with skill_manage(create), and save any durable user preference with "
                                    "memory(add)."
                                ),
                            },
                        }
                    ],
                )
            )
        return self._result(AIMessage(content="Done — reflected on the session and persisted what was reusable."))


class DemoReviewModel(_ScriptedModel):
    """The review subagent's scripted brain (the *aux* model).

    On its first turn it calls the real ``skill_manage`` + ``memory`` tools; once
    their results come back it returns a short final summary.
    """

    skill_name: str = DEMO_SKILL_NAME
    skill_description: str = DEMO_SKILL_DESCRIPTION
    skill_body: str = DEMO_SKILL_BODY
    skill_category: str = DEMO_SKILL_CATEGORY
    memory_target: str = DEMO_MEMORY_TARGET
    memory_entry: str = DEMO_MEMORY_ENTRY

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        already_acted = any(isinstance(m, ToolMessage) for m in messages)
        if not already_acted:
            return self._result(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "skill_manage",
                            "id": "skill-create",
                            "args": {
                                "action": "create",
                                "name": self.skill_name,
                                "description": self.skill_description,
                                "body": self.skill_body,
                                "category": self.skill_category,
                            },
                        },
                        {
                            "name": "memory",
                            "id": "memory-add",
                            "args": {
                                "action": "add",
                                "target": self.memory_target,
                                "entry": self.memory_entry,
                            },
                        },
                    ],
                )
            )
        return self._result(
            AIMessage(content=f"Crystallised the '{self.skill_name}' skill and saved one {self.memory_target} note.")
        )


# ── result ───────────────────────────────────────────────────────────────────


@dataclass
class DemoResult:
    """Everything the loop produced, for the CLI to render or a test to assert."""

    home: Path
    workspace: Path
    nudge_interval: int
    tool_iterations: int
    skill_created: bool
    skill_name: str | None = None
    skill_path: Path | None = None
    skill_frontmatter: dict[str, Any] = field(default_factory=dict)
    skill_body: str = ""
    memory_target: str | None = None
    memory_path: Path | None = None
    memory_entries: list[str] = field(default_factory=list)
    audit_actions: list[str] = field(default_factory=list)
    sessions_recorded: int = 0
    final_answer: str = ""


# ── driver ───────────────────────────────────────────────────────────────────


def run_demo(
    *,
    home: Path,
    workspace: Path | None = None,
    nudge_interval: int = 3,
    session_id: str = "demo-001",
) -> DemoResult:
    """Drive the real reflection→skill-creation loop offline and return the results.

    Args:
        home: Throwaway ``HERMES_HOME`` the demo writes its side effects under.
            Created if it does not exist. The caller owns cleanup.
        workspace: Filesystem root the agent's file tools operate in; defaults to
            ``home / "workspace"``.
        nudge_interval: Tool-using iterations before the review is spawned (drives
            both the skills and memory nudge intervals). Kept small so the demo is
            quick; the machinery is identical at the shipped default of 10.
        session_id: Session id for the run (also the FTS5 store key).

    Returns:
        A :class:`DemoResult` describing the generated skill, memory note, and the
        genuine side effects recorded (audit log + FTS5 store).
    """
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    workspace = Path(workspace) if workspace is not None else home / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # HERMES_HOME must be visible to config resolution AND to the memory tool /
    # store / library (which read it lazily during .invoke()). Set it for the
    # duration of the run and restore the prior environment afterwards so a
    # library caller (or the test suite) sees no lasting mutation.
    prev_env = {k: os.environ.get(k) for k in ("HERMES_HOME", "DEEPAGENT_HERMES_HOME")}
    os.environ["HERMES_HOME"] = str(home)
    os.environ["DEEPAGENT_HERMES_HOME"] = str(home)
    agent = None
    try:
        from langstage_hermes import HermesConfig, create_hermes_agent

        cfg = HermesConfig.resolve(
            overrides={
                "skills_creation_nudge_interval": nudge_interval,
                "memory_nudge_interval": nudge_interval,
            }
        )
        agent = create_hermes_agent(
            cfg,
            workspace=workspace,
            session_id=session_id,
            model=DemoMainModel(nudge_interval=nudge_interval),
            aux_model=DemoReviewModel(),
        )
        state = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Investigate why a couple of Python scripts are slow, using the same repeatable "
                            "procedure each time. When you spot a reusable pattern, persist it for next time."
                        ),
                    }
                ],
                "session_id": session_id,
                # Generous budget: nudge_interval tool turns + task + final.
                "iteration_budget_remaining": nudge_interval + 20,
            },
            config={"configurable": {"thread_id": session_id}},
        )
    finally:
        # Release the SQLite handles (store + audit log) BEFORE the caller
        # removes HERMES_HOME — on Windows an open connection locks state.db and
        # leaves the "cleaned up" dir behind (the whole point of gh #68).
        _close_agent_resources(agent)
        for key, val in prev_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    return _collect_result(
        home=home,
        workspace=workspace,
        nudge_interval=nudge_interval,
        state=state,
    )


def _close_agent_resources(agent: Any) -> None:
    """Close the agent's SQLite-backed resources (store + skill audit log).

    The FTS5 store and the audit log each hold a persistent ``sqlite3``
    connection to ``<HERMES_HOME>/state.db``. On Windows those open handles lock
    the file, so a caller that then ``rmtree``s the throwaway home is left with a
    half-removed directory. Closing here makes ``run_demo``'s side effects fully
    removable. Best-effort — a close failure must never break the demo.
    """
    if agent is None:
        return
    store = getattr(agent, "langstage_hermes_store", None)
    library = getattr(agent, "langstage_hermes_library", None)
    audit = getattr(library, "audit_log", None) if library is not None else None
    for resource in (store, audit):
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - defensive
                pass


def _last_ai_text(messages: list[Any]) -> str:
    last_ai = next((m for m in reversed(messages) if getattr(m, "type", None) == "ai"), None)
    content = getattr(last_ai, "content", "") if last_ai is not None else ""
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def _collect_result(*, home: Path, workspace: Path, nudge_interval: int, state: dict[str, Any]) -> DemoResult:
    """Read the real on-disk side effects the loop produced."""
    import frontmatter

    result = DemoResult(
        home=home,
        workspace=workspace,
        nudge_interval=nudge_interval,
        tool_iterations=nudge_interval,
        skill_created=False,
        final_answer=_last_ai_text(state.get("messages", [])),
    )

    # Generated SKILL.md — search the user skills dir the library writes to.
    skill_files = sorted((home / "skills").rglob("SKILL.md")) if (home / "skills").is_dir() else []
    if skill_files:
        skill_path = skill_files[0]
        post = frontmatter.load(str(skill_path))
        result.skill_created = True
        result.skill_path = skill_path
        result.skill_frontmatter = dict(post.metadata)
        result.skill_name = str(post.metadata.get("name", skill_path.parent.name))
        result.skill_body = post.content

    # Memory note — MEMORY.md / USER.md under memories/.
    for target, fname in (("user", "USER.md"), ("memory", "MEMORY.md")):
        mem_path = home / "memories" / fname
        if mem_path.is_file() and mem_path.read_text(encoding="utf-8").strip():
            entries = [e.strip() for e in mem_path.read_text(encoding="utf-8").split("\n§\n") if e.strip()]
            if entries:
                result.memory_target = target
                result.memory_path = mem_path
                result.memory_entries = entries
                break

    # Audit log + FTS5 session count — proof the genuine subsystems ran.
    db = home / "state.db"
    if db.exists():
        conn = sqlite3.connect(str(db))
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if "skill_mutations" in tables:
                result.audit_actions = [
                    r[0] for r in conn.execute("SELECT action FROM skill_mutations ORDER BY rowid").fetchall()
                ]
            if "sessions" in tables:
                result.sessions_recorded = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        except sqlite3.Error:  # pragma: no cover - defensive
            pass
        finally:
            conn.close()

    return result


__all__ = [
    "DemoMainModel",
    "DemoResult",
    "DemoReviewModel",
    "run_demo",
]
