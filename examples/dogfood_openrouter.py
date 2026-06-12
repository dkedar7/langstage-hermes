"""Substantive multi-turn dogfood through OpenRouter + gpt-4o-mini.

Goes broader than ``tests/test_real_model.py`` — that test verifies the
plumbing holds. This script exercises the same surfaces a user would touch
on day one:

- Multi-turn substantive arc designed to give the review subagent both
  preference-style content (→ memory) and procedural content (→ skills).
- Runs through the same code paths the chat REPL uses (pretty extractors
  surface ``◆`` callouts for skill/memory events).
- Aggressive nudge intervals so we get at least two review fires inside
  the run.

Run::

    OPENROUTER_API_KEY=... python examples/dogfood_openrouter.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Five-prompt arc:
# 1. Preference declaration → memory candidate
# 2-4. Three uses of the same procedure → skill candidate
# 5. Explicit invitation → final review
PROMPTS = [
    # 1. preference (memory)
    "From now on when you write Python for me: full type hints, Google-"
    "style docstrings (Args / Returns / Raises), pytest tests in the same "
    "reply if the function is non-trivial. Confirm briefly.",
    # 2-4. apply the procedure to three small functions
    "Write `parse_duration(s: str) -> int` that converts strings like "
    "'30s', '5m', '2h' into seconds. Save to durations.py with write_file.",
    "Now write `format_bytes(n: int) -> str` that formats 1024 -> '1.0 KiB', 1048576 -> '1.0 MiB'. Save to bytes_fmt.py.",
    "One more: `truncate(s: str, max_len: int = 80) -> str` that adds an ellipsis when cut. Save to truncate.py.",
    # 5. invite persistence
    "Across these three, what's the pattern you're following? If you have "
    "tools to persist that pattern (skills / memory), use them now so a "
    "fresh session starts knowing it.",
]


def _last_ai_text(messages: list) -> str:
    last_ai = next((m for m in reversed(messages) if getattr(m, "type", None) == "ai"), None)
    content = getattr(last_ai, "content", "")
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


def _setup_openrouter() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set; aborting.", file=sys.stderr)
        sys.exit(2)
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    os.environ["DEEPAGENT_HERMES_MODEL_DEFAULT"] = "openai:openai/gpt-4o-mini"
    os.environ["DEEPAGENT_HERMES_MODEL_AUX"] = "openai:openai/gpt-4o-mini"


def main() -> int:
    _setup_openrouter()

    tmp_home = Path(tempfile.mkdtemp(prefix="dah-openrouter-"))
    workspace = tmp_home / "workspace"
    workspace.mkdir()
    os.environ["DEEPAGENT_HERMES_HOME"] = str(tmp_home)
    os.environ["HERMES_HOME"] = str(tmp_home)
    os.environ["DEEPAGENT_HERMES_SKILLS_CREATION_NUDGE_INTERVAL"] = "3"
    os.environ["DEEPAGENT_HERMES_MEMORY_NUDGE_INTERVAL"] = "3"

    print(f"HERMES_HOME = {tmp_home}")
    print(f"workspace   = {workspace}")
    print("model       = openai:openai/gpt-4o-mini (via OpenRouter)")
    print("thresholds  = skills=3, memory=3")
    print()

    t0 = time.perf_counter()
    from langstage_hermes import HermesConfig, create_hermes_agent

    cfg = HermesConfig.resolve()
    sid = "openrouter-dogfood-001"
    agent = create_hermes_agent(cfg, workspace=workspace, session_id=sid)
    print(f"agent built in {time.perf_counter() - t0:.2f}s ({len(agent.langstage_hermes_library.list())} bundled skills)")
    print()

    memory_md = tmp_home / "memories" / "MEMORY.md"
    user_md = tmp_home / "memories" / "USER.md"
    user_skill_dir = tmp_home / "skills"

    def _size(p: Path) -> int:
        return p.stat().st_size if p.exists() else 0

    def _count_user_skills() -> int:
        return len(list(user_skill_dir.rglob("SKILL.md"))) if user_skill_dir.exists() else 0

    last_mem = 0
    last_user = 0
    last_skills = 0
    config = {"configurable": {"thread_id": sid}}

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"=== Turn {i} ============================================")
        print(f"  user: {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
        t = time.perf_counter()
        try:
            result = agent.invoke(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "session_id": sid,
                    "iteration_budget_remaining": cfg.agent_max_iterations,
                },
                config=config,
            )
        except Exception as exc:
            print(f"  !! {type(exc).__name__}: {exc}")
            return 1
        elapsed = time.perf_counter() - t

        text = _last_ai_text(result.get("messages", []))
        snippet = text[:160].replace("\n", " ")
        print(f"  reply ({elapsed:.1f}s, {len(text)} chars): {snippet!r}")

        iss = result.get("iters_since_skill", "?")
        tsm = result.get("turns_since_memory", "?")
        prk = result.get("pending_review_kind")
        print(f"  counters: iters_since_skill={iss}  turns_since_memory={tsm}  pending={prk!r}")

        mem_delta = _size(memory_md) - last_mem
        usr_delta = _size(user_md) - last_user
        skills_delta = _count_user_skills() - last_skills
        if mem_delta or usr_delta or skills_delta:
            parts: list[str] = []
            if skills_delta > 0:
                parts.append(f"+{skills_delta} SKILL.md")
            if mem_delta:
                parts.append(f"MEMORY.md +{mem_delta}B")
            if usr_delta:
                parts.append(f"USER.md +{usr_delta}B")
            print(f"  *** DELTA: {', '.join(parts)}")
        last_mem = _size(memory_md)
        last_user = _size(user_md)
        last_skills = _count_user_skills()
        print()

    # ── Final state ──
    print("=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    print(f"\nUSER.md  ({_size(user_md)}B):")
    if user_md.exists():
        print("  " + user_md.read_text(encoding="utf-8").replace("\n", "\n  ")[:1200])
    print(f"\nMEMORY.md  ({_size(memory_md)}B):")
    if memory_md.exists():
        print("  " + memory_md.read_text(encoding="utf-8").replace("\n", "\n  ")[:1200])

    skills = sorted(user_skill_dir.rglob("SKILL.md")) if user_skill_dir.exists() else []
    print(f"\nUser-written SKILL.md files: {len(skills)}")
    for p in skills:
        body = p.read_text(encoding="utf-8")
        print(f"\n--- {p.relative_to(tmp_home)} ({len(body)} chars) ---")
        print(body[:800])

    # FTS5
    import sqlite3

    db = tmp_home / "state.db"
    if db.exists():
        conn = sqlite3.connect(str(db))
        try:
            n_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            n_sess = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            conn.close()
        print(f"\nFTS5 store: {n_sess} session(s), {n_msgs} message row(s)")

    # Workspace files
    print(f"\nWorkspace files at {workspace}:")
    if workspace.exists():
        for p in sorted(workspace.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(workspace)} ({p.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
