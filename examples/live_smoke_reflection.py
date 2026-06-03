"""Multi-turn live smoke that aims to trip the reflection trigger.

Sets `skills.creation_nudge_interval = 3` (down from default 10) and issues a
sequence of tool-using prompts so the agent crosses the threshold inside one
script run. Then reads the skill library to see if a SKILL.md was created.

Run:  python examples/live_smoke_reflection.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; aborting.", file=sys.stderr)
        return 2

    tmp_home = Path(tempfile.mkdtemp(prefix="deepagent-hermes-reflect-"))
    os.environ["DEEPAGENT_HERMES_HOME"] = str(tmp_home)
    os.environ["HERMES_HOME"] = str(tmp_home)
    # Aggressive trigger so we see reflection inside one run.
    os.environ["DEEPAGENT_HERMES_SKILLS_CREATION_NUDGE_INTERVAL"] = "3"
    os.environ["DEEPAGENT_HERMES_MEMORY_NUDGE_INTERVAL"] = "3"

    print(f"HERMES_HOME = {tmp_home}")
    print("Building agent (skills nudge_interval=3, memory nudge_interval=3)...")
    t0 = time.perf_counter()

    from deepagent_hermes import HermesConfig, create_hermes_agent

    cfg = HermesConfig.resolve()
    sid = "reflect-smoke-001"
    agent = create_hermes_agent(cfg, workspace=tmp_home, session_id=sid)
    print(f"  built in {time.perf_counter() - t0:.2f}s")
    print(f"  bundled skills loaded: {len(agent.deepagent_hermes_library.list())}")
    print()

    prompts = [
        "Use the write_file tool to create a file called notes.txt with the content 'first line'.",
        "Use the read_file tool to read notes.txt and tell me what it says.",
        "Use the write_file tool to append a second line. Just call it once with the new full content.",
        "List the files in the current directory using the ls tool.",
        "Use the read_file tool one more time to confirm notes.txt has both lines now.",
    ]

    config = {"configurable": {"thread_id": sid}}

    for i, prompt in enumerate(prompts, 1):
        print(f"=== Turn {i}: {prompt[:60]}{'...' if len(prompt) > 60 else ''} ===")
        t = time.perf_counter()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
        )
        elapsed = time.perf_counter() - t
        msgs = result.get("messages", [])
        last_ai = next((m for m in reversed(msgs) if getattr(m, "type", None) == "ai"), None)
        last_content = getattr(last_ai, "content", "<no AIMessage>")
        if isinstance(last_content, list):
            # Anthropic content-block format
            last_content = "".join(
                b.get("text", "") for b in last_content if isinstance(b, dict)
            )
        snippet = (last_content[:100] + "...") if len(last_content) > 100 else last_content
        print(f"  ({elapsed:.2f}s) -> {snippet!r}")
        print(
            f"  iters_since_skill={result.get('iters_since_skill', '?')} "
            f"turns_since_memory={result.get('turns_since_memory', '?')} "
            f"pending_review_kind={result.get('pending_review_kind', None)!r} "
            f"messages={len(msgs)}"
        )

    print()
    print("--- Skill library inspection ---")
    library = agent.deepagent_hermes_library
    bundled = library.list()
    print(f"Total skills in library: {len(bundled)}")
    # Filter to user-created (not under the bundled dir)
    user_skills = [
        s for s in bundled
        if "deepagent-hermes\\skills" not in str(s.path)
        and "deepagent-hermes/skills" not in str(s.path)
    ]
    print(f"User-created skills: {len(user_skills)}")
    for s in user_skills:
        print(f"  - {s.name}: {s.description[:80]}")

    user_skill_dir = tmp_home / "skills"
    new_skills = list(user_skill_dir.rglob("SKILL.md")) if user_skill_dir.exists() else []
    print(f"SKILL.md files written under HERMES_HOME/skills: {len(new_skills)}")
    for p in new_skills:
        print(f"  - {p.relative_to(tmp_home)}")

    print()
    print("--- FTS5 store inspection ---")
    import sqlite3

    db = tmp_home / "state.db"
    conn = sqlite3.connect(str(db))
    try:
        n_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        n_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        print(f"FTS5: {n_sessions} session(s), {n_msgs} message row(s)")
    finally:
        conn.close()

    print()
    print("--- Memory inspection ---")
    memory_md = tmp_home / "memories" / "MEMORY.md"
    user_md = tmp_home / "memories" / "USER.md"
    print(f"MEMORY.md exists: {memory_md.exists()}  size: {memory_md.stat().st_size if memory_md.exists() else 0}")
    print(f"USER.md exists: {user_md.exists()}  size: {user_md.stat().st_size if user_md.exists() else 0}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
