"""Dogfood run: drive the agent through a substantive 12-turn arc.

I (the operator) am simulating a developer who:
  1. teaches the agent a specific code-style preference,
  2. exercises it across multiple functions,
  3. shifts domain to test whether the agent generalizes the preference,
  4. asks a meta question.

The point is to give the review subagent *something* to crystallize.
Trivial file-editing didn't trip a write; this should.

After every turn we dump counters + skill/memory deltas + FTS5 row count.
At the end we tail any SKILL.md or memory file the review wrote.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

PROMPTS = [
    # 1. Establish a preference — this is the kind of signal the review prompt
    # explicitly asks reviewers to capture under "Has the user expressed
    # expectations about how you should behave".
    "From now on, when you write any Python function for me, follow these "
    "rules without being asked: (a) full type hints on all params and return, "
    "(b) a Google-style docstring with Args / Returns / Raises sections, "
    "(c) pytest-style tests in the same response if I asked for code that's "
    "non-trivial, (d) prefer dataclasses over plain dicts for return values. "
    "Confirm you've understood.",
    # 2. Light application — small function. Use the write_file tool.
    "Write me a Python function `parse_duration(s: str) -> int` that converts "
    "strings like '30s', '5m', '2h' into seconds. Save it to durations.py "
    "with the write_file tool.",
    # 3. Reinforce by applying again — different function, same style.
    "Now write a function `format_bytes(n: int) -> str` that formats a byte "
    "count like 1024 -> '1.0 KiB', 1048576 -> '1.0 MiB'. Save to bytes_fmt.py.",
    # 4. Third instance of the same pattern.
    "And one more: `truncate(s: str, max_len: int = 80) -> str` that adds "
    "an ellipsis if the string was cut. Save to truncate.py.",
    # 5. Now a slightly bigger task — agent should apply the preference here.
    "Build me a small data class `RateLimitConfig` with fields "
    "(requests_per_second: float, burst: int = 10, retry_after_seconds: int = 60). "
    "Save to rate_limit.py.",
    # 6. Switch domain — ask for a README. The review should NOT save this as
    # a Python-only skill, but a thoughtful one might generalize to
    # "user values structured docs + examples".
    "Write a short README for the rate_limit.py module. Save as README.md.",
    # 7. Test the agent's recall of the preference.
    "Why did you put a docstring on RateLimitConfig and not just bare fields? Was that your call or was I particular about it?",
    # 8. Direct meta-instruction that points the review at memory rather than
    # skill — "I work like X" is squarely in MEMORY.md territory per the
    # prompt.
    "Worth knowing about me: I'm a data scientist, I publish OSS Python "
    "libraries, and I'd rather you flag dependency choices to me than silently "
    "install things. Don't tell me you've noted this, just behave that way.",
    # 9. Make the agent re-engage with the files it wrote so the conversation
    # has real procedural content for the reviewer to point at.
    "Read durations.py and bytes_fmt.py and tell me whether they handle "
    "negative inputs gracefully. Don't fix anything yet — just diagnose.",
    # 10. Have the agent apply its diagnosis.
    "OK, patch both so a negative input raises ValueError with a helpful "
    "message. Run the tests we wrote earlier — if they exist — to confirm.",
    # 11. Compress the arc into a takeaway. The review subagent's prompt
    # explicitly looks for "user's expectations about how you should behave".
    "Summarize: across this session, what conventions am I clearly asking "
    "you to follow? Keep it to a bulleted list, no preamble.",
    # 12. Explicit invitation — closes the loop deliberately.
    "Anything from this session you should remember for next time? If yes, use the appropriate tool to persist it now.",
]


def _last_ai_text(messages: list) -> str:
    last_ai = next((m for m in reversed(messages) if getattr(m, "type", None) == "ai"), None)
    content = getattr(last_ai, "content", "")
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


def _list_user_skills(library, hermes_home: Path) -> list[Path]:
    user_skill_dir = hermes_home / "skills"
    if not user_skill_dir.exists():
        return []
    return sorted(user_skill_dir.rglob("SKILL.md"))


def _file_size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; aborting.", file=sys.stderr)
        return 2

    tmp_home = Path(tempfile.mkdtemp(prefix="deepagent-hermes-dogfood-"))
    workspace = tmp_home / "workspace"
    workspace.mkdir()
    os.environ["DEEPAGENT_HERMES_HOME"] = str(tmp_home)
    os.environ["HERMES_HOME"] = str(tmp_home)
    # Default thresholds (10/10) — let one or two natural review fires
    # happen across 12 turns.

    print(f"HERMES_HOME = {tmp_home}")
    print(f"workspace = {workspace}")
    print("Building agent (default thresholds: 10 iters / 10 turns)...")
    t0 = time.perf_counter()

    from deepagent_hermes import HermesConfig, create_hermes_agent

    cfg = HermesConfig.resolve()
    sid = "dogfood-001"
    agent = create_hermes_agent(cfg, workspace=workspace, session_id=sid)
    print(f"  built in {time.perf_counter() - t0:.2f}s")
    print(f"  bundled skills: {len(agent.deepagent_hermes_library.list())}")
    print()

    memory_md = tmp_home / "memories" / "MEMORY.md"
    user_md = tmp_home / "memories" / "USER.md"

    last_skill_count = 0
    last_memory_size = 0
    last_user_size = 0

    config = {"configurable": {"thread_id": sid}}
    cumulative_response_chars = 0

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"=== Turn {i:2d}: {prompt[:75]}{'...' if len(prompt) > 75 else ''}")
        t = time.perf_counter()
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
            )
        except Exception as exc:
            print(f"  !! invoke raised: {type(exc).__name__}: {exc}")
            return 1

        elapsed = time.perf_counter() - t
        text = _last_ai_text(result.get("messages", []))
        cumulative_response_chars += len(text)
        snippet = text[:140].replace("\n", " ")
        print(f"  ({elapsed:5.1f}s, {len(text):4d} chars) -> {snippet!r}")

        # State counters
        iss = result.get("iters_since_skill", "?")
        tsm = result.get("turns_since_memory", "?")
        prk = result.get("pending_review_kind", None)
        msgs = len(result.get("messages", []))
        print(f"    counters: iters_since_skill={iss} turns_since_memory={tsm} pending={prk!r} msgs={msgs}")

        # Skill / memory deltas
        user_skills = _list_user_skills(agent.deepagent_hermes_library, tmp_home)
        new_skills = len(user_skills) - last_skill_count
        mem_delta = _file_size(memory_md) - last_memory_size
        user_delta = _file_size(user_md) - last_user_size
        deltas: list[str] = []
        if new_skills:
            deltas.append(f"+{new_skills} skill(s) ({', '.join(p.parent.name for p in user_skills[last_skill_count:])})")
        if mem_delta:
            deltas.append(f"+{mem_delta} bytes MEMORY.md")
        if user_delta:
            deltas.append(f"+{user_delta} bytes USER.md")
        if deltas:
            print(f"    *** DELTA: {'; '.join(deltas)}")
        last_skill_count = len(user_skills)
        last_memory_size = _file_size(memory_md)
        last_user_size = _file_size(user_md)
        print()

    # Final dump
    print("=" * 78)
    print("FINAL STATE")
    print("=" * 78)

    user_skills = _list_user_skills(agent.deepagent_hermes_library, tmp_home)
    print(f"\nUser-written skills: {len(user_skills)}")
    for p in user_skills:
        body = p.read_text(encoding="utf-8")
        print(f"\n--- {p.relative_to(tmp_home)} ({len(body)} chars) ---")
        # Show first 800 chars
        print(body[:800])
        if len(body) > 800:
            print(f"... (+{len(body) - 800} more chars)")

    print(f"\nMEMORY.md: {_file_size(memory_md)} bytes")
    if memory_md.exists():
        print(memory_md.read_text(encoding="utf-8")[:1500])

    print(f"\nUSER.md: {_file_size(user_md)} bytes")
    if user_md.exists():
        print(user_md.read_text(encoding="utf-8")[:1500])

    # Workspace files
    print(f"\nWorkspace files at {workspace}:")
    for p in sorted(workspace.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(workspace)} ({p.stat().st_size} bytes)")

    # FTS5
    import sqlite3

    db = tmp_home / "state.db"
    conn = sqlite3.connect(str(db))
    try:
        n_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        n_sess = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        print(f"\nFTS5: {n_sess} session(s), {n_msgs} messages")
    finally:
        conn.close()

    print(f"\nTotal model response chars: {cumulative_response_chars}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
