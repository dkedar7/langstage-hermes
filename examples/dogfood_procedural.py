"""Procedural-arc dogfood: target the skill-write path.

The first dogfood (`dogfood.py`) gave the review subagent code-style
preferences — which correctly went to MEMORY.md, not skills. This one
gives it a multi-step *procedure* that's reusable across tasks — exactly
the shape SKILL.md is meant for.

Arc: "help me investigate why a Python script is slow", then repeat the
investigation pattern on a second script. The reusable procedure is:
read the script → guess hotspots → instrument with cProfile → analyze
output → propose fix. If the review subagent works at all on procedural
content, it should crystallize that pattern.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Two scripts with deliberate inefficiencies — the agent will exercise the
# same investigation pattern on both.
SLOW_SCRIPT_1 = '''\
"""Compute pairwise distances between 2000 points. Slow on purpose."""
import math
import random

points = [(random.random(), random.random()) for _ in range(2000)]

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

total = 0.0
for i, p in enumerate(points):
    for j, q in enumerate(points):
        if i < j:
            total += dist(p, q)
print(f"sum of pairwise distances: {total:.2f}")
'''

SLOW_SCRIPT_2 = '''\
"""Count word frequencies in a synthetic corpus. Slow on purpose."""
import random
import string

def random_word():
    return "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))

corpus = [random_word() for _ in range(50_000)]

counts = {}
for word in corpus:
    if word in counts:
        counts[word] = counts[word] + 1
    else:
        counts[word] = 1

top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
print("top 5:", top)
'''


PROMPTS = [
    # 0. Drop both scripts into the workspace via the agent.
    f"Save this script to slow1.py and then we'll work with it:\n\n```python\n{SLOW_SCRIPT_1}```",
    # 1. Procedural step 1.
    "I think slow1.py is slow. Walk me through how you'd investigate it: "
    "your investigation process, not the answer yet. Be specific about which "
    "tools/commands you'd run.",
    # 2. Procedural step 2 — execute the investigation.
    "OK, do the investigation. Read the file, identify the hotspot from first "
    "principles, then propose ONE concrete fix. Don't apply the fix.",
    # 3. Apply.
    "Apply the fix and save it back to slow1.py.",
    # 4. Repeat the pattern on a different script — this is the key reinforcement
    # the review subagent should pick up as "procedure to crystallize".
    f"Now do the same thing for this other script. Save it to slow2.py first, "
    f"then run your investigation procedure:\n\n```python\n{SLOW_SCRIPT_2}```",
    # 5. Make the pattern explicit.
    "Across slow1 and slow2, what's your repeatable investigation procedure? "
    "Describe it as steps a less-experienced engineer could follow.",
    # 6. Reinforce + invite persistence.
    "Good. We're going to do this kind of investigation a lot. If you have "
    "tools to persist that procedure so future sessions start with it, use them now.",
    # 7. Light test of recall in a new context.
    "If I bring you a third slow script tomorrow, what's the first thing you'd do? Keep it brief.",
]


def _last_ai_text(messages: list) -> str:
    last_ai = next((m for m in reversed(messages) if getattr(m, "type", None) == "ai"), None)
    content = getattr(last_ai, "content", "")
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content


def _file_size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; aborting.", file=sys.stderr)
        return 2

    tmp_home = Path(tempfile.mkdtemp(prefix="deepagent-hermes-proc-"))
    workspace = tmp_home / "workspace"
    workspace.mkdir()
    os.environ["DEEPAGENT_HERMES_HOME"] = str(tmp_home)
    os.environ["HERMES_HOME"] = str(tmp_home)
    # Aggressive thresholds — 8 turns means we want review fires by turn 5-6.
    os.environ["DEEPAGENT_HERMES_SKILLS_CREATION_NUDGE_INTERVAL"] = "4"
    os.environ["DEEPAGENT_HERMES_MEMORY_NUDGE_INTERVAL"] = "4"

    print(f"HERMES_HOME = {tmp_home}")
    print(f"workspace  = {workspace}")
    print("Building agent (skills/memory nudge_interval=4)...")
    t0 = time.perf_counter()

    from deepagent_hermes import HermesConfig, create_hermes_agent

    cfg = HermesConfig.resolve()
    sid = "proc-dogfood-001"
    agent = create_hermes_agent(cfg, workspace=workspace, session_id=sid)
    print(f"  built in {time.perf_counter() - t0:.2f}s")
    print()

    user_skill_dir = tmp_home / "skills"
    memory_md = tmp_home / "memories" / "MEMORY.md"
    user_md = tmp_home / "memories" / "USER.md"

    last_skill_count = 0
    last_memory_size = 0
    last_user_size = 0

    config = {"configurable": {"thread_id": sid}}

    for i, prompt in enumerate(PROMPTS, 1):
        first_line = prompt.split("\n", 1)[0]
        print(f"=== Turn {i}: {first_line[:78]}{'...' if len(first_line) > 78 else ''}")
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
        snippet = text[:150].replace("\n", " ")
        print(f"  ({elapsed:5.1f}s, {len(text):4d} chars) -> {snippet!r}")
        iss = result.get("iters_since_skill", "?")
        tsm = result.get("turns_since_memory", "?")
        prk = result.get("pending_review_kind", None)
        print(f"    counters: iters={iss} mem={tsm} pending={prk!r}")

        # Deltas
        new_skills = []
        if user_skill_dir.exists():
            cur = sorted(user_skill_dir.rglob("SKILL.md"))
            new_skills = cur[last_skill_count:]
            last_skill_count = len(cur)
        if new_skills:
            for p in new_skills:
                print(f"    *** NEW SKILL: {p.relative_to(tmp_home)}")

        mem_delta = _file_size(memory_md) - last_memory_size
        user_delta = _file_size(user_md) - last_user_size
        if mem_delta:
            print(f"    *** MEMORY.md +{mem_delta} bytes")
        if user_delta:
            print(f"    *** USER.md +{user_delta} bytes")
        last_memory_size = _file_size(memory_md)
        last_user_size = _file_size(user_md)
        print()

    # ── Final dump ──
    print("=" * 70)
    print("FINAL STATE")
    print("=" * 70)

    if user_skill_dir.exists():
        skills = sorted(user_skill_dir.rglob("SKILL.md"))
        print(f"\nUser-written SKILL.md files: {len(skills)}")
        for p in skills:
            body = p.read_text(encoding="utf-8")
            print(f"\n--- {p.relative_to(tmp_home)} ({len(body)} chars) ---")
            print(body[:1200])
            if len(body) > 1200:
                print(f"... (+{len(body) - 1200} more)")
    else:
        print("\nNo user skill dir created.")

    print(f"\nMEMORY.md: {_file_size(memory_md)} bytes")
    if memory_md.exists():
        print(memory_md.read_text(encoding="utf-8")[:1000])
    print(f"\nUSER.md: {_file_size(user_md)} bytes")
    if user_md.exists():
        print(user_md.read_text(encoding="utf-8")[:1000])

    return 0


if __name__ == "__main__":
    sys.exit(main())
