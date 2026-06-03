"""Live smoke test against the Anthropic API.

Hits Anthropic once with a trivial prompt to prove the agent graph builds,
compiles, and successfully round-trips through the model. Writes any new
SKILL.md / MEMORY.md state under a tmp HERMES_HOME so it doesn't pollute
the user's library.

Run:  python examples/live_smoke.py
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

    tmp_home = Path(tempfile.mkdtemp(prefix="deepagent-hermes-live-"))
    os.environ["DEEPAGENT_HERMES_HOME"] = str(tmp_home)
    os.environ["HERMES_HOME"] = str(tmp_home)

    print(f"HERMES_HOME = {tmp_home}")
    print("Building agent...")
    t0 = time.perf_counter()

    from deepagent_hermes import HermesConfig, create_hermes_agent

    cfg = HermesConfig.resolve()
    agent = create_hermes_agent(cfg, workspace=tmp_home, session_id="live-smoke-001")

    print(f"  built in {time.perf_counter() - t0:.2f}s")
    print(f"  model: {cfg.model_default}")
    print(f"  session_id: {agent.deepagent_hermes_session_id}")
    print(f"  state.db: {tmp_home / 'state.db'} ({(tmp_home / 'state.db').stat().st_size} bytes)")
    print()
    print("Invoking with: 'Reply with exactly one word: hello'")
    print("---")

    t0 = time.perf_counter()
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Reply with exactly one word: hello"}],
            "session_id": "live-smoke-001",
            "iteration_budget_remaining": cfg.agent_max_iterations,
        }
    )
    elapsed = time.perf_counter() - t0

    # Last AIMessage content.
    msgs = result.get("messages", [])
    last_ai = next((m for m in reversed(msgs) if getattr(m, "type", None) == "ai"), None)
    last_content = getattr(last_ai, "content", "<no AIMessage found>")

    print(f"Response ({elapsed:.2f}s): {last_content!r}")
    print()
    print(f"Total messages in state: {len(msgs)}")
    print(f"iters_since_skill: {result.get('iters_since_skill', 0)}")
    print(f"turns_since_memory: {result.get('turns_since_memory', 0)}")
    print(f"iteration_budget_remaining: {result.get('iteration_budget_remaining', '?')}")

    # Verify FTS5 recorded the turn.
    import sqlite3

    conn = sqlite3.connect(str(tmp_home / "state.db"))
    try:
        n_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        n_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        print(f"FTS5: {n_sessions} session(s), {n_msgs} message row(s)")
    finally:
        conn.close()

    print()
    print("SUCCESS" if last_content else "EMPTY RESPONSE")
    return 0 if last_content else 1


if __name__ == "__main__":
    sys.exit(main())
