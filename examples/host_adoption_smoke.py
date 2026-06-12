"""Host-adoption smoke: load the Hermes graph via the deepagent-code host machinery.

Validates the contract every deepagent-* host depends on:
``DEEPAGENT_AGENT_SPEC=langstage_hermes.agent:graph`` resolves through
``langgraph_stream_parser.host.load_agent_spec`` and the resulting graph
is invokable.

Does NOT actually launch the deepagent-code REPL (interactive), just
proves the loader contract holds end-to-end and the graph can stream
its first turn.
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

    tmp_home = Path(tempfile.mkdtemp(prefix="deepagent-hermes-host-"))
    os.environ["DEEPAGENT_HERMES_HOME"] = str(tmp_home)
    os.environ["HERMES_HOME"] = str(tmp_home)
    os.environ["DEEPAGENT_AGENT_SPEC"] = "langstage_hermes.agent:graph"

    print(f"HERMES_HOME = {tmp_home}")
    print(f"DEEPAGENT_AGENT_SPEC = {os.environ['DEEPAGENT_AGENT_SPEC']}")
    print()

    # ─── 1. Verify load_agent_spec resolves the entry point ────────────
    print("[1] load_agent_spec('langstage_hermes.agent:graph') ...")
    t0 = time.perf_counter()
    from langgraph_stream_parser.host import load_agent_spec

    graph = load_agent_spec("langstage_hermes.agent:graph")
    print(f"    resolved in {time.perf_counter() - t0:.2f}s -> {type(graph).__name__}")
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "stream")
    print(f"    config = {getattr(graph, 'langstage_hermes_config', None)!r}"[:120])
    print(f"    session_id = {getattr(graph, 'langstage_hermes_session_id', None)!r}")
    print(f"    bundled skills = {len(graph.langstage_hermes_library.list())}")
    print()

    # ─── 2. Verify deepagent-code can construct its config + load same spec ─
    print("[2] deepagent_code.config picks up DEEPAGENT_AGENT_SPEC ...")
    try:
        from deepagent_code.config import CodeConfig
    except ImportError:
        # config module may have moved across deepagent-code versions.
        from deepagent_code import config as code_config  # type: ignore[no-redef]

        CodeConfig = getattr(code_config, "CodeConfig", None)
    if CodeConfig is None:
        print("    !! could not locate CodeConfig; skipping deepagent-code config check")
    else:
        cfg = CodeConfig.resolve()
        spec = getattr(cfg, "agent_spec", None)
        print(f"    cfg.agent_spec = {spec!r}")
        assert spec == "langstage_hermes.agent:graph", f"expected hermes spec, got {spec!r}"
        print("    OK — deepagent-code would load the hermes agent")
    print()

    # ─── 3. Drive one real turn through the StreamParser like a host would ─
    print("[3] StreamParser round-trip (real Anthropic call) ...")
    from langgraph_stream_parser import StreamParser

    parser = StreamParser()
    sid = "host-smoke-001"
    config = {"configurable": {"thread_id": sid}}
    events_seen: dict[str, int] = {}
    t = time.perf_counter()
    try:
        for event in parser.parse(
            graph.stream(
                {
                    "messages": [{"role": "user", "content": "Reply with one word: yes"}],
                    "session_id": sid,
                },
                config=config,
                stream_mode="updates",
            )
        ):
            etype = type(event).__name__
            events_seen[etype] = events_seen.get(etype, 0) + 1
    except Exception as exc:
        print(f"    !! stream raised: {type(exc).__name__}: {exc}")
        return 1
    print(f"    streamed in {time.perf_counter() - t:.2f}s")
    print(f"    events by type: {events_seen}")
    print()

    # ─── 4. Confirm side effects landed ────────────────────────────────
    print("[4] side-effect verification ...")
    import sqlite3

    db = tmp_home / "state.db"
    if db.exists():
        conn = sqlite3.connect(str(db))
        try:
            n_sess = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            n_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        finally:
            conn.close()
        print(f"    FTS5 store: {n_sess} session(s), {n_msgs} message row(s)")
    else:
        print("    !! state.db not created")
        return 1
    print()

    print("HOST-ADOPTION SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
