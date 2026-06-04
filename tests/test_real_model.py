"""Opt-in real-model eval: a live reflection cycle through the full stack.

Every other test uses fakes/stubs. This one runs hermes's real middleware
stack (prompt assembly, caching, the reflection trigger, the state recorder)
with a live model (OpenRouter / gpt-4o-mini) over several tool-using turns and
asserts the *integration* holds:

- multi-turn tool use completes without error through the real graph,
- the reflection-counter machinery engages,
- the state recorder persists the session + messages to the FTS5 store.

It deliberately does **not** assert skill/memory *creation* — whether the agent
chooses to author a SKILL.md is model-quality-dependent and flaky on a cheap
model. That belongs in a separate, statistical LLM-as-judge eval.

Opt-in: skips unless ``OPENROUTER_API_KEY`` is set and ``langchain-openai`` (the
``real`` extra) is installed, so default CI stays free and deterministic. Run::

    uv pip install -e ".[dev,real]"
    OPENROUTER_API_KEY=... pytest tests/test_real_model.py -v
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

pytestmark = pytest.mark.real_model


def _setup_openrouter_env() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set — skipping real-model eval")
    pytest.importorskip("langchain_openai")
    key = os.environ["OPENROUTER_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["OPENAI_API_KEY"] = key
    os.environ["DEEPAGENT_HERMES_MODEL_DEFAULT"] = "openai:openai/gpt-4o-mini"
    os.environ["DEEPAGENT_HERMES_MODEL_AUX"] = "openai:openai/gpt-4o-mini"


def test_real_reflection_cycle_runs_and_persists_state(tmp_path: Path):
    _setup_openrouter_env()

    home = tmp_path / "hermes-home"
    home.mkdir()
    os.environ["DEEPAGENT_HERMES_HOME"] = str(home)
    os.environ["HERMES_HOME"] = str(home)
    # Aggressive trigger so the reflection machinery fires inside one run.
    os.environ["DEEPAGENT_HERMES_SKILLS_CREATION_NUDGE_INTERVAL"] = "2"
    os.environ["DEEPAGENT_HERMES_MEMORY_NUDGE_INTERVAL"] = "2"

    from deepagent_hermes import HermesConfig, create_hermes_agent

    cfg = HermesConfig.resolve()
    assert cfg.model_default == "openai:openai/gpt-4o-mini"

    session_id = "real-reflect-test"
    agent = create_hermes_agent(cfg, workspace=home, session_id=session_id)
    config = {"configurable": {"thread_id": session_id}}

    prompts = [
        "Use the write_file tool to create notes.txt with content 'first line'.",
        "Use the read_file tool to read notes.txt and tell me what it says.",
        "Use the ls tool to list the files in the current directory.",
    ]

    last_iters = None
    for prompt in prompts:
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
        msgs = result.get("messages", [])
        assert any(getattr(m, "type", None) == "ai" for m in msgs), "expected an AI reply"
        last_iters = result.get("iters_since_skill", last_iters)

    # Reflection-counter machinery engaged (the key is present + an int).
    assert isinstance(last_iters, int), "reflection counter should be tracked in state"

    # State recorder persisted the conversation to the FTS5 store.
    db = home / "state.db"
    assert db.exists(), "state.db (FTS5 store) should have been created"
    conn = sqlite3.connect(str(db))
    try:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()
    assert sessions >= 1, "expected the session to be recorded"
    assert messages > 0, "expected messages to be persisted to the store"
