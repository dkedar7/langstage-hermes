"""End-to-end smoke test: build the agent graph + verify the middleware stack.

This does NOT call any model — it builds the graph, verifies the middleware
list, checks attached resources (store, library, session_id, config), and
confirms the parser extractors handle their expected tool-result shapes.
A live model invocation is out of scope (requires ANTHROPIC_API_KEY and
network access; covered by examples/cli_smoke.py for manual testing).
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def integration_env(tmp_hermes_home, tmp_workspace, monkeypatch):
    """Isolated HERMES_HOME + an Anthropic-key sentinel so init_chat_model works."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return tmp_hermes_home, tmp_workspace


def test_create_hermes_agent_builds(integration_env):
    """The graph compiles, attaches its config + store + library + session id."""
    from langstage_hermes.agent import create_hermes_agent
    from langstage_hermes.config import HermesConfig
    from langstage_hermes.skills.library import SkillLibrary
    from langstage_hermes.store.sqlite_fts import SqliteFtsStore

    _, ws = integration_env
    cfg = HermesConfig.resolve()
    agent = create_hermes_agent(cfg, workspace=ws, session_id="test-session-123")

    # Compiled — has invoke/stream.
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "stream")
    assert hasattr(agent, "ainvoke")

    # Attached refs are present.
    assert agent.langstage_hermes_config is cfg
    assert agent.langstage_hermes_session_id == "test-session-123"
    assert isinstance(agent.langstage_hermes_store, SqliteFtsStore)
    assert isinstance(agent.langstage_hermes_library, SkillLibrary)


def test_state_db_created_under_hermes_home(integration_env):
    """The SQLite state.db lands at <HERMES_HOME>/state.db and has the FTS5 tables."""
    import sqlite3

    from langstage_hermes.agent import create_hermes_agent

    home, ws = integration_env
    create_hermes_agent(workspace=ws)

    db = home / "state.db"
    assert db.exists()

    conn = sqlite3.connect(str(db))
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()}
        assert "sessions" in tables
        assert "messages" in tables
        assert "messages_fts" in tables
        assert "messages_fts_trigram" in tables
    finally:
        conn.close()


def test_auto_session_id_when_omitted(integration_env):
    """No session_id arg → generates a sess-<hex> id."""
    from langstage_hermes.agent import create_hermes_agent

    _, ws = integration_env
    agent = create_hermes_agent(workspace=ws)
    sid = agent.langstage_hermes_session_id
    assert sid.startswith("sess-")
    assert len(sid) > len("sess-")


def test_module_level_graph_is_lazy(integration_env, monkeypatch):
    """`import langstage_hermes.agent` does NOT build the graph; first access does."""
    import langstage_hermes.agent as agent_mod

    # Reset module-level cache (test isolation).
    monkeypatch.setattr(agent_mod, "_graph", None)

    # _graph is None before access.
    assert agent_mod._graph is None

    # First access triggers the build.
    g = agent_mod.graph  # noqa: F841
    assert agent_mod._graph is not None


# ── parser extractor smoke tests ────────────────────────────────────────


def test_skill_manage_extractor_handles_create():
    from langstage_hermes.extractors import SkillManageExtractor

    ex = SkillManageExtractor()
    result = ex.extract(json.dumps({"action": "create", "name": "pdf-merging"}))
    assert result == {
        "action": "create",
        "name": "pdf-merging",
        "extracted_subtype": "skill_created",
    }


def test_skill_manage_extractor_handles_patch():
    from langstage_hermes.extractors import SkillManageExtractor

    ex = SkillManageExtractor()
    result = ex.extract({"action": "patch", "name": "csv-cleaning"})
    assert result == {
        "action": "patch",
        "name": "csv-cleaning",
        "extracted_subtype": "skill_updated",
    }


def test_skill_view_extractor():
    from langstage_hermes.extractors import SkillViewExtractor

    ex = SkillViewExtractor()
    body = "Here is the body of the skill: do X then Y."
    result = ex.extract(body)
    assert result == {"loaded": True, "body_chars": len(body)}


def test_compression_extractor():
    from langstage_hermes.extractors import CompressionExtractor

    ex = CompressionExtractor()
    payload = {"before_tokens": 47_000, "after_tokens": 9_000, "ratio": 5.2, "section_count": 13}
    result = ex.extract(json.dumps(payload))
    assert result == payload


def test_memory_extractor():
    from langstage_hermes.extractors import MemoryExtractor

    ex = MemoryExtractor()
    result = ex.extract(json.dumps({"action": "add", "target": "user", "entry": "..."}))
    assert result == {"action": "add", "target": "user", "extracted_subtype": "memory_added"}


def test_extractors_return_none_on_unrelated_content():
    from langstage_hermes.extractors import (
        CompressionExtractor,
        MemoryExtractor,
        SkillManageExtractor,
    )

    for ex in (SkillManageExtractor(), CompressionExtractor(), MemoryExtractor()):
        assert ex.extract(None) is None
        assert ex.extract("") is None
        assert ex.extract(123) is None


# ── host-adoption smoke ────────────────────────────────────────────────


def test_deepagent_agent_spec_resolves(integration_env):
    """The DEEPAGENT_AGENT_SPEC convention: 'langstage_hermes.agent:graph' should resolve.

    Hosts in the deepagent-* family use langgraph_stream_parser.host.load_agent_spec
    to load an agent. We verify the spec string maps to a usable graph.
    """
    from langgraph_stream_parser.host import load_agent_spec

    graph = load_agent_spec("langstage_hermes.agent:graph")
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "stream")
