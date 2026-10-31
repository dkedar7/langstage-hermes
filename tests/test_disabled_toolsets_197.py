"""gh #197: `[agent].disabled_toolsets` removes the toolsets' tools from the built agent,
not just from the prompt."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langstage_hermes.agent import create_hermes_agent
from langstage_hermes.cli import _slash_toolsets
from langstage_hermes.config import HermesConfig
from langstage_hermes.tools.toolsets import tool_allowed, toolset_of, unknown_toolsets

ALL_BOUND = {
    "delete", "edit_file", "execute", "glob", "grep", "ls", "memory", "read_file",
    "session_search", "skill_manage", "skill_view", "skills_list", "task", "write_file", "write_todos",
}  # fmt: skip


class _Fake(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LANGSTAGE_HERMES_AGENT_DISABLED_TOOLSETS", raising=False)
    monkeypatch.delenv("DEEPAGENT_HERMES_AGENT_DISABLED_TOOLSETS", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _build(disabled: list[str], ws: Path, replies: list[AIMessage] | None = None) -> Any:
    cfg = HermesConfig.resolve(overrides={"agent_disabled_toolsets": disabled})
    model = _Fake(messages=iter(replies or [AIMessage("ok")]))
    return create_hermes_agent(cfg, model=model, aux_model=_Fake(messages=iter([AIMessage("x")] * 5)), workspace=ws)


def _bound(graph: Any) -> set[str]:
    return set(graph.nodes["tools"].bound.tools_by_name)


def test_nothing_disabled_binds_every_tool(home: Path):
    assert _bound(_build([], home)) == ALL_BOUND


def test_issue_repro_disabled_file_toolset_cannot_write(home: Path):
    ws = home / "ws"
    ws.mkdir()
    call = AIMessage("", tool_calls=[{"name": "write_file", "args": {"file_path": "/pwned.txt", "content": "x"}, "id": "c1"}])
    graph = _build(["file", "memory", "skills", "session_search"], ws, [call, AIMessage("done")])

    assert _bound(graph) == {"execute", "task", "write_todos"}
    out = graph.invoke({"messages": [HumanMessage("go")]})
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert "not a valid tool" in tool_msgs[0].content
    assert list(ws.iterdir()) == []


@pytest.mark.parametrize(
    ("toolset", "gone"),
    [
        ("file", {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep"}),
        ("terminal", {"execute"}),
        ("todo", {"write_todos"}),
        ("delegation", {"task"}),
        ("memory", {"memory"}),
        ("skills", {"skills_list", "skill_view", "skill_manage"}),
        ("session_search", {"session_search"}),
    ],
)
def test_each_toolset_removes_exactly_its_tools(home: Path, toolset: str, gone: set[str]):
    assert _bound(_build([toolset], home)) == ALL_BOUND - gone


def test_disabled_todo_and_delegation_drop_their_middleware(home: Path):
    from deepagents.middleware.subagents import SubAgentMiddleware
    from langchain.agents.middleware import TodoListMiddleware

    graph = _build(["todo", "delegation"], home)
    names = " ".join(graph.nodes)
    assert TodoListMiddleware.__name__ not in names
    assert SubAgentMiddleware.__name__ not in names


def test_env_var_disables_too(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGSTAGE_HERMES_AGENT_DISABLED_TOOLSETS", "file,terminal")
    cfg = HermesConfig.resolve()
    graph = create_hermes_agent(cfg, model=_Fake(messages=iter([AIMessage("ok")])), workspace=home)
    assert not _bound(graph) & {"write_file", "execute"}


def test_toml_disables_too(home: Path):
    (home / "langstage-hermes.toml").write_text('[agent]\ndisabled_toolsets = ["memory"]\n', encoding="utf-8")
    graph = create_hermes_agent(HermesConfig.resolve(), model=_Fake(messages=iter([AIMessage("ok")])), workspace=home)
    assert "memory" not in _bound(graph)


def test_review_fork_loses_disabled_tools_too(home: Path, monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}
    import langstage_hermes.agent as agent_mod

    real = agent_mod.build_review_subagent

    def spy(**kw: Any) -> Any:
        seen["tools"] = [t.name for t in kw["tools"]]
        return real(**kw)

    monkeypatch.setattr(agent_mod, "build_review_subagent", spy)
    _build(["skills"], home)
    assert "skill_manage" not in seen["tools"]
    assert "memory" in seen["tools"]


def test_unknown_toolset_is_reported(home: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture):
    with caplog.at_level(logging.WARNING, logger="langstage_hermes.agent"):
        graph = _build(["fiel"], home)
    assert _bound(graph) == ALL_BOUND
    assert "unknown toolset(s) ['fiel']" in caplog.text

    cfg = HermesConfig.resolve(overrides={"agent_disabled_toolsets": ["fiel", "memory"]})
    _slash_toolsets("", {"cfg": cfg})
    out = capsys.readouterr().out
    assert "unknown toolset(s) ['fiel'] ignored" in out
    assert "✗ memory" in out


def test_cron_strips_its_toolsets_and_the_users(home: Path, monkeypatch: pytest.MonkeyPatch):
    from langstage_hermes.cron import scheduler

    seen: dict[str, Any] = {}

    def fake_build(cfg: Any = None, **kw: Any) -> Any:
        seen["disabled"] = list(cfg.agent_disabled_toolsets)
        raise RuntimeError("stop after build")

    monkeypatch.setenv("LANGSTAGE_HERMES_AGENT_DISABLED_TOOLSETS", "file")
    monkeypatch.setattr("langstage_hermes.agent.create_hermes_agent", fake_build)
    scheduler.run_job({"id": "j1", "prompt": "hi"})
    assert {"cronjob", "messaging", "clarify", "file"} <= set(seen["disabled"])


def test_toolset_helpers():
    assert toolset_of("write_file") == "file"
    assert toolset_of("execute") == "terminal"
    assert toolset_of("web_search") == "web"  # the primary toolset, not the "search" composite
    assert toolset_of("my_own_tool") is None
    assert tool_allowed("my_own_tool", set())  # a caller's own tool is never filtered
    assert not tool_allowed("write_file", {"terminal"})
    assert unknown_toolsets(["file", "nope", "nope"]) == ["nope"]
