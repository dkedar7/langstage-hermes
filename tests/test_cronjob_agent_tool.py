"""The opt-in, approval-gated `cronjob` agent tool (`[cron] agent_tool = true`).

Off by default, the agent has no `cronjob` tool. When it's on, `list` / `show` /
`pause` / `delete` run straight away, while `create` / `resume` / `run-now` (the
calls that start or restart unattended runs) pause for a person's approval, on
every stage that hosts the graph. A cron run's own agent never gets the tool.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from langstage_hermes.config import HermesConfig
from langstage_hermes.cron import jobs as cron_jobs
from langstage_hermes.cron.tool import cronjob_interrupt_config, needs_approval


class _Scripted(GenericFakeChatModel):
    """Main model: replays scripted turns and records the tools it was given."""

    seen: ClassVar[list[str]] = []

    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        type(self).seen = sorted(getattr(t, "name", None) or t["name"] for t in tools)
        return self


class _Aux(FakeListChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


@pytest.fixture
def home(tmp_hermes_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.delenv("LANGSTAGE_HERMES_INTERRUPT_ON", raising=False)
    monkeypatch.delenv("DEEPAGENT_HERMES_INTERRUPT_ON", raising=False)
    return tmp_hermes_home


def _call(action: str, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "cronjob", "args": {"action": action, **args}, "id": f"c-{action}"}])


def _agent(*turns: AIMessage, cron_agent_tool: bool = True, **overrides: Any) -> Any:
    from langstage_hermes.agent import create_hermes_agent

    cfg = HermesConfig.resolve(overrides={"cron_agent_tool": cron_agent_tool, **overrides})
    model = _Scripted(disable_streaming=True, messages=iter([*turns, *[AIMessage("done")] * 4]))
    return create_hermes_agent(cfg, model=model, aux_model=_Aux(responses=["ok"] * 10), workspace=os.getcwd())


def _bound(graph: Any) -> set[str]:
    return set(graph.nodes["tools"].bound.tools_by_name)


def _thread(graph: Any) -> dict[str, Any]:
    return {"configurable": {"thread_id": graph.langstage_hermes_session_id}}


CREATE = {"schedule": "0 9 * * *", "prompt": "summarise the repo", "name": "digest"}


# ── the flag ────────────────────────────────────────────────────────────


def test_off_by_default(home: Path):
    assert HermesConfig.resolve().cron_agent_tool is False
    g = _agent(cron_agent_tool=False)
    assert "cronjob" not in _bound(g)
    g.invoke({"messages": [HumanMessage("hi")]}, _thread(g))
    assert "cronjob" not in _Scripted.seen


def test_flag_on_binds_the_tool(home: Path):
    g = _agent()
    assert "cronjob" in _bound(g)
    g.invoke({"messages": [HumanMessage("hi")]}, _thread(g))
    assert "cronjob" in _Scripted.seen


@pytest.mark.parametrize(
    ("source", "value"),
    [("env", "true"), ("toml", "true")],
)
def test_flag_resolves_from_env_and_toml(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, value: str):
    if source == "env":
        monkeypatch.setenv("DEEPAGENT_HERMES_CRON_AGENT_TOOL", value)
    else:
        (Path.cwd() / "langstage-hermes.toml").write_text(f"[cron]\nagent_tool = {value}\n", encoding="utf-8")
    assert HermesConfig.resolve().cron_agent_tool is True


def test_a_disabled_cronjob_toolset_wins_over_the_flag(home: Path):
    g = _agent(agent_disabled_toolsets=["cronjob"])
    assert "cronjob" not in _bound(g)


# ── the gate ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("action", "gated"),
    [
        ("create", True),
        ("resume", True),
        ("run-now", True),
        ("run_now", True),
        ("trigger", True),
        (" Create ", True),
        ("something-new", True),  # fails closed
        ("", True),
        ("list", False),
        ("show", False),
        ("pause", False),
        ("delete", False),
    ],
)
def test_which_actions_need_approval(action: str, gated: bool):
    assert needs_approval({"action": action}) is gated


def test_create_pauses_for_approval_and_approve_creates_the_job(home: Path):
    g = _agent(_call("create", **CREATE))
    out = g.invoke({"messages": [HumanMessage("remind me every morning")]}, _thread(g))
    assert "__interrupt__" in out
    request = out["__interrupt__"][0].value["action_requests"][0]
    assert request["name"] == "cronjob"
    assert "schedule a new cron job [0 9 * * *]" in request["description"]
    assert "summarise the repo" in request["description"]
    assert cron_jobs.list_jobs() == []  # nothing scheduled while it waits

    g.invoke(Command(resume={"decisions": [{"type": "approve"}]}), _thread(g))
    jobs = cron_jobs.list_jobs()
    assert [j["name"] for j in jobs] == ["digest"]


def test_reject_creates_no_job(home: Path):
    g = _agent(_call("create", **CREATE))
    out = g.invoke({"messages": [HumanMessage("remind me every morning")]}, _thread(g))
    assert "__interrupt__" in out
    final = g.invoke(Command(resume={"decisions": [{"type": "reject", "message": "not now"}]}), _thread(g))
    assert cron_jobs.list_jobs() == []
    rejected = [m for m in final["messages"] if getattr(m, "type", "") == "tool"]
    assert rejected and "not now" in rejected[-1].content


def test_list_runs_without_approval(home: Path):
    cron_jobs.create_job(prompt="p", schedule="30m", name="existing")
    g = _agent(_call("list"))
    out = g.invoke({"messages": [HumanMessage("what's scheduled?")]}, _thread(g))
    assert "__interrupt__" not in out
    results = [m.content for m in out["messages"] if getattr(m, "type", "") == "tool"]
    assert any("existing" in r for r in results)


@pytest.mark.parametrize("action", ["resume", "run-now"])
def test_resume_and_run_now_pause_for_approval(home: Path, action: str):
    job = cron_jobs.create_job(prompt="p", schedule="30m", name="existing")
    cron_jobs.pause_job(job["id"])
    g = _agent(_call(action, id=job["id"]))
    out = g.invoke({"messages": [HumanMessage("go")]}, _thread(g))
    assert "__interrupt__" in out
    assert job["id"] in out["__interrupt__"][0].value["action_requests"][0]["description"]
    assert cron_jobs.get_job(job["id"])["enabled"] is False  # still paused while it waits


def test_interrupt_on_env_gates_every_cronjob_call(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANGSTAGE_HERMES_INTERRUPT_ON", "cronjob")
    g = _agent(_call("list"))
    out = g.invoke({"messages": [HumanMessage("what's scheduled?")]}, _thread(g))
    assert "__interrupt__" in out


def test_interrupt_config_shape():
    cfg = cronjob_interrupt_config()
    assert cfg["allowed_decisions"] == ["approve", "reject"]
    assert callable(cfg["when"]) and callable(cfg["description"])


# ── cron runs never get it ──────────────────────────────────────────────


def test_cron_runs_strip_the_tool_even_with_the_flag_on(home: Path, monkeypatch: pytest.MonkeyPatch):
    from langstage_hermes import agent as agent_mod
    from langstage_hermes.cron import scheduler

    assert "cronjob" in scheduler._CRON_ALWAYS_STRIPPED
    monkeypatch.setenv("DEEPAGENT_HERMES_CRON_AGENT_TOOL", "true")
    real = agent_mod.create_hermes_agent
    built: list[set[str]] = []

    def _spy(cfg: Any = None, **kw: Any) -> Any:
        assert cfg is not None and cfg.cron_agent_tool is True  # the flag reached the cron build
        g = real(cfg, model=_Scripted(messages=iter([AIMessage("done")] * 4)), aux_model=_Aux(responses=["ok"] * 10))
        built.append(_bound(g))
        return g

    monkeypatch.setattr(agent_mod, "create_hermes_agent", _spy)
    ok, _ = scheduler._build_cron_response({"id": "j1", "name": "j1"}, prompt="hi")
    assert ok
    assert built and "cronjob" not in built[0]


# ── the chat REPL can answer the pause ──────────────────────────────────


def _chat_state(graph: Any) -> dict[str, Any]:
    return {
        "session_id": graph.langstage_hermes_session_id,
        "cfg": graph.langstage_hermes_config,
        "verbose": False,
        "yolo": False,
    }


def _answer(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": text)


def test_chat_asks_and_approve_creates_the_job(home: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    from langstage_hermes.cli import _run_agent_turn_agui

    g = _agent(_call("create", **CREATE))
    _answer(monkeypatch, "y")
    _run_agent_turn_agui(g, "remind me every morning", _chat_state(g))
    out = capsys.readouterr().out
    assert "approval needed: cronjob" in out
    assert "schedule a new cron job [0 9 * * *]" in out
    assert "✓ approved" in out
    assert [j["name"] for j in cron_jobs.list_jobs()] == ["digest"]


@pytest.mark.parametrize("answer", ["", "n", "no", "not today"])
def test_chat_reject_creates_no_job(home: Path, monkeypatch: pytest.MonkeyPatch, capsys, answer: str):
    from langstage_hermes.cli import _run_agent_turn_agui

    g = _agent(_call("create", **CREATE))
    _answer(monkeypatch, answer)
    _run_agent_turn_agui(g, "remind me every morning", _chat_state(g))
    assert "✗ rejected" in capsys.readouterr().out
    assert cron_jobs.list_jobs() == []


def test_chat_with_no_stdin_stops_the_turn_without_answering(home: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    from langstage_hermes.cli import _run_agent_turn_agui

    def _closed(prompt: str = "") -> str:
        raise EOFError

    g = _agent(_call("create", **CREATE))
    monkeypatch.setattr("builtins.input", _closed)
    _run_agent_turn_agui(g, "remind me every morning", _chat_state(g))
    assert "paused for approval" in capsys.readouterr().out
    assert cron_jobs.list_jobs() == []
