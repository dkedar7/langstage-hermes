"""A paused curator skips its scheduled pass (gh #204); `tools` lists the names the
model is given and stops advertising unbound tools (gh #205); `/cron`, `/curator` and
`/toolsets` act on their arguments instead of dropping them (gh #206)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from langstage_hermes.cli import (
    BUILTIN_SLASH_COMMANDS,
    _curator_state_get,
    _curator_state_save,
    _slash_cron,
    _slash_curator,
    _slash_reload,
    _slash_toolsets,
    cli,
)
from langstage_hermes.config import HermesConfig
from langstage_hermes.tools.toolsets import BOUND_TOOLS, IMPLEMENTED_TOOLSETS


class _Fake(FakeListChatModel):
    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        return self


class _Recorder(FakeListChatModel):
    """Records the tool names the agent binds to the model."""

    seen: ClassVar[list[str]] = []

    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        type(self).seen = sorted(getattr(t, "name", None) or t["name"] for t in tools)
        return self


@pytest.fixture
def home(tmp_hermes_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return tmp_hermes_home


def _state(**kw: Any) -> dict[str, Any]:
    return {"session_id": "s1", "cfg": HermesConfig.resolve(), "verbose": False, "yolo": False, **kw}


def _install_old_skill(home: Path) -> Path:
    src = home.parent / "src" / "my-notes"
    src.mkdir(parents=True)
    (src / "SKILL.md").write_text(
        "---\nname: my-notes\ndescription: Hand-written procedure; must not be archived while paused.\n---\n# My notes\n",
        encoding="utf-8",
    )
    res = CliRunner().invoke(cli, ["skills", "install", str(src)])
    assert res.exit_code == 0, res.output
    path = next(home.rglob("my-notes/SKILL.md"))
    old = time.time() - 120 * 86400
    os.utime(path, (old, old))
    return path


def _make_curator_due() -> None:
    s = _curator_state_get()
    s.update(last_run_at=time.time() - 200 * 3600, last_user_activity=time.time() - 10 * 3600)
    _curator_state_save(s)


# ── gh #204: curator pause ─────────────────────────────────────────────


@pytest.mark.parametrize("paused", [True, False])
def test_paused_curator_does_not_archive_on_a_chat_turn(home: Path, paused: bool):
    from langstage_hermes.agent import create_hermes_agent

    _install_old_skill(home)
    if paused:
        assert CliRunner().invoke(cli, ["curator", "pause"]).exit_code == 0
    _make_curator_due()

    g = create_hermes_agent(model=_Fake(responses=["hi there"]), aux_model=_Fake(responses=["ok"] * 5))
    g.invoke({"messages": [HumanMessage("hello")]}, {"configurable": {"thread_id": "t1"}})

    live = [p for p in home.rglob("my-notes*/SKILL.md") if "_archived" not in p.parts]
    archived = [p for p in home.rglob("my-notes*/SKILL.md") if "_archived" in p.parts]
    if paused:
        assert live and not archived
    else:  # the control: the same setup unpaused does run the pass
        assert archived and not live


# ── gh #205: advertised tools ──────────────────────────────────────────


def test_tools_lists_the_names_the_model_is_given(home: Path):
    from langstage_hermes.agent import create_hermes_agent

    m = _Recorder(responses=["ok"] * 5)
    g = create_hermes_agent(model=m, aux_model=m, workspace=os.getcwd())
    g.invoke({"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": "t"}})
    bound = set(_Recorder.seen)

    advertised = {name for ts in IMPLEMENTED_TOOLSETS for name in BOUND_TOOLS[ts]}
    # Every bound tool is advertised; the only advertised name not bound is the
    # sandbox-only shell, which `tools` says needs a sandbox backend.
    assert bound <= advertised
    assert advertised - bound == {"execute"}

    out = CliRunner().invoke(cli, ["tools", "--implemented-only"]).output
    for name in ("edit_file", "write_todos", "glob", "task", "execute"):
        assert f"· {name}" in out
    for hermes_name in ("· patch", "· search_files", "· todo\n", "· process"):
        assert hermes_name not in out
    assert "sandbox backend" in out


@pytest.mark.parametrize("toolset", ["clarify", "cronjob"])
def test_unbound_tools_are_not_advertised_as_implemented(home: Path, toolset: str):
    res = CliRunner().invoke(cli, ["tools", "--toolset", toolset])
    assert res.exit_code == 0
    assert "○" in res.output and "not bound into the agent" in res.output
    assert toolset not in IMPLEMENTED_TOOLSETS


# ── gh #206: /cron, /curator, /toolsets ───────────────────────────────


def test_slash_cron_create_creates_a_job(home: Path, capsys):
    from langstage_hermes.cron.jobs import list_jobs

    _slash_cron('create --prompt "summarise my inbox" --schedule "every 1h"', _state())
    assert [j["prompt"] for j in list_jobs()] == ["summarise my inbox"]
    _slash_cron("", _state())
    assert "Cron jobs (1)" in capsys.readouterr().out


def test_slash_cron_daemon_is_refused_not_run(home: Path, capsys):
    _slash_cron("daemon", _state())
    assert "would block the chat" in capsys.readouterr().out


def test_slash_cron_bad_arguments_print_an_error(home: Path, capsys):
    _slash_cron("frobnicate", _state())
    captured = capsys.readouterr()
    assert "No such command" in captured.out + captured.err
    _slash_cron('create --prompt "unterminated', _state())
    assert "could not parse arguments" in capsys.readouterr().out


def test_slash_curator_pause_and_resume(home: Path, capsys):
    _slash_curator("pause", _state())
    assert _curator_state_get().get("paused") is True
    _slash_curator("", _state())
    assert "paused=True" in capsys.readouterr().out
    _slash_curator("resume", _state())
    assert _curator_state_get().get("paused") is False


def _builtin_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr("langstage_hermes.agent._init_chat_model", lambda mid: _Fake(responses=["hello back"] * 5))
    from langstage_hermes.agent import create_hermes_agent

    cfg = HermesConfig.resolve()
    agent = create_hermes_agent(cfg)
    sid = agent.langstage_hermes_session_id
    agent.invoke({"messages": [HumanMessage("remember this")]}, config={"configurable": {"thread_id": sid}})
    return _state(session_id=sid, cfg=cfg, agent=agent, cli_overrides={}, _agui_agent=object())


def _tool_names(agent: Any) -> set[str]:
    return set(agent.nodes["tools"].bound.tools_by_name)


def test_slash_toolsets_disable_removes_the_tools_and_keeps_the_conversation(home: Path, monkeypatch, capsys):
    st = _builtin_state(monkeypatch)
    old = st["agent"]
    assert "write_todos" in _tool_names(old)

    _slash_toolsets("disable todo", st)

    out = capsys.readouterr().out
    assert "disabled todo for this session" in out
    assert st["agent"] is not old and "_agui_agent" not in st
    assert "write_todos" not in _tool_names(st["agent"])
    assert "✗ todo" in out
    thread = {"configurable": {"thread_id": st["session_id"]}}
    assert "remember this" in [m.content for m in st["agent"].get_state(thread).values["messages"]]

    # /reload keeps the session's choice; enable brings the tool back.
    _slash_reload("", st)
    assert "write_todos" not in _tool_names(st["agent"])
    _slash_toolsets("enable todo", st)
    assert "write_todos" in _tool_names(st["agent"])
    assert st["cfg"].agent_disabled_toolsets == []


def test_slash_toolsets_rejects_unknown_names_and_spec_graphs(home: Path, monkeypatch, capsys):
    st = _builtin_state(monkeypatch)
    agent = st["agent"]
    _slash_toolsets("disable terminl", st)
    assert "unknown toolset(s) ['terminl']" in capsys.readouterr().out
    assert st["agent"] is agent

    spec = _state(agent=object(), cli_overrides={})
    _slash_toolsets("disable todo", spec)
    assert "only changes the built-in agent" in capsys.readouterr().out

    _slash_toolsets("toggle todo", st)
    assert "Usage: /toolsets" in capsys.readouterr().out


def test_help_describes_what_the_commands_do():
    assert "Pass-through" not in BUILTIN_SLASH_COMMANDS["cron"]
    assert "disable|enable" in BUILTIN_SLASH_COMMANDS["toolsets"]
    assert "pause|resume|run" in BUILTIN_SLASH_COMMANDS["curator"]
