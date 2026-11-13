"""gh #220: `langstage-hermes tools` marks toolsets the resolved config turns off,
so it agrees with what the model is actually given."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langstage_hermes.cli import _slash_tools, cli
from langstage_hermes.config import HermesConfig
from langstage_hermes.tools.toolsets import IMPLEMENTED_TOOLSETS, bound_tools_for, config_off_reason


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    for var in (
        "LANGSTAGE_HERMES_AGENT_DISABLED_TOOLSETS",
        "DEEPAGENT_HERMES_AGENT_DISABLED_TOOLSETS",
        "LANGSTAGE_HERMES_CRON_AGENT_TOOL",
        "DEEPAGENT_HERMES_CRON_AGENT_TOOL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _toml(ws: Path, text: str) -> None:
    (ws / "langstage-hermes.toml").write_text(text, encoding="utf-8")


def _header(output: str, ts: str) -> str:
    m = re.search(rf"^\s+(\S)\s+{re.escape(ts)}\s+\d+ tool\(s\)(.*)$", output, re.M)
    assert m, f"no header for {ts!r} in:\n{output}"
    return m.group(1) + m.group(2)


def test_disabled_toolsets_are_marked_off(ws: Path) -> None:
    _toml(ws, '[agent]\ndisabled_toolsets = ["file", "memory"]\n')
    out = CliRunner().invoke(cli, ["tools"]).output
    assert _header(out, "file").startswith("✗")
    assert "disabled by agent.disabled_toolsets" in _header(out, "file")
    assert _header(out, "memory").startswith("✗")
    assert _header(out, "skills").startswith("●")
    assert "turned off by your config" in out


def test_filter_shows_off_reason(ws: Path) -> None:
    _toml(ws, '[agent]\ndisabled_toolsets = ["file"]\n')
    result = CliRunner().invoke(cli, ["tools", "--toolset", "file"])
    assert result.exit_code == 0
    assert _header(result.output, "file").startswith("✗")


def test_cronjob_follows_agent_tool_flag(ws: Path) -> None:
    out = CliRunner().invoke(cli, ["tools", "--toolset", "cronjob"]).output
    assert _header(out, "cronjob").startswith("✗")
    assert "agent_tool = false" in out

    _toml(ws, "[cron]\nagent_tool = true\n")
    out = CliRunner().invoke(cli, ["tools", "--toolset", "cronjob"]).output
    assert _header(out, "cronjob").startswith("●")

    _toml(ws, '[cron]\nagent_tool = true\n[agent]\ndisabled_toolsets = ["cronjob"]\n')
    out = CliRunner().invoke(cli, ["tools", "--toolset", "cronjob"]).output
    assert _header(out, "cronjob").startswith("✗")
    assert "disabled by agent.disabled_toolsets" in out


def test_default_config_only_cronjob_off(ws: Path) -> None:
    out = CliRunner().invoke(cli, ["tools", "--implemented-only"]).output
    off = {ts for ts in IMPLEMENTED_TOOLSETS if _header(out, ts).startswith("✗")}
    assert off == {"cronjob"}


def test_slash_tools_uses_session_config(capsys: pytest.CaptureFixture[str], ws: Path) -> None:
    cfg = HermesConfig.resolve(overrides={"agent_disabled_toolsets": ["memory"]})
    _slash_tools("", {"cfg": cfg})
    out = capsys.readouterr().out
    assert re.search(r"✗ memory\s+\(disabled by agent.disabled_toolsets\)", out)
    assert re.search(r"● skills", out)


class _Rec(GenericFakeChatModel):
    seen: list[str] = []

    def bind_tools(self, tools: Any, **kw: Any) -> Any:
        type(self).seen = sorted(t.name for t in tools)
        return self


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"agent_disabled_toolsets": ["file", "memory"]},
        {"cron_agent_tool": True},
        {"cron_agent_tool": True, "agent_disabled_toolsets": ["cronjob", "skills"]},
    ],
)
def test_listing_matches_bound_tools(ws: Path, overrides: dict[str, Any]) -> None:
    """For every implemented toolset, 'on' in `tools` iff its tools reach the model."""
    from langstage_hermes.agent import create_hermes_agent

    cfg = HermesConfig.resolve(overrides={"workspace_root": str(ws), **overrides})
    m = _Rec(messages=iter([AIMessage("ok")] * 5))
    g = create_hermes_agent(cfg, model=m, aux_model=m, workspace=str(ws))
    g.invoke({"messages": [{"role": "user", "content": "hi"}]}, config={"configurable": {"thread_id": "t"}})
    bound = set(_Rec.seen)
    for ts in IMPLEMENTED_TOOLSETS:
        names = set(bound_tools_for(ts))
        if not names or ts == "terminal":  # terminal needs a sandbox backend
            continue
        on = config_off_reason(ts, cfg) is None
        assert on == names.issubset(bound), (ts, on, names & bound)
