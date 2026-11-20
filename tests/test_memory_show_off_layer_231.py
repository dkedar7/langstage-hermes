"""`memory show` and `/memory` mark a layer that config switches off (gh #231).

Since 0.5.7 (gh #228) an off layer stays on disk but is left out of the agent's
prompt and refused by the memory tool. The two readers people use to answer
"what does the agent know about me?" printed it like a live layer. They still
print the file, but now say it's off, and ``--json`` carries ``enabled``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import _slash_memory, cli
from langstage_hermes.config import HermesConfig

USER_TEXT = "- Name: Sam; prefers terse answers\n"
MEM_TEXT = "- shipping the release\n"


@pytest.fixture
def home(tmp_hermes_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    mem = tmp_hermes_home / "memories"
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "USER.md").write_text(USER_TEXT, encoding="utf-8")
    (mem / "MEMORY.md").write_text(MEM_TEXT, encoding="utf-8")
    return tmp_hermes_home


def _section(output: str, label: str) -> str:
    """The heading line for ``label`` (USER.md / MEMORY.md)."""
    return next(line for line in output.splitlines() if line.startswith(label))


def test_toml_switch_marks_user_layer_off(home: Path, tmp_path: Path) -> None:
    (tmp_path / "langstage-hermes.toml").write_text("[memory]\nuser_profile_enabled = false\n", encoding="utf-8")
    res = CliRunner().invoke(cli, ["memory", "show"])
    assert res.exit_code == 0, res.output
    # The file is still shown (it's untouched on disk) ...
    assert "Name: Sam" in res.output
    # ... but marked off, naming the switch; the live layer isn't.
    assert "OFF (memory.user_profile_enabled = false)" in _section(res.output, "USER.md")
    assert "OFF" not in _section(res.output, "MEMORY.md")


def test_env_switch_marks_session_layer_off(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSTAGE_HERMES_MEMORY_ENABLED", "false")
    res = CliRunner().invoke(cli, ["memory", "show", "--session"])
    assert res.exit_code == 0, res.output
    assert "OFF (memory.memory_enabled = false)" in _section(res.output, "MEMORY.md")


@pytest.mark.parametrize(("mem", "usr"), [(True, True), (True, False), (False, True), (False, False)])
def test_json_enabled_follows_the_switches(home: Path, monkeypatch: pytest.MonkeyPatch, mem: bool, usr: bool) -> None:
    monkeypatch.setenv("LANGSTAGE_HERMES_MEMORY_ENABLED", str(mem).lower())
    monkeypatch.setenv("LANGSTAGE_HERMES_MEMORY_USER_PROFILE_ENABLED", str(usr).lower())
    res = CliRunner().invoke(cli, ["memory", "show", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["user"]["enabled"] is usr
    assert data["memory"]["enabled"] is mem
    # Content is still what's on disk either way.
    assert data["user"]["content"] == USER_TEXT


def test_empty_off_layer_says_so(tmp_hermes_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSTAGE_HERMES_MEMORY_USER_PROFILE_ENABLED", "false")
    res = CliRunner().invoke(cli, ["memory", "show", "--user"])
    assert res.exit_code == 0, res.output
    assert "no user memory yet" in res.output
    assert "memory.user_profile_enabled = false" in res.output


def test_slash_memory_marks_off_layer(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = HermesConfig.resolve(overrides={"memory_user_profile_enabled": False})
    _slash_memory("", {"cfg": cfg})
    out = capsys.readouterr().out
    assert "Name: Sam" in out
    assert "OFF (memory.user_profile_enabled = false)" in _section(out, "USER.md")
    assert "OFF" not in _section(out, "MEMORY.md")


def test_slash_memory_without_cfg_is_unchanged(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _slash_memory("", {})
    out = capsys.readouterr().out
    assert "OFF" not in out
    assert "Name: Sam" in out
