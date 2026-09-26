"""The LangStage family exit codes (core ADR 0007).

0 success / 1 failure / 2 paused on a HITL interrupt / 64 usage error. hermes used to
exit 2 for every failed readiness check (``verify``, ``doctor``, the ``chat`` key
preflight), which the family reserves for "paused", and click's own usage errors
(also 2) collided the same way.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes import cli as cli_mod
from langstage_hermes.cli import cli


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "LANGSTAGE_HERMES_HOME",
        "DEEPAGENT_HERMES_HOME",
        "LANGSTAGE_HERMES_MODEL_DEFAULT",
        "DEEPAGENT_HERMES_MODEL_DEFAULT",
        "LANGSTAGE_HERMES_MODEL_AUX",
        "DEEPAGENT_HERMES_MODEL_AUX",
        "LANGSTAGE_AGENT_SPEC",
        "DEEPAGENT_AGENT_SPEC",
    ):
        monkeypatch.delenv(k, raising=False)


def test_constants_are_the_family_scheme():
    assert (cli_mod.EXIT_OK, cli_mod.EXIT_FAIL, cli_mod.EXIT_PAUSED, cli_mod.EXIT_USAGE) == (0, 1, 2, 64)


# ── 0 ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("argv", [["--help"], ["--version"], ["verify", "--help"], []])
def test_success_is_0(argv):
    assert CliRunner().invoke(cli, argv).exit_code == 0


# ── 64: usage errors (click's default is 2) ───────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["no-such-command"],
        ["--bogus"],
        ["verify", "--bogus"],
        ["--json"],  # --json without --show-config
        ["search", "q", "--window", "abc"],  # bad type
        ["search", "q", "--session", "s1"],  # --session without --around
        ["cron", "create", "--schedule", "every 2h"],  # missing required --prompt
        ["skills", "remove"],  # missing argument
        ["skills", "no-such-subcommand"],  # nested group
    ],
)
def test_usage_errors_exit_64(argv, monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    r = CliRunner().invoke(cli, argv)
    assert r.exit_code == 64, r.output


def test_cron_create_invalid_schedule_is_usage_64(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    r = CliRunner().invoke(cli, ["cron", "create", "--prompt", "x", "--schedule", "not-a-schedule"])
    assert r.exit_code == 64, r.output


def test_console_script_usage_error_exits_64():
    # Through main(), the real console-script entry point.
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.argv=['langstage-hermes','--bogus']; from langstage_hermes.cli import main; main()",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 64, r.stderr


# ── 1: failures (hermes used 2) ───────────────────────────────────────


def test_verify_keyless_fails_1(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    r = CliRunner().invoke(cli, ["verify"])
    assert r.exit_code == 1, r.output


def test_verify_build_failure_is_1(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")

    def boom(*a, **kw):
        raise RuntimeError("simulated build failure")

    monkeypatch.setattr("langstage_hermes.create_hermes_agent", boom)
    r = CliRunner().invoke(cli, ["verify"])
    assert r.exit_code == 1, r.output


def test_doctor_not_ready_is_1(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **kw: object())
    assert CliRunner().invoke(cli, ["doctor"]).exit_code == 1
    assert CliRunner().invoke(cli, ["doctor", "--json"]).exit_code == 1


def test_chat_keyless_preflight_is_1(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    r = CliRunner().invoke(cli, ["chat"], input="hi\n/quit\n")
    assert r.exit_code == 1, r.output


def test_chat_bad_agent_spec_is_1(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    r = CliRunner().invoke(cli, ["chat", "--agent", "/nope/x.py:graph"], input="/quit\n")
    assert r.exit_code == 1, r.output


def test_skills_install_without_skill_md_is_1(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    r = CliRunner().invoke(cli, ["skills", "install", str(empty)])
    assert r.exit_code == 1, r.output


def test_cron_daemon_start_failure_is_1(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    from langstage_hermes.cron import __main__ as cron_main
    from langstage_hermes.cron.scheduler import HermesCron

    def locked(self):
        raise RuntimeError("another daemon holds the lock")

    monkeypatch.setattr(HermesCron, "run_forever", locked)
    assert cron_main.main() == 1
    assert CliRunner().invoke(cli, ["cron", "daemon"]).exit_code == 1
