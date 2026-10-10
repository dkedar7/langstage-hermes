"""``python -m langstage_hermes.cron`` parses its arguments (gh #172).

It used to ignore ``sys.argv`` and go straight into ``run_forever()``, so ``--help``,
``--version`` and a typo'd flag all started a daemon that never returned. Now ``-h`` /
``--help`` and ``--version`` print and exit 0, any other argument is a usage error
(exit 64, core ADR 0007), and no arguments still starts the daemon.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from click.testing import CliRunner

from langstage_hermes import __version__
from langstage_hermes.cli import cli
from langstage_hermes.cron import __main__ as cron_main
from langstage_hermes.cron.scheduler import HermesCron


@pytest.fixture
def daemon_must_not_start(monkeypatch):
    def run_forever(self):
        pytest.fail("the cron daemon started")

    monkeypatch.setattr(HermesCron, "run_forever", run_forever)


@pytest.fixture
def daemon_runs(monkeypatch):
    """Record a daemon start instead of looping forever."""
    started = []

    def run_forever(self):
        started.append(self)

    monkeypatch.setattr(HermesCron, "run_forever", run_forever)
    return started


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_and_exits_0(flag, capsys, daemon_must_not_start):
    assert cron_main.main([flag]) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: python -m langstage_hermes.cron"), out
    flat = " ".join(out.split())  # argparse wraps to the terminal width
    assert "cron daemon forever" in flat, out
    assert "langstage-hermes cron --help" in flat, out  # points at the full CLI


def test_version_matches_the_main_cli_and_exits_0(capsys, daemon_must_not_start):
    assert cron_main.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert out == f"langstage-hermes {__version__}\n", out
    assert out == CliRunner().invoke(cli, ["--version"]).output


@pytest.mark.parametrize("arg", ["--totally-bogus-flag", "daemon"])
def test_unknown_argument_is_a_usage_error_64(arg, capsys, daemon_must_not_start):
    assert cron_main.main([arg]) == 64
    err = capsys.readouterr().err
    assert "usage: python -m langstage_hermes.cron" in err, err
    assert f"error: unrecognized arguments: {arg}" in err, err


def test_no_arguments_still_starts_the_daemon(capsys, daemon_runs):
    assert cron_main.main() == 0
    assert len(daemon_runs) == 1
    assert "langstage-hermes cron starting" in capsys.readouterr().out


def test_cron_daemon_command_does_not_parse_the_click_argv(monkeypatch, daemon_runs):
    # `langstage-hermes cron daemon` calls main() — sys.argv there is the click command
    # line (`cron daemon`), which must not reach the module parser as unknown args.
    monkeypatch.setattr(sys, "argv", ["langstage-hermes", "cron", "daemon"])
    r = CliRunner().invoke(cli, ["cron", "daemon"])
    assert r.exit_code == 0, r.output
    assert len(daemon_runs) == 1


@pytest.mark.parametrize(
    "flag, code, text",
    [
        ("--help", 0, "usage: python -m langstage_hermes.cron"),
        ("--version", 0, f"langstage-hermes {__version__}"),
        ("--totally-bogus-flag", 64, "unrecognized arguments: --totally-bogus-flag"),
    ],
)
def test_module_entry_end_to_end(flag, code, text):
    # The real `python -m` entry. A regression that starts the daemon hangs, so the
    # timeout turns it into a failure instead of a stuck suite.
    r = subprocess.run(
        [sys.executable, "-m", "langstage_hermes.cron", flag],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == code, (r.stdout, r.stderr)
    assert text in r.stdout + r.stderr, (r.stdout, r.stderr)
    assert "cron starting" not in r.stdout, r.stdout
