"""Sanity checks for the CLI banner.

The banner itself is hard-coded ASCII art and a `_print_banner` helper
that gates on `sys.stdout.isatty()`. These tests pin the contract so a
future refactor can't accidentally drop the banner from interactive
sessions, blast it through CI / piped output, or quietly hard-code a
stale version.
"""

from __future__ import annotations

import io

from click.testing import CliRunner

from deepagent_hermes.cli import _BANNER_ASCII, _print_banner, cli


def test_banner_constant_shape():
    """4 lines tall, all under 30 columns wide — sanity bounds for the
    pre-rendered FIGlet so a slip in `cli.py` doesn't ship a 12-line
    monster that takes over a terminal."""
    lines = _BANNER_ASCII.splitlines()
    assert len(lines) == 4
    assert all(len(line) <= 30 for line in lines), [len(line) for line in lines]


def test_print_banner_silent_on_non_tty(capsys):
    """Piped invocations (CI, `| grep`, test harnesses) must not see
    the banner — keeps `deepagent-hermes --version | head -1` clean."""
    _print_banner()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_banner_prints_on_tty(monkeypatch, capsys):
    """When stdout claims TTY, banner + tagline both print."""

    class _FakeTTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    # Direct stub on sys.stdout's isatty so the helper sees a TTY.
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    _print_banner(tagline="testing")
    captured = capsys.readouterr()
    # ANSI-stripped substring checks — color codes shouldn't be the
    # contract.
    out = captured.out
    assert "hermes" in out.lower() or "_||_" in out, "ascii body missing"
    assert "testing" in out, "custom tagline missing"
    # Version line must be live (read from __version__), not hard-coded.
    from deepagent_hermes import __version__

    assert __version__ in out


def test_help_does_not_explode_on_banner(monkeypatch):
    """Bare invocation prints the help. Make sure the banner-print path
    doesn't raise even when stdout is a TTY-claiming non-TTY (the env
    inside `CliRunner` — Click captures stdout in a non-TTY buffer)."""
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    # Banner is silent on non-TTY; help text always shows.
    assert "deepagent-hermes" in result.output.lower()
