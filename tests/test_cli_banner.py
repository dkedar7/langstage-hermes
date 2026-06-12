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

from langstage_hermes.cli import _BANNER_ASCII, _print_banner, cli


def test_banner_constant_shape():
    """4 lines tall, all under 30 columns wide — sanity bounds for the
    pre-rendered FIGlet so a slip in `cli.py` doesn't ship a 12-line
    monster that takes over a terminal."""
    lines = _BANNER_ASCII.splitlines()
    assert len(lines) == 4
    assert all(len(line) <= 30 for line in lines), [len(line) for line in lines]


def test_print_banner_silent_on_non_tty(capsys):
    """Piped invocations (CI, `| grep`, test harnesses) must not see
    the banner — keeps `langstage-hermes --version | head -1` clean."""
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
    from langstage_hermes import __version__

    assert __version__ in out


def test_help_does_not_explode_on_banner(monkeypatch):
    """Bare invocation prints the help. Make sure the banner-print path
    doesn't raise even when stdout is a TTY-claiming non-TTY (the env
    inside `CliRunner` — Click captures stdout in a non-TTY buffer)."""
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    # Banner is silent on non-TTY; help text always shows.
    assert "langstage-hermes" in result.output.lower()


# ── _print_chat_context ─────────────────────────────────────────────


def test_chat_context_silent_on_non_tty(capsys):
    """Same TTY gate as the banner — piped chat invocations don't leak
    config to stdout (though `chat` itself is interactive, so this is
    a defence-in-depth check, not a primary use case)."""
    from langstage_hermes.cli import _print_chat_context

    _print_chat_context(
        cfg=type("C", (), {"model_default": "anthropic:foo", "hermes_home": "/tmp/h"})(),
        workspace="/tmp/ws",
        session_id="sess-abc",
        agent=lambda x: x,
    )
    assert capsys.readouterr().out == ""


def test_chat_context_renders_core_fields_on_tty(monkeypatch, capsys):
    from langstage_hermes.cli import _print_chat_context

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.delenv("DEEPAGENT_AGENT_SPEC", raising=False)

    def my_factory(cfg):  # so we can read its qualname back out
        return None

    cfg = type("C", (), {"model_default": "anthropic:claude-sonnet-4-5", "hermes_home": "/h/dah"})()
    _print_chat_context(
        cfg=cfg,
        workspace="/path/to/ws",
        session_id="sess-7094182fad7f",
        agent=my_factory,
    )
    out = capsys.readouterr().out
    # Core fields all present.
    assert "agent" in out and "my_factory" in out
    assert "anthropic:claude-sonnet-4-5" in out
    assert "/path/to/ws" in out
    assert "/h/dah" in out
    assert "sess-7094182fad7f" in out
    # No spec line when env var is unset.
    assert "spec" not in out.lower()


def test_chat_context_shows_advisory_spec_when_set(monkeypatch, capsys):
    """If DEEPAGENT_AGENT_SPEC is in env, surface it with an annotation
    that it's not consumed by this CLI — keeps the user from being
    confused when the spec they set "doesn't work" here."""
    from langstage_hermes.cli import _print_chat_context

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", "custom.module:graph")

    cfg = type("C", (), {"model_default": "m", "hermes_home": "/h"})()
    _print_chat_context(cfg=cfg, workspace="/ws", session_id="s", agent=lambda c: None)
    out = capsys.readouterr().out
    assert "custom.module:graph" in out
    assert "advisory" in out.lower()


def test_chat_context_shortens_long_paths(monkeypatch, capsys):
    """Long absolute paths get middle-ellipsised so the block fits in
    a normal terminal width."""
    from langstage_hermes.cli import _print_chat_context, _shorten_path

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    long_ws = "/very/long/absolute/path/" + "x" * 200 + "/workspace"
    cfg = type("C", (), {"model_default": "m", "hermes_home": long_ws})()
    _print_chat_context(cfg=cfg, workspace=long_ws, session_id="s", agent=lambda c: None)
    out = capsys.readouterr().out
    # Ellipsis marker present (paths got shortened).
    assert "..." in out
    # Direct shortener check too.
    assert _shorten_path(long_ws) != long_ws
    assert "..." in _shorten_path(long_ws)
    assert _shorten_path("/short/path") == "/short/path"
