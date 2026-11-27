"""gh #240 — Hermes' own error lines name an unset provider key.

With no key, langchain-anthropic fails with ``AnthropicConnectionError:
Connection error.``, which reads like a network problem. langstage-core 1.0.69
appends ``(hint: ANTHROPIC_API_KEY is not set)`` on the lines core formats, but
Hermes formats its own for cron results, ``verify`` and chat's stream failures,
so they printed the bare error. They now go through
:mod:`langstage_hermes.keyhint`. Only the variable's name is printed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

HINT = "(hint: ANTHROPIC_API_KEY is not set)"
SECRET = "sk-ant-not-a-real-key-240"

pytest.importorskip("langstage_core.keyhint")


class AnthropicConnectionError(Exception):
    """Shaped like langchain-anthropic's wrapper: the class name names the provider."""


class _FailingAgent:
    def invoke(self, *args, **kwargs):
        raise AnthropicConnectionError("Connection error.")

    def stream(self, *args, **kwargs):
        raise AnthropicConnectionError("Connection error.")


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "LANGSTAGE_HERMES_MODEL_DEFAULT",
        "DEEPAGENT_HERMES_MODEL_DEFAULT",
        "LANGSTAGE_HERMES_MODEL_AUX",
        "DEEPAGENT_HERMES_MODEL_AUX",
        "LANGSTAGE_HERMES_HOME",
        "DEEPAGENT_HERMES_HOME",
    ):
        monkeypatch.delenv(k, raising=False)


def _stub_agent(monkeypatch) -> None:
    import langstage_hermes
    import langstage_hermes.agent as agent_mod

    monkeypatch.setattr(langstage_hermes, "create_hermes_agent", lambda *a, **k: _FailingAgent())
    monkeypatch.setattr(agent_mod, "create_hermes_agent", lambda *a, **k: _FailingAgent())


def test_wrappers_add_the_hint_once_and_never_a_value(monkeypatch):
    from langstage_hermes.keyhint import describe_error, with_key_hint

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exc = AnthropicConnectionError("Connection error.")
    assert describe_error(exc) == f"AnthropicConnectionError: Connection error. {HINT}"
    assert with_key_hint("Connection error.", exc) == f"Connection error. {HINT}"
    once = with_key_hint(describe_error(exc))
    assert once.count("hint:") == 1

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    assert describe_error(exc) == "AnthropicConnectionError: Connection error."
    assert SECRET not in describe_error(exc)


def test_wrappers_fall_back_without_core_keyhint(monkeypatch):
    """An older core (no langstage_core.keyhint) gets the plain text, no crash."""
    import sys

    from langstage_hermes.keyhint import describe_error, with_key_hint

    monkeypatch.setitem(sys.modules, "langstage_core.keyhint", None)  # import -> ImportError
    exc = AnthropicConnectionError("Connection error.")
    assert describe_error(exc) == "AnthropicConnectionError: Connection error."
    assert with_key_hint("x", exc) == "x"


def test_cron_job_error_names_the_unset_key(monkeypatch, tmp_hermes_home):
    """The issue's example: ``[agent invoke failed: Connection error.]``."""
    from langstage_hermes.cron.scheduler import _build_cron_response

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _stub_agent(monkeypatch)
    ok, text = _build_cron_response({"id": "j1"}, prompt="hi")
    assert ok is False
    assert text == f"[agent invoke failed: Connection error. {HINT}]"


def test_cron_failure_log_line_names_the_unset_key(monkeypatch, tmp_hermes_home, caplog):
    """The stderr WARNING (what crontab mail shows) carries the hint too."""
    import logging

    from langstage_hermes.cron.scheduler import _build_cron_response

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LANGSTAGE_DEBUG", raising=False)
    _stub_agent(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="langstage_hermes.cron.scheduler"):
        _build_cron_response({"id": "j1"}, prompt="hi")
    assert f"Cron job j1: agent invoke failed: Connection error. {HINT}" in caplog.text


def test_cron_job_error_has_no_hint_when_the_key_is_set(monkeypatch, tmp_hermes_home):
    from langstage_hermes.cron.scheduler import _build_cron_response

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    _stub_agent(monkeypatch)
    ok, text = _build_cron_response({"id": "j1"}, prompt="hi")
    assert ok is False
    assert text == "[agent invoke failed: Connection error.]"
    assert SECRET not in text


@pytest.mark.parametrize("as_json", [False, True])
def test_verify_names_a_blank_key(monkeypatch, tmp_path, as_json):
    """A blank ``ANTHROPIC_API_KEY`` passes the preflight (it is set), then the turn
    fails as a connection error; the verify line now says which key to fix."""
    from langstage_hermes.cli import cli

    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
    _stub_agent(monkeypatch)
    args = ["verify", "--json"] if as_json else ["verify"]
    r = CliRunner().invoke(cli, args)
    assert r.exit_code == 1, r.output
    if as_json:
        payload = json.loads(r.output)
        detail = next(c["detail"] for c in payload["checks"] if c["name"] == "round_trip")
        assert detail == f"model invoke failed: AnthropicConnectionError: Connection error. {HINT}"
    else:
        assert f"model invoke failed: AnthropicConnectionError: Connection error. {HINT}" in r.output


def _chat_state() -> dict:
    from types import SimpleNamespace

    return {"session_id": "s", "_agui_agent": object(), "cfg": SimpleNamespace(agent_max_iterations=5)}


def test_chat_stream_failure_names_the_unset_key(monkeypatch, capsys):
    """An exception escaping chat's stream loop gets the hint too."""
    from langstage_hermes import agui_stream
    from langstage_hermes.cli import _run_agent_turn_agui

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def _boom(*a, **k):
        raise AnthropicConnectionError("Connection error.")

    monkeypatch.setattr(agui_stream, "stream_frames_sync", _boom)
    _run_agent_turn_agui(object(), "hi", _chat_state())
    assert f"Agent stream failed: Connection error. {HINT}" in capsys.readouterr().out


def test_chat_error_frame_gets_the_hint_once(monkeypatch, capsys):
    """Core >= 1.0.69 already hints its error frames; chat must not double it."""
    from langstage_hermes import agui_stream
    from langstage_hermes.cli import _run_agent_turn_agui

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frames = [{"type": "error", "error": f"AnthropicConnectionError: Connection error. {HINT}"}]
    monkeypatch.setattr(agui_stream, "stream_frames_sync", lambda *a, **k: iter(frames))
    _run_agent_turn_agui(object(), "hi", _chat_state())
    out = capsys.readouterr().out
    assert f"Agent stream failed: AnthropicConnectionError: Connection error. {HINT}" in out
    assert out.count("hint:") == 1
