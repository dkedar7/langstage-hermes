"""`chat` runs the same provider-aware API-key preflight verify/doctor do (gh #76).

`chat` — the headline Quick Start command — used to have NO key preflight: with a
missing/blank provider key the default ``anthropic:*`` path opened the REPL,
accepted a message, and only THEN leaked a raw ``TypeError`` mid-session, while
the ``openai:*`` path leaked a bare ``Missing credentials`` at build. Meanwhile
``verify``/``doctor`` already gave clean, actionable guidance. This pins that
``chat`` now gates BEFORE building the agent / entering the REPL, reusing the
same ``_preflight_model_key`` primitive ``verify`` uses — and that a valid key
(or a BYO spec graph) still proceeds untouched.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import _preflight_model_key, cli


def _isolate(monkeypatch, tmp_path):
    """No stray key/env/toml from the host — a fresh, keyless environment."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)  # no stray langstage.toml from the repo
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "LANGSTAGE_HERMES_HOME",
        "DEEPAGENT_HERMES_HOME",
        "LANGSTAGE_HERMES_MODEL_DEFAULT",
        "DEEPAGENT_HERMES_MODEL_DEFAULT",
        "LANGSTAGE_AGENT_SPEC",
        "DEEPAGENT_AGENT_SPEC",
    ):
        monkeypatch.delenv(k, raising=False)


class _FakeGraph:
    """Stand-in graph; the only contract chat checks is ``.invoke``."""

    def invoke(self, *args, **kwargs):
        return {"messages": []}


# A module-level graph instance the spec loader (module:attr) can resolve.
_FAKE_GRAPH_INSTANCE = _FakeGraph()


# ── chat integration: the gate fires BEFORE the REPL, cleanly ──────────────


def test_chat_missing_anthropic_key_exits_clean_no_raw_traceback(monkeypatch, tmp_path):
    """The headline repro: default anthropic:* model, no key. Before the fix the
    REPL opened, accepted 'hi', then leaked a raw TypeError. Now it exits 2 with
    verify's clean guidance BEFORE any input is consumed."""
    _isolate(monkeypatch, tmp_path)  # default model is anthropic:*

    r = CliRunner().invoke(cli, ["chat"], input="hi\n/quit\n")

    assert r.exit_code == 2, r.output
    assert "model is anthropic:* but ANTHROPIC_API_KEY not set" in r.output
    # Points the user at the full preflight (chat-only pointer).
    assert "verify" in r.output
    # The bug's fingerprints must be gone — no raw provider internals leaked.
    assert "TypeError" not in r.output
    assert "Could not resolve authentication" not in r.output
    assert "Agent stream failed" not in r.output
    assert "Traceback" not in r.output


def test_chat_missing_openai_key_exits_clean(monkeypatch, tmp_path):
    """The openai:* variant: before the fix chat leaked a bare 'Missing
    credentials' at build; now it gives the same clean, provider-aware message
    and exits 2 BEFORE trying to build (so it doesn't even need the [openai]
    extra installed)."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")

    r = CliRunner().invoke(cli, ["chat"], input="hi\n/quit\n")

    assert r.exit_code == 2, r.output
    assert "model is openai:* but neither OPENAI_API_KEY nor OPENROUTER_API_KEY set" in r.output
    # No raw build error leaked.
    assert "Missing credentials" not in r.output
    assert "Failed to build agent" not in r.output


def test_chat_openrouter_key_satisfies_openai_model(monkeypatch, tmp_path):
    """OPENROUTER_API_KEY is an accepted credential for openai:* (mirrors
    verify) — so the gate must NOT fire when only it is set."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-dummy")
    built: dict[str, bool] = {}

    def _fake_factory(target, cfg):
        built["yes"] = True
        return _FakeGraph()

    monkeypatch.setattr("langstage_hermes.cli._instantiate_factory", _fake_factory)

    r = CliRunner().invoke(cli, ["chat"], input="/quit\n")

    assert r.exit_code == 0, r.output
    assert "neither OPENAI_API_KEY nor OPENROUTER_API_KEY set" not in r.output
    assert built.get("yes") is True  # gate passed → chat proceeded to build the agent


def test_chat_valid_anthropic_key_proceeds_to_repl(monkeypatch, tmp_path):
    """A valid key must NOT be blocked by the gate: chat gets past the preflight,
    builds the agent, and enters the REPL (which /quit then exits cleanly)."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    built: dict[str, bool] = {}

    def _fake_factory(target, cfg):
        # The preflight is what's under test; don't build/hit the real model.
        built["yes"] = True
        return _FakeGraph()

    monkeypatch.setattr("langstage_hermes.cli._instantiate_factory", _fake_factory)

    r = CliRunner().invoke(cli, ["chat"], input="/quit\n")

    assert r.exit_code == 0, r.output
    assert "ANTHROPIC_API_KEY not set" not in r.output
    assert built.get("yes") is True  # preflight passed → agent got built


def test_chat_spec_graph_bypasses_key_gate(monkeypatch, tmp_path):
    """A BYO spec graph owns its own model, so cfg.model_default doesn't describe
    it — the preflight must NOT second-guess it (avoids the #33 false-positive
    class). Even with no key, a spec graph reaches the REPL."""
    _isolate(monkeypatch, tmp_path)  # keyless
    monkeypatch.setenv("LANGSTAGE_AGENT_SPEC", f"{__name__}:_FAKE_GRAPH_INSTANCE")

    r = CliRunner().invoke(cli, ["chat"], input="/quit\n")

    assert r.exit_code == 0, r.output
    assert "ANTHROPIC_API_KEY not set" not in r.output


# ── _preflight_model_key unit contract (shared with verify) ─────────────────


def test_preflight_exits_2_when_anthropic_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        _preflight_model_key("anthropic:claude-sonnet-4-6")
    assert exc.value.code == 2


def test_preflight_exits_2_when_openai_key_missing(monkeypatch):
    for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit) as exc:
        _preflight_model_key("openai:openai/gpt-4o-mini")
    assert exc.value.code == 2


def test_preflight_passes_when_anthropic_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    assert _preflight_model_key("anthropic:claude-sonnet-4-6") is None


def test_preflight_ignores_unknown_provider(monkeypatch):
    """A custom/local provider prefix asserts no specific key — don't block it."""
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert _preflight_model_key("ollama:llama3") is None
