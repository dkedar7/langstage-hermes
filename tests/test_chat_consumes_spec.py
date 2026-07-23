"""Tests for the DEEPAGENT_AGENT_SPEC consumption path in the chat REPL.

The chat REPL used to always import ``langstage_hermes.agent`` and call
``create_hermes_agent``. The ``DEEPAGENT_AGENT_SPEC`` env var was read
for display only (``"advisory"`` annotation). After this change, chat
honours the spec — loading any importable ``module:object`` and using
its graph in place of the built-in factory.

These tests pin the resolution logic and the context-block annotations
without launching a real chat REPL (which would block on stdin).
"""

from __future__ import annotations

import pytest

from langstage_hermes.cli import (
    _instantiate_factory,
    _print_chat_context,
    _resolve_agent,
)

# ── _resolve_agent ─────────────────────────────────────────────────


class _FakeGraph:
    """Stand-in for a CompiledStateGraph. The only contract bit
    chat actually checks is ``.invoke``."""

    def __init__(self):
        self.invoked = False

    def invoke(self, *args, **kwargs):
        self.invoked = True
        return {"messages": []}


_FAKE_GRAPH_INSTANCE = _FakeGraph()


def _make_graph() -> _FakeGraph:
    """Factory the spec resolver should be able to call when the spec
    points at a callable (not the graph directly)."""
    return _FakeGraph()


def test_resolve_agent_falls_back_to_builtin_when_env_unset(monkeypatch):
    monkeypatch.delenv("DEEPAGENT_AGENT_SPEC", raising=False)
    target, source, err = _resolve_agent()
    assert source == "builtin"
    assert err is None
    # built-in path returns the ``create_hermes_agent`` factory
    assert callable(target)
    assert getattr(target, "__name__", "") == "create_hermes_agent"


def test_resolve_agent_loads_via_spec_when_env_set(monkeypatch):
    """A spec pointing at a module-level graph instance loads cleanly."""
    monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", f"{__name__}:_FAKE_GRAPH_INSTANCE")
    # `pytest` keeps the test module in sys.modules; the spec loader's
    # importlib path picks it up.
    target, source, err = _resolve_agent()
    assert source == "spec"
    assert err is None
    assert target is _FAKE_GRAPH_INSTANCE


def test_resolve_agent_loads_factory_via_spec(monkeypatch):
    """A spec pointing at a *callable* (factory) loads — chat calls
    it later, the resolver just verifies it's callable."""
    monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", f"{__name__}:_make_graph")
    target, source, err = _resolve_agent()
    assert source == "spec"
    assert err is None
    assert target is _make_graph  # callable, not instance — chat will call


def test_resolve_agent_explicit_arg_works_without_env(monkeypatch):
    """The -a/--agent CLI flag path: explicit spec, env unset."""
    monkeypatch.delenv("DEEPAGENT_AGENT_SPEC", raising=False)
    target, source, err = _resolve_agent(f"{__name__}:_FAKE_GRAPH_INSTANCE")
    assert source == "spec"
    assert err is None
    assert target is _FAKE_GRAPH_INSTANCE


def test_resolve_agent_explicit_arg_beats_env(monkeypatch):
    monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", "nope.does.not.exist:agent")
    target, source, err = _resolve_agent(f"{__name__}:_FAKE_GRAPH_INSTANCE")
    assert source == "spec"
    assert err is None
    assert target is _FAKE_GRAPH_INSTANCE


def test_resolve_agent_returns_error_for_unknown_module(monkeypatch):
    monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", "nope.does.not.exist:agent")
    target, source, err = _resolve_agent()
    assert source == "spec"
    assert target is None
    assert err and "nope.does.not.exist" in err


def test_resolve_agent_returns_error_for_missing_attr(monkeypatch):
    monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", f"{__name__}:NoSuchAttr")
    target, source, err = _resolve_agent()
    assert source == "spec"
    assert target is None
    assert err and "NoSuchAttr" in err


def test_resolve_agent_returns_error_for_non_invokable_target(monkeypatch):
    """Pointing the spec at a string / module / int — anything without
    .invoke and not callable — surfaces a clean error rather than
    crashing inside the REPL on the first turn."""
    monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", f"{__name__}:_NOT_A_GRAPH")
    target, source, err = _resolve_agent()
    assert source == "spec"
    assert target is None
    assert err is not None
    assert "invokable" in err.lower() or "callable" in err.lower()


_NOT_A_GRAPH = "just a string"  # for the test above


# ── _instantiate_factory ───────────────────────────────────────────


def test_instantiate_factory_passes_cfg_when_signature_accepts_it():
    sentinel = object()
    called_with = []

    def factory(cfg):
        called_with.append(cfg)
        return "graph"

    result = _instantiate_factory(factory, sentinel)
    assert result == "graph"
    assert called_with == [sentinel]


def test_instantiate_factory_falls_back_to_bare_call_when_factory_rejects_cfg():
    """User factories loaded via spec may take no args."""
    call_count = []

    def zero_arg_factory():
        call_count.append(1)
        return "ok"

    result = _instantiate_factory(zero_arg_factory, "irrelevant_cfg")
    assert result == "ok"
    assert call_count == [1]


def test_instantiate_factory_reraises_unrelated_typeerror():
    """A TypeError from inside the factory body is NOT a signature
    mismatch — propagate it so real bugs aren't swallowed."""

    def broken_factory(cfg):
        # Looks like a normal call, but the body fails for an
        # unrelated reason. Heuristic shouldn't catch this.
        return None.nonexistent_attr  # raises AttributeError, but...

    # Actually trigger a TypeError that doesn't match the heuristic:
    def broken_factory_typeerror(cfg):
        raise TypeError("something completely different")

    with pytest.raises(TypeError, match="completely different"):
        _instantiate_factory(broken_factory_typeerror, "cfg")


# ── _print_chat_context (spec annotations) ─────────────────────────


def test_context_drops_advisory_when_spec_consumed(monkeypatch, capsys):
    """When source=spec, the spec line should NOT carry the
    'advisory' annotation — the env var is actually doing work."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", "my.module:graph")

    cfg = type("C", (), {"model_default": "m", "hermes_home": "/h"})()
    _print_chat_context(
        cfg=cfg,
        workspace="/ws",
        session_id="s",
        agent=_FakeGraph(),
        agent_source="spec",
    )
    out = capsys.readouterr().out
    assert "my.module:graph" in out
    assert "(active)" in out
    assert "advisory" not in out.lower()


def test_context_keeps_advisory_when_spec_not_consumed(monkeypatch, capsys):
    """Defensive: if spec is set but agent_source is 'builtin'
    (e.g., the user invoked a path that bypasses _resolve_agent),
    we still surface the advisory framing — better than silently
    misleading."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", "my.module:graph")

    cfg = type("C", (), {"model_default": "m", "hermes_home": "/h"})()
    _print_chat_context(
        cfg=cfg,
        workspace="/ws",
        session_id="s",
        agent=lambda c: None,
        agent_source="builtin",
    )
    out = capsys.readouterr().out
    assert "my.module:graph" in out
    assert "advisory" in out.lower()


def test_context_model_line_annotates_when_spec_active(monkeypatch, capsys):
    """When spec is active, cfg.model_default is misleading (the spec
    graph owns its real model). We annotate to flag that."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    cfg = type("C", (), {"model_default": "anthropic:foo", "hermes_home": "/h"})()
    _print_chat_context(
        cfg=cfg,
        workspace="/ws",
        session_id="s",
        agent=_FakeGraph(),
        agent_source="spec",
    )
    out = capsys.readouterr().out
    assert "anthropic:foo" in out
    assert "spec graph owns" in out


def test_context_model_line_plain_when_spec_inactive(monkeypatch, capsys):
    """No annotation in built-in mode — the model line means exactly
    what it says (cfg.model_default IS what's used)."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    cfg = type("C", (), {"model_default": "anthropic:foo", "hermes_home": "/h"})()
    _print_chat_context(
        cfg=cfg,
        workspace="/ws",
        session_id="s",
        agent=lambda c: None,
        agent_source="builtin",
    )
    out = capsys.readouterr().out
    assert "anthropic:foo" in out
    assert "spec graph owns" not in out


# ── Integration: graph-with-.invoke is preferred over calling it ────


def test_chat_uses_spec_graph_as_is_when_invokable(monkeypatch):
    """When the spec target has .invoke, chat should treat it as the
    ready graph — not try to call it like a factory."""
    monkeypatch.setenv("DEEPAGENT_AGENT_SPEC", f"{__name__}:_FAKE_GRAPH_INSTANCE")
    target, source, err = _resolve_agent()
    assert source == "spec"
    assert err is None
    # _FAKE_GRAPH_INSTANCE has .invoke; the chat handler checks for
    # this and uses target directly.
    assert hasattr(target, "invoke")
    # And the factory path is NOT taken — calling it would AttributeError.


# ── [agent] spec from TOML (gh #85) ─────────────────────────────────

_SPEC = f"{__name__}:_FAKE_GRAPH_INSTANCE"


def _clear_spec_env(monkeypatch) -> None:
    monkeypatch.delenv("LANGSTAGE_AGENT_SPEC", raising=False)
    monkeypatch.delenv("DEEPAGENT_AGENT_SPEC", raising=False)


def test_resolve_agent_honors_config_spec_when_flag_and_env_unset(monkeypatch):
    """The regression: a resolved ``[agent] spec`` TOML value (passed as
    ``config_spec``, i.e. ``cfg.agent_spec``) is loaded by chat's resolver
    instead of silently falling back to the built-in hermes agent (gh #85)."""
    _clear_spec_env(monkeypatch)
    target, source, err = _resolve_agent(None, config_spec=_SPEC)
    assert source == "spec"
    assert err is None
    assert target is _FAKE_GRAPH_INSTANCE


def test_resolve_agent_env_beats_config_spec(monkeypatch):
    """env (LANGSTAGE_AGENT_SPEC) still wins over the TOML config_spec."""
    _clear_spec_env(monkeypatch)
    monkeypatch.setenv("LANGSTAGE_AGENT_SPEC", _SPEC)
    target, source, err = _resolve_agent(None, config_spec="nope.does.not.exist:agent")
    assert source == "spec"
    assert err is None
    assert target is _FAKE_GRAPH_INSTANCE


def test_resolve_agent_flag_beats_config_spec(monkeypatch):
    """The -a/--agent flag still wins over the TOML config_spec."""
    _clear_spec_env(monkeypatch)
    target, source, err = _resolve_agent(_SPEC, config_spec="nope.does.not.exist:agent")
    assert source == "spec"
    assert err is None
    assert target is _FAKE_GRAPH_INSTANCE


def test_resolve_agent_ignores_config_spec_when_none(monkeypatch):
    """No flag, no env, no TOML → built-in factory (unchanged default)."""
    _clear_spec_env(monkeypatch)
    target, source, err = _resolve_agent(None, config_spec=None)
    assert source == "builtin"
    assert err is None
    assert getattr(target, "__name__", "") == "create_hermes_agent"


def test_chat_honors_toml_agent_spec_end_to_end(monkeypatch, tmp_path):
    """End-to-end: a project ``langstage-hermes.toml`` with ``[agent] spec``
    resolves onto ``cfg.agent_spec`` and is then loaded by ``_resolve_agent``
    exactly as chat wires it — proving the file value is honored (gh #85)."""
    from langstage_hermes.config import HermesConfig

    _clear_spec_env(monkeypatch)
    # Isolate HERMES_HOME so no real global config.toml leaks in.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(home))

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "langstage-hermes.toml").write_text(f'[agent]\nspec = "{_SPEC}"\n', encoding="utf-8")

    cfg = HermesConfig.resolve(toml_start=proj)
    # --show-config already resolves this; the bug was chat ignoring it.
    assert cfg.agent_spec == _SPEC
    assert cfg.sources.get("agent_spec", "").startswith("toml")

    # chat's wiring: _resolve_agent(cli_flag, config_spec=cfg.agent_spec)
    target, source, err = _resolve_agent(None, config_spec=cfg.agent_spec)
    assert source == "spec"
    assert err is None
    assert target is _FAKE_GRAPH_INSTANCE


def test_context_surfaces_toml_spec_as_active(monkeypatch, capsys):
    """The chat context block shows a TOML-sourced spec as ``(active)`` even
    though it never touched an env var (gh #85 — the diagnostic must not go
    dark just because the spec came from a file)."""
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    _clear_spec_env(monkeypatch)
    cfg = type("C", (), {"model_default": "m", "hermes_home": "/h"})()
    _print_chat_context(
        cfg=cfg,
        workspace="/ws",
        session_id="s",
        agent=_FakeGraph(),
        agent_source="spec",
        spec=_SPEC,
    )
    out = capsys.readouterr().out
    assert _SPEC in out
    assert "(active)" in out
    assert "advisory" not in out.lower()
