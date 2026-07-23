"""Tests for the unknown-key warning on ``langstage-hermes.toml`` (gh #84).

A syntactically valid but unrecognized TOML key is silently dropped — the CLI
already warns on a malformed file and on deprecated env vars, but a mistyped key
had no signal at all, made likely by the internally-inconsistent key naming
(``memory.memory_enabled`` next to ``memory.nudge_interval``). These tests pin:

* genuinely-unknown keys are detected with a "did you mean" hint,
* recognized keys (including the renamed/prefixed forms) are NOT flagged,
* section headers and free-form tables (``skills.platform_disabled``,
  ``configurable``) never produce false positives,
* the recognized set is built from the resolver's own map (can't drift),
* the note is warning-only (ASCII, stderr) and deduped.
"""

from __future__ import annotations

from pathlib import Path

from langstage_hermes.config import (
    HermesConfig,
    _format_unknown_key_note,
    _warn_unknown_hermes_keys,
    _warned_unknown_toml_keys,
    hermes_unknown_toml_keys,
)

_TOML = Path("langstage-hermes.toml")


def _keys(parsed):
    return {dotted for _p, dotted, _s in parsed}


def _suggestion_for(parsed, dotted):
    for _p, d, s in parsed:
        if d == dotted:
            return s
    return None


# ── detection ──────────────────────────────────────────────────────


def test_near_miss_memory_key_flagged_with_suggestion():
    data = {"memory": {"enabled": False}}
    parsed = hermes_unknown_toml_keys([(_TOML, data)])
    assert _keys(parsed) == {"memory.enabled"}
    assert _suggestion_for(parsed, "memory.enabled") == "memory_enabled"


def test_near_miss_model_key_flagged_with_suggestion():
    data = {"model": {"aux": "provider:model"}}
    parsed = hermes_unknown_toml_keys([(_TOML, data)])
    assert _keys(parsed) == {"model.aux"}
    assert _suggestion_for(parsed, "model.aux") == "aux_model"


def test_recognized_inconsistent_keys_not_flagged():
    """The accepted-but-inconsistent names must stay silent: prefixed
    (memory_enabled), unprefixed (nudge_interval), and renamed (aux_model)."""
    data = {
        "memory": {"memory_enabled": False, "nudge_interval": 5, "provider": "markdown"},
        "model": {"default": "anthropic:x", "aux_model": "anthropic:y"},
        "agent": {"spec": "mod:graph"},  # base HostConfig key
    }
    assert hermes_unknown_toml_keys([(_TOML, data)]) == []


def test_section_header_itself_never_flagged():
    """An empty recognized section table must not warn on the header."""
    data = {"memory": {}, "skills": {"platform_disabled": {}}}
    assert hermes_unknown_toml_keys([(_TOML, data)]) == []


def test_platform_disabled_subkeys_are_data_not_flagged():
    """[skills.platform_disabled] is keyed by arbitrary platform names —
    those are data, never unknown keys."""
    data = {"skills": {"platform_disabled": {"telegram": ["foo"], "cli": ["bar"]}}}
    assert hermes_unknown_toml_keys([(_TOML, data)]) == []


def test_configurable_passthrough_table_not_flagged():
    data = {"configurable": {"thread_id": "abc", "anything": 1}}
    assert hermes_unknown_toml_keys([(_TOML, data)]) == []


def test_unknown_top_level_and_unknown_section_both_flagged():
    data = {"totally_unknown": 1, "bogus": {"nested": 2}}
    assert _keys(hermes_unknown_toml_keys([(_TOML, data)])) == {"totally_unknown", "bogus.nested"}


# ── recognized set is drift-proof ──────────────────────────────────


def test_recognized_set_is_the_resolver_map():
    recognized = HermesConfig._recognized_toml_keys()
    assert recognized == set(HermesConfig._toml_map().values())
    # includes both a base HostConfig key and a Hermes-specific one
    assert "agent.spec" in recognized
    assert "memory.memory_enabled" in recognized
    assert "model.aux_model" in recognized


def test_platform_disabled_is_a_container():
    assert "skills.platform_disabled" in HermesConfig._toml_container_keys()
    assert "configurable" in HermesConfig._toml_container_keys()


# ── note formatting ────────────────────────────────────────────────


def test_note_format_matches_issue_shape_and_is_ascii():
    note = _format_unknown_key_note(_TOML, "memory.enabled", "memory_enabled")
    assert note == (
        "note: unknown config key '[memory] enabled' in langstage-hermes.toml (ignored). Did you mean 'memory_enabled'?"
    )
    note.encode("ascii")  # must not raise on a cp1252 console


def test_note_without_suggestion_omits_did_you_mean():
    note = _format_unknown_key_note(_TOML, "totally_unknown", None)
    assert note == "note: unknown config key 'totally_unknown' in langstage-hermes.toml (ignored)."


# ── emitter: stderr, dedupe, opt-out (pytest-suppressed by default) ─


def test_emitter_prints_note_when_not_under_pytest(monkeypatch, capsys):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("LANGSTAGE_SUPPRESS_UNKNOWN_KEY_NOTICE", raising=False)
    _warned_unknown_toml_keys.clear()
    path = Path("emit-test-a.toml")
    _warn_unknown_hermes_keys([(path, {"memory": {"enabled": False}})])
    err = capsys.readouterr().err
    assert "unknown config key '[memory] enabled'" in err
    # dedup: a second call for the same (file, key) is silent
    _warn_unknown_hermes_keys([(path, {"memory": {"enabled": False}})])
    assert capsys.readouterr().err == ""


def test_emitter_suppressed_by_optout(monkeypatch, capsys):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("LANGSTAGE_SUPPRESS_UNKNOWN_KEY_NOTICE", "1")
    _warned_unknown_toml_keys.clear()
    _warn_unknown_hermes_keys([(Path("emit-test-b.toml"), {"memory": {"enabled": False}})])
    assert capsys.readouterr().err == ""


def test_emitter_suppressed_under_pytest(capsys):
    """PYTEST_CURRENT_TEST is set during a test run — the note stays quiet so
    it can't pollute this suite or other repos'."""
    _warned_unknown_toml_keys.clear()
    _warn_unknown_hermes_keys([(Path("emit-test-c.toml"), {"memory": {"enabled": False}})])
    assert capsys.readouterr().err == ""
