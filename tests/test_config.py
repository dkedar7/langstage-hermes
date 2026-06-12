"""Tests for ``langstage_hermes.config.HermesConfig``.

Verify SPEC §2 defaults, env-var precedence, and that ``describe()`` reports
the resolution source for each field.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from langstage_hermes.config import HermesConfig, hermes_home

# ── defaults (SPEC §2 verbatim) ──────────────────────────────────────


def _strip_env(monkeypatch):
    """Remove any DEEPAGENT_* / DEEPAGENT_HERMES_* env vars so defaults stand."""
    for var in list(os.environ.keys()):
        if var.startswith("DEEPAGENT_"):
            monkeypatch.delenv(var, raising=False)


def test_defaults_match_spec_model(monkeypatch):
    """[model] block defaults."""
    _strip_env(monkeypatch)
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.model_default == "anthropic:claude-sonnet-4-6"
    assert cfg.model_provider == "auto"
    assert cfg.model_context_length is None
    assert cfg.model_max_tokens is None
    assert cfg.model_aux == "anthropic:claude-haiku-4-5-20251001"


def test_defaults_match_spec_agent(monkeypatch):
    _strip_env(monkeypatch)
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.agent_api_max_retries == 3
    assert cfg.agent_max_iterations == 90
    assert cfg.agent_delegation_max_iterations == 50
    assert cfg.agent_task_completion_guidance is True
    assert cfg.agent_environment_probe is True
    assert cfg.agent_tool_use_enforcement == "auto"
    assert cfg.agent_disabled_toolsets == []


def test_defaults_match_spec_memory(monkeypatch):
    _strip_env(monkeypatch)
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.memory_enabled is True
    assert cfg.memory_user_profile_enabled is True
    assert cfg.memory_nudge_interval == 10
    assert cfg.memory_char_limit == 2200
    assert cfg.memory_user_char_limit == 1375
    assert cfg.memory_provider == ""


def test_defaults_match_spec_skills(monkeypatch):
    _strip_env(monkeypatch)
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.skills_creation_nudge_interval == 10
    assert cfg.skills_external_dirs == []
    assert cfg.skills_disabled == []
    assert cfg.skills_platform_disabled == {}


def test_defaults_match_spec_compression(monkeypatch):
    _strip_env(monkeypatch)
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.compression_enabled is True
    assert cfg.compression_threshold == pytest.approx(0.50)
    assert cfg.compression_target_ratio == pytest.approx(0.20)
    assert cfg.compression_protect_first_n == 3
    assert cfg.compression_protect_last_n == 20
    assert cfg.compression_abort_on_summary_failure is False


def test_defaults_match_spec_delegation(monkeypatch):
    _strip_env(monkeypatch)
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.delegation_max_concurrent_children == 4
    assert cfg.delegation_max_spawn_depth == 3
    assert cfg.delegation_max_iterations == 50


def test_defaults_match_spec_curator(monkeypatch):
    _strip_env(monkeypatch)
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.curator_enabled is True
    assert cfg.curator_interval_hours == 168
    assert cfg.curator_min_idle_hours == 2
    assert cfg.curator_stale_after_days == 30
    assert cfg.curator_archive_after_days == 90
    assert cfg.curator_prune_builtins is True


def test_defaults_match_spec_cron_and_plugins(monkeypatch):
    _strip_env(monkeypatch)
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.cron_tick_seconds == 60
    assert cfg.plugins_enabled == []
    assert cfg.plugins_disabled == []


# ── env-var resolution ───────────────────────────────────────────────


def test_env_override_skills_nudge_interval(monkeypatch):
    """The flagship env-precedence test from the task spec."""
    monkeypatch.setenv("DEEPAGENT_HERMES_SKILLS_CREATION_NUDGE_INTERVAL", "5")
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.skills_creation_nudge_interval == 5
    assert cfg.sources["skills_creation_nudge_interval"] == ("env:DEEPAGENT_HERMES_SKILLS_CREATION_NUDGE_INTERVAL")


def test_env_override_model_default(monkeypatch):
    monkeypatch.setenv("DEEPAGENT_HERMES_MODEL_DEFAULT", "anthropic:claude-opus-4-7-20251001")
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.model_default == "anthropic:claude-opus-4-7-20251001"
    assert cfg.sources["model_default"] == "env:DEEPAGENT_HERMES_MODEL_DEFAULT"


def test_env_override_bool(monkeypatch):
    monkeypatch.setenv("DEEPAGENT_HERMES_COMPRESSION_ENABLED", "false")
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.compression_enabled is False


def test_env_override_csv_list(monkeypatch):
    monkeypatch.setenv("DEEPAGENT_HERMES_PLUGINS_ENABLED", "markdown, foo , bar")
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.plugins_enabled == ["markdown", "foo", "bar"]


def test_env_override_float(monkeypatch):
    monkeypatch.setenv("DEEPAGENT_HERMES_COMPRESSION_THRESHOLD", "0.75")
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.compression_threshold == pytest.approx(0.75)


def test_explicit_override_beats_env(monkeypatch):
    monkeypatch.setenv("DEEPAGENT_HERMES_AGENT_MAX_ITERATIONS", "42")
    cfg = HermesConfig.resolve(use_toml=False, overrides={"agent_max_iterations": 7})
    assert cfg.agent_max_iterations == 7
    assert cfg.sources["agent_max_iterations"] == "override"


# ── TOML resolution ──────────────────────────────────────────────────


def test_project_toml_overrides_defaults(monkeypatch, tmp_path):
    """A ``langstage-hermes.toml`` in the toml_start dir wins over defaults."""
    # Isolate from any real config files / env on the host.
    _strip_env(monkeypatch)
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(tmp_path / "no_global"))
    monkeypatch.chdir(tmp_path)

    (tmp_path / "langstage-hermes.toml").write_text(
        "[skills]\ncreation_nudge_interval = 25\n[memory]\nnudge_interval = 3\n",
        encoding="utf-8",
    )

    cfg = HermesConfig.resolve(toml_start=tmp_path)
    assert cfg.skills_creation_nudge_interval == 25
    assert cfg.memory_nudge_interval == 3
    # Source should be the TOML file we just wrote.
    assert "langstage-hermes.toml" in cfg.sources["skills_creation_nudge_interval"]


def test_env_beats_toml(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(tmp_path / "no_global"))
    monkeypatch.setenv("DEEPAGENT_HERMES_SKILLS_CREATION_NUDGE_INTERVAL", "99")
    monkeypatch.chdir(tmp_path)

    (tmp_path / "langstage-hermes.toml").write_text("[skills]\ncreation_nudge_interval = 25\n", encoding="utf-8")

    cfg = HermesConfig.resolve(toml_start=tmp_path)
    assert cfg.skills_creation_nudge_interval == 99
    assert cfg.sources["skills_creation_nudge_interval"].startswith("env:")


# ── describe() ───────────────────────────────────────────────────────


def test_describe_outputs_source_of_each_field(monkeypatch):
    """``describe()`` should print every field with its origin."""
    monkeypatch.setenv("DEEPAGENT_HERMES_SKILLS_CREATION_NUDGE_INTERVAL", "5")
    cfg = HermesConfig.resolve(use_toml=False)
    out = cfg.describe()

    # Every Hermes-specific field appears as its own line.
    for fname in (
        "model_default",
        "agent_max_iterations",
        "memory_nudge_interval",
        "skills_creation_nudge_interval",
        "compression_threshold",
        "delegation_max_iterations",
        "curator_interval_hours",
        "cron_tick_seconds",
        "plugins_enabled",
    ):
        assert fname in out, f"missing field {fname!r} in describe() output"

    # The env-overridden field reports its env source inline.
    assert "env:DEEPAGENT_HERMES_SKILLS_CREATION_NUDGE_INTERVAL" in out


def test_describe_includes_env_var_hint_for_hermes_fields(monkeypatch):
    cfg = HermesConfig.resolve(use_toml=False)
    out = cfg.describe()
    # Hints surface the env var for at least one Hermes-specific field.
    assert "DEEPAGENT_HERMES_MODEL_DEFAULT" in out
    assert "DEEPAGENT_HERMES_COMPRESSION_THRESHOLD" in out


# ── HERMES_HOME resolver ─────────────────────────────────────────────


def test_hermes_home_default(monkeypatch):
    monkeypatch.delenv("DEEPAGENT_HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert hermes_home() == Path.home() / ".langstage-hermes"


def test_hermes_home_from_deepagent_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(tmp_path / "deephome"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "legacyhome"))
    assert hermes_home() == tmp_path / "deephome"


def test_hermes_home_falls_back_to_legacy(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPAGENT_HERMES_HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "legacyhome"))
    assert hermes_home() == tmp_path / "legacyhome"


def test_hermes_home_exposed_via_config(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(tmp_path / "deephome"))
    cfg = HermesConfig.resolve(use_toml=False)
    assert cfg.hermes_home == tmp_path / "deephome"
