"""gh #108 — ``--json`` on the two "run this first" readiness checks.

``doctor`` and ``verify`` are the commands the README tells a fresh adopter to run
*first*, and the two you'd gate a setup script / CI job / Dockerfile healthcheck
on — yet they were the only user-facing commands with no ``--json``, so a script
was left grepping human prose and inferring cause from an exit code.

These tests pin the fix: both commands emit one JSON object on stdout (mirroring
the ``skills audit --json`` shape — top-level ``ok`` + a per-check list), the
exit-code semantics are unchanged, ``.ok`` matches ``exit code == 0``, and the
human render is untouched. The live round-trip is reported ``skipped`` when a key
is absent, so a keyless CI run never triggers a paid call.
"""

from __future__ import annotations

import json
import tempfile as tempfile_mod
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langstage_hermes.cli import cli


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)  # no stray langstage-hermes.toml from the repo
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


# ── doctor --json ───────────────────────────────────────────────────────


def test_doctor_json_shape_and_ok_matches_exit(monkeypatch, tmp_path):
    """Keyless default anthropic:* config: doctor --json is one valid object with a
    top-level ``ok`` + per-check list, ``ok`` is false, and it equals exit==0."""
    _isolate(monkeypatch, tmp_path)
    # Hold the provider-package dimension fixed (present) so the only failure is
    # the missing key — isolates the JSON shape from CI-env extras.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    r = CliRunner().invoke(cli, ["doctor", "--json"])
    assert r.exit_code == 2, r.output
    data = json.loads(r.output)
    assert set(data) == {"ok", "checks"}
    assert data["ok"] is False
    assert data["ok"] is (r.exit_code == 0)  # `.ok` == readiness == exit 0

    checks = {c["name"]: c for c in data["checks"]}
    # Every check carries at least name/ok/detail.
    for c in data["checks"]:
        assert {"name", "ok", "detail"} <= set(c)
    # The configured model is surfaced, and the missing required key is the ✗.
    assert checks["model"]["detail"] == "anthropic:claude-sonnet-4-6"
    assert checks["api_key"]["ok"] is False
    assert "ANTHROPIC_API_KEY" in checks["api_key"]["detail"]
    assert "hint" in checks["api_key"]
    # Names the issue's proposal enumerates are all present.
    for name in ("python", "langstage-core", "provider_pkg", "hermes_home", "cron_dir", "bash"):
        assert name in checks


def test_doctor_json_ok_true_when_ready(monkeypatch, tmp_path):
    """With the required key present (and provider pkg fixed present), doctor --json
    reports ``ok: true`` and exits 0 — proving ``.ok`` tracks the exit code."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    r = CliRunner().invoke(cli, ["doctor", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["ok"] is True
    assert data["ok"] is (r.exit_code == 0)
    checks = {c["name"]: c for c in data["checks"]}
    assert checks["api_key"]["ok"] is True


def test_doctor_human_output_unchanged(monkeypatch, tmp_path):
    """No --json → the human report, never JSON."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    r = CliRunner().invoke(cli, ["doctor"])
    assert "langstage-hermes doctor:" in r.output
    assert "ANTHROPIC_API_KEY: not set (required for the configured anthropic:* model)" in r.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(r.output)


# ── verify --json ───────────────────────────────────────────────────────


def test_verify_json_keyless_skips_round_trip(monkeypatch, tmp_path):
    """Keyless default config: verify --json is one valid object, ``ok`` is false,
    exit 2, and the live round-trip is reported ``skipped`` (never a paid call)."""
    _isolate(monkeypatch, tmp_path)

    r = CliRunner().invoke(cli, ["verify", "--json"])
    assert r.exit_code == 2, r.output
    data = json.loads(r.output)
    assert set(data) == {"ok", "model", "model_aux", "checks"}
    assert data["ok"] is False
    assert data["model"] == "anthropic:claude-sonnet-4-6"
    assert data["model_aux"]  # the aux model is carried too (the #96 payload home)

    checks = {c["name"]: c for c in data["checks"]}
    assert checks["bundled_prompts"]["ok"] is True
    assert checks["bundled_skills"]["ok"] is True
    assert checks["hermes_home"]["ok"] is True
    assert checks["model_key"]["ok"] is False
    # round_trip present, skipped, and explicitly because there's no key.
    assert checks["round_trip"]["ok"] is False
    assert "skipped" in checks["round_trip"]["detail"]
    assert "no key" in checks["round_trip"]["detail"]
    # never built a workspace / made a call
    assert "fts5_store" not in checks


class _YesModel(BaseChatModel):
    """A tool-free fake that answers ``YES`` — enough for verify's one round-trip,
    with no network and no API key."""

    @property
    def _llm_type(self) -> str:  # pragma: no cover - identity only
        return "test-yes-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="YES"))])


def test_verify_json_passes_with_key_and_does_round_trip(monkeypatch, tmp_path, tmp_hermes_home):
    """With a key present + a fake model, verify --json runs the real round-trip on
    the fake, reports ``ok: true`` / exit 0, and records the round_trip + FTS5
    checks — and leaks no workspace."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-not-real")
    # route mkdtemp into an isolated dir so we can assert no leak
    scratch = tmp_path / "tmproot"
    scratch.mkdir()
    monkeypatch.setattr(tempfile_mod, "tempdir", str(scratch))

    from langstage_hermes.agent import create_hermes_agent as real_factory

    def fake_factory(cfg: Any = None, **kwargs: Any) -> Any:
        return real_factory(cfg, model=_YesModel(), **kwargs)

    monkeypatch.setattr("langstage_hermes.create_hermes_agent", fake_factory)

    r = CliRunner().invoke(cli, ["verify", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["ok"] is True
    checks = {c["name"]: c for c in data["checks"]}
    assert checks["model_key"]["ok"] is True
    assert checks["round_trip"]["ok"] is True
    assert checks["fts5_store"]["ok"] is True
    # workspace cleaned up (no leak), matching the human path's contract (gh #68)
    assert list(scratch.glob("dah-verify-*")) == []


def test_verify_human_path_unchanged(monkeypatch, tmp_path):
    """No --json → the human banner + fail-fast exit, never JSON."""
    _isolate(monkeypatch, tmp_path)
    r = CliRunner().invoke(cli, ["verify"])
    assert r.exit_code == 2, r.output
    assert "langstage-hermes verify — live end-to-end smoke" in r.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(r.output)
