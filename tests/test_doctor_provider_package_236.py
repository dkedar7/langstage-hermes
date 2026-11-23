"""`doctor` checks the provider package for every LangChain provider (gh #236).

`_PROVIDER_PACKAGES` only covers openai:* and anthropic:*, so `doctor` asserted
nothing for `ollama:`, `google_genai:`, `groq:` and the rest: it exited 0 with
`--json` `ok: true` while `verify` and `chat` then failed with an ImportError.
`doctor` now falls back to LangChain's own prefix -> package map, for the main
and the aux model, and names the distribution to install.

The package's absence is always SIMULATED through `importlib.util.find_spec`, so
these tests don't depend on what is installed.
"""

from __future__ import annotations

import importlib.util
import json

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import (
    _PROVIDER_PACKAGES,
    _doctor_provider_package,
    _langchain_provider_package,
    cli,
)

_real_find_spec = importlib.util.find_spec


def _isolate(monkeypatch, tmp_path):
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


def _missing(monkeypatch, *modules):
    """find_spec reports ``modules`` as absent and everything else as present."""

    def fake(name, *a, **kw):
        if name in modules:
            return None
        return object()

    monkeypatch.setattr("importlib.util.find_spec", fake)


def _doctor_json(monkeypatch):
    r = CliRunner().invoke(cli, ["doctor", "--json"])
    payload = json.loads(r.output)
    return r, payload, {c["name"]: c for c in payload["checks"]}


# ── the issue's repro ──────────────────────────────────────────────────────


def test_missing_ollama_package_fails_doctor(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "ollama:llama3.2")
    _missing(monkeypatch, "langchain_ollama")

    r = CliRunner().invoke(cli, ["doctor"])

    assert r.exit_code == 1, r.output
    assert "provider package 'langchain_ollama' not importable for ollama:llama3.2" in r.output
    assert "pip install langchain-ollama" in r.output


def test_missing_ollama_package_json_not_ok(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "ollama:llama3.2")
    _missing(monkeypatch, "langchain_ollama")

    r, payload, checks = _doctor_json(monkeypatch)

    assert r.exit_code == 1
    assert payload["ok"] is False
    assert checks["provider_pkg"]["ok"] is False
    assert checks["provider_pkg"]["hint"] == "pip install langchain-ollama"


def test_installed_ollama_package_passes_the_check(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "ollama:llama3.2")
    _missing(monkeypatch)  # nothing missing

    _, _, checks = _doctor_json(monkeypatch)

    assert checks["provider_pkg"] == {"name": "provider_pkg", "ok": True, "detail": "langchain_ollama installed"}


def test_missing_aux_provider_package_fails_doctor(monkeypatch, tmp_path):
    """A working openai:* main model with an ollama:* aux model: the reflection
    subagent would crash after the turn, so doctor must fail now."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_AUX", "ollama:llama3.2")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    _missing(monkeypatch, "langchain_ollama")

    r, payload, checks = _doctor_json(monkeypatch)

    assert r.exit_code == 1
    assert payload["ok"] is False
    assert checks["provider_pkg"]["ok"] is True
    assert checks["provider_pkg_aux"]["ok"] is False
    assert checks["provider_pkg_aux"]["hint"] == "pip install langchain-ollama"


# ── what stays unasserted ──────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["mycorp:internal-model", "gpt-4o-mini"])
def test_unknown_prefix_and_bare_name_stay_unasserted(monkeypatch, tmp_path, model):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", model)
    _missing(monkeypatch, "langchain_ollama")

    _, _, checks = _doctor_json(monkeypatch)

    assert "provider_pkg" not in checks
    assert _doctor_provider_package(model) is None


def test_hermes_extra_still_wins_for_openai(monkeypatch):
    assert _doctor_provider_package("openai:openai/gpt-4o-mini") is _PROVIDER_PACKAGES["openai:"]
    assert _doctor_provider_package("anthropic:claude-sonnet-4-6") is _PROVIDER_PACKAGES["anthropic:"]


# ── the LangChain map lookup ───────────────────────────────────────────────


def test_dotted_module_uses_the_top_level_package(monkeypatch):
    from langchain.chat_models import base as lc_base

    monkeypatch.setattr(
        lc_base,
        "_BUILTIN_PROVIDERS",
        {"azure_ai": ("langchain_azure_ai.chat_models", "AzureAIOpenAIApiChatModel", None)},
        raising=False,
    )
    entry = _langchain_provider_package("azure_ai:gpt-4o")
    assert entry is not None
    assert entry.module == "langchain_azure_ai"
    assert entry.install == "pip install langchain-azure-ai"


@pytest.mark.parametrize("shape", [None, {}, {"ollama": "langchain_ollama"}, {"ollama": ()}, {"ollama": (None,)}])
def test_missing_or_reshaped_map_asserts_nothing(monkeypatch, shape):
    from langchain.chat_models import base as lc_base

    if shape is None:
        monkeypatch.delattr(lc_base, "_BUILTIN_PROVIDERS", raising=False)
    else:
        monkeypatch.setattr(lc_base, "_BUILTIN_PROVIDERS", shape, raising=False)
    assert _langchain_provider_package("ollama:llama3.2") is None


# ── the key line for providers hermes has no key check for ─────────────────


def test_other_provider_key_not_reported_as_passing(monkeypatch, tmp_path):
    """google_genai:* with only OPENROUTER_API_KEY set used to print
    `OPENROUTER_API_KEY: set`, which read as a passing check for a key that
    model never uses."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "google_genai:gemini-2.0-flash")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    _missing(monkeypatch)

    r = CliRunner().invoke(cli, ["doctor"])

    assert "OPENROUTER_API_KEY: set" not in r.output
    assert "API key: not checked for google_genai:* models" in r.output


def test_bare_model_name_still_lists_present_keys(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    _missing(monkeypatch)

    r = CliRunner().invoke(cli, ["doctor"])

    assert "OPENAI_API_KEY: set" in r.output
