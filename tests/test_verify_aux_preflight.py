"""gh #96 — ``verify`` must preflight ``model_aux``, not only the main model.

The reflection review subagent (the project's headline closed-loop feature) runs
on ``model_aux``. On the documented mixed-provider path — an ``openai:*`` main
model + the default ``anthropic:*`` aux — a user who sets only the main model and
``OPENAI_API_KEY`` used to sail through ``verify`` green, then hit an Anthropic
auth error at the first reflection (~iteration 10). ``verify`` now runs the same
provider-aware key preflight against the aux model too, so the misconfig fails at
preflight (exit 2) *before* the agent is ever built — which is exactly what
``verify`` exists to catch.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

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


def test_verify_flags_unauthenticated_aux_model(monkeypatch, tmp_path):
    """openai:* main (+ key) with the default anthropic:* aux and no
    ANTHROPIC_API_KEY must FAIL preflight, naming the aux model + its missing key,
    and must not reach the agent-build / workspace stage."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-not-real")  # main model key present

    r = CliRunner().invoke(cli, ["verify"])

    assert r.exit_code == 2, r.output
    # Surfaced the aux model row and the specific missing key...
    assert "model (aux)" in r.output
    assert "ANTHROPIC_API_KEY" in r.output
    # ...and caught it at PREFLIGHT — before building the agent or its workspace.
    assert "isolated workspace" not in r.output
