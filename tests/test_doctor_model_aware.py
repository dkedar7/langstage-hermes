"""doctor checks the CONFIGURED model's provider key, like verify (gh #35).

doctor used to hardcode ANTHROPIC_API_KEY and stay silent on a missing
OpenAI/OpenRouter key, so on the README's openai:* / OpenRouter path it both
cited the wrong key and hid the one actually required.
"""

from click.testing import CliRunner

from langstage_hermes.cli import cli


def _isolate(monkeypatch, tmp_path):
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
    ):
        monkeypatch.delenv(k, raising=False)


def test_doctor_flags_openai_key_for_openai_model(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")

    r = CliRunner().invoke(cli, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "openai:openai/gpt-4o-mini" in r.output  # reports the configured model
    assert "OPENAI_API_KEY / OPENROUTER_API_KEY: not set" in r.output  # the right missing key
    assert "anthropic:* model" not in r.output  # no longer wrongly cites anthropic


def test_doctor_checks_anthropic_for_default_model(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    r = CliRunner().invoke(cli, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "ANTHROPIC_API_KEY: not set (required for the configured anthropic:* model)" in r.output
