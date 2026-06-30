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
    # This test is about the KEY check; hold the provider-package dimension fixed
    # (present) so it isolates the key behavior regardless of whether the CI env
    # installed the [openai] extra. The missing-package path has its own test below.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

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


def test_doctor_fails_when_openai_provider_pkg_missing(monkeypatch, tmp_path):
    """The bug (gh #41): doctor green-lit an openai:* model with the [openai]
    extra absent, while verify (exit 2) named the fix. doctor must now agree."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    # Simulate `langchain_openai` not installed (the [openai] extra is missing)
    # regardless of what the dev venv happens to have.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    r = CliRunner().invoke(cli, ["doctor"])
    assert r.exit_code == 2, r.output  # matches verify, no longer a clean bill
    assert "provider package 'langchain_openai' not importable" in r.output
    assert 'pip install "langstage-hermes[openai]"' in r.output  # verify's gold-standard hint


def test_doctor_passes_when_provider_pkg_present(monkeypatch, tmp_path):
    """The default anthropic:* path: langchain-anthropic is a core dep, so the
    provider-package check passes and doctor still exits 0."""
    _isolate(monkeypatch, tmp_path)

    r = CliRunner().invoke(cli, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "provider package: langchain_anthropic installed" in r.output
