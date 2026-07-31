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
    # The MAIN model line must not wrongly cite anthropic (the original #35 bug).
    # The separately-labeled aux row legitimately does now (gh #96) — but as
    # "...configured aux anthropic:* model", never the un-qualified main phrasing.
    assert "required for the configured anthropic:* model" not in r.output


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


def test_doctor_surfaces_mixed_provider_aux_model(monkeypatch, tmp_path):
    """gh #96: an openai:* main model + the default anthropic:* aux with no
    ANTHROPIC_API_KEY must surface the AUX model and its missing key — the
    reflection loop runs on the aux model, so doctor can no longer report only the
    main model and call the runtime healthy."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")  # main model key present
    # Hold the provider-package dimension fixed (present) so this isolates the aux
    # key row regardless of whether the CI env installed the [openai] extra.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    r = CliRunner().invoke(cli, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "model (aux):" in r.output  # the aux row exists
    assert "anthropic:claude-haiku" in r.output  # names the default aux model
    assert "aux anthropic:* model" in r.output  # and its distinct key requirement


def test_doctor_no_aux_row_when_same_provider(monkeypatch, tmp_path):
    """Default config: main + aux are both anthropic, so the aux shares the main
    model's key check — doctor prints a single anthropic key row, not a redundant
    aux one (gh #96 only surfaces a DIFFERENT-provider aux)."""
    _isolate(monkeypatch, tmp_path)

    r = CliRunner().invoke(cli, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "model (aux):" not in r.output
