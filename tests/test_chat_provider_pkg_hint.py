"""`chat` names the hermes extra for a missing provider package (gh #78).

On a plain ``pip install langstage-hermes`` (no extras), the README's documented
OpenAI/OpenRouter Quick start failed with langchain's raw ImportError —
``Please install it with `pip install langchain-openai``` — because ``chat``'s
agent-build ``except`` handler printed ``Failed to build agent: {e}`` verbatim.
That points AWAY from the documented install path (``pip install
"langstage-hermes[openai]"``) and contradicts ``verify``/``doctor``, which had
already been taught to name the extra (#33, #41, #76).

This pins that ``chat`` now appends the same extra guidance, that all three
commands read it from ONE shared table (``_PROVIDER_PACKAGES``) so they can't
drift a fourth time, and that an unrelated build failure gets NO misleading
"install the extra" hint.

The ImportError is always SIMULATED rather than relying on ``langchain_openai``
being absent from the venv, so these assertions hold identically on a machine
that has the ``[openai]`` extra installed.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import (
    _PROVIDER_PACKAGES,
    _missing_provider_install_line,
    _provider_package,
    cli,
)

# langchain's verbatim message when the openai provider package is absent.
_LANGCHAIN_OPENAI_IMPORT_ERROR = (
    "Initializing ChatOpenAI requires the langchain-openai package. Please install it with `pip install langchain-openai`"
)


def _isolate(monkeypatch, tmp_path):
    """No stray key/env/toml from the host."""
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


def _fail_build_with(monkeypatch, exc: BaseException) -> None:
    """Make the agent build raise ``exc`` (simulating the missing package)."""

    def _boom(target, cfg):
        raise exc

    monkeypatch.setattr("langstage_hermes.cli._instantiate_factory", _boom)


# ── chat integration: the headline repro ───────────────────────────────────


def test_chat_missing_openai_pkg_names_the_hermes_extra(monkeypatch, tmp_path):
    """The bug: chat leaked `pip install langchain-openai`, contradicting the
    README. It must now also name `langstage-hermes[openai]`."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-anything")  # past the #76 key gate
    _fail_build_with(monkeypatch, ImportError(_LANGCHAIN_OPENAI_IMPORT_ERROR))

    r = CliRunner().invoke(cli, ["chat"], input="hi\n/quit\n")

    assert r.exit_code == 2, r.output  # shape preserved: exit 2, no crash
    assert "Failed to build agent" in r.output  # still reports the failure
    # The fix: verify/doctor's gold-standard hint, naming the documented extra.
    assert 'pip install "langstage-hermes[openai]"' in r.output
    assert "for OpenAI-compatible models install:" in r.output
    assert "Traceback" not in r.output


def test_chat_missing_anthropic_pkg_hint_is_not_openai_only(monkeypatch, tmp_path):
    """Coverage is provider-general, not an openai:*-shaped special case: the
    anthropic:* path gets its own package hint (a base dep, so the plain
    package — there is no hermes extra to name)."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")  # past the key gate
    _fail_build_with(
        monkeypatch,
        ImportError("Initializing ChatAnthropic requires the langchain-anthropic package."),
    )

    r = CliRunner().invoke(cli, ["chat"], input="hi\n/quit\n")

    assert r.exit_code == 2, r.output
    assert "for Anthropic models install: pip install langchain-anthropic" in r.output
    # Must not misattribute to the openai extra.
    assert "langstage-hermes[openai]" not in r.output


def test_chat_unrelated_build_failure_gets_no_install_hint(monkeypatch, tmp_path):
    """The hint must be specific: a build that failed for any other reason must
    NOT be told to install an extra that has nothing to do with it."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-anything")
    _fail_build_with(monkeypatch, RuntimeError("checkpointer database is locked"))

    r = CliRunner().invoke(cli, ["chat"], input="hi\n/quit\n")

    assert r.exit_code == 2, r.output
    assert "checkpointer database is locked" in r.output
    assert "langstage-hermes[openai]" not in r.output
    assert "install:" not in r.output


# ── the shared table: one source of truth for chat/verify/doctor ───────────


def test_verify_reads_the_same_table(monkeypatch, tmp_path):
    """verify's gold-standard line is unchanged and now comes from the shared
    helper — the parity chat was brought up to."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-anything")
    monkeypatch.setattr(
        "langstage_hermes.create_hermes_agent",
        lambda *a, **kw: (_ for _ in ()).throw(ImportError(_LANGCHAIN_OPENAI_IMPORT_ERROR)),
    )

    r = CliRunner().invoke(cli, ["verify"])

    assert r.exit_code == 2, r.output
    assert 'for OpenAI-compatible models install: pip install "langstage-hermes[openai]"' in r.output


def test_doctor_reads_the_same_table(monkeypatch, tmp_path):
    """doctor's not-importable line is unchanged and now comes from the shared
    table (the third caller that must not drift)."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    r = CliRunner().invoke(cli, ["doctor"])

    assert r.exit_code == 2, r.output
    assert "provider package 'langchain_openai' not importable" in r.output
    assert _PROVIDER_PACKAGES["openai:"].install in r.output


def test_table_covers_every_provider_doctor_knows(monkeypatch):
    """The install command hermes advertises must be the hermes extra wherever
    one exists — never a bare langchain distribution the README doesn't mention.
    `[openai]` is the only model-provider extra pyproject declares; anthropic is
    a base dep."""
    assert _PROVIDER_PACKAGES["openai:"].install == 'pip install "langstage-hermes[openai]"'
    assert _PROVIDER_PACKAGES["anthropic:"].module == "langchain_anthropic"
    for prefix, entry in _PROVIDER_PACKAGES.items():
        assert prefix.endswith(":")
        assert _provider_package(f"{prefix}some/model") is entry


# ── _missing_provider_install_line unit contract ───────────────────────────


def test_hint_matches_distribution_spelling_even_when_installed(monkeypatch):
    """Guards against the test passing only by ambient accident: the exception
    text alone is enough, so the hint fires even where the package IS present."""
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    line = _missing_provider_install_line("openai:openai/gpt-4o-mini", ImportError(_LANGCHAIN_OPENAI_IMPORT_ERROR))
    assert line == 'for OpenAI-compatible models install: pip install "langstage-hermes[openai]"'


def test_hint_falls_back_to_importability_when_message_is_reworded(monkeypatch):
    """If langchain rewords its message we still catch it via find_spec."""
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    line = _missing_provider_install_line("openai:openai/gpt-4o-mini", ImportError("no module found, sorry"))
    assert line is not None and "langstage-hermes[openai]" in line


def test_no_hint_for_non_import_error():
    assert _missing_provider_install_line("openai:o", RuntimeError("langchain-openai")) is None


def test_no_hint_for_unknown_provider(monkeypatch):
    """A custom/local provider we ship no path for asserts nothing."""
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert _missing_provider_install_line("ollama:llama3", ImportError("boom")) is None


@pytest.mark.parametrize("model", ["openai:openai/gpt-4o-mini", "anthropic:claude-sonnet-4-6"])
def test_no_hint_when_package_present_and_unnamed(monkeypatch, model):
    """An ImportError about something else entirely, with the provider package
    installed, must not be blamed on the provider package."""
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    assert _missing_provider_install_line(model, ImportError("cannot import name 'foo' from 'bar'")) is None
