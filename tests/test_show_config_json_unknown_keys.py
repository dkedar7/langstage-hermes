"""``--show-config --json`` reports the unknown keys the stderr notes warn about (gh #170).

The JSON's ``toml.unknown_keys`` and ``issues`` came from core's ``config_dict()``,
which reads an attribute hermes' own ``resolve()`` never set, so both were always
empty: a CI config-lint gate on the documented JSON passed on typos the human CLI
flagged.
"""

from __future__ import annotations

import json
import os
import textwrap

from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.config import HermesConfig

_TYPOS = textwrap.dedent(
    """
    [model]
    default = "anthropic:claude-sonnet-4-6"
    defaultt = "typo"
    [memory]
    memory_char_limit = 3000
    bogus_key = 1
    """
)


def _clean_env(monkeypatch):
    for var in [v for v in os.environ if v.startswith(("LANGSTAGE_", "HERMES_"))]:
        if var != "HERMES_HOME":
            monkeypatch.delenv(var, raising=False)


def test_resolved_config_reports_hermes_unknown_keys(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    (tmp_path / "langstage-hermes.toml").write_text(_TYPOS)
    cfg = HermesConfig.resolve(toml_start=tmp_path)
    assert cfg.unknown_toml_keys() == ["memory.bogus_key", "model.defaultt"]
    issues = {i["key"]: i for i in cfg.config_issues() if i["kind"] == "unknown_toml_key"}
    assert set(issues) == {"memory.bogus_key", "model.defaultt"}
    assert issues["model.defaultt"]["did_you_mean"] == "default"
    assert "did you mean 'default'" in issues["model.defaultt"]["message"]


def test_show_config_json_is_a_usable_lint_gate(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "langstage-hermes.toml").write_text(_TYPOS)
    r = CliRunner().invoke(cli, ["--show-config", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["toml"]["unknown_keys"] == ["memory.bogus_key", "model.defaultt"]
    assert {i["key"] for i in data["issues"] if i["kind"] == "unknown_toml_key"} == {
        "memory.bogus_key",
        "model.defaultt",
    }


def test_clean_config_stays_empty(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    (tmp_path / "langstage-hermes.toml").write_text('[model]\ndefault = "anthropic:claude-sonnet-4-6"\n')
    cfg = HermesConfig.resolve(toml_start=tmp_path)
    assert cfg.unknown_toml_keys() == []
    assert [i for i in cfg.config_issues() if i["kind"] == "unknown_toml_key"] == []


def test_shared_langstage_toml_is_not_linted(monkeypatch, tmp_path):
    """Other stages' keys in the shared langstage.toml are not hermes typos."""
    _clean_env(monkeypatch)
    (tmp_path / "langstage.toml").write_text("[server]\nport = 8050\n")
    cfg = HermesConfig.resolve(toml_start=tmp_path)
    assert cfg.unknown_toml_keys() == []
