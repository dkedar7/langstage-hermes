"""``HermesConfig.resolve(overrides=...)`` coerces like TOML (the hermes twin of
langstage-core #225): a str "false" for a bool field used to stay truthy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from langstage_hermes.config import HermesConfig


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in [v for v in os.environ if v.startswith(("LANGSTAGE_", "DEEPAGENT_"))]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_string_overrides_are_coerced_to_the_field_type():
    cfg = HermesConfig.resolve(
        overrides={"curator_enabled": "false", "memory_nudge_interval": "5", "compression_threshold": "0.6"},
        use_toml=False,
    )
    assert cfg.curator_enabled is False
    assert cfg.memory_nudge_interval == 5
    assert cfg.compression_threshold == 0.6
    assert cfg.sources["curator_enabled"] == "override"


def test_typed_and_list_overrides_pass_through():
    cfg = HermesConfig.resolve(overrides={"curator_enabled": False, "agent_disabled_toolsets": ["todo"]}, use_toml=False)
    assert cfg.curator_enabled is False
    assert cfg.agent_disabled_toolsets == ["todo"]


def test_an_uncoercible_override_keeps_the_layer_beneath(capsys):
    default = HermesConfig.resolve(use_toml=False).memory_nudge_interval
    cfg = HermesConfig.resolve(overrides={"memory_nudge_interval": "lots"}, use_toml=False)
    assert cfg.memory_nudge_interval == default
    assert cfg.sources["memory_nudge_interval"] == "default"
    issues = [i for i in cfg.config_issues() if i.get("kind") == "malformed_value"]
    assert [(i["field"], i["source"]) for i in issues] == [("memory_nudge_interval", "override")]
    assert "malformed override memory_nudge_interval='lots'" in capsys.readouterr().err
