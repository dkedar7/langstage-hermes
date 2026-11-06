"""List-valued config keys written as a bare string (gh #209).

``skills.external_dirs = "extskills"`` used to pass through verbatim: ``skills list``
and ``chat`` crashed iterating a ``Path``, ``agent.disabled_toolsets = "file"``
disabled the toolsets ``f``/``i``/``l``/``e`` instead of ``file``, and
``--show-config`` reported no issues. A bare string now reads like the env layer
reads it (comma-split), and any other wrong shape is a reported malformed value.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from langstage_hermes.config import HermesConfig


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_hermes_home: Path, tmp_path: Path) -> Path:
    for var in [v for v in os.environ if v.startswith(("LANGSTAGE_", "DEEPAGENT_")) and "HOME" not in v]:
        monkeypatch.delenv(var, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    return project


def _toml(text: str) -> None:
    Path("langstage-hermes.toml").write_text(text, encoding="utf-8")


def _write_skill(root: Path, name: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: A test skill.\n---\nBody.\n", encoding="utf-8")


def test_bare_string_disabled_toolsets_disables_that_toolset():
    _toml('[agent]\ndisabled_toolsets = "file"\n')
    cfg = HermesConfig.resolve()
    assert cfg.agent_disabled_toolsets == ["file"]
    assert cfg.config_issues() == []


def test_bare_string_is_comma_split_like_the_env_layer():
    _toml('[plugins]\nenabled = "alpha, beta"\ndisabled = "gamma"\n')
    cfg = HermesConfig.resolve()
    assert cfg.plugins_enabled == ["alpha", "beta"]
    assert cfg.plugins_disabled == ["gamma"]


def test_bare_string_external_dirs_is_one_dir_rebased_on_the_toml(tmp_hermes_home):
    _write_skill(Path("extskills"), "my-skill")
    _toml('[skills]\nexternal_dirs = "extskills"\n')
    cfg = HermesConfig.resolve()
    assert cfg.skills_external_dirs == [str(Path("extskills").resolve())]

    from langstage_hermes.cli import _skill_library

    names = [s.name for s in _skill_library(with_audit=False).list()]
    assert "my-skill" in names


def test_bare_string_skills_disabled_hides_the_skill(tmp_hermes_home):
    _write_skill(tmp_hermes_home / "skills", "alpha")
    _write_skill(tmp_hermes_home / "skills", "writing-plans")
    _toml('[skills]\ndisabled = "writing-plans"\n')

    from langstage_hermes.cli import _skill_library

    names = [s.name for s in _skill_library(with_audit=False).list()]
    assert "alpha" in names
    assert "writing-plans" not in names


def test_platform_disabled_accepts_a_bare_string_per_platform():
    _toml('[skills.platform_disabled]\ncli = "a"\nweb = ["b", "c"]\n')
    cfg = HermesConfig.resolve()
    assert cfg.skills_platform_disabled == {"cli": ["a"], "web": ["b", "c"]}


@pytest.mark.parametrize(
    ("text", "field", "default"),
    [
        ("[skills]\ndisabled = 5\n", "skills_disabled", []),
        ('[skills]\nexternal_dirs = ["a", 3]\n', "skills_external_dirs", []),
        ("[agent]\ndisabled_toolsets = { file = true }\n", "agent_disabled_toolsets", []),
        ('[skills]\nplatform_disabled = "cli"\n', "skills_platform_disabled", {}),
        ("[skills.platform_disabled]\ncli = 1\n", "skills_platform_disabled", {}),
    ],
)
def test_other_wrong_shapes_are_reported_and_fall_back(capsys, text, field, default):
    _toml(text)
    cfg = HermesConfig.resolve()
    assert getattr(cfg, field) == default
    issues = cfg.config_issues()
    assert [(i["kind"], i["field"]) for i in issues] == [("malformed_value", field)]
    assert "note: ignoring malformed" in capsys.readouterr().err


def test_a_bad_project_value_keeps_the_global_one(tmp_hermes_home):
    (tmp_hermes_home / "config.toml").write_text('[skills]\ndisabled = ["alpha"]\n', encoding="utf-8")
    _toml("[skills]\ndisabled = 7\n")
    cfg = HermesConfig.resolve()
    assert cfg.skills_disabled == ["alpha"]
    assert cfg.sources["skills_disabled"].startswith("toml (config.toml)")


def test_list_overrides_go_through_the_same_check(capsys):
    cfg = HermesConfig.resolve(overrides={"agent_disabled_toolsets": "file,todo"}, use_toml=False)
    assert cfg.agent_disabled_toolsets == ["file", "todo"]
    cfg = HermesConfig.resolve(overrides={"skills_disabled": 3}, use_toml=False)
    assert cfg.skills_disabled == []
    assert [i["field"] for i in cfg.config_issues()] == ["skills_disabled"]
