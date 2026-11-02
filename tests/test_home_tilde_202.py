"""gh #202: a literal ``~`` in HERMES_HOME (quoted export, .env, Docker ENV) must
expand to the home directory, not become ``<cwd>/~``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def fake_home(monkeypatch, tmp_path: Path) -> Path:
    """Point ``~`` at a tmp dir (HOME on POSIX, USERPROFILE on Windows) and run in another cwd."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for var in ("LANGSTAGE_HERMES_HOME", "HERMES_HOME", "DEEPAGENT_HERMES_HOME"):
        monkeypatch.delenv(var, raising=False)
    cwd = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return home


@pytest.mark.parametrize("var", ["HERMES_HOME", "LANGSTAGE_HERMES_HOME", "DEEPAGENT_HERMES_HOME"])
def test_tilde_in_every_home_var_expands(monkeypatch, fake_home: Path, var: str):
    from langstage_hermes.config import hermes_home, hermes_home_override

    monkeypatch.setenv("LANGSTAGE_SUPPRESS_LEGACY_NOTICE", "1")
    monkeypatch.setenv(var, "~/.langstage-hermes")
    assert hermes_home() == fake_home / ".langstage-hermes"
    assert Path(hermes_home_override()) == fake_home / ".langstage-hermes"


def test_configured_home_used_by_demo_expands_too(monkeypatch, fake_home: Path):
    from langstage_hermes.cli import _configured_hermes_home

    monkeypatch.setenv("HERMES_HOME", "~/.langstage-hermes")
    assert _configured_hermes_home() == fake_home / ".langstage-hermes"


def test_absolute_and_relative_values_are_unchanged(monkeypatch, fake_home: Path, tmp_path: Path):
    from langstage_hermes.config import hermes_home

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "abs"))
    assert hermes_home() == tmp_path / "abs"
    monkeypatch.setenv("HERMES_HOME", "rel/home")
    assert hermes_home() == Path("rel/home")


def test_doctor_writes_to_the_real_home_and_no_tilde_dir(monkeypatch, fake_home: Path):
    from langstage_hermes.cli import cli

    monkeypatch.setenv("HERMES_HOME", "~/.langstage-hermes")
    result = CliRunner().invoke(cli, ["doctor", "--json"])
    checks = {c["name"]: c for c in json.loads(result.output)["checks"]}
    assert Path(checks["hermes_home"]["detail"]) == fake_home / ".langstage-hermes"
    assert (fake_home / ".langstage-hermes").is_dir()
    assert not (Path.cwd() / "~").exists()
    assert "stray_tilde_home" not in checks


def test_doctor_points_at_a_leftover_tilde_dir_without_touching_it(monkeypatch, fake_home: Path):
    from langstage_hermes.cli import cli

    stray = Path.cwd() / "~" / ".langstage-hermes"
    stray.mkdir(parents=True)
    (stray / "state.db").write_bytes(b"")
    monkeypatch.setenv("HERMES_HOME", "~/.langstage-hermes")
    result = CliRunner().invoke(cli, ["doctor", "--json"])
    checks = {c["name"]: c for c in json.loads(result.output)["checks"]}
    note = checks["stray_tilde_home"]
    assert note["ok"] is True
    assert "rm -rf ./~" in note["detail"]
    assert (stray / "state.db").exists()


def test_unrelated_tilde_dir_is_not_reported(fake_home: Path):
    from langstage_hermes.config import stray_tilde_home

    (Path.cwd() / "~").mkdir()
    assert stray_tilde_home() is None


def test_other_path_env_vars_expand_tilde(monkeypatch, fake_home: Path):
    from langstage_hermes.tools.environments import local, singularity

    bash = fake_home / "bin" / "bash"
    bash.parent.mkdir()
    bash.write_text("")
    monkeypatch.setenv("LANGSTAGE_HERMES_BASH_PATH", "~/bin/bash")
    assert Path(local._find_bash()) == bash

    monkeypatch.setenv("LANGSTAGE_HERMES_SINGULARITY_WORKSPACE", "~/sing")
    assert singularity._resolve_workspace() == fake_home / "sing"
    assert not (Path.cwd() / "~").exists()


def test_ssh_key_env_expands_tilde(monkeypatch, fake_home: Path):
    pytest.importorskip("paramiko")
    from langstage_hermes.tools.environments.ssh import SshEnvironment

    monkeypatch.setenv("LANGSTAGE_HERMES_SSH_HOST", "example.org")
    monkeypatch.setenv("LANGSTAGE_HERMES_SSH_KEY", "~/.ssh/acme_key")
    assert Path(SshEnvironment(session_id="s").key_path) == fake_home / ".ssh" / "acme_key"
