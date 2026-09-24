"""gh #159: `skills install` copies only the skill, never its surroundings.

A FILE install used to ``copytree`` the file's whole parent directory, so
``skills install ./SKILL.md`` at a project root dragged ``.env`` secrets,
``.git/`` and unrelated binaries into the skill library. A file now installs
only itself (as ``SKILL.md``); a directory install copies the skill's contents
but never hidden/VCS/secret entries, symlinks, or dependency/cache dirs, and
says what it skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli

_SKILL = "---\nname: proj-skill\ndescription: A skill authored at a project root.\n---\n# Proj skill\nSteps here.\n"


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _project_root(base: Path, skill_filename: str = "SKILL.md") -> Path:
    """The issue's repro: a skill file sitting in a real working directory."""
    root = base / "projroot"
    root.mkdir()
    (root / skill_filename).write_text(_SKILL, encoding="utf-8")
    (root / ".env").write_text("AWS_SECRET=topsecret\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("gitdata\n", encoding="utf-8")
    (root / "data.bin").write_bytes(os.urandom(4096))
    (root / "README.md").write_text("unrelated project readme\n", encoding="utf-8")
    (root / "node_modules" / "left-pad").mkdir(parents=True)
    (root / "node_modules" / "left-pad" / "index.js").write_text("x\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("filename", ["SKILL.md", "my-draft.md"])
def test_file_install_copies_only_that_file(tmp_hermes_home: Path, tmp_path: Path, filename: str):
    root = _project_root(tmp_path, filename)

    r = CliRunner().invoke(cli, ["skills", "install", str(root / filename)])
    assert r.exit_code == 0, r.output

    target = tmp_hermes_home / "skills" / "proj-skill"
    assert _files(target) == {"SKILL.md"}
    assert (target / "SKILL.md").read_text(encoding="utf-8") == _SKILL
    # The skill is loadable and passes the audit validator.
    audit = CliRunner().invoke(cli, ["skills", "validate", str(target)])
    assert audit.exit_code == 0, audit.output


def test_directory_install_excludes_secrets_vcs_and_junk(tmp_hermes_home: Path, tmp_path: Path):
    src = tmp_path / "clean-skill"
    src.mkdir()
    (src / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    # Legit skill layout (SPEC §9: references/, templates/, assets/; plus scripts/
    # and top-level support files as the bundled skills use).
    for rel in ("references/api.md", "templates/t.txt", "assets/logo.svg", "scripts/run.py", "editing.md"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text("ok\n", encoding="utf-8")
    # Things that must never enter the library.
    junk = {
        ".env": "SECRET=1\n",
        ".env.local": "SECRET=2\n",
        ".git/config": "gitdata\n",
        ".venv/pyvenv.cfg": "home = x\n",
        "myenv/pyvenv.cfg": "home = x\n",
        "venv/lib.py": "x\n",
        "node_modules/pkg/index.js": "x\n",
        "scripts/__pycache__/run.cpython-311.pyc": "x\n",
        "scripts/.secret": "x\n",
        "scripts/stale.pyc": "x\n",
        "pkg.egg-info/PKG-INFO": "x\n",
    }
    for rel, text in junk.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text, encoding="utf-8")

    r = CliRunner().invoke(cli, ["skills", "install", str(src)])
    assert r.exit_code == 0, r.output

    target = tmp_hermes_home / "skills" / "proj-skill"
    assert _files(target) == {
        "SKILL.md",
        "references/api.md",
        "templates/t.txt",
        "assets/logo.svg",
        "scripts/run.py",
        "editing.md",
    }
    # The user is told what was dropped.
    assert "Skipped" in r.output
    for shown in (".env", ".env.local", ".git/", ".venv/", "myenv/", "node_modules/", "scripts/.secret"):
        assert shown in r.output, (shown, r.output)
    # The audit row records the installed SKILL.md.
    log = CliRunner().invoke(cli, ["audit", "log"])
    assert "proj-skill" in log.output


def test_directory_install_skips_symlinks(tmp_hermes_home: Path, tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "id_rsa").write_text("PRIVATE KEY\n", encoding="utf-8")
    src = tmp_path / "linky"
    src.mkdir()
    (src / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    try:
        (src / "keys").symlink_to(outside, target_is_directory=True)
        (src / "key-file").symlink_to(outside / "id_rsa")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this platform")

    r = CliRunner().invoke(cli, ["skills", "install", str(src)])
    assert r.exit_code == 0, r.output
    assert _files(tmp_hermes_home / "skills" / "proj-skill") == {"SKILL.md"}
    assert "keys" in r.output and "key-file" in r.output


def test_clean_directory_install_reports_nothing_skipped(tmp_hermes_home: Path, tmp_path: Path):
    src = tmp_path / "clean"
    src.mkdir()
    (src / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    r = CliRunner().invoke(cli, ["skills", "install", str(src)])
    assert r.exit_code == 0, r.output
    assert "Skipped" not in r.output
