"""gh #98 — ``skills validate PATH``: a non-mutating pre-install SKILL.md check.

The keyless-preview pattern (``search`` gh #79, ``memory notes`` gh #94) applied
to skill authoring. ``skills validate`` runs the exact agentskills.io frontmatter
validator ``skills install`` / ``skills audit`` use, against an arbitrary working
tree, and writes NOTHING — no copy into ``<HERMES_HOME>/skills`` and no ``audit``
row (the side effects that made ``skills install`` an awkward "did I get the
frontmatter right?" check). Exit 0 valid, 1 invalid, ``--json`` for CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import frontmatter
from click.testing import CliRunner

from langstage_hermes.cli import cli


def _write_skill_dir(base: Path, name: str, metadata: dict) -> Path:
    d = base / name
    d.mkdir(parents=True)
    post = frontmatter.Post("# body\n\n## When to use\n\nwhen validating\n", **metadata)
    (d / "SKILL.md").write_text(frontmatter.dumps(post), encoding="utf-8")
    return d


def test_validate_valid_skill_exits_0_and_mutates_nothing(tmp_hermes_home: Path, tmp_path: Path):
    d = _write_skill_dir(tmp_path / "wt", "vv", {"name": "vv", "description": "A valid skill to VALIDATE, not install."})

    r = CliRunner().invoke(cli, ["skills", "validate", str(d)])
    assert r.exit_code == 0, r.output
    assert "valid" in r.output.lower()

    # The whole point: no install-side effects. Nothing copied into the store...
    assert not (tmp_hermes_home / "skills" / "vv").exists()
    # ...and no audit row landed (contrast `skills install`, which logs a create).
    audit = CliRunner().invoke(cli, ["audit", "log"]).output
    assert "vv" not in audit


def test_validate_accepts_skill_md_file_path(tmp_hermes_home: Path, tmp_path: Path):
    """PATH mirrors `skills install`: a SKILL.md file works, not only a dir."""
    d = _write_skill_dir(tmp_path / "wt", "ff", {"name": "ff", "description": "file path form"})
    r = CliRunner().invoke(cli, ["skills", "validate", str(d / "SKILL.md")])
    assert r.exit_code == 0, r.output
    assert "valid" in r.output.lower()


def test_validate_invalid_frontmatter_exits_1(tmp_hermes_home: Path, tmp_path: Path):
    # Uppercase name — the validator rejects it (lowercase/digits/hyphens only).
    d = _write_skill_dir(tmp_path / "wt", "Bad-Name", {"name": "Bad-Name", "description": "uppercase rejected"})
    r = CliRunner().invoke(cli, ["skills", "validate", str(d)])
    assert r.exit_code == 1, r.output
    assert "invalid" in r.output.lower()


def test_validate_missing_description_exits_1(tmp_hermes_home: Path, tmp_path: Path):
    d = _write_skill_dir(tmp_path / "wt", "nd", {"name": "nd"})  # no description
    r = CliRunner().invoke(cli, ["skills", "validate", str(d)])
    assert r.exit_code == 1, r.output
    assert "description" in r.output.lower()


def test_validate_json_valid_shape(tmp_hermes_home: Path, tmp_path: Path):
    d = _write_skill_dir(tmp_path / "wt", "jj", {"name": "jj", "description": "valid json skill"})
    r = CliRunner().invoke(cli, ["skills", "validate", str(d), "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["valid"] is True
    assert data["name"] == "jj"
    assert data["errors"] == []


def test_validate_json_invalid_shape(tmp_hermes_home: Path, tmp_path: Path):
    d = _write_skill_dir(tmp_path / "wt", "kk", {"name": "kk"})  # missing description
    r = CliRunner().invoke(cli, ["skills", "validate", str(d), "--json"])
    assert r.exit_code == 1, r.output
    data = json.loads(r.output)
    assert data["valid"] is False
    assert data["errors"]  # non-empty error list


def test_validate_no_skill_md_exits_1(tmp_hermes_home: Path, tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    r = CliRunner().invoke(cli, ["skills", "validate", str(empty)])
    assert r.exit_code == 1, r.output
    assert "no skill.md" in r.output.lower()


# ── gh #107: a FILE path validates THAT file, never a sibling SKILL.md ──


def test_validate_named_file_is_acted_on_not_sibling_skill_md(tmp_hermes_home: Path, tmp_path: Path):
    """gh #107: pointing at a file must validate THAT file, not a sibling SKILL.md.

    The dangerous case the issue reports: an INVALID draft sits next to a VALID
    SKILL.md. The old resolution (``path.parent / "SKILL.md"``) discarded the
    filename and validated the sibling, returning a false ``✓ valid`` / exit 0 —
    defeating a command advertised as a CI/pre-commit gate. It must now report the
    draft as invalid (exit 1).
    """
    # A VALID SKILL.md sibling in the same directory.
    d = _write_skill_dir(tmp_path / "wt", "sib", {"name": "sib", "description": "a valid sibling skill"})
    # The INVALID file the user actually names (no frontmatter at all).
    draft = d / "draft.md"
    draft.write_text("this is not a valid skill\n", encoding="utf-8")

    r = CliRunner().invoke(cli, ["skills", "validate", str(draft)])
    assert r.exit_code == 1, r.output  # NOT a false ✓ valid about the sibling
    assert "invalid" in r.output.lower()

    # --json makes the target explicit: the reported path is the file named, not
    # the sibling SKILL.md the tool used to silently swap in.
    rj = CliRunner().invoke(cli, ["skills", "validate", str(draft), "--json"])
    assert rj.exit_code == 1, rj.output
    data = json.loads(rj.output)
    assert data["valid"] is False
    assert Path(data["path"]).name == "draft.md", data["path"]
    assert data["errors"]


def test_install_named_file_is_acted_on_not_sibling_skill_md(tmp_hermes_home: Path, tmp_path: Path):
    """gh #107: `install FILE` acts on FILE, never a sibling SKILL.md.

    Pointing `install` at an invalid draft beside a valid SKILL.md must REJECT the
    draft (exit 2), not silently install the sibling skill under its own name.
    """
    d = _write_skill_dir(tmp_path / "wt", "sib2", {"name": "sibling-skill", "description": "a valid sibling"})
    draft = d / "draft.md"
    draft.write_text("this is not a valid skill\n", encoding="utf-8")

    r = CliRunner().invoke(cli, ["skills", "install", str(draft)])
    assert r.exit_code == 2, r.output  # invalid frontmatter → rejected
    # And it did NOT install the sibling skill.
    assert not (tmp_hermes_home / "skills" / "sibling-skill").exists()
