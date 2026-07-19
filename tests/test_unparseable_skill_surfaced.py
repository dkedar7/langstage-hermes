"""gh #81 — an unparseable SKILL.md must not vanish *silently*.

A SKILL.md whose YAML frontmatter fails to parse (one missing quote) is dropped
from ``SkillLibrary.list()``. Dropping it is correct — one bad file must not take
down the other 26 skills — but pre-0.4.19 the only trace was a ``logger.debug``
below the CLI's default level, so ``skills list`` and the **live agent** both went
silent: no warning, no error, exit 0. Only ``skills audit`` said anything.

These tests are built on the clean-room repro from the issue: one valid skill and
one neighbor with a single deliberate YAML typo, in an external skills dir.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import frontmatter
import pytest
import yaml
from click.testing import CliRunner
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langstage_hermes.skills.library import SkillLibrary, format_load_error

# The issue's repro, verbatim in shape: a valid neighbor and one typo'd skill.
VALID_SKILL = """\
---
name: valid-alpha
description: A valid neighbor skill that loads fine.
---
# Alpha
ok
"""

# The defect trigger: `description` opens a quote and never closes it, so
# yaml.scanner.ScannerError("while scanning a quoted scalar") is raised.
TYPO_SKILL = """\
---
name: typo-beta
description: "one missing quote here
---
# Beta
this skill has a single YAML typo in its description
"""


@pytest.fixture
def ext_skills(tmp_path: Path) -> Path:
    """An external skills dir holding valid-alpha + typo-beta."""
    ext = tmp_path / "ext"
    (ext / "valid-alpha").mkdir(parents=True)
    (ext / "typo-beta").mkdir(parents=True)
    (ext / "valid-alpha" / "SKILL.md").write_text(VALID_SKILL, encoding="utf-8")
    (ext / "typo-beta" / "SKILL.md").write_text(TYPO_SKILL, encoding="utf-8")
    return ext


@pytest.fixture
def ext_skills_configured(ext_skills: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``skills.external_dirs`` at the repro dir (as the issue does)."""
    monkeypatch.delenv("DEEPAGENT_HERMES_SKILLS_EXTERNAL_DIRS", raising=False)
    monkeypatch.setenv("LANGSTAGE_HERMES_SKILLS_EXTERNAL_DIRS", str(ext_skills))
    return ext_skills


def _sanity_check_repro(ext: Path) -> None:
    """The typo must be a genuine *parse* failure, not a validation choice."""
    frontmatter.load(str(ext / "valid-alpha" / "SKILL.md"))  # parses fine
    with pytest.raises(yaml.YAMLError):
        frontmatter.load(str(ext / "typo-beta" / "SKILL.md"))


# ---------------------------------------------------------------------------
# Library: the drop is recorded, not just swallowed
# ---------------------------------------------------------------------------


def test_list_records_the_dropped_skill_and_keeps_its_neighbor(ext_skills: Path):
    """``list()`` still skips the bad file (no crash) but now reports it."""
    _sanity_check_repro(ext_skills)
    lib = SkillLibrary(dirs=[ext_skills])

    names = [s.name for s in lib.list()]
    assert names == ["valid-alpha"], "the valid neighbor must still load"

    assert len(lib.load_errors) == 1, "the dropped skill must be recorded, not silently swallowed"
    err = lib.load_errors[0]
    assert err.parent_name == "typo-beta"
    assert err.path == ext_skills / "typo-beta" / "SKILL.md"
    assert err.message.startswith("parse failure: ")


def test_load_errors_reset_between_scans(ext_skills: Path):
    """``load_errors`` describes the *last* scan — fixing the file clears it."""
    lib = SkillLibrary(dirs=[ext_skills])
    lib.list()
    assert len(lib.load_errors) == 1

    post = frontmatter.Post("# Beta", name="typo-beta", description="now valid")
    (ext_skills / "typo-beta" / "SKILL.md").write_bytes(frontmatter.dumps(post).encode("utf-8"))

    names = [s.name for s in lib.list()]
    assert names == ["typo-beta", "valid-alpha"]
    assert lib.load_errors == [], "a repaired skill must not keep reporting a stale error"


def test_warning_and_audit_are_generated_from_the_same_detection(ext_skills: Path):
    """Guard against a second, drifting copy of the parse-failure logic.

    ``skills audit`` (``validate_all``) already detected and worded this exact
    failure; the warning path must reuse it rather than re-derive it. Assert both
    surfaces agree on the key *and* the message, so a change to one that skips
    the other trips here.
    """
    lib = SkillLibrary(dirs=[ext_skills])
    lib.list()
    err = lib.load_errors[0]

    audit_results = lib.validate_all()
    assert err.key in audit_results, "the warning's key must match the audit key (__error__/<dir>)"
    assert audit_results[err.key] == [err.message], "the warning and the audit row must use identical wording"


def test_formatted_warning_is_one_actionable_ascii_line(ext_skills: Path):
    """Names the dir, the file and the error; ASCII-only; single line.

    Windows cp1252 consoles raise UnicodeEncodeError on non-ASCII glyphs, and a
    ``yaml.scanner.ScannerError`` stringifies to four lines, which would smear a
    warning across the terminal.
    """
    lib = SkillLibrary(dirs=[ext_skills])
    lib.list()
    line = format_load_error(lib.load_errors[0])

    assert "typo-beta" in line, "must name the offending directory"
    assert str(ext_skills / "typo-beta" / "SKILL.md") in line, "must name the offending file"
    assert "parse failure" in line and "quoted scalar" in line, "must name the parse error"
    assert "skills audit" in line, "must point at the command with the full report"
    assert "\n" not in line, "multi-line YAML errors must be collapsed to one line"
    line.encode("ascii")  # raises UnicodeEncodeError if a non-ASCII glyph crept in
    line.encode("cp1252")


# ---------------------------------------------------------------------------
# `skills list` — the natural place a user checks "did my skill load?"
# ---------------------------------------------------------------------------


def _run_skills_list(args: list[str] | None = None) -> Any:
    from langstage_hermes.cli import cli

    return CliRunner().invoke(cli, ["skills", "list", *(args or [])])


def test_skills_list_warns_on_stderr_and_still_exits_zero(tmp_hermes_home, ext_skills_configured: Path):
    """The core regression: warning on stderr, exit code contract unchanged."""
    result = _run_skills_list()

    assert result.exit_code == 0, "a broken skill is a warning, not a failure -- the other skills still work"
    assert "valid-alpha" in result.stdout, "the valid neighbor must still be listed"
    assert "typo-beta" not in result.stdout, "the unparseable skill is still (correctly) not listed"

    assert "skipping unparseable skill" in result.stderr
    assert "typo-beta" in result.stderr
    assert "parse failure" in result.stderr
    assert "skills audit" in result.stderr


def test_skills_list_warning_never_touches_stdout(tmp_hermes_home, ext_skills_configured: Path):
    """stdout is a listing that gets piped/grepped -- keep the diagnostic out of it."""
    result = _run_skills_list()

    assert "warning" not in result.stdout.lower()
    assert "unparseable" not in result.stdout
    assert "parse failure" not in result.stdout
    # Every stdout line is either blank, a category header, a skill row, or the count.
    assert "__error__" not in result.stdout


def test_skills_list_warns_even_when_a_filter_hides_everything(tmp_hermes_home, ext_skills_configured: Path):
    """The drop is independent of --query/--category, including the early return."""
    result = _run_skills_list(["--query", "zzz-matches-nothing"])

    assert result.exit_code == 0
    assert "No skills match." in result.stdout
    assert "skipping unparseable skill" in result.stderr


def test_skills_list_is_silent_when_every_skill_parses(tmp_hermes_home, ext_skills, monkeypatch):
    """No false alarm: a healthy library emits nothing on stderr."""
    monkeypatch.delenv("DEEPAGENT_HERMES_SKILLS_EXTERNAL_DIRS", raising=False)
    post = frontmatter.Post("# Beta", name="typo-beta", description="now valid")
    (ext_skills / "typo-beta" / "SKILL.md").write_bytes(frontmatter.dumps(post).encode("utf-8"))
    monkeypatch.setenv("LANGSTAGE_HERMES_SKILLS_EXTERNAL_DIRS", str(ext_skills))

    result = _run_skills_list()

    assert result.exit_code == 0
    assert "skipping unparseable skill" not in result.stderr


# ---------------------------------------------------------------------------
# The live agent -- the worse half of the issue
# ---------------------------------------------------------------------------


def test_agent_build_warns_about_the_skill_it_silently_dropped(tmp_hermes_home, ext_skills_configured: Path, caplog):
    """A user starting `chat` never runs `skills list`; the agent must say it itself."""
    from langstage_hermes.agent import create_hermes_agent
    from langstage_hermes.config import HermesConfig

    with caplog.at_level(logging.WARNING, logger="langstage_hermes.agent"):
        graph = create_hermes_agent(
            HermesConfig.resolve(),
            model=FakeListChatModel(responses=["stub response"]),
        )

    assert graph is not None, "one broken skill must not stop the agent from starting"
    warnings = [r for r in caplog.records if "skipping unparseable skill" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly one build-time warning, got {len(warnings)}"
    msg = warnings[0].getMessage()
    assert warnings[0].levelno == logging.WARNING
    assert "typo-beta" in msg and "parse failure" in msg


def test_scanning_never_warns_so_the_repl_cannot_be_spammed(ext_skills: Path, caplog):
    """SkillLoaderMiddleware re-scans the library on *every* model call.

    If the warning were emitted from inside the scan it would repeat the same
    line for the whole session -- intolerable in a REPL. The warning therefore
    belongs to the one-shot build path (asserted above), and the scan itself
    must stay quiet no matter how many times it runs.
    """
    lib = SkillLibrary(dirs=[ext_skills])

    with caplog.at_level(logging.DEBUG, logger="langstage_hermes.skills.library"):
        for _ in range(5):
            lib.list()

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == [], (
        "the per-model-call scan must not log at WARNING -- it would repeat every turn"
    )
    # The drop is still recorded every scan, so a caller can surface it on demand.
    assert len(lib.load_errors) == 1
