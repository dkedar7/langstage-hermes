"""Regression tests for the Wave 4 "advertised but not honored" fixes.

verify's keyless FTS5 check (gh #128), empty-body skills (gh #129), the loader
skipping skills the validator rejects (gh #132), the `memory show` over-budget
label (gh #130), standard cron syntax (gh #147), the legacy
`deepagent-hermes.toml` notice (gh #125), `--show-config --json` (gh #117), the
runnable extractors docstring (gh #156) and the exact `skills remove` restore
command (gh #157). The MarkdownProvider heading context (gh #121) lives in
``test_markdown_provider.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.cron import jobs as cron_jobs
from langstage_hermes.skills.library import SkillLibrary, format_load_error
from langstage_hermes.skills.prompt import build_skills_system_prompt
from langstage_hermes.skills.validator import validate


def _keyless(monkeypatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("LANGSTAGE_HERMES_HOME", raising=False)
    monkeypatch.delenv("DEEPAGENT_HERMES_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "LANGSTAGE_HERMES_MODEL_DEFAULT",
        "DEEPAGENT_HERMES_MODEL_DEFAULT",
        "LANGSTAGE_HERMES_MODEL_AUX",
        "DEEPAGENT_HERMES_MODEL_AUX",
    ):
        monkeypatch.delenv(k, raising=False)
    return home


# ── verify: FTS5 store checked on the keyless path (gh #128) ─────────


def test_verify_json_keyless_checks_fts5_store(monkeypatch, tmp_path):
    home = _keyless(monkeypatch, tmp_path)
    r = CliRunner().invoke(cli, ["verify", "--json"])
    assert r.exit_code == 1, r.output  # still no key
    checks = {c["name"]: c for c in json.loads(r.output)["checks"]}
    assert checks["fts5_init"]["ok"] is True, checks["fts5_init"]
    assert "FTS5" in checks["fts5_init"]["detail"]
    # The store was actually opened (and its schema created) without a model call.
    assert (home / "state.db").is_file()
    assert "fts5_store" not in checks  # the live side-effect check still needs a key


def test_verify_json_reports_fts5_init_failure(monkeypatch, tmp_path):
    _keyless(monkeypatch, tmp_path)

    def _boom(*_a, **_k):
        raise RuntimeError("no fts5 here")

    monkeypatch.setattr("langstage_hermes.store.sqlite_fts.SqliteFtsStore.__init__", _boom)
    r = CliRunner().invoke(cli, ["verify", "--json"])
    checks = {c["name"]: c for c in json.loads(r.output)["checks"]}
    assert checks["fts5_init"]["ok"] is False
    assert "no fts5 here" in checks["fts5_init"]["detail"]


def test_verify_human_keyless_prints_fts5_line(monkeypatch, tmp_path):
    _keyless(monkeypatch, tmp_path)
    r = CliRunner().invoke(cli, ["verify"])
    assert r.exit_code != 0  # keyless still fails at the key preflight
    assert "FTS5 store" in r.output
    # ...and the store line comes BEFORE the key failure, i.e. it ran keyless.
    assert r.output.index("FTS5 store") < r.output.index("ANTHROPIC_API_KEY")


# ── skills: empty body rejected everywhere (gh #129) ─────────────────


def _skill_dir(root: Path, name: str, fm: str, body: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\n{fm}---\n{body}", encoding="utf-8")
    return d


def test_validator_rejects_empty_body():
    fm = {"name": "x", "description": "d"}
    assert validate(fm) == []  # frontmatter-only callers are unchanged
    assert validate(fm, body="# Real\nsteps") == []
    errs = validate(fm, body="  \n\n")
    assert len(errs) == 1 and errs[0].startswith("body:")


def test_skills_validate_install_audit_reject_empty_body(tmp_hermes_home: Path, tmp_path: Path):
    src = _skill_dir(tmp_path / "src", "no-body", "name: no-body\ndescription: Forgot the body.\n", "")
    runner = CliRunner()

    v = runner.invoke(cli, ["skills", "validate", str(src)])
    assert v.exit_code == 1, v.output
    assert "body" in v.output

    i = runner.invoke(cli, ["skills", "install", str(src)])
    assert i.exit_code == 1, i.output
    assert not (tmp_hermes_home / "skills" / "no-body").exists()

    # A hand-placed one is flagged by audit.
    _skill_dir(tmp_hermes_home / "skills", "no-body", "name: no-body\ndescription: Forgot the body.\n", "\n")
    a = runner.invoke(cli, ["skills", "audit"])
    assert a.exit_code == 1, a.output
    assert "no-body" in a.output and "body" in a.output


def test_library_write_rejects_empty_body(tmp_hermes_home: Path):
    lib = SkillLibrary(dirs=[tmp_hermes_home / "skills"])
    with pytest.raises(ValueError, match="body"):
        lib.write("empty", {"name": "empty", "description": "d"}, "")


# ── skills: loader agrees with the validator (gh #132) ───────────────


@pytest.mark.parametrize(
    ("fm", "body", "reason"),
    [
        ("name: broken-skill\n", "# Broken\nBody present.\n", "description"),
        ("name: broken-skill\ndescription: ''\n", "# Broken\nBody present.\n", "description"),
        ("name: broken-skill\ndescription: Has one.\n", "", "body"),
    ],
)
def test_loader_skips_signal_less_skill_with_a_note(tmp_path: Path, fm: str, body: str, reason: str):
    skills = tmp_path / "skills"
    _skill_dir(skills, "broken-skill", fm, body)
    _skill_dir(skills, "good-skill", "name: good-skill\ndescription: Works.\n", "# Good\nsteps\n")
    lib = SkillLibrary(dirs=[skills])

    names = [s.name for s in lib.list()]
    assert names == ["good-skill"]  # the neighbour still loads; nothing crashes
    assert len(lib.load_errors) == 1
    err = lib.load_errors[0]
    assert err.parent_name == "broken-skill"
    assert err.message.startswith(f"{reason}:")
    line = format_load_error(err)
    assert "skipping invalid skill 'broken-skill'" in line
    assert "unparseable" not in line
    # Never reaches the agent's boot index.
    assert "broken-skill" not in build_skills_system_prompt(lib)


def test_skills_list_json_reports_the_skipped_skill(tmp_hermes_home: Path):
    _skill_dir(tmp_hermes_home / "skills", "broken-skill", "name: broken-skill\n", "# Broken\nBody.\n")
    r = CliRunner().invoke(cli, ["skills", "list", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert not any(s["name"] == "broken-skill" for s in data["skills"])
    assert any("broken-skill" in e and "description" in e for e in data["load_errors"])


# ── memory show: honest over-budget label (gh #130) ──────────────────


def test_memory_show_over_budget_label_matches_runtime(tmp_hermes_home: Path):
    (tmp_hermes_home / "memories" / "USER.md").write_text("x" * 1600, encoding="utf-8")
    r = CliRunner().invoke(cli, ["memory", "show", "--user"])
    assert r.exit_code == 0, r.output
    assert "over budget" in r.output
    assert "truncated" not in r.output  # the runtime never truncates...
    assert "injected in full" in r.output  # ...it injects the whole layer

    from langstage_hermes.memory import tool as mt

    snap = mt.build_snapshot(user_char_limit=1375, memory_char_limit=2200)
    assert "x" * 1600 in snap["user_snapshot"]  # pin the behavior the label describes


# ── cron: standard cron syntax (gh #147) ─────────────────────────────


@pytest.mark.parametrize(
    "expr",
    [
        "0 9 * * MON-FRI",
        "0 9 * * mon",
        "0 0 1 JAN *",
        "*/15 9-17 * * MON-FRI",
        "@daily",
        "@hourly",
        "@weekly",
        "@monthly",
        "@yearly",
        "@annually",
        "@midnight",
        "@DAILY",
    ],
)
def test_parse_schedule_accepts_named_fields_and_shortcuts(expr):
    s = cron_jobs.parse_schedule(expr)
    assert s["kind"] == "cron"
    assert s["expr"] == expr
    job = {"schedule": s, "created_at": "2026-01-01T00:00:00+00:00"}
    assert cron_jobs.compute_next_run(job) is not None


@pytest.mark.parametrize("expr", ["0 9 * * FUNDAY", "@fortnightly", "0 9 * * MON-XYZ"])
def test_parse_schedule_still_rejects_bad_names(expr):
    with pytest.raises(ValueError):
        cron_jobs.parse_schedule(expr)


def test_parse_schedule_plain_words_are_not_cron():
    with pytest.raises(ValueError, match="Invalid schedule"):
        cron_jobs.parse_schedule("run it every day please")


def test_cron_create_accepts_weekday_names(tmp_hermes_home: Path):
    r = CliRunner().invoke(cli, ["cron", "create", "--prompt", "p", "--schedule", "0 9 * * MON-FRI"])
    assert r.exit_code == 0, r.output


# ── legacy deepagent-hermes.toml notice (gh #125) ────────────────────


def test_legacy_project_toml_emits_deprecation(monkeypatch, tmp_path: Path):
    from langstage_core.host import config as core_config

    from langstage_hermes.config import HermesConfig

    (tmp_path / "deepagent-hermes.toml").write_text('[model]\ndefault = "openai:x"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LANGSTAGE_SUPPRESS_LEGACY_NOTICE", raising=False)
    monkeypatch.setattr(core_config, "_warned_legacy_toml", set())
    with pytest.warns(DeprecationWarning, match="langstage-hermes.toml"):
        cfg = HermesConfig.resolve()
    assert cfg.model_default == "openai:x"  # still honored


def test_new_project_toml_is_silent(monkeypatch, tmp_path: Path):
    from langstage_core.host import config as core_config

    from langstage_hermes.config import HermesConfig

    (tmp_path / "langstage-hermes.toml").write_text('[model]\ndefault = "openai:x"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core_config, "_warned_legacy_toml", set())
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        HermesConfig.resolve()


def test_legacy_project_toml_prints_one_stderr_note(tmp_path: Path):
    (tmp_path / "deepagent-hermes.toml").write_text('[model]\ndefault = "openai:x"\n', encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTEST_CURRENT_TEST", "LANGSTAGE_SUPPRESS_LEGACY_NOTICE")}
    res = subprocess.run(
        [sys.executable, "-m", "langstage_hermes.cli", "--show-config"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    notes = [ln for ln in res.stderr.splitlines() if "legacy name" in ln]
    assert len(notes) == 1, res.stderr
    assert "deepagent-hermes.toml" in notes[0] and "langstage-hermes.toml" in notes[0]

    env["LANGSTAGE_SUPPRESS_LEGACY_NOTICE"] = "1"
    quiet = subprocess.run(
        [sys.executable, "-m", "langstage_hermes.cli", "--show-config"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
    )
    assert "legacy name" not in quiet.stderr


# ── --show-config --json (gh #117) ───────────────────────────────────


def test_show_config_json_mirrors_config_dict(monkeypatch, tmp_path: Path):
    (tmp_path / "langstage-hermes.toml").write_text('[model]\ndefault = "openai:x"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli, ["--show-config", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert {"config", "toml", "issues"} <= set(data)
    md = data["config"]["model_default"]
    assert md["value"] == "openai:x"
    assert md["source"] == "toml (langstage-hermes.toml)"
    assert md["env"] == "LANGSTAGE_HERMES_MODEL_DEFAULT"
    assert md["toml"] == "model.default"
    assert {"value", "source", "env", "legacy_env", "toml"} <= set(md)
    assert data["toml"]["found"] is True
    assert data["toml"]["path"].endswith("langstage-hermes.toml")


def test_show_config_json_key_set_matches_human(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    human = CliRunner().invoke(cli, ["--show-config"]).output
    data = json.loads(CliRunner().invoke(cli, ["--show-config", "--json"]).output)
    for field in data["config"]:
        assert f"  {field} " in human


def test_json_without_show_config_is_a_usage_error():
    r = CliRunner().invoke(cli, ["--json"])
    assert r.exit_code == 64  # usage error (ADR 0007)
    assert "--show-config" in r.output


# ── extractors docstring example runs (gh #156) ──────────────────────


def test_extractors_docstring_example_runs():
    import langstage_hermes.extractors as ex

    doc = ex.__doc__ or ""
    assert "StreamParser" not in doc
    block = doc.split("::", 1)[1]
    code = textwrap.dedent(block.split("\n\n", 1)[1] if block.startswith("\n\n") else block)
    ns: dict = {}
    exec(compile(code, "<extractors docstring>", "exec"), ns)  # our own docstring
    assert len(ns["extractors"]) == len(ex.ALL_EXTRACTORS) == 4
    for e in ns["extractors"]:
        assert e.tool_name and e.extracted_type and callable(e.extract)
    assert callable(ns["stream"])


# ── skills remove prints the exact restore command (gh #157) ─────────


def test_skills_remove_prints_runnable_rollback(tmp_hermes_home: Path):
    lib = SkillLibrary(dirs=[tmp_hermes_home / "skills"])
    lib.write("goner", {"name": "goner", "description": "d"}, "# Goner\nsteps\n")

    runner = CliRunner()
    r = runner.invoke(cli, ["skills", "remove", "goner"])
    assert r.exit_code == 0, r.output
    marker = "langstage-hermes audit rollback goner "
    assert marker in r.output
    mutation_id = r.output.split(marker, 1)[1].split("`", 1)[0].strip()
    assert mutation_id.isdigit()

    # Followed verbatim (minus the program name), it restores the skill.
    rb = runner.invoke(cli, ["audit", "rollback", "goner", mutation_id], input="y\n")
    assert rb.exit_code == 0, rb.output
    assert SkillLibrary(dirs=[tmp_hermes_home / "skills"]).get("goner") is not None
