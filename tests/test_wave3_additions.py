"""Regression tests for the Wave 3 local-correctness fixes that extend existing suites.

Kept in one module so each can import its neighbour's fixtures without editing them:
search half-specified SCROLL (gh #135), failed one-shot cron retention (gh #137),
skill_manage(create) overwrite labelling (gh #148), audit diff newline handling
(gh #153), verify --json round-trip reason (gh #146), doctor aux provider package
(gh #158) and TOML-relative paths (gh #162).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.config import HermesConfig
from langstage_hermes.cron import jobs as cron_jobs
from langstage_hermes.skills.audit import SkillAuditLog
from langstage_hermes.skills.library import SkillLibrary
from langstage_hermes.skills.tools import _skill_manage_impl
from langstage_hermes.store.sqlite_fts import SqliteFtsStore

# ── search: --session / --around alone (gh #135) ───────────────────


@pytest.fixture
def search_home(tmp_hermes_home: Path) -> Path:
    store = SqliteFtsStore(db_path=str(tmp_hermes_home / "state.db"))
    store.ensure_session("sess-a", source="user", title="Docker setup chat")
    store.record_message("sess-a", "user", "set up docker compose for the api")
    store.close()
    return tmp_hermes_home


@pytest.mark.parametrize(
    "args",
    [
        ["search", "docker", "--session", "sess-a"],
        ["search", "docker", "--session", "DOES-NOT-EXIST"],
        ["search", "docker", "--session", "DOES-NOT-EXIST", "--json"],
        ["search", "--session", "sess-a"],
        ["search", "docker", "--around", "3"],
    ],
)
def test_session_or_around_alone_is_rejected_not_ignored(search_home, args):
    """--session / --around were silently dropped outside SCROLL, so a "scoped"
    query returned unscoped hits (even for a bogus session id) with exit 0. Given
    alone they are now a usage error naming the missing flag."""
    res = CliRunner().invoke(cli, args)
    assert res.exit_code == 2, res.output
    assert "requires" in res.output
    assert "BM25" not in res.output  # no DISCOVERY results were printed
    assert "Recent sessions" not in res.output  # nor a BROWSE listing


def test_full_scroll_still_works(search_home):
    res = CliRunner().invoke(cli, ["search", "--session", "sess-a", "--around", "1"])
    assert res.exit_code == 0, res.output


# ── cron: failed one-shot is kept (gh #137) ────────────────────────


def test_failed_one_shot_is_kept_with_its_error(tmp_hermes_home: Path):
    """A one-shot job whose only run FAILED was popped exactly like a successful
    one, taking last_status / last_error with it. It must stay visible."""
    job = cron_jobs.create_job("summarize inbox", "once at 2099-01-01T00:00", name="daily")
    cron_jobs.mark_job_run(job["id"], success=False, error="no API key")

    after = cron_jobs.get_job(job["id"])
    assert after is not None, "a failed one-shot must not vanish"
    assert after["last_status"] == "error"
    assert after["last_error"] == "no API key"
    assert after["state"] == "error"
    assert after["enabled"] is False
    assert after["next_run_at"] is None
    assert job["id"] not in [j["id"] for j in cron_jobs.get_due_jobs()]  # not re-fired
    assert job["id"] in [j["id"] for j in cron_jobs.list_jobs()]  # but still listed


def test_failed_one_shot_shows_in_cron_list_json(tmp_hermes_home: Path):
    job = cron_jobs.create_job("summarize inbox", "once at 2099-01-01T00:00", name="daily")
    cron_jobs.mark_job_run(job["id"], success=False, error="boom")
    res = CliRunner().invoke(cli, ["cron", "list", "--json"])
    assert res.exit_code == 0, res.output
    assert job["id"] in res.output


def test_successful_one_shot_is_still_removed(tmp_hermes_home: Path):
    job = cron_jobs.create_job("summarize inbox", "once at 2099-01-01T00:00", name="daily")
    cron_jobs.mark_job_run(job["id"], success=True)
    assert cron_jobs.get_job(job["id"]) is None


# ── skills: create-over-existing label + trailing newline (gh #148, #153) ──


@pytest.fixture
def audit_log(tmp_path: Path):
    (tmp_path / "skills").mkdir()
    log = SkillAuditLog(db_path=str(tmp_path / "state.db"))
    yield log
    log.close()


@pytest.fixture
def library(tmp_path: Path, audit_log: SkillAuditLog) -> SkillLibrary:
    return SkillLibrary(dirs=[tmp_path / "skills"], audit_log=audit_log)


def _create_via_agent(library: SkillLibrary, name: str, body: str) -> None:
    _skill_manage_impl(
        library,
        action="create",
        name=name,
        description=f"{name} skill",
        body=body,
        category="",
        old_str="",
        new_str="",
        frontmatter_data=None,
        tool_call_id="tc-create",
    )


def test_agent_create_over_existing_skill_is_logged_write_file(library, audit_log):
    """skill_manage(create) on an existing name overwrote it but was logged as
    `create`, indistinguishable from a first-time create (gh #148)."""
    _create_via_agent(library, "delta", "v1 body")
    _create_via_agent(library, "delta", "v2 body")
    rows = audit_log.list(skill_name="delta")
    assert [r.action for r in rows] == ["write_file", "create"]
    assert rows[0].before_content is not None
    assert rows[1].before_content is None


def test_written_skill_md_ends_with_newline(library):
    """Agent-authored SKILL.md files had no trailing newline (gh #153)."""
    path = library.write("eps", {"name": "eps", "description": "e"}, "last line without newline")
    assert path.read_bytes().endswith(b"\n")


def test_diff_keeps_each_line_on_its_own_line_without_trailing_newline():
    """A snapshot without a trailing newline ran the removed line straight into the
    added ones (gh #153). Each hunk line must be its own physical line."""
    from langstage_hermes.skills.audit import render_unified_diff

    diff = render_unified_diff(b"a\nlast", b"a\nlast\nextra\n", fromfile="x@1", tofile="x@disk")
    lines = diff.splitlines()
    assert "-last" in lines
    assert "\\ No newline at end of file" in lines
    assert "+last" in lines
    assert "+extra" in lines


def test_audit_diff_cli_legible_for_pre_fix_snapshot(tmp_hermes_home: Path):
    """`audit diff` against a mutation recorded before the fix (no trailing newline)."""
    from langstage_hermes.skills.audit import SkillAuditLog as _Log

    lib_log = _Log(db_path=str(tmp_hermes_home / "state.db"))
    try:
        lib = SkillLibrary(dirs=[tmp_hermes_home / "skills"], audit_log=lib_log)
        path = lib.write("zeta", {"name": "zeta", "description": "z"}, "- keep\n- tail")
        row = lib_log.list(skill_name="zeta")[0]
        # Simulate a pre-fix snapshot: the recorded content had no trailing newline.
        lib_log._conn.execute(
            "UPDATE skill_mutations SET after_content = ? WHERE id = ?",
            (row.after_content.rstrip(b"\n"), row.id),
        )
        lib_log._conn.commit()
        path.write_bytes(path.read_bytes() + b"- appended\n")
    finally:
        lib_log.close()

    res = CliRunner().invoke(cli, ["audit", "diff", "zeta", str(row.id)])
    assert res.exit_code == 0, res.output
    out_lines = res.output.splitlines()
    assert "-- tail" in out_lines
    assert "+- appended" in out_lines
    assert "- tail+" not in res.output


# ── verify --json round-trip reason (gh #146) ──────────────────────


def _isolate_models(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "LANGSTAGE_HERMES_MODEL_DEFAULT",
        "DEEPAGENT_HERMES_MODEL_DEFAULT",
        "LANGSTAGE_HERMES_MODEL_AUX",
        "DEEPAGENT_HERMES_MODEL_AUX",
        "LANGSTAGE_HERMES_HOME",
        "DEEPAGENT_HERMES_HOME",
    ):
        monkeypatch.delenv(k, raising=False)


def test_verify_json_round_trip_not_no_key_when_only_aux_key_missing(monkeypatch, tmp_path):
    """Primary key present, aux key missing: round_trip said "no key", contradicting
    model_key ("required key present") in the same payload."""
    _isolate_models(monkeypatch, tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-not-real")

    r = CliRunner().invoke(cli, ["verify", "--json"])
    assert r.exit_code == 2, r.output
    checks = {c["name"]: c for c in json.loads(r.output)["checks"]}
    assert checks["model_key"]["ok"] is True
    assert checks["model_key_aux"]["ok"] is False
    assert "no key" not in checks["round_trip"]["detail"]
    assert "preflight failed" in checks["round_trip"]["detail"]


def test_verify_json_round_trip_says_no_key_when_primary_key_missing(monkeypatch, tmp_path):
    _isolate_models(monkeypatch, tmp_path)
    r = CliRunner().invoke(cli, ["verify", "--json"])
    assert r.exit_code == 2, r.output
    checks = {c["name"]: c for c in json.loads(r.output)["checks"]}
    assert checks["model_key"]["ok"] is False
    assert checks["round_trip"]["detail"] == "skipped — no key"


# ── doctor: aux provider package (gh #158) ─────────────────────────


def test_doctor_fails_when_aux_provider_pkg_missing(monkeypatch, tmp_path):
    """anthropic main + openai aux with langchain_openai absent: doctor probed only
    the MAIN model's package and exited 0, then the reflection subagent crashed."""
    _isolate_models(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "anthropic:claude-sonnet-4-6")
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_AUX", "openai:openai/gpt-4o-mini")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name, *a, **kw: None if name == "langchain_openai" else real_find_spec(name, *a, **kw),
    )

    r = CliRunner().invoke(cli, ["doctor"])
    assert r.exit_code == 2, r.output
    assert "provider package 'langchain_openai' not importable for openai:openai/gpt-4o-mini (aux)" in r.output
    assert 'pip install "langstage-hermes[openai]"' in r.output

    r = CliRunner().invoke(cli, ["doctor", "--json"])
    assert r.exit_code == 2, r.output
    data = json.loads(r.output)
    assert data["ok"] is False
    aux = next(c for c in data["checks"] if c["name"] == "provider_pkg_aux")
    assert aux["ok"] is False


def test_doctor_reports_aux_provider_pkg_when_present(monkeypatch, tmp_path):
    _isolate_models(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_AUX", "openai:openai/gpt-4o-mini")
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a, **kw: object())

    r = CliRunner().invoke(cli, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "provider package (aux): langchain_openai installed" in r.output


# ── TOML-relative paths (gh #162) ──────────────────────────────────


def _strip_config_env(monkeypatch, tmp_path: Path) -> None:
    for k in (
        "LANGSTAGE_WORKSPACE_ROOT",
        "DEEPAGENT_WORKSPACE_ROOT",
        "LANGSTAGE_AGENT_SPEC",
        "DEEPAGENT_AGENT_SPEC",
        "LANGSTAGE_HERMES_SKILLS_EXTERNAL_DIRS",
        "DEEPAGENT_HERMES_SKILLS_EXTERNAL_DIRS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(tmp_path / "no_home"))
    monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(tmp_path / "no_core_home"))


def test_relative_paths_in_hermes_toml_resolve_against_the_toml_dir(monkeypatch, tmp_path):
    """Relative paths in langstage-hermes.toml resolved against the cwd, so running
    from a subdirectory silently pointed elsewhere. They now resolve against the
    toml's own directory, like langstage-core 1.0.36."""
    _strip_config_env(monkeypatch, tmp_path)
    proj = tmp_path / "proj"
    sub = proj / "deep" / "sub"
    sub.mkdir(parents=True)
    (proj / "langstage-hermes.toml").write_text(
        '[workspace]\nroot = "ws"\n[agent]\nspec = "agents/my.py:graph"\n'
        '[skills]\nexternal_dirs = ["shared-skills", "~/abs-skills"]\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(sub)

    cfg = HermesConfig.resolve(toml_start=sub)
    assert Path(cfg.workspace_root) == proj / "ws"
    assert cfg.agent_spec == f"{proj / 'agents' / 'my.py'}:graph"
    assert cfg.skills_external_dirs[0] == str(proj / "shared-skills")
    assert cfg.skills_external_dirs[1] == str(Path("~/abs-skills").expanduser())
    assert all(isinstance(d, str) for d in cfg.skills_external_dirs)
    assert cfg.toml_dir_for("workspace_root") == proj
    assert cfg.toml_dir_for("skills_external_dirs") == proj


def test_relative_paths_in_deepagents_toml_resolve_against_the_toml_dir(monkeypatch, tmp_path):
    _strip_config_env(monkeypatch, tmp_path)
    proj = tmp_path / "proj"
    sub = proj / "sub"
    sub.mkdir(parents=True)
    (proj / "deepagents.toml").write_text('[workspace]\nroot = "ws"\n', encoding="utf-8")
    monkeypatch.chdir(sub)

    cfg = HermesConfig.resolve(toml_start=sub)
    assert Path(cfg.workspace_root) == proj / "ws"
    assert cfg.toml_dir_for("workspace_root") == proj


def test_env_paths_stay_cwd_relative(monkeypatch, tmp_path):
    _strip_config_env(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGSTAGE_HERMES_SKILLS_EXTERNAL_DIRS", "rel-skills")
    cfg = HermesConfig.resolve(toml_start=tmp_path)
    assert cfg.skills_external_dirs == ["rel-skills"]
    assert cfg.toml_dir_for("skills_external_dirs") is None
