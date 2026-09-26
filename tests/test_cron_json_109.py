"""gh #109 — ``--json`` on the ``cron`` subcommands.

The scheduler is the subsystem you'd actually build automation *around*, yet it
was the one automation surface with no machine-readable output: monitoring which
jobs exist, their state / next run, or what a tick executed meant scraping a
fixed-width text table. These tests pin the fix — ``list`` / ``create`` /
``run-due`` / ``delete`` / ``pause`` / ``resume`` now emit one JSON object with
stable keys, mirroring the existing ``--json`` convention — while preserving the
not-found exit codes a prior fix established (gh #97): ``delete`` / ``pause`` /
``resume`` still exit 1 on a missing id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli

# ── cron list --json ────────────────────────────────────────────────────


def test_cron_list_json_empty_is_clean(tmp_hermes_home: Path):
    r = CliRunner().invoke(cli, ["cron", "list", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output) == {"jobs": [], "count": 0}


def test_cron_list_json_shape(tmp_hermes_home: Path):
    from langstage_hermes.cron import jobs as cron_jobs

    job = cron_jobs.create_job("do things", "every 5m", name="nightly", model="anthropic:claude-haiku-4-5-20251001")
    r = CliRunner().invoke(cli, ["cron", "list", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["count"] == len(data["jobs"]) == 1
    row = data["jobs"][0]
    assert set(row) == {"id", "name", "schedule", "state", "next_run", "last_run", "last_status", "model"}
    assert row["id"] == job["id"]
    assert row["name"] == "nightly"
    assert row["schedule"] == "every 5m"
    assert row["state"] == "scheduled"
    assert row["next_run"]  # populated
    assert row["last_run"] is None
    assert row["model"] == "anthropic:claude-haiku-4-5-20251001"


def test_cron_list_human_output_unchanged(tmp_hermes_home: Path):
    from langstage_hermes.cron import jobs as cron_jobs

    cron_jobs.create_job("do things", "every 5m", name="nightly")
    r = CliRunner().invoke(cli, ["cron", "list"])
    assert r.exit_code == 0, r.output
    assert "nightly" in r.output
    assert "state=scheduled" in r.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(r.output)


# ── cron create --json ──────────────────────────────────────────────────


def test_cron_create_json_shape(tmp_hermes_home: Path):
    r = CliRunner().invoke(cli, ["cron", "create", "--prompt", "summarize inbox", "--schedule", "every 2h", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert set(data) == {"id", "name", "schedule", "next_run"}
    assert data["name"] == "summarize inbox"
    # `schedule` is the canonical display form (exact, largest whole unit; gh #149).
    assert data["schedule"] == "every 2h"
    assert data["next_run"]
    # it really persisted — list sees it
    listed = json.loads(CliRunner().invoke(cli, ["cron", "list", "--json"]).output)
    assert any(j["id"] == data["id"] for j in listed["jobs"])


def test_cron_create_json_invalid_schedule_exits_64(tmp_hermes_home: Path):
    r = CliRunner().invoke(cli, ["cron", "create", "--prompt", "x", "--schedule", "not-a-schedule", "--json"])
    assert r.exit_code == 64, r.output
    assert "error" in json.loads(r.output)


# ── cron run-due --json ─────────────────────────────────────────────────


def test_cron_run_due_json_none_due(tmp_hermes_home: Path):
    r = CliRunner().invoke(cli, ["cron", "run-due", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output) == {"ran": [], "count": 0}


def test_cron_run_due_json_reports_what_ran(tmp_hermes_home: Path, monkeypatch):
    """The tick summary carries id/name/status/error per job — the thing that makes
    an external tick loop observable. We stub the execution so the shape is tested
    without building a live agent."""
    import langstage_hermes.cron.scheduler as sched
    from langstage_hermes.cron import jobs as cron_jobs

    job = cron_jobs.create_job("do a thing", "every 5m", name="nightly")
    monkeypatch.setattr(
        sched.HermesCron,
        "tick",
        lambda self: [
            {"job_id": job["id"], "success": True, "output_path": None, "silent": False, "error": None, "delivery_error": None}
        ],
    )
    r = CliRunner().invoke(cli, ["cron", "run-due", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["count"] == 1
    ran = data["ran"][0]
    assert set(ran) == {"id", "name", "status", "error"}
    assert ran["id"] == job["id"]
    assert ran["name"] == "nightly"  # best-effort lookup filled the name in
    assert ran["status"] == "ok"
    assert ran["error"] is None


def test_cron_run_due_human_output_unchanged(tmp_hermes_home: Path):
    r = CliRunner().invoke(cli, ["cron", "run-due"])
    assert r.exit_code == 0, r.output
    assert "Tick complete: 0 job(s) run." in r.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(r.output)


# ── cron delete / pause / resume --json (exit codes preserved, gh #97) ──


@pytest.mark.parametrize("subcommand", ["delete", "pause", "resume"])
def test_cron_mutation_json_missing_id_exits_1(subcommand: str, tmp_hermes_home: Path):
    r = CliRunner().invoke(cli, ["cron", subcommand, "no-such-job-123", "--json"])
    assert r.exit_code == 1, r.output  # the gh #97 not-found exit is preserved
    data = json.loads(r.output)
    assert data == {"id": "no-such-job-123", "action": subcommand, "ok": False, "error": "not found"}


def test_cron_delete_json_success(tmp_hermes_home: Path):
    from langstage_hermes.cron import jobs as cron_jobs

    job = cron_jobs.create_job("do things", "every 5m")
    r = CliRunner().invoke(cli, ["cron", "delete", job["id"], "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output) == {"id": job["id"], "action": "delete", "ok": True}


def test_cron_pause_then_resume_json_success(tmp_hermes_home: Path):
    from langstage_hermes.cron import jobs as cron_jobs

    job = cron_jobs.create_job("do things", "every 5m")

    paused = CliRunner().invoke(cli, ["cron", "pause", job["id"], "--json"])
    assert paused.exit_code == 0, paused.output
    assert json.loads(paused.output) == {"id": job["id"], "action": "pause", "ok": True}

    resumed = CliRunner().invoke(cli, ["cron", "resume", job["id"], "--json"])
    assert resumed.exit_code == 0, resumed.output
    assert json.loads(resumed.output) == {"id": job["id"], "action": "resume", "ok": True}
