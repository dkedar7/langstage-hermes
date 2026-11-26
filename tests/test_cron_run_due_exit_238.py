"""gh #238 — ``cron run-due`` exits 1 when a job in the tick failed.

``run-due`` is what an external tick loop calls (crontab, a systemd timer, a CI
schedule), and those read only the exit code. A failed job used to show up in the
text / JSON / ``last_status`` but the command exited 0, so nothing ever alerted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli


def _result(job_id: str, ok: bool) -> dict:
    return {
        "job_id": job_id,
        "success": ok,
        "output_path": None,
        "silent": False,
        "error": None if ok else "[agent invoke failed: Connection error.]",
        "delivery_error": None,
    }


def _stub_tick(monkeypatch, results):
    import langstage_hermes.cron.scheduler as sched

    monkeypatch.setattr(sched.HermesCron, "tick", lambda self: results)


@pytest.mark.parametrize("as_json", [False, True])
def test_failed_job_fails_the_tick(tmp_hermes_home: Path, monkeypatch, as_json: bool):
    from langstage_hermes.cron import jobs as cron_jobs

    good = cron_jobs.create_job("a", "every 5m", name="good")
    bad = cron_jobs.create_job("b", "every 5m", name="bad")
    _stub_tick(monkeypatch, [_result(good["id"], True), _result(bad["id"], False)])

    args = ["cron", "run-due"] + (["--json"] if as_json else [])
    r = CliRunner().invoke(cli, args)
    assert r.exit_code == 1, r.output
    # stdout keeps the per-job detail
    if as_json:
        data = json.loads(r.stdout)
        assert data["count"] == 2
        assert [row["status"] for row in data["ran"]] == ["ok", "error"]
    else:
        assert "Tick complete: 2 job(s) run." in r.stdout
        assert f"{bad['id']}: error" in r.stdout
    # one-line summary on stderr naming the failed job
    assert r.stderr.strip() == f"error: 1 of 2 cron job(s) failed: {bad['id']}"


@pytest.mark.parametrize("as_json", [False, True])
def test_all_ok_tick_exits_0(tmp_hermes_home: Path, monkeypatch, as_json: bool):
    from langstage_hermes.cron import jobs as cron_jobs

    job = cron_jobs.create_job("a", "every 5m", name="good")
    _stub_tick(monkeypatch, [_result(job["id"], True)])
    r = CliRunner().invoke(cli, ["cron", "run-due"] + (["--json"] if as_json else []))
    assert r.exit_code == 0, r.output
    assert r.stderr == ""


@pytest.mark.parametrize("as_json", [False, True])
def test_nothing_due_exits_0(tmp_hermes_home: Path, as_json: bool):
    r = CliRunner().invoke(cli, ["cron", "run-due"] + (["--json"] if as_json else []))
    assert r.exit_code == 0, r.output


class _RaisingAgent:
    def invoke(self, *a, **k):
        raise RuntimeError("Connection error.")


def test_real_failing_job_exits_1(tmp_hermes_home: Path, monkeypatch):
    """Through the real scheduler and run_job: a due job whose agent invoke raises is
    recorded as an error and fails the tick (only the agent build is stubbed)."""
    from unittest.mock import patch

    import langstage_hermes.cron.scheduler as sched
    from langstage_hermes.cron import jobs as cron_jobs

    job = cron_jobs.create_job("say hi", "every 5m", name="quick")
    monkeypatch.setattr(cron_jobs, "get_due_jobs", lambda **k: [cron_jobs.get_job(job["id"])])
    with (
        patch("langstage_hermes.agent.create_hermes_agent", lambda *a, **k: _RaisingAgent()),
        patch.object(sched, "_deliver_output"),
    ):
        r = CliRunner().invoke(cli, ["cron", "run-due"])
    assert r.exit_code == 1, r.output
    assert f"{job['id']}: error" in r.stdout
    assert r.stderr.strip() == f"error: 1 of 1 cron job(s) failed: {job['id']}"
    assert cron_jobs.get_job(job["id"])["last_status"] == "error"
