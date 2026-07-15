"""Tests for ``langstage_hermes.cron.jobs`` — CRUD + schedule parsing."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from langstage_hermes.cron import jobs as cron_jobs


def test_parse_schedule_interval():
    """'every 30m' → interval kind, 1800s."""
    s = cron_jobs.parse_schedule("every 30m")
    assert s["kind"] == "interval"
    assert s["seconds"] == 30 * 60


def test_parse_schedule_bare_duration_is_recurring_interval():
    """Bare '30m' is a recurring interval — identical to 'every 30m' (gh #71).

    The docstring / tool example / error-hint all present a bare duration as a
    recurring interval; it must not silently parse to a fire-once one-shot.
    """
    bare = cron_jobs.parse_schedule("30m")
    every = cron_jobs.parse_schedule("every 30m")
    assert bare["kind"] == "interval"
    assert bare["seconds"] == 30 * 60
    # A bare duration and its "every" alias must mean exactly the same thing.
    assert bare["kind"] == every["kind"]
    assert bare["seconds"] == every["seconds"]
    assert bare["display"] == every["display"]


def test_parse_schedule_bare_durations_all_recurring():
    """Every documented bare-duration form parses to a recurring interval (gh #71)."""
    for expr, secs in [("30s", 30), ("5m", 300), ("2h", 7200), ("1d", 86400)]:
        s = cron_jobs.parse_schedule(expr)
        assert s["kind"] == "interval", f"{expr!r} should be a recurring interval"
        assert s["seconds"] == secs


def test_parse_schedule_cron():
    """5-field cron expression parses + validates via croniter."""
    s = cron_jobs.parse_schedule("0 9 * * *")
    assert s["kind"] == "cron"
    assert s["expr"] == "0 9 * * *"


def test_parse_schedule_once_at_iso():
    """'once at 2026-06-15T09:00' parses to a future one-shot."""
    s = cron_jobs.parse_schedule("once at 2026-06-15T09:00")
    assert s["kind"] == "once"
    assert "2026-06-15" in s["run_at"]


def test_parse_schedule_rejects_garbage():
    """Unparseable input raises ValueError with hints."""
    with pytest.raises(ValueError):
        cron_jobs.parse_schedule("whenever I feel like it")


def test_parse_duration_units():
    assert cron_jobs.parse_duration("30s") == 30
    assert cron_jobs.parse_duration("5m") == 300
    assert cron_jobs.parse_duration("2h") == 7200
    assert cron_jobs.parse_duration("1d") == 86400


def test_create_and_list_job(tmp_hermes_home: Path):
    """create_job persists a job; list_jobs returns it."""
    job = cron_jobs.create_job("ping", "1m", name="smoke")
    assert job["name"] == "smoke"
    assert job["prompt"] == "ping"
    # A bare '1m' schedule is a recurring interval (gh #71).
    assert job["schedule"]["kind"] == "interval"
    listed = cron_jobs.list_jobs()
    assert any(j["id"] == job["id"] for j in listed)


def test_bare_duration_job_refires_and_does_not_retire(tmp_hermes_home: Path):
    """A bare-duration job keeps rescheduling after it fires (gh #71).

    Pre-fix, '1m' parsed to a one-shot: after the first run ``next_run_at``
    went ``None`` and the job flipped to ``completed`` (retired). As a recurring
    interval it must stay ``scheduled`` with a fresh ``next_run_at``.
    """
    job = cron_jobs.create_job("say hi", "1m", name="recurring")
    assert job["schedule"]["kind"] == "interval"
    # One-shots auto-set repeat=1; a recurring interval must not.
    assert job["repeat"]["times"] is None

    cron_jobs.mark_job_run(job["id"], success=True)

    after = cron_jobs.get_job(job["id"])
    assert after is not None, "recurring job must not be auto-deleted after one run"
    assert after["state"] == "scheduled"
    assert after["next_run_at"] is not None, "interval job must reschedule its next run"


def test_compute_next_run_about_one_minute_out(tmp_hermes_home: Path):
    """For schedule='1m', compute_next_run lands ~1 minute from now."""
    job = cron_jobs.create_job("ping", "1m")
    next_run = cron_jobs.compute_next_run(job)
    assert isinstance(next_run, datetime)
    delta = next_run - datetime.now().astimezone()
    # '1m' is a recurring interval (gh #71); with no last_run_at yet the next
    # run is ~1 minute out — in the future and within ~120s of "now".
    assert -timedelta(seconds=5) <= delta <= timedelta(seconds=120)


def test_get_job_round_trip(tmp_hermes_home: Path):
    job = cron_jobs.create_job("a", "5m")
    fetched = cron_jobs.get_job(job["id"])
    assert fetched is not None
    assert fetched["id"] == job["id"]
    assert cron_jobs.get_job("does-not-exist") is None


def test_pause_and_resume(tmp_hermes_home: Path):
    job = cron_jobs.create_job("a", "every 5m")
    assert cron_jobs.pause_job(job["id"], reason="testing")
    paused = cron_jobs.get_job(job["id"])
    assert paused["state"] == "paused"
    assert paused["enabled"] is False
    assert cron_jobs.resume_job(job["id"])
    resumed = cron_jobs.get_job(job["id"])
    assert resumed["state"] == "scheduled"
    assert resumed["enabled"] is True


def test_delete_job(tmp_hermes_home: Path):
    job = cron_jobs.create_job("a", "5m")
    assert cron_jobs.delete_job(job["id"])
    assert cron_jobs.get_job(job["id"]) is None
    assert cron_jobs.delete_job(job["id"]) is False


def test_no_agent_requires_script(tmp_hermes_home: Path):
    """no_agent=True without a script is a clear ValueError at create time."""
    with pytest.raises(ValueError):
        cron_jobs.create_job(None, "5m", no_agent=True)


def test_update_job_rejects_id_change(tmp_hermes_home: Path):
    job = cron_jobs.create_job("a", "5m")
    with pytest.raises(ValueError):
        cron_jobs.update_job(job["id"], id="hijack")


def test_job_shape_has_all_spec_fields(tmp_hermes_home: Path):
    """Sanity: the created job dict carries every SPEC §14 field."""
    job = cron_jobs.create_job("hi", "5m", name="x")
    required = {
        "id",
        "name",
        "prompt",
        "skills",
        "skill",
        "model",
        "provider",
        "base_url",
        "script",
        "no_agent",
        "context_from",
        "schedule",
        "schedule_display",
        "repeat",
        "enabled",
        "state",
        "paused_at",
        "paused_reason",
        "created_at",
        "next_run_at",
        "last_run_at",
        "last_status",
        "last_error",
        "last_delivery_error",
        "deliver",
        "origin",
        "enabled_toolsets",
        "workdir",
        "profile",
    }
    assert required.issubset(job.keys())


def test_job_output_dir_rejects_escape(tmp_hermes_home: Path):
    """Path-escape attempts in job IDs are rejected."""
    for bad in ("..", "../escape", "with/slash", "back\\slash"):
        with pytest.raises(ValueError):
            cron_jobs.job_output_dir(bad)
