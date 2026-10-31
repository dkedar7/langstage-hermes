"""gh #200: cron jobs keep their wall-clock time across a DST change.

The next run used to be computed at the anchor's fixed UTC offset
(``datetime.now().astimezone()``), so after DST ended a "9am" job ran at 08:00,
and after DST started at 10:00. Cron steps now happen in the local zone.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from langstage_hermes.cron import jobs as cron_jobs

NY = ZoneInfo("America/New_York")
EDT = timezone(timedelta(hours=-4))
EST = timezone(timedelta(hours=-5))


@pytest.fixture
def new_york(monkeypatch):
    monkeypatch.setattr(cron_jobs, "_local_zone", lambda: NY)


def _run(expr: str, base: datetime, n: int = 1) -> list[str]:
    """Chain ``n`` next runs the way mark_job_run does, each anchored on the last."""
    out = []
    job = {"schedule": {"kind": "cron", "expr": expr}}
    for _ in range(n):
        nxt = cron_jobs.compute_next_run({**job, "last_run_at": base.isoformat()})
        out.append(nxt.isoformat())
        base = nxt
    return out


def _wall(iso: str) -> str:
    return f"{datetime.fromisoformat(iso).astimezone(NY):%Y-%m-%d %H:%M %Z}"


# The issue's repro: last run stored with _now()'s fixed offset.
def test_nine_am_stays_nine_am_when_dst_ends(new_york):
    ran = datetime(2026, 10, 31, 9, 0, 3, tzinfo=EDT)
    assert _run("0 9 * * *", ran) == ["2026-11-01T09:00:00-05:00"]


def test_nine_am_stays_nine_am_when_dst_starts(new_york):
    ran = datetime(2027, 3, 13, 9, 0, 3, tzinfo=EST)
    assert _run("0 9 * * *", ran) == ["2027-03-14T09:00:00-04:00"]


def test_new_job_created_before_the_change_gets_the_right_first_run(new_york):
    # No completed run: create_job anchors on now.
    job = {"schedule": {"kind": "cron", "expr": "0 9 * * *"}}
    nxt = cron_jobs.compute_next_run(job, now=datetime(2026, 10, 31, 10, 0, tzinfo=EDT))
    assert nxt.isoformat() == "2026-11-01T09:00:00-05:00"


def test_weekday_job_across_the_change(new_york):
    ran = datetime(2026, 10, 30, 9, 0, 3, tzinfo=EDT)  # a Friday
    assert [_wall(x) for x in _run("0 9 * * MON-FRI", ran, 2)] == [
        "2026-11-02 09:00 EST",
        "2026-11-03 09:00 EST",
    ]


def test_fixed_time_in_the_repeated_hour_runs_once(new_york):
    # 01:30 happens twice on 2026-11-01; a daily 01:30 job must not fire twice.
    ran = datetime(2026, 10, 31, 1, 30, 2, tzinfo=EDT)
    assert _run("30 1 * * *", ran, 3) == [
        "2026-11-01T01:30:00-04:00",
        "2026-11-02T01:30:00-05:00",
        "2026-11-03T01:30:00-05:00",
    ]


def test_every_half_hour_runs_through_both_repeated_hours(new_york):
    ran = datetime(2026, 11, 1, 0, 45, tzinfo=EDT)
    assert _run("*/30 * * * *", ran, 6) == [
        "2026-11-01T01:00:00-04:00",
        "2026-11-01T01:30:00-04:00",
        "2026-11-01T01:00:00-05:00",
        "2026-11-01T01:30:00-05:00",
        "2026-11-01T02:00:00-05:00",
        "2026-11-01T02:30:00-05:00",
    ]


def test_time_skipped_when_dst_starts_runs_once_then_resumes(new_york):
    # 02:30 doesn't exist on 2027-03-14: it runs at 03:00, then 02:30 again.
    ran = datetime(2027, 3, 13, 2, 30, 2, tzinfo=EST)
    assert [_wall(x) for x in _run("30 2 * * *", ran, 3)] == [
        "2027-03-14 03:00 EDT",
        "2027-03-15 02:30 EDT",
        "2027-03-16 02:30 EDT",
    ]


def test_every_half_hour_through_the_skipped_hour(new_york):
    ran = datetime(2027, 3, 14, 1, 15, tzinfo=EST)
    assert [_wall(x) for x in _run("*/30 * * * *", ran, 3)] == [
        "2027-03-14 01:30 EST",
        "2027-03-14 03:00 EDT",
        "2027-03-14 03:30 EDT",
    ]


def test_interval_jobs_still_count_elapsed_time(new_york):
    job = {"schedule": {"kind": "interval", "seconds": 7200}, "last_run_at": "2026-11-01T00:30:00-04:00"}
    nxt = cron_jobs.compute_next_run(job)
    assert nxt.astimezone(UTC) == datetime(2026, 11, 1, 6, 30, tzinfo=UTC)


def test_through_mark_job_run_and_get_due_jobs(new_york, tmp_hermes_home: Path, monkeypatch):
    job = cron_jobs.create_job(prompt="morning report", schedule="0 9 * * *")
    monkeypatch.setattr(cron_jobs, "_now", lambda: datetime(2026, 10, 31, 9, 0, 3, tzinfo=EDT))
    cron_jobs.mark_job_run(job["id"], success=True)
    stored = cron_jobs.get_job(job["id"])["next_run_at"]
    assert stored == "2026-11-01T09:00:00-05:00"
    # Not due at 08:00 EST (13:00 UTC), due at 09:00 EST.
    assert not cron_jobs.get_due_jobs(now=datetime(2026, 11, 1, 8, 0, tzinfo=EST))
    assert [j["id"] for j in cron_jobs.get_due_jobs(now=datetime(2026, 11, 1, 9, 0, tzinfo=EST))] == [job["id"]]


def test_a_record_saved_by_the_old_code_fires_at_its_wall_time(new_york, tmp_hermes_home: Path):
    # A job saved before the fix holds 09:00 at the stale EDT offset (08:00 EST).
    job = cron_jobs.create_job(prompt="morning report", schedule="0 9 * * *")
    cron_jobs.update_job(job["id"], next_run_at="2026-11-01T09:00:00-04:00")
    assert cron_jobs.get_job(job["id"])["next_run_at"] == "2026-11-01T09:00:00-05:00"
    assert not cron_jobs.get_due_jobs(now=datetime(2026, 11, 1, 8, 0, tzinfo=EST))
    assert cron_jobs.get_due_jobs(now=datetime(2026, 11, 1, 9, 0, tzinfo=EST))


def test_a_correct_record_is_left_alone(new_york, tmp_hermes_home: Path):
    job = cron_jobs.create_job(prompt="night", schedule="30 1 * * *")
    for value in ("2026-11-01T01:30:00-04:00", "2026-11-01T01:30:00-05:00", "2026-11-02T01:30:00-05:00"):
        cron_jobs.update_job(job["id"], next_run_at=value)
        assert cron_jobs.get_job(job["id"])["next_run_at"] == value


def test_non_cron_records_are_left_alone(new_york, tmp_hermes_home: Path):
    job = cron_jobs.create_job(prompt="tick", schedule="every 2h")
    cron_jobs.update_job(job["id"], next_run_at="2026-11-01T09:00:00-04:00")
    assert cron_jobs.get_job(job["id"])["next_run_at"] == "2026-11-01T09:00:00-04:00"


def test_local_zone_is_a_real_zone():
    zone = cron_jobs._local_zone()
    assert zone is not None
    # A real zone gives different offsets across the year where it observes DST;
    # a fixed offset never would. Only assert it's usable as tzinfo.
    assert datetime(2026, 1, 1, tzinfo=zone).utcoffset() is not None


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset (POSIX)")
def test_without_zone_data_the_os_clock_still_keeps_wall_time(monkeypatch):
    monkeypatch.setattr(cron_jobs, "_local_zone", lambda: None)
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        ran = datetime(2026, 10, 31, 9, 0, 3, tzinfo=EDT)
        assert _run("0 9 * * *", ran) == ["2026-11-01T09:00:00-05:00"]
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
