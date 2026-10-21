"""gh #185: a zero interval (``every 0m`` / ``0s`` / ``0h``) is rejected, not saved.

An interval of zero seconds has no next run, so the job used to be listed as
``scheduled`` and never fire. It is now a usage error, like ``every -5m``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.cron import jobs as cron_jobs
from langstage_hermes.cron.tool import _cronjob_impl

ZEROS = ["every 0m", "every 0s", "every 0h", "0m", "0d", "every 00m"]


@pytest.mark.parametrize("expr", ["0s", "0m", "0h", "0d", "00m"])
def test_parse_duration_rejects_zero(expr):
    with pytest.raises(ValueError, match="greater than zero"):
        cron_jobs.parse_duration(expr)


@pytest.mark.parametrize("expr", ZEROS)
def test_parse_schedule_rejects_zero_interval(expr):
    with pytest.raises(ValueError):
        cron_jobs.parse_schedule(expr)


def test_smallest_interval_still_works():
    assert cron_jobs.parse_schedule("every 1s")["seconds"] == 1
    assert cron_jobs.parse_schedule("1m")["seconds"] == 60


@pytest.mark.parametrize("expr", ["every 0m", "every 0s", "every 0h"])
def test_cli_create_zero_interval_exits_64_and_saves_nothing(tmp_hermes_home: Path, expr):
    r = CliRunner().invoke(cli, ["cron", "create", "--prompt", "x", "--schedule", expr])
    assert r.exit_code == 64, r.output
    assert "greater than zero" in r.output
    listed = json.loads(CliRunner().invoke(cli, ["cron", "list", "--json"]).output)
    assert listed["jobs"] == []


def test_cli_create_zero_interval_json(tmp_hermes_home: Path):
    r = CliRunner().invoke(cli, ["cron", "create", "--prompt", "x", "--schedule", "every 0m", "--json"])
    assert r.exit_code == 64, r.output
    assert "error" in json.loads(r.output)


def test_agent_tool_create_zero_interval_is_an_error(tmp_hermes_home: Path):
    out = _cronjob_impl(action="create", prompt="x", schedule="every 0m")
    assert out.startswith("Error creating cron job") and "greater than zero" in out
    assert cron_jobs.list_jobs() == []


def test_update_to_zero_interval_is_rejected(tmp_hermes_home: Path):
    job = cron_jobs.create_job(prompt="x", schedule="every 5m")
    with pytest.raises(ValueError, match="greater than zero"):
        cron_jobs.update_job(job["id"], schedule="every 0m")
    assert cron_jobs.get_job(job["id"])["schedule"]["seconds"] == 300
