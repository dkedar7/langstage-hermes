"""gh #97 — ``cron delete`` / ``pause`` / ``resume`` must exit 1 on a missing id.

The three cron mutation subcommands printed ``No cron job with id 'X'.`` but
exited 0, so a CI/script wrapper around the cron lifecycle could not tell a
typo'd or stale id from a real success — ``cron pause $ID && echo ok`` printed
``ok`` even when nothing was paused. Every other not-found path in the CLI
(``skills remove``, ``audit rollback``, ...) exits 1; these now do too, keeping
the same message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes.cli import cli


@pytest.mark.parametrize("subcommand", ["delete", "pause", "resume"])
def test_cron_mutation_missing_id_exits_1(subcommand: str, tmp_hermes_home: Path):
    r = CliRunner().invoke(cli, ["cron", subcommand, "no-such-job-123"])
    assert r.exit_code == 1, r.output
    assert "No cron job with id" in r.output


def test_cron_delete_existing_id_still_exits_0(tmp_hermes_home: Path):
    """The success path is unchanged — a real delete reports success and exits 0."""
    from langstage_hermes.cron import jobs as cron_jobs

    job = cron_jobs.create_job("do things", "every 5m")
    r = CliRunner().invoke(cli, ["cron", "delete", job["id"]])
    assert r.exit_code == 0, r.output
    assert "Deleted." in r.output


def test_cron_pause_then_resume_existing_id_exits_0(tmp_hermes_home: Path):
    from langstage_hermes.cron import jobs as cron_jobs

    job = cron_jobs.create_job("do things", "every 5m")

    paused = CliRunner().invoke(cli, ["cron", "pause", job["id"]])
    assert paused.exit_code == 0, paused.output
    assert "Paused." in paused.output

    resumed = CliRunner().invoke(cli, ["cron", "resume", job["id"]])
    assert resumed.exit_code == 0, resumed.output
    assert "Resumed." in resumed.output
