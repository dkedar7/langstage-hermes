"""Tests for ``langstage_hermes.cron.scheduler.run_job`` — the agent path's
failure bookkeeping + delivery gating (gh #72).

A failed agent invoke must be recorded as a *failure* (``last_status="error"``)
and must **not** deliver the error string as if it were the job's result —
mirroring the ``no_agent`` (script) path. These tests monkeypatch the real
``create_hermes_agent`` seam so no live LLM is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from langstage_hermes.cron import jobs as cron_jobs
from langstage_hermes.cron import scheduler


class _RaisingAgent:
    """Stand-in agent whose ``invoke`` raises — models an expired key / outage."""

    def invoke(self, *_args, **_kwargs):
        raise RuntimeError("Could not resolve authentication method")


class _OkAgent:
    """Stand-in agent returning a normal final AIMessage."""

    def __init__(self, text: str = "your morning brief") -> None:
        self._text = text

    def invoke(self, *_args, **_kwargs):
        return {"messages": [SimpleNamespace(content=self._text)]}


def test_failed_agent_invoke_records_failure_and_suppresses_delivery(tmp_hermes_home: Path):
    """A raised agent invoke → success=False, last_status='error', no delivery (gh #72)."""
    job = cron_jobs.create_job("write my morning brief", "every 1m", name="brief")

    with (
        patch("langstage_hermes.agent.create_hermes_agent", lambda *a, **k: _RaisingAgent()),
        patch.object(scheduler, "_deliver_output") as mock_deliver,
    ):
        result = scheduler.run_job(job)

    # 1. The tick result reports a failure (so `run-due` prints '<id>: error').
    assert result["success"] is False
    assert result["error"] and "agent invoke failed" in result["error"]

    # 2. The error string is NOT delivered as the job's output.
    mock_deliver.assert_not_called()

    # 3. Bookkeeping in jobs.json records the failure — not a phantom success.
    rec = cron_jobs.get_job(job["id"])
    assert rec is not None
    assert rec["last_status"] == "error"
    assert rec["last_error"] and "agent invoke failed" in rec["last_error"]

    # 4. The output doc is still saved (parity with the script path), carrying
    #    the error as an audit trail — it just isn't delivered.
    assert result["output_path"] is not None
    saved = Path(result["output_path"]).read_text(encoding="utf-8")
    assert "agent invoke failed" in saved


def test_successful_agent_invoke_delivers_and_records_ok(tmp_hermes_home: Path):
    """Guard the new (ok, body) contract: a good invoke still delivers + records ok."""
    job = cron_jobs.create_job("write my morning brief", "every 1m", name="brief")

    with (
        patch("langstage_hermes.agent.create_hermes_agent", lambda *a, **k: _OkAgent("your morning brief")),
        patch.object(scheduler, "_deliver_output") as mock_deliver,
    ):
        result = scheduler.run_job(job)

    assert result["success"] is True
    assert result["error"] is None
    # The real content — not an error sentinel — is delivered exactly once.
    mock_deliver.assert_called_once()
    delivered_body = mock_deliver.call_args.args[1]
    assert delivered_body == "your morning brief"

    rec = cron_jobs.get_job(job["id"])
    assert rec is not None
    assert rec["last_status"] == "ok"
    assert rec["last_error"] is None


def test_failed_job_logs_clean_one_line_by_default(
    tmp_hermes_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Default path: a failed job logs ONE clean line, no ~126-line traceback (gh #111).

    ``cron run-due`` / the daemon are automation surfaces; the failure is already
    captured cleanly in ``last_status``/``error``, so the raw traceback is pure
    duplicate noise. Assert the record is a single-line ``WARNING`` with no
    ``exc_info`` attached (nothing for a formatter to expand into a stack).
    """
    monkeypatch.delenv("LANGSTAGE_DEBUG", raising=False)
    job = cron_jobs.create_job("write my morning brief", "every 1m", name="brief")

    with (
        patch("langstage_hermes.agent.create_hermes_agent", lambda *a, **k: _RaisingAgent()),
        patch.object(scheduler, "_deliver_output"),
        caplog.at_level(logging.DEBUG, logger="langstage_hermes.cron.scheduler"),
    ):
        result = scheduler.run_job(job)

    # Failure bookkeeping is unchanged — only the console noise differs.
    assert result["success"] is False

    failures = [r for r in caplog.records if "agent invoke failed" in r.getMessage()]
    assert len(failures) == 1
    rec = failures[0]
    assert rec.levelno == logging.WARNING
    # No traceback attached to the record...
    assert rec.exc_info is None
    # ...and even fully formatted it is a single clean line with the concise cause.
    formatted = logging.Formatter().format(rec)
    assert "Traceback (most recent call last)" not in formatted
    assert "\n" not in formatted
    assert "Could not resolve authentication method" in formatted


def test_failed_job_preserves_full_traceback_under_debug(
    tmp_hermes_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """LANGSTAGE_DEBUG=1: the full traceback is preserved at ERROR (gh #111)."""
    monkeypatch.setenv("LANGSTAGE_DEBUG", "1")
    job = cron_jobs.create_job("write my morning brief", "every 1m", name="brief")

    with (
        patch("langstage_hermes.agent.create_hermes_agent", lambda *a, **k: _RaisingAgent()),
        patch.object(scheduler, "_deliver_output"),
        caplog.at_level(logging.DEBUG, logger="langstage_hermes.cron.scheduler"),
    ):
        result = scheduler.run_job(job)

    assert result["success"] is False

    failures = [r for r in caplog.records if "agent invoke failed" in r.getMessage()]
    assert len(failures) == 1
    rec = failures[0]
    assert rec.levelno == logging.ERROR
    # exc_info is attached → the formatter renders the multi-line stack.
    assert rec.exc_info is not None
    formatted = logging.Formatter().format(rec)
    assert "Traceback (most recent call last)" in formatted
    assert "RuntimeError" in formatted


def _clear_debug_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LANGSTAGE_DEBUG", "DEEPAGENT_DEBUG"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    "source",
    ["env:LANGSTAGE_DEBUG", "env:DEEPAGENT_DEBUG", "toml:langstage-hermes.toml", "toml:langstage.toml"],
)
def test_debug_follows_the_resolved_config(
    tmp_hermes_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
):
    """The cron traceback switch is the resolved ``debug``, not a raw LANGSTAGE_DEBUG read.

    The legacy ``DEEPAGENT_DEBUG`` and a TOML ``debug = true`` (hermes or cross-host
    file) set ``HostConfig.debug`` but were ignored by the scheduler, which read only
    ``LANGSTAGE_DEBUG`` from the environment.
    """
    _clear_debug_env(monkeypatch)
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    monkeypatch.setenv("LANGSTAGE_CONFIG_HOME", str(tmp_path / "core_global"))
    assert scheduler._debug_enabled() is False

    kind, where = source.split(":", 1)
    if kind == "env":
        monkeypatch.setenv(where, "1")
    else:
        (proj / where).write_text("debug = true\n", encoding="utf-8")
    assert scheduler._debug_enabled() is True


def test_debug_lookup_failure_falls_back_to_the_env_switch(tmp_hermes_home: Path, monkeypatch: pytest.MonkeyPatch):
    """A config-resolution error must never mask the job failure being logged."""
    from langstage_hermes.config import HermesConfig

    def boom(*a, **k):
        raise RuntimeError("config exploded")

    _clear_debug_env(monkeypatch)
    monkeypatch.setattr(HermesConfig, "resolve", classmethod(boom))
    assert scheduler._debug_enabled() is False
    monkeypatch.setenv("LANGSTAGE_DEBUG", "1")
    assert scheduler._debug_enabled() is True
