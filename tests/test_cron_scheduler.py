"""Tests for ``langstage_hermes.cron.scheduler.run_job`` — the agent path's
failure bookkeeping + delivery gating (gh #72).

A failed agent invoke must be recorded as a *failure* (``last_status="error"``)
and must **not** deliver the error string as if it were the job's result —
mirroring the ``no_agent`` (script) path. These tests monkeypatch the real
``create_hermes_agent`` seam so no live LLM is needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
