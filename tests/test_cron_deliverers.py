"""Tests for the pluggable cron deliverer pattern + bundled deliverers.

Covers the :class:`Deliverer` ABC + registry, the three bundled
deliverers (``local`` / ``stdout`` / ``agentmail``), and the
``[SILENT]`` suppression contract enforced by ``scheduler._deliver_output``.

No real network calls — ``requests.post`` is mocked in every AgentMail
test so this suite is safe to run offline / in CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest

from deepagent_hermes.cron import deliverers as deliverer_mod
from deepagent_hermes.cron import scheduler as scheduler_mod
from deepagent_hermes.cron.deliverers import (
    AgentMailDeliverer,
    Deliverer,
    LocalDeliverer,
    StdoutDeliverer,
    get_deliverer,
    register_deliverer,
)

# ── helpers ─────────────────────────────────────────────────────────


def _job(**overrides: Any) -> dict[str, Any]:
    """Minimal job dict for deliverer tests."""
    base: dict[str, Any] = {
        "id": "abc123",
        "name": "test-job",
        "deliver": "local",
    }
    base.update(overrides)
    return base


# ── LocalDeliverer ──────────────────────────────────────────────────


def test_local_deliverer_noop(tmp_path: Path):
    """LocalDeliverer is a no-op even if output_path already exists."""
    output_path = tmp_path / "out.md"
    output_path.write_text("already on disk")
    # Must not raise, must not modify or touch the file.
    LocalDeliverer().deliver(_job(), "anything", output_path=output_path)
    assert output_path.read_text() == "already on disk"


# ── StdoutDeliverer ─────────────────────────────────────────────────


def test_stdout_deliverer_prints_banner(capsys):
    """StdoutDeliverer prints a recognizable banner before the output."""
    job = _job(id="xyz789", name="my-job")
    StdoutDeliverer().deliver(job, "hello world", output_path=None)
    captured = capsys.readouterr()
    assert "=== Cron job: my-job (xyz789) ===" in captured.out
    assert "hello world" in captured.out


# ── AgentMailDeliverer ──────────────────────────────────────────────


def test_agentmail_deliverer_requires_api_key(monkeypatch):
    """Without AGENTMAIL_API_KEY in env, deliver() raises immediately."""
    monkeypatch.delenv("AGENTMAIL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AGENTMAIL_API_KEY"):
        AgentMailDeliverer().deliver(_job(deliver="agentmail"), "body", output_path=None)


def test_agentmail_deliverer_posts_to_endpoint(monkeypatch):
    """deliver() POSTs to the AgentMail endpoint with Bearer auth + correct payload."""
    monkeypatch.setenv("AGENTMAIL_API_KEY", "test-key-123")

    fake_resp = MagicMock(status_code=200, text="{}")
    with patch("requests.post", return_value=fake_resp) as mock_post:
        job = _job(
            deliver="agentmail",
            name="weekly-digest",
            deliver_to="someone@example.com",
        )
        AgentMailDeliverer().deliver(job, "## Digest\n- item 1", output_path=None)

    assert mock_post.call_count == 1
    args, kwargs = mock_post.call_args

    # URL is positional; verify it.
    url = args[0] if args else kwargs.get("url")
    assert url == "https://api.agentmail.to/v0/inboxes/kzest@agentmail.to/messages/send"

    # Headers carry Bearer auth + JSON content type.
    headers = kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-key-123"
    assert headers["Content-Type"] == "application/json"

    # JSON body has the canonical AgentMail shape.
    payload = kwargs["json"]
    assert payload["subject"] == "[deepagent-hermes] weekly-digest"
    assert payload["text"] == "## Digest\n- item 1"
    assert payload["to"] == ["someone@example.com"]


def test_agentmail_deliverer_uses_default_recipient(monkeypatch):
    """When job['deliver_to'] is absent, falls back to Kedar's known address."""
    monkeypatch.setenv("AGENTMAIL_API_KEY", "test-key-123")

    fake_resp = MagicMock(status_code=200, text="{}")
    with patch("requests.post", return_value=fake_resp) as mock_post:
        AgentMailDeliverer().deliver(_job(deliver="agentmail"), "body", output_path=None)

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["to"] == ["kdabhadk@gmail.com"]


def test_agentmail_deliverer_raises_on_4xx(monkeypatch):
    """A 4xx response surfaces as a RuntimeError carrying the response body."""
    monkeypatch.setenv("AGENTMAIL_API_KEY", "bad-key")

    fake_resp = MagicMock(status_code=401, text='{"error": "unauthorized"}')
    with patch("requests.post", return_value=fake_resp):
        with pytest.raises(RuntimeError, match="HTTP 401"):
            AgentMailDeliverer().deliver(_job(deliver="agentmail"), "body", output_path=None)


# ── scheduler SILENT_MARKER integration ────────────────────────────


def test_agentmail_silent_marker_suppression_in_scheduler(monkeypatch):
    """Scheduler's _deliver_output skips ALL deliverers when output starts with [SILENT]."""
    monkeypatch.setenv("AGENTMAIL_API_KEY", "test-key-123")

    calls: list[tuple[dict[str, Any], str]] = []

    class _Recording(Deliverer):
        name: ClassVar[str] = "recording-silent-test"

        def deliver(self, job, output, *, output_path):
            calls.append((job, output))

    register_deliverer(_Recording)
    try:
        job = _job(deliver="recording-silent-test", name="silent-job")
        scheduler_mod._deliver_output(
            job,
            f"{scheduler_mod.SILENT_MARKER} nothing to report",
            None,
        )
    finally:
        # Clean up registry pollution between tests.
        deliverer_mod._REGISTRY.pop("recording-silent-test", None)

    assert calls == [], "deliverer should not be invoked when output starts with [SILENT]"


# ── registry ────────────────────────────────────────────────────────


def test_register_get_deliverer_round_trip():
    """A custom Deliverer registers and is retrievable by name."""

    class _Custom(Deliverer):
        name: ClassVar[str] = "custom-roundtrip-test"

        def deliver(self, job, output, *, output_path):
            return None

    register_deliverer(_Custom)
    try:
        fetched = get_deliverer("custom-roundtrip-test")
        assert fetched is _Custom
    finally:
        deliverer_mod._REGISTRY.pop("custom-roundtrip-test", None)


def test_unknown_deliverer_falls_back_to_local_with_warning(caplog):
    """Unknown deliverer name → warning log + LocalDeliverer fallback (no crash)."""
    job = _job(deliver="nope-not-real")
    with caplog.at_level(logging.WARNING, logger="deepagent_hermes.cron.scheduler"):
        # Must not raise; LocalDeliverer is a no-op.
        scheduler_mod._deliver_output(job, "some output", None)
    assert any("no deliverer registered for 'nope-not-real'" in rec.getMessage() for rec in caplog.records)


# ── bundled registry sanity ────────────────────────────────────────


def test_bundled_deliverers_are_pre_registered():
    """The three bundled deliverers ship registered out of the box."""
    assert get_deliverer("local") is LocalDeliverer
    assert get_deliverer("stdout") is StdoutDeliverer
    assert get_deliverer("agentmail") is AgentMailDeliverer
