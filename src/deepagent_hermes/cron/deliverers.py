"""Pluggable cron job output deliverers.

The cron scheduler writes every job's output to disk; *delivering* the
output is a separate, swappable concern. This module defines the
:class:`Deliverer` ABC + a module-level registry so callers (and tests,
plugins, future channels) can plug in additional delivery channels
without modifying ``scheduler.py``.

Bundled deliverers
------------------

- ``"local"``      — :class:`LocalDeliverer`. No-op beyond the on-disk
                     write the scheduler already performed.
- ``"stdout"``     — :class:`StdoutDeliverer`. Prints the output to
                     stdout with a banner — handy for foreground / dev
                     runs of the daemon.
- ``"agentmail"``  — :class:`AgentMailDeliverer`. Sends the output via
                     the AgentMail REST API (``api.agentmail.to``).

SILENT_MARKER handling
----------------------

The ``"[SILENT]"`` prefix convention (Hermes's "nothing new to report"
signal) is enforced by the *scheduler*, not the deliverers. By the time
a deliverer's ``deliver()`` runs, the scheduler has already decided the
output is worth delivering. Individual deliverers should NOT re-check
for the marker — that would couple them to a scheduler invariant.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# ── ABC ─────────────────────────────────────────────────────────────


class Deliverer(ABC):
    """Strategy ABC for cron job output delivery.

    Subclasses set a class-level :attr:`name` (used as the registry key
    and matched against ``job["deliver"]``) and implement
    :meth:`deliver`. Failures should raise — the scheduler catches the
    exception, logs it, and records the message as
    ``last_delivery_error`` on the job record.
    """

    name: ClassVar[str]

    @abstractmethod
    def deliver(
        self,
        job: dict[str, Any],
        output: str,
        *,
        output_path: Path | None,
    ) -> None:
        """Deliver ``output`` for ``job``.

        Parameters
        ----------
        job
            The full job dict (so deliverers can read fields like
            ``deliver_to`` / ``name`` / ``id`` / channel-specific
            overrides).
        output
            The rendered job output string (markdown / plaintext).
        output_path
            Path to the on-disk copy already written by the scheduler.
            ``None`` only in edge cases (e.g. dry-run); deliverers that
            need a file must guard for it.

        Raises
        ------
        Exception
            Anything raised propagates to the scheduler, which records
            ``last_delivery_error`` on the job and continues with the
            next tick.
        """
        raise NotImplementedError


# ── registry ────────────────────────────────────────────────────────


_REGISTRY: dict[str, type[Deliverer]] = {}


def register_deliverer(cls: type[Deliverer]) -> type[Deliverer]:
    """Register a Deliverer subclass under its ``name``.

    Usable as a decorator. Re-registration replaces the previous entry
    so tests can swap deliverers safely.
    """
    name = getattr(cls, "name", None)
    if not name or not isinstance(name, str):
        raise ValueError(f"Deliverer {cls.__name__} must set a non-empty class-level 'name'")
    _REGISTRY[name] = cls
    return cls


def get_deliverer(name: str) -> type[Deliverer] | None:
    """Return the Deliverer class registered for ``name``, or ``None``."""
    return _REGISTRY.get(name)


def registered_deliverers() -> dict[str, type[Deliverer]]:
    """Snapshot of the current registry (for introspection / tests)."""
    return dict(_REGISTRY)


# ── bundled deliverers ─────────────────────────────────────────────


@register_deliverer
class LocalDeliverer(Deliverer):
    """No-op deliverer: the scheduler has already written ``output_path``.

    The ``[SILENT]`` suppression contract lives in the scheduler's
    pre-check (see :data:`deepagent_hermes.cron.scheduler.SILENT_MARKER`);
    by the time this deliverer is invoked the output is already on disk
    and (per scheduler policy) is worth surfacing. Local delivery means
    "filesystem-only" — nothing else to do.
    """

    name: ClassVar[str] = "local"

    def deliver(
        self,
        job: dict[str, Any],
        output: str,
        *,
        output_path: Path | None,
    ) -> None:
        # Intentionally empty: the scheduler already persisted output_path.
        return None


@register_deliverer
class StdoutDeliverer(Deliverer):
    """Print the output to stdout with a banner.

    Intended for running the daemon in the foreground / during
    development. Uses ``print`` rather than the logger so the output
    isn't reformatted by log handlers.
    """

    name: ClassVar[str] = "stdout"

    def deliver(
        self,
        job: dict[str, Any],
        output: str,
        *,
        output_path: Path | None,
    ) -> None:
        banner = f"\n=== Cron job: {job.get('name', '?')} ({job.get('id', '?')}) ===\n"
        sys.stdout.write(f"{banner}{output}\n")
        sys.stdout.flush()


@register_deliverer
class AgentMailDeliverer(Deliverer):
    """Send the output via Kedar's AgentMail inbox (``kzest@agentmail.to``).

    Mirrors ``C:\\kzest\\scripts\\Send-AgentMail.ps1`` — the canonical
    kzest-side reference for AgentMail self-sends. The endpoint shape is:

        POST https://api.agentmail.to/v0/inboxes/{from_inbox}/messages/send
        Authorization: Bearer $AGENTMAIL_API_KEY
        Body: {"subject": ..., "text": ..., "to": [...]}

    Configuration
    -------------

    - ``AGENTMAIL_API_KEY`` (env)  — required; raises ``RuntimeError``
      if missing.
    - ``job["deliver_to"]``        — recipient address. Defaults to
      Kedar's known address (``kdabhadk@gmail.com``).
    - ``job["from_inbox"]``        — sender inbox. Defaults to
      ``kzest@agentmail.to``.

    Subject:  ``[deepagent-hermes] {job.name}``
    Body:     the output as plain text (markdown rendering is the
              recipient's concern).

    Uses ``requests`` if importable, otherwise falls back to
    :mod:`urllib.request`. 4xx/5xx responses are surfaced as a
    ``RuntimeError`` carrying the response body so the scheduler can
    record a useful ``last_delivery_error``.
    """

    name: ClassVar[str] = "agentmail"

    # Known constants — matched to Send-AgentMail.ps1 so the
    # AgentMail-side audit trail looks the same regardless of caller.
    API_BASE: ClassVar[str] = "https://api.agentmail.to/v0"
    DEFAULT_FROM_INBOX: ClassVar[str] = "kzest@agentmail.to"
    DEFAULT_RECIPIENT: ClassVar[str] = "kdabhadk@gmail.com"

    def deliver(
        self,
        job: dict[str, Any],
        output: str,
        *,
        output_path: Path | None,
    ) -> None:
        api_key = os.environ.get("AGENTMAIL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "AGENTMAIL_API_KEY is not set; cannot deliver via AgentMail. "
                "Set it at user scope or pull from 1Password "
                "(see C:\\kzest\\scripts\\Send-AgentMail.ps1)."
            )

        recipient = job.get("deliver_to") or self.DEFAULT_RECIPIENT
        from_inbox = job.get("from_inbox") or self.DEFAULT_FROM_INBOX
        subject = f"[deepagent-hermes] {job.get('name', job.get('id', 'cron job'))}"
        recipients = [recipient] if isinstance(recipient, str) else list(recipient)

        payload = {
            "subject": subject,
            "text": output,
            "to": recipients,
        }
        url = f"{self.API_BASE}/inboxes/{from_inbox}/messages/send"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Prefer requests (richer error info); fall back to urllib so
        # this deliverer remains importable in stripped envs.
        try:
            import requests  # type: ignore[import-not-found]
        except ImportError:
            requests = None  # type: ignore[assignment]

        if requests is not None:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            status = resp.status_code
            body = resp.text
            if status >= 400:
                raise RuntimeError(f"AgentMail API call failed: HTTP {status} — {body}")
            return None

        # urllib fallback
        from urllib import error as _urlerror
        from urllib import request as _urlrequest

        data = json.dumps(payload).encode("utf-8")
        req = _urlrequest.Request(url, data=data, headers=headers, method="POST")
        try:
            with _urlrequest.urlopen(req, timeout=30) as r:
                _ = r.read()
        except _urlerror.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"AgentMail API call failed: HTTP {e.code} — {body}") from e


__all__ = [
    "AgentMailDeliverer",
    "Deliverer",
    "LocalDeliverer",
    "StdoutDeliverer",
    "get_deliverer",
    "register_deliverer",
    "registered_deliverers",
]
