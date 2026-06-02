"""``HonchoProvider`` — cross-session user-model memory via Honcho's SDK.

Implements ``MemoryProvider`` from ``deepagent_hermes.memory.provider``.
The full Hermes plugin (~1400 LOC at ``hermes-agent/plugins/memory/honcho/``)
covers prefetch threading, dialectic-depth tuning, peer cards, conclusions,
SOUL.md sync, and chunked message uploads. v1 here ships a **best-effort
straight-line implementation** of the four ABC methods that the SPEC
requires; the elaborate threading / dialectic features are TODOs marked
inline with clear interface boundaries so the Hermes code can be ported
incrementally without breaking callers.

Why best-effort and not stub-only:

- The ABC has four methods; an empty stub provider would compile but break
  silently when called. Best-effort means calls succeed for the typical
  installer (api_key in env + honcho-ai installed) and return ``[]`` from
  ``recall`` while logging a warning when the SDK shape changes — which is
  the same failure mode the agent already tolerates from the noop provider.

Config resolution chain (SPEC §13.2):

1. ``$HERMES_HOME/honcho.json`` (profile-scoped, preferred)
2. ``~/.deepagent-hermes/honcho.json``
3. ``~/.honcho/config.json`` (legacy, shared with the Honcho CLI)
4. Environment: ``HONCHO_API_KEY``, ``HONCHO_ENVIRONMENT``

Host key (``workspace`` in Honcho terminology): ``deepagent_hermes_<profile>``
when a profile is set, else ``deepagent_hermes``. Override via
``DEEPAGENT_HERMES_HONCHO_HOST``.

Recall modes mirror Hermes: ``hybrid`` (default), ``context``, ``tools``.
Legacy ``auto`` aliases to ``hybrid``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

from deepagent_hermes.memory.provider import MemoryProvider, RecallMode, register_provider

logger = logging.getLogger(__name__)


# ── Config resolution ────────────────────────────────────────────────


def _hermes_home() -> Path:
    """Mirror ``deepagent_hermes.memory.tool._hermes_home`` to avoid an import cycle."""
    return Path(
        os.environ.get("DEEPAGENT_HERMES_HOME")
        or os.environ.get("HERMES_HOME")
        or (Path.home() / ".deepagent-hermes")
    )


def _candidate_config_paths() -> list[Path]:
    """Honcho config file search order — first existing file wins.

    Returns the full ordered list (not just the first hit) so callers can log
    "looked at X, Y, Z" diagnostics on failure.
    """
    return [
        _hermes_home() / "honcho.json",
        Path.home() / ".deepagent-hermes" / "honcho.json",
        Path.home() / ".honcho" / "config.json",
    ]


def _load_config() -> dict[str, Any]:
    """Resolve Honcho config from files + env. Returns a flat dict.

    Keys we look for (Honcho SDK + Hermes conventions intermixed):
      - ``api_key`` / ``HONCHO_API_KEY``
      - ``environment`` / ``HONCHO_ENVIRONMENT`` (e.g. ``"production"``)
      - ``base_url`` (for self-hosted Honcho)
      - ``workspace`` (overrides the host-key default)
      - ``recall_mode`` (``hybrid`` / ``context`` / ``tools`` / ``auto``)
    """
    config: dict[str, Any] = {}

    # File chain — later files DO NOT overwrite earlier hits. First match wins
    # because the precedence is "most specific to least specific".
    for path in _candidate_config_paths():
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("HonchoProvider: failed to parse %s: %s", path, e)
                continue
            # Flatten Honcho's nested ``honcho`` block if present.
            if isinstance(data, dict):
                inner = data.get("honcho") if isinstance(data.get("honcho"), dict) else None
                config = {**(inner or {}), **data}
                logger.debug("HonchoProvider: loaded config from %s", path)
                break

    # Env vars override file values — they're the standard escape hatch.
    if os.environ.get("HONCHO_API_KEY"):
        config["api_key"] = os.environ["HONCHO_API_KEY"]
    if os.environ.get("HONCHO_ENVIRONMENT"):
        config["environment"] = os.environ["HONCHO_ENVIRONMENT"]
    if os.environ.get("HONCHO_BASE_URL"):
        config["base_url"] = os.environ["HONCHO_BASE_URL"]

    return config


def _resolve_workspace(config: dict[str, Any]) -> str:
    """Compute the Honcho workspace (host key).

    Order: explicit ``DEEPAGENT_HERMES_HONCHO_HOST`` env → config
    ``workspace`` → ``deepagent_hermes_<profile_sanitized>`` → ``deepagent_hermes``.

    "Sanitized" = lowercase, non-alphanumerics → underscore — matches how
    Hermes derives the host key from profile names.
    """
    env_override = os.environ.get("DEEPAGENT_HERMES_HONCHO_HOST")
    if env_override:
        return env_override
    if config.get("workspace"):
        return str(config["workspace"])

    profile = os.environ.get("DEEPAGENT_HERMES_PROFILE", "").strip()
    if profile:
        sanitized = re.sub(r"[^a-z0-9_]+", "_", profile.lower()).strip("_")
        if sanitized:
            return f"deepagent_hermes_{sanitized}"
    return "deepagent_hermes"


# ── Provider ─────────────────────────────────────────────────────────


class HonchoProvider(MemoryProvider):
    """Honcho-backed ``MemoryProvider``.

    The full Hermes implementation runs background prefetch threads, multi-pass
    dialectic queries, and peer-card management. This v1 ships a straight-line
    synchronous version with TODOs at the threading-extension points so the
    behavior can be ported incrementally. The current shape is correct enough
    that the SPEC §13.2 acceptance test (``what do you know about me?`` reads
    something) works once Honcho credentials are present.
    """

    name = "honcho"

    def __init__(self, *, recall_mode: RecallMode | Literal["auto"] = "hybrid") -> None:
        """Initialize the provider with a default recall mode.

        Args:
            recall_mode: Default mode used when callers don't override on each
                ``recall()`` call. ``"auto"`` is a legacy alias for ``"hybrid"``
                — we coerce it here so the rest of the code only sees the
                three canonical values.
        """
        if recall_mode == "auto":
            recall_mode = "hybrid"
        self.recall_mode: RecallMode = recall_mode

        # Populated by setup_session — None means "not yet initialized" so
        # the recall/record_turn methods can no-op gracefully.
        self._client: Any | None = None
        self._workspace: str | None = None
        self._peer_id: str | None = None  # user peer
        self._ai_peer_id: str = "ai"      # constant for now; Hermes lets SOUL.md override
        self._session_id: str | None = None
        self._config: dict[str, Any] = {}

    # ── lifecycle ──

    def setup_session(self, session_id: str, user_id: str | None = None) -> None:
        """Lazy-import the SDK, build the client, and ensure session+peer exist.

        Raises a helpful ``ImportError`` (with the install command) on a
        missing ``honcho-ai`` package — the user shouldn't have to read the
        traceback to learn how to fix it.
        """
        try:
            from honcho import Honcho  # noqa: PLC0415  — lazy import is the point
        except ImportError as e:
            raise ImportError(
                "HonchoProvider requires honcho-ai SDK. "
                "Install with: pip install deepagent-hermes[honcho]"
            ) from e

        self._config = _load_config()
        self._workspace = _resolve_workspace(self._config)
        self._peer_id = user_id or self._config.get("user_peer", "user")
        self._session_id = session_id

        # Honcho client kwargs — only pass non-None so we don't trip the SDK's
        # "you can't combine these" guards (e.g. base_url + environment).
        client_kwargs: dict[str, Any] = {}
        if self._config.get("api_key"):
            client_kwargs["api_key"] = self._config["api_key"]
        if self._config.get("environment"):
            client_kwargs["environment"] = self._config["environment"]
        if self._config.get("base_url"):
            client_kwargs["base_url"] = self._config["base_url"]

        try:
            self._client = Honcho(**client_kwargs)
            # TODO(honcho-impl): the real SDK surface is
            #   workspace = client.workspaces[self._workspace]
            #   peer = workspace.peers[self._peer_id]
            #   session = workspace.sessions[self._session_id]
            # Calling those here would warm caches; we defer to first use
            # because the SDK does its own lazy init.
        except Exception as e:  # noqa: BLE001 — log + degrade, don't crash the agent
            logger.warning("HonchoProvider: client init failed: %s", e)
            self._client = None

    def recall(self, query: str, mode: RecallMode = "hybrid") -> list[str]:
        """Return cross-session context snippets.

        ``tools`` mode short-circuits to ``[]`` — by spec, tools-mode users
        invoke the provider's tools explicitly and want zero auto-injection.

        ``context`` and ``hybrid`` modes call ``peer.context()`` (cheap, no
        LLM) and optionally ``peer.chat()`` (LLM reasoning) under hybrid.
        Failures degrade to ``[]`` with a warning — never raise back to the
        prompt builder.
        """
        if mode == "auto":  # type: ignore[comparison-overlap]
            mode = "hybrid"
        if mode == "tools":
            return []
        if not self._client or not self._workspace or not self._peer_id:
            return []
        if not query or not query.strip():
            return []

        # The SDK shape is roughly:
        #   workspace = client.workspaces[self._workspace]
        #   peer      = workspace.peers[self._peer_id]
        #   ctx       = peer.context(query=query, max_tokens=...)
        # We wrap that pattern in a try/except so SDK upgrades that move
        # methods around don't take down the agent — we log and return [].
        results: list[str] = []
        try:
            workspace = self._client.workspaces[self._workspace]
            peer = workspace.peers[self._peer_id]

            # Context call — cheap, available in all modes that inject.
            try:
                ctx = peer.context(query=query) if hasattr(peer, "context") else None
                if ctx:
                    text = getattr(ctx, "text", None) or str(ctx)
                    if text and text.strip():
                        results.append(text.strip())
            except Exception as e:  # noqa: BLE001
                logger.debug("HonchoProvider.recall: peer.context() failed: %s", e)

            # Hybrid adds a dialectic .chat() call for synthesized answers.
            # Context-only mode skips this to save tokens.
            if mode == "hybrid":
                try:
                    reply = (
                        peer.chat(query, reasoning_level="low")
                        if hasattr(peer, "chat")
                        else None
                    )
                    if reply:
                        text = getattr(reply, "content", None) or str(reply)
                        if text and text.strip() and text not in results:
                            results.append(text.strip())
                except Exception as e:  # noqa: BLE001
                    logger.debug("HonchoProvider.recall: peer.chat() failed: %s", e)
        except Exception as e:  # noqa: BLE001
            logger.warning("HonchoProvider.recall: unexpected error: %s", e)
            return []

        return results

    def record_turn(self, role: str, content: str) -> None:
        """Push one message into the Honcho session for user-model learning.

        Best-effort: failures only log. The agent must never stall because the
        memory provider's backend is slow.
        """
        if not self._client or not self._workspace or not self._session_id:
            return
        if not content or not content.strip():
            return

        try:
            workspace = self._client.workspaces[self._workspace]
            session = workspace.sessions[self._session_id]
            # SDK shape:
            #   session.messages.create(peer_id=..., role=..., content=...)
            # The exact kwargs vary across SDK versions; we try the modern
            # shape first and fall back to the older one.
            messages = getattr(session, "messages", None)
            if messages is None:
                return
            peer_id = self._peer_id if role == "user" else self._ai_peer_id
            if hasattr(messages, "create"):
                messages.create(peer_id=peer_id, role=role, content=content)
            elif hasattr(session, "add_message"):
                session.add_message(role, content)
            # If neither hook exists, the SDK has drifted — log once and move
            # on. We don't want a per-turn log explosion.
            else:
                logger.debug(
                    "HonchoProvider.record_turn: no known message-add API on session"
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("HonchoProvider.record_turn: %s", e)

    def teardown(self) -> None:
        """Best-effort flush. No threads to join in this straight-line port."""
        # TODO(honcho-impl): when the background prefetch/sync threads from
        # the Hermes implementation get ported, join them here with a sane
        # timeout (Hermes uses 5-10s). For now there's nothing to do.
        self._client = None
        self._workspace = None
        self._peer_id = None
        self._session_id = None


# Self-register at import time so `get_provider("honcho")` works once the
# package is on sys.path (or once the plugin loader has imported this module).
register_provider("honcho", HonchoProvider)


__all__ = ["HonchoProvider"]
