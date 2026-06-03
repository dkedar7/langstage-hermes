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

SDK shape (honcho-ai >= 2.0, imported as ``honcho``):

    from honcho import Honcho, MessageCreateParams
    client = Honcho(api_key=..., environment=..., base_url=..., workspace_id=...)
    user_peer = client.peer("user")
    ai_peer   = client.peer("assistant")
    session   = client.session(session_id, peers=[user_peer, ai_peer])
    session.add_messages(MessageCreateParams(peer_id="user", content="..."))
    page      = session.messages(size=20, reverse=True)
    answer    = user_peer.chat("what does the user like?", session=session)

The workspace is set **on the client**, not on a sub-resource. There is no
``workspaces.get_or_create(name=...)`` chain in v2 — pre-v2 doc snippets
that show that pattern are stale.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
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

    "Sanitized" = lowercase, non-alphanumerics → underscore, truncate to 50 —
    matches how Hermes derives the host key from profile names.
    """
    env_override = os.environ.get("DEEPAGENT_HERMES_HONCHO_HOST")
    if env_override:
        return env_override
    if config.get("workspace"):
        return str(config["workspace"])

    profile = os.environ.get("DEEPAGENT_HERMES_PROFILE", "").strip()
    if profile:
        sanitized = re.sub(r"[^a-z0-9_]+", "_", profile.lower()).strip("_")
        sanitized = sanitized[:50]  # truncate per spec
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
        self._ai_peer_id: str = "assistant"  # constant for now; Hermes lets SOUL.md override
        self._session_id: str | None = None
        self._config: dict[str, Any] = {}

        # SDK resource handles, cached after setup_session.
        self._user_peer: Any | None = None
        self._ai_peer: Any | None = None
        self._session: Any | None = None

        # Provider can be hit from the main agent thread AND the reflection
        # subagent thread. Honcho SDK calls aren't documented thread-safe;
        # serialise writes (and the setup/teardown lifecycle) with one lock.
        self._lock = threading.RLock()

    # ── lifecycle ──

    def setup_session(self, session_id: str, user_id: str | None = None) -> None:
        """Lazy-import the SDK, build the client, and ensure session+peer exist.

        Raises a helpful ``ImportError`` (with the install command) on a
        missing ``honcho-ai`` package — the user shouldn't have to read the
        traceback to learn how to fix it.

        Idempotent: safe to call twice. Uses ``client.peer(id)`` /
        ``client.session(id)`` which are get-or-create under the hood.
        """
        try:
            from honcho import Honcho
        except ImportError as e:
            raise ImportError(
                "HonchoProvider requires honcho-ai SDK. "
                "Install with: pip install deepagent-hermes[honcho]"
            ) from e

        with self._lock:
            self._config = _load_config()
            self._workspace = _resolve_workspace(self._config)
            self._peer_id = user_id or self._config.get("user_peer", "user")
            self._session_id = session_id

            # Honcho client kwargs — only pass non-None so we don't trip the SDK's
            # default-handling (some kwargs use sentinel FieldInfo defaults that
            # break if you pass an explicit None for an unset value).
            client_kwargs: dict[str, Any] = {"workspace_id": self._workspace}
            if self._config.get("api_key"):
                client_kwargs["api_key"] = self._config["api_key"]
            if self._config.get("environment"):
                client_kwargs["environment"] = self._config["environment"]
            if self._config.get("base_url"):
                client_kwargs["base_url"] = self._config["base_url"]

            try:
                self._client = Honcho(**client_kwargs)
                # get_or_create on the SDK — these calls are idempotent and
                # cheap (the SDK caches the local object; server-side it's a
                # PUT-with-if-not-exists).
                self._user_peer = self._client.peer(self._peer_id)
                self._ai_peer = self._client.peer(self._ai_peer_id)
                # Attach both peers at session creation so dialectic queries
                # have the right relational graph from turn 1. The SDK accepts
                # PeerBase instances directly.
                self._session = self._client.session(
                    self._session_id,
                    peers=[self._user_peer, self._ai_peer],
                )
            except Exception as e:
                logger.warning("HonchoProvider: client init failed: %s", e)
                self._client = None
                self._user_peer = None
                self._ai_peer = None
                self._session = None

    def recall(self, query: str, mode: RecallMode = "hybrid") -> list[str]:
        """Return cross-session context snippets.

        Mode mapping (per SPEC):

        - ``hybrid`` — combine ``peer.chat(query)`` with last 5 session messages
          for short-term context. Default; the most useful all-purpose mix.
        - ``context`` — only last 20 ``session.messages()`` (no LLM call).
        - ``tools`` — only ``peer.chat(query)`` (longer-term reasoning, no
          short-term echo).

        Failures **always** degrade to ``[]`` with a logged warning — losing
        recall is preferable to crashing the agent.
        """
        # auto → hybrid (legacy alias)
        if mode == "auto":  # type: ignore[comparison-overlap]
            mode = "hybrid"
        if not self._client or not self._user_peer or not self._session:
            return []
        if not query or not query.strip():
            return []

        results: list[str] = []
        try:
            # --- dialectic chat call (hybrid + tools) ---
            if mode in ("hybrid", "tools"):
                try:
                    reply = self._user_peer.chat(
                        query,
                        session=self._session,
                        reasoning_level="low",
                    )
                    # peer.chat returns Optional[str] per the v2 SDK signature.
                    if reply:
                        text = reply if isinstance(reply, str) else (
                            getattr(reply, "content", None) or str(reply)
                        )
                        if text and text.strip():
                            results.append(text.strip())
                except Exception as e:
                    logger.debug("HonchoProvider.recall: peer.chat() failed: %s", e)

            # --- short-term context (hybrid + context) ---
            if mode in ("hybrid", "context"):
                try:
                    # hybrid wants a small recent tail; context wants more.
                    size = 5 if mode == "hybrid" else 20
                    # session.messages() returns a SyncPage iterable of Message
                    # objects with .content + .peer_id attrs.
                    page = self._session.messages(size=size, reverse=True)
                    items = list(page) if page is not None else []
                    for msg in items:
                        text = getattr(msg, "content", None)
                        if not text:
                            continue
                        peer_id = getattr(msg, "peer_id", "") or ""
                        snippet = f"[{peer_id}] {text}".strip() if peer_id else text.strip()
                        if snippet and snippet not in results:
                            results.append(snippet)
                except Exception as e:
                    logger.debug("HonchoProvider.recall: session.messages() failed: %s", e)
        except Exception as e:
            logger.warning("HonchoProvider.recall: unexpected error: %s", e)
            return []

        return results

    def record_turn(self, role: str, content: str) -> None:
        """Push one message into the Honcho session for user-model learning.

        Role mapping:
          - ``user`` → user_peer
          - ``assistant`` → assistant_peer
          - anything else (``tool``, ``system``, …) → skip silently. Honcho's
            user model expects bilateral dialogue; tool traces would just be
            noise in the dialectic index.

        Best-effort: failures only log at DEBUG. The agent must never stall
        because the memory provider's backend is slow.
        """
        if not self._client or not self._session:
            return
        if not content or not content.strip():
            return

        # Map role → peer_id. Unknown roles are dropped on the floor on purpose.
        if role == "user":
            peer_id = self._peer_id
        elif role == "assistant":
            peer_id = self._ai_peer_id
        else:
            return

        if not peer_id:
            return

        try:
            # Lazy-import MessageCreateParams here — if the import fails (SDK
            # missing/drifted) we want the existing except to swallow it
            # rather than crashing record_turn.
            from honcho import MessageCreateParams
        except ImportError as e:
            logger.debug("HonchoProvider.record_turn: MessageCreateParams import failed: %s", e)
            return

        with self._lock:
            try:
                params = MessageCreateParams(peer_id=peer_id, content=content)
                self._session.add_messages(params)
            except Exception as e:
                logger.debug("HonchoProvider.record_turn: %s", e)

    def teardown(self) -> None:
        """Best-effort flush + handle release.

        Honcho's v2 SDK doesn't expose a ``Session.close()`` (sessions are
        server-side resources; clients are stateless HTTP wrappers). We still
        try ``.close()`` if it appears in a future SDK version, then drop
        local refs so a re-``setup_session`` rebuilds cleanly.

        We do NOT delete the session — keep it for future recall across runs.
        """
        with self._lock:
            session = self._session
            client = self._client

            # Future: when the background prefetch/sync threads from the full
            # Hermes port land, join them here with a sane timeout (Hermes uses
            # 5-10s). Straight-line v1 has no threads of its own so there's
            # nothing extra to flush beyond the SDK-level close() calls below.

            for resource in (session, client):
                if resource is None:
                    continue
                close = getattr(resource, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as e:
                        logger.debug("HonchoProvider.teardown: close() failed: %s", e)

            self._client = None
            self._user_peer = None
            self._ai_peer = None
            self._session = None
            self._workspace = None
            self._peer_id = None
            self._session_id = None


# Self-register at import time so `get_provider("honcho")` works once the
# package is on sys.path (or once the plugin loader has imported this module).
register_provider("honcho", HonchoProvider)


__all__ = ["HonchoProvider"]
