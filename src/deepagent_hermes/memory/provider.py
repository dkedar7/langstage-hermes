"""``MemoryProvider`` ABC + provider registry (SPEC §13.2).

A memory provider is a single-select plug-in slot. The agent loads at most one
provider — selected by ``config.memory.provider`` (default ``""`` → no-op).

Bundled providers register themselves at import time via
``register_provider("honcho", HonchoProvider)``. Third-party plug-ins discovered
through the plugin loader (SPEC §15) can register additional providers the
same way.

Lifecycle:

1. ``setup_session(session_id, user_id)`` — called by ``HonchoMiddleware`` at
   ``before_agent`` time, once per session.
2. ``recall(query, mode)`` — called by the volatile-layer prompt builder to
   fetch relevant cross-session context; returns a list of plain-text excerpts
   that get joined and injected into the system prompt.
3. ``record_turn(role, content)`` — called by ``after_model`` for every
   message, so the provider can update its user model.
4. ``teardown()`` — called by ``after_agent`` at session end; the provider
   flushes any buffered writes here.

Errors raised by any of these methods should be caught by the middleware and
logged — the agent must keep running even when the provider is misbehaving.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

# Recall modes mirror Hermes's vocabulary. "hybrid" is the recommended default
# (auto-injected context + tool calls); "context" disables tools; "tools"
# disables auto-injection. Legacy "auto" is aliased to "hybrid" inside the
# Honcho provider for back-compat with old honcho.json files.
RecallMode = Literal["hybrid", "context", "tools"]


class MemoryProvider(ABC):
    """Abstract base class for cross-session memory providers."""

    @abstractmethod
    def setup_session(self, session_id: str, user_id: str | None = None) -> None:
        """Initialize provider state for a new session.

        Called from ``HonchoMiddleware.before_agent``. Implementations should
        be quick — long-running setup belongs in a background thread.
        """

    @abstractmethod
    def recall(self, query: str, mode: RecallMode = "hybrid") -> list[str]:
        """Return relevant cross-session context as plain-text excerpts.

        ``mode`` controls the recall strategy:

        - ``"hybrid"`` — provider's default mix of context + reasoning calls.
        - ``"context"`` — context-only (no LLM reasoning, cheaper).
        - ``"tools"`` — return an empty list (the provider exposes its own
          tools instead of auto-injecting).

        Returns an empty list when there's nothing useful to surface; callers
        join with ``"\\n\\n"`` and skip the system-prompt block when empty.
        """

    @abstractmethod
    def record_turn(self, role: str, content: str) -> None:
        """Record one conversation turn for the provider's user model.

        Called once per message after the model emits its response. ``role`` is
        a free-form string (``"user"`` / ``"assistant"`` / ``"tool"`` / …);
        providers should accept anything they don't recognize without raising.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Flush + release resources. Called once at session end."""


# ── Provider registry ─────────────────────────────────────────────────

_REGISTRY: dict[str, type[MemoryProvider]] = {}


def register_provider(name: str, cls: type[MemoryProvider]) -> None:
    """Register a memory provider class under ``name``.

    Re-registration overwrites silently — last-write-wins lets a project
    plug-in shadow a bundled provider intentionally.

    The empty string ``""`` is reserved for the no-op provider (the
    documented "disabled" config value). External callers must use a
    non-empty name; the bundled noop is the only registration that's
    allowed to use ``""``.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"register_provider: name must be a string, got {type(name).__name__}"
        )
    if not isinstance(cls, type) or not issubclass(cls, MemoryProvider):
        raise TypeError(
            f"register_provider: cls must be a MemoryProvider subclass, got {cls!r}"
        )
    # External callers can't claim "" — that's the noop slot. The noop
    # registration below this function uses _REGISTRY directly.
    if name == "" and cls is not NoopMemoryProvider:
        raise ValueError(
            "register_provider: empty name is reserved for NoopMemoryProvider"
        )
    _REGISTRY[name] = cls


def get_provider(name: str) -> type[MemoryProvider]:
    """Return the registered provider class for ``name``.

    Raises ``KeyError`` if the name isn't registered — callers should catch
    this and fall back to ``NoopMemoryProvider`` with a warning, so a stale
    config value doesn't crash the agent.
    """
    try:
        return _REGISTRY[name]
    except KeyError as e:
        registered = sorted(_REGISTRY)
        raise KeyError(
            f"No memory provider registered as {name!r}. "
            f"Available: {registered!r}"
        ) from e


def available_providers() -> list[str]:
    """Return registered provider names, sorted alphabetically."""
    return sorted(_REGISTRY)


# ── No-op provider (default when memory.provider is unset) ───────────


class NoopMemoryProvider(MemoryProvider):
    """Default provider — does nothing, returns nothing. Safe everywhere.

    Selected when ``config.memory.provider == ""``. Lets every middleware
    call into ``provider.recall()`` / ``provider.record_turn()`` without
    a ``None`` check.
    """

    def setup_session(self, session_id: str, user_id: str | None = None) -> None:  # noqa: D401
        """No-op."""
        return None

    def recall(self, query: str, mode: RecallMode = "hybrid") -> list[str]:  # noqa: D401
        """Return ``[]`` — nothing to recall without a backing provider."""
        return []

    def record_turn(self, role: str, content: str) -> None:  # noqa: D401
        """No-op."""
        return None

    def teardown(self) -> None:  # noqa: D401
        """No-op."""
        return None


# Always register the noop under "" (empty string = default) and "noop" so
# `get_provider(config.memory_provider)` works without a None check.
register_provider("", NoopMemoryProvider)
register_provider("noop", NoopMemoryProvider)


__all__ = [
    "MemoryProvider",
    "NoopMemoryProvider",
    "RecallMode",
    "available_providers",
    "get_provider",
    "register_provider",
]
