"""`HermesToolRegistry` — central tool registry with TTL-cached availability checks.

Mirrors Hermes's ``tools/registry.py`` API shape (SPEC §11) on top of LangChain
``BaseTool`` objects instead of bare ``(schema, handler)`` pairs. The 30 s TTL on
``check_fn`` mirrors Hermes's amortization of expensive probes (Docker daemon
ping, Modal SDK import, playwright binary presence): for a long-lived CLI or
daemon, hitting those on every ``get_tools()`` call is pure waste — external
state changes on human timescales.

The registry itself imports nothing from ``langchain`` / ``deepagents`` so it
can be exercised by the test suite without the full middleware stack installed.
Tools are accepted as opaque objects with at minimum a ``.name`` attribute,
which is the contract every ``BaseTool`` honors.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

# Default TTL for cached check_fn results. Hermes uses 30 s; we keep parity so
# downstream `deepagent-hermes tools` CLI behavior matches user expectations.
_CHECK_FN_TTL_SECONDS = 30.0


@dataclass
class _ToolEntry:
    """Single registration record. Stored internally; not part of the public API."""

    tool: Any
    toolset: str
    check_fn: Callable[[], tuple[bool, str | None]] | None = None
    requires_env: tuple[str, ...] = ()


class HermesToolRegistry:
    """Registry of tools grouped by toolset, with optional availability checks.

    The registry is intentionally minimal: ``register`` accepts any object with
    a ``.name`` attribute (LangChain ``BaseTool`` satisfies this natively).
    ``get_tools`` returns the filtered list a downstream agent factory should
    bind to the model. ``check_status`` runs the user-supplied availability
    probe, caching the result for ~30 s so repeat callers don't re-probe
    Docker/Modal/playwright/network endpoints every turn.
    """

    def __init__(self) -> None:
        # name -> _ToolEntry
        self._tools: dict[str, _ToolEntry] = {}
        # name -> (ts_monotonic, ok, reason)
        self._check_cache: dict[str, tuple[float, bool, str | None]] = {}
        self._lock = threading.RLock()
        self._ttl = _CHECK_FN_TTL_SECONDS

    # ── registration ──────────────────────────────────────────────────

    def register(
        self,
        tool: Any,
        *,
        toolset: str,
        check_fn: Callable[[], tuple[bool, str | None]] | None = None,
        requires_env: Iterable[str] = (),
    ) -> None:
        """Register ``tool`` under ``toolset``.

        Args:
            tool: Any object with a ``.name`` attribute (e.g. a LangChain
                ``BaseTool``). The registry stores it opaquely and returns it
                from ``get_tools`` to the caller.
            toolset: Logical group name (one of the 33 in
                :mod:`deepagent_hermes.tools.toolsets`).
            check_fn: Optional zero-arg callable returning
                ``(ok: bool, reason: str | None)``. Results are TTL-cached
                (~30 s) so the same probe isn't re-run on every turn.
            requires_env: Optional iterable of env var names the tool requires;
                surfaced by ``get_status()`` for diagnostic output.
        """
        name = getattr(tool, "name", None)
        if not name:
            raise ValueError(
                "register() requires a tool with a non-empty .name attribute; "
                f"got {tool!r}"
            )
        with self._lock:
            self._tools[name] = _ToolEntry(
                tool=tool,
                toolset=toolset,
                check_fn=check_fn,
                requires_env=tuple(requires_env),
            )
            # New registration invalidates any prior cached result for this name.
            self._check_cache.pop(name, None)

    def deregister(self, name: str) -> None:
        """Drop a tool by name. Silent no-op when the name is unknown."""
        with self._lock:
            self._tools.pop(name, None)
            self._check_cache.pop(name, None)

    # ── retrieval ─────────────────────────────────────────────────────

    def get_tools(
        self,
        *,
        enabled_toolsets: set[str] | None = None,
        platform: str = "cli",
        disabled: set[str] = frozenset(),  # type: ignore[assignment]
    ) -> list[Any]:
        """Return tools currently available for the given context.

        Filters applied (in order):
          1. ``enabled_toolsets`` — if provided, only tools in those toolsets pass.
          2. ``disabled`` — tool names in this set are dropped unconditionally.
          3. ``check_fn`` — tools whose availability probe returns ``(False, ...)``
             are dropped. Probe results are TTL-cached.

        ``platform`` is accepted for parity with Hermes's per-platform overrides;
        v1 doesn't gate on it directly here (callers do so by passing
        ``enabled_toolsets``), but future platform-specific filtering belongs
        in this method.
        """
        with self._lock:
            entries = list(self._tools.values())

        result: list[Any] = []
        for entry in entries:
            name = entry.tool.name
            if enabled_toolsets is not None and entry.toolset not in enabled_toolsets:
                continue
            if name in disabled:
                continue
            if entry.check_fn is not None:
                ok, _reason = self.check_status(name)
                if not ok:
                    continue
            result.append(entry.tool)
        # Sort for determinism (mirrors Hermes's ``get_definitions`` behavior).
        result.sort(key=lambda t: t.name)
        return result

    def list_toolsets(self) -> dict[str, list[str]]:
        """Return ``{toolset: [sorted tool names]}`` for every registered tool."""
        with self._lock:
            entries = list(self._tools.values())
        out: dict[str, list[str]] = {}
        for entry in entries:
            out.setdefault(entry.toolset, []).append(entry.tool.name)
        for names in out.values():
            names.sort()
        return out

    def get_tool(self, name: str) -> Any | None:
        """Return the raw tool object for ``name`` (bypasses check_fn). ``None`` if unknown."""
        with self._lock:
            entry = self._tools.get(name)
        return entry.tool if entry else None

    def get_toolset_for_tool(self, name: str) -> str | None:
        """Return the toolset a tool belongs to, or ``None`` if unknown."""
        with self._lock:
            entry = self._tools.get(name)
        return entry.toolset if entry else None

    # ── availability ──────────────────────────────────────────────────

    def check_status(self, tool_name: str) -> tuple[bool, str | None]:
        """Return ``(ok, reason)`` for ``tool_name``, TTL-cached for 30 s.

        Tools registered without a ``check_fn`` are always available
        (``(True, None)``). Unknown names return ``(False, "unknown tool")``.

        Exceptions raised by ``check_fn`` are swallowed and reported as
        unavailable; the same probe is retried on the next call after TTL
        expiry rather than poisoning the cache.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._tools.get(tool_name)
            if entry is None:
                return (False, "unknown tool")
            if entry.check_fn is None:
                return (True, None)
            cached = self._check_cache.get(tool_name)
            if cached is not None:
                ts, ok, reason = cached
                if now - ts < self._ttl:
                    return (ok, reason)
            fn = entry.check_fn

        # Call the probe outside the lock — it may do network / disk I/O.
        try:
            ok_raw, reason = fn()
            ok = bool(ok_raw)
        except Exception as exc:  # pragma: no cover - defensive
            ok, reason = False, f"{type(exc).__name__}: {exc}"

        with self._lock:
            self._check_cache[tool_name] = (now, ok, reason)
        return (ok, reason)

    def invalidate_check_cache(self, tool_name: str | None = None) -> None:
        """Drop cached check results for ``tool_name`` or every tool when ``None``.

        Call after a config flip that affects tool availability (e.g.
        ``deepagent-hermes tools enable foo``) so the next ``get_tools()``
        re-probes immediately.
        """
        with self._lock:
            if tool_name is None:
                self._check_cache.clear()
            else:
                self._check_cache.pop(tool_name, None)

    # ── introspection ─────────────────────────────────────────────────

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return isinstance(name, str) and name in self._tools

    def names(self) -> list[str]:
        """Return all registered tool names, sorted."""
        with self._lock:
            return sorted(self._tools.keys())


# Module-level singleton — mirrors Hermes's ``registry = ToolRegistry()`` shape.
registry = HermesToolRegistry()


__all__ = ["HermesToolRegistry", "registry"]
