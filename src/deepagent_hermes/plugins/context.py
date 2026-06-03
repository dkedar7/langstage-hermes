"""``PluginContext`` + ``LoadedPlugin`` + hook-name registry.

SPEC §15.3 enumerates 17 lifecycle hook names; we expose them as
``VALID_HOOKS`` and validate any ``register_hook(name, ...)`` call against
that set so typos surface immediately.

The plugin contract (per-plugin ``register(ctx: PluginContext)``) gets four
pluggable surfaces in v1:

  - **Tools**         (``register_tool``)        — slots into ``HermesToolRegistry``.
  - **Memory provider** (``register_memory_provider``) — single-select; deny on conflict.
  - **Context engine** (``register_context_engine``)   — single-select; deny on conflict.
  - **Hooks**         (``register_hook``)        — multi-callback; declared-order fire.
  - **Slash commands**(``register_slash_command``) — appears in the chat REPL.

Most hooks are wired in v1 through deepagents middleware (``pre_tool_call``
↔ ``wrap_tool_call``, ``pre_llm_call`` ↔ ``wrap_model_call``, etc.). The
rest are accepted but quietly no-op until v0.2 ships ``PluginEventBus``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ── global hook registry (read by PluginEventBus) ──────────────────
#
# Every ``PluginContext.register_hook(name, fn)`` call appends ``fn`` to
# this module-level mapping in addition to the per-context ``hooks`` store
# passed by the loader. ``PluginEventBus`` reads from here on each fire so
# plugins registered after agent build (e.g. via a future hot-reload path)
# still get their callbacks fired, and so test harnesses don't need to
# thread a registry through the loader to make hooks observable.
#
# Keys are hook names from ``VALID_HOOKS``; values are lists of callables
# in registration order. Use ``get_global_hook_registry()`` to access.

_GLOBAL_HOOK_REGISTRY: dict[str, list[Callable[..., Any]]] = defaultdict(list)


def get_global_hook_registry() -> dict[str, list[Callable[..., Any]]]:
    """Return the module-level hook registry that ``PluginEventBus`` reads.

    Mutating the returned dict mutates the live registry — this is
    intentional so test fixtures can clear it between cases:

        get_global_hook_registry().clear()
    """
    return _GLOBAL_HOOK_REGISTRY


# ── hook registry ───────────────────────────────────────────────────


VALID_HOOKS: set[str] = {
    # tool surface
    "pre_tool_call",
    "post_tool_call",
    "transform_terminal_output",
    "transform_tool_result",
    # llm surface
    "transform_llm_output",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    # session lifecycle
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    # subagent
    "subagent_stop",
    # gateway / approvals
    "pre_gateway_dispatch",
    "pre_approval_request",
    "post_approval_response",
}

# Hooks wired through deepagents middleware in v1 (the rest no-op until 0.2).
_WIRED_HOOKS_V1: dict[str, str] = {
    "pre_tool_call": "wrap_tool_call (middleware)",
    "post_tool_call": "wrap_tool_call (middleware)",
    "pre_llm_call": "wrap_model_call (middleware)",
    "post_llm_call": "wrap_model_call (middleware)",
    "on_session_start": "before_agent (middleware)",
    "on_session_end": "after_agent (middleware)",
}


# ── data shapes ────────────────────────────────────────────────────


@dataclass
class LoadedPlugin:
    """Runtime record of a plugin discovered + ``register()``-ed.

    ``register_fn`` is captured at discovery so the loader can re-run
    registration (e.g. ``deepagent-hermes plugins reload`` in a future
    release) without re-importing the module.
    """

    name: str
    version: str = ""
    description: str = ""
    source: Literal["bundled", "user", "project", "entry_point"] = "bundled"
    path: Path | None = None
    register_fn: Callable[[PluginContext], None] | None = None
    error: str | None = None
    enabled: bool = True

    # populated by PluginContext during register()
    tools_registered: list[str] = field(default_factory=list)
    hooks_registered: list[str] = field(default_factory=list)
    commands_registered: list[str] = field(default_factory=list)


# ── PluginContext ──────────────────────────────────────────────────


class PluginContext:
    """The handle passed to each plugin's ``register(ctx)`` function.

    Constructed once per loader run with references to the host's tool /
    memory / slash-command / hook registries. Plugin authors call the
    ``register_*`` methods to wire their integrations into the host.

    The registries are intentionally typed as ``Any`` so this module stays
    importable without the heavy ``HermesToolRegistry`` / middleware
    machinery — the loader passes whatever the caller hands it. v1 callers
    can pass plain dicts; production wiring (in ``agent.py``) will pass the
    real singletons.
    """

    def __init__(
        self,
        *,
        registry: Any,
        memory_registry: Any,
        slash_commands: Any,
        hooks: Any,
        plugin: LoadedPlugin | None = None,
    ) -> None:
        self.registry = registry
        self.memory_registry = memory_registry
        self.slash_commands = slash_commands
        self.hooks = hooks
        # ``plugin`` is set by the loader before calling ``register_fn`` so the
        # context can track which plugin owns each registration (for
        # introspection / unload).
        self._plugin = plugin or LoadedPlugin(name="<anonymous>")

    # ── tool registration ──

    def register_tool(self, tool: Any, *, toolset: str) -> None:
        """Register a LangChain ``BaseTool`` (or compatible) under ``toolset``."""
        if hasattr(self.registry, "register"):
            self.registry.register(tool, toolset=toolset)
        elif isinstance(self.registry, dict):
            self.registry.setdefault(toolset, []).append(tool)
        else:
            raise TypeError(f"PluginContext.register_tool: unknown registry type {type(self.registry).__name__}")
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "<tool>")
        self._plugin.tools_registered.append(name)
        logger.debug("Plugin %s registered tool %s in toolset %s", self._plugin.name, name, toolset)

    # ── memory provider (single-select) ──

    def register_memory_provider(self, name: str, cls: Any) -> None:
        """Register a ``MemoryProvider`` subclass under ``name``.

        Memory providers are single-select — only one is active per session,
        chosen via ``[memory] provider = "..."`` in config. Registering two
        with the same name overwrites the first (last-loaded wins; matches
        Hermes's user-over-bundled precedence).
        """
        if hasattr(self.memory_registry, "register"):
            self.memory_registry.register(name, cls)
        elif isinstance(self.memory_registry, dict):
            self.memory_registry[name] = cls
        else:
            raise TypeError(
                f"PluginContext.register_memory_provider: unknown registry type {type(self.memory_registry).__name__}"
            )
        logger.debug("Plugin %s registered memory provider %s", self._plugin.name, name)

    # ── context engine (single-select) ──

    def register_context_engine(self, name: str, cls: Any) -> None:
        """Register a context-engine implementation under ``name``.

        Same single-select semantics as memory providers — selection happens
        through config (default: ``HermesCompressionMiddleware``).
        """
        # We reuse the memory_registry slot via a namespaced key so the
        # loader doesn't need a fourth registry argument for v1.
        key = f"context_engine:{name}"
        if hasattr(self.memory_registry, "register"):
            self.memory_registry.register(key, cls)
        elif isinstance(self.memory_registry, dict):
            self.memory_registry[key] = cls
        else:
            raise TypeError(f"PluginContext.register_context_engine: unknown registry type {type(self.memory_registry).__name__}")
        logger.debug("Plugin %s registered context engine %s", self._plugin.name, name)

    # ── hooks ──

    def register_hook(self, name: str, fn: Callable[..., Any]) -> None:
        """Subscribe ``fn`` to lifecycle hook ``name``.

        ``name`` must be one of ``VALID_HOOKS`` — typos raise ``ValueError``
        so plugin authors find them at install time. v1 wires a subset of
        hooks through deepagents middleware; the others register but no-op
        until ``PluginEventBus`` ships in v0.2.
        """
        if name not in VALID_HOOKS:
            raise ValueError(f"Unknown hook {name!r}. Valid hooks: {sorted(VALID_HOOKS)}")
        if not callable(fn):
            raise TypeError(f"Hook callback must be callable, got {type(fn).__name__}")

        if hasattr(self.hooks, "register"):
            self.hooks.register(name, fn)
        elif isinstance(self.hooks, dict):
            self.hooks.setdefault(name, []).append(fn)
        else:
            raise TypeError(f"PluginContext.register_hook: unknown hooks store {type(self.hooks).__name__}")
        # Also append to the module-level registry that PluginEventBus reads.
        # Loader-provided ``hooks`` store stays the authoritative per-discovery
        # record; the global mirror exists so the event bus has a stable
        # lookup target without being threaded through every constructor.
        _GLOBAL_HOOK_REGISTRY[name].append(fn)
        self._plugin.hooks_registered.append(name)
        if name not in _WIRED_HOOKS_V1:
            logger.debug(
                "Plugin %s registered hook %r (not yet wired in v0.1 — registration tracked, "
                "callback is a no-op until PluginEventBus ships in v0.2)",
                self._plugin.name,
                name,
            )

    # ── slash commands ──

    def register_slash_command(self, name: str, fn: Callable[..., Any]) -> None:
        """Add ``/name`` to the chat REPL command table.

        ``name`` may include or omit the leading ``/`` — we normalize.
        Collisions with built-in commands are rejected with a warning so the
        plugin doesn't silently shadow ``/help`` or ``/quit``.
        """
        clean = name.strip().lstrip("/").lower().replace(" ", "-")
        if not clean:
            raise ValueError("slash command name cannot be empty")
        if not callable(fn):
            raise TypeError(f"Slash command handler must be callable, got {type(fn).__name__}")

        # Reject built-in collision (best-effort; the CLI may not be imported yet).
        try:
            from deepagent_hermes.cli import BUILTIN_SLASH_COMMANDS

            if clean in BUILTIN_SLASH_COMMANDS:
                logger.warning(
                    "Plugin %s tried to register /%s, which is a built-in. Skipping.",
                    self._plugin.name,
                    clean,
                )
                return
        except Exception:
            pass

        if hasattr(self.slash_commands, "register"):
            self.slash_commands.register(clean, fn)
        elif isinstance(self.slash_commands, dict):
            self.slash_commands[clean] = fn
        else:
            raise TypeError(f"PluginContext.register_slash_command: unknown registry type {type(self.slash_commands).__name__}")
        self._plugin.commands_registered.append(clean)
        logger.debug("Plugin %s registered slash command /%s", self._plugin.name, clean)


__all__ = [
    "VALID_HOOKS",
    "LoadedPlugin",
    "PluginContext",
    "get_global_hook_registry",
]
