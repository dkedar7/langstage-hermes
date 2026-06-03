"""Plugin discovery (4 sources) + PluginContext + lifecycle hook registration.

Public surface (SPEC §15):

  - :class:`deepagent_hermes.plugins.context.PluginContext`     — registration handle
  - :class:`deepagent_hermes.plugins.context.LoadedPlugin`      — discovery record
  - :data:`deepagent_hermes.plugins.context.VALID_HOOKS`        — 17 hook names
  - :class:`deepagent_hermes.plugins.loader.HermesPluginLoader` — 4-source discovery
"""

from deepagent_hermes.plugins.context import (
    VALID_HOOKS,
    LoadedPlugin,
    PluginContext,
    get_global_hook_registry,
)
from deepagent_hermes.plugins.event_bus import PluginEventBus
from deepagent_hermes.plugins.loader import (
    ENTRY_POINTS_GROUP,
    HermesPluginLoader,
)

__all__ = [
    "ENTRY_POINTS_GROUP",
    "VALID_HOOKS",
    "HermesPluginLoader",
    "LoadedPlugin",
    "PluginContext",
    "PluginEventBus",
    "get_global_hook_registry",
]
