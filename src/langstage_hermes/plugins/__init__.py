"""Plugin discovery (4 sources) + PluginContext + lifecycle hook registration.

Public surface (SPEC §15):

  - :class:`langstage_hermes.plugins.context.PluginContext`     — registration handle
  - :class:`langstage_hermes.plugins.context.LoadedPlugin`      — discovery record
  - :data:`langstage_hermes.plugins.context.VALID_HOOKS`        — 17 hook names
  - :class:`langstage_hermes.plugins.loader.HermesPluginLoader` — 4-source discovery
"""

from langstage_hermes.plugins.context import (
    VALID_HOOKS,
    LoadedPlugin,
    PluginContext,
    get_global_hook_registry,
)
from langstage_hermes.plugins.event_bus import PluginEventBus
from langstage_hermes.plugins.loader import (
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
