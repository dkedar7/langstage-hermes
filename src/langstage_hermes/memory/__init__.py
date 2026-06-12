"""Frozen-snapshot memory tool + MemoryProvider plugin ABC.

Public re-exports keep ``from langstage_hermes.memory import X`` short for
the rest of the package. See the submodule docstrings for design notes:

- ``tool``: ``MemoryToolMiddleware``, the ``memory`` tool, snapshot helpers.
- ``provider``: ``MemoryProvider`` ABC + provider registry.
- ``threat_patterns``: prompt-injection / promptware / exfiltration scanner.
"""

from __future__ import annotations

from langstage_hermes.memory.provider import (
    MemoryProvider,
    NoopMemoryProvider,
    RecallMode,
    available_providers,
    get_provider,
    register_provider,
)
from langstage_hermes.memory.threat_patterns import scan, scan_for_threats
from langstage_hermes.memory.tool import (
    DEFAULT_MEMORY_CHAR_LIMIT,
    DEFAULT_USER_CHAR_LIMIT,
    ENTRY_DELIMITER,
    MemoryStateExt,
    MemoryToolMiddleware,
    build_snapshot,
)

__all__ = [
    "DEFAULT_MEMORY_CHAR_LIMIT",
    "DEFAULT_USER_CHAR_LIMIT",
    "ENTRY_DELIMITER",
    "MemoryProvider",
    "MemoryStateExt",
    "MemoryToolMiddleware",
    "NoopMemoryProvider",
    "RecallMode",
    "available_providers",
    "build_snapshot",
    "get_provider",
    "register_provider",
    "scan",
    "scan_for_threats",
]
