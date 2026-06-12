"""Deprecated alias package: ``deepagent_hermes`` is now ``langstage_hermes``.

Kept for one transition window so host spec strings like
``deepagent_hermes.agent:graph`` and existing imports keep working. Use
``langstage_hermes`` (spec: ``langstage_hermes.agent:graph``) instead.
"""

import sys as _sys
import warnings as _warnings

import langstage_hermes as _new
from langstage_hermes import *  # noqa: F403
from langstage_hermes import agent, cli, config

_warnings.warn(
    "deepagent_hermes has been renamed to langstage_hermes; this alias package will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# The documented host-integration entry points stay importable under the
# old dotted paths (DEEPAGENT_AGENT_SPEC=deepagent_hermes.agent:graph).
_sys.modules[__name__ + ".agent"] = agent
_sys.modules[__name__ + ".cli"] = cli
_sys.modules[__name__ + ".config"] = config
__version__ = getattr(_new, "__version__", "0")
