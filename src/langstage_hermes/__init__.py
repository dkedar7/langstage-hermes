"""langstage-hermes — closed-loop reflection / skill-creation agent on LangGraph + deepagents.

Faithful reproduction of Nous Research's Hermes Agent design ideas. See SPEC.md and NOTICE.
"""

# Read the version from installed package metadata so it can never drift from
# pyproject.toml again (the hand-maintained constant was stuck at "0.1.4"
# through the 0.1.5 and 0.2.0 releases).
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("langstage-hermes")
except PackageNotFoundError:  # pragma: no cover — running from a source tree without install
    __version__ = "0.0.0+unknown"

# Re-exports populated by submodule integration (see agent.py).
# Subagents wire these in; importing here would create circular deps during build.
__all__ = [
    "HermesConfig",
    "HermesState",
    "__version__",
    "create_hermes_agent",
]


def __getattr__(name: str):
    """Lazy re-export to defer heavy imports until first access."""
    if name == "create_hermes_agent":
        from langstage_hermes.agent import create_hermes_agent

        return create_hermes_agent
    if name == "HermesConfig":
        from langstage_hermes.config import HermesConfig

        return HermesConfig
    if name == "HermesState":
        from langstage_hermes.state import HermesState

        return HermesState
    raise AttributeError(f"module 'langstage_hermes' has no attribute {name!r}")
