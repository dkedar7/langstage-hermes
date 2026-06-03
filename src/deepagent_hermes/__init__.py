"""deepagent-hermes — closed-loop reflection / skill-creation agent on LangGraph + deepagents.

Faithful reproduction of Nous Research's Hermes Agent design ideas. See SPEC.md and NOTICE.
"""

__version__ = "0.1.0"

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
        from deepagent_hermes.agent import create_hermes_agent

        return create_hermes_agent
    if name == "HermesConfig":
        from deepagent_hermes.config import HermesConfig

        return HermesConfig
    if name == "HermesState":
        from deepagent_hermes.state import HermesState

        return HermesState
    raise AttributeError(f"module 'deepagent_hermes' has no attribute {name!r}")
