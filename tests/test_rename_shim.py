"""The deepagent_hermes → langstage_hermes rename ships a deprecated alias package."""

import sys

import pytest


def test_legacy_import_works_and_warns():
    for name in list(sys.modules):
        if name == "deepagent_hermes" or name.startswith("deepagent_hermes."):
            sys.modules.pop(name)
    with pytest.warns(DeprecationWarning, match="langstage_hermes"):
        import deepagent_hermes  # noqa: F401


def test_legacy_host_spec_path_still_imports():
    """The documented host integration (spec deepagent_hermes.agent:graph)
    keeps resolving — the submodule alias is the same module object."""
    import deepagent_hermes.agent as old_agent

    import langstage_hermes.agent as new_agent

    assert old_agent is new_agent
    assert hasattr(old_agent, "graph") or hasattr(old_agent, "create_hermes_agent")


def test_legacy_package_version_matches():
    import deepagent_hermes
    import langstage_hermes

    assert deepagent_hermes.__version__ == langstage_hermes.__version__
