"""`tools --toolset <unknown>` is a named-target-not-found lookup: exit 1 (gh #134)."""

from __future__ import annotations

from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.tools.toolsets import IMPLEMENTED_TOOLSETS, TOOLSETS


def test_unknown_toolset_exits_1():
    res = CliRunner().invoke(cli, ["tools", "--toolset", "nope"])
    assert res.exit_code == 1, res.output
    assert "No toolset named 'nope'" in res.output


def test_known_toolset_exits_0():
    res = CliRunner().invoke(cli, ["tools", "--toolset", "file"])
    assert res.exit_code == 0, res.output
    assert "file" in res.output


def test_help_example_names_a_real_toolset():
    """The help advertised a `filesystem` toolset that doesn't exist; it's `file`."""
    res = CliRunner().invoke(cli, ["tools", "--help"])
    assert res.exit_code == 0
    assert "filesystem" not in res.output
    assert "file" in TOOLSETS


def test_declared_but_filtered_toolset_is_not_a_lookup_failure():
    """A real (stubbed) toolset hidden by --implemented-only exists, so it isn't a
    not-found; the message says why nothing was shown."""
    stubbed = sorted(set(TOOLSETS) - set(IMPLEMENTED_TOOLSETS))
    if not stubbed:  # pragma: no cover - every toolset implemented
        return
    res = CliRunner().invoke(cli, ["tools", "--toolset", stubbed[0], "--implemented-only"])
    assert res.exit_code == 0, res.output
    assert "not implemented" in res.output
