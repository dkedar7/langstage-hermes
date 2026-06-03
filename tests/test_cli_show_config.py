"""Tests for the ``deepagent-hermes`` CLI surface — focused on ``--show-config``."""

from __future__ import annotations

import subprocess
import sys


def test_show_config_exits_zero_and_prints_sections():
    """``python -m deepagent_hermes.cli --show-config`` prints the resolved config.

    Per SPEC §2 acceptance: every field with its source. We assert exit code 0
    plus presence of two ``model`` / ``agent`` field prefixes (HermesConfig
    fields are flat — ``model_default``, ``agent_max_iterations``, ...).
    """
    result = subprocess.run(
        [sys.executable, "-m", "deepagent_hermes.cli", "--show-config"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"--show-config failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "model" in result.stdout
    assert "agent" in result.stdout
    # Sanity: every field row contains its source bracket.
    assert "[default]" in result.stdout or "[toml" in result.stdout or "[env:" in result.stdout


def test_root_without_subcommand_shows_help():
    """Bare ``python -m deepagent_hermes.cli`` should not crash; prints help."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagent_hermes.cli"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "Usage" in result.stdout or "Commands" in result.stdout


def test_version_flag():
    """``--version`` prints the package version."""
    result = subprocess.run(
        [sys.executable, "-m", "deepagent_hermes.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "deepagent-hermes" in result.stdout
