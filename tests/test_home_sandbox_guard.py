"""The test suite must never touch a developer's real hermes home.

``tests/conftest.py`` points HERMES_HOME / LANGSTAGE_HERMES_HOME /
DEEPAGENT_HERMES_HOME (and HOME / USERPROFILE) at temp dirs before any test runs.
These tests prove it: in-process, and by running a suite file that writes
``state.db`` without setting a home, in a child pytest whose environment points
every home variable at a "real" directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from langstage_hermes.config import hermes_home

REPO = Path(__file__).resolve().parents[1]


def test_home_resolves_inside_the_sandbox(_sandbox_hermes_home: Path):
    assert hermes_home() == _sandbox_hermes_home
    for var in ("HERMES_HOME", "LANGSTAGE_HERMES_HOME", "DEEPAGENT_HERMES_HOME"):
        assert os.environ[var] == str(_sandbox_hermes_home)
    assert "sandbox_home" in str(Path.home())  # the ~ default is sandboxed too


def test_monkeypatched_home_still_wins(monkeypatch, tmp_path: Path):
    mine = tmp_path / "mine"
    monkeypatch.setenv("LANGSTAGE_HERMES_HOME", str(mine))
    assert hermes_home() == mine


def test_suite_leaves_a_real_hermes_home_untouched(tmp_path: Path):
    real = tmp_path / "real-hermes-home"
    real.mkdir()
    sentinel = real / "USER_DATA.txt"
    sentinel.write_text("do not touch", encoding="utf-8")
    before = {p.name: p.stat().st_mtime_ns for p in real.iterdir()}

    env = dict(os.environ)
    for var in ("HERMES_HOME", "LANGSTAGE_HERMES_HOME", "DEEPAGENT_HERMES_HOME"):
        env[var] = str(real)
    env.pop("PYTEST_CURRENT_TEST", None)
    # test_rename_shim.py writes <home>/state.db via the CLI without setting a home
    # of its own; before the conftest guard it wrote into the ambient HERMES_HOME.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_rename_shim.py"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    after = {p.name: p.stat().st_mtime_ns for p in real.iterdir()}
    assert after == before, f"suite wrote into the real home: {sorted(set(after) - set(before))}"
