"""A stale cron tick lock (owner process dead) must not wedge the daemon (gh #136)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from langstage_hermes.cron import jobs as cron_jobs
from langstage_hermes.cron import scheduler


def _dead_pid() -> int:
    """The PID of a process that has already exited."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_filelock_is_a_declared_dependency():
    """The kernel-released lock must be the default path, not an optional upgrade."""
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    assert any(d.lower().startswith("filelock") for d in deps), deps


def test_pid_alive_for_self_and_dead():
    assert scheduler._pid_alive(os.getpid())
    assert not scheduler._pid_alive(_dead_pid())


def test_fallback_reclaims_lock_left_by_dead_pid(tmp_hermes_home: Path):
    lock = cron_jobs.tick_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(_dead_pid()), encoding="utf-8")

    with scheduler._pid_lockfile(lock):
        assert lock.read_text(encoding="utf-8") == str(os.getpid())
    assert not lock.exists()


@pytest.mark.parametrize("content", ["", "not-a-pid"])
def test_fallback_reclaims_empty_or_garbage_lock(tmp_hermes_home: Path, content: str):
    lock = cron_jobs.tick_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(content, encoding="utf-8")
    with scheduler._pid_lockfile(lock):
        pass
    assert not lock.exists()


def test_fallback_refuses_lock_held_by_live_process(tmp_hermes_home: Path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock = cron_jobs.tick_lock_path()
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(str(proc.pid), encoding="utf-8")
        with pytest.raises(RuntimeError, match="Another cron daemon"):
            with scheduler._pid_lockfile(lock):
                pass
        assert lock.read_text(encoding="utf-8") == str(proc.pid)  # left alone
    finally:
        proc.kill()
        proc.wait()


def test_tick_lock_ignores_a_stranded_fallback_lockfile(tmp_hermes_home: Path):
    """With filelock (the default) a `.tick.lock` stranded by a crashed daemon is
    irrelevant: the OS lock on `.tick.lock.flock` is what's checked."""
    lock = cron_jobs.tick_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(_dead_pid()), encoding="utf-8")
    with scheduler._tick_lock():
        pass
