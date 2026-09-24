"""Cron daemon (60s tick) + ``tick()`` + ``run_job()`` for ``langstage-hermes``.

SPEC §14 reproduction. Single long-running process (``python -m
langstage_hermes.cron``) sweeps ``<HERMES_HOME>/cron/jobs.json`` every
``cfg.cron_tick_seconds`` seconds, runs each due job, and writes output.

A file lock at ``<HERMES_HOME>/cron/.tick.lock`` prevents two daemon
processes from double-firing. ``filelock`` (a declared dependency) holds an OS
lock the kernel releases when the process dies, so a crash never strands it.
If ``filelock`` is somehow missing we fall back to an ``O_CREAT | O_EXCL``
lockfile that records the owner's PID and is reclaimed when that PID is dead.

Job execution paths:

  - ``no_agent=True``  →  ``subprocess.run`` the script, deliver stdout verbatim.
  - ``no_agent=False`` →  spawn ``create_hermes_agent`` with ``platform="cron"``,
                          restricted toolset (always strips ``cronjob`` /
                          ``messaging`` / ``clarify`` per SPEC §14.3), invoke
                          synchronously with the job prompt, capture the final
                          ``AIMessage`` content.

Output is appended to ``<HERMES_HOME>/cron/output/{job_id}/{timestamp}.md``.
If the agent response starts with ``"[SILENT]"`` we save to disk but suppress
delivery — Hermes's "nothing new to report" convention.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from langstage_core.host.config import _env_bool

from langstage_hermes.cron import jobs as cron_jobs
from langstage_hermes.cron.deliverers import get_deliverer

logger = logging.getLogger(__name__)

# Sentinel prefix on agent output → save locally, skip delivery.
SILENT_MARKER = "[SILENT]"

# Toolsets a cron-spawned agent must never receive (per SPEC §14.3).
_CRON_ALWAYS_STRIPPED = ("cronjob", "messaging", "clarify")


# ── failure logging (gh #111) ──────────────────────────────────────


def _debug_enabled() -> bool:
    """True when full tracebacks are wanted on the console.

    Follows the resolved ``debug`` setting
    (:attr:`langstage_core.host.config.HostConfig.debug`) through the whole config chain:
    ``LANGSTAGE_DEBUG``, the legacy ``DEEPAGENT_DEBUG``, and ``debug = true`` in a TOML
    file, the same way langstage-core resolves it for error-frame tracebacks (core
    1.0.36, #137). Reading only ``LANGSTAGE_DEBUG`` ignored the other two. Off by default
    so the automation surfaces (``cron run-due`` and the long-running daemon) stay quiet.
    Resolved per call, and only on the failure path, so a healthy tick never pays for it.
    If resolution itself fails, it falls back to the env switch alone rather than
    masking the original failure.
    """
    try:
        from langstage_hermes.config import HermesConfig

        return bool(HermesConfig.resolve().debug)
    except Exception:  # never let the debug lookup hide the real error
        return _env_bool(os.getenv("LANGSTAGE_DEBUG"))


def _log_cron_failure(msg: str, *args: Any, exc: BaseException) -> None:
    """Log a cron failure without flooding the console (gh #111).

    Default path: one clean ``WARNING`` line — ``"<msg>: <exc>"`` — carrying the
    same concise cause already surfaced structurally in ``last_status``/``error``
    and in ``run-due --json``, so the raw ~126-line traceback would be pure
    duplicate noise on an automation surface. Under ``LANGSTAGE_DEBUG`` the full
    traceback is preserved, at ``ERROR`` with ``exc_info``. Either way the caller
    still records the run as failed and returns unchanged — only console noise
    changes.
    """
    if _debug_enabled():
        logger.error(msg, *args, exc_info=True)
    else:
        logger.warning("%s: %s", msg % args if args else msg, exc)


# ── tick lock ──────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    """True if process ``pid`` is (probably) still running.

    Errs on the side of "alive": only a definite "no such process" answer lets the
    caller reclaim a lockfile, so a live daemon is never stolen from.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) on Windows TERMINATES the process, so ask the kernel instead.
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        process_query_limited_information = 0x1000
        still_active = 259
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            # ERROR_ACCESS_DENIED (5): the process exists but belongs to someone else.
            return kernel32.GetLastError() == 5
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except OSError:
        return True
    return True


def _lockfile_is_stale(lock_path: Path) -> bool:
    """True when ``lock_path`` was left behind by a daemon that is no longer running.

    The fallback lock records its owner's PID. An empty or unparseable file (a crash
    between create and write) is stale too, as is a dead PID.
    """
    try:
        raw = lock_path.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        pid = int(raw)
    except ValueError:
        return True
    if pid == os.getpid():
        return False
    return not _pid_alive(pid)


@contextmanager
def _tick_lock() -> Iterator[None]:
    """Acquire the cross-process tick lock; yield + release on exit.

    ``filelock`` first: its OS-level lock is released by the kernel when the
    process dies, so a SIGTERM / OOM kill / reboot can't wedge the next start.
    On ImportError fall back to an atomic ``O_CREAT | O_EXCL`` lockfile holding
    the owner's PID. A lockfile whose PID is dead (or that is empty) is stale:
    it is reclaimed once instead of blocking the daemon forever (gh #136). Only a
    lock held by a live process refuses the start.
    """
    lock_path = cron_jobs.tick_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from filelock import FileLock, Timeout
    except ImportError:
        FileLock = None  # type: ignore[assignment,misc]

    if FileLock is not None:
        lock = FileLock(str(lock_path) + ".flock", timeout=0)
        try:
            lock.acquire()
        except Timeout as e:
            raise RuntimeError(f"Another cron daemon already holds {lock_path}.flock") from e
        try:
            yield
        finally:
            try:
                lock.release()
            except Exception:  # pragma: no cover
                pass
        return

    with _pid_lockfile(lock_path):
        yield


@contextmanager
def _pid_lockfile(lock_path: Path) -> Iterator[None]:
    """The ``O_CREAT | O_EXCL`` fallback lock, with stale-PID recovery (gh #136)."""
    fd: int | None = None
    for attempt in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as e:
            if attempt == 0 and _lockfile_is_stale(lock_path):
                logger.warning("Reclaiming stale cron lockfile %s (owner process is gone)", lock_path)
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
                except OSError as unlink_err:
                    raise RuntimeError(f"Stale lockfile at {lock_path} could not be removed: {unlink_err}") from e
                continue
            raise RuntimeError(f"Another cron daemon (PID in {lock_path}) is running.") from e
    assert fd is not None
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


# ── job execution ──────────────────────────────────────────────────


def _run_script(script_path: str, *, timeout: int = 300) -> tuple[bool, str]:
    """Run a script (under ``<HERMES_HOME>/scripts/``) and return (ok, stdout).

    ``.sh`` / ``.bash`` execute via ``bash``; everything else via the current
    Python interpreter. Non-zero exit / timeout → ``ok=False`` and stderr is
    folded into the returned text.
    """
    home = cron_jobs._hermes_home()
    candidate = Path(script_path)
    if not candidate.is_absolute():
        candidate = home / "scripts" / candidate
    candidate = candidate.expanduser()

    if not candidate.exists():
        return False, f"script not found: {candidate}"

    is_shell = candidate.suffix.lower() in {".sh", ".bash"}
    cmd: list[str] = ["bash", str(candidate)] if is_shell else [sys.executable, str(candidate)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return False, f"script timed out after {timeout}s: {e}"
    except OSError as e:
        return False, f"script exec failed: {e}"
    if proc.returncode != 0:
        return False, (proc.stdout or "") + (proc.stderr or "")
    return True, proc.stdout or ""


def _build_cron_response(job: dict[str, Any], *, prompt: str) -> tuple[bool, str]:
    """Spawn a Hermes agent for one cron job and return ``(ok, text)``.

    Mirrors ``_run_script``'s ``(ok, body)`` contract: ``ok`` is ``False`` only
    when the agent invoke *raises* (expired key, rate-limit, provider outage,
    …). A normal completion — including an empty one — is ``ok=True``. This lets
    ``run_job`` record a failed invoke as an actual failure and suppress
    delivery, instead of the error string masquerading as a successful result
    (gh #72).

    Pulled into its own function so ``run_job`` stays readable and tests can
    monkeypatch this seam without standing up a real LLM.
    """
    try:
        from langstage_hermes.agent import create_hermes_agent
    except ImportError as e:
        logger.warning(
            "Cron job %s: langstage_hermes.agent not available (%s). Returning placeholder response.",
            job.get("id"),
            e,
        )
        return True, f"[langstage-hermes.agent unavailable; prompt was] {prompt}"

    from langstage_hermes.config import HermesConfig

    overrides: dict[str, Any] = {}
    if job.get("model"):
        overrides["model_default"] = job["model"]
    if job.get("provider") and job["provider"] != "auto":
        overrides["model_provider"] = job["provider"]
    # Always-strip toolsets plus any user policy.
    disabled = list(_CRON_ALWAYS_STRIPPED)
    cfg = HermesConfig.resolve(overrides=overrides)
    for name in cfg.agent_disabled_toolsets:
        if name and name not in disabled:
            disabled.append(name)
    cfg = HermesConfig.resolve(overrides={**overrides, "agent_disabled_toolsets": disabled})

    # The agent factory may accept a richer context object; we pass what we
    # know. If the signature differs we still try a positional invoke and
    # fall back to a string placeholder so the daemon never crashes a tick.
    try:
        agent = create_hermes_agent(cfg)
    except TypeError:  # pragma: no cover - factory shape may evolve
        agent = create_hermes_agent()  # type: ignore[call-arg]

    try:
        from langchain_core.messages import HumanMessage

        result = agent.invoke(
            {
                "messages": [HumanMessage(content=prompt)],
                "session_id": f"cron_{job.get('id', '?')}",
                "active_skills": list(job.get("skills") or []),
            }
        )
    except Exception as e:
        _log_cron_failure("Cron job %s: agent invoke failed", job.get("id"), exc=e)
        return False, f"[agent invoke failed: {e}]"

    messages = (result or {}).get("messages") or []
    # Return content of the last AIMessage (or any non-empty message tail).
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return True, content
        if isinstance(content, list):
            # Anthropic-style content blocks → concat text parts.
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
            if text.strip():
                return True, text
    return True, ""


def _build_output_doc(
    job: dict[str, Any],
    *,
    started_at: datetime,
    body: str,
    mode: str = "agent",
) -> str:
    """Render the markdown saved under ``cron/output/{job_id}/{timestamp}.md``."""
    return (
        f"## Job {job.get('name', job.get('id', '?'))}\n"
        f"## Started at: {started_at.isoformat()}\n\n"
        f"_Mode_: {mode}\n\n"
        f"---\n\n"
        f"{body}\n"
    )


def _deliver_output(
    job: dict[str, Any],
    output: str,
    output_path: Path | None,
) -> None:
    """Dispatch ``output`` to the deliverer named by ``job['deliver']``.

    SILENT_MARKER suppression is honored here (no deliverer is invoked).
    Unknown deliverer names log a warning and fall back to ``"local"``
    so a typo in jobs.json never silently drops output. The selected
    deliverer's exceptions propagate; callers (``run_job``) catch and
    record them as ``last_delivery_error``.
    """
    if output.startswith(SILENT_MARKER):
        return
    deliverer_name = (job.get("deliver") or "local").lower()
    if deliverer_name == "origin":
        # 'origin' = "use whatever the job's origin specified"; for v1
        # we fall through to local. Origin-aware routing is future work.
        deliverer_name = "local"
    deliverer_cls = get_deliverer(deliverer_name)
    if deliverer_cls is None:
        logger.warning("cron: no deliverer registered for %r; using local", deliverer_name)
        deliverer_cls = get_deliverer("local")
        assert deliverer_cls is not None, "LocalDeliverer should always be registered"
    try:
        deliverer_cls().deliver(job, output, output_path=output_path)
    except Exception as e:
        _log_cron_failure("cron deliverer %r failed", deliverer_name, exc=e)
        # caller updates last_delivery_error
        raise


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one cron job; persist output; update bookkeeping.

    Returns a dict ``{"job_id", "success", "output_path", "silent", "error"}``
    so callers (tests, ``tick()``, the ``cronjob`` ``run-now`` action) can
    inspect outcome without re-reading jobs.json.
    """
    job_id = job.get("id", "?")
    started = cron_jobs._now()
    silent = False
    success = True
    error: str | None = None
    delivery_error: str | None = None
    output_path: Path | None = None

    try:
        if job.get("no_agent"):
            script = job.get("script")
            if not script:
                raise ValueError("no_agent=True but no script set")
            ok, body = _run_script(script)
            success = ok
            if not ok:
                error = body
            doc = _build_output_doc(job, started_at=started, body=body, mode="no_agent")
            output_path = cron_jobs.save_job_output(job_id, doc)
            silent = not body.strip()
            if not silent and ok:
                try:
                    _deliver_output(job, body, output_path)
                except Exception as e:  # pragma: no cover - defensive
                    delivery_error = f"{type(e).__name__}: {e}"
        else:
            prompt = job.get("prompt") or ""
            ok, response = _build_cron_response(job, prompt=prompt)
            success = ok
            if not ok:
                error = response
            doc = _build_output_doc(job, started_at=started, body=response, mode="agent")
            output_path = cron_jobs.save_job_output(job_id, doc)
            silent = response.strip().startswith(SILENT_MARKER)
            # Mirror the script path: only deliver a genuine success. A failed
            # invoke's error string must never be delivered as the result (gh #72).
            if not silent and ok:
                try:
                    _deliver_output(job, response, output_path)
                except Exception as e:  # pragma: no cover - defensive
                    delivery_error = f"{type(e).__name__}: {e}"
    except Exception as e:  # pragma: no cover - top-level safety net
        _log_cron_failure("Cron job %s crashed", job_id, exc=e)
        success = False
        error = f"{type(e).__name__}: {e}"

    cron_jobs.mark_job_run(
        job_id,
        success=success,
        error=error,
        delivery_error=delivery_error,
    )
    return {
        "job_id": job_id,
        "success": success,
        "output_path": str(output_path) if output_path else None,
        "silent": silent,
        "error": error,
        "delivery_error": delivery_error,
    }


# ── HermesCron daemon ──────────────────────────────────────────────


class HermesCron:
    """Long-running daemon driving the cron tick loop.

    ``tick()`` is a single sweep (one call per scheduled wake). ``run_forever``
    is the daemon entry point — holds the file lock, then sleeps
    ``tick_seconds`` between sweeps until ``stop()`` is called.
    """

    def __init__(self, *, tick_seconds: int = 60) -> None:
        self.tick_seconds = tick_seconds
        self._stop = threading.Event()

    # -- lifecycle --

    def stop(self) -> None:
        """Signal ``run_forever`` to exit at the next tick boundary."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # -- single tick (testable without daemon) --

    def tick(self) -> list[dict[str, Any]]:
        """Find due jobs and run each. Returns one result dict per job run."""
        results: list[dict[str, Any]] = []
        for job in cron_jobs.get_due_jobs():
            logger.info("Cron tick: running job %s (%s)", job.get("id"), job.get("name"))
            results.append(run_job(job))
        return results

    # -- daemon loop --

    def run_forever(self) -> None:
        """Hold the lock and tick every ``tick_seconds`` until stopped."""
        with _tick_lock():
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception as e:  # pragma: no cover - never let one bad tick kill the loop
                    _log_cron_failure("Cron tick raised; continuing", exc=e)
                self._stop.wait(self.tick_seconds)


__all__ = ["SILENT_MARKER", "HermesCron", "run_job"]
