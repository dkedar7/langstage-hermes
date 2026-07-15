"""Cron job CRUD + storage for ``langstage-hermes`` (SPEC §14).

Jobs live in ``<HERMES_HOME>/cron/jobs.json`` (mode 0600 best-effort on
Windows). Each job is a ~30-field dict — shape ported verbatim from Hermes's
``cron/jobs.py`` so existing job records load unchanged.

Schedule expressions:

  - ``"30s"`` / ``"5m"`` / ``"2h"`` / ``"1d"``  → interval (recurring every N)
  - ``"every 30m"``                              → same as above (alias)
  - ``"0 9 * * *"``                              → cron (5/6-field expression)
  - ``"once at 2026-06-15T09:00"``               → one-shot
  - ISO timestamp (``"2026-06-15T09:00"``)       → one-shot

Concurrency: ``_jobs_lock`` (in-process ``threading.Lock``) guards
load→modify→save cycles so parallel ticks don't clobber each other. The cron
daemon also takes a *file* lock (``.tick.lock``) to prevent two daemon
processes from double-firing; see ``cron/scheduler.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from croniter import croniter

    _HAS_CRONITER = True
except ImportError:  # pragma: no cover - croniter is a pinned dep
    croniter = None  # type: ignore[assignment]
    _HAS_CRONITER = False


# ── paths ───────────────────────────────────────────────────────────


def _hermes_home() -> Path:
    """Resolve HERMES_HOME at call time so tests can monkeypatch env vars."""
    from langstage_hermes.config import hermes_home

    return hermes_home()


def _cron_dir() -> Path:
    return _hermes_home() / "cron"


def _jobs_file() -> Path:
    return _cron_dir() / "jobs.json"


def _output_dir() -> Path:
    return _cron_dir() / "output"


def tick_lock_path() -> Path:
    """Filesystem lock for the cron daemon (one tick at a time across processes)."""
    return _cron_dir() / ".tick.lock"


# ── permissions helpers ────────────────────────────────────────────


def _secure_dir(path: Path) -> None:
    """Owner-only access (0700). No-op on Windows / unsupported FS."""
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass


def _secure_file(path: Path) -> None:
    """Owner-only read/write (0600). No-op on Windows / unsupported FS."""
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _ensure_dirs() -> None:
    """Create cron + output dirs with secure perms (best-effort on Windows)."""
    _cron_dir().mkdir(parents=True, exist_ok=True)
    _output_dir().mkdir(parents=True, exist_ok=True)
    _secure_dir(_cron_dir())
    _secure_dir(_output_dir())


# ── duration / schedule parsing ────────────────────────────────────


_DURATION_RE = re.compile(
    r"^(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|"
    r"h|hr|hrs|hour|hours|d|day|days)$"
)


def parse_duration(expr: str) -> int:
    """Parse a duration like ``"30s"`` / ``"5m"`` / ``"2h"`` / ``"7d"`` → seconds.

    Returns total seconds. Raises ``ValueError`` on garbage.
    """
    text = expr.strip().lower()
    m = _DURATION_RE.match(text)
    if not m:
        raise ValueError(f"Invalid duration {expr!r}. Use forms like '30s', '5m', '2h', '7d'.")
    value = int(m.group(1))
    unit = m.group(2)[0]  # s / m / h / d
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]


def parse_schedule(expr: str) -> dict[str, Any]:
    """Parse a schedule expression into a structured ``{"kind": ..., ...}`` dict.

    Returns one of:
      - ``{"kind": "interval", "seconds": N, "expr": "...", "display": "..."}``
      - ``{"kind": "cron",     "expr": "...",            "display": "..."}``
      - ``{"kind": "once",     "run_at": "ISO-8601",     "display": "..."}``
    """
    original = expr.strip()
    text = original.lower()

    # "every Xm/Xh/Xd" → recurring interval
    if text.startswith("every "):
        seconds = parse_duration(original[6:].strip())
        return {
            "kind": "interval",
            "seconds": seconds,
            "expr": original,
            "display": f"every {seconds // 60}m" if seconds >= 60 else f"every {seconds}s",
        }

    # "once at TIMESTAMP" → one-shot
    if text.startswith("once at "):
        ts = original[8:].strip()
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid 'once at' timestamp {ts!r}: {e}") from e
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return {
            "kind": "once",
            "run_at": dt.isoformat(),
            "expr": original,
            "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}",
        }

    # 5/6-field cron expression
    parts = original.split()
    if len(parts) in (5, 6) and all(re.match(r"^[\d\*\-,/]+$", p) for p in parts[:5]):
        if not _HAS_CRONITER:
            raise ValueError("Cron expressions require the 'croniter' package (pip install croniter).")
        try:
            croniter(original)
        except Exception as e:
            raise ValueError(f"Invalid cron expression {original!r}: {e}") from e
        return {"kind": "cron", "expr": original, "display": original}

    # ISO timestamp → one-shot
    if "T" in original or re.match(r"^\d{4}-\d{2}-\d{2}", original):
        try:
            dt = datetime.fromisoformat(original.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.astimezone()
            return {
                "kind": "once",
                "run_at": dt.isoformat(),
                "expr": original,
                "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}",
            }
        except ValueError:
            pass

    # Bare duration → recurring interval (identical to "every <duration>").
    # The module docstring, the `cronjob` tool example (tool.py), and the
    # invalid-schedule hint below all present a bare "30m" as a recurring
    # interval alongside "every 2h" — so "30m" must mean the same thing as
    # "every 30m", not a silent one-shot that fires once and stops (gh #71).
    # Use "once at <ts>" / an ISO timestamp to request a genuine one-shot.
    try:
        seconds = parse_duration(original)
    except ValueError:
        pass
    else:
        return {
            "kind": "interval",
            "seconds": seconds,
            "expr": original,
            "display": f"every {seconds // 60}m" if seconds >= 60 else f"every {seconds}s",
        }

    raise ValueError(f"Invalid schedule {expr!r}. Try '30m' / 'every 2h' / '0 9 * * *' / 'once at 2026-06-15T09:00'.")


# ── time helpers ────────────────────────────────────────────────────


def _now() -> datetime:
    """Timezone-aware 'now' in local time (matches Hermes ``hermes_time.now``)."""
    return datetime.now().astimezone()


def _ensure_aware(dt: datetime) -> datetime:
    """Coerce a naive datetime to local-tz aware (back-compat with old records)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt


def compute_next_run(
    job: dict[str, Any],
    *,
    now: datetime | None = None,
) -> datetime | None:
    """Return the next scheduled run time for ``job`` (or ``None`` if exhausted).

    Accepts the full job dict (so ``last_run_at`` anchors the next run for
    interval/cron schedules across restarts).
    """
    now = now or _now()
    schedule = job.get("schedule") or {}
    kind = schedule.get("kind")
    last_run_at = job.get("last_run_at")

    if kind == "once":
        # One-shot: only eligible if it hasn't already run.
        if last_run_at:
            return None
        run_at = schedule.get("run_at")
        if not run_at:
            return None
        return _ensure_aware(datetime.fromisoformat(run_at))

    if kind == "interval":
        seconds = int(schedule.get("seconds") or schedule.get("minutes", 0) * 60)
        if seconds <= 0:
            return None
        if last_run_at:
            base = _ensure_aware(datetime.fromisoformat(last_run_at))
            return base + timedelta(seconds=seconds)
        return now + timedelta(seconds=seconds)

    if kind == "cron":
        if not _HAS_CRONITER:
            logger.warning(
                "Cron schedule %r requires croniter; skipping next_run computation",
                schedule.get("expr"),
            )
            return None
        base = _ensure_aware(datetime.fromisoformat(last_run_at)) if last_run_at else now
        c = croniter(schedule["expr"], base)
        return c.get_next(datetime)

    return None


# ── storage I/O ─────────────────────────────────────────────────────


_jobs_lock = threading.Lock()


def _load() -> list[dict[str, Any]]:
    """Read jobs.json from disk (returns ``[]`` if absent)."""
    _ensure_dirs()
    path = _jobs_file()
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return list(json.load(f).get("jobs", []))
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to read %s: %s", path, e)
        raise


def _save(jobs: list[dict[str, Any]]) -> None:
    """Atomically write jobs.json (tmp + rename) with 0600 perms."""
    _ensure_dirs()
    path = _jobs_file()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".jobs_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                {"jobs": jobs, "updated_at": _now().isoformat()},
                f,
                indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _secure_file(path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── normalization helpers ──────────────────────────────────────────


def _normalize_skill_list(skill: str | None, skills: Any) -> list[str]:
    """Merge legacy ``skill`` (single) + new ``skills`` (list) into a unique ordered list."""
    if skills is None:
        items: list[Any] = [skill] if skill else []
    elif isinstance(skills, str):
        items = [skills]
    else:
        items = list(skills)
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


# ── public API ─────────────────────────────────────────────────────


def create_job(
    prompt: str | None,
    schedule: str,
    *,
    name: str | None = None,
    skills: list[str] | None = None,
    skill: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    script: str | None = None,
    no_agent: bool = False,
    context_from: list[str] | str | None = None,
    repeat: int | None = None,
    deliver: str = "local",
    origin: dict[str, Any] | None = None,
    enabled_toolsets: list[str] | None = None,
    workdir: str | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Create + persist a new cron job. Returns the created job dict.

    Job IDs are 12-char hex (uuid4 prefix). The full 30-field shape matches
    SPEC §14 / Hermes verbatim so existing tooling reads our jobs unchanged.
    """
    parsed = parse_schedule(schedule)
    normalized_skills = _normalize_skill_list(skill, skills)

    if no_agent and not script:
        raise ValueError("no_agent=True requires a script — with no agent and no script there is nothing for the job to run.")

    # Auto-set repeat=1 for one-shot schedules.
    if parsed["kind"] == "once" and repeat is None:
        repeat = 1
    if repeat is not None and repeat <= 0:
        repeat = None

    if isinstance(context_from, str):
        ctx_from: list[str] | None = [context_from.strip()] if context_from.strip() else None
    elif isinstance(context_from, list):
        ctx_from = [str(j).strip() for j in context_from if str(j).strip()] or None
    else:
        ctx_from = None

    prompt_text = "" if prompt is None else str(prompt)
    label_source = prompt_text or (normalized_skills[0] if normalized_skills else "") or (script or "") or "cron job"
    job_id = uuid.uuid4().hex[:12]
    now_iso = _now().isoformat()

    job: dict[str, Any] = {
        "id": job_id,
        "name": (name or label_source[:50]).strip() or "cron job",
        "prompt": prompt_text,
        "skills": normalized_skills,
        "skill": normalized_skills[0] if normalized_skills else None,
        "model": model.strip() if isinstance(model, str) and model.strip() else None,
        "provider": (provider.strip() if isinstance(provider, str) and provider.strip() else None),
        "base_url": (base_url.strip().rstrip("/") if isinstance(base_url, str) and base_url.strip() else None),
        "script": script.strip() if isinstance(script, str) and script.strip() else None,
        "no_agent": bool(no_agent),
        "context_from": ctx_from,
        "schedule": parsed,
        "schedule_display": parsed.get("display", schedule),
        "repeat": {"times": repeat, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": now_iso,
        "next_run_at": None,  # computed below
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "last_delivery_error": None,
        "deliver": deliver,
        "origin": origin,
        "enabled_toolsets": list(enabled_toolsets) if enabled_toolsets else None,
        "workdir": workdir,
        "profile": profile,
    }
    next_run = compute_next_run(job)
    job["next_run_at"] = next_run.isoformat() if next_run else None

    with _jobs_lock:
        jobs = _load()
        jobs.append(job)
        _save(jobs)

    return job


def list_jobs(*, include_disabled: bool = True) -> list[dict[str, Any]]:
    """Return all stored jobs. Disabled jobs included by default."""
    with _jobs_lock:
        jobs = _load()
    if not include_disabled:
        jobs = [j for j in jobs if j.get("enabled", True)]
    return jobs


def get_job(job_id: str) -> dict[str, Any] | None:
    """Lookup a job by exact ID. Returns ``None`` if missing."""
    with _jobs_lock:
        for job in _load():
            if job.get("id") == job_id:
                return job
    return None


def delete_job(job_id: str) -> bool:
    """Remove a job by ID. Returns ``True`` if a job was removed."""
    with _jobs_lock:
        jobs = _load()
        new = [j for j in jobs if j.get("id") != job_id]
        if len(new) == len(jobs):
            return False
        _save(new)
    return True


def pause_job(job_id: str, reason: str = "") -> bool:
    """Disable + mark a job ``paused``. Returns ``False`` if the job is missing."""
    return update_job(
        job_id,
        enabled=False,
        state="paused",
        paused_at=_now().isoformat(),
        paused_reason=reason or None,
    )


def resume_job(job_id: str) -> bool:
    """Re-enable a paused job and recompute its ``next_run_at`` from now."""
    job = get_job(job_id)
    if not job:
        return False
    next_run = compute_next_run({**job, "last_run_at": None})
    return update_job(
        job_id,
        enabled=True,
        state="scheduled",
        paused_at=None,
        paused_reason=None,
        next_run_at=next_run.isoformat() if next_run else None,
    )


def update_job(job_id: str, **fields: Any) -> bool:
    """Patch ``job_id`` with ``fields`` (skip ``id`` — immutable). Returns success."""
    if "id" in fields:
        raise ValueError("cron job 'id' is immutable and cannot be updated")
    with _jobs_lock:
        jobs = _load()
        for i, job in enumerate(jobs):
            if job.get("id") != job_id:
                continue
            updated = {**job, **fields}
            # If schedule changed and not paused, recompute next_run_at.
            if "schedule" in fields and updated.get("state") != "paused":
                if isinstance(updated["schedule"], str):
                    updated["schedule"] = parse_schedule(updated["schedule"])
                next_run = compute_next_run(updated)
                updated["next_run_at"] = next_run.isoformat() if next_run else None
            jobs[i] = updated
            _save(jobs)
            return True
    return False


def mark_job_run(
    job_id: str,
    *,
    success: bool,
    error: str | None = None,
    delivery_error: str | None = None,
) -> None:
    """Record a completed run: timestamp + status + repeat-count bookkeeping.

    Auto-deletes one-shot jobs at their repeat limit; recurring jobs that
    fail to compute a next run are marked ``state="error"`` (not silently
    disabled).
    """
    with _jobs_lock:
        jobs = _load()
        for i, job in enumerate(jobs):
            if job.get("id") != job_id:
                continue
            now_iso = _now().isoformat()
            job["last_run_at"] = now_iso
            job["last_status"] = "ok" if success else "error"
            job["last_error"] = None if success else error
            job["last_delivery_error"] = delivery_error

            if job.get("repeat"):
                job["repeat"]["completed"] = job["repeat"].get("completed", 0) + 1
                times = job["repeat"].get("times")
                if times is not None and times > 0 and job["repeat"]["completed"] >= times:
                    jobs.pop(i)
                    _save(jobs)
                    return

            next_run = compute_next_run(job)
            job["next_run_at"] = next_run.isoformat() if next_run else None

            if job["next_run_at"] is None:
                kind = (job.get("schedule") or {}).get("kind")
                if kind in {"cron", "interval"}:
                    job["state"] = "error"
                    if not job.get("last_error"):
                        job["last_error"] = "Failed to compute next run for recurring schedule"
                else:
                    job["enabled"] = False
                    job["state"] = "completed"
            elif job.get("state") != "paused":
                job["state"] = "scheduled"

            jobs[i] = job
            _save(jobs)
            return


def get_due_jobs(*, now: datetime | None = None) -> list[dict[str, Any]]:
    """Return all enabled jobs whose ``next_run_at`` has elapsed."""
    now = now or _now()
    due: list[dict[str, Any]] = []
    for job in list_jobs(include_disabled=False):
        if not job.get("enabled", True):
            continue
        next_run = job.get("next_run_at")
        if not next_run:
            continue
        try:
            run_at = _ensure_aware(datetime.fromisoformat(next_run))
        except ValueError:
            continue
        if run_at <= now:
            due.append(job)
    return due


def job_output_dir(job_id: str) -> Path:
    """Resolve a job's output directory under ``<HERMES_HOME>/cron/output/``.

    Rejects path-escape attempts (``..``, separators, absolute paths) — job
    IDs are filesystem path components, so anything fancier is unsafe.
    """
    text = str(job_id or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"Invalid cron job id for output path: {job_id!r}")
    if Path(text).is_absolute() or Path(text).drive:
        raise ValueError(f"Invalid cron job id for output path: {job_id!r}")
    return _output_dir() / text


def save_job_output(job_id: str, body: str) -> Path:
    """Append a timestamped ``.md`` file under the job's output dir; return its path."""
    out_dir = job_output_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    _secure_dir(out_dir)
    timestamp = _now().strftime("%Y-%m-%d_%H-%M-%S")
    target = out_dir / f"{timestamp}.md"

    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".tmp", prefix=".out_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        _secure_file(target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


__all__ = [
    "compute_next_run",
    "create_job",
    "delete_job",
    "get_due_jobs",
    "get_job",
    "job_output_dir",
    "list_jobs",
    "mark_job_run",
    "parse_duration",
    "parse_schedule",
    "pause_job",
    "resume_job",
    "save_job_output",
    "tick_lock_path",
    "update_job",
]
