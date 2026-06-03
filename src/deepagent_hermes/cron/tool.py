"""``cronjob`` agent tool — CRUD for scheduled jobs from inside the agent.

Exposes ``deepagent_hermes.cron.jobs`` to the LLM as a single multi-action
tool, mirroring Hermes's ``tools/cronjob_tools.py`` shape:

    cronjob(action="create",   prompt=..., schedule="30m")
    cronjob(action="list")
    cronjob(action="show",     id="abc123")
    cronjob(action="delete",   id="abc123")
    cronjob(action="pause",    id="abc123")
    cronjob(action="resume",   id="abc123")
    cronjob(action="run-now",  id="abc123")

Returns a short markdown string the model can quote back to the user.
"""

from __future__ import annotations

from typing import Any, Literal

from deepagent_hermes.cron import jobs as cron_jobs
from deepagent_hermes.cron.scheduler import run_job

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - langchain-core is required at runtime
    tool = None  # type: ignore[assignment]


def _format_job_summary(job: dict[str, Any]) -> str:
    """Single-line summary used by ``list`` and as the header of ``show``."""
    return (
        f"- `{job['id']}` **{job.get('name','?')}** "
        f"[{job.get('schedule_display','?')}] "
        f"state={job.get('state','?')} "
        f"next={job.get('next_run_at') or '—'}"
    )


def _format_job_detail(job: dict[str, Any]) -> str:
    """Multi-line markdown detail used by ``show``."""
    lines = [f"### Cron job `{job['id']}` — {job.get('name','?')}", ""]
    for key in (
        "schedule_display",
        "prompt",
        "skills",
        "model",
        "provider",
        "script",
        "no_agent",
        "context_from",
        "deliver",
        "enabled",
        "state",
        "created_at",
        "next_run_at",
        "last_run_at",
        "last_status",
        "last_error",
        "workdir",
        "profile",
    ):
        if key in job:
            lines.append(f"- **{key}**: `{job[key]!r}`")
    return "\n".join(lines)


def _cronjob_impl(
    action: str,
    *,
    id: str = "",
    prompt: str = "",
    schedule: str = "",
    name: str = "",
    skills: list[str] | None = None,
    model: str = "",
    no_agent: bool = False,
    deliver: str = "local",
) -> str:
    """Pure-Python implementation, decoupled from the ``@tool`` decorator.

    Lives separate from the tool wrapper so unit tests can call it directly
    without spinning up a LangGraph runtime.
    """
    act = action.lower().strip()

    if act == "create":
        if not schedule:
            return "Error: `schedule` is required for create (e.g. '30m', '0 9 * * *')."
        try:
            job = cron_jobs.create_job(
                prompt=prompt or None,
                schedule=schedule,
                name=name or None,
                skills=skills,
                model=model or None,
                no_agent=no_agent,
                deliver=deliver,
            )
        except (ValueError, RuntimeError) as e:
            return f"Error creating cron job: {e}"
        return (
            f"Created cron job `{job['id']}` ({job['name']}) "
            f"[{job['schedule_display']}], next run at {job['next_run_at']}."
        )

    if act == "list":
        items = cron_jobs.list_jobs()
        if not items:
            return "No cron jobs scheduled."
        return "Cron jobs:\n" + "\n".join(_format_job_summary(j) for j in items)

    if act == "show":
        if not id:
            return "Error: `id` is required for show."
        job = cron_jobs.get_job(id)
        if not job:
            return f"No cron job with id `{id}`."
        return _format_job_detail(job)

    if act == "delete":
        if not id:
            return "Error: `id` is required for delete."
        return (
            f"Deleted cron job `{id}`."
            if cron_jobs.delete_job(id)
            else f"No cron job with id `{id}`."
        )

    if act == "pause":
        if not id:
            return "Error: `id` is required for pause."
        return (
            f"Paused cron job `{id}`."
            if cron_jobs.pause_job(id)
            else f"No cron job with id `{id}`."
        )

    if act == "resume":
        if not id:
            return "Error: `id` is required for resume."
        return (
            f"Resumed cron job `{id}`."
            if cron_jobs.resume_job(id)
            else f"No cron job with id `{id}`."
        )

    if act in {"run-now", "run_now", "trigger"}:
        if not id:
            return "Error: `id` is required for run-now."
        job = cron_jobs.get_job(id)
        if not job:
            return f"No cron job with id `{id}`."
        result = run_job(job)
        status = "ok" if result["success"] else f"error: {result.get('error')}"
        return (
            f"Ran cron job `{id}` ({status}). "
            f"Output: {result.get('output_path')}"
        )

    return (
        f"Unknown cron action `{action}`. "
        "Valid: create, list, show, delete, pause, resume, run-now."
    )


if tool is not None:

    @tool("cronjob")
    def cronjob(
        action: Literal["create", "list", "show", "delete", "pause", "resume", "run-now"],
        id: str = "",
        prompt: str = "",
        schedule: str = "",
        name: str = "",
        skills: list[str] | None = None,
        model: str = "",
        no_agent: bool = False,
        deliver: str = "local",
    ) -> str:
        """Manage scheduled cron jobs.

        Actions:
          - `create`  → needs `schedule` (e.g. ``"30m"`` or ``"0 9 * * *"``) and
            usually `prompt`; optional `skills`, `model`, `name`, `no_agent`,
            `deliver` (``"local"`` only in v1).
          - `list`    → all jobs.
          - `show`    → details for one job (needs `id`).
          - `delete`  → remove job (needs `id`).
          - `pause`   → disable without deleting (needs `id`).
          - `resume`  → re-enable a paused job (needs `id`).
          - `run-now` → execute immediately without waiting for schedule.
        """
        return _cronjob_impl(
            action,
            id=id,
            prompt=prompt,
            schedule=schedule,
            name=name,
            skills=skills,
            model=model,
            no_agent=no_agent,
            deliver=deliver,
        )

else:  # pragma: no cover - exercised only without langchain-core
    cronjob = _cronjob_impl  # type: ignore[assignment]


__all__ = ["_cronjob_impl", "cronjob"]
