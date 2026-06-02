"""Cron daemon, job storage, cronjob tool.

Public surface (SPEC §14):

  - :mod:`deepagent_hermes.cron.jobs`      — job CRUD + ``parse_schedule``
  - :mod:`deepagent_hermes.cron.scheduler` — ``HermesCron`` daemon + ``run_job``
  - :mod:`deepagent_hermes.cron.tool`      — ``cronjob`` agent tool
  - ``python -m deepagent_hermes.cron``    — run the daemon
"""

from deepagent_hermes.cron.jobs import (
    create_job,
    delete_job,
    get_due_jobs,
    get_job,
    list_jobs,
    pause_job,
    resume_job,
    update_job,
)
from deepagent_hermes.cron.scheduler import SILENT_MARKER, HermesCron, run_job

__all__ = [
    "HermesCron",
    "SILENT_MARKER",
    "create_job",
    "delete_job",
    "get_due_jobs",
    "get_job",
    "list_jobs",
    "pause_job",
    "resume_job",
    "run_job",
    "update_job",
]
