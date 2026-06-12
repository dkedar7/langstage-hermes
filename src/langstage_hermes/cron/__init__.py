"""Cron daemon, job storage, cronjob tool.

Public surface (SPEC §14):

  - :mod:`langstage_hermes.cron.jobs`      — job CRUD + ``parse_schedule``
  - :mod:`langstage_hermes.cron.scheduler` — ``HermesCron`` daemon + ``run_job``
  - :mod:`langstage_hermes.cron.tool`      — ``cronjob`` agent tool
  - ``python -m langstage_hermes.cron``    — run the daemon
"""

from langstage_hermes.cron.deliverers import (
    AgentMailDeliverer,
    Deliverer,
    LocalDeliverer,
    StdoutDeliverer,
    get_deliverer,
    register_deliverer,
)
from langstage_hermes.cron.jobs import (
    create_job,
    delete_job,
    get_due_jobs,
    get_job,
    list_jobs,
    pause_job,
    resume_job,
    update_job,
)
from langstage_hermes.cron.scheduler import SILENT_MARKER, HermesCron, run_job

__all__ = [
    "SILENT_MARKER",
    "AgentMailDeliverer",
    "Deliverer",
    "HermesCron",
    "LocalDeliverer",
    "StdoutDeliverer",
    "create_job",
    "delete_job",
    "get_deliverer",
    "get_due_jobs",
    "get_job",
    "list_jobs",
    "pause_job",
    "register_deliverer",
    "resume_job",
    "run_job",
    "update_job",
]
