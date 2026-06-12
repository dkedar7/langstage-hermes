"""Entry point: ``python -m langstage_hermes.cron``.

Starts the cron daemon. The daemon holds a file lock so only one instance
runs per ``HERMES_HOME`` — start it on logon via Task Scheduler (Windows),
``systemd --user`` (Linux), or ``launchd`` (macOS); see SPEC §14.5.
"""

from __future__ import annotations

import logging
import sys

from langstage_hermes.config import HermesConfig, hermes_home
from langstage_hermes.cron.jobs import _jobs_file, _output_dir, tick_lock_path
from langstage_hermes.cron.scheduler import HermesCron


def main() -> int:
    """Resolve config + spin up the cron loop. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = HermesConfig.resolve()
    cron = HermesCron(tick_seconds=cfg.cron_tick_seconds)

    home = hermes_home()
    print(
        "langstage-hermes cron starting "
        f"(tick={cron.tick_seconds}s, home={home}, "
        f"jobs={_jobs_file()}, output={_output_dir()}, lock={tick_lock_path()})",
        flush=True,
    )
    try:
        cron.run_forever()
    except KeyboardInterrupt:
        print("\nlangstage-hermes cron stopping (SIGINT)", flush=True)
        return 0
    except RuntimeError as e:
        print(f"langstage-hermes cron failed to start: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
