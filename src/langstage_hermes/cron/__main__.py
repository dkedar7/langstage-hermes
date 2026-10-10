"""Entry point: ``python -m langstage_hermes.cron``.

Starts the cron daemon. The daemon holds a file lock so only one instance
runs per ``HERMES_HOME`` — start it on logon via Task Scheduler (Windows),
``systemd --user`` (Linux), or ``launchd`` (macOS); see SPEC §14.5.

``-h/--help`` and ``--version`` print and exit 0; any other argument is a usage
error, exit 64 (family exit codes, core ADR 0007). No arguments starts the daemon.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import NoReturn

from langstage_hermes import __version__
from langstage_hermes.cli import EXIT_FAIL, EXIT_OK, EXIT_USAGE
from langstage_hermes.config import HermesConfig, hermes_home
from langstage_hermes.cron.jobs import _jobs_file, _output_dir, tick_lock_path
from langstage_hermes.cron.scheduler import HermesCron


class _ArgumentParser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error, which the family reserves for "paused";
    exit 64 instead, like the click CLI's ``_FamilyExitGroup`` (gh #172)."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="python -m langstage_hermes.cron",
        description=(
            "Run the langstage-hermes cron daemon forever: each tick runs the jobs that "
            "are due. One daemon runs per HERMES_HOME (it holds a file lock). Stop it "
            "with Ctrl+C."
        ),
        epilog="To list, create, pause, resume or delete jobs, use the full cron CLI: `langstage-hermes cron --help`.",
    )
    parser.add_argument("--version", action="version", version=f"langstage-hermes {__version__}")
    return parser


def main(argv: Sequence[str] = ()) -> int:
    """Parse ``argv``, then resolve config + spin up the cron loop. Returns a process exit code.

    ``argv`` defaults to no arguments, so ``langstage-hermes cron daemon`` (which calls
    ``main()``) never sees the click command line; the module entry passes ``sys.argv[1:]``.
    """
    try:
        _parser().parse_args(list(argv))
    except SystemExit as e:  # --help / --version (0) or a usage error (64)
        return e.code if isinstance(e.code, int) else EXIT_USAGE

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
        return EXIT_OK
    except RuntimeError as e:
        print(f"langstage-hermes cron failed to start: {e}", file=sys.stderr)
        return EXIT_FAIL  # can't start is a failure (family exit codes, core ADR 0007)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
