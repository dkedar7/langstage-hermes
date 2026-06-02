"""Daytona terminal backend — STUB (full implementation deferred to v0.2.0).

When implemented, will use the ``daytona-sdk`` extra (``pip install
deepagent-hermes[daytona]``) and wrap blocking ``sandbox.exec`` calls in a
threaded ``ProcessHandle`` adapter. See SPEC §12.
"""

from __future__ import annotations

from deepagent_hermes.tools.environments.base import BaseEnvironment, ProcessHandle


class DaytonaEnvironment(BaseEnvironment):
    """Stub. ``_run_bash`` raises NotImplementedError; ``cleanup`` is a no-op."""

    def _run_bash(
        self,
        cmd: str,
        *,
        login: bool = False,
        timeout: int = 60,
        stdin_data: str | None = None,
    ) -> ProcessHandle:
        raise NotImplementedError(
            "DaytonaEnvironment is not implemented in v0.1.0. "
            "Use LocalEnvironment, or contribute the implementation. "
            "See SPEC §12."
        )

    def cleanup(self) -> None:
        pass


__all__ = ["DaytonaEnvironment"]
