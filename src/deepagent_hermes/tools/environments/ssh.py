"""SSH terminal backend — STUB (full implementation deferred to v0.2.0).

When implemented, will use ``paramiko`` for the SSH session and ``rsync``
(or ``scp`` fallback) for file sync, configured via ``[ssh].host`` / ``.user``
/ ``.key_path`` in ``deepagent-hermes.toml``. See SPEC §12.
"""

from __future__ import annotations

from deepagent_hermes.tools.environments.base import BaseEnvironment, ProcessHandle


class SshEnvironment(BaseEnvironment):
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
            "SshEnvironment is not implemented in v0.1.0. "
            "Use LocalEnvironment, or contribute the implementation. "
            "See SPEC §12."
        )

    def cleanup(self) -> None:
        pass


__all__ = ["SshEnvironment"]
