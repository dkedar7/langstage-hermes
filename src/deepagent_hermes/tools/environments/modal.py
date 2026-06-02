"""Modal terminal backend — STUB (full implementation deferred to v0.2.0).

When implemented, will use the ``modal`` SDK (``pip install
deepagent-hermes[modal]``) with a ``_ThreadedProcessHandle`` adapter wrapping
``sandbox.exec`` blocking calls. See SPEC §12.
"""

from __future__ import annotations

from deepagent_hermes.tools.environments.base import BaseEnvironment, ProcessHandle


class ModalEnvironment(BaseEnvironment):
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
            "ModalEnvironment is not implemented in v0.1.0. "
            "Use LocalEnvironment, or contribute the implementation. "
            "See SPEC §12."
        )

    def cleanup(self) -> None:
        pass


__all__ = ["ModalEnvironment"]
