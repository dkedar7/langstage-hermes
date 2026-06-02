"""Docker terminal backend — STUB (full implementation deferred to v0.2.0).

When implemented, will spawn commands via ``docker run --rm -v <bind> <image>
bash -c ...``, gated on the ``DEEPAGENT_HERMES_DOCKER_IMAGE`` env var (default
``python:3.13-slim``) and the Docker daemon being reachable.

See SPEC §12 for the full requirements.
"""

from __future__ import annotations

from deepagent_hermes.tools.environments.base import BaseEnvironment, ProcessHandle


class DockerEnvironment(BaseEnvironment):
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
            "DockerEnvironment is not implemented in v0.1.0. "
            "Use LocalEnvironment, or contribute the implementation. "
            "See SPEC §12."
        )

    def cleanup(self) -> None:
        # No resources to release in the stub. Implementation will tear down
        # the container handle here.
        pass


__all__ = ["DockerEnvironment"]
