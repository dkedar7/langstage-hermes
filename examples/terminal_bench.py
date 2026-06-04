"""Run deepagent-hermes against Terminal-Bench 2.0 via Harbor.

This file is a **Harbor agent adapter**. Harbor (the framework that
backs Terminal-Bench) loads agents by import path; the adapter is the
glue between Harbor's contract (``BaseAgent.run(instruction, environment,
context)``) and our compiled hermes graph (``create_hermes_agent(...)``
returning a ``CompiledStateGraph``).

How it fits together
--------------------

1. Harbor spins up a sandbox (``--env docker`` / ``e2b`` / etc.) and
   constructs a :class:`harbor.environments.base.BaseEnvironment` pointing
   at it. The only primitive offered to the agent is ``env.exec(command)``.
2. We adapt that primitive to deepagents' filesystem/exec surface by
   subclassing :class:`deepagents.backends.sandbox.BaseSandbox`. The
   subclass overrides three abstract methods — ``execute``, ``upload_files``,
   ``download_files`` — bridging the sync→async gap via
   ``asyncio.run_coroutine_threadsafe``.
3. We pass that backend to ``create_hermes_agent(backend=...)``. Hermes's
   ``FilesystemMiddleware`` and ``SubAgentMiddleware`` then use the
   sandbox for every file/exec tool call, so the *entire* deepagent-hermes
   middleware stack (reflection, memory, skills, FTS5 recorder, …) runs
   *inside* the Terminal-Bench task container.
4. After the graph terminates we walk the final state's messages, sum
   ``usage_metadata`` across every ``AIMessage``, and populate Harbor's
   ``AgentContext``.

Running it
----------

The task containers are Linux, and Harbor calls the ``docker`` CLI
directly. On Windows boxes where Docker is exposed via WSL2, run this
from inside WSL — Harbor will then find ``docker`` on ``$PATH`` and the
mounts will work natively.

::

    # inside WSL Ubuntu
    python3 -m venv ~/.venvs/dah-bench && source ~/.venvs/dah-bench/bin/activate
    pip install harbor "deepagent-hermes[openai]"
    export OPENROUTER_API_KEY=sk-or-v1-...   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
    harbor run \\
        --agent examples/terminal_bench.py::DeepagentHermesAgent \\
        --model openrouter/anthropic/claude-sonnet-4-5 \\
        --env docker \\
        --tasks terminal-bench-core==head \\
        --n-tasks 5             # start small; drop to run the full 89

Known limitations of this v1 adapter
------------------------------------

- **Stateless across tasks** — each task gets a fresh ``HERMES_HOME`` in
  ``/tmp``. That's intentional (no skill/memory leakage between tasks)
  but means hermes can't learn from earlier tasks during the same suite.
- **No streaming back to Harbor** — we run ``.ainvoke(...)`` and
  populate the context at the end. Trajectory mid-run isn't surfaced.
  Harbor still logs the agent's stdout via ``self.logger``.
- **Heavy-handed timeouts** — every ``exec`` defaults to 300 s. Tasks
  that legitimately need longer (compile huge projects) can raise
  ``HERMES_BENCH_DEFAULT_EXEC_TIMEOUT``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

log = logging.getLogger(__name__)


_DEFAULT_EXEC_TIMEOUT = int(os.environ.get("HERMES_BENCH_DEFAULT_EXEC_TIMEOUT", "300"))


class HarborSandboxBackend(BaseSandbox):
    """Bridge from deepagents' sandbox protocol to Harbor's async env.

    We're called from sync tool handlers (``ls``, ``read``, ``write``,
    ``edit``, ``execute``). Each of those needs to await ``env.exec(...)``.
    We use ``asyncio.run_coroutine_threadsafe`` to schedule the coroutine
    on the loop that owns the environment, then block on the future.

    This works because deepagents' middleware invokes the *async* tool
    paths (``aexecute``, ``awrite``, …) which wrap our sync methods in
    ``asyncio.to_thread`` — so the sync call lives on a worker thread,
    leaving the env's loop free to actually run the coroutine.
    """

    def __init__(self, env: BaseEnvironment, loop: asyncio.AbstractEventLoop, *, default_timeout: int | None = None) -> None:
        self._env = env
        self._loop = loop
        self._default_timeout = default_timeout or _DEFAULT_EXEC_TIMEOUT
        self._id = f"harbor-{env.session_id}-{uuid.uuid4().hex[:6]}"

    @property
    def id(self) -> str:
        return self._id

    def _await(self, coro: Any, *, timeout: float | None) -> Any:
        """Schedule *coro* on ``self._loop`` from a worker thread and block."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # Give the future itself slightly longer than the inner timeout so
        # the inner cancellation can propagate before we time out the wait.
        wait_timeout = (timeout + 30) if timeout is not None else None
        return fut.result(timeout=wait_timeout)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        effective = timeout if timeout is not None else self._default_timeout
        try:
            result = self._await(
                self._env.exec(command, timeout_sec=effective),
                timeout=effective,
            )
        except Exception as exc:
            # BaseSandbox.read/ls/grep parse stdout; surfacing the error as
            # a non-zero exit code keeps them on the error path instead of
            # crashing the agent.
            log.warning("HarborSandboxBackend.execute crashed: %s", exc)
            return ExecuteResponse(output=f"hermes-bench: env.exec raised: {exc}", exit_code=124)

        combined = (result.stdout or "") + (result.stderr or "")
        return ExecuteResponse(
            output=combined,
            exit_code=result.return_code,
            truncated=False,
        )

    # File transfer ---------------------------------------------------------
    # BaseSandbox.write() calls upload_files([(path, bytes)]) after a
    # preflight check. We avoid round-tripping to disk on the host by
    # base64-piping the content through a single env.exec.

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            b64 = base64.b64encode(content).decode("ascii")
            cmd = (
                f"mkdir -p {shlex.quote(str(Path(path).parent))} && "
                f"printf '%s' {shlex.quote(b64)} | base64 -d > {shlex.quote(path)}"
            )
            result = self._await(
                self._env.exec(cmd, timeout_sec=self._default_timeout),
                timeout=self._default_timeout,
            )
            if result.return_code != 0:
                err = (result.stderr or result.stdout or "non-zero exit").strip()
                if "Permission denied" in err:
                    responses.append(FileUploadResponse(path=path, error="permission_denied"))
                else:
                    responses.append(FileUploadResponse(path=path, error=err))
            else:
                responses.append(FileUploadResponse(path=path))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            # ``test -f`` first so we can return a structured error rather than
            # treating the cat failure as a generic backend error.
            check = self._await(
                self._env.exec(f"test -f {shlex.quote(path)}", timeout_sec=30),
                timeout=30,
            )
            if check.return_code != 0:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue
            cmd = f"base64 -w0 {shlex.quote(path)}"
            result = self._await(
                self._env.exec(cmd, timeout_sec=self._default_timeout),
                timeout=self._default_timeout,
            )
            if result.return_code != 0:
                err = (result.stderr or result.stdout or "non-zero exit").strip()
                responses.append(FileDownloadResponse(path=path, content=None, error=err))
                continue
            try:
                content = base64.b64decode((result.stdout or "").strip())
            except Exception as exc:
                responses.append(FileDownloadResponse(path=path, content=None, error=f"decode_error: {exc}"))
                continue
            responses.append(FileDownloadResponse(path=path, content=content))
        return responses


def _sum_usage(messages: list[Any]) -> dict[str, int]:
    """Walk ``AIMessage.usage_metadata`` across the run and total it up."""
    totals = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0}
    for msg in messages or []:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        details = usage.get("input_token_details") or {}
        totals["cache_read"] += int(details.get("cache_read") or 0)
        totals["cache_creation"] += int(details.get("cache_creation") or 0)
    return totals


class DeepagentHermesAgent(BaseAgent):
    """Harbor entrypoint for the deepagent-hermes runtime.

    Subclasses :class:`harbor.agents.base.BaseAgent`. The class is auto-
    discovered by Harbor via the ``--agent <path>::Class`` flag.

    Each ``run()`` call builds a fresh compiled hermes graph wired to a
    fresh ``HarborSandboxBackend`` and a per-task ``HERMES_HOME`` under
    ``/tmp``. That keeps tasks isolated (no skill/memory bleed between
    tasks) and keeps the home directory inside the container, where it
    can write freely without bumping into mounted-volume permissions.
    """

    # Harbor surfaces these as class metadata in reports.
    SUPPORTS_ATIF: bool = False  # We don't emit Harbor-format trajectories yet.
    SUPPORTS_WINDOWS: bool = False  # We exec POSIX shell commands directly.

    @staticmethod
    def name() -> str:
        return "deepagent-hermes"

    def version(self) -> str | None:
        try:
            from deepagent_hermes import __version__

            return __version__
        except Exception:
            return None

    async def setup(self, environment: BaseEnvironment) -> None:
        # Nothing to install — the agent itself runs on the host; only the
        # tool calls reach into the container. We *do* make sure base64 and
        # /tmp/hermes-home exist (some minimal images ship neither).
        for cmd in (
            "command -v base64 >/dev/null || { echo 'base64 missing — install coreutils'; exit 1; }",
            "mkdir -p /tmp/hermes-home",
        ):
            result = await environment.exec(cmd, timeout_sec=30)
            if result.return_code != 0:
                raise RuntimeError(
                    f"deepagent-hermes setup failed (rc={result.return_code}): {(result.stderr or result.stdout or '').strip()}"
                )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # Lazy import — saves Harbor's `--help` from pulling the whole graph.
        from deepagent_hermes.agent import create_hermes_agent
        from deepagent_hermes.config import HermesConfig

        loop = asyncio.get_running_loop()
        backend = HarborSandboxBackend(environment, loop)

        # Per-task HERMES_HOME — isolated state.db, fresh memory, fresh
        # skill index. We don't bind-mount: the *graph* runs on the host,
        # so the host's tmp is the right place for SQLite + memory files.
        host_home = Path(os.environ.get("HERMES_BENCH_HOME_ROOT") or "/tmp/hermes-bench") / f"task-{uuid.uuid4().hex[:8]}"
        host_home.mkdir(parents=True, exist_ok=True)
        os.environ["DEEPAGENT_HERMES_HOME"] = str(host_home)

        cfg = HermesConfig.resolve()
        # Honor Harbor's model selection. Harbor passes ``self.model_name``
        # as ``provider/model`` (e.g. ``openrouter/anthropic/claude-…``).
        # ``init_chat_model`` accepts that format directly.
        if self.model_name:
            cfg.model_default = self.model_name
            cfg.model_aux = self.model_name

        start = time.monotonic()
        graph = create_hermes_agent(cfg, backend=backend)

        # The instruction is the user's first turn. LangGraph's recursion
        # limit (1000) covers tool-call ping-pong; the iteration budget
        # middleware enforces the agent-level cap (default 10).
        try:
            final = await graph.ainvoke(
                {"messages": [{"role": "user", "content": instruction}]},
                config={"configurable": {"thread_id": graph.deepagent_hermes_session_id}},
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            self.logger.exception("deepagent-hermes agent crashed after %.1fs: %s", elapsed, exc)
            context.metadata["error"] = str(exc)
            return

        elapsed = time.monotonic() - start
        usage = _sum_usage(final.get("messages") or [])

        # Populate Harbor's context — these fields drive the trial report
        # (token totals, cost estimate, rollout metadata).
        context.n_input_tokens = usage["input_tokens"]
        context.n_cache_tokens = usage["cache_read"]
        context.n_output_tokens = usage["output_tokens"]
        # Cost is provider-specific; we leave it 0.0 unless the user
        # exports HERMES_BENCH_COST_PER_INPUT_KTOK / _OUTPUT_KTOK as a
        # rough back-of-envelope estimator.
        ipk = float(os.environ.get("HERMES_BENCH_COST_PER_INPUT_KTOK", "0") or 0)
        opk = float(os.environ.get("HERMES_BENCH_COST_PER_OUTPUT_KTOK", "0") or 0)
        context.cost_usd = (usage["input_tokens"] / 1000 * ipk) + (usage["output_tokens"] / 1000 * opk)
        context.metadata.update(
            {
                "elapsed_sec": round(elapsed, 1),
                "hermes_home": str(host_home),
                "session_id": graph.deepagent_hermes_session_id,
                "n_messages": len(final.get("messages") or []),
                "version": self.version() or "unknown",
            }
        )

        self.logger.info(
            "deepagent-hermes finished task in %.1fs: %d in / %d out tokens (cache_read=%d)",
            elapsed,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["cache_read"],
        )
