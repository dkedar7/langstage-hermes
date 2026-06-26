"""The compiled Hermes agent.

This module is the entry point hosts target via
``DEEPAGENT_AGENT_SPEC=langstage_hermes.agent:graph``. It owns the
**middleware stack ordering** (see SPEC §4) and is the only place that
knows how the subsystems fit together.

Two public surfaces:

* :func:`create_hermes_agent` — build a fresh compiled graph from a
  :class:`~langstage_hermes.config.HermesConfig`. Each call returns an
  independent ``CompiledStateGraph`` with its own checkpointer + store
  references; callers can swap models or workspaces per agent.
* :data:`graph` — a module-level instance built from
  ``HermesConfig.resolve()`` for hosts that want a ready-to-use graph.
  Constructed lazily on first attribute access so ``import``-time has
  no side effects.

Per SPEC §1 (D8), we deliberately do NOT use ``deepagents.create_deep_agent``
because it appends user middleware *after* the defaults and always prepends
``BASE_AGENT_PROMPT``. We need to own the middleware list end-to-end and
own the system prompt, so we call ``langchain.agents.create_agent`` directly
and assemble the middleware ourselves.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from langstage_hermes.budget import IterationBudgetMiddleware
from langstage_hermes.caching import AnthropicCachingS3Middleware
from langstage_hermes.compression import HermesCompressionMiddleware
from langstage_hermes.config import HermesConfig
from langstage_hermes.curator import CuratorMiddleware
from langstage_hermes.memory.provider import get_provider
from langstage_hermes.memory.tool import MemoryToolMiddleware
from langstage_hermes.plugins.event_bus import PluginEventBus
from langstage_hermes.prompts import PromptAssemblyMiddleware
from langstage_hermes.reflection import ReflectionMiddleware, build_review_subagent
from langstage_hermes.search.session_search import make_session_search_tool
from langstage_hermes.skills.library import SkillLibrary
from langstage_hermes.skills.loader import SkillLoaderMiddleware
from langstage_hermes.skills.tools import make_skill_tools
from langstage_hermes.store.recorder import HermesStateRecorderMiddleware
from langstage_hermes.store.sqlite_fts import SqliteFtsStore
from langstage_hermes.tools.toolsets import resolve_enabled

log = logging.getLogger(__name__)


def _default_skill_dirs(cfg: HermesConfig) -> list[Path]:
    """Resolution order matches SPEC §10.2 — later wins on name collision.

    Bundled skills ship inside the wheel at ``langstage_hermes/_bundled_skills/``
    (not at the repo root). Pre-v0.1.2 this looked three parents up from
    ``agent.py``, which worked for editable installs but resolved to ``Lib/``
    on PyPI installs — leaving fresh users with no bundled skills.
    """
    dirs: list[Path] = []
    bundled = Path(__file__).resolve().parent / "_bundled_skills"
    if bundled.is_dir():
        dirs.append(bundled)
    # User-global.
    dirs.append(cfg.hermes_home / "skills")
    # Project shadow.
    project = Path.cwd() / ".langstage-hermes" / "skills"
    if project.is_dir():
        dirs.append(project)
    # Extra dirs from config.
    for extra in cfg.skills_external_dirs:
        dirs.append(Path(extra).expanduser())
    return dirs


def _alias_openrouter_key(model_id: str) -> None:
    """Wire ``OPENROUTER_API_KEY`` to the OpenAI client for ``openai:*`` models.

    The README advertises ``OPENROUTER_API_KEY`` as a drop-in alternative to
    ``OPENAI_API_KEY``, but ``ChatOpenAI`` only reads ``OPENAI_API_KEY``. When an
    ``openai:*`` model is selected with only ``OPENROUTER_API_KEY`` set, alias it
    (and default the base URL to OpenRouter) so the documented path — which
    ``verify``/``doctor`` already accept as a satisfied key — actually works
    instead of failing with "Missing credentials" at build. (gh #33)
    """
    if model_id.startswith("openai:") and os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
        os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def _init_chat_model(model_id: str | None) -> Any:
    """Wrap ``langchain.chat_models.init_chat_model`` so a ``None`` returns a sentinel default."""
    from langchain.chat_models import init_chat_model

    if not model_id:
        # Must match HermesConfig.model_default (SPEC §2) — a stale literal here
        # silently disagreed with the configured default (gh #-dogfood).
        model_id = "anthropic:claude-sonnet-4-6"

    _alias_openrouter_key(model_id)
    return init_chat_model(model_id)


def create_hermes_agent(
    config: HermesConfig | None = None,
    *,
    workspace: str | Path | None = None,
    session_id: str | None = None,
    extra_middleware: list[Any] | None = None,
    backend: Any = None,
    model: Any = None,
    aux_model: Any = None,
) -> Any:
    """Build a fresh Hermes agent graph.

    Args:
        config: Resolved configuration; defaults to ``HermesConfig.resolve()``.
        workspace: Filesystem root for the file toolset; defaults to ``cwd``.
            Ignored when ``backend`` is supplied — the backend owns its own
            root.
        session_id: Optional session id; auto-generated UUID if not provided.
        extra_middleware: Additional middleware appended after the standard
            stack — useful for hosts that want to inject tracing or auth.
        backend: Optional ``BackendProtocol`` instance to use for filesystem
            and exec tools (and the review subagent). When ``None`` (default),
            a local :class:`~deepagents.backends.filesystem.FilesystemBackend`
            rooted at ``workspace`` is constructed. Pass a
            :class:`~deepagents.backends.protocol.SandboxBackendProtocol`
            implementation (e.g. the Harbor adapter in
            ``examples/terminal_bench.py``) to run hermes against a remote
            sandbox.
        model: Optional ``langchain_core.language_models.BaseChatModel``
            instance to use as the main model. When supplied, overrides
            ``config.model_default`` and bypasses ``init_chat_model``. Use
            this for bring-your-own-model setups that ``init_chat_model``'s
            ``provider:name`` string can't express cleanly —
            ``AzureChatOpenAI``, ``ChatBedrock``, an OpenAI-compatible
            proxy with a custom ``base_url``, etc. The instance is used
            as-is (no rewrapping); responsibility for credentials,
            ``cache_control`` headers, retry config, etc. stays with the
            caller.
        aux_model: Same as ``model`` but for the auxiliary model the
            reflection subagent and the compression summariser use.
            Defaults to ``model`` if ``model`` is set and ``aux_model`` is
            not; otherwise falls back to ``config.model_aux`` via
            ``init_chat_model``.

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready for ``.invoke()`` /
        ``.stream()``. The graph carries a SQLite checkpointer and FTS5 store
        rooted at ``<HERMES_HOME>/state.db``.
    """
    from langchain.agents import create_agent
    from langgraph.checkpoint.sqlite import SqliteSaver

    cfg = config or HermesConfig.resolve()
    sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
    ws = Path(workspace).resolve() if workspace else Path.cwd()

    # ── shared resources ─────────────────────────────────────────────────
    db_path = cfg.hermes_home / "state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteFtsStore(db_path=str(db_path))
    # Audit log shares the state.db file with the store (separate table).
    # We give every library the audit log so agent skill mutations are
    # automatically recorded; the CLI's `audit rollback` reads from the
    # same table.
    from langstage_hermes.skills.audit import SkillAuditLog

    audit_log = SkillAuditLog(db_path=str(db_path))
    library = SkillLibrary(_default_skill_dirs(cfg), audit_log=audit_log)
    library.set_mutation_context(session_id=sid, source="agent")

    # Bring-your-own-model: caller-supplied instances bypass init_chat_model
    # entirely so Azure / Bedrock / OpenAI-compatible proxies / anything
    # whose config doesn't fit the `provider:name` string format works
    # without forcing the user to wrap it. Default behaviour (None,None)
    # is unchanged — string-driven init via cfg.model_default / model_aux.
    main_model = model if model is not None else _init_chat_model(cfg.model_default)
    if aux_model is not None:
        pass  # caller supplied it explicitly
    elif model is not None:
        # Caller gave a main model but no aux — share it. This matches the
        # string-driven path's behaviour when only `model_default` is set.
        aux_model = main_model
    else:
        aux_model = _init_chat_model(cfg.model_aux) if cfg.model_aux else main_model

    # ── memory provider plugin (single-select) ───────────────────────────
    provider_name = cfg.memory_provider or ""
    provider_cls = get_provider(provider_name)
    provider = provider_cls() if provider_cls else None

    # ── enabled toolsets (after disabled_toolsets filter) ────────────────
    enabled_toolsets = resolve_enabled(
        disabled_toolsets=set(cfg.agent_disabled_toolsets),
        platform=os.getenv("HERMES_PLATFORM", "cli"),
    )

    # ── tools (kept as a flat list; deepagents'/langchain's create_agent merges from middleware too) ──
    skill_tools = make_skill_tools(library)
    session_search_tool = make_session_search_tool(store, current_session_id_getter=lambda: sid)
    # FilesystemBackend tools come in via the FilesystemMiddleware (below).
    tools: list[Any] = [*skill_tools, session_search_tool]

    # ── deepagents middleware (filesystem + subagents + todos) ───────────
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from deepagents.middleware.subagents import SubAgentMiddleware
    from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware

    if backend is not None:
        # External sandbox backend (e.g. Harbor proxy) — paths are real and
        # the host has full control. virtual_mode/root_dir semantics belong
        # to the supplied backend, not us.
        fs_backend = backend
    else:
        # virtual_mode=True so the agent's '/workspace/foo.py' (its natural
        # absolute-path convention from the system prompt) resolves under our
        # configured root rather than literal C:\workspace\foo.py. Without this,
        # the agent silently writes to / reads from a path the user can't
        # introspect — surfaced in the 2026-06-02 dogfood when reported file
        # writes didn't appear on disk.
        fs_backend = FilesystemBackend(root_dir=str(ws), virtual_mode=True)

    # ── review subagent (reflection target) ──────────────────────────────
    # Wire the memory + skill_manage tools so the review fork can actually
    # write — without these, the subagent runs but has no way to act on its
    # conclusions, and the closed loop never closes.
    _memory_mw_for_tools = MemoryToolMiddleware(
        memory_char_limit=cfg.memory_char_limit,
        user_char_limit=cfg.memory_user_char_limit,
    )
    review_tools = [*skill_tools, *_memory_mw_for_tools.tools]
    review_subagent = build_review_subagent(library=library, store=store, aux_model=aux_model, tools=review_tools)

    # ── compose the middleware stack (SPEC §4 order) ─────────────────────
    # Note: deepagents inserts TodoList + Filesystem + SubAgent earlier in
    # its own create_deep_agent; we do it ourselves to control ordering.
    middleware: list[Any] = [
        # PluginEventBus is OUTERMOST so plugin hooks see the unmodified
        # request and the final response (per its module docstring).
        PluginEventBus(),
        # Budget next — it can short-circuit before anything else runs.
        IterationBudgetMiddleware(max_iterations=cfg.agent_max_iterations),
        # Prompt assembly owns the system prompt (outermost wrap so the
        # skill loader's mutation lands on top of the assembled prompt).
        PromptAssemblyMiddleware(
            enabled_toolsets=list(enabled_toolsets),
            platform=os.getenv("HERMES_PLATFORM", "cli"),
            workspace_root=ws,
        ),
        SkillLoaderMiddleware(library),
        # Memory snapshot loader / writer.
        MemoryToolMiddleware(
            memory_char_limit=cfg.memory_char_limit,
            user_char_limit=cfg.memory_user_char_limit,
        ),
        # FTS5 recorder — writes every turn to the SQLite store.
        HermesStateRecorderMiddleware(store=store),
        # Reflection — counts tool calls, spawns review subagent on threshold.
        ReflectionMiddleware(
            skill_nudge_interval=cfg.skills_creation_nudge_interval,
            memory_nudge_interval=cfg.memory_nudge_interval,
            library=library,
            store=store,
            model=main_model,
            aux_model=aux_model,
        ),
        # Curator — runs on session start if idle gates open.
        CuratorMiddleware(
            library,
            store,
            interval_hours=cfg.curator_interval_hours,
            min_idle_hours=cfg.curator_min_idle_hours,
            stale_days=cfg.curator_stale_after_days,
            archive_days=cfg.curator_archive_after_days,
            enabled=cfg.curator_enabled,
        ),
        # Deepagents built-ins.
        TodoListMiddleware(),
        FilesystemMiddleware(backend=fs_backend),
        SubAgentMiddleware(
            backend=fs_backend,
            subagents=[review_subagent],
        ),
        # Compression near the end so it sees the fully-assembled prompt + state.
        HermesCompressionMiddleware(
            model=main_model,
            aux_model=aux_model,
            threshold_percent=cfg.compression_threshold,
            protect_first_n=cfg.compression_protect_first_n,
            protect_last_n=cfg.compression_protect_last_n,
            summary_target_ratio=cfg.compression_target_ratio,
            abort_on_summary_failure=cfg.compression_abort_on_summary_failure,
        ),
        # Caching wraps the actual model call — must be near the inner edge.
        AnthropicCachingS3Middleware(ttl="5m"),
        # PatchToolCalls fixes orphaned tool_call ids after interrupted runs.
        PatchToolCallsMiddleware(),
    ]

    # Optional human-in-the-loop (only added if any tool is gated).
    # Hosts can override via DEEPAGENT_HERMES_INTERRUPT_ON env (CSV of tool names).
    interrupt_csv = os.getenv("LANGSTAGE_HERMES_INTERRUPT_ON") or os.getenv("DEEPAGENT_HERMES_INTERRUPT_ON", "")
    if interrupt_csv:
        interrupt_on = {name.strip(): True for name in interrupt_csv.split(",") if name.strip()}
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    if extra_middleware:
        middleware.extend(extra_middleware)

    # ── checkpointer ─────────────────────────────────────────────────────
    # Shares the state.db file with the FTS store (disjoint table namespaces).
    # We hold a long-lived connection ourselves rather than using
    # ``SqliteSaver.from_conn_string`` (which is a context manager and would
    # close the connection on GC of the temporary). ``check_same_thread=False``
    # lets the graph stream from a different thread than the constructor.
    import sqlite3

    saver_conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(saver_conn)

    # ── compile ──────────────────────────────────────────────────────────
    # System prompt is set by PromptAssemblyMiddleware via wrap_model_call,
    # so we pass an empty string here — the middleware will replace it.
    compiled = create_agent(
        main_model,
        system_prompt="",
        tools=tools,
        middleware=middleware,
        checkpointer=checkpointer,
        store=store,
    ).with_config({"recursion_limit": 1000, "configurable": {"thread_id": sid}})

    # Attach references the hosts may want to introspect.
    compiled.langstage_hermes_config = cfg  # type: ignore[attr-defined]
    compiled.langstage_hermes_session_id = sid  # type: ignore[attr-defined]
    compiled.langstage_hermes_store = store  # type: ignore[attr-defined]
    compiled.langstage_hermes_library = library  # type: ignore[attr-defined]
    compiled.langstage_hermes_provider = provider  # type: ignore[attr-defined]
    # Keep the checkpointer connection alive for the lifetime of the graph.
    compiled._langstage_hermes_saver_conn = saver_conn  # type: ignore[attr-defined]

    return compiled


# ── module-level lazy graph for host adoption ────────────────────────────


_graph: Any = None


def __getattr__(name: str) -> Any:
    """Lazy ``graph`` instantiation so ``import langstage_hermes.agent`` is cheap.

    Hosts using ``DEEPAGENT_AGENT_SPEC=langstage_hermes.agent:graph`` will
    trigger the build on first attribute access.
    """
    global _graph
    if name == "graph":
        if _graph is None:
            _graph = create_hermes_agent()
        return _graph
    raise AttributeError(f"module 'langstage_hermes.agent' has no attribute {name!r}")
