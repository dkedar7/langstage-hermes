"""Add Hermes's learning loop to an agent you already have.

:func:`hermes_middleware` returns a list of ``AgentMiddleware`` for
``langchain.agents.create_agent`` (or ``deepagents.create_deep_agent``). Your
model, tools and system prompt stay yours. Hermes adds the parts that make an
agent learn from its own sessions:

* ``reflection``: every few turns, a review subagent looks back over the
  conversation and saves what worked as a skill or a memory entry.
* ``skills``: the skill library (``skills_list`` / ``skill_view`` /
  ``skill_manage``) and its index in the system prompt.
* ``memory``: the ``memory`` tool plus the frozen memory and user snapshots.
* ``recorder``: every turn is written to the FTS5 store at
  ``<HERMES_HOME>/state.db``; ``session_search`` searches it.
* ``curator``: archives skills nobody has used in a while.

Opt-in by name with ``include=``: ``compression`` (summarise long contexts),
``todo`` (``write_todos``), ``cronjob`` (the agent manages its own scheduled
jobs; ``create`` / ``resume`` / ``run-now`` wait for your approval) and
``budget`` (cap tool-loop iterations).

Hermes never replaces your system prompt. Its sections (tool guidance, then the
memory and user snapshots, then the skills index) are appended after yours.

Not included, because they belong to the host agent: the filesystem and
workspace-path middleware, model override, prompt caching, patching of dangling
tool calls, and Hermes's own identity prompt. Use :func:`create_hermes_agent`
when you want the whole Hermes agent instead.

Everything here is a plain ``langchain`` ``AgentMiddleware``; nothing depends on
hooks that only ``deepagents`` provides.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from langstage_hermes.config import HermesConfig

DEFAULT_COMPONENTS: tuple[str, ...] = ("skills", "memory", "recorder", "reflection", "curator")
"""Returned unless excluded."""

OPTIONAL_COMPONENTS: tuple[str, ...] = ("compression", "todo", "cronjob", "budget")
"""Returned only when named in ``include``."""

COMPONENTS: tuple[str, ...] = DEFAULT_COMPONENTS + OPTIONAL_COMPONENTS


class HermesMiddleware(list):
    """The middleware list :func:`hermes_middleware` returns.

    It is a plain ``list`` (unpack it into ``middleware=[...]``) that also keeps
    the shared resources, so a caller can inspect them or close them:

    * ``config``: the resolved :class:`~langstage_hermes.config.HermesConfig`.
    * ``store``: the FTS5 store the recorder writes to.
    * ``library``: the skill library.
    * ``session_id``: the id skill changes are attributed to.
    * ``components``: the component names that were built, in order.
    """

    config: HermesConfig
    store: Any
    library: Any
    session_id: str
    components: tuple[str, ...]

    def close(self) -> None:
        """Close the SQLite connections (store and skill audit log). Best-effort.

        On Windows an open connection locks ``state.db``; close before removing
        a throwaway ``HERMES_HOME``.
        """
        audit = getattr(self.library, "audit_log", None)
        for resource in (self.store, audit):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - defensive
                    pass


def _names(value: Iterable[str] | str | None, arg: str) -> set[str]:
    if value is None:
        return set()
    names = {value} if isinstance(value, str) else set(value)
    unknown = sorted(n for n in names if n not in COMPONENTS)
    if unknown:
        raise ValueError(
            f"hermes_middleware({arg}=...): unknown component(s) {', '.join(map(repr, unknown))}. "
            f"Valid names: {', '.join(COMPONENTS)} (default: {', '.join(DEFAULT_COMPONENTS)})."
        )
    return names


def _cron_approval_middleware(interrupt_on: dict[str, Any]) -> Any:
    """A ``HumanInTheLoopMiddleware`` under its own name.

    ``create_agent`` refuses two middleware with the same name, so the cron gate
    gets a name of its own and can sit next to the host's own approval
    middleware.
    """
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    class HermesCronApprovalMiddleware(HumanInTheLoopMiddleware):
        """Approval for ``cronjob`` create / resume / run-now (see cron/tool.py)."""

    return HermesCronApprovalMiddleware(interrupt_on=interrupt_on)


def hermes_middleware(
    config: HermesConfig | None = None,
    *,
    store: Any = None,
    aux_model: Any = None,
    session_id: str | None = None,
    include: Iterable[str] | str | None = None,
    exclude: Iterable[str] | str | None = None,
) -> HermesMiddleware:
    """Hermes's learning loop as middleware for an existing agent.

    Example::

        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver
        from langstage_hermes import hermes_middleware

        hermes = hermes_middleware()
        agent = create_agent(
            "anthropic:claude-sonnet-4-6",
            tools=[my_tool],
            system_prompt="You are Acme's support agent.",
            middleware=[*hermes],
            checkpointer=InMemorySaver(),
        )

    Args:
        config: Resolved configuration; defaults to ``HermesConfig.resolve()``
            (``langstage-hermes.toml`` and ``LANGSTAGE_HERMES_*``). It decides
            ``HERMES_HOME``, the nudge intervals, memory limits, curator timing
            and ``[agent].disabled_toolsets``, as for the full agent.
        store: The FTS5 store to record into. Defaults to
            ``<HERMES_HOME>/state.db``.
        aux_model: The model the review subagent (and ``compression``'s
            summariser) uses: a chat model instance or a ``"provider:name"``
            string. Defaults to ``config.model_aux``, else ``config.model_default``.
            Pass your own model here to keep Hermes on the same provider and key
            as your agent.
        session_id: The session conversations are recorded under in ``state.db``,
            the id skill changes are attributed to in the audit log, and the
            session ``session_search`` leaves out as current. Auto-generated if
            omitted; ``.session_id`` reports it. Every thread run through this list
            records into that one session unless the run's input state carries its
            own ``session_id`` (e.g. ``{"messages": [...], "session_id": "chat-7"}``),
            which then wins for that thread.
        include: Extra components to add: any of ``compression``, ``todo``,
            ``cronjob``, ``budget``. Default components may be named too (a no-op).
        exclude: Components to leave out, e.g. ``exclude=["curator"]``.

    Returns:
        A :class:`HermesMiddleware` list. Unpack it into ``middleware=``. The tools
        Hermes adds (``memory``, the skill tools, ``session_search``, ``cronjob``)
        ride on the middleware, so ``create_agent`` binds them without you
        listing them.

    Raises:
        ValueError: ``include`` or ``exclude`` names an unknown component.

    Notes:
        ``todo`` adds langchain's ``TodoListMiddleware``. ``create_deep_agent``
        already includes one, and ``create_agent`` refuses duplicates, so leave
        ``todo`` out there. ``cronjob``'s approval pause, like any interrupt,
        needs the host agent to have a checkpointer.
    """
    # Shared construction lives in agent.py so this and create_hermes_agent agree.
    from langstage_hermes import agent as _agent
    from langstage_hermes.curator import CuratorMiddleware
    from langstage_hermes.memory.tool import MemoryToolMiddleware
    from langstage_hermes.prompts import PromptAssemblyMiddleware
    from langstage_hermes.reflection import ReflectionMiddleware
    from langstage_hermes.search.session_search import make_session_search_tool
    from langstage_hermes.skills.loader import SkillLoaderMiddleware
    from langstage_hermes.skills.tools import make_skill_tools
    from langstage_hermes.store.recorder import HermesStateRecorderMiddleware
    from langstage_hermes.tools.toolsets import tool_allowed

    wanted = (set(DEFAULT_COMPONENTS) | _names(include, "include")) - _names(exclude, "exclude")

    cfg = config or HermesConfig.resolve()
    sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
    store, db_path = _agent._open_store(cfg, store)
    library = _agent._open_library(cfg, sid, db_path)
    enabled_toolsets = _agent._enabled_toolsets(cfg)

    def _allowed(tools: Iterable[Any]) -> list[Any]:
        return [t for t in tools if tool_allowed(getattr(t, "name", ""), enabled_toolsets)]

    if aux_model is None or isinstance(aux_model, str):
        aux_model = _agent._init_chat_model(aux_model or cfg.model_aux or cfg.model_default)

    skill_tools = _allowed(make_skill_tools(library, store=store)) if "skills" in wanted else []
    memory_mw = (
        MemoryToolMiddleware(memory_char_limit=cfg.memory_char_limit, user_char_limit=cfg.memory_user_char_limit)
        if "memory" in wanted
        else None
    )
    if memory_mw is not None:
        memory_mw.tools = _allowed(memory_mw.tools)
    search_tools = (
        _allowed([make_session_search_tool(store, current_session_id_getter=lambda: sid)]) if "recorder" in wanted else []
    )

    # Prompt guidance only for the Hermes toolsets that are actually bound.
    guidance: list[str] = []
    if memory_mw is not None and memory_mw.tools:
        guidance.append("memory")
    if search_tools:
        guidance.append("session_search")
    if skill_tools:
        guidance.append("skills")

    out = HermesMiddleware()
    built: list[str] = []

    if "budget" in wanted:
        from langstage_hermes.budget import IterationBudgetMiddleware

        out.append(IterationBudgetMiddleware(max_iterations=cfg.agent_max_iterations))
        built.append("budget")
    # Appends after the host's prompt; the skill loader then adds the skills index.
    if guidance or memory_mw is not None:
        out.append(PromptAssemblyMiddleware(enabled_toolsets=guidance, embedded=True))
    if "skills" in wanted:
        out.append(SkillLoaderMiddleware(library, tools=skill_tools))
        built.append("skills")
    if memory_mw is not None:
        out.append(memory_mw)
        built.append("memory")
    if "recorder" in wanted:
        out.append(HermesStateRecorderMiddleware(store=store, tools=search_tools, session_id=sid))
        built.append("recorder")
    if "reflection" in wanted:
        # The review fork writes through the same skill and memory tools the agent has.
        review_tools = [*skill_tools, *(memory_mw.tools if memory_mw is not None else [])]
        _, review_graph = _agent._build_review(library, store, aux_model, review_tools)
        out.append(
            ReflectionMiddleware(
                skill_nudge_interval=cfg.skills_creation_nudge_interval,
                memory_nudge_interval=cfg.memory_nudge_interval,
                library=library,
                store=store,
                model=aux_model,
                aux_model=aux_model,
                skills_toolset_enabled=bool(skill_tools),
                review_graph=review_graph,
            )
        )
        built.append("reflection")
    if "curator" in wanted:
        out.append(
            CuratorMiddleware(
                library,
                store,
                interval_hours=cfg.curator_interval_hours,
                min_idle_hours=cfg.curator_min_idle_hours,
                stale_days=cfg.curator_stale_after_days,
                archive_days=cfg.curator_archive_after_days,
                enabled=cfg.curator_enabled,
            )
        )
        built.append("curator")
    if "todo" in wanted:
        from langchain.agents.middleware import TodoListMiddleware

        out.append(TodoListMiddleware())
        built.append("todo")
    if "compression" in wanted:
        from langstage_hermes.compression import HermesCompressionMiddleware

        out.append(
            HermesCompressionMiddleware(
                model=aux_model,
                aux_model=aux_model,
                threshold_percent=cfg.compression_threshold,
                protect_first_n=cfg.compression_protect_first_n,
                protect_last_n=cfg.compression_protect_last_n,
                summary_target_ratio=cfg.compression_target_ratio,
                abort_on_summary_failure=cfg.compression_abort_on_summary_failure,
            )
        )
        built.append("compression")
    if "cronjob" in wanted and "cronjob" in enabled_toolsets:
        # Naming it in include= is the opt-in here ([cron] agent_tool is for the full
        # agent). A disabled `cronjob` toolset still wins, as it does there.
        from langchain.agents.middleware import AgentMiddleware

        from langstage_hermes.cron.tool import cronjob, cronjob_interrupt_config

        class HermesCronToolMiddleware(AgentMiddleware):
            """Carries the ``cronjob`` tool so the host's create_agent binds it."""

            def __init__(self) -> None:
                super().__init__()
                self.tools = [cronjob]

        out.append(HermesCronToolMiddleware())
        out.append(_cron_approval_middleware({"cronjob": cronjob_interrupt_config()}))
        built.append("cronjob")

    out.config = cfg
    out.store = store
    out.library = library
    out.session_id = sid
    out.components = tuple(built)
    return out


__all__ = [
    "COMPONENTS",
    "DEFAULT_COMPONENTS",
    "OPTIONAL_COMPONENTS",
    "HermesMiddleware",
    "hermes_middleware",
]
