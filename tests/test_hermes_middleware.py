"""``hermes_middleware()``: Hermes's learning loop inside someone else's agent.

The host builds its own agent with plain ``langchain.agents.create_agent`` (or
``deepagents.create_deep_agent``), keeps its model, tools and system prompt, and
adds ``*hermes_middleware()``. Keyless: scripted models only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import Field

from langstage_hermes import HermesConfig, hermes_middleware
from langstage_hermes.cron import jobs as cron_jobs
from langstage_hermes.demo import DEMO_SKILL_NAME, DemoReviewModel, _ScriptedModel
from langstage_hermes.embed import COMPONENTS, DEFAULT_COMPONENTS, HermesMiddleware
from langstage_hermes.prompts import PromptAssemblyMiddleware

HOST_PROMPT = "You are Acme's support agent. Always answer in one short paragraph."


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by id."""
    return f"order {order_id}: shipped"


class _HostModel(_ScriptedModel):
    """The host's model: a few `lookup_order` calls, then an answer.

    Records the tools it was bound with and the system prompt of every call.
    """

    tool_turns: int = 0
    turns: list[AIMessage] = Field(default_factory=list)
    seen_tools: ClassVar[list[str]] = []
    seen_system: ClassVar[list[str]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        type(self).seen_tools = sorted(getattr(t, "name", None) or t["name"] for t in tools)
        return self

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any):
        if messages and getattr(messages[0], "type", "") == "system":
            type(self).seen_system.append(messages[0].text)
        n = sum(isinstance(m, AIMessage) for m in messages)
        if n < len(self.turns):
            return self._result(self.turns[n])
        if n < self.tool_turns:
            return self._result(
                AIMessage(content="", tool_calls=[{"name": "lookup_order", "id": f"lo-{n}", "args": {"order_id": str(n)}}])
            )
        return self._result(AIMessage(content="Your order has shipped."))


class _QuietAux(_ScriptedModel):
    def _generate(self, messages: list[Any], *a: Any, **k: Any):
        return self._result(AIMessage(content="nothing to save"))


@pytest.fixture
def home(tmp_hermes_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    _HostModel.seen_tools = []
    _HostModel.seen_system = []
    return tmp_hermes_home


def _cfg(**overrides: Any) -> HermesConfig:
    return HermesConfig.resolve(overrides=overrides) if overrides else HermesConfig.resolve()


def _host(hermes: list[Any], model: Any = None) -> Any:
    return create_agent(
        model or _HostModel(),
        tools=[lookup_order],
        system_prompt=HOST_PROMPT,
        middleware=[*hermes],
        checkpointer=InMemorySaver(),
    )


def _run(agent: Any, text: str = "where is my order?", thread: str = "t1") -> dict[str, Any]:
    return agent.invoke({"messages": [HumanMessage(text)]}, {"configurable": {"thread_id": thread}})


@pytest.fixture
def made():
    """Close every HermesMiddleware a test builds (Windows locks an open state.db)."""
    built: list[HermesMiddleware] = []
    yield built
    for h in built:
        h.close()


def _hermes(made: list[HermesMiddleware], cfg: HermesConfig | None = None, **kw: Any) -> HermesMiddleware:
    kw.setdefault("aux_model", _QuietAux())
    h = hermes_middleware(cfg or _cfg(), **kw)
    made.append(h)
    return h


# ── shape ───────────────────────────────────────────────────────────────


def test_default_components_and_a_plain_list(home: Path, made):
    h = _hermes(made)
    assert isinstance(h, list)
    assert h.components == DEFAULT_COMPONENTS
    names = [type(m).__name__ for m in h]
    assert names == [
        "PromptAssemblyMiddleware",
        "SkillLoaderMiddleware",
        "MemoryToolMiddleware",
        "HermesStateRecorderMiddleware",
        "ReflectionMiddleware",
        "CuratorMiddleware",
    ]
    # Nothing that belongs to the host.
    host_owned = {"FilesystemMiddleware", "ModelOverrideMiddleware", "AnthropicCachingS3Middleware", "PatchToolCallsMiddleware"}
    assert not host_owned & set(names)
    assert h.store is not None and h.library is not None and h.session_id.startswith("sess-")


def test_include_adds_and_exclude_removes(home: Path, made):
    h = _hermes(made, include=["compression", "budget", "todo"], exclude=["curator", "recorder"])
    assert set(h.components) == {"skills", "memory", "reflection", "compression", "budget", "todo"}
    names = [type(m).__name__ for m in h]
    assert names[0] == "IterationBudgetMiddleware"  # outermost: it can stop the loop before anything runs
    assert "CuratorMiddleware" not in names and "HermesStateRecorderMiddleware" not in names
    assert {"HermesCompressionMiddleware", "TodoListMiddleware"} <= set(names)


@pytest.mark.parametrize("arg", ["include", "exclude"])
def test_unknown_component_names_raise(home: Path, arg: str):
    with pytest.raises(ValueError) as e:
        hermes_middleware(_cfg(), aux_model=_QuietAux(), **{arg: ["reflection", "filesystem"]})
    msg = str(e.value)
    assert "'filesystem'" in msg and arg in msg
    for name in COMPONENTS:
        assert name in msg


def test_a_single_string_is_one_name(home: Path, made):
    assert "curator" not in _hermes(made, exclude="curator").components


# ── inside a plain create_agent ──────────────────────────────────────────


def test_host_and_hermes_tools_are_both_bound(home: Path, made):
    agent = _host(_hermes(made))
    _run(agent)
    assert set(_HostModel.seen_tools) == {
        "lookup_order",
        "memory",
        "session_search",
        "skill_manage",
        "skill_view",
        "skills_list",
    }


def test_the_host_prompt_is_kept_verbatim_and_first(home: Path, made):
    (home / "memories").mkdir(parents=True, exist_ok=True)
    (home / "memories" / "USER.md").write_text("Prefers metric units.", encoding="utf-8")
    _run(_host(_hermes(made)))
    system = _HostModel.seen_system[0]
    assert system.startswith(HOST_PROMPT + "\n\n")
    assert system.count(HOST_PROMPT) == 1
    rest = system[len(HOST_PROMPT) :]
    assert "Prefers metric units." in rest  # the frozen user snapshot
    assert "skill" in rest.lower()  # skills guidance / index
    # None of the full agent's own sections.
    assert "Conversation started:" not in system
    assert "You are Hermes" not in system


def test_embedded_prompt_without_a_host_prompt(home: Path):
    mw = PromptAssemblyMiddleware(enabled_toolsets=["memory"], embedded=True)
    text = mw.assemble_embedded(state={"memory_snapshot": "M", "user_snapshot": "U"})
    assert text.endswith("M\n\nU")


def test_a_turn_is_recorded_in_the_store(home: Path, made):
    h = _hermes(made)
    out = _run(_host(h), "where is order 4242?")
    sid = out["session_id"]
    assert h.store.get_session(sid) is not None
    hits = h.store.search_messages("4242")
    assert hits and all(hit["session_id"] == sid for hit in hits)


def test_reflection_runs_the_review_at_its_interval(home: Path, made):
    cfg = _cfg(skills_creation_nudge_interval=3, memory_nudge_interval=50)
    h = _hermes(made, cfg, aux_model=DemoReviewModel())
    _run(_host(h, _HostModel(tool_turns=3)))
    assert [p.parent.name for p in (home / "skills").rglob("SKILL.md")] == [DEMO_SKILL_NAME]


def test_no_review_below_the_interval(home: Path, made):
    cfg = _cfg(skills_creation_nudge_interval=10, memory_nudge_interval=50)
    h = _hermes(made, cfg, aux_model=DemoReviewModel())
    _run(_host(h, _HostModel(tool_turns=2)))
    skills_dir = home / "skills"
    assert not skills_dir.is_dir() or not list(skills_dir.rglob("SKILL.md"))


def test_a_disabled_toolset_is_not_bound(home: Path, made):
    _run(_host(_hermes(made, _cfg(agent_disabled_toolsets=["memory"]))))
    assert "memory" not in _HostModel.seen_tools
    assert "lookup_order" in _HostModel.seen_tools


# ── cronjob (opt-in, approval-gated) ─────────────────────────────────────


def _cron(action: str, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "cronjob", "args": {"action": action, **args}, "id": f"c-{action}"}])


CREATE = {"schedule": "0 9 * * *", "prompt": "summarise the repo", "name": "digest"}


def test_cronjob_is_off_by_default(home: Path, made):
    _run(_host(_hermes(made)))
    assert "cronjob" not in _HostModel.seen_tools


def test_cronjob_create_waits_for_approval(home: Path, made, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LANGSTAGE_HERMES_INTERRUPT_ON", raising=False)
    agent = _host(_hermes(made, include=["cronjob"]), _HostModel(turns=[_cron("create", **CREATE)]))
    out = _run(agent, "remind me every morning")
    assert "cronjob" in _HostModel.seen_tools
    assert "__interrupt__" in out
    assert "schedule a new cron job [0 9 * * *]" in out["__interrupt__"][0].value["action_requests"][0]["description"]
    assert cron_jobs.list_jobs() == []
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), {"configurable": {"thread_id": "t1"}})
    assert [j["name"] for j in cron_jobs.list_jobs()] == ["digest"]


def test_cronjob_list_does_not_wait(home: Path, made):
    cron_jobs.create_job(prompt="p", schedule="30m", name="existing")
    out = _run(_host(_hermes(made, include=["cronjob"]), _HostModel(turns=[_cron("list")])))
    assert "__interrupt__" not in out
    assert any("existing" in str(m.content) for m in out["messages"] if m.type == "tool")


def test_cron_gate_sits_next_to_a_host_approval_middleware(home: Path, made):
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    h = _hermes(made, include=["cronjob"])
    create_agent(
        _HostModel(),
        tools=[lookup_order],
        middleware=[*h, HumanInTheLoopMiddleware(interrupt_on={"lookup_order": True})],
        checkpointer=InMemorySaver(),
    )


# ── inside deepagents.create_deep_agent ──────────────────────────────────


def test_works_with_create_deep_agent(home: Path, made):
    from deepagents import create_deep_agent

    agent = create_deep_agent(
        model=_HostModel(),
        tools=[lookup_order],
        system_prompt=HOST_PROMPT,
        middleware=[*_hermes(made)],
        checkpointer=InMemorySaver(),
    )
    _run(agent)
    assert {"lookup_order", "memory", "skills_list", "session_search"} <= set(_HostModel.seen_tools)
    assert HOST_PROMPT in _HostModel.seen_system[0]
