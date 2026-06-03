"""Tests for ``deepagent_hermes.skills.loader.SkillLoaderMiddleware``.

We exercise ``wrap_model_call`` directly with a stub ``ModelRequest`` so the
test stays out of the langchain wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import frontmatter
import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from deepagent_hermes.skills.library import SkillLibrary
from deepagent_hermes.skills.loader import SkillLoaderMiddleware
from deepagent_hermes.skills.prompt import clear_prompt_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill(base: Path, *, name: str, description: str, body: str = "Body") -> Path:
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    skill_md = root / "SKILL.md"
    post = frontmatter.Post(body, name=name, description=description)
    skill_md.write_bytes(frontmatter.dumps(post).encode("utf-8"))
    return skill_md


def _build_request(*, system_prompt: str | None, state: dict[str, Any]) -> ModelRequest:
    """A minimally valid ModelRequest using a fake chat model."""
    model = FakeMessagesListChatModel(responses=[HumanMessage(content="ok")])
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="hi")],
        system_prompt=system_prompt,
        tool_choice=None,
        tools=[],
        response_format=None,
        state=state,
        runtime=Runtime(context=None),
        model_settings={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    """Wipe the prompt LRU between tests so disk changes are observed."""
    clear_prompt_cache()
    yield
    clear_prompt_cache()


@pytest.fixture
def library(tmp_hermes_home) -> SkillLibrary:
    skills_dir = tmp_hermes_home / "skills"
    _write_skill(skills_dir, name="alpha-skill", description="Use for alpha tasks.")
    _write_skill(skills_dir, name="beta-skill", description="Use for beta tasks.")
    return SkillLibrary(dirs=[skills_dir])


# ---------------------------------------------------------------------------
# wrap_model_call behaviour
# ---------------------------------------------------------------------------


def test_skills_block_appended_to_system_prompt(library):
    mw = SkillLoaderMiddleware(library)
    request = _build_request(system_prompt="ROOT-PROMPT", state={"messages": []})

    captured: dict[str, ModelRequest] = {}

    def handler(req: ModelRequest):
        captured["request"] = req

    mw.wrap_model_call(request, handler)
    final_prompt = captured["request"].system_prompt
    assert final_prompt.startswith("ROOT-PROMPT")
    assert "## Skills (mandatory)" in final_prompt
    assert "alpha-skill" in final_prompt
    assert "beta-skill" in final_prompt
    assert "Use for alpha tasks." in final_prompt
    assert "Use for beta tasks." in final_prompt


def test_skills_block_present_even_when_base_prompt_empty(library):
    mw = SkillLoaderMiddleware(library)
    request = _build_request(system_prompt=None, state={"messages": []})

    captured: dict[str, ModelRequest] = {}
    mw.wrap_model_call(request, lambda r: captured.setdefault("request", r))

    final_prompt = captured["request"].system_prompt
    assert final_prompt is not None
    assert "## Skills (mandatory)" in final_prompt
    assert "alpha-skill" in final_prompt


def test_loaded_skill_bodies_injected(library):
    mw = SkillLoaderMiddleware(library)
    state = {
        "messages": [],
        "loaded_skill_bodies": {"alpha-skill": "ALPHA BODY CONTENT"},
    }
    request = _build_request(system_prompt="ROOT", state=state)

    captured: dict[str, ModelRequest] = {}
    mw.wrap_model_call(request, lambda r: captured.setdefault("request", r))

    final_prompt = captured["request"].system_prompt
    assert "## Loaded skills" in final_prompt
    assert "ALPHA BODY CONTENT" in final_prompt
    assert "<skill name=\"alpha-skill\">" in final_prompt


def test_no_skills_no_block(tmp_hermes_home):
    empty_lib = SkillLibrary(dirs=[tmp_hermes_home / "skills"])
    mw = SkillLoaderMiddleware(empty_lib)
    request = _build_request(system_prompt="ROOT", state={"messages": []})

    captured: dict[str, ModelRequest] = {}
    mw.wrap_model_call(request, lambda r: captured.setdefault("request", r))

    # ROOT unchanged (no skills block to append)
    assert captured["request"].system_prompt == "ROOT"


def test_handler_return_value_propagated(library):
    mw = SkillLoaderMiddleware(library)
    request = _build_request(system_prompt="ROOT", state={"messages": []})
    result = mw.wrap_model_call(request, lambda r: "RESULT")
    assert result == "RESULT"


def test_state_missing_loaded_bodies_tolerated(library):
    mw = SkillLoaderMiddleware(library)
    # state without loaded_skill_bodies key
    request = _build_request(system_prompt="ROOT", state={"messages": []})
    captured: dict[str, ModelRequest] = {}
    mw.wrap_model_call(request, lambda r: captured.setdefault("request", r))
    final_prompt = captured["request"].system_prompt
    assert "## Skills (mandatory)" in final_prompt
    # No "Loaded skills" header when nothing has been viewed.
    assert "## Loaded skills" not in final_prompt
