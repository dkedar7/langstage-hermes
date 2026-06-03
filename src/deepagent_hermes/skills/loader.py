"""``SkillLoaderMiddleware`` — appends the skills block to the system prompt.

Per SPEC §5 the prompt-assembly module owns the *full* system prompt. This
middleware only contributes its slice: the rendered skills index plus the
bodies of any skills the model has already loaded via ``skill_view`` (which
live in ``state["loaded_skill_bodies"]`` per SPEC §10.4).

We override ``wrap_model_call`` instead of using ``@dynamic_prompt`` because
``@dynamic_prompt`` *replaces* the system prompt — we want to *append* to
whatever the upstream ``PromptAssemblyMiddleware`` produced.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest

from deepagent_hermes.skills.library import SkillLibrary
from deepagent_hermes.skills.prompt import build_skills_system_prompt

logger = logging.getLogger(__name__)

__all__ = ["SkillLoaderMiddleware"]


_BANNER = "## Loaded skills"


def _union_active_skills(a: list[str] | None, b: list[str] | None) -> list[str]:
    seen: dict[str, None] = {}
    for src in (a or []), (b or []):
        for name in src:
            seen.setdefault(name, None)
    return list(seen.keys())


def _merge_loaded_bodies(a: dict[str, str] | None, b: dict[str, str] | None) -> dict[str, str]:
    return {**(a or {}), **(b or {})}


class _SkillLoaderStateExt(AgentState):
    """Declare ``active_skills`` + ``loaded_skill_bodies`` on the merged graph
    state schema so the ``skill_view`` tool's state updates persist.

    Reducer-annotated so parallel ``skill_view`` calls from different
    middleware paths in the same superstep accumulate rather than crash.
    """

    active_skills: NotRequired[Annotated[list[str], _union_active_skills]]
    loaded_skill_bodies: NotRequired[Annotated[dict[str, str], _merge_loaded_bodies]]


class SkillLoaderMiddleware(AgentMiddleware):
    """Inject the skills index + any loaded skill bodies into the system prompt.

    The skills block is appended to ``request.system_prompt`` (or its
    SystemMessage equivalent) on every model call. Loaded skill bodies are
    pulled from ``state["loaded_skill_bodies"]`` (populated by the
    ``skill_view`` tool — see ``tools.py``) so a viewed skill persists in
    context for the rest of the session without re-emitting through tool
    results.
    """

    state_schema = _SkillLoaderStateExt

    def __init__(self, library: SkillLibrary) -> None:
        super().__init__()
        self.library = library

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        """Append the skills block to the system prompt for this call."""
        addition = self._build_addition(request)
        if addition:
            base = request.system_prompt or ""
            joined = (base + "\n\n" + addition).strip() if base else addition
            request = request.override(system_prompt=joined)
        return handler(request)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_addition(self, request: ModelRequest) -> str:
        """Return the skills block + loaded bodies, or empty string."""
        try:
            skills_block = build_skills_system_prompt(self.library)
        except Exception:
            logger.exception("failed to build skills system prompt; skipping")
            skills_block = ""

        loaded_bodies = self._loaded_bodies(request)
        loaded_block = self._render_loaded(loaded_bodies)

        parts = [p for p in (skills_block, loaded_block) if p]
        return "\n\n".join(parts)

    @staticmethod
    def _loaded_bodies(request: ModelRequest) -> dict[str, str]:
        state = getattr(request, "state", None) or {}
        bodies = state.get("loaded_skill_bodies") if isinstance(state, dict) else None
        if not isinstance(bodies, dict):
            return {}
        # Defensive: cast values to str so a corrupt state entry can't blow up rendering.
        return {str(k): str(v) for k, v in bodies.items() if v}

    @staticmethod
    def _render_loaded(bodies: dict[str, str]) -> str:
        if not bodies:
            return ""
        chunks: list[str] = [_BANNER]
        for name in sorted(bodies):
            body = bodies[name].strip()
            chunks.append(f"<skill name=\"{name}\">\n{body}\n</skill>")
        return "\n\n".join(chunks)
