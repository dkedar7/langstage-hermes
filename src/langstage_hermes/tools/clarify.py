"""`clarify` tool — ask the human a question and pause.

Renders as a structured prompt the host UI (CLI, TUI, lab notebook, Slack
gateway, …) can detect and display as a clarification request. In Hermes the
analogous tool drives an interactive ``input()`` call; on ``deepagents`` we
return the question as a tool message and rely on the host's
``HumanInTheLoopMiddleware`` (or equivalent) to surface it to the user.

**Platform note:** when the agent is spawned in ``cron`` platform mode there
is no human to respond, so the agent factory MUST strip this tool from the
toolset. See ``cron/scheduler.py`` for the analogous restriction in Hermes.
"""

from __future__ import annotations

from typing import Any

# Tag prefix the host UI can grep for to render a special panel rather than a
# normal tool result. Kept short + ALL_CAPS so a model that paraphrases the
# string in its own output is harder to confuse with a genuine clarify call.
CLARIFY_TAG = "[CLARIFY]"


def _format_clarify(question: str, options: list[str] | None) -> str:
    """Return the structured text the host UI consumes.

    Pure function so unit tests can verify the format without spinning up a
    LangChain tool wrapper.
    """
    body = f"{CLARIFY_TAG} {question.strip()}"
    if options:
        rendered = "\n".join(f"  - {opt}" for opt in options)
        return f"{body}\nOptions:\n{rendered}"
    return body


def _clarify_impl(question: str, options: list[str] | None = None) -> str:
    """Underlying handler — kept independent of the ``@tool`` decorator.

    Allows tests + cron platform code to exercise the formatting logic
    without importing ``langchain_core`` (which is a heavy optional dep
    in the v0.1.0 wheel build).
    """
    if not isinstance(question, str) or not question.strip():
        return f"{CLARIFY_TAG} (empty question — clarify call ignored)"
    return _format_clarify(question, options)


def make_clarify_tool() -> Any:
    """Build the LangChain ``BaseTool`` wrapping :func:`_clarify_impl`.

    Imports ``langchain_core`` lazily so modules that only need the toolset
    enumeration (or the registry, or this module's formatting helpers) don't
    pay the import cost or fail when the optional langchain stack isn't
    installed.

    The returned object has a stable ``.name = "clarify"`` so the registry
    can index it without any special-casing.
    """
    try:
        from langchain_core.tools import tool
    except ImportError as exc:  # pragma: no cover - guarded for headless test envs
        raise RuntimeError(
            "make_clarify_tool() requires langchain_core. Install `langstage-hermes[dev]` or add `langchain-core` to your env."
        ) from exc

    @tool("clarify")
    def clarify(question: str, options: list[str] | None = None) -> str:
        """Ask the user a clarifying question.

        ONLY call this when you genuinely cannot make further progress
        without more information from the user. If you can reasonably infer
        the answer (or pick a sensible default and ask for correction
        after), do that instead — clarify pauses the entire conversation
        until a human replies.

        Args:
            question: The clarifying question, phrased as you'd ask a colleague.
            options: Optional multiple-choice options. When provided, the host
                UI may render them as buttons or a numbered list.
        """
        return _clarify_impl(question, options)

    return clarify


__all__ = ["CLARIFY_TAG", "_clarify_impl", "_format_clarify", "make_clarify_tool"]
