"""``PromptAssemblyMiddleware`` — three-layer system prompt with prefix-cache discipline.

SPEC §5 maps the Hermes ``build_system_prompt_parts`` (in ``agent/system_prompt.py``)
onto a ``@dynamic_prompt``-style ``AgentMiddleware`` that overrides
``wrap_model_call`` and replaces ``request.system_message`` with our assembled
prompt before delegating to the handler.

Three layers, joined with ``"\\n\\n"``:

* **stable**   — identity (SOUL.md > ``default_identity.md``), tool-aware guidance
  (memory / session_search / skills), ``task_completion.md``, optional
  ``computer_use.md``, ``tool_use_enforcement.md`` (gated by model id), Google /
  OpenAI execution-discipline blocks (gated by model id), environment hints
  (python / platform / cwd), platform hint from ``platform_hints/<platform>.md``.
* **context**  — caller-supplied ``system_message`` plus context-file content
  (``AGENTS.md`` / ``.cursorrules`` / ``HERMES.md``) discovered by walking up
  from cwd. Each file is scanned through
  :func:`langstage_hermes.memory.threat_patterns.scan`; matches are replaced
  with ``"[BLOCKED: <reason>]"`` so prompt-injection payloads don't reach the
  model.
* **volatile** — ``state["memory_snapshot"]``, ``state["user_snapshot"]``, then
  a **date-only** line ``"Conversation started: <Weekday, Month DD, YYYY>"``
  (no minute precision — byte-stable for the entire day, which is what keeps
  the upstream prefix cache warm across turns). Optional trailing lines for
  ``Session ID:`` / ``Model:`` / ``Provider:`` if set.

The middleware does NOT cache its own output — that's the caller's job (see
:class:`langstage_hermes.caching.AnthropicCachingS3Middleware`). All we do is
assemble deterministically so the *bytes* are identical from turn N to turn
N+1 within a session.
"""

from __future__ import annotations

import os
import platform as _platform_mod
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, SystemMessage

from langstage_hermes.memory.threat_patterns import scan as _threat_scan

if TYPE_CHECKING:
    pass


# ── prompt loading ───────────────────────────────────────────────────


_PROMPT_PACKAGE = "langstage_hermes._prompts"
_PROMPTS_DIR = Path(__file__).resolve().parent / "_prompts"


def load_prompt(name: str) -> str:
    """Read a prompt file by relative name (e.g. ``"task_completion.md"``).

    Prompts live inside the package at ``langstage_hermes/_prompts/`` (as of
    v0.1.2 — previously the broken ``shared-data`` config left wheels without
    them). Direct filesystem fallback covers loaders that don't expose
    ``importlib.resources``.

    Returns ``""`` if the file does not exist — callers append-and-strip, so a
    missing optional prompt should not break assembly.
    """
    try:
        return resources.files(_PROMPT_PACKAGE).joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, AttributeError, NotADirectoryError):
        path = _PROMPTS_DIR.joinpath(name)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return ""


# ── gating constants (SPEC §5) ───────────────────────────────────────


TOOL_USE_ENFORCEMENT_MODELS: tuple[str, ...] = (
    "gpt",
    "codex",
    "gemini",
    "gemma",
    "grok",
    "glm",
    "qwen",
    "deepseek",
)
"""Model-name substrings that opt-in to ``tool_use_enforcement.md``."""

_GOOGLE_MODEL_TOKENS: tuple[str, ...] = ("gemini", "gemma")
_OPENAI_MODEL_TOKENS: tuple[str, ...] = ("gpt", "codex", "grok")

_CONTEXT_FILENAMES: tuple[str, ...] = ("AGENTS.md", ".cursorrules", "HERMES.md")
"""Filenames considered for the context layer, in order of preference."""

_TOOLSET_GUIDANCE_FILES: dict[str, str] = {
    "memory": "memory_guidance.md",
    "session_search": "session_search_guidance.md",
    "skills": "skills_guidance.md",
}
"""Map ``enabled_toolsets`` entries to the guidance file they unlock."""


# ── soul / identity helpers ──────────────────────────────────────────


def _resolve_hermes_home() -> Path | None:
    """Return ``<HERMES_HOME>`` if set in the environment, else ``None``.

    Respects both ``DEEPAGENT_HERMES_HOME`` (our convention) and ``HERMES_HOME``
    (Hermes-native). The first set wins.
    """
    raw = os.environ.get("LANGSTAGE_HERMES_HOME") or os.environ.get("DEEPAGENT_HERMES_HOME") or os.environ.get("HERMES_HOME")
    if not raw:
        return None
    return Path(raw).expanduser()


def _load_identity() -> str:
    """``<HERMES_HOME>/SOUL.md`` if present, else ``default_identity.md``.

    Per Hermes convention, ``SOUL.md`` lets a user override the agent's
    identity globally; the default is always present and ships with the
    package so a fresh install still has a coherent persona.
    """
    home = _resolve_hermes_home()
    if home is not None:
        soul = home / "SOUL.md"
        if soul.is_file():
            text = soul.read_text(encoding="utf-8").strip()
            if text:
                return text
    return load_prompt("default_identity.md").strip()


# ── context-file discovery ───────────────────────────────────────────


def build_context_files_prompt(cwd: Path | None = None) -> str:
    """Walk up from ``cwd`` looking for ``AGENTS.md`` / ``.cursorrules`` / ``HERMES.md``.

    Each file found is scanned through
    :func:`langstage_hermes.memory.threat_patterns.scan` with ``scope="context"``.
    On a hit, the file body is replaced with ``f"[BLOCKED: {reason}]"`` so a
    poisoned ``AGENTS.md`` (e.g. checked in by a hostile dependency) cannot
    inject instructions into the system prompt.

    Files are joined with ``"\\n\\n"``; the directory order is **deepest first**
    (current cwd before its parent) so the most-specific guidance wins by
    appearing closest to the model's "recency window" within the context layer.

    Returns ``""`` if nothing was found, so the caller can skip the section
    cleanly.
    """
    start = Path(cwd or Path.cwd()).resolve()
    seen: set[Path] = set()
    blocks: list[str] = []

    # Walk from cwd up to root, deepest first.
    current: Path | None = start
    while current is not None:
        for fname in _CONTEXT_FILENAMES:
            candidate = current / fname
            if not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)

            try:
                body = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not body:
                continue

            reason = _threat_scan(body, scope="context")
            if reason is not None:
                blocks.append(f"[BLOCKED: {reason}]")
            else:
                blocks.append(f"# {fname} ({current})\n{body}")

        parent = current.parent
        if parent == current:
            break
        current = parent

    return "\n\n".join(blocks)


# ── environment hints ────────────────────────────────────────────────


def _build_environment_hints(*, workspace_root: Path | None) -> str:
    """One-line environment summary: python version, platform, cwd.

    Deterministic for the lifetime of the process — does NOT include any
    wall-clock value (the date line is the only time-varying piece, and it
    sits in the volatile layer alongside ``memory_snapshot``).
    """
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    plat = _platform_mod.platform(terse=True)
    cwd = (workspace_root or Path.cwd()).resolve()
    return f"Environment: Python {py} on {plat}. Working directory: {cwd}"


# ── stable-layer assembly ────────────────────────────────────────────


def _model_id_lower(request: ModelRequest) -> str:
    """Best-effort lowercase model identifier from a ModelRequest.

    LangChain chat models expose the id under different attribute names per
    provider (``model``, ``model_name``, ``deployment``, ...). Try a few; if
    none stick, return empty string so all model-id gates fall to default.
    """
    model = request.model
    for attr in ("model", "model_name", "deployment", "deployment_name", "deployment_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value.lower()
    return ""


def _append_nonempty(parts: list[str], *items: str) -> None:
    """Append each non-empty stripped item to ``parts`` — quiet little helper."""
    for item in items:
        if item:
            parts.append(item)


def _model_family_blocks(model_id: str) -> list[str]:
    """Return the model-family-gated guidance blocks for ``model_id``."""
    if not model_id or not any(token in model_id for token in TOOL_USE_ENFORCEMENT_MODELS):
        return []
    out: list[str] = []
    _append_nonempty(out, load_prompt("tool_use_enforcement.md").strip())
    if any(token in model_id for token in _GOOGLE_MODEL_TOKENS):
        _append_nonempty(out, load_prompt("google_execution.md").strip())
    if any(token in model_id for token in _OPENAI_MODEL_TOKENS):
        _append_nonempty(out, load_prompt("openai_execution.md").strip())
    return out


def _toolset_guidance_blocks(enabled_toolsets: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for toolset in enabled_toolsets:
        fname = _TOOLSET_GUIDANCE_FILES.get(toolset)
        if not fname:
            continue
        _append_nonempty(out, load_prompt(fname).strip())
    return out


def _build_stable_layer(
    *,
    enabled_toolsets: tuple[str, ...],
    model_id: str,
    platform: str,
    workspace_root: Path | None,
) -> str:
    """Assemble the stable layer (identity + guidance + env hints + platform hint)."""
    parts: list[str] = []

    _append_nonempty(parts, _load_identity())
    parts.extend(_toolset_guidance_blocks(enabled_toolsets))
    _append_nonempty(parts, load_prompt("task_completion.md").strip())

    if "computer_use" in enabled_toolsets:
        _append_nonempty(parts, load_prompt("computer_use.md").strip())

    parts.extend(_model_family_blocks(model_id))

    _append_nonempty(parts, _build_environment_hints(workspace_root=workspace_root))

    if platform:
        _append_nonempty(parts, load_prompt(f"platform_hints/{platform.lower()}.md").strip())

    return "\n\n".join(p for p in parts if p)


# ── context-layer assembly ───────────────────────────────────────────


def _build_context_layer(
    *,
    system_message: str,
    workspace_root: Path | None,
) -> str:
    parts: list[str] = []
    if system_message.strip():
        parts.append(system_message.strip())
    files_block = build_context_files_prompt(workspace_root).strip()
    if files_block:
        parts.append(files_block)
    return "\n\n".join(parts)


# ── volatile-layer assembly ──────────────────────────────────────────


def _build_volatile_layer(
    *,
    state: Any,
    model_id: str,
    provider: str,
) -> str:
    """Assemble the volatile layer.

    Pulls ``memory_snapshot`` and ``user_snapshot`` from state (frozen at
    session start per SPEC §13.1 — they DO NOT mutate mid-session, which is
    what makes the byte-stable prompt possible). Adds the date-only
    ``Conversation started: ...`` line, plus optional ``Session ID`` /
    ``Model`` / ``Provider`` trailers.

    State access is duck-typed via ``.get`` so a missing/empty value falls back
    to the empty string without raising.
    """
    parts: list[str] = []

    def _get(key: str, default: Any = "") -> Any:
        if state is None:
            return default
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    memory_snapshot = _get("memory_snapshot", "")
    if isinstance(memory_snapshot, str) and memory_snapshot.strip():
        parts.append(memory_snapshot.strip())

    user_snapshot = _get("user_snapshot", "")
    if isinstance(user_snapshot, str) and user_snapshot.strip():
        parts.append(user_snapshot.strip())

    # Date-only — byte-stable for the full day. See SPEC §5 + comment in
    # hermes-agent/agent/system_prompt.py: minute precision invalidates the
    # prefix cache on every rebuild path.
    date_line = f"Conversation started: {datetime.now().strftime('%A, %B %d, %Y')}"
    trailer_bits: list[str] = []
    session_id = _get("session_id", "")
    if isinstance(session_id, str) and session_id:
        trailer_bits.append(f"Session ID: {session_id}")
    if model_id:
        trailer_bits.append(f"Model: {model_id}")
    if provider:
        trailer_bits.append(f"Provider: {provider}")
    if trailer_bits:
        date_line = "\n".join([date_line, *trailer_bits])
    parts.append(date_line)

    return "\n\n".join(p for p in parts if p)


# ── middleware ───────────────────────────────────────────────────────


class PromptAssemblyMiddleware(AgentMiddleware):
    """Build the 3-layer system prompt on every model call.

    The middleware is intentionally idempotent and pure: given the same
    state + enabled toolsets + workspace + date, it returns the same bytes.
    That's the entire point — caching downstream relies on it.

    Constructor args:
        enabled_toolsets: Which toolsets are wired into this agent. Used to
            decide which tool-aware guidance blocks (memory / session_search /
            skills / computer_use) to include. Order is preserved so callers
            can pin the in-prompt order. Default: ``()`` (no tool guidance,
            useful for smoke tests).
        platform: Platform key matching a file under ``prompts/platform_hints/``
            (e.g. ``"cli"``, ``"cron"``, ``"telegram"``). Default ``"cli"``.
        system_message: Caller-supplied system-prompt override that goes into
            the context layer. Use this to thread per-instance customisation
            without touching the stable layer. Default: empty.
        workspace_root: Working directory for context-file discovery and the
            environment hint. Default: ``None`` (use the process cwd at
            assembly time).
    """

    def __init__(
        self,
        *,
        enabled_toolsets: tuple[str, ...] | list[str] = (),
        platform: str = "cli",
        system_message: str = "",
        workspace_root: Path | None = None,
    ) -> None:
        self.enabled_toolsets: tuple[str, ...] = tuple(enabled_toolsets)
        self.platform = platform
        self.system_message = system_message
        self.workspace_root = workspace_root

    # ── public assembly entry point (handy for tests & debugging) ────

    def assemble(
        self,
        *,
        state: Any = None,
        model_id: str = "",
        provider: str = "",
    ) -> str:
        """Return the full prompt without going through ``wrap_model_call``.

        Tests call this directly; the middleware just wires the request shape
        into ``assemble(...)`` and overrides the request's system message.
        """
        stable = _build_stable_layer(
            enabled_toolsets=self.enabled_toolsets,
            model_id=model_id,
            platform=self.platform,
            workspace_root=self.workspace_root,
        )
        context = _build_context_layer(
            system_message=self.system_message,
            workspace_root=self.workspace_root,
        )
        volatile = _build_volatile_layer(
            state=state,
            model_id=model_id,
            provider=provider,
        )
        return "\n\n".join(part for part in (stable, context, volatile) if part)

    # ── middleware hooks ─────────────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        model_id = _model_id_lower(request)
        # Provider is rarely exposed cleanly on chat models — best-effort.
        provider = ""
        for attr in ("_llm_type", "provider", "_provider"):
            value = getattr(request.model, attr, None)
            if isinstance(value, str) and value:
                provider = value
                break

        # ``request.state`` may be a dict or BaseModel — pass it as-is to the
        # volatile-layer builder which duck-types both shapes.
        prompt = self.assemble(state=request.state, model_id=model_id, provider=provider)
        new_request = request.override(system_message=SystemMessage(content=prompt))
        return handler(new_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        # Identical to sync path — assembly is pure / sync.
        model_id = _model_id_lower(request)
        provider = ""
        for attr in ("_llm_type", "provider", "_provider"):
            value = getattr(request.model, attr, None)
            if isinstance(value, str) and value:
                provider = value
                break
        prompt = self.assemble(state=request.state, model_id=model_id, provider=provider)
        new_request = request.override(system_message=SystemMessage(content=prompt))
        return await handler(new_request)


__all__ = [
    "TOOL_USE_ENFORCEMENT_MODELS",
    "PromptAssemblyMiddleware",
    "build_context_files_prompt",
    "load_prompt",
]
