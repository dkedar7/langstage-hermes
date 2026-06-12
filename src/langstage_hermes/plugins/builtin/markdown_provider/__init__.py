"""``MarkdownProvider`` — bundled, zero-dependency ``MemoryProvider``.

Recalls relevant sections from ``<HERMES_HOME>/memories/notes/*.md`` via
plain keyword overlap. Designed for the case where the user has
hand-authored or agent-written context that's too long for the
frozen-snapshot ``MEMORY.md`` (which is 2200 chars) but worth surfacing
when relevant — e.g. project briefs, architectural notes, personal cheat
sheets.

Why markdown + a Python function instead of a service:

- ``MEMORY.md`` and ``USER.md`` are already in the system prompt every
  turn (the frozen-snapshot layer). They don't *need* a recall provider.
- The interesting surface is "long-form context the agent should pull in
  only when relevant" — and for that, plain markdown files + substring
  search are sufficient. No external service, no auth, no embeddings.
- The agent can write its own notes here via the filesystem tools when
  it discovers something worth remembering at length.

Notes file format
-----------------

Each ``.md`` file is split into sections at every ``\\n## `` boundary
(or ``\\n# `` if no H2s). The leading heading is kept with its section
for context. ``recall(query)`` returns sections whose lowered text
contains any query token of length ≥ 3.

This is deliberately dumb — keyword overlap is robust, debuggable, and
fast. Anyone wanting BM25 / vector / dialectic recall can write their
own provider against the :class:`MemoryProvider` ABC and ship it as a
plugin.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import ClassVar, Literal

from langstage_hermes.memory.provider import (
    MemoryProvider,
    RecallMode,
    available_providers,
    register_provider,
)

logger = logging.getLogger(__name__)


def _resolve_notes_dir() -> Path:
    """``<HERMES_HOME>/memories/notes`` — matches the rest of the memory layout."""
    from langstage_hermes.config import hermes_home

    home = str(hermes_home())
    return Path(home) / "memories" / "notes"


_SECTION_HEAD = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _split_into_sections(text: str) -> list[str]:
    """Split a markdown file at H1 / H2 / H3 boundaries. Heading lines stay
    with their section so the recall result carries its own context."""
    if not text.strip():
        return []
    matches = list(_SECTION_HEAD.finditer(text))
    if not matches:
        # No headings — the whole file is one section.
        return [text.strip()]
    sections: list[str] = []
    # Anything before the first heading is its own preamble section.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(preamble)
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            sections.append(chunk)
    return sections


def _tokenize_query(query: str) -> list[str]:
    """Lowercased query tokens of length ≥ 3 — short tokens ('a', 'in', 'is')
    create too much noise. Punctuation is stripped."""
    raw = re.findall(r"[a-zA-Z0-9_]+", query.lower())
    return [t for t in raw if len(t) >= 3]


def search_notes(query: str, notes_dir: Path, *, limit: int = 5) -> list[str]:
    """Pure recall function — exported for direct use without instantiating
    the provider (so the agent's tooling, tests, or CLI can call it too).

    Ranks sections by the count of distinct query tokens present, then by
    section length (shorter wins on ties — more focused snippets).
    """
    if not notes_dir.is_dir():
        return []
    tokens = _tokenize_query(query)
    if not tokens:
        return []
    scored: list[tuple[int, int, str]] = []  # (hit_count, -length, section)
    for path in sorted(notes_dir.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("MarkdownProvider: could not read %s: %s", path, exc)
            continue
        for section in _split_into_sections(content):
            lowered = section.lower()
            hits = sum(1 for t in tokens if t in lowered)
            if hits > 0:
                # Prepend the source file so the agent knows where it came from.
                prefixed = f"_From {path.name}:_\n{section}"
                scored.append((hits, -len(section), prefixed))
    scored.sort(reverse=True)
    return [s for _hits, _neg_len, s in scored[:limit]]


class MarkdownProvider(MemoryProvider):
    """Recall over ``<HERMES_HOME>/memories/notes/*.md`` via keyword search."""

    name: ClassVar[str] = "markdown"

    def __init__(self, *, notes_dir: Path | None = None, limit: int = 5) -> None:
        """Args:
        notes_dir: Override the default ``<HERMES_HOME>/memories/notes``.
            Tests pass a tmp dir; production resolves from env.
        limit: Max sections to return per ``recall()`` call.
        """
        self._notes_dir = notes_dir
        self._limit = limit

    def setup_session(self, session_id: str, user_id: str | None = None) -> None:
        # Nothing to set up; recall reads files on demand. The session_id /
        # user_id args are kept for ABC compatibility.
        del session_id, user_id

    def recall(self, query: str, mode: RecallMode | Literal["auto"] = "hybrid") -> list[str]:
        # All three modes behave identically here — there's no dialectic /
        # short-vs-long distinction to make against a flat notes corpus.
        # We accept the param for ABC compatibility and to keep config-swap
        # behavior with other providers consistent.
        del mode
        notes_dir = self._notes_dir or _resolve_notes_dir()
        return search_notes(query, notes_dir, limit=self._limit)

    def record_turn(self, role: str, content: str) -> None:
        # Markdown notes are user-curated (or agent-curated via the
        # filesystem tools), not auto-appended on every turn. The frozen-
        # snapshot MemoryToolMiddleware already handles the "agent decides
        # this is worth remembering" path via the `memory` tool.
        del role, content

    def teardown(self) -> None:
        # No handles to release.
        pass


# Self-register on import so the plugin loader picks us up.
register_provider("markdown", MarkdownProvider)


def register(ctx: object) -> None:
    """No-op plugin-loader entry point.

    The plugin loader (SPEC §15) calls ``register(ctx)`` on every discovered
    plugin's ``__init__.py``. We don't *need* it — registration into the
    memory-provider registry already happens at import time via the
    ``register_provider(...)`` call above, and that's what the agent factory
    consults. But the loader's contract requires a callable named ``register``,
    so we provide one that just confirms the import side-effect happened.

    Before v0.1.2 this function was missing, which made the loader log
    ``"Plugin 'markdown-provider' has no callable register()"`` on every
    fresh-install ``plugins list`` — embarrassing.
    """
    del ctx
    if "markdown" not in available_providers():  # pragma: no cover — defensive
        register_provider("markdown", MarkdownProvider)


__all__ = ["MarkdownProvider", "register", "search_notes"]
