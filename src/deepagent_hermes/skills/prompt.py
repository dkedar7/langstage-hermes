"""Render the ``## Skills (mandatory)`` system-prompt block.

Two-layer cache (mirrors Hermes ``agent/prompt_builder.py``):

1. In-process ``functools.lru_cache(maxsize=1)`` keyed on a hash of
   (file paths, mtimes, sizes) — invalidated whenever any SKILL.md
   changes on disk.
2. Disk snapshot at ``<HERMES_HOME>/.skills_prompt_snapshot.json``
   (or a caller-supplied path) reused at process startup when the
   on-disk manifest still matches.

The preface text is a paraphrase of Hermes's preface (see
``hermes-agent/agent/prompt_builder.py`` lines 1236-1262), worded for a
deepagents host rather than for Hermes-specific slash commands.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

from deepagent_hermes.skills.library import Skill, SkillLibrary

__all__ = ["build_skills_system_prompt", "clear_prompt_cache"]

logger = logging.getLogger(__name__)


PREFACE = (
    "Before replying, scan the skills below. If a skill matches or is even "
    "partially relevant to your task, you MUST load it with skill_view(name) "
    "and follow its instructions. Err on the side of loading — it is always "
    "better to have context you don't need than to miss critical steps, "
    "pitfalls, or established workflows. Skills contain specialized knowledge "
    "— API endpoints, tool-specific commands, and proven workflows that "
    "outperform general-purpose approaches. Load the skill even if you think "
    "you could handle the task with basic tools. Skills also encode the user's "
    "preferred approach, conventions, and quality standards — load them even "
    "for tasks you already know how to do, because the skill defines how it "
    "should be done here.\n"
    "If a skill has issues, fix it with skill_manage(action='patch'). "
    "After difficult or iterative tasks, offer to save the workflow as a new "
    "skill. If a skill you loaded was missing steps, had wrong commands, or "
    "needed pitfalls you discovered, update it before finishing."
)

_EMPTY_PROMPT = ""  # returned when the library has no visible skills


# ---------------------------------------------------------------------------
# Cache plumbing
# ---------------------------------------------------------------------------

# OrderedDict keyed on the manifest hash. Holds at most one entry — this is the
# in-process LRU layer.
_PROMPT_CACHE: OrderedDict[str, str] = OrderedDict()
_PROMPT_CACHE_LOCK = threading.Lock()
_PROMPT_CACHE_MAX = 1


def clear_prompt_cache() -> None:
    """Drop the in-process prompt cache. Disk snapshot is left alone."""
    with _PROMPT_CACHE_LOCK:
        _PROMPT_CACHE.clear()


def _manifest_for(skills: Iterable[Skill]) -> list[tuple[str, int, int]]:
    """Return a deterministic manifest of (path, mtime_ns, size) tuples.

    Used to detect any disk change to the underlying SKILL.md files.
    """
    manifest: list[tuple[str, int, int]] = []
    for skill in skills:
        try:
            stat = skill.path.stat()
            manifest.append((str(skill.path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            # File vanished between scan and stat; record a sentinel so the
            # manifest hash changes on the next scan.
            manifest.append((str(skill.path), 0, -1))
    manifest.sort()
    return manifest


def _hash_manifest(manifest: list[tuple[str, int, int]]) -> str:
    h = hashlib.sha256()
    for path, mtime_ns, size in manifest:
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(str(mtime_ns).encode("ascii"))
        h.update(b"\0")
        h.update(str(size).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _read_disk_snapshot(cache_path: Path) -> tuple[str, str] | None:
    """Return ``(manifest_hash, prompt)`` from the snapshot file, or None."""
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("could not read skills prompt snapshot %s: %s", cache_path, exc)
        return None
    manifest_hash = data.get("manifest_hash")
    prompt = data.get("prompt")
    if isinstance(manifest_hash, str) and isinstance(prompt, str):
        return manifest_hash, prompt
    return None


def _write_disk_snapshot(cache_path: Path, manifest_hash: str, prompt: str) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"manifest_hash": manifest_hash, "prompt": prompt}),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("could not write skills prompt snapshot %s: %s", cache_path, exc)


def _default_cache_path() -> Path:
    home = os.environ.get("DEEPAGENT_HERMES_HOME") or os.environ.get("HERMES_HOME")
    if home:
        return Path(home).expanduser() / ".skills_prompt_snapshot.json"
    return Path.home() / ".deepagent-hermes" / ".skills_prompt_snapshot.json"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(skills: list[Skill]) -> str:
    if not skills:
        return _EMPTY_PROMPT

    # Group by category; "" sorts as "uncategorized"
    by_category: dict[str, list[Skill]] = {}
    for skill in skills:
        cat = skill.category or "uncategorized"
        by_category.setdefault(cat, []).append(skill)

    lines: list[str] = []
    for category in sorted(by_category):
        lines.append(f"  {category}:")
        seen: set[str] = set()
        for skill in sorted(by_category[category], key=lambda s: s.name):
            if skill.name in seen:
                continue
            seen.add(skill.name)
            desc = skill.description.strip()
            if desc:
                lines.append(f"    - {skill.name}: {desc}")
            else:
                lines.append(f"    - {skill.name}")

    return (
        "## Skills (mandatory)\n"
        + PREFACE
        + "\n\n<available_skills>\n"
        + "\n".join(lines)
        + "\n</available_skills>\n\n"
        "Only proceed without loading a skill if genuinely none are relevant to the task."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_skills_system_prompt(
    library: SkillLibrary,
    *,
    cache_path: Path | None = None,
) -> str:
    """Render the ``## Skills (mandatory)`` block for the system prompt.

    Args:
        library: The skill library to scan.
        cache_path: Override the disk snapshot location. ``None`` uses
            ``$HERMES_HOME/.skills_prompt_snapshot.json``.

    Returns:
        The rendered prompt block, or an empty string if the library is empty.
    """
    skills = library.list()
    manifest = _manifest_for(skills)
    manifest_hash = _hash_manifest(manifest)

    # 1. In-process LRU hit?
    with _PROMPT_CACHE_LOCK:
        cached = _PROMPT_CACHE.get(manifest_hash)
        if cached is not None:
            _PROMPT_CACHE.move_to_end(manifest_hash)
            return cached

    # 2. Disk snapshot hit?
    snapshot_path = cache_path if cache_path is not None else _default_cache_path()
    snapshot = _read_disk_snapshot(snapshot_path)
    if snapshot is not None and snapshot[0] == manifest_hash:
        prompt = snapshot[1]
        _store_prompt(manifest_hash, prompt)
        return prompt

    # 3. Render afresh and populate both caches.
    prompt = _render(skills)
    _store_prompt(manifest_hash, prompt)
    _write_disk_snapshot(snapshot_path, manifest_hash, prompt)
    return prompt


def _store_prompt(manifest_hash: str, prompt: str) -> None:
    with _PROMPT_CACHE_LOCK:
        _PROMPT_CACHE[manifest_hash] = prompt
        _PROMPT_CACHE.move_to_end(manifest_hash)
        while len(_PROMPT_CACHE) > _PROMPT_CACHE_MAX:
            _PROMPT_CACHE.popitem(last=False)
