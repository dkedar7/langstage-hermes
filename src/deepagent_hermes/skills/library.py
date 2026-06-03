"""Skill library — filesystem scanner + CRUD over SKILL.md directories.

A *skill* is a directory containing a ``SKILL.md`` file (YAML frontmatter +
markdown body), per the agentskills.io spec. Skills live in one or more
search directories; later directories in the search list win on name
collision (per SPEC §10.2: project > user > bundled means the caller
passes ``[bundled, user, project]`` and the project dir overrides).

This module is intentionally pure: it does not import any langchain or
langgraph machinery. The middleware in ``loader.py`` and the tools in
``tools.py`` wrap this library.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import frontmatter

from deepagent_hermes.skills.validator import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
)
from deepagent_hermes.skills.validator import (
    validate as validate_frontmatter,
)

logger = logging.getLogger(__name__)

__all__ = ["Skill", "SkillLibrary"]


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

# Map agentskills/Hermes platform strings to ``sys.platform`` prefixes.
_OS_PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}

# Directories we never treat as skills.
_EXCLUDED_DIR_NAMES = frozenset(
    {
        "_archived",
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
    }
)


def _current_os_platform() -> str:
    """Return the agentskills platform name matching the current OS."""
    sp = sys.platform
    for name, prefix in _OS_PLATFORM_MAP.items():
        if sp.startswith(prefix):
            return name
    return sp


def _current_session_platform() -> str:
    """Resolve the active *session* platform (cli, telegram, cron, ...).

    SPEC §10 uses this for ``skills.platform_disabled`` lookups. Defaults to
    ``"cli"``. ``HERMES_PLATFORM`` wins over ``HERMES_SESSION_PLATFORM``.
    """
    return os.environ.get("HERMES_PLATFORM") or os.environ.get("HERMES_SESSION_PLATFORM") or "cli"


# ---------------------------------------------------------------------------
# Default directory resolution
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    """Resolve the user's hermes-home directory."""
    explicit = os.environ.get("DEEPAGENT_HERMES_HOME") or os.environ.get("HERMES_HOME")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".deepagent-hermes"


def _bundled_skills_dir() -> Path:
    """The bundled ``skills/`` directory shipped with the package."""
    # ``library.py`` lives at src/deepagent_hermes/skills/library.py
    # bundled skills live at <repo>/skills/
    return Path(__file__).resolve().parents[3] / "skills"


def _default_search_dirs() -> list[Path]:
    """Default ordered search dirs: bundled < user < project (later wins)."""
    dirs: list[Path] = []
    bundled = _bundled_skills_dir()
    if bundled.exists():
        dirs.append(bundled)
    user_dir = _hermes_home() / "skills"
    dirs.append(user_dir)
    project_dir = Path.cwd() / ".deepagent-hermes" / "skills"
    if project_dir.exists():
        dirs.append(project_dir)
    return dirs


# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A single skill loaded from disk.

    Attributes:
        name: The ``name`` frontmatter field (canonical identifier).
        description: The ``description`` frontmatter field, possibly truncated
            to ``MAX_DESCRIPTION_LENGTH``.
        body: The markdown body of SKILL.md (everything after the frontmatter).
        path: Absolute path to the SKILL.md file.
        metadata: Full parsed frontmatter dict.
        category: Top-level subdirectory under a search dir (e.g.
            ``"software-development"``), if any.
        platforms: Optional list of OS platforms the skill targets.
        version: Optional Hermes extension.
        pinned: Convenience accessor for ``metadata.hermes.pinned``.
    """

    name: str
    description: str
    body: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    category: str | None = None
    platforms: list[str] | None = None
    version: str | None = None

    @property
    def directory(self) -> Path:
        """The directory that contains SKILL.md."""
        return self.path.parent

    @property
    def pinned(self) -> bool:
        meta = self.metadata.get("metadata")
        if not isinstance(meta, dict):
            return False
        hermes = meta.get("hermes")
        if not isinstance(hermes, dict):
            return False
        return bool(hermes.get("pinned"))


# ---------------------------------------------------------------------------
# SkillLibrary
# ---------------------------------------------------------------------------


class SkillLibrary:
    """Filesystem-backed library of skills.

    Args:
        dirs: Ordered list of search directories. Later entries win on
            name collision. If omitted, uses the default set
            (bundled + user + project — see :func:`_default_search_dirs`).
        config: Optional dict shaped like ``{"disabled": [...],
            "platform_disabled": {"<platform>": [...]}}`` — see SPEC §2.

    The library performs no caching across calls — every ``list()`` /
    ``get()`` re-scans disk. The prompt builder in ``prompt.py`` adds the
    LRU + disk cache on top, keyed on file mtimes.
    """

    def __init__(
        self,
        dirs: list[Path] | None = None,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.dirs: list[Path] = [Path(d) for d in (dirs if dirs is not None else _default_search_dirs())]
        self.config: dict[str, Any] = config or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list(self) -> list[Skill]:
        """Return all skills visible under the current platform/config.

        Filter rules (any condition removes the skill):

        1. ``name`` in ``config["disabled"]``
        2. ``name`` in ``config["platform_disabled"][<session_platform>]``
        3. ``platforms`` field set and does NOT include current OS platform

        On name collisions across directories, the *later* search dir wins
        (matches SPEC §10.2 ordering when caller passes [bundled, user, project]).
        """
        disabled = set(self.config.get("disabled", []) or [])
        platform_disabled_map = self.config.get("platform_disabled", {}) or {}
        session_platform = _current_session_platform()
        platform_disabled = set(platform_disabled_map.get(session_platform, []) or [])
        current_os = _current_os_platform()

        # name -> Skill; later dirs overwrite earlier ones.
        seen: dict[str, Skill] = {}
        for directory in self.dirs:
            if not directory.exists():
                continue
            for skill in self._scan_directory(directory):
                if skill.platforms and current_os not in skill.platforms:
                    continue
                if skill.name in disabled or skill.name in platform_disabled:
                    continue
                seen[skill.name] = skill

        return sorted(seen.values(), key=lambda s: (s.category or "", s.name))

    def get(self, name: str) -> Skill | None:
        """Find a single skill by name.

        Resolution mirrors ``list()`` ordering (later dir wins).
        """
        for skill in self.list():
            if skill.name == name:
                return skill
        return None

    def write(
        self,
        name: str,
        frontmatter_data: dict[str, Any],
        body: str,
        *,
        category: str | None = None,
        target_dir: Path | None = None,
    ) -> Path:
        """Create or overwrite ``<dir>/<category>/<name>/SKILL.md``.

        Writes to the *last* writable search dir (the user/project dir)
        unless ``target_dir`` is supplied. Validates the frontmatter first
        and raises ``ValueError`` on failure.

        Returns the path to the written SKILL.md.
        """
        # Ensure the supplied name matches frontmatter
        frontmatter_data = dict(frontmatter_data)
        frontmatter_data.setdefault("name", name)
        if frontmatter_data.get("name") != name:
            raise ValueError(f"frontmatter name {frontmatter_data.get('name')!r} does not match supplied name {name!r}")

        errors = validate_frontmatter(frontmatter_data, parent_dir_name=name)
        if errors:
            raise ValueError(f"SKILL.md frontmatter for {name!r} is invalid:\n- " + "\n- ".join(errors))

        base = Path(target_dir) if target_dir is not None else self._default_write_dir()
        skill_root = base / category / name if category else base / name
        skill_root.mkdir(parents=True, exist_ok=True)
        skill_md = skill_root / "SKILL.md"

        post = frontmatter.Post(body, **frontmatter_data)
        skill_md.write_bytes(frontmatter.dumps(post).encode("utf-8"))
        return skill_md

    def delete(self, name: str) -> bool:
        """Archive a skill to ``<dir>/_archived/<name>-<timestamp>/``.

        Returns ``True`` if the skill was found and archived, ``False`` if no
        such skill exists.
        """
        skill = self.get(name)
        if skill is None:
            return False

        # Archive under the writable dir of origin (which is `skill.directory`'s
        # nearest search-dir ancestor).
        origin_search_dir = self._find_search_dir_for(skill.path)
        if origin_search_dir is None:
            # Couldn't locate the search dir — fall back to default writable dir.
            origin_search_dir = self._default_write_dir()

        archive_root = origin_search_dir / "_archived"
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        dest = archive_root / f"{skill.name}-{stamp}"
        shutil.move(str(skill.directory), str(dest))
        return True

    def validate_all(self) -> dict[str, list[str]]:
        """Validate every SKILL.md under every search dir.

        Returns ``{skill_name: [errors...]}``. Skills whose frontmatter has no
        valid ``name`` are keyed by their parent dir name with a ``__error__``
        prefix.
        """
        results: dict[str, list[str]] = {}
        for directory in self.dirs:
            if not directory.exists():
                continue
            for skill_md in self._iter_skill_md_files(directory):
                parent_name = skill_md.parent.name
                try:
                    post = frontmatter.load(skill_md)
                    meta = dict(post.metadata)
                except Exception as exc:
                    results[f"__error__/{parent_name}"] = [f"parse failure: {exc}"]
                    continue
                errors = validate_frontmatter(meta, parent_dir_name=parent_name)
                key = meta.get("name") or f"__error__/{parent_name}"
                results[key] = errors
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _default_write_dir(self) -> Path:
        """The last directory in the search list (highest precedence).

        Per SPEC §10.2: user / project / bundled — the caller is expected to
        order them so that the last entry is the one new skills should land in.
        Falls back to ``$HERMES_HOME/skills``.
        """
        for directory in reversed(self.dirs):
            # Prefer a non-bundled dir. Bundled lives inside the package source tree.
            if directory != _bundled_skills_dir():
                directory.mkdir(parents=True, exist_ok=True)
                return directory
        fallback = _hermes_home() / "skills"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _find_search_dir_for(self, path: Path) -> Path | None:
        for directory in self.dirs:
            try:
                path.resolve().relative_to(directory.resolve())
                return directory
            except (ValueError, OSError):
                continue
        return None

    def _scan_directory(self, directory: Path) -> Iterator[Skill]:
        for skill_md in self._iter_skill_md_files(directory):
            try:
                skill = self._load_skill(skill_md, base_dir=directory)
            except Exception as exc:
                logger.debug("skipping unparseable skill at %s: %s", skill_md, exc)
                continue
            if skill is not None:
                yield skill

    @staticmethod
    def _iter_skill_md_files(directory: Path) -> Iterator[Path]:
        """Yield all SKILL.md files under ``directory``, skipping excluded dirs."""
        for skill_md in directory.rglob("SKILL.md"):
            parts = set(skill_md.parts)
            if parts & _EXCLUDED_DIR_NAMES:
                continue
            yield skill_md

    def _load_skill(self, skill_md: Path, *, base_dir: Path) -> Skill | None:
        post = frontmatter.load(skill_md)
        meta = dict(post.metadata)

        raw_name = meta.get("name")
        if not isinstance(raw_name, str) or not raw_name:
            # Fall back to parent dir name — better to surface in listings than skip.
            raw_name = skill_md.parent.name
        name = raw_name[:MAX_NAME_LENGTH]

        description = meta.get("description", "")
        if not isinstance(description, str):
            description = str(description)
        if len(description) > MAX_DESCRIPTION_LENGTH:
            description = description[: MAX_DESCRIPTION_LENGTH - 3] + "..."

        category = _extract_category(skill_md, base_dir=base_dir)

        platforms_raw = meta.get("platforms")
        platforms: list[str] | None = None
        if isinstance(platforms_raw, (list, tuple)):
            platforms = [str(p) for p in platforms_raw if isinstance(p, str)]
        elif isinstance(platforms_raw, str):
            platforms = [platforms_raw]

        version_raw = meta.get("version")
        version = str(version_raw) if isinstance(version_raw, str) else None

        return Skill(
            name=name,
            description=description,
            body=post.content,
            path=skill_md,
            metadata=meta,
            category=category,
            platforms=platforms,
            version=version,
        )


def _extract_category(skill_md: Path, *, base_dir: Path) -> str | None:
    """Top-level subdir under the search dir is the category.

    Example: ``<dir>/software-development/git-workflow/SKILL.md`` -> ``"software-development"``.
    A skill placed directly under the search dir (``<dir>/foo/SKILL.md``) has
    no category.
    """
    try:
        rel = skill_md.relative_to(base_dir)
    except ValueError:
        return None
    parts = rel.parts
    # parts = (category?, ..., skill_dir, "SKILL.md")
    if len(parts) >= 3:
        return parts[0]
    return None
