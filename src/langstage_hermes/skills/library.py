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

from langstage_hermes.skills.validator import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
)
from langstage_hermes.skills.validator import (
    validate as validate_frontmatter,
)

logger = logging.getLogger(__name__)

__all__ = ["Skill", "SkillLibrary", "SkillLoadError", "format_load_error"]


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
    from langstage_hermes.config import hermes_home

    return hermes_home().expanduser()


def _bundled_skills_dir() -> Path:
    """The bundled skills directory shipped *inside* the package.

    ``library.py`` lives at ``langstage_hermes/skills/library.py`` and the
    bundled SKILL.md tree is packaged at ``langstage_hermes/_bundled_skills/``
    (``parent.parent`` is the ``langstage_hermes`` package dir). Resolving it
    relative to the package — not the repo root — works identically in a source
    checkout and an installed wheel. The old ``parents[3] / "skills"`` pointed
    at a nonexistent repo-root ``skills/`` dir, so zero of the bundled skills
    loaded anywhere (gh #-dogfood).
    """
    return Path(__file__).resolve().parent.parent / "_bundled_skills"


def _default_search_dirs() -> list[Path]:
    """Default ordered search dirs: bundled < user < project (later wins)."""
    dirs: list[Path] = []
    bundled = _bundled_skills_dir()
    if bundled.exists():
        dirs.append(bundled)
    user_dir = _hermes_home() / "skills"
    dirs.append(user_dir)
    project_dir = Path.cwd() / ".langstage-hermes" / "skills"
    if project_dir.exists():
        dirs.append(project_dir)
    return dirs


# ---------------------------------------------------------------------------
# Parse failures — one vocabulary, shared by every surface
# ---------------------------------------------------------------------------
#
# A SKILL.md whose YAML frontmatter won't parse is *dropped* from ``list()``:
# it disappears from ``skills list`` and from the live agent. That is the right
# behaviour (one bad file must not take down the other 26 skills) but it has to
# be **said out loud** — silence is the defect (gh #81).
#
# ``validate_all()`` (behind ``skills audit``) already detected and worded this
# exact failure. Rather than grow a second copy of "how do we spot and phrase a
# parse failure" for the warning path — the triplication #78 had to unwind —
# both surfaces are built from the three primitives below.


def _error_key(parent_dir_name: str) -> str:
    """The ``validate_all()`` / ``skills audit`` key for a skill with no usable
    ``name``: ``__error__/<parent dir>``. Shared so a warning and an audit row
    name the same thing and a user can connect the two."""
    return f"__error__/{parent_dir_name}"


def _parse_failure(exc: BaseException) -> str:
    """The canonical wording for an unparseable SKILL.md."""
    return f"parse failure: {exc}"


@dataclass(frozen=True)
class SkillLoadError:
    """A SKILL.md that could not be parsed, and is therefore absent from ``list()``.

    Collected on :attr:`SkillLibrary.load_errors` by every scan so callers can
    surface the omission on their own channel (``click.echo(err=True)`` for the
    CLI, ``logging`` for the agent runtime) instead of each re-deriving it.

    Attributes:
        path: Absolute path to the offending SKILL.md.
        parent_name: Name of the directory containing it.
        message: ``"parse failure: <exception>"`` — identical to the string
            ``validate_all()`` records, so ``skills audit`` and the warning
            agree verbatim.
    """

    path: Path
    parent_name: str
    message: str

    @property
    def key(self) -> str:
        """The ``__error__/<dir>`` key ``skills audit`` reports this under."""
        return _error_key(self.parent_name)


def format_load_error(err: SkillLoadError) -> str:
    """Render a one-line, ASCII-only diagnostic for a dropped skill.

    Names the directory, the file and the parse error so the message is
    actionable, and points at ``skills audit`` for the full report. Whitespace
    is collapsed because a ``yaml.scanner.ScannerError`` stringifies to four
    lines — fine in ``audit``'s indented block, unusable as a warning. ASCII
    only: this CLI runs on Windows cp1252 consoles where a stray non-ASCII
    glyph raises ``UnicodeEncodeError``.
    """
    detail = " ".join(err.message.split())
    return (
        f"skipping unparseable skill '{err.parent_name}' at {err.path}: {detail} "
        f"-- it is absent from `skills list` and from the agent; "
        f"run `langstage-hermes skills audit` for the full report ({err.key})"
    )


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
        audit_log: Optional :class:`~langstage_hermes.skills.audit.SkillAuditLog`.
            When provided, every successful ``write()`` / ``delete()``
            appends a row capturing the full SKILL.md before+after so
            the change can be diffed and rolled back. Pass ``None`` for
            test fixtures and read-only callers.

    The library performs no caching across calls — every ``list()`` /
    ``get()`` re-scans disk. The prompt builder in ``prompt.py`` adds the
    LRU + disk cache on top, keyed on file mtimes.
    """

    def __init__(
        self,
        dirs: list[Path] | None = None,
        *,
        config: dict[str, Any] | None = None,
        audit_log: Any = None,
    ) -> None:
        self.dirs: list[Path] = [Path(d) for d in (dirs if dirs is not None else _default_search_dirs())]
        self.config: dict[str, Any] = config or {}
        # Typed as Any so audit.py doesn't have to be imported eagerly at
        # library import time — the library is used in read-only contexts
        # (tests, validators) that should not pay for sqlite startup.
        self.audit_log: Any = audit_log
        # Per-call provenance overrides. The default values are picked up
        # by record-write/delete; callers can swap them in for a single
        # mutation using ``set_mutation_context``.
        self._mutation_source: str | None = None
        self._mutation_session_id: str | None = None
        self._mutation_tool_call_id: str | None = None
        # SKILL.md files the last scan had to drop. Refreshed by every
        # ``list()``; read by the CLI and the agent factory to warn about the
        # omission (gh #81). Empty until the first scan.
        self.load_errors: list[SkillLoadError] = []

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

        Unparseable SKILL.md files are skipped rather than raised — one bad file
        must not hide the rest of the library — but each is recorded on
        :attr:`load_errors` so the caller can say so (gh #81).
        """
        self.load_errors = []
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

    def set_mutation_context(
        self,
        *,
        source: str | None = None,
        session_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        """Stash provenance fields the audit log should record on the next
        mutation. Callers (the agent's ``skill_manage`` tool, the CLI's
        ``audit rollback``) set these before invoking ``write`` / ``delete``.

        The fields stay set until explicitly cleared or overridden — they
        are not auto-reset after a single mutation. That matches the way
        we use them: a tool handler sets them once and may issue several
        related writes (e.g. ``write_file`` then a follow-up validate).
        """
        if source is not None:
            self._mutation_source = source
        if session_id is not None:
            self._mutation_session_id = session_id
        if tool_call_id is not None:
            self._mutation_tool_call_id = tool_call_id

    def _record_mutation(
        self,
        *,
        skill_name: str,
        action: str,
        skill_path: Path | None,
        before_content: bytes | None,
        after_content: bytes | None,
    ) -> None:
        """Best-effort write to the audit log. Failure is logged but never
        raised — the actual file mutation already succeeded by the time
        this is called, and breaking the caller because we can't write to
        the log would be worse than a missing audit row."""
        if self.audit_log is None:
            return
        try:
            self.audit_log.record(
                skill_name=skill_name,
                action=action,
                before_content=before_content,
                after_content=after_content,
                source=self._mutation_source,
                session_id=self._mutation_session_id,
                tool_call_id=self._mutation_tool_call_id,
                skill_path=skill_path,
            )
        except Exception:
            logger.warning("Failed to record %s mutation for skill %s", action, skill_name, exc_info=True)

    def write(
        self,
        name: str,
        frontmatter_data: dict[str, Any],
        body: str,
        *,
        category: str | None = None,
        target_dir: Path | None = None,
        audit_action: str | None = None,
    ) -> Path:
        """Create or overwrite ``<dir>/<category>/<name>/SKILL.md``.

        Writes to the *last* writable search dir (the user/project dir)
        unless ``target_dir`` is supplied. Validates the frontmatter first
        and raises ``ValueError`` on failure.

        Args:
            audit_action: Override the action label recorded in the audit
                log (e.g. ``"create"`` vs ``"write_file"`` vs ``"patch"``).
                Defaults to ``"write_file"`` for an overwrite of an
                existing skill, ``"create"`` for a new one.

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

        before_content = skill_md.read_bytes() if skill_md.exists() else None
        post = frontmatter.Post(body, **frontmatter_data)
        after_bytes = frontmatter.dumps(post).encode("utf-8")
        skill_md.write_bytes(after_bytes)

        action = audit_action or ("create" if before_content is None else "write_file")
        self._record_mutation(
            skill_name=name,
            action=action,
            skill_path=skill_md,
            before_content=before_content,
            after_content=after_bytes,
        )
        return skill_md

    def delete(self, name: str) -> bool:
        """Archive a skill to ``<dir>/_archived/<name>-<timestamp>/``.

        Returns ``True`` if the skill was found and archived, ``False`` if no
        such skill exists.
        """
        skill = self.get(name)
        if skill is None:
            return False

        # Snapshot the SKILL.md before moving so the audit log can record
        # the pre-delete content (recovery aid — rollback uses this).
        before_content = skill.path.read_bytes() if skill.path.exists() else None
        original_path = skill.path

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

        self._record_mutation(
            skill_name=name,
            action="delete",
            skill_path=original_path,
            before_content=before_content,
            after_content=None,
        )
        return True

    def record_install(self, name: str, skill_path: Path) -> None:
        """Log a ``create`` mutation for a skill installed out-of-band.

        The CLI's ``skills install`` copies a whole directory tree (scripts +
        assets, not just SKILL.md), so it can't go through ``write()`` — but it
        should still land an audit row so ``audit log`` shows it and the install
        is consistent with an agent-side ``create`` (rollback then points the
        user at ``delete``, the same as any other create). Best-effort. (gh #31)
        """
        after = skill_path.read_bytes() if skill_path.exists() else None
        self._record_mutation(
            skill_name=name,
            action="create",
            skill_path=skill_path,
            before_content=None,
            after_content=after,
        )

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
                    results[_error_key(parent_name)] = [_parse_failure(exc)]
                    continue
                errors = validate_frontmatter(meta, parent_dir_name=parent_name)
                key = meta.get("name") or _error_key(parent_name)
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
                # Record, don't raise, and don't emit here: ``list()`` is called
                # on *every* model call by SkillLoaderMiddleware, so warning from
                # inside the scan would repeat the same line for the whole
                # session. The two production callers (the CLI's `skills list`,
                # agent.py's one-shot build-time scan) each surface
                # ``load_errors`` once, on their own channel. (gh #81)
                self.load_errors.append(
                    SkillLoadError(
                        path=skill_md,
                        parent_name=skill_md.parent.name,
                        message=_parse_failure(exc),
                    )
                )
                logger.debug("skipping unparseable skill at %s: %s", skill_md, exc)
                continue
            if skill is not None:
                yield skill

    @staticmethod
    def _iter_skill_md_files(directory: Path) -> Iterator[Path]:
        """Yield all SKILL.md files under ``directory``, skipping excluded dirs.

        Exclusions are matched against the path **relative to** ``directory`` —
        i.e. junk dirs *inside* the skill tree (``.venv``, ``node_modules``,
        ``__pycache__`` …). They must NOT be matched against the absolute path,
        or the install *prefix* collides with them: a package installed into a
        venv named ``.venv`` (e.g. the README's ``uv venv .venv``) would have
        ``.venv`` in every bundled skill's absolute path and load ZERO skills
        (gh #-dogfood).
        """
        directory = Path(directory)
        for skill_md in directory.rglob("SKILL.md"):
            try:
                rel_parts = set(skill_md.relative_to(directory).parts)
            except ValueError:  # pragma: no cover - rglob result is always under directory
                rel_parts = set(skill_md.parts)
            if rel_parts & _EXCLUDED_DIR_NAMES:
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
