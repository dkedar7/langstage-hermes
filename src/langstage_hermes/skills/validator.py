"""agentskills.io frontmatter validation for SKILL.md files.

Implements the rules from https://agentskills.io/specification.md verbatim,
plus Hermes-specific extensions (``version``, ``platforms``,
``prerequisites``, ``metadata.hermes.*``) that remain spec-valid because
agentskills.io permits an arbitrary ``metadata`` map and treats unknown
top-level keys as forward-compatible.

The validator is intentionally tolerant of extra keys (rather than rejecting
them) because skill authors routinely add custom fields and the spec does
not forbid that.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["ValidationError", "is_valid", "validate"]


# agentskills.io constants
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500

# Strict per the spec: unicode lowercase ASCII letters, digits, hyphens.
# (The spec text says "unicode lowercase alphanumeric" but the examples and
# the reference validator use ASCII; we follow the reference validator.)
_NAME_CHARS_RE = re.compile(r"^[a-z0-9-]+$")
_CONSECUTIVE_HYPHENS_RE = re.compile(r"--")

# Hermes-specific platform whitelist (loader/runtime extension)
_VALID_PLATFORMS = frozenset({"macos", "linux", "windows"})


class ValidationError(ValueError):
    """Raised when frontmatter fails validation and a hard error is desired.

    The functional ``validate()`` API returns a list of errors instead — this
    exception exists for callers that want exception-based control flow.
    """


def _err(errors: list[str], msg: str) -> None:
    errors.append(msg)


def _validate_name(value: Any, *, parent_dir_name: str | None, errors: list[str]) -> None:
    if value is None or value == "":
        _err(errors, "name: required field is missing or empty")
        return
    if not isinstance(value, str):
        _err(errors, f"name: must be a string, got {type(value).__name__}")
        return
    if len(value) < 1 or len(value) > MAX_NAME_LENGTH:
        _err(errors, f"name: must be 1-{MAX_NAME_LENGTH} characters (got {len(value)})")
    if not _NAME_CHARS_RE.match(value):
        _err(
            errors,
            f"name: may only contain lowercase letters, digits, and hyphens (got {value!r})",
        )
    if value.startswith("-") or value.endswith("-"):
        _err(errors, f"name: must not start or end with a hyphen (got {value!r})")
    if _CONSECUTIVE_HYPHENS_RE.search(value):
        _err(errors, f"name: must not contain consecutive hyphens (got {value!r})")
    if parent_dir_name is not None and value != parent_dir_name:
        _err(
            errors,
            f"name: must match parent directory name (name={value!r}, parent_dir={parent_dir_name!r})",
        )


def _validate_description(value: Any, *, errors: list[str]) -> None:
    if value is None or value == "":
        _err(errors, "description: required field is missing or empty")
        return
    if not isinstance(value, str):
        _err(errors, f"description: must be a string, got {type(value).__name__}")
        return
    stripped = value.strip()
    if not stripped:
        _err(errors, "description: must be non-empty")
        return
    if len(value) > MAX_DESCRIPTION_LENGTH:
        _err(
            errors,
            f"description: must be 1-{MAX_DESCRIPTION_LENGTH} characters (got {len(value)})",
        )


def _validate_compatibility(value: Any, *, errors: list[str]) -> None:
    if not isinstance(value, str):
        _err(errors, f"compatibility: must be a string, got {type(value).__name__}")
        return
    if len(value) < 1 or len(value) > MAX_COMPATIBILITY_LENGTH:
        _err(
            errors,
            f"compatibility: must be 1-{MAX_COMPATIBILITY_LENGTH} characters (got {len(value)})",
        )


def _validate_license(value: Any, *, errors: list[str]) -> None:
    if not isinstance(value, str):
        _err(errors, f"license: must be a string, got {type(value).__name__}")


def _validate_metadata(value: Any, *, errors: list[str]) -> None:
    if not isinstance(value, dict):
        _err(errors, f"metadata: must be a mapping, got {type(value).__name__}")
        return
    # Validate the Hermes extension shape if present, but don't reject other keys.
    hermes = value.get("hermes")
    if hermes is not None and not isinstance(hermes, dict):
        _err(errors, f"metadata.hermes: must be a mapping, got {type(hermes).__name__}")
        return
    if isinstance(hermes, dict):
        tags = hermes.get("tags")
        if tags is not None and not isinstance(tags, (list, tuple, str)):
            _err(
                errors,
                f"metadata.hermes.tags: must be a list (or comma-separated string), got {type(tags).__name__}",
            )
        related = hermes.get("related_skills")
        if related is not None and not isinstance(related, (list, tuple, str)):
            _err(
                errors,
                f"metadata.hermes.related_skills: must be a list (or comma-separated string), got {type(related).__name__}",
            )


def _validate_allowed_tools(value: Any, *, errors: list[str]) -> None:
    if not isinstance(value, str):
        _err(
            errors,
            f"allowed-tools: must be a space-separated string, got {type(value).__name__}",
        )


def _validate_platforms(value: Any, *, errors: list[str]) -> None:
    """Hermes extension: ``platforms`` restricts which OSes load the skill."""
    if not isinstance(value, (list, tuple)):
        _err(errors, f"platforms: must be a list, got {type(value).__name__}")
        return
    for item in value:
        if not isinstance(item, str):
            _err(errors, f"platforms: each entry must be a string, got {type(item).__name__}")
            continue
        if item not in _VALID_PLATFORMS:
            _err(
                errors,
                f"platforms: {item!r} is not a valid platform (allowed: {sorted(_VALID_PLATFORMS)})",
            )


def _validate_prerequisites(value: Any, *, errors: list[str]) -> None:
    """Hermes extension: ``prerequisites: {env_vars, commands}``."""
    if not isinstance(value, dict):
        _err(errors, f"prerequisites: must be a mapping, got {type(value).__name__}")
        return
    for key in ("env_vars", "commands"):
        if key in value and not isinstance(value[key], (list, tuple)):
            _err(
                errors,
                f"prerequisites.{key}: must be a list, got {type(value[key]).__name__}",
            )


def _validate_version(value: Any, *, errors: list[str]) -> None:
    """Hermes extension: free-form version string."""
    if not isinstance(value, str):
        _err(errors, f"version: must be a string, got {type(value).__name__}")


def validate(
    frontmatter: dict[str, Any] | None,
    *,
    parent_dir_name: str | None = None,
) -> list[str]:
    """Validate a parsed SKILL.md frontmatter dict.

    Args:
        frontmatter: Parsed YAML frontmatter as a dict (e.g. from
            ``frontmatter.loads(content).metadata``).
        parent_dir_name: If provided, ``name`` must equal this. agentskills.io
            requires ``name`` to match the parent directory name.

    Returns:
        A list of error strings. Empty list means the frontmatter is valid.
    """
    errors: list[str] = []

    if frontmatter is None:
        return ["frontmatter: missing — SKILL.md must begin with a YAML frontmatter block"]
    if not isinstance(frontmatter, dict):
        return [f"frontmatter: must be a mapping, got {type(frontmatter).__name__}"]

    # Required fields
    _validate_name(frontmatter.get("name"), parent_dir_name=parent_dir_name, errors=errors)
    _validate_description(frontmatter.get("description"), errors=errors)

    # Optional agentskills.io fields
    if "license" in frontmatter:
        _validate_license(frontmatter["license"], errors=errors)
    if "compatibility" in frontmatter:
        _validate_compatibility(frontmatter["compatibility"], errors=errors)
    if "metadata" in frontmatter:
        _validate_metadata(frontmatter["metadata"], errors=errors)
    if "allowed-tools" in frontmatter:
        _validate_allowed_tools(frontmatter["allowed-tools"], errors=errors)

    # Hermes extensions (still valid agentskills.io frontmatter because the spec
    # treats unknown top-level keys as forward-compatible).
    if "version" in frontmatter:
        _validate_version(frontmatter["version"], errors=errors)
    if "platforms" in frontmatter:
        _validate_platforms(frontmatter["platforms"], errors=errors)
    if "prerequisites" in frontmatter:
        _validate_prerequisites(frontmatter["prerequisites"], errors=errors)

    return errors


def is_valid(
    frontmatter: dict[str, Any] | None,
    *,
    parent_dir_name: str | None = None,
) -> bool:
    """True iff ``validate()`` returns no errors."""
    return not validate(frontmatter, parent_dir_name=parent_dir_name)
