"""`HermesConfig` — full TOML/env-resolved configuration for ``langstage-hermes``.

Extends ``langstage_core.host.HostConfig`` so we inherit the cross-host
``DEEPAGENT_AGENT_SPEC`` / workspace / port / debug / title plumbing, and adds
every Hermes-specific knob enumerated in SPEC §2 (model / agent / memory /
skills / compression / delegation / curator / cron / plugins).

Resolution chain (lowest to highest precedence):

    defaults  <  $HERMES_HOME/config.toml  <  ./langstage-hermes.toml
              <  LANGSTAGE_HERMES_* env vars  <  explicit overrides

(The global config lives under ``HERMES_HOME`` — ``$HERMES_HOME/config.toml``,
default ``~/.langstage-hermes/config.toml`` — so it moves with a custom
``HERMES_HOME`` rather than staying at the default path.)

(The legacy ``DEEPAGENT_HERMES_*`` spelling still resolves as a fallback and
emits a ``DeprecationWarning``; the canonical ``LANGSTAGE_HERMES_*`` wins.)

The base class' resolution chain still loads ``deepagents.toml`` first, so
shared keys (agent_spec, workspace, ...) keep working unchanged. Hermes-specific
keys live only in ``langstage-hermes.toml``.

Discoverability: ``HermesConfig.resolve().describe()`` prints every field with
its source and the env var / TOML key that sets it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

from langstage_core.host.config import (
    HostConfig,
    _coerce,
    _deep_merge,
    _env_bool,
    _env_pair,
    _get_dotted,
    _malformed_toml,
    _read_toml,
    _warn_legacy_env,
    load_toml_config,
)

# ── Hermes TOML locations ────────────────────────────────────────────

# The global config's DEFAULT location. The *effective* path honors HERMES_HOME —
# it is `hermes_home() / "config.toml"` (see `_hermes_global_toml_path`), so it moves
# with a custom HERMES_HOME; this constant only names where it lives by default.
HERMES_GLOBAL_TOML = Path.home() / ".langstage-hermes" / "config.toml"
HERMES_PROJECT_TOML = "langstage-hermes.toml"
# Pre-rename locations, still honoured. The legacy home dir keeps winning for
# existing installs so skills/memories/state are never orphaned by an upgrade.
LEGACY_HERMES_HOME = Path.home() / ".deepagent-hermes"
LEGACY_HERMES_PROJECT_TOML = "deepagent-hermes.toml"


def hermes_home() -> Path:
    """Resolve ``HERMES_HOME`` with the documented precedence.

    Order: ``LANGSTAGE_HERMES_HOME`` > legacy ``DEEPAGENT_HERMES_HOME`` >
    ``HERMES_HOME`` env > existing ``~/.langstage-hermes`` > existing legacy
    ``~/.deepagent-hermes`` (so pre-rename installs keep their skills and
    memories) > default ``~/.langstage-hermes``. The result is not created —
    that's the caller's job (`tests/conftest.py::tmp_hermes_home` already
    does this).
    """
    override = os.getenv("LANGSTAGE_HERMES_HOME") or os.getenv("DEEPAGENT_HERMES_HOME") or os.getenv("HERMES_HOME")
    if override:
        return Path(override)
    new_home = Path.home() / ".langstage-hermes"
    if new_home.is_dir():
        return new_home
    if LEGACY_HERMES_HOME.is_dir():
        return LEGACY_HERMES_HOME
    return new_home


def _hermes_global_toml_path() -> Path:
    """Compute the global Hermes config path, honoring the home overrides."""
    base = hermes_home()
    return base / "config.toml"


def _find_hermes_project_toml(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (or cwd) looking for ``langstage-hermes.toml``
    (or the legacy ``deepagent-hermes.toml``; new name wins per directory)."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        for fname in (HERMES_PROJECT_TOML, LEGACY_HERMES_PROJECT_TOML):
            candidate = directory / fname
            if candidate.is_file():
                return candidate
    return None


def load_hermes_toml_config(start: Path | None = None) -> tuple[dict, list[Path]]:
    """Load + deep-merge the global and project ``langstage-hermes.toml`` files.

    Project wins on conflicts. Returns ``(merged_config, sources_used)`` —
    ``({}, [])`` if no TOML reader is available.
    """
    sources: list[Path] = []
    merged: dict = {}
    gpath = _hermes_global_toml_path()
    if gpath.is_file():
        merged = _deep_merge(merged, _read_toml(gpath))
        # _read_toml catches a TOMLDecodeError, returns {}, and records the path as
        # malformed — so don't list an ignored file as read (the gh #61 mislabel).
        if str(gpath) not in _malformed_toml:
            sources.append(gpath)
    ppath = _find_hermes_project_toml(start)
    if ppath is not None:
        merged = _deep_merge(merged, _read_toml(ppath))
        if str(ppath) not in _malformed_toml:
            sources.append(ppath)
    return merged, sources


def _parse_toml_files(paths: list[Path]) -> list[tuple[Path, dict]]:
    """Parse each TOML path once, keeping per-file dicts (ascending precedence).

    ``load_*_toml_config`` deep-merge the stack into one dict, which loses which
    file supplied which key — so a value that lives only in the global
    ``config.toml`` can't be told apart from a project override. Keeping the
    per-file dicts lets ``--show-config`` attribute a value to its real origin. (gh #55)
    """
    parsed: list[tuple[Path, dict]] = []
    for p in paths:
        try:
            parsed.append((p, _read_toml(p)))
        except Exception:  # pragma: no cover - loader already skipped malformed files
            continue
    return parsed


def _toml_source_label(parsed: list[tuple[Path, dict]], tkey: str) -> str:
    """Name the HIGHEST-precedence parsed file that actually sets ``tkey``.

    ``parsed`` is in ascending precedence, so the last file containing the key is
    the one whose value wins the merge — that's the file to attribute it to,
    instead of unconditionally the last file *read* (the gh #55 mislabel).
    """
    winner: Path | None = None
    for p, data in parsed:
        if _get_dotted(data, tkey) is not None:
            winner = p
    return f"toml ({winner.name})" if winner is not None else "toml"


# ── Field casters ────────────────────────────────────────────────────


def _env_list_csv(value: str | None, default: list[str] | None = None) -> list[str]:
    """Parse a comma-separated env string into ``list[str]``; empty → ``[]``."""
    if value is None or value == "":
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_float(value: str | None) -> float:
    return float(value) if value not in (None, "") else 0.0


# ── HermesConfig ─────────────────────────────────────────────────────


@dataclass
class HermesConfig(HostConfig):
    """Full ``langstage-hermes`` runtime config.

    Inherits the shared ``DEEPAGENT_*`` core (agent_spec, workspace_root,
    host, port, debug, title) from ``HostConfig``. Adds every Hermes-specific
    knob from SPEC §2, env-bound under ``DEEPAGENT_HERMES_*``.
    """

    # ── [model] ──
    model_default: str = "anthropic:claude-sonnet-4-6"
    model_provider: str = "auto"
    model_context_length: int | None = None
    model_max_tokens: int | None = None
    model_aux: str = "anthropic:claude-haiku-4-5-20251001"

    # ── [agent] ──
    agent_api_max_retries: int = 3
    agent_max_iterations: int = 90
    agent_delegation_max_iterations: int = 50
    agent_task_completion_guidance: bool = True
    agent_environment_probe: bool = True
    agent_tool_use_enforcement: str = "auto"
    agent_disabled_toolsets: list[str] = field(default_factory=list)

    # ── [memory] ──
    memory_enabled: bool = True
    memory_user_profile_enabled: bool = True
    memory_nudge_interval: int = 10
    memory_char_limit: int = 2200
    memory_user_char_limit: int = 1375
    memory_provider: str = ""

    # ── [skills] ──
    skills_creation_nudge_interval: int = 10
    skills_external_dirs: list[str] = field(default_factory=list)
    skills_disabled: list[str] = field(default_factory=list)
    skills_platform_disabled: dict[str, list[str]] = field(default_factory=dict)

    # ── [compression] ──
    compression_enabled: bool = True
    compression_threshold: float = 0.50
    compression_target_ratio: float = 0.20
    compression_protect_first_n: int = 3
    compression_protect_last_n: int = 20
    compression_abort_on_summary_failure: bool = False

    # ── [delegation] ──
    delegation_max_concurrent_children: int = 4
    delegation_max_spawn_depth: int = 3
    delegation_max_iterations: int = 50

    # ── [curator] ──
    curator_enabled: bool = True
    curator_interval_hours: int = 168
    curator_min_idle_hours: int = 2
    curator_stale_after_days: int = 30
    curator_archive_after_days: int = 90
    curator_prune_builtins: bool = True

    # ── [cron] ──
    cron_tick_seconds: int = 60

    # ── [plugins] ──
    plugins_enabled: list[str] = field(default_factory=list)
    plugins_disabled: list[str] = field(default_factory=list)

    # ── env mapping (additive: base ``DEEPAGENT_*`` core still resolves) ──
    _ENV: ClassVar[dict[str, tuple[str, Callable[[str], Any]]]] = {
        # [model]
        "model_default": ("DEEPAGENT_HERMES_MODEL_DEFAULT", str),
        "model_provider": ("DEEPAGENT_HERMES_MODEL_PROVIDER", str),
        "model_context_length": ("DEEPAGENT_HERMES_MODEL_CONTEXT_LENGTH", int),
        "model_max_tokens": ("DEEPAGENT_HERMES_MODEL_MAX_TOKENS", int),
        "model_aux": ("DEEPAGENT_HERMES_MODEL_AUX", str),
        # [agent]
        "agent_api_max_retries": ("DEEPAGENT_HERMES_AGENT_API_MAX_RETRIES", int),
        "agent_max_iterations": ("DEEPAGENT_HERMES_AGENT_MAX_ITERATIONS", int),
        "agent_delegation_max_iterations": (
            "DEEPAGENT_HERMES_AGENT_DELEGATION_MAX_ITERATIONS",
            int,
        ),
        "agent_task_completion_guidance": (
            "DEEPAGENT_HERMES_AGENT_TASK_COMPLETION_GUIDANCE",
            _env_bool,
        ),
        "agent_environment_probe": ("DEEPAGENT_HERMES_AGENT_ENVIRONMENT_PROBE", _env_bool),
        "agent_tool_use_enforcement": (
            "DEEPAGENT_HERMES_AGENT_TOOL_USE_ENFORCEMENT",
            str,
        ),
        "agent_disabled_toolsets": (
            "DEEPAGENT_HERMES_AGENT_DISABLED_TOOLSETS",
            _env_list_csv,
        ),
        # [memory]
        "memory_enabled": ("DEEPAGENT_HERMES_MEMORY_ENABLED", _env_bool),
        "memory_user_profile_enabled": (
            "DEEPAGENT_HERMES_MEMORY_USER_PROFILE_ENABLED",
            _env_bool,
        ),
        "memory_nudge_interval": ("DEEPAGENT_HERMES_MEMORY_NUDGE_INTERVAL", int),
        "memory_char_limit": ("DEEPAGENT_HERMES_MEMORY_CHAR_LIMIT", int),
        "memory_user_char_limit": ("DEEPAGENT_HERMES_MEMORY_USER_CHAR_LIMIT", int),
        "memory_provider": ("DEEPAGENT_HERMES_MEMORY_PROVIDER", str),
        # [skills]
        "skills_creation_nudge_interval": (
            "DEEPAGENT_HERMES_SKILLS_CREATION_NUDGE_INTERVAL",
            int,
        ),
        "skills_external_dirs": ("DEEPAGENT_HERMES_SKILLS_EXTERNAL_DIRS", _env_list_csv),
        "skills_disabled": ("DEEPAGENT_HERMES_SKILLS_DISABLED", _env_list_csv),
        # [compression]
        "compression_enabled": ("DEEPAGENT_HERMES_COMPRESSION_ENABLED", _env_bool),
        "compression_threshold": ("DEEPAGENT_HERMES_COMPRESSION_THRESHOLD", _env_float),
        "compression_target_ratio": (
            "DEEPAGENT_HERMES_COMPRESSION_TARGET_RATIO",
            _env_float,
        ),
        "compression_protect_first_n": (
            "DEEPAGENT_HERMES_COMPRESSION_PROTECT_FIRST_N",
            int,
        ),
        "compression_protect_last_n": (
            "DEEPAGENT_HERMES_COMPRESSION_PROTECT_LAST_N",
            int,
        ),
        "compression_abort_on_summary_failure": (
            "DEEPAGENT_HERMES_COMPRESSION_ABORT_ON_SUMMARY_FAILURE",
            _env_bool,
        ),
        # [delegation]
        "delegation_max_concurrent_children": (
            "DEEPAGENT_HERMES_DELEGATION_MAX_CONCURRENT_CHILDREN",
            int,
        ),
        "delegation_max_spawn_depth": (
            "DEEPAGENT_HERMES_DELEGATION_MAX_SPAWN_DEPTH",
            int,
        ),
        "delegation_max_iterations": (
            "DEEPAGENT_HERMES_DELEGATION_MAX_ITERATIONS",
            int,
        ),
        # [curator]
        "curator_enabled": ("DEEPAGENT_HERMES_CURATOR_ENABLED", _env_bool),
        "curator_interval_hours": ("DEEPAGENT_HERMES_CURATOR_INTERVAL_HOURS", int),
        "curator_min_idle_hours": ("DEEPAGENT_HERMES_CURATOR_MIN_IDLE_HOURS", int),
        "curator_stale_after_days": ("DEEPAGENT_HERMES_CURATOR_STALE_AFTER_DAYS", int),
        "curator_archive_after_days": (
            "DEEPAGENT_HERMES_CURATOR_ARCHIVE_AFTER_DAYS",
            int,
        ),
        "curator_prune_builtins": (
            "DEEPAGENT_HERMES_CURATOR_PRUNE_BUILTINS",
            _env_bool,
        ),
        # [cron]
        "cron_tick_seconds": ("DEEPAGENT_HERMES_CRON_TICK_SECONDS", int),
        # [plugins]
        "plugins_enabled": ("DEEPAGENT_HERMES_PLUGINS_ENABLED", _env_list_csv),
        "plugins_disabled": ("DEEPAGENT_HERMES_PLUGINS_DISABLED", _env_list_csv),
    }

    # ── TOML key map (dotted path within langstage-hermes.toml) ──
    _TOML: ClassVar[dict[str, str]] = {
        # [model]
        "model_default": "model.default",
        "model_provider": "model.provider",
        "model_context_length": "model.context_length",
        "model_max_tokens": "model.max_tokens",
        "model_aux": "model.aux_model",
        # [agent]
        "agent_api_max_retries": "agent.api_max_retries",
        "agent_max_iterations": "agent.max_iterations",
        "agent_delegation_max_iterations": "agent.delegation_max_iterations",
        "agent_task_completion_guidance": "agent.task_completion_guidance",
        "agent_environment_probe": "agent.environment_probe",
        "agent_tool_use_enforcement": "agent.tool_use_enforcement",
        "agent_disabled_toolsets": "agent.disabled_toolsets",
        # [memory]
        "memory_enabled": "memory.memory_enabled",
        "memory_user_profile_enabled": "memory.user_profile_enabled",
        "memory_nudge_interval": "memory.nudge_interval",
        "memory_char_limit": "memory.memory_char_limit",
        "memory_user_char_limit": "memory.user_char_limit",
        "memory_provider": "memory.provider",
        # [skills]
        "skills_creation_nudge_interval": "skills.creation_nudge_interval",
        "skills_external_dirs": "skills.external_dirs",
        "skills_disabled": "skills.disabled",
        "skills_platform_disabled": "skills.platform_disabled",
        # [compression]
        "compression_enabled": "compression.enabled",
        "compression_threshold": "compression.threshold",
        "compression_target_ratio": "compression.target_ratio",
        "compression_protect_first_n": "compression.protect_first_n",
        "compression_protect_last_n": "compression.protect_last_n",
        "compression_abort_on_summary_failure": "compression.abort_on_summary_failure",
        # [delegation]
        "delegation_max_concurrent_children": "delegation.max_concurrent_children",
        "delegation_max_spawn_depth": "delegation.max_spawn_depth",
        "delegation_max_iterations": "delegation.max_iterations",
        # [curator]
        "curator_enabled": "curator.enabled",
        "curator_interval_hours": "curator.interval_hours",
        "curator_min_idle_hours": "curator.min_idle_hours",
        "curator_stale_after_days": "curator.stale_after_days",
        "curator_archive_after_days": "curator.archive_after_days",
        "curator_prune_builtins": "curator.prune_builtins",
        # [cron]
        "cron_tick_seconds": "cron.tick_seconds",
        # [plugins]
        "plugins_enabled": "plugins.enabled",
        "plugins_disabled": "plugins.disabled",
    }

    # ── convenience property ──

    @property
    def hermes_home(self) -> Path:
        """Resolved HERMES_HOME path — same precedence as the module helper."""
        return hermes_home()

    # ── resolution ──

    @classmethod
    def resolve(  # type: ignore[override]
        cls,
        *,
        overrides: dict[str, Any] | None = None,
        toml_start: Path | None = None,
        env: dict[str, str] | None = None,
        use_toml: bool = True,
    ) -> HermesConfig:
        """Resolve through ``defaults < TOML < env < overrides``.

        Layers (lowest precedence first):
          1. Dataclass defaults (SPEC §2 verbatim)
          2. ``deepagents.toml`` (cross-host shared keys only — base behavior)
          3. ``$HERMES_HOME/config.toml`` (default ``~/.langstage-hermes/config.toml``;
             moves with a custom ``HERMES_HOME``) then ``./langstage-hermes.toml``
          4. ``LANGSTAGE_*`` (core) and ``LANGSTAGE_HERMES_*`` (this class) env
             vars — the legacy ``DEEPAGENT_*`` spellings still resolve as a
             fallback (canonical wins) and warn.
          5. Explicit ``overrides`` keyword

        ``use_toml=False`` skips both TOML layers.
        """
        overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        env = os.environ if env is None else env

        # Layer 2: cross-host TOML (deepagents.toml) — same as base.
        base_toml_data, base_toml_paths = load_toml_config(toml_start) if use_toml else ({}, [])
        # Layer 3: hermes-specific TOML.
        hermes_toml_data, hermes_toml_paths = load_hermes_toml_config(toml_start) if use_toml else ({}, [])
        # Per-file parses so a field's source names the file it actually came from,
        # not just the last file read across a layered global+project stack (gh #55).
        base_toml_parsed = _parse_toml_files(base_toml_paths)
        hermes_toml_parsed = _parse_toml_files(hermes_toml_paths)

        env_map = cls._env_map()
        toml_map = cls._toml_map()

        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for f in fields(cls):
            name = f.name
            # default
            if f.default is not MISSING:
                val: Any = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                val = f.default_factory()  # type: ignore[misc]
            else:
                val = None
            src = "default"

            tkey = toml_map.get(name)
            if tkey is not None:
                # Look up the field in BOTH TOML stacks — hermes TOML wins
                # because we check it second (overwrites).
                tv = _get_dotted(base_toml_data, tkey)
                if tv is not None:
                    val = _coerce(f, tv)
                    src = _toml_source_label(base_toml_parsed, tkey)
                tv2 = _get_dotted(hermes_toml_data, tkey)
                if tv2 is not None:
                    val = _coerce(f, tv2)
                    src = _toml_source_label(hermes_toml_parsed, tkey)

            if name in env_map:
                var, caster = env_map[name]
                # Normalize to (canonical LANGSTAGE_*, legacy DEEPAGENT_*) and
                # check canonical first, falling back to legacy with a warning —
                # identical to the base HostConfig.resolve(). The override used to
                # read only the raw declared name, so the canonical names this
                # class advertises (and describe() prints) were silently dead
                # while only the legacy spelling worked (gh #24).
                canonical, legacy = _env_pair(var)
                ev = env.get(canonical)
                used = canonical
                if ev is None or ev == "":
                    ev = env.get(legacy)
                    used = legacy
                    if ev not in (None, "") and legacy != canonical:
                        _warn_legacy_env(legacy, canonical)
                if ev is not None and ev != "":
                    val = caster(ev)
                    src = f"env:{used}"

            if name in overrides:
                val = overrides[name]
                src = "override"

            values[name] = val
            sources[name] = src

        obj = cls(**values)
        obj._sources = sources  # type: ignore[attr-defined]
        obj._toml_paths = base_toml_paths + hermes_toml_paths  # type: ignore[attr-defined]
        return obj

    def describe(self) -> str:
        """Like the base dump, but the 'no TOML found' line lists the search
        order Hermes actually uses — leading with the documented
        ``langstage-hermes.toml`` (the base message only named the cross-host
        ``langstage.toml``/``deepagents.toml``). (gh #-dogfood)

        The global path is shown as the *resolved* ``$HERMES_HOME/config.toml`` —
        NOT a hardcoded ``~/.langstage-hermes/config.toml`` — because the global
        config lives under ``HERMES_HOME`` and moves with it. Printing the fixed
        default misdirected anyone who set a custom ``HERMES_HOME`` to the wrong
        path (gh #57).
        """
        text = super().describe()
        global_toml = _hermes_global_toml_path()
        return text.replace(
            "TOML: no langstage.toml (or legacy deepagents.toml) found",
            f"TOML: no config found (looked for ./langstage-hermes.toml, {global_toml}, ./langstage.toml, ./deepagents.toml)",
        )


__all__ = [
    "HERMES_GLOBAL_TOML",
    "HERMES_PROJECT_TOML",
    "HermesConfig",
    "hermes_home",
    "load_hermes_toml_config",
]
