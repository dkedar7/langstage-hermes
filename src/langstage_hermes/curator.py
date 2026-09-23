"""Curator middleware — background skill-library maintenance (SPEC §9.4).

The curator runs **inactivity-triggered**: when the user has been idle for at
least ``min_idle_hours`` AND the last curator pass was longer than
``interval_hours`` ago, ``CuratorMiddleware.before_agent`` invokes a curator
subagent that consolidates the skill library (merges narrow siblings into
class-level umbrellas, marks stale skills, archives long-dead ones).

Two layers run on every tick:

1. **Pure lifecycle pass** (no LLM) — ``mark_stale_and_archive`` walks the
   library, marks skills unused for ``stale_days`` as ``"stale"`` in their
   frontmatter, and moves skills unused for ``archive_days`` into the
   ``_archived/`` directory. Pinned skills are exempt.
2. **LLM consolidation pass** — the curator subagent reads the skill catalog
   and proposes umbrella mergers via ``skill_manage`` calls. Its run is logged
   to ``<HERMES_HOME>/logs/curator/{YYYYMMDD-HHMMSS}/{run.json, REPORT.md}``.

State is persisted in the ``store`` under namespace ``curator_state`` (key
``state``) so the schedule survives process restarts.

Skill "last used" timestamps live in the ``state_meta`` namespace under keys
``skill_last_used:<name>`` (value ``{"ts": <unix float>}``). The agent's
``skill_view`` and ``skill_manage`` tools stamp them via
:func:`record_skill_use` (gh #141); :func:`read_skill_last_used` reads them. The
lifecycle function tolerates missing entries (a skill never used through the
agent falls back to its SKILL.md mtime).

Bundled skills (shipped inside the installed package) are never stale-marked or
archived: the package is shared by every ``HERMES_HOME``, so a per-home pass must
not mutate it (gh #154). Hide an unwanted bundled skill with ``skills.disabled``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from langstage_hermes.reflection import load_prompt
from langstage_hermes.skills.library import is_pinned

logger = logging.getLogger(__name__)


# ── store helpers (namespace constants) ──────────────────────────────


_CURATOR_NS = ("curator_state",)
_CURATOR_KEY = "state"
_STATE_META_NS = ("state_meta",)


def _now() -> float:
    return time.time()


def _hermes_home() -> Path:
    """Resolve ``<HERMES_HOME>`` with the same precedence Hermes uses.

    ``DEEPAGENT_HERMES_HOME`` wins, then ``HERMES_HOME``, then the bundled
    default of ``~/.langstage-hermes``. Created on demand.
    """

    from langstage_hermes.config import hermes_home

    home = hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


# ── skill usage stamps (gh #141) ─────────────────────────────────────


def _last_used_key(name: str) -> str:
    return f"skill_last_used:{name}"


def record_skill_use(store: Any, name: str, *, now: float | None = None) -> None:
    """Stamp ``skill_last_used:<name>`` = now. Best-effort: never breaks a tool call."""
    try:
        store.put(_STATE_META_NS, _last_used_key(name), {"ts": now if now is not None else _now()})
    except Exception as exc:
        logger.debug("curator: failed to record use of %r: %s", name, exc)


def read_skill_last_used(store: Any, name: str) -> float | None:
    """Return the ``skill_last_used:<name>`` UNIX timestamp, or ``None`` if never recorded."""
    try:
        item = store.get(_STATE_META_NS, _last_used_key(name))
    except Exception:
        return None
    if item is None:
        return None
    value = getattr(item, "value", item)
    if isinstance(value, dict):
        value = value.get("ts") or value.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    return None


# ── skill lifecycle (pure, no LLM) ───────────────────────────────────


def mark_stale_and_archive(
    library: Any,
    *,
    stale_days: int = 30,
    archive_days: int = 90,
    state_meta_get: Callable[[str], float | None] | None = None,
    now: float | None = None,
) -> dict[str, list[str]]:
    """Walk the library; mark stale skills stale; archive long-dead ones.

    Pinned skills (``metadata.hermes.pinned == true``) are exempt from both
    transitions. The "last used" timestamp comes from ``state_meta_get``
    (typically ``store.get(("state_meta",), f"skill_last_used:{name}")`` →
    UNIX float), with a fallback to the skill file's mtime when the lookup
    returns ``None``.

    Bundled skills (``skill.bundled``) are skipped entirely — they live inside
    the installed package, which a per-home pass must never mutate (gh #154).
    Pins are read with :func:`~langstage_hermes.skills.library.is_pinned`, the
    same accessor ``Skill.pinned`` uses (gh #119).

    Args:
        library: A ``SkillLibrary``-shaped object with ``list()``,
            ``update_frontmatter(name, frontmatter_data, *, audit_action)`` and
            ``delete(name)`` methods. ``list`` must yield objects with
            ``.name``, ``.metadata`` (the full frontmatter dict) and ``.path``
            (``Path``).
        stale_days: Inactivity threshold before ``metadata.hermes.lifecycle``
            flips to ``"stale"``.
        archive_days: Inactivity threshold before the skill is archived via
            ``library.delete``.
        state_meta_get: Callable to read a ``skill_last_used:<name>`` value
            from the store. When ``None``, mtime-only mode is used.
        now: Override "now" (test seam). Defaults to ``time.time()``.

    Returns:
        ``{"marked_stale": [...], "archived": [...], "skipped_pinned": [...]}``.
    """
    now = now if now is not None else _now()
    stale_cutoff = now - stale_days * 86400
    archive_cutoff = now - archive_days * 86400

    marked_stale: list[str] = []
    archived: list[str] = []
    skipped_pinned: list[str] = []

    for skill in library.list():
        name = getattr(skill, "name", None) or skill.get("name")  # type: ignore[union-attr]
        if not name:
            continue

        if getattr(skill, "bundled", False):
            continue

        # ``metadata`` is the full frontmatter; the Hermes extensions live under
        # its nested ``metadata.hermes`` block (SPEC §9), not a top-level ``hermes``.
        frontmatter_data = getattr(skill, "metadata", None) or (skill.get("metadata") if isinstance(skill, dict) else {})
        frontmatter_data = dict(frontmatter_data or {})

        if is_pinned(frontmatter_data):
            skipped_pinned.append(name)
            continue

        last_used = state_meta_get(name) if state_meta_get else None
        if last_used is None:
            # Fall back to file mtime — the next-best signal we have when the
            # `state_meta` table has no entry for this skill (e.g. installed
            # but never used).
            path = getattr(skill, "path", None)
            if isinstance(path, Path) and path.exists():
                last_used = path.stat().st_mtime
            else:
                # No mtime to fall back on; assume "just installed" so we
                # don't archive freshly-created skills on first pass.
                last_used = now

        if last_used <= archive_cutoff:
            try:
                library.delete(name)
                archived.append(name)
            except Exception as exc:
                logger.warning("curator: archive of %r failed: %s", name, exc)
            continue

        if last_used <= stale_cutoff:
            nested = dict(frontmatter_data.get("metadata") or {})
            hermes_meta = dict(nested.get("hermes") or {})
            if hermes_meta.get("lifecycle") == "stale":
                # Already marked — nothing to do.
                continue
            hermes_meta["lifecycle"] = "stale"
            nested["hermes"] = hermes_meta
            frontmatter_data["metadata"] = nested
            try:
                # In-place frontmatter rewrite through the real library API (the
                # old ``library.write(skill)`` matched no real signature, so the
                # TypeError was swallowed and nothing was ever marked — gh #120).
                library.update_frontmatter(name, frontmatter_data, audit_action="curator-stale")
                marked_stale.append(name)
            except Exception as exc:
                logger.warning("curator: stale mark on %r failed: %s", name, exc)

    return {
        "marked_stale": marked_stale,
        "archived": archived,
        "skipped_pinned": skipped_pinned,
    }


# ── curator state persistence ────────────────────────────────────────


def _load_curator_state(store: Any) -> dict[str, Any]:
    """Read curator-state JSON from the store. Returns ``{}`` on miss."""
    try:
        item = store.get(_CURATOR_NS, _CURATOR_KEY)
    except Exception:
        return {}
    if item is None:
        return {}
    # `BaseStore.get` returns an `Item` with a `.value` dict in production; some
    # fakes return the value dict directly. Handle both.
    value = getattr(item, "value", None)
    if value is None and isinstance(item, dict):
        value = item
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _save_curator_state(store: Any, state: dict[str, Any]) -> None:
    """Persist curator-state JSON back to the store."""
    try:
        store.put(_CURATOR_NS, _CURATOR_KEY, state)
    except Exception as exc:
        logger.warning("curator: failed to persist state: %s", exc)


# ── reports ──────────────────────────────────────────────────────────


def _write_run_report(
    *,
    started_at: datetime,
    elapsed_seconds: float,
    lifecycle_result: dict[str, list[str]],
    llm_summary: str | None,
    error: str | None,
) -> Path | None:
    """Persist a per-run report under ``<HERMES_HOME>/logs/curator/``.

    Writes ``run.json`` (machine-readable) and ``REPORT.md`` (human-readable).
    Best-effort: returns the directory path on success, ``None`` on I/O
    failure (caller logs and continues — reporting must never break the run).
    """
    root = _hermes_home() / "logs" / "curator"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("curator: report root mkdir failed: %s", exc)
        return None

    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    run_dir = root / stamp
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{stamp}-{suffix}"

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        logger.debug("curator: run dir mkdir failed: %s", exc)
        return None

    payload = {
        "started_at": started_at.isoformat(),
        "duration_seconds": round(elapsed_seconds, 2),
        "lifecycle": lifecycle_result,
        "llm_summary": llm_summary or "",
        "error": error,
    }
    try:
        (run_dir / "run.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("curator: run.json write failed: %s", exc)

    md_lines: list[str] = [
        f"# Curator run — {payload['started_at']}",
        "",
        f"- duration: {payload['duration_seconds']}s",
        f"- marked stale: {len(lifecycle_result.get('marked_stale', []))}",
        f"- archived: {len(lifecycle_result.get('archived', []))}",
        f"- skipped (pinned): {len(lifecycle_result.get('skipped_pinned', []))}",
        "",
    ]
    if error:
        md_lines.extend([f"> Error: `{error}`", ""])
    if llm_summary:
        md_lines.extend(["## LLM consolidation summary", "", llm_summary, ""])
    try:
        (run_dir / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")
    except OSError as exc:
        logger.debug("curator: REPORT.md write failed: %s", exc)
    return run_dir


# ── curator subagent factory ─────────────────────────────────────────


def build_curator_subagent(
    *,
    library: Any,
    store: Any,
    aux_model: Any,
    tools: list[Any] | None = None,
) -> dict[str, Any]:
    """Return a ``SubAgent`` spec for the curator review fork.

    Mirrors ``build_review_subagent`` in ``reflection.py``: a TypedDict-shaped
    dict the caller passes to ``SubAgentMiddleware(subagents=[...])``. The
    system prompt is ``curator_review.md``.
    """
    del library, store
    spec: dict[str, Any] = {
        "name": "curator",
        "description": (
            "Background skill-library curator. Consolidates narrow skills into "
            "class-level umbrellas, marks stale skills, archives long-dead ones. "
            "Scheduled by CuratorMiddleware on an inactivity-triggered cadence."
        ),
        "system_prompt": load_prompt("curator_review.md"),
        "tools": tools or [],
    }
    if aux_model is not None:
        spec["model"] = aux_model
    return spec


# ── CuratorMiddleware ────────────────────────────────────────────────


class CuratorMiddleware(AgentMiddleware):
    """Schedule + run the curator on an inactivity-triggered cadence.

    Two gates must be open for a run to fire on ``before_agent``:

    1. ``now - last_run_at >= interval_hours * 3600``
    2. ``now - last_user_activity >= min_idle_hours * 3600``

    When both are open, the middleware runs the pure lifecycle pass, then
    invokes the LLM consolidation subagent (if wired), then writes a report
    and persists the new ``last_run_at`` / ``last_user_activity`` state.

    The middleware also updates ``last_user_activity`` on every turn so the
    idle gate stays meaningful.
    """

    state_schema = AgentState

    def __init__(
        self,
        library: Any,
        store: Any,
        *,
        interval_hours: int = 168,
        min_idle_hours: int = 2,
        stale_days: int = 30,
        archive_days: int = 90,
        enabled: bool = True,
        curator_graph: Any | None = None,
    ) -> None:
        super().__init__()
        self.library = library
        self.store = store
        self.interval_hours = interval_hours
        self.min_idle_hours = min_idle_hours
        self.stale_days = stale_days
        self.archive_days = archive_days
        self.enabled = enabled
        self._curator_graph = curator_graph
        self.tools: list[Any] = []

    # ── before_agent: maybe run ──────────────────────────────────

    def before_agent(self, state: Any, runtime: Runtime[Any] | None = None) -> dict[str, Any] | None:
        """If the cadence gates are open, run the curator pass."""
        if not self.enabled:
            return None

        now = _now()
        cstate = _load_curator_state(self.store)
        last_run_at = float(cstate.get("last_run_at") or 0.0)
        last_user_activity = float(cstate.get("last_user_activity") or 0.0)

        interval = self.interval_hours * 3600
        idle = self.min_idle_hours * 3600

        # First-run behaviour: seed last_run_at to now so we don't fire on
        # the very first ever invocation; defer one full interval. Mirrors
        # Hermes's `should_run_now` first-run logic.
        if last_run_at == 0.0:
            cstate.update({"last_run_at": now, "last_user_activity": now})
            _save_curator_state(self.store, cstate)
            return None

        if (now - last_run_at) < interval:
            return None
        if (now - last_user_activity) < idle:
            return None

        self._run_pass(cstate, now)
        return None

    # ── after_agent: stamp last_user_activity ────────────────────

    def after_agent(self, state: Any, runtime: Runtime[Any] | None = None) -> dict[str, Any] | None:
        """Stamp ``last_user_activity`` so the idle gate stays calibrated."""
        if not self.enabled:
            return None
        cstate = _load_curator_state(self.store)
        cstate["last_user_activity"] = _now()
        _save_curator_state(self.store, cstate)
        return None

    # ── implementation ──────────────────────────────────────────

    def _run_pass(self, cstate: dict[str, Any], now: float) -> None:
        """Run lifecycle + LLM consolidation. Persist state and write a report."""
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()
        error: str | None = None
        llm_summary: str | None = None

        try:
            lifecycle_result = mark_stale_and_archive(
                self.library,
                stale_days=self.stale_days,
                archive_days=self.archive_days,
                state_meta_get=lambda name: read_skill_last_used(self.store, name),
                now=now,
            )
        except Exception as exc:
            logger.warning("curator: lifecycle pass failed: %s", exc)
            lifecycle_result = {"marked_stale": [], "archived": [], "skipped_pinned": []}
            error = f"lifecycle: {exc}"

        if self._curator_graph is not None:
            try:
                sub_state = {"messages": [HumanMessage(content=load_prompt("curator_review.md"))]}
                result = self._curator_graph.invoke(sub_state)
                # Extract the LLM final message text for the report.
                msgs = result.get("messages") if isinstance(result, dict) else None
                if msgs:
                    last = msgs[-1]
                    llm_summary = getattr(last, "content", None) or str(last)
            except Exception as exc:
                logger.warning("curator: LLM consolidation failed: %s", exc)
                error = f"{error + '; ' if error else ''}llm: {exc}"

        elapsed = time.perf_counter() - t0
        _write_run_report(
            started_at=started_at,
            elapsed_seconds=elapsed,
            lifecycle_result=lifecycle_result,
            llm_summary=llm_summary,
            error=error,
        )

        cstate.update(
            {
                "last_run_at": now,
                "last_user_activity": now,
                "last_run_duration_seconds": round(elapsed, 2),
                "last_lifecycle_counts": {k: len(v) for k, v in lifecycle_result.items()},
            }
        )
        _save_curator_state(self.store, cstate)


__all__ = [
    "CuratorMiddleware",
    "build_curator_subagent",
    "mark_stale_and_archive",
    "read_skill_last_used",
    "record_skill_use",
]
