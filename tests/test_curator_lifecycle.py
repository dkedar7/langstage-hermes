"""Tests for the curator skill lifecycle pass.

We construct a tiny fake ``SkillLibrary`` over a temp directory, scatter three
SKILL.md files with varying mtimes, and verify ``mark_stale_and_archive``:

1. archives skills whose ``state_meta`` "last used" timestamp is older than
   ``archive_days``;
2. flips ``metadata.hermes.lifecycle`` to ``"stale"`` for skills older than
   ``stale_days`` but younger than ``archive_days``;
3. leaves pinned skills entirely alone;
4. leaves fresh / recently-used skills untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from langstage_hermes.curator import (
    CuratorMiddleware,
    mark_stale_and_archive,
)

# ── tiny fake library ────────────────────────────────────────────────


@dataclass
class FakeSkill:
    """In-memory analog of the eventual ``Skill`` dataclass."""

    name: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeLibrary:
    """Disk-backed test double for ``SkillLibrary``.

    Each skill is a ``<root>/<name>/SKILL.md`` file. ``write`` overwrites the
    markdown body; ``delete`` archives by moving the directory to
    ``<root>/_archived/<name>/`` (recoverable, per SPEC §10).
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # ── helpers ──

    def _skill_dir(self, name: str) -> Path:
        return self.root / name

    def _archive_dir(self) -> Path:
        return self.root / "_archived"

    # ── public API used by the curator ──

    def list(self) -> list[FakeSkill]:
        out: list[FakeSkill] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            out.append(FakeSkill(name=child.name, path=skill_md, metadata=self._read_meta(skill_md)))
        return out

    def get(self, name: str) -> FakeSkill:
        skill_md = self._skill_dir(name) / "SKILL.md"
        return FakeSkill(name=name, path=skill_md, metadata=self._read_meta(skill_md))

    def write(self, skill: FakeSkill) -> None:
        skill.path.parent.mkdir(parents=True, exist_ok=True)
        skill.path.write_text(self._serialize(skill), encoding="utf-8")

    def delete(self, name: str) -> None:
        src = self._skill_dir(name)
        dst_root = self._archive_dir()
        dst_root.mkdir(parents=True, exist_ok=True)
        dst = dst_root / name
        if dst.exists():
            # Suffix collisions to avoid races on tight timestamps.
            i = 2
            while (dst_root / f"{name}-{i}").exists():
                i += 1
            dst = dst_root / f"{name}-{i}"
        src.rename(dst)

    # ── frontmatter (we keep it dead simple) ──

    @staticmethod
    def _read_meta(path: Path) -> dict[str, Any]:
        import frontmatter

        post = frontmatter.load(str(path))
        return dict(post.metadata)

    @staticmethod
    def _serialize(skill: FakeSkill) -> str:
        import frontmatter

        post = frontmatter.Post("(test body)", **skill.metadata)
        return frontmatter.dumps(post)


# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def library(tmp_path: Path) -> FakeLibrary:
    return FakeLibrary(tmp_path / "skills")


def _write_skill(library: FakeLibrary, name: str, **meta: Any) -> Path:
    import frontmatter

    skill_dir = library.root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    post = frontmatter.Post(f"# {name}\n", name=name, description=f"{name} test fixture", **meta)
    skill_md.write_text(frontmatter.dumps(post), encoding="utf-8")
    return skill_md


# ── tests ────────────────────────────────────────────────────────────


def test_lifecycle_archives_long_unused_and_marks_stale(library: FakeLibrary):
    """Two stale skills get the right transition; the fresh one is untouched."""
    _write_skill(library, "fresh-skill")
    _write_skill(library, "stale-skill")
    _write_skill(library, "ancient-skill")

    now = time.time()
    DAY = 86400
    state_meta = {
        "fresh-skill": now - 1 * DAY,
        "stale-skill": now - 45 * DAY,  # > 30 days → stale
        "ancient-skill": now - 120 * DAY,  # > 90 days → archive
    }

    def _meta_get(name: str) -> float | None:
        return state_meta.get(name)

    result = mark_stale_and_archive(
        library,
        stale_days=30,
        archive_days=90,
        state_meta_get=_meta_get,
        now=now,
    )

    assert result["archived"] == ["ancient-skill"]
    assert result["marked_stale"] == ["stale-skill"]
    assert result["skipped_pinned"] == []

    # On-disk effects.
    assert not (library.root / "ancient-skill").exists(), "should be archived"
    assert (library.root / "_archived" / "ancient-skill" / "SKILL.md").exists()

    fresh = library.get("fresh-skill")
    assert fresh.metadata.get("hermes", {}).get("lifecycle") != "stale"

    stale = library.get("stale-skill")
    assert stale.metadata.get("hermes", {}).get("lifecycle") == "stale"


def test_pinned_skills_are_immune(library: FakeLibrary):
    """A pinned skill is left alone even when it would otherwise be archived."""
    _write_skill(library, "ancient-pinned", hermes={"pinned": True})
    _write_skill(library, "ancient-unpinned")

    now = time.time()
    state_meta = {
        "ancient-pinned": now - 365 * 86400,
        "ancient-unpinned": now - 365 * 86400,
    }

    result = mark_stale_and_archive(
        library,
        stale_days=30,
        archive_days=90,
        state_meta_get=state_meta.get,
        now=now,
    )

    assert "ancient-pinned" in result["skipped_pinned"]
    assert result["archived"] == ["ancient-unpinned"]
    # Pinned dir survives on disk.
    assert (library.root / "ancient-pinned" / "SKILL.md").exists()


def test_missing_state_meta_falls_back_to_mtime(library: FakeLibrary):
    """When state_meta returns None for a skill, we use the file mtime."""
    path = _write_skill(library, "no-record")
    # Pretend the file was last touched 120 days ago.
    long_ago = time.time() - 120 * 86400
    Path(path).touch()
    import os

    os.utime(path, (long_ago, long_ago))

    result = mark_stale_and_archive(
        library,
        stale_days=30,
        archive_days=90,
        state_meta_get=lambda _n: None,
    )
    assert result["archived"] == ["no-record"]


def test_already_marked_stale_is_idempotent(library: FakeLibrary):
    """A skill already at ``lifecycle == "stale"`` doesn't appear in
    ``marked_stale`` on the next pass — no re-write churn."""
    _write_skill(library, "stale-skill", hermes={"lifecycle": "stale"})
    now = time.time()
    result = mark_stale_and_archive(
        library,
        stale_days=30,
        archive_days=90,
        state_meta_get=lambda _n: now - 45 * 86400,
        now=now,
    )
    assert result["marked_stale"] == []
    assert result["archived"] == []


def test_curator_middleware_respects_interval_gate(library: FakeLibrary):
    """``CuratorMiddleware.before_agent`` is a no-op when the interval hasn't elapsed."""
    # In-memory store stand-in: tracks the curator state dict via put/get.
    store = _FakeStore()
    # Seed last_run_at to 1 second ago so the interval gate is closed.
    store.put(
        ("curator_state",),
        "state",
        {
            "last_run_at": time.time() - 1,
            "last_user_activity": time.time() - 100_000,  # idle gate would be open
        },
    )

    mw = CuratorMiddleware(library, store, interval_hours=168, min_idle_hours=2)
    mw.before_agent(state={"messages": []})

    # State unchanged on the curator side; nothing archived.
    after = store.get(("curator_state",), "state").value  # type: ignore[union-attr]
    # last_run_at preserved (not overwritten with "now").
    assert abs(after["last_run_at"] - (time.time() - 1)) < 5


def test_curator_middleware_first_run_seeds_state(library: FakeLibrary):
    """First-run path seeds ``last_run_at`` but does NOT execute the pass."""
    store = _FakeStore()
    mw = CuratorMiddleware(library, store)
    mw.before_agent(state={"messages": []})

    item = store.get(("curator_state",), "state")
    assert item is not None
    assert item.value["last_run_at"] > 0


# ── helpers ──────────────────────────────────────────────────────────


class _FakeStoreItem:
    def __init__(self, value: Any) -> None:
        self.value = value


class _FakeStore:
    """Minimal ``BaseStore``-shaped fake — get / put on (namespace, key)."""

    def __init__(self) -> None:
        self._db: dict[tuple[tuple[str, ...], str], Any] = {}

    def get(self, namespace: tuple[str, ...], key: str) -> _FakeStoreItem | None:
        v = self._db.get((namespace, key))
        if v is None:
            return None
        return _FakeStoreItem(v)

    def put(self, namespace: tuple[str, ...], key: str, value: Any) -> None:
        self._db[(namespace, key)] = value
