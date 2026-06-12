"""Tests for ``langstage_hermes.plugins.loader.HermesPluginLoader``.

Covers SPEC §15: directory discovery (bundled / user / project), entry-point
discovery, allow-list / deny-list, hook validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from langstage_hermes.plugins.context import VALID_HOOKS, PluginContext
from langstage_hermes.plugins.loader import HermesPluginLoader

# ── fixtures ────────────────────────────────────────────────────────


def _write_plugin(parent: Path, name: str, *, register_body: str = "pass") -> Path:
    """Write a minimal directory plugin and return its path.

    ``register_body`` is the body of the ``register(ctx)`` function — defaults
    to a no-op. Tests can pass a snippet that mutates a sentinel attribute
    on ``ctx.registry`` to confirm registration ran.
    """
    plugin_dir = parent / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\nversion: 0.0.1\ndescription: test plugin\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n" + "\n".join("    " + line for line in register_body.splitlines() or ["pass"]) + "\n",
        encoding="utf-8",
    )
    return plugin_dir


# ── tests ──────────────────────────────────────────────────────────


def test_user_plugin_discovered_and_registered(tmp_hermes_home: Path):
    """Plugin in <HERMES_HOME>/plugins/<name>/ is discovered, register() runs."""
    plugins_dir = tmp_hermes_home / "plugins"
    plugins_dir.mkdir(exist_ok=True)
    _write_plugin(
        plugins_dir,
        "fake",
        register_body="ctx.registry['ran'] = True",
    )

    registry: dict = {}
    loader = HermesPluginLoader(registry=registry)
    loaded = loader.discover()

    fake = next((p for p in loaded if p.name == "fake"), None)
    assert fake is not None, f"fake plugin not loaded; got {[p.name for p in loaded]}"
    assert fake.enabled is True
    assert fake.source == "user"
    assert fake.error is None
    assert registry.get("ran") is True


def test_disabled_plugin_skipped(tmp_hermes_home: Path):
    """Deny-list always wins, even when register() would have succeeded."""
    plugins_dir = tmp_hermes_home / "plugins"
    plugins_dir.mkdir(exist_ok=True)
    _write_plugin(
        plugins_dir,
        "blocked",
        register_body="raise RuntimeError('register should not run when disabled')",
    )

    registry: dict = {}
    loader = HermesPluginLoader(registry=registry, disabled=["blocked"])
    loaded = loader.discover()

    blocked = next((p for p in loaded if p.name == "blocked"), None)
    assert blocked is not None
    assert blocked.enabled is False
    assert "disabled" in (blocked.error or "")


def test_allow_list_excludes_unlisted(tmp_hermes_home: Path):
    """When [plugins.enabled] is set, plugins not in it stay disabled."""
    plugins_dir = tmp_hermes_home / "plugins"
    plugins_dir.mkdir(exist_ok=True)
    _write_plugin(plugins_dir, "alpha")
    _write_plugin(plugins_dir, "beta")

    loader = HermesPluginLoader(enabled=["alpha"])
    loaded = {p.name: p for p in loader.discover()}

    assert loaded["alpha"].enabled is True
    assert loaded["beta"].enabled is False
    assert "not listed" in (loaded["beta"].error or "")


def test_register_hook_validates_name():
    """Unknown hook names raise ValueError listing the valid set."""
    ctx = PluginContext(registry={}, memory_registry={}, slash_commands={}, hooks={})
    with pytest.raises(ValueError):
        ctx.register_hook("not_a_real_hook", lambda: None)
    # known names succeed
    ctx.register_hook("pre_tool_call", lambda *a, **kw: None)


def test_valid_hooks_has_17_entries():
    """SPEC §15.3 enumerates exactly 17 lifecycle hooks."""
    assert len(VALID_HOOKS) == 17


def test_project_plugins_off_by_default(tmp_path: Path, monkeypatch):
    """./.langstage-hermes/plugins/ is ignored unless the env var opts in."""
    project = tmp_path / "proj" / ".langstage-hermes" / "plugins"
    project.mkdir(parents=True)
    _write_plugin(project, "projonly")
    monkeypatch.delenv("DEEPAGENT_HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

    loader = HermesPluginLoader(project_dir=project)
    loaded = loader.discover()
    # project source not scanned, so projonly should NOT appear.
    assert all(p.name != "projonly" for p in loaded)


def test_project_plugins_on_when_opted_in(tmp_path: Path, monkeypatch):
    """Setting DEEPAGENT_HERMES_ENABLE_PROJECT_PLUGINS=1 turns project discovery on."""
    project = tmp_path / "proj" / ".langstage-hermes" / "plugins"
    project.mkdir(parents=True)
    _write_plugin(project, "projonly")
    monkeypatch.setenv("DEEPAGENT_HERMES_ENABLE_PROJECT_PLUGINS", "1")

    loader = HermesPluginLoader(project_dir=project)
    loaded = {p.name: p for p in loader.discover()}
    assert "projonly" in loaded
    assert loaded["projonly"].source == "project"


def test_plugin_with_failing_register_records_error(tmp_hermes_home: Path):
    """A plugin whose register() raises is captured (not propagated)."""
    plugins_dir = tmp_hermes_home / "plugins"
    plugins_dir.mkdir(exist_ok=True)
    _write_plugin(
        plugins_dir,
        "broken",
        register_body="raise RuntimeError('intentional')",
    )

    loader = HermesPluginLoader()
    loaded = {p.name: p for p in loader.discover()}
    assert loaded["broken"].enabled is False
    assert "intentional" in (loaded["broken"].error or "")


def test_discovery_order_user_before_bundled(tmp_hermes_home: Path):
    """The first three sources are scanned in order: bundled → user → project."""
    plugins_dir = tmp_hermes_home / "plugins"
    plugins_dir.mkdir(exist_ok=True)
    _write_plugin(plugins_dir, "alphabetical")

    loader = HermesPluginLoader()
    loaded = loader.discover()
    sources = [p.source for p in loaded]
    # bundled (if any) appears before user; verify by source string ordering.
    # We can't assume bundled plugins exist in the test repo, but if they do,
    # they precede user plugins.
    bundled_indices = [i for i, s in enumerate(sources) if s == "bundled"]
    user_indices = [i for i, s in enumerate(sources) if s == "user"]
    if bundled_indices and user_indices:
        assert max(bundled_indices) < min(user_indices)
