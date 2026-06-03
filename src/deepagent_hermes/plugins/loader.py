"""``HermesPluginLoader`` — 4-source discovery (SPEC §15).

Discovery order, lowest-to-highest precedence (later wins on name collision):

  1. **bundled**     — ``<package>/plugins/builtin/<name>/``
  2. **user**        — ``~/.deepagent-hermes/plugins/<name>/``
  3. **project**     — ``./.deepagent-hermes/plugins/<name>/``
                        (opt-in: ``DEEPAGENT_HERMES_ENABLE_PROJECT_PLUGINS=1``)
  4. **entry_point** — ``importlib.metadata.entry_points(group="deepagent_hermes.plugins")``

Each directory plugin needs:

    plugin.yaml     ← manifest (``name``, ``version``, ``description``, ...)
    __init__.py     ← exports ``register(ctx: PluginContext) -> None``

Entry-point plugins skip ``plugin.yaml`` (the manifest comes from the entry
point name + package metadata). The loaded module must still export
``register(ctx)``.

Allow-list / deny-list (``[plugins.enabled]`` / ``[plugins.disabled]`` in
config) are applied at load time — **deny wins** so a malicious or broken
plugin can be force-disabled even if it's listed as enabled elsewhere.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

from deepagent_hermes.plugins.context import LoadedPlugin, PluginContext

logger = logging.getLogger(__name__)

ENTRY_POINTS_GROUP = "deepagent_hermes.plugins"


class HermesPluginLoader:
    """Discover + load plugins from the 4 sources, applying enable/disable lists.

    Construct with the registries the plugins should mutate (tools, memory,
    slash commands, hooks). Call :meth:`discover` to scan all four sources;
    returns a list of :class:`LoadedPlugin` records (one per plugin found,
    even those that failed to load — ``error`` carries the reason).

    The ``enabled`` / ``disabled`` args mirror ``[plugins.enabled]`` /
    ``[plugins.disabled]`` from ``deepagent-hermes.toml``. ``enabled=None``
    (default) means "no allow-list — load everything not denied".
    """

    def __init__(
        self,
        *,
        registry: Any = None,
        memory_registry: Any = None,
        slash_commands: Any = None,
        hooks: Any = None,
        enabled: list[str] | None = None,
        disabled: list[str] | None = None,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
    ) -> None:
        self.registry = registry if registry is not None else {}
        self.memory_registry = memory_registry if memory_registry is not None else {}
        self.slash_commands = slash_commands if slash_commands is not None else {}
        self.hooks = hooks if hooks is not None else {}
        self.enabled = set(enabled) if enabled is not None else None
        self.disabled = set(disabled or [])
        self._user_dir_override = user_dir
        self._project_dir_override = project_dir
        self._loaded: list[LoadedPlugin] = []

    # ── public API ──

    def discover(self) -> list[LoadedPlugin]:
        """Scan all four sources, apply enable/disable, run each ``register()``.

        Returns the list of plugin records in deterministic order (bundled
        → user → project → entry_point). Plugins that fail to import or
        whose ``register()`` raises are still returned with ``error`` set
        and ``enabled=False`` so callers can surface the diagnostic.
        """
        self._loaded = []

        # Source 1: bundled.
        for plugin in self._scan_dir(self._bundled_dir(), source="bundled"):
            self._maybe_register(plugin)

        # Source 2: user.
        for plugin in self._scan_dir(self._user_dir(), source="user"):
            self._maybe_register(plugin)

        # Source 3: project (opt-in).
        if self._project_plugins_enabled():
            for plugin in self._scan_dir(self._project_dir(), source="project"):
                self._maybe_register(plugin)
        else:
            logger.debug("Project plugin discovery disabled (set DEEPAGENT_HERMES_ENABLE_PROJECT_PLUGINS=1 to enable)")

        # Source 4: pip entry points.
        for plugin in self._scan_entry_points():
            self._maybe_register(plugin)

        return self._loaded

    @property
    def loaded(self) -> list[LoadedPlugin]:
        """Plugin records from the most recent :meth:`discover` call."""
        return list(self._loaded)

    # ── source resolution ──

    @staticmethod
    def _bundled_dir() -> Path:
        """Return ``<package>/plugins/builtin``."""
        return Path(__file__).parent / "builtin"

    def _user_dir(self) -> Path:
        """Return ``<HERMES_HOME>/plugins`` (or the test override)."""
        if self._user_dir_override is not None:
            return self._user_dir_override
        from deepagent_hermes.config import hermes_home

        return hermes_home() / "plugins"

    def _project_dir(self) -> Path:
        """Return ``./.deepagent-hermes/plugins`` (or the test override)."""
        if self._project_dir_override is not None:
            return self._project_dir_override
        return Path.cwd() / ".deepagent-hermes" / "plugins"

    @staticmethod
    def _project_plugins_enabled() -> bool:
        """``True`` iff ``DEEPAGENT_HERMES_ENABLE_PROJECT_PLUGINS`` is truthy."""
        val = os.getenv("DEEPAGENT_HERMES_ENABLE_PROJECT_PLUGINS", "")
        return val.strip().lower() in {"1", "true", "yes", "on"}

    # ── scanning ──

    def _scan_dir(self, base: Path, *, source: str) -> list[LoadedPlugin]:
        """Find every ``<base>/<name>/__init__.py`` and build a plugin record per dir."""
        results: list[LoadedPlugin] = []
        if not base.is_dir():
            return results
        for child in sorted(base.iterdir()):
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            init_file = child / "__init__.py"
            if not init_file.is_file():
                logger.debug("Skipping %s (no __init__.py)", child)
                continue
            manifest = self._read_manifest(child / "plugin.yaml")
            results.append(
                LoadedPlugin(
                    name=manifest.get("name") or child.name,
                    version=str(manifest.get("version") or ""),
                    description=str(manifest.get("description") or ""),
                    source=source,  # type: ignore[arg-type]
                    path=child,
                )
            )
        return results

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        """Parse ``plugin.yaml`` if present; return ``{}`` on absence or parse error."""
        if not path.is_file():
            return {}
        try:
            import yaml

            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                logger.warning("Plugin manifest %s did not parse to a mapping", path)
                return {}
            return data
        except ImportError:  # pragma: no cover - pyyaml is pinned
            logger.warning("pyyaml not installed; skipping manifest %s", path)
            return {}
        except Exception as e:
            logger.warning("Failed to read plugin manifest %s: %s", path, e)
            return {}

    def _scan_entry_points(self) -> list[LoadedPlugin]:
        """Discover pip-installed plugins via ``importlib.metadata`` entry points."""
        results: list[LoadedPlugin] = []
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.10+: ``.select`` is the right API; older returns a dict.
            if hasattr(eps, "select"):
                selected = list(eps.select(group=ENTRY_POINTS_GROUP))
            else:  # pragma: no cover - py<3.10
                selected = list(eps.get(ENTRY_POINTS_GROUP, []))  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover - importlib.metadata always present
            logger.warning("Entry-point discovery failed: %s", e)
            return results

        for ep in selected:
            try:
                module = ep.load()
            except Exception as e:
                logger.warning("Failed to load entry-point plugin %s: %s", ep.name, e)
                results.append(
                    LoadedPlugin(
                        name=ep.name,
                        source="entry_point",
                        error=f"import failed: {e}",
                        enabled=False,
                    )
                )
                continue
            register_fn = getattr(module, "register", None)
            results.append(
                LoadedPlugin(
                    name=ep.name,
                    version=str(getattr(module, "__version__", "") or ""),
                    description=str(getattr(module, "__doc__", "") or "").strip().split("\n")[0],
                    source="entry_point",
                    path=Path(getattr(module, "__file__", "") or ""),
                    register_fn=register_fn,
                    error=None if callable(register_fn) else "no register() exported",
                    enabled=callable(register_fn),
                )
            )
        return results

    # ── load + register ──

    def _maybe_register(self, plugin: LoadedPlugin) -> None:
        """Apply enable/disable lists, import + run ``register()``, capture errors."""
        # deny wins
        if plugin.name in self.disabled:
            plugin.enabled = False
            plugin.error = "disabled via [plugins.disabled]"
            self._loaded.append(plugin)
            return
        # allow-list (when set)
        if self.enabled is not None and plugin.name not in self.enabled:
            plugin.enabled = False
            plugin.error = "not listed in [plugins.enabled]"
            self._loaded.append(plugin)
            return

        # Directory plugin: import the package by file path.
        if plugin.register_fn is None and plugin.path and plugin.path.is_dir():
            try:
                plugin.register_fn = self._load_register_fn(plugin)
            except Exception as e:
                plugin.error = f"import failed: {e}"
                plugin.enabled = False
                self._loaded.append(plugin)
                logger.warning("Plugin %s failed to import: %s", plugin.name, e)
                return

        if plugin.register_fn is None:
            plugin.error = plugin.error or "no register() callable"
            plugin.enabled = False
            self._loaded.append(plugin)
            return

        ctx = PluginContext(
            registry=self.registry,
            memory_registry=self.memory_registry,
            slash_commands=self.slash_commands,
            hooks=self.hooks,
            plugin=plugin,
        )
        try:
            plugin.register_fn(ctx)
            plugin.enabled = True
        except Exception as e:
            plugin.error = f"register() raised: {e}"
            plugin.enabled = False
            logger.exception("Plugin %s register() failed", plugin.name)
        self._loaded.append(plugin)

    @staticmethod
    def _load_register_fn(plugin: LoadedPlugin) -> Any:
        """Import a directory plugin's ``__init__.py`` and return its ``register``."""
        assert plugin.path is not None
        init_file = plugin.path / "__init__.py"
        # Namespace the import so two plugins with the same name from
        # different sources (bundled vs user vs project) don't collide in
        # ``sys.modules``.
        mod_name = f"deepagent_hermes_plugin_{plugin.source}_{plugin.name}"
        spec = importlib.util.spec_from_file_location(mod_name, init_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build module spec for {init_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        register_fn = getattr(module, "register", None)
        if not callable(register_fn):
            raise AttributeError(f"Plugin {plugin.name!r} at {init_file} has no callable register()")
        return register_fn


__all__ = ["ENTRY_POINTS_GROUP", "HermesPluginLoader"]
