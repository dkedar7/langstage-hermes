"""Tests for ``HermesToolRegistry`` — registration, retrieval, and TTL cache."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from langstage_hermes.tools.registry import HermesToolRegistry, registry

# ── Test doubles ─────────────────────────────────────────────────────


@dataclass
class _FakeTool:
    """Minimal ``BaseTool`` stand-in: anything with ``.name`` works."""

    name: str


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def reg() -> HermesToolRegistry:
    """Fresh registry per test — isolates state from the module singleton."""
    return HermesToolRegistry()


# ── Registration / retrieval ─────────────────────────────────────────


def test_register_and_get_tool(reg: HermesToolRegistry) -> None:
    """A registered tool round-trips via ``get_tool`` and ``get_tools``."""
    t = _FakeTool(name="echo")
    reg.register(t, toolset="terminal")

    assert reg.get_tool("echo") is t
    assert reg.get_toolset_for_tool("echo") == "terminal"
    assert "echo" in reg
    assert reg.names() == ["echo"]
    assert reg.get_tools() == [t]


def test_register_requires_name(reg: HermesToolRegistry) -> None:
    """Tools without a ``.name`` are rejected at registration time."""

    class NoName:
        pass

    with pytest.raises(ValueError, match=r"non-empty \.name"):
        reg.register(NoName(), toolset="terminal")


def test_get_tools_filters_by_enabled_toolsets(reg: HermesToolRegistry) -> None:
    """``enabled_toolsets`` whitelist narrows the returned set."""
    reg.register(_FakeTool("ls"), toolset="terminal")
    reg.register(_FakeTool("read_file"), toolset="file")
    reg.register(_FakeTool("memory"), toolset="memory")

    out = reg.get_tools(enabled_toolsets={"terminal", "file"})
    assert sorted(t.name for t in out) == ["ls", "read_file"]


def test_get_tools_filters_by_disabled(reg: HermesToolRegistry) -> None:
    """``disabled`` tool names are dropped even when their toolset is enabled."""
    reg.register(_FakeTool("ls"), toolset="terminal")
    reg.register(_FakeTool("rm"), toolset="terminal")
    out = reg.get_tools(disabled={"rm"})
    assert [t.name for t in out] == ["ls"]


def test_list_toolsets_groups_tool_names(reg: HermesToolRegistry) -> None:
    reg.register(_FakeTool("ls"), toolset="terminal")
    reg.register(_FakeTool("kill"), toolset="terminal")
    reg.register(_FakeTool("read_file"), toolset="file")

    grouped = reg.list_toolsets()
    assert grouped["terminal"] == ["kill", "ls"]
    assert grouped["file"] == ["read_file"]


def test_deregister_drops_tool(reg: HermesToolRegistry) -> None:
    reg.register(_FakeTool("ls"), toolset="terminal")
    reg.deregister("ls")
    assert "ls" not in reg
    assert reg.get_tool("ls") is None


# ── check_fn TTL cache ───────────────────────────────────────────────


def test_check_status_no_check_fn_is_always_available(reg: HermesToolRegistry) -> None:
    reg.register(_FakeTool("ls"), toolset="terminal")
    ok, reason = reg.check_status("ls")
    assert ok is True
    assert reason is None


def test_check_status_unknown_tool(reg: HermesToolRegistry) -> None:
    ok, reason = reg.check_status("nonexistent")
    assert ok is False
    assert reason == "unknown tool"


def test_check_status_runs_check_fn(reg: HermesToolRegistry) -> None:
    calls = {"n": 0}

    def check() -> tuple[bool, str | None]:
        calls["n"] += 1
        return (True, None)

    reg.register(_FakeTool("docker_cmd"), toolset="terminal", check_fn=check)

    ok, _ = reg.check_status("docker_cmd")
    assert ok is True
    assert calls["n"] == 1


def test_check_status_caches_for_30s(monkeypatch, reg: HermesToolRegistry) -> None:
    """Repeated calls inside the 30s window must NOT re-invoke the probe;
    once the TTL elapses, the probe is called again."""
    calls = {"n": 0}

    def check() -> tuple[bool, str | None]:
        calls["n"] += 1
        return (calls["n"] % 2 == 1, f"call #{calls['n']}")

    reg.register(_FakeTool("docker_cmd"), toolset="terminal", check_fn=check)

    # Freeze monotonic clock at t=100.
    fake_now = [100.0]
    monkeypatch.setattr(
        "langstage_hermes.tools.registry.time.monotonic",
        lambda: fake_now[0],
    )

    ok1, reason1 = reg.check_status("docker_cmd")
    assert ok1 is True
    assert reason1 == "call #1"
    assert calls["n"] == 1

    # Five seconds later: still cached.
    fake_now[0] = 105.0
    ok2, reason2 = reg.check_status("docker_cmd")
    assert (ok2, reason2) == (True, "call #1")
    assert calls["n"] == 1

    # 29 seconds later: still cached.
    fake_now[0] = 129.0
    reg.check_status("docker_cmd")
    assert calls["n"] == 1

    # 31 seconds later: TTL expired, probe re-runs.
    fake_now[0] = 131.0
    ok3, reason3 = reg.check_status("docker_cmd")
    assert ok3 is False  # second call returns False per our toggle
    assert reason3 == "call #2"
    assert calls["n"] == 2


def test_check_status_swallows_exceptions(reg: HermesToolRegistry) -> None:
    """A check_fn that raises is reported as unavailable, not propagated."""

    def boom() -> tuple[bool, str | None]:
        raise RuntimeError("docker daemon down")

    reg.register(_FakeTool("docker_cmd"), toolset="terminal", check_fn=boom)
    ok, reason = reg.check_status("docker_cmd")
    assert ok is False
    assert reason is not None and "RuntimeError" in reason


def test_get_tools_skips_unavailable_check_fn(reg: HermesToolRegistry) -> None:
    """A tool whose check_fn returns False is dropped from ``get_tools()``."""

    reg.register(_FakeTool("ls"), toolset="terminal")
    reg.register(
        _FakeTool("docker_cmd"),
        toolset="terminal",
        check_fn=lambda: (False, "no docker"),
    )
    names = [t.name for t in reg.get_tools()]
    assert names == ["ls"]


def test_invalidate_check_cache_forces_re_probe(monkeypatch, reg: HermesToolRegistry) -> None:
    calls = {"n": 0}

    def check() -> tuple[bool, str | None]:
        calls["n"] += 1
        return (True, None)

    reg.register(_FakeTool("x"), toolset="terminal", check_fn=check)

    fake_now = [100.0]
    monkeypatch.setattr(
        "langstage_hermes.tools.registry.time.monotonic",
        lambda: fake_now[0],
    )

    reg.check_status("x")
    reg.check_status("x")
    assert calls["n"] == 1

    reg.invalidate_check_cache("x")
    reg.check_status("x")
    assert calls["n"] == 2


# ── Module singleton ─────────────────────────────────────────────────


def test_module_singleton_is_a_registry() -> None:
    assert isinstance(registry, HermesToolRegistry)
