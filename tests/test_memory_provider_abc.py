"""Tests for ``deepagent_hermes.memory.provider`` — ABC + registry + noop."""

from __future__ import annotations

import pytest

from deepagent_hermes.memory.provider import (
    MemoryProvider,
    NoopMemoryProvider,
    available_providers,
    get_provider,
    register_provider,
)


def test_noop_provider_instantiates_and_does_nothing() -> None:
    """The default provider must accept all four ABC methods as no-ops."""
    p = NoopMemoryProvider()
    # Each method returns None / [] and raises nothing
    assert p.setup_session("session-123", user_id="user-456") is None
    assert p.recall("any query") == []
    assert p.recall("query", mode="context") == []
    assert p.recall("query", mode="tools") == []
    assert p.recall("query", mode="hybrid") == []
    assert p.record_turn("user", "hello") is None
    assert p.record_turn("assistant", "hi there") is None
    assert p.teardown() is None


def test_register_provider_and_get_provider_roundtrip() -> None:
    class FakeProvider(MemoryProvider):
        """Minimal subclass used purely for registry round-trip testing."""

        def setup_session(self, session_id: str, user_id: str | None = None) -> None:
            self.session = session_id  # type: ignore[attr-defined]

        def recall(self, query: str, mode: str = "hybrid") -> list[str]:
            return [f"echo: {query}"]

        def record_turn(self, role: str, content: str) -> None:
            return None

        def teardown(self) -> None:
            return None

    register_provider("fake-test-only", FakeProvider)
    cls = get_provider("fake-test-only")
    assert cls is FakeProvider

    inst = cls()
    inst.setup_session("s1")
    assert inst.session == "s1"  # type: ignore[attr-defined]
    assert inst.recall("ping") == ["echo: ping"]


def test_get_provider_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="No memory provider"):
        get_provider("definitely-does-not-exist")


def test_empty_string_resolves_to_noop() -> None:
    """``config.memory.provider == ""`` is the documented "disabled" value.
    The registry must expose ``NoopMemoryProvider`` under that key so callers
    can do ``get_provider(config.memory_provider)`` without a None check."""
    assert get_provider("") is NoopMemoryProvider
    assert get_provider("noop") is NoopMemoryProvider


def test_available_providers_lists_registered() -> None:
    names = available_providers()
    assert "" in names
    assert "noop" in names
    # Sorted alphabetically — keeps test snapshots stable
    assert names == sorted(names)


def test_register_provider_rejects_non_subclass() -> None:
    class NotAProvider:
        pass

    with pytest.raises(TypeError, match="MemoryProvider subclass"):
        register_provider("bad", NotAProvider)  # type: ignore[arg-type]


def test_register_provider_rejects_empty_name_for_non_noop() -> None:
    """The empty-string key is reserved for the noop provider — other classes
    can't claim it (would silently disable cross-session memory)."""

    class OtherProvider(NoopMemoryProvider):
        """Any non-noop subclass — content doesn't matter for this test."""

    with pytest.raises(ValueError, match="reserved for NoopMemoryProvider"):
        register_provider("", OtherProvider)


def test_abc_cannot_be_instantiated_directly() -> None:
    """``MemoryProvider`` is abstract — subclasses must implement all four
    methods. This guards against accidental instantiation in tests."""
    with pytest.raises(TypeError, match="abstract"):
        MemoryProvider()  # type: ignore[abstract]


def test_honcho_provider_registers_on_import() -> None:
    """Importing the bundled honcho plug-in module should register it under
    the name 'honcho' — even if the honcho-ai SDK isn't installed (init
    is lazy, setup_session is what raises)."""
    # Import the bundled plug-in package — this is the only side effect we
    # care about for this test.
    import deepagent_hermes.plugins.builtin.honcho_provider  # noqa: F401

    cls = get_provider("honcho")
    assert cls is not None
    # Subclass check — confirms it implements the ABC contract
    assert issubclass(cls, MemoryProvider)


def test_honcho_provider_raises_helpful_error_when_sdk_missing() -> None:
    """When honcho-ai isn't installed, setup_session must raise an ImportError
    with the install command in the message — not just a bare 'no module' error.
    """
    import sys

    import deepagent_hermes.plugins.builtin.honcho_provider as hp_module  # noqa: F401

    HonchoProvider = get_provider("honcho")
    inst = HonchoProvider()

    # If honcho IS installed (CI variant), we skip — the error path is what
    # we're testing here.
    if "honcho" in sys.modules or _honcho_importable():
        pytest.skip("honcho-ai is installed; can't test missing-SDK error path")

    with pytest.raises(ImportError, match="pip install deepagent-hermes\\[honcho\\]"):
        inst.setup_session("s1", user_id="u1")


def _honcho_importable() -> bool:
    """True iff ``import honcho`` would succeed right now."""
    import importlib

    try:
        importlib.import_module("honcho")
        return True
    except ImportError:
        return False
