"""Tests for the bundled ``HonchoProvider`` plug-in.

The real ``honcho-ai`` SDK (PyPI ``honcho-ai``, imported as ``honcho``) is
heavyweight (httpx + pydantic + a live API). These tests install a fake
``honcho`` module into ``sys.modules`` and assert the provider drives the
expected SDK surface without ever touching the network.

Why mock instead of using a recording transport:
  - The provider's contract is "make these specific SDK calls in this order".
    That's a structural assertion, best tested with ``MagicMock``.
  - Tests must pass with or without ``honcho-ai`` installed in the dev venv.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from deepagent_hermes.memory.provider import MemoryProvider, get_provider

# ── Fake SDK plumbing ────────────────────────────────────────────────


def _fresh_provider_module():
    """Re-import the provider module under a clean ``honcho`` mock.

    The provider registers itself at import time, so a stale module reference
    is fine for `get_provider("honcho")` — but for tests that need to assert
    on the lazy import path we re-run the import after patching sys.modules.
    """
    name = "deepagent_hermes.plugins.builtin.honcho_provider"
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def _make_fake_honcho_module() -> tuple[types.ModuleType, MagicMock, MagicMock]:
    """Build a fake ``honcho`` package exposing ``Honcho`` + ``MessageCreateParams``.

    Returns ``(module, HonchoClass, MessageCreateParamsClass)`` so individual
    tests can install per-call side effects on the mock client.
    """
    fake = types.ModuleType("honcho")
    honcho_cls = MagicMock(name="HonchoClass")
    mcp_cls = MagicMock(name="MessageCreateParamsClass")
    # MessageCreateParams is called like a constructor — make it return a
    # sentinel that we can identify in assertions on add_messages.
    mcp_cls.side_effect = lambda **kwargs: {"_kind": "MessageCreateParams", **kwargs}
    fake.Honcho = honcho_cls
    fake.MessageCreateParams = mcp_cls
    return fake, honcho_cls, mcp_cls


def _wire_client_for_setup(honcho_cls: MagicMock) -> dict[str, Any]:
    """Configure the Honcho mock so .peer() / .session() return inspectable mocks.

    Returns a dict of the inner mocks so tests can assert against them.
    """
    client = MagicMock(name="client_instance")
    user_peer = MagicMock(name="user_peer")
    ai_peer = MagicMock(name="ai_peer")
    session = MagicMock(name="session")

    def peer_side_effect(peer_id: str, **kwargs):
        if peer_id == "assistant":
            return ai_peer
        return user_peer

    client.peer.side_effect = peer_side_effect
    client.session.return_value = session
    honcho_cls.return_value = client

    return {
        "client": client,
        "user_peer": user_peer,
        "ai_peer": ai_peer,
        "session": session,
    }


@pytest.fixture
def fake_honcho(monkeypatch):
    """Install a fake ``honcho`` package into sys.modules + wire .peer/.session.

    Yields a dict with the mocks: ``honcho_cls``, ``mcp_cls``, ``client``,
    ``user_peer``, ``ai_peer``, ``session``.
    """
    fake, honcho_cls, mcp_cls = _make_fake_honcho_module()
    monkeypatch.setitem(sys.modules, "honcho", fake)
    wired = _wire_client_for_setup(honcho_cls)
    yield {"honcho_cls": honcho_cls, "mcp_cls": mcp_cls, **wired}


@pytest.fixture
def clean_honcho_env(monkeypatch):
    """Wipe HONCHO_* + DEEPAGENT_HERMES_* env so each test starts with a blank slate."""
    for var in (
        "HONCHO_API_KEY",
        "HONCHO_ENVIRONMENT",
        "HONCHO_BASE_URL",
        "DEEPAGENT_HERMES_HONCHO_HOST",
        "DEEPAGENT_HERMES_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)


# ── Tests: setup_session ─────────────────────────────────────────────


def test_setup_session_lazy_imports_honcho(monkeypatch, clean_honcho_env):
    """When ``honcho`` cannot be imported, ``setup_session`` raises a helpful
    ImportError that mentions the extras install command."""
    # Make `import honcho` fail. Setting to None in sys.modules makes Python's
    # import machinery raise ImportError on next `import honcho`.
    monkeypatch.setitem(sys.modules, "honcho", None)

    # Reload the provider module so it re-runs at import time without a real
    # honcho. The module *itself* doesn't import honcho at top level — the
    # import is inside setup_session — so reloading is mainly to defeat any
    # caching from previous tests.
    mod = _fresh_provider_module()
    inst = mod.HonchoProvider()

    with pytest.raises(ImportError, match=r"pip install deepagent-hermes\[honcho\]"):
        inst.setup_session("s1", user_id="u1")


def test_setup_session_resolves_config_chain(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """A ``honcho.json`` in HERMES_HOME provides api_key + environment."""
    cfg = tmp_hermes_home / "honcho.json"
    cfg.write_text(
        '{"api_key": "sk-from-file", "environment": "production"}',
        encoding="utf-8",
    )

    mod = _fresh_provider_module()
    inst = mod.HonchoProvider()
    inst.setup_session("session-abc", user_id="alice")

    # Honcho(...) called with the file-sourced kwargs.
    kwargs = fake_honcho["honcho_cls"].call_args.kwargs
    assert kwargs["api_key"] == "sk-from-file"
    assert kwargs["environment"] == "production"
    assert kwargs["workspace_id"] == "deepagent_hermes"


def test_setup_session_uses_env_fallback(monkeypatch, fake_honcho, clean_honcho_env):
    """With no config files but HONCHO_API_KEY set, env wins."""
    monkeypatch.setenv("HONCHO_API_KEY", "sk-from-env")
    monkeypatch.setenv("HONCHO_ENVIRONMENT", "local")
    # No HERMES_HOME fixture → _hermes_home() points at ~/.deepagent-hermes,
    # which is the user's real dir. Override to a tmp empty dir to be safe.
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(monkeypatch.delenv  # placeholder
        if False else "")
    )
    # Better: set HERMES_HOME to a tmp path that has no honcho.json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("DEEPAGENT_HERMES_HOME", td)
        monkeypatch.setenv("HERMES_HOME", td)
        # And aim Path.home() at the same empty dir to defeat ~/.honcho lookup.
        monkeypatch.setenv("HOME", td)
        monkeypatch.setenv("USERPROFILE", td)

        mod = _fresh_provider_module()
        inst = mod.HonchoProvider()
        inst.setup_session("s-env", user_id="bob")

        kwargs = fake_honcho["honcho_cls"].call_args.kwargs
        assert kwargs["api_key"] == "sk-from-env"
        assert kwargs["environment"] == "local"


def test_setup_session_creates_workspace_peers_session(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """setup_session must call client.peer twice + client.session once."""
    monkeypatch.setenv("HONCHO_API_KEY", "sk-test")

    mod = _fresh_provider_module()
    inst = mod.HonchoProvider()
    inst.setup_session("sess-1", user_id="kedar")

    client = fake_honcho["client"]
    # user peer + assistant peer
    peer_calls = [c.args[0] for c in client.peer.call_args_list]
    assert "kedar" in peer_calls
    assert "assistant" in peer_calls

    # session created with both peers attached
    client.session.assert_called_once()
    sess_args = client.session.call_args
    assert sess_args.args[0] == "sess-1"
    peers_kwarg = sess_args.kwargs.get("peers") or (
        sess_args.args[1] if len(sess_args.args) > 1 else None
    )
    assert peers_kwarg is not None
    assert fake_honcho["user_peer"] in peers_kwarg
    assert fake_honcho["ai_peer"] in peers_kwarg


def test_setup_session_workspace_key_sanitization(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """Profile ``'My Project!'`` → workspace ``'deepagent_hermes_my_project'``."""
    monkeypatch.setenv("HONCHO_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPAGENT_HERMES_PROFILE", "My Project!")

    mod = _fresh_provider_module()
    inst = mod.HonchoProvider()
    inst.setup_session("s-x")

    kwargs = fake_honcho["honcho_cls"].call_args.kwargs
    assert kwargs["workspace_id"] == "deepagent_hermes_my_project"


def test_setup_session_is_idempotent(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """Calling setup_session twice should not crash — client.peer/session are
    idempotent get_or_create per the SDK contract."""
    monkeypatch.setenv("HONCHO_API_KEY", "sk-test")

    mod = _fresh_provider_module()
    inst = mod.HonchoProvider()
    inst.setup_session("s-1", user_id="u-1")
    inst.setup_session("s-1", user_id="u-1")  # should not raise


# ── Tests: recall ────────────────────────────────────────────────────


def _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home):
    """Helper: clean env, set api key, return an initialized provider."""
    monkeypatch.setenv("HONCHO_API_KEY", "sk-test")
    mod = _fresh_provider_module()
    inst = mod.HonchoProvider()
    inst.setup_session("s-recall", user_id="u-recall")
    return mod, inst


def _make_message(content: str, peer_id: str = "u-recall"):
    """Build a fake ``honcho.Message`` with .content + .peer_id attrs."""
    m = MagicMock()
    m.content = content
    m.peer_id = peer_id
    return m


def test_recall_hybrid_combines_chat_and_messages(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """hybrid mode calls BOTH peer.chat() and session.messages()."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    user_peer = fake_honcho["user_peer"]
    session = fake_honcho["session"]
    user_peer.chat.return_value = "user likes mountain biking"
    session.messages.return_value = [
        _make_message("hello there", peer_id="u-recall"),
        _make_message("hi back", peer_id="assistant"),
    ]

    results = inst.recall("what does the user like?", mode="hybrid")

    user_peer.chat.assert_called_once()
    session.messages.assert_called_once()
    # hybrid uses size=5 per spec
    assert session.messages.call_args.kwargs.get("size") == 5
    assert any("mountain biking" in r for r in results)
    assert any("hello there" in r for r in results)


def test_recall_context_only_lists_messages(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """context mode skips peer.chat() entirely."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    user_peer = fake_honcho["user_peer"]
    session = fake_honcho["session"]
    session.messages.return_value = [_make_message("recent turn")]

    inst.recall("anything", mode="context")

    user_peer.chat.assert_not_called()
    session.messages.assert_called_once()
    # context mode uses size=20 per spec
    assert session.messages.call_args.kwargs.get("size") == 20


def test_recall_tools_only_calls_chat(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """tools mode skips session.messages() entirely."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    user_peer = fake_honcho["user_peer"]
    session = fake_honcho["session"]
    user_peer.chat.return_value = "dialectic answer"

    results = inst.recall("query", mode="tools")

    user_peer.chat.assert_called_once()
    session.messages.assert_not_called()
    assert "dialectic answer" in results


def test_recall_auto_aliases_to_hybrid(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """Legacy mode='auto' must behave identically to mode='hybrid'."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    user_peer = fake_honcho["user_peer"]
    session = fake_honcho["session"]
    user_peer.chat.return_value = "x"
    session.messages.return_value = []

    inst.recall("q", mode="auto")  # type: ignore[arg-type]

    user_peer.chat.assert_called_once()
    session.messages.assert_called_once()


def test_recall_swallows_errors(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """If the SDK raises, recall returns [] — never propagates."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    fake_honcho["user_peer"].chat.side_effect = RuntimeError("honcho down")
    fake_honcho["session"].messages.side_effect = RuntimeError("also down")

    # Must not raise even though both sub-calls explode.
    result = inst.recall("q", mode="hybrid")
    assert result == []


def test_recall_returns_empty_when_not_setup(monkeypatch, clean_honcho_env):
    """Calling recall before setup_session must safely return []."""
    mod = _fresh_provider_module()
    inst = mod.HonchoProvider()
    assert inst.recall("hello") == []


def test_recall_empty_query_returns_empty(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """Whitespace/empty queries are dropped before hitting the SDK."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    assert inst.recall("") == []
    assert inst.recall("   ") == []
    fake_honcho["user_peer"].chat.assert_not_called()


# ── Tests: record_turn ───────────────────────────────────────────────


def test_record_turn_user_vs_assistant(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """role='user' → peer_id=user; role='assistant' → peer_id=assistant."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    session = fake_honcho["session"]
    inst.record_turn("user", "I like trail running")
    inst.record_turn("assistant", "noted")

    assert session.add_messages.call_count == 2
    first_msg = session.add_messages.call_args_list[0].args[0]
    second_msg = session.add_messages.call_args_list[1].args[0]
    # The fake MessageCreateParams returns a dict with kwargs we can inspect.
    assert first_msg["peer_id"] == "u-recall"
    assert first_msg["content"] == "I like trail running"
    assert second_msg["peer_id"] == "assistant"
    assert second_msg["content"] == "noted"


def test_record_turn_skips_unknown_role(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """role='tool'/'system'/etc. → no SDK call, no exception."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    inst.record_turn("tool", "tool output")
    inst.record_turn("system", "system message")
    inst.record_turn("weird-role", "something")

    fake_honcho["session"].add_messages.assert_not_called()


def test_record_turn_skips_empty_content(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """Empty / whitespace content is dropped before hitting the SDK."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    inst.record_turn("user", "")
    inst.record_turn("user", "   ")
    fake_honcho["session"].add_messages.assert_not_called()


def test_record_turn_swallows_errors(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """SDK exceptions in add_messages must not bubble out of record_turn."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)
    fake_honcho["session"].add_messages.side_effect = RuntimeError("write failed")
    # Should NOT raise.
    inst.record_turn("user", "anything")


def test_record_turn_noop_when_not_setup(monkeypatch, clean_honcho_env):
    """record_turn before setup_session → safe no-op."""
    mod = _fresh_provider_module()
    inst = mod.HonchoProvider()
    inst.record_turn("user", "hi")  # must not raise


# ── Tests: teardown ──────────────────────────────────────────────────


def test_teardown_best_effort_calls_close_when_available(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """If session.close exists, teardown should call it once."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    session = fake_honcho["session"]
    session.close = MagicMock()  # explicit attr — MagicMock auto-attrs anyway

    inst.teardown()
    session.close.assert_called()


def test_teardown_no_exception_when_close_missing(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """If close() isn't callable, teardown should still clear state cleanly."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)

    # Replace .close with a non-callable so getattr finds it but `callable()` is False.
    fake_honcho["session"].close = "not callable"
    fake_honcho["client"].close = None

    inst.teardown()
    # Internal state cleared regardless.
    assert inst._client is None
    assert inst._session is None


def test_teardown_swallows_close_errors(
    monkeypatch, fake_honcho, tmp_hermes_home, clean_honcho_env
):
    """Exceptions raised by .close() must not propagate out of teardown."""
    _, inst = _setup_inst(monkeypatch, fake_honcho, tmp_hermes_home)
    fake_honcho["session"].close = MagicMock(side_effect=RuntimeError("close boom"))
    inst.teardown()  # must not raise


def test_teardown_is_safe_without_setup(monkeypatch, clean_honcho_env):
    """teardown before setup_session → safe no-op."""
    mod = _fresh_provider_module()
    inst = mod.HonchoProvider()
    inst.teardown()  # must not raise


# ── Tests: registry / module import ──────────────────────────────────


def test_provider_self_registers():
    """Importing the module must put the provider into the registry under 'honcho'."""
    import deepagent_hermes.plugins.builtin.honcho_provider as hp  # noqa: F401

    cls = get_provider("honcho")
    assert issubclass(cls, MemoryProvider)
    assert cls.__name__ == "HonchoProvider"


def test_provider_instantiates_without_sdk(monkeypatch, clean_honcho_env):
    """Construction must not import honcho — only setup_session does."""
    monkeypatch.setitem(sys.modules, "honcho", None)
    mod = _fresh_provider_module()
    # If __init__ tried to import honcho, this would raise.
    inst = mod.HonchoProvider()
    assert inst is not None
    assert inst.recall_mode in ("hybrid", "context", "tools")


def test_auto_recall_mode_coerced_to_hybrid_in_init(monkeypatch, clean_honcho_env):
    """recall_mode='auto' on the constructor must coerce to 'hybrid'."""
    mod = _fresh_provider_module()
    inst = mod.HonchoProvider(recall_mode="auto")  # type: ignore[arg-type]
    assert inst.recall_mode == "hybrid"
