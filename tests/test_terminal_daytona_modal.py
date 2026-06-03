"""Tests for the lazy-SDK error path on :class:`DaytonaEnvironment` /
:class:`ModalEnvironment`, plus light verification that the SDK auth env vars
are honoured when the SDK *is* mockable.

These tests don't touch a real cloud sandbox — they patch the SDK module in
``sys.modules`` so we can assert on the import path and the constructor
arguments without paying network cost.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

# Convenience for cleanly reloading the backend module under test after we
# poke at sys.modules. Both backends do lazy ``import daytona_sdk`` / ``import
# modal`` inside the constructor, so a fresh module-level reload isn't even
# required — but we do it for symmetry and to make sure cached state from
# previous tests doesn't leak.

DAYTONA_MODPATH = "deepagent_hermes.tools.environments.daytona"
MODAL_MODPATH = "deepagent_hermes.tools.environments.modal"


@pytest.fixture(autouse=True)
def _restore_modules():
    """Snapshot + restore ``sys.modules`` mutations after each test.

    Both backends do a lazy ``import daytona_sdk`` / ``import modal``; once a
    test sticks a fake (or ``None``) into ``sys.modules`` we have to undo it
    before the next test runs or cross-test bleed makes failures inscrutable.
    """
    snapshot = {
        key: sys.modules.get(key)
        for key in ("daytona_sdk", "modal", DAYTONA_MODPATH, MODAL_MODPATH)
    }
    yield
    for key, value in snapshot.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value


# ── Lazy-import error path ────────────────────────────────────────────


def test_daytona_raises_without_sdk():
    """Instantiating DaytonaEnvironment without daytona-sdk must raise ImportError.

    We force the missing-package case by stuffing ``None`` into
    ``sys.modules['daytona_sdk']`` — that's the canonical way to make
    ``import daytona_sdk`` raise ``ImportError`` even on a machine where
    the SDK is actually installed.
    """
    sys.modules["daytona_sdk"] = None  # type: ignore[assignment]
    # Reload the backend module to ensure no cached SDK reference survives.
    if DAYTONA_MODPATH in sys.modules:
        importlib.reload(sys.modules[DAYTONA_MODPATH])
    from deepagent_hermes.tools.environments.daytona import DaytonaEnvironment

    with pytest.raises(ImportError) as excinfo:
        DaytonaEnvironment(session_id="x")
    assert "pip install deepagent-hermes[daytona]" in str(excinfo.value)


def test_modal_raises_without_sdk():
    """Instantiating ModalEnvironment without modal must raise ImportError."""
    sys.modules["modal"] = None  # type: ignore[assignment]
    if MODAL_MODPATH in sys.modules:
        importlib.reload(sys.modules[MODAL_MODPATH])
    from deepagent_hermes.tools.environments.modal import ModalEnvironment

    with pytest.raises(ImportError) as excinfo:
        ModalEnvironment(session_id="x")
    assert "pip install deepagent-hermes[modal]" in str(excinfo.value)


# ── Auth env-var resolution (with mocked SDK) ─────────────────────────


def _install_fake_daytona_sdk() -> tuple[types.ModuleType, MagicMock, MagicMock]:
    """Build a fake ``daytona_sdk`` module exposing ``Daytona`` + ``CreateSandboxParams``.

    Returns (module, Daytona_cls_mock, sandbox_mock) so tests can assert on
    construction args and verify the sandbox is built correctly.
    """
    fake_sdk = types.ModuleType("daytona_sdk")
    daytona_cls = MagicMock(name="Daytona")
    sandbox = MagicMock(name="Sandbox")
    daytona_cls.return_value.create.return_value = sandbox
    # process.exec returns a result-like object with .result + .exit_code.
    sandbox.process.exec.return_value = MagicMock(result="", exit_code=0)
    fake_sdk.Daytona = daytona_cls  # type: ignore[attr-defined]
    fake_sdk.CreateSandboxParams = MagicMock(name="CreateSandboxParams")  # type: ignore[attr-defined]
    sys.modules["daytona_sdk"] = fake_sdk
    return fake_sdk, daytona_cls, sandbox


def _install_fake_modal_sdk() -> tuple[types.ModuleType, MagicMock]:
    """Build a fake ``modal`` module shaped enough to satisfy init_session()."""
    fake_sdk = types.ModuleType("modal")
    sandbox = MagicMock(name="Sandbox")
    sandbox_cls = MagicMock(name="SandboxCls")
    sandbox_cls.create.return_value = sandbox
    fake_sdk.Sandbox = sandbox_cls  # type: ignore[attr-defined]

    app = MagicMock(name="App")
    app_cls = MagicMock(name="AppCls")
    app_cls.lookup.return_value = app
    fake_sdk.App = app_cls  # type: ignore[attr-defined]

    image_cls = MagicMock(name="Image")
    image_cls.from_registry = MagicMock(name="from_registry", return_value=MagicMock(name="ImageObj"))
    fake_sdk.Image = image_cls  # type: ignore[attr-defined]

    sys.modules["modal"] = fake_sdk
    return fake_sdk, sandbox_cls


def test_daytona_uses_api_key_env(monkeypatch):
    """``Daytona(api_key=...)`` must be called with the value of DAYTONA_API_KEY."""
    monkeypatch.setenv("DAYTONA_API_KEY", "test-key-123")
    monkeypatch.delenv("DEEPAGENT_HERMES_DAYTONA_API_KEY", raising=False)

    _, daytona_cls, _ = _install_fake_daytona_sdk()
    # Reload to bind the freshly-installed fake SDK.
    if DAYTONA_MODPATH in sys.modules:
        importlib.reload(sys.modules[DAYTONA_MODPATH])
    from deepagent_hermes.tools.environments.daytona import DaytonaEnvironment

    env = DaytonaEnvironment(session_id="test")
    env.init_session()  # this is where the Daytona client is constructed
    daytona_cls.assert_called_once_with(api_key="test-key-123")


def test_daytona_prefers_namespaced_env_when_official_missing(monkeypatch):
    """When DAYTONA_API_KEY is unset, fall back to the namespaced override."""
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setenv("DEEPAGENT_HERMES_DAYTONA_API_KEY", "fallback-key")

    _, daytona_cls, _ = _install_fake_daytona_sdk()
    if DAYTONA_MODPATH in sys.modules:
        importlib.reload(sys.modules[DAYTONA_MODPATH])
    from deepagent_hermes.tools.environments.daytona import DaytonaEnvironment

    env = DaytonaEnvironment(session_id="test")
    env.init_session()
    daytona_cls.assert_called_once_with(api_key="fallback-key")


def test_modal_uses_token_env(monkeypatch):
    """ModalEnvironment.init_session must accept MODAL_TOKEN_ID + MODAL_TOKEN_SECRET.

    The Modal SDK reads the tokens from env itself — we don't pass them through
    a constructor call. So the contract is: with the env vars set, init_session
    succeeds and builds a sandbox; with them unset, it raises a clear
    RuntimeError mentioning the missing vars.
    """
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-test")

    fake_sdk, sandbox_cls = _install_fake_modal_sdk()
    if MODAL_MODPATH in sys.modules:
        importlib.reload(sys.modules[MODAL_MODPATH])
    from deepagent_hermes.tools.environments.modal import ModalEnvironment

    env = ModalEnvironment(session_id="test")
    env.init_session()

    # App lookup must have used the documented app name.
    fake_sdk.App.lookup.assert_called_once()
    args, kwargs = fake_sdk.App.lookup.call_args
    assert args[0] == "deepagent-hermes"
    assert kwargs.get("create_if_missing") is True

    # And Sandbox.create must have been called with the image + app.
    sandbox_cls.create.assert_called_once()
    create_kwargs = sandbox_cls.create.call_args.kwargs
    assert "image" in create_kwargs
    assert "app" in create_kwargs


def test_modal_raises_without_tokens(monkeypatch):
    """Missing MODAL_TOKEN_ID/SECRET must produce a clear pre-SDK error."""
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    _install_fake_modal_sdk()
    if MODAL_MODPATH in sys.modules:
        importlib.reload(sys.modules[MODAL_MODPATH])
    from deepagent_hermes.tools.environments.modal import ModalEnvironment

    env = ModalEnvironment(session_id="test")
    with pytest.raises(RuntimeError) as excinfo:
        env.init_session()
    assert "MODAL_TOKEN_ID" in str(excinfo.value)
    assert "MODAL_TOKEN_SECRET" in str(excinfo.value)
