"""Tests for `create_hermes_agent(model=...)` — the BYOM kwarg path.

Bring-your-own-model lets callers hand a fully-configured langchain
`BaseChatModel` instance (Azure, Bedrock, OpenAI-compatible proxy, etc.)
to the factory without going through `init_chat_model`'s
``provider:name`` string. These tests don't need a real model — they
use a stub that satisfies the langchain interface enough to build the
graph, plus a monkeypatch on `_init_chat_model` to detect whether the
factory fell through to the string-driven path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from deepagent_hermes.agent import create_hermes_agent
from deepagent_hermes.config import HermesConfig


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test HERMES_HOME so state.db / skills / memories are isolated."""
    monkeypatch.setenv("DEEPAGENT_HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _stub_model() -> Any:
    """Smallest valid `BaseChatModel` we can hand the factory.

    `FakeListChatModel` ships with langchain-core for exactly this kind
    of test scaffolding — it satisfies `bind_tools()`, `with_config()`,
    `invoke()`, etc. without making a real API call.
    """
    return FakeListChatModel(responses=["stub response"])


def test_supplied_model_bypasses_init_chat_model(home: Path):
    """If `model=...` is passed, `init_chat_model` is never called."""
    cfg = HermesConfig.resolve()
    model = _stub_model()

    with patch("deepagent_hermes.agent._init_chat_model") as init_mock:
        graph = create_hermes_agent(cfg, model=model)

    assert init_mock.call_count == 0, (
        f"_init_chat_model was called {init_mock.call_count} times; expected 0 "
        "because the caller supplied `model=...` and `aux_model` was not set "
        "(should default to the supplied main model)"
    )
    assert graph is not None
    # Belt-and-suspenders: the compiled graph should be useable.
    assert hasattr(graph, "deepagent_hermes_session_id")


def test_no_model_kwarg_uses_init_chat_model(home: Path):
    """Default path still works — falls through to init_chat_model."""
    cfg = HermesConfig.resolve()
    cfg.model_default = "anthropic:claude-haiku-4-5-20250929"

    stub = _stub_model()
    with patch("deepagent_hermes.agent._init_chat_model", return_value=stub) as init_mock:
        graph = create_hermes_agent(cfg)

    assert init_mock.call_count >= 1, "init_chat_model should be the fallback when no model is supplied"
    # The first call should ask for the default model id.
    first_call_arg = init_mock.call_args_list[0].args[0]
    assert first_call_arg == "anthropic:claude-haiku-4-5-20250929"
    assert graph is not None


def test_explicit_aux_model_kwarg_is_respected(home: Path):
    """Supplying `aux_model` independently bypasses init_chat_model for both."""
    cfg = HermesConfig.resolve()
    main = _stub_model()
    aux = _stub_model()
    assert main is not aux  # sanity: distinct instances

    with patch("deepagent_hermes.agent._init_chat_model") as init_mock:
        graph = create_hermes_agent(cfg, model=main, aux_model=aux)

    assert init_mock.call_count == 0, "Both main and aux were supplied — init_chat_model should not be touched"
    assert graph is not None


def test_main_model_only_shares_with_aux(home: Path):
    """Supplying `model=` alone defaults `aux_model` to the same instance.

    This matches the string-driven path's behaviour when only
    `model_default` is set (aux falls back to main).
    """
    cfg = HermesConfig.resolve()
    cfg.model_aux = None
    main = _stub_model()

    # If the factory wrongly tried to init aux from the config string when
    # only `model` was supplied, this patch would catch it.
    with patch("deepagent_hermes.agent._init_chat_model") as init_mock:
        graph = create_hermes_agent(cfg, model=main)

    assert init_mock.call_count == 0
    assert graph is not None


def test_byom_model_accepted_with_no_config(home: Path):
    """The most ergonomic call site: just hand a model, take all other defaults."""
    model = _stub_model()
    with patch("deepagent_hermes.agent._init_chat_model") as init_mock:
        graph = create_hermes_agent(model=model)
    assert init_mock.call_count == 0
    assert graph is not None
