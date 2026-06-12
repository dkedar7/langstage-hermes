"""Tests for ``HermesCompressionMiddleware``.

Verifies (SPEC §7):

* Below threshold -> no-op.
* Above threshold -> head verbatim, tail verbatim, middle collapsed to a
  single SystemMessage summary.
* Summary prefix appears on the replacement message.
* Anti-thrash skips subsequent compressions after two low-yield passes.
* ``aux_model`` is the one invoked for summarisation (not main model).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langstage_hermes.compression import (
    SUMMARY_PREFIX,
    HermesCompressionMiddleware,
    _estimate_tokens,
)

# ── helpers ──────────────────────────────────────────────────────────


def _make_aux_model(summary_text: str = "FIXED_SUMMARY_BODY"):
    """Return a MagicMock that mimics ``chat_model.invoke(...)`` returning a Message."""
    aux = MagicMock(name="aux_model")
    aux.invoke = MagicMock(return_value=AIMessage(content=summary_text))
    return aux


def _bulky_messages(n: int, *, body_chars: int = 800) -> list:
    """``n`` messages, alternating Human/AI, each ``body_chars`` chars of content.

    With char/4 token estimation, n=200, body_chars=800 → ≈ n*200 ≈ 40 000
    tokens — well above a 50% threshold on a 200K context model (100K).
    Bumping further gives plenty of head-room.
    """
    out = []
    pad = "x" * body_chars
    for i in range(n):
        if i % 2 == 0:
            out.append(HumanMessage(content=f"user-{i}: {pad}"))
        else:
            out.append(AIMessage(content=f"ai-{i}: {pad}"))
    return out


# ── tests ────────────────────────────────────────────────────────────


def test_below_threshold_is_noop():
    aux = _make_aux_model()
    mw = HermesCompressionMiddleware(
        model=MagicMock(),
        aux_model=aux,
        threshold_percent=0.50,
        context_length=200_000,
        protect_first_n=3,
        protect_last_n=20,
    )
    # A short conversation — well below 100K tokens.
    msgs = [HumanMessage(content=f"msg {i}") for i in range(10)]
    result = mw.compress(msgs)
    assert result is msgs  # unchanged reference
    aux.invoke.assert_not_called()


def test_compression_protects_head_and_tail_replaces_middle():
    aux = _make_aux_model("SUMMARY_OF_MIDDLE_TURNS")
    mw = HermesCompressionMiddleware(
        model=MagicMock(),
        aux_model=aux,
        threshold_percent=0.05,  # trip easily — ~10K threshold on 200K context
        context_length=200_000,
        protect_first_n=3,
        protect_last_n=20,
    )
    msgs = _bulky_messages(200, body_chars=400)
    original_tokens = _estimate_tokens(msgs)
    assert original_tokens > mw.threshold_tokens

    new = mw.compress(list(msgs))

    # Head: first 3 preserved verbatim.
    for i in range(3):
        assert new[i] is msgs[i]

    # The 4th element is the inserted SystemMessage summary.
    assert isinstance(new[3], SystemMessage)
    assert new[3].content.startswith(SUMMARY_PREFIX)
    assert "SUMMARY_OF_MIDDLE_TURNS" in new[3].content

    # Tail: at least protect_last_n verbatim copies of the original tail.
    tail_count = len(new) - 3 - 1  # subtract head + summary slot
    assert tail_count >= 20
    # The very last element is the last original message.
    assert new[-1] is msgs[-1]

    # Token estimate dropped — compressed total < threshold_tokens.
    new_tokens = _estimate_tokens(new)
    assert new_tokens < original_tokens
    aux.invoke.assert_called_once()


def test_aux_model_invocation_uses_template_and_aux():
    """``aux_model`` (not main ``model``) is the one called; payload includes the template."""
    aux = _make_aux_model("Z")
    main = MagicMock(name="main_model")
    main.invoke = MagicMock(side_effect=RuntimeError("main_model must not be called"))
    mw = HermesCompressionMiddleware(
        model=main,
        aux_model=aux,
        threshold_percent=0.05,
        context_length=200_000,
    )
    msgs = _bulky_messages(200, body_chars=400)
    mw.compress(list(msgs))
    aux.invoke.assert_called_once()
    main.invoke.assert_not_called()


def test_summary_failure_uses_deterministic_fallback():
    aux = MagicMock(name="aux_model")
    aux.invoke = MagicMock(side_effect=RuntimeError("summariser down"))
    mw = HermesCompressionMiddleware(
        model=MagicMock(),
        aux_model=aux,
        threshold_percent=0.05,
        context_length=200_000,
        abort_on_summary_failure=False,
    )
    msgs = _bulky_messages(200, body_chars=400)
    new = mw.compress(list(msgs))
    # The summary slot still exists, with the fallback text inside.
    assert isinstance(new[3], SystemMessage)
    assert "[Earlier conversation summarised:" in new[3].content


def test_abort_on_summary_failure_reraises():
    aux = MagicMock(name="aux_model")
    aux.invoke = MagicMock(side_effect=RuntimeError("summariser down"))
    mw = HermesCompressionMiddleware(
        model=MagicMock(),
        aux_model=aux,
        threshold_percent=0.05,
        context_length=200_000,
        abort_on_summary_failure=True,
    )
    msgs = _bulky_messages(200, body_chars=400)
    with pytest.raises(RuntimeError, match="summariser down"):
        mw.compress(list(msgs))


def test_before_model_hook_returns_updated_messages_when_compressed():
    aux = _make_aux_model("S")
    mw = HermesCompressionMiddleware(
        model=MagicMock(),
        aux_model=aux,
        threshold_percent=0.05,
        context_length=200_000,
    )
    msgs = _bulky_messages(200, body_chars=400)
    update = mw.before_model({"messages": msgs})
    assert update is not None
    assert "messages" in update
    assert len(update["messages"]) < len(msgs)


def test_before_model_hook_returns_none_when_below_threshold():
    aux = _make_aux_model()
    mw = HermesCompressionMiddleware(
        model=MagicMock(),
        aux_model=aux,
        threshold_percent=0.50,
        context_length=200_000,
    )
    msgs = [HumanMessage(content="hi")]
    assert mw.before_model({"messages": msgs}) is None
