"""`/compress` forces context compression on demand (gh #59).

The README Quick start lists `/compress` ("force context compression"), but the REPL
handler used to be an unwired stub that printed a "not yet wired… v0.2 task" message. It
now builds a HermesCompressionMiddleware from the resolved config (mirroring how the agent
wires it) and compresses the in-REPL history in place, falling back to a non-model summary
when the summariser can't run (so it works keyless).
"""

import os

from langchain_core.messages import AIMessage, HumanMessage

from langstage_hermes.cli import _slash_compress
from langstage_hermes.config import HermesConfig


def _no_keys(monkeypatch):
    # Force the keyless fallback-summary path: deterministic, no network/model call.
    for var in list(os.environ):
        if "API_KEY" in var or var.startswith(("LANGSTAGE_HERMES", "DEEPAGENT_HERMES")):
            monkeypatch.delenv(var, raising=False)


def _state(messages):
    return {"messages": messages, "cfg": HermesConfig.resolve(use_toml=False)}


def test_compress_reduces_a_long_history(monkeypatch):
    _no_keys(monkeypatch)
    # 3 small head + 10 large middle + 20 small tail (protect_first_n=3, protect_last_n=20),
    # so there is a real middle to summarise and tokens genuinely drop.
    messages = [HumanMessage(content="hi"), AIMessage(content="hello"), HumanMessage(content="ok")]
    for i in range(10):
        messages.append(AIMessage(content=f"middle {i} " + ("lorem ipsum dolor sit amet " * 60)))
    for i in range(20):
        messages.append(HumanMessage(content=f"tail {i}"))

    st = _state(messages)
    n_before = len(st["messages"])
    _slash_compress("", st)

    assert len(st["messages"]) < n_before  # compressed to fewer messages
    # a summary message replaced the middle
    joined = " ".join(str(getattr(m, "content", "")) for m in st["messages"]).lower()
    assert "summary" in joined or "summariz" in joined or "compress" in joined


def test_compress_on_empty_history_is_a_no_op(monkeypatch, capsys):
    _no_keys(monkeypatch)
    st = _state([])
    _slash_compress("", st)
    assert st["messages"] == []
    assert "nothing to compress" in capsys.readouterr().out.lower()


def test_compress_on_tiny_history_reports_already_minimal(monkeypatch, capsys):
    _no_keys(monkeypatch)
    st = _state([HumanMessage(content="hi"), AIMessage(content="hello")])
    _slash_compress("", st)
    # too little to gain from compressing — history left untouched, said so.
    assert len(st["messages"]) == 2
    assert "minimal" in capsys.readouterr().out.lower()
