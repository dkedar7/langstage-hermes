"""Regression tests for gh #68 — ``verify`` must not leak its ``/tmp`` workspace.

Before the fix, ``verify`` created ``tempfile.mkdtemp(prefix="dah-verify-")`` and
never removed it: the build-failure, invoke-failure, and ``VERIFY: PASS`` paths
all returned without cleanup, leaking one directory per run — precisely on the
command the README tells users to "run first on any fresh install" (i.e. re-run
while debugging their key/model setup).

These tests drive the real ``verify`` command through ``CliRunner`` with a
controlled temp root and assert nothing is left behind — on BOTH the failure and
the success path — plus the ``--keep-workspace`` opt-out for post-mortem
inspection. Run against the pre-fix ``cli.py`` they fail (a ``dah-verify-*`` dir
survives); against the fix they pass.
"""

from __future__ import annotations

import tempfile as tempfile_mod
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langstage_hermes.cli import cli


class _YesModel(BaseChatModel):
    """A tool-free fake that just answers ``YES`` — enough for verify's one
    ≤20-token round-trip, with no network and no API key."""

    @property
    def _llm_type(self) -> str:  # pragma: no cover - identity only
        return "test-yes-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="YES"))])


@pytest.fixture
def scratch_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Route ``tempfile.mkdtemp`` into an isolated dir we can scan for leaks."""
    scratch = tmp_path / "tmproot"
    scratch.mkdir()
    monkeypatch.setattr(tempfile_mod, "tempdir", str(scratch))
    return scratch


def _leaked(scratch: Path) -> list[Path]:
    return list(scratch.glob("dah-verify-*"))


def _inject_fake_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``verify``'s ``create_hermes_agent`` build a real graph on a fake model."""
    from langstage_hermes.agent import create_hermes_agent as real_factory

    def fake_factory(cfg: Any = None, **kwargs: Any) -> Any:
        return real_factory(cfg, model=_YesModel(), **kwargs)

    monkeypatch.setattr("langstage_hermes.create_hermes_agent", fake_factory)


def test_verify_removes_workspace_on_build_failure(tmp_hermes_home: Path, scratch_tmp: Path, monkeypatch):
    """The build-failure path (issue repro) must clean up its workspace."""
    # An openai:* model + a dummy key gets past verify's key gate, so the
    # workspace IS created — then we force the build to fail after it.
    monkeypatch.setenv("LANGSTAGE_HERMES_MODEL_DEFAULT", "openai:openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-not-real")

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated agent build failure")

    monkeypatch.setattr("langstage_hermes.create_hermes_agent", boom)

    result = CliRunner().invoke(cli, ["verify"])

    assert result.exit_code == 2
    # Sanity: we really did get past the key gate and create the workspace.
    assert "isolated workspace" in result.output
    assert _leaked(scratch_tmp) == [], f"verify leaked its workspace on build failure: {_leaked(scratch_tmp)}"


def test_verify_removes_workspace_on_success(tmp_hermes_home: Path, scratch_tmp: Path, monkeypatch):
    """The ``VERIFY: PASS`` success path must also clean up its workspace."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-not-real")
    _inject_fake_factory(monkeypatch)

    result = CliRunner().invoke(cli, ["verify"])

    assert result.exit_code == 0, result.output
    assert "VERIFY: PASS" in result.output
    assert _leaked(scratch_tmp) == [], f"verify leaked its workspace on success: {_leaked(scratch_tmp)}"


def test_verify_keep_workspace_flag_preserves_it(tmp_hermes_home: Path, scratch_tmp: Path, monkeypatch):
    """``--keep-workspace`` is the opt-out: the dir survives and is reported."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-not-real")
    _inject_fake_factory(monkeypatch)

    result = CliRunner().invoke(cli, ["verify", "--keep-workspace"])

    assert result.exit_code == 0, result.output
    assert "kept" in result.output
    kept = _leaked(scratch_tmp)
    assert len(kept) == 1, f"--keep-workspace should preserve exactly one workspace, found: {kept}"
