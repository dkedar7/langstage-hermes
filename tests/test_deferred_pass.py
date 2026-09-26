"""Deferred-backlog fixes: gh #140 (clean `search --json` snippets), #144 (past
`once at` rejected), #149 (interval display keeps sub-minute remainder), #155
(deprecated `deepagent-hermes` command announces itself)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from langstage_hermes import cli as cli_mod
from langstage_hermes.cli import cli
from langstage_hermes.cron import jobs as cron_jobs
from langstage_hermes.store.sqlite_fts import SqliteFtsStore

# ── #140 ────────────────────────────────────────────────────────────────


def test_search_json_snippet_is_clean_text_with_match_ranges(tmp_hermes_home: Path):
    store = SqliteFtsStore(db_path=str(tmp_hermes_home / "state.db"))
    store.ensure_session("sess-1", source="user", title="notes")
    store.record_message("sess-1", "assistant", "python >>> profile() showed a slow loop; fix it")
    store.close()

    r = CliRunner().invoke(cli, ["search", "profile", "--json"])
    assert r.exit_code == 0, r.output
    hit = json.loads(r.output)["results"][0]
    snippet = hit["snippet"]
    assert "<<<" not in snippet
    assert snippet.count(">>>") == 1  # only the real REPL prompt survives
    assert "python >>> profile()" in snippet
    assert [snippet[a:b] for a, b in hit["match_ranges"]] == ["profile"]


def test_search_human_has_no_sentinels(tmp_hermes_home: Path):
    store = SqliteFtsStore(db_path=str(tmp_hermes_home / "state.db"))
    store.ensure_session("sess-1", source="user", title="notes")
    store.record_message("sess-1", "assistant", "the dockerfile is ready")
    store.close()
    r = CliRunner().invoke(cli, ["search", "dockerfile"])
    assert r.exit_code == 0, r.output
    assert "dockerfile" in r.output
    assert "<<<" not in r.output and "\x02" not in r.output


# ── #144 ────────────────────────────────────────────────────────────────


def test_once_at_in_the_past_is_rejected(tmp_hermes_home: Path):
    past = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
    with pytest.raises(ValueError, match="in the past"):
        cron_jobs.create_job("oops", f"once at {past}")
    with pytest.raises(ValueError, match="in the past"):
        cron_jobs.create_job("oops", past)
    assert cron_jobs.list_jobs() == []


def test_once_at_past_cli_exits_nonzero(tmp_hermes_home: Path):
    past = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M")
    r = CliRunner().invoke(cli, ["cron", "create", "--prompt", "x", "--schedule", f"once at {past}"])
    assert r.exit_code != 0
    assert "in the past" in r.output


def test_once_at_in_the_future_still_accepted(tmp_hermes_home: Path):
    future = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
    job = cron_jobs.create_job("later", f"once at {future}")
    assert job["schedule"]["kind"] == "once"


# ── #149 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("expr", "display"),
    [
        ("every 90s", "every 90s"),
        ("every 150s", "every 150s"),
        ("every 30s", "every 30s"),
        ("every 5m", "every 5m"),
        ("every 120s", "every 2m"),
        ("every 2h", "every 2h"),
        ("every 90m", "every 90m"),
        ("every 1d", "every 1d"),
        ("45m", "every 45m"),
    ],
)
def test_interval_display_is_exact(expr: str, display: str):
    parsed = cron_jobs.parse_schedule(expr)
    assert parsed["display"] == display
    # and it round-trips to the same cadence
    assert cron_jobs.parse_schedule(display)["seconds"] == parsed["seconds"]


# ── #155 ────────────────────────────────────────────────────────────────


def _run_main(monkeypatch, argv0: str, capsys) -> tuple[str, str]:
    monkeypatch.setattr(sys, "argv", [argv0, "--version"])
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(SystemExit):
        cli_mod.main()
    out = capsys.readouterr()
    return out.out, out.err


@pytest.mark.parametrize("argv0", ["/venv/bin/deepagent-hermes", r"C:\venv\Scripts\deepagent-hermes.exe"])
def test_deprecated_command_warns_and_names_itself(monkeypatch, capsys, argv0):
    monkeypatch.delenv("LANGSTAGE_SUPPRESS_LEGACY_NOTICE", raising=False)
    out, err = _run_main(monkeypatch, argv0, capsys)
    assert "deprecated" in err and "langstage-hermes" in err
    assert out.startswith("deepagent-hermes")


def test_deprecated_command_notice_can_be_silenced(monkeypatch, capsys):
    monkeypatch.setenv("LANGSTAGE_SUPPRESS_LEGACY_NOTICE", "1")
    _out, err = _run_main(monkeypatch, "/venv/bin/deepagent-hermes", capsys)
    assert "deprecated" not in err


def test_canonical_command_is_silent(monkeypatch, capsys):
    monkeypatch.delenv("LANGSTAGE_SUPPRESS_LEGACY_NOTICE", raising=False)
    out, err = _run_main(monkeypatch, "/venv/bin/langstage-hermes", capsys)
    assert "deprecated" not in err
    assert out.startswith("langstage-hermes")
