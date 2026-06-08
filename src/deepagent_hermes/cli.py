"""``deepagent-hermes`` CLI — subcommands + chat REPL + slash-command dispatch.

SPEC §16 reproduction. Built on ``click`` for the same hyphenated subcommand
grammar Hermes ships (``deepagent-hermes chat``, ``... cron list``, ...).

Subcommands:

  - ``chat``             — interactive REPL with slash-command dispatch
  - ``tools``            — list registered toolsets + check status
  - ``skills``           — list / show / install / audit (validates)
  - ``audit``            — log / show / diff / rollback (skill mutation history)
  - ``cron``             — list / create / delete / pause / resume / run-due / daemon
  - ``curator``          — status / run / pause / resume / pin / unpin
  - ``plugins``          — list / enable / disable
  - ``doctor``           — env + permissions sanity check

Top-level ``--show-config`` short-circuits and prints the resolved config
(SPEC §2.acceptance). When no subcommand is given the CLI prints ``--help``.

The ``chat`` command imports the agent module lazily — if ``deepagent_hermes.agent``
isn't integrated yet, we print a helpful message instead of crashing so the
rest of the CLI surface remains usable during partial builds.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

# ── BUILTIN_SLASH_COMMANDS ──────────────────────────────────────────
#
# Module-level so ``plugins.context.register_slash_command`` can detect
# collisions without circular-importing the rest of the CLI.

BUILTIN_SLASH_COMMANDS: dict[str, str] = {
    # v1 essentials (must ship — SPEC §16.2)
    "new": "Start a new conversation (clears messages, keeps config).",
    "reset": "Reset the agent state to a fresh session.",
    "compress": "Force-run context compression.",
    "stop": "Cancel the in-flight turn.",
    "help": "List available slash commands.",
    "quit": "Exit the chat REPL.",
    "exit": "Alias for /quit.",
    "model": "Switch model for subsequent turns: /model anthropic:claude-haiku-4-5",
    "config": "Show the resolved config.",
    "skills": "List/manage skills.",
    "cron": "Pass-through to the cron subcommand.",
    "curator": "Run / inspect the skill curator.",
    "memory": "View MEMORY.md / USER.md snapshots.",
    "tools": "List registered toolsets.",
    "toolsets": "Toggle which toolsets are active for this session.",
    "verbose": "Toggle verbose stream output.",
    "yolo": "Toggle auto-approval of dangerous tool calls.",
    "reload": "Reload plugins / config from disk.",
    # v2 nice-to-haves: stub-only (print a one-liner).
    "rollback": "Rewind to a prior turn (v2).",
    "snapshot": "Save a state snapshot (v2).",
    "queue": "View the input queue (v2).",
    "steer": "Inject a steering message (v2).",
    "voice": "Toggle voice I/O (v2).",
    "skin": "Switch UI skin (v2).",
    "insights": "Show session insights (v2).",
}


# ── banner ─────────────────────────────────────────────────────────
# Pre-rendered FIGlet (font: small) for "hermes". Hard-coded rather than
# generated at startup so we don't pull in pyfiglet as a runtime dep,
# don't pay the render cost on every invocation, and don't risk a font
# lookup failing on a stripped install. Width: 28 cols, height: 4 lines
# — fits any terminal that can run a CLI at all.
_BANNER_ASCII = r""" _
| |_  ___ _ _ _ __  ___ ___
| ' \/ -_) '_| '  \/ -_|_-<
|_||_\___|_| |_|_|_\___/__/"""


def _print_banner(*, tagline: str | None = None) -> None:
    """Render the splash banner — cyan ASCII art over a dim tagline + version.

    Called from the chat REPL on startup and from the bare ``deepagent-hermes``
    invocation (above the help text). Skipped silently if stdout isn't a TTY
    so piped invocations stay clean (``deepagent-hermes --version | grep``,
    test harnesses, etc.).
    """
    try:
        is_tty = sys.stdout.isatty()
    except Exception:
        is_tty = False
    if not is_tty:
        return
    from deepagent_hermes import __version__

    click.echo(click.style(_BANNER_ASCII, fg="cyan", bold=True))
    # ASCII separators — middle-dot looks nicer but mojibakes in cp1252
    # terminals (Windows default before sys.stdout.reconfigure runs).
    sub = tagline or "reflection | skills | memory"
    click.echo(click.style(f"  {sub}  |  v{__version__}", fg="bright_black"))
    click.echo()


def _shorten_path(p: str, *, max_len: int = 56) -> str:
    """Trim long absolute paths to fit on one line — keeps the head + tail,
    ellipsises the middle. Pure cosmetic; never used for any lookup."""
    if len(p) <= max_len:
        return p
    keep = (max_len - 3) // 2
    return f"{p[:keep]}...{p[-keep:]}"


def _print_chat_context(*, cfg: Any, workspace: Any, session_id: str, agent: Any) -> None:
    """Print the per-session config block below the banner.

    Surfaces *what's actually being rendered* — the model, workspace,
    HERMES_HOME, session id, and the (built-in) agent factory. The
    historical chat startup printed just the session id, which left a
    user staring at a prompt with no way to tell which model they were
    burning credits on or where their state.db would land.

    Also surfaces the ``DEEPAGENT_AGENT_SPEC`` env var if set — even
    though this CLI doesn't consume it (it always uses the built-in
    agent factory), the env var is the host-adoption knob other
    deepagent-* surfaces respect. If the user has it set and is
    expecting this REPL to obey it, the line acts as a heads-up that
    it's been read but ignored.
    """
    try:
        is_tty = sys.stdout.isatty()
    except Exception:
        is_tty = False
    if not is_tty:
        return

    # The "agent" the chat REPL is using = the one returned by
    # _try_import_agent — always the built-in factory unless someone
    # patches it. Show its import path so the user knows what they're
    # talking to.
    agent_qualname = (
        f"{getattr(agent, '__module__', '?')}.{getattr(agent, '__name__', '?')}"
        if callable(agent)
        else f"{type(agent).__module__}.{type(agent).__name__}"
    )

    # Model: prefer explicit override on cfg, fall back to default.
    model_id = getattr(cfg, "model_default", None) or "(default: anthropic:claude-sonnet-4-5-20250929)"

    rows: list[tuple[str, str]] = [
        ("agent", agent_qualname),
        ("model", str(model_id)),
        ("workspace", _shorten_path(str(workspace))),
        ("home", _shorten_path(str(getattr(cfg, "hermes_home", "(unset)")))),
        ("session", session_id),
    ]
    spec = os.environ.get("DEEPAGENT_AGENT_SPEC")
    if spec:
        # Flag that this CLI ignores the spec — the spec is consumed by
        # external hosts (deepagent-code / deepagent-lab / cowork-dash).
        rows.append(("spec", f"{spec}  (advisory -- built-in agent used here)"))

    label_w = max(len(k) for k, _ in rows)
    for label, value in rows:
        line = f"  {label:<{label_w}} : {value}"
        click.echo(click.style(line, fg="bright_black"))
    click.echo()


# ── helpers ────────────────────────────────────────────────────────


def _load_config() -> Any:
    """Resolve ``HermesConfig`` (caught here so subcommands stay clean)."""
    from deepagent_hermes.config import HermesConfig

    return HermesConfig.resolve()


def _try_import_agent() -> tuple[Any | None, str | None]:
    """Lazy-import the agent module; return (factory, error_message)."""
    try:
        agent_mod = importlib.import_module("deepagent_hermes.agent")
    except ImportError as e:
        return None, (f"Agent module not yet integrated (import error: {e}). Run 'pytest' to verify subsystems work.")
    factory = getattr(agent_mod, "create_hermes_agent", None) or getattr(agent_mod, "graph", None)
    if factory is None:
        return None, (
            "deepagent_hermes.agent loaded but exposes neither create_hermes_agent "
            "nor a 'graph' symbol. CLI cannot start an agent yet."
        )
    return factory, None


# ── root ───────────────────────────────────────────────────────────


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--show-config",
    is_flag=True,
    help="Print the resolved HermesConfig (defaults < TOML < env < CLI) and exit.",
)
@click.option(
    "--version",
    is_flag=True,
    help="Print the deepagent-hermes version and exit.",
)
@click.pass_context
def cli(ctx: click.Context, show_config: bool, version: bool) -> None:
    """deepagent-hermes — Hermes-style reflection / skill-creation agent."""
    if version:
        from deepagent_hermes import __version__

        click.echo(f"deepagent-hermes {__version__}")
        ctx.exit(0)
    if show_config:
        cfg = _load_config()
        click.echo(cfg.describe())
        ctx.exit(0)
    if ctx.invoked_subcommand is None:
        _print_banner()
        click.echo(ctx.get_help())


# ── chat ───────────────────────────────────────────────────────────


@cli.command()
@click.option("--model", "model_id", default=None, help="Override model for this session.")
@click.option(
    "--workspace",
    "workspace",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Working directory for the agent's filesystem tools.",
)
def chat(model_id: str | None, workspace: str | None) -> None:
    """Interactive chat REPL with slash-command dispatch."""
    factory, err = _try_import_agent()
    if err:
        click.echo(click.style(err, fg="yellow"), err=True)
        click.echo(
            "(The rest of the CLI — cron, plugins, doctor, --show-config — still works.)",
            err=True,
        )
        sys.exit(1)

    cfg = _load_config()
    overrides: dict[str, Any] = {}
    if model_id:
        overrides["model_default"] = model_id
    if workspace:
        overrides["workspace_root"] = workspace
    if overrides:
        from deepagent_hermes.config import HermesConfig

        cfg = HermesConfig.resolve(overrides=overrides)

    try:
        agent = factory(cfg) if callable(factory) else factory
    except Exception as e:
        click.echo(click.style(f"Failed to build agent: {e}", fg="red"), err=True)
        sys.exit(2)

    import uuid

    # One session_id for the entire REPL lifetime so the FTS5 recorder
    # threads turns together; /new can mint a fresh one.
    session_id = getattr(agent, "deepagent_hermes_session_id", None) or f"sess-{uuid.uuid4().hex[:12]}"

    # Session-mutable state surfaced to slash commands.
    state: dict[str, Any] = {
        "messages": [],
        "model_override": model_id,
        "verbose": False,
        "yolo": False,
        "cfg": cfg,
        "session_id": session_id,
        "agent": agent,
    }

    _print_banner(tagline="chat | type /help for commands, /quit to exit")
    _print_chat_context(
        cfg=cfg,
        workspace=workspace or os.getcwd(),
        session_id=session_id,
        agent=factory,
    )
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo()
            break
        if not line:
            continue
        if line.startswith("/"):
            should_exit = _dispatch_slash(line, state)
            if should_exit:
                break
            continue

        _run_agent_turn(agent, line, state)


# ── slash command dispatch ─────────────────────────────────────────


def _dispatch_slash(line: str, state: dict[str, Any]) -> bool:
    """Parse + dispatch a ``/command [args]`` line. Returns ``True`` to exit REPL."""
    parts = line[1:].split(None, 1)
    if not parts:
        click.echo("(empty slash command)")
        return False
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    handler: Callable[[str, dict[str, Any]], bool] | None = _SLASH_HANDLERS.get(name)
    if handler is None:
        # Stub everything not implemented as a stub or built-in.
        if name in BUILTIN_SLASH_COMMANDS:
            click.echo(f"/{name} not implemented in v0.1.0")
            return False
        click.echo(f"Unknown slash command: /{name}. Try /help.")
        return False
    return bool(handler(args, state))


# Individual slash command handlers — each returns ``True`` to terminate REPL.


def _slash_quit(args: str, state: dict[str, Any]) -> bool:
    return True


def _slash_help(args: str, state: dict[str, Any]) -> bool:
    click.echo("Available slash commands:")
    for name in sorted(BUILTIN_SLASH_COMMANDS):
        click.echo(f"  /{name:<10}  {BUILTIN_SLASH_COMMANDS[name]}")
    return False


def _slash_new(args: str, state: dict[str, Any]) -> bool:
    """Fresh session — clears messages AND mints a new session_id so the FTS5
    recorder starts a new lineage. Previous session stays in the store and
    will surface in `session_search` queries.
    """
    import uuid

    state["messages"].clear()
    new_id = f"sess-{uuid.uuid4().hex[:12]}"
    state["session_id"] = new_id
    click.echo(click.style(f"(new session: {new_id})", fg="bright_black"))
    return False


def _slash_reset(args: str, state: dict[str, Any]) -> bool:
    state["messages"].clear()
    state["model_override"] = None
    state["verbose"] = False
    state["yolo"] = False
    click.echo("(state reset — session_id unchanged; use /new to start a fresh session)")
    return False


def _slash_model(args: str, state: dict[str, Any]) -> bool:
    if not args.strip():
        click.echo(f"Current model: {state.get('model_override') or state['cfg'].model_default}")
        return False
    state["model_override"] = args.strip()
    click.echo(f"(model override set: {args.strip()})")
    return False


def _slash_config(args: str, state: dict[str, Any]) -> bool:
    click.echo(state["cfg"].describe())
    return False


def _slash_verbose(args: str, state: dict[str, Any]) -> bool:
    state["verbose"] = not state["verbose"]
    click.echo(f"(verbose = {state['verbose']})")
    return False


def _slash_yolo(args: str, state: dict[str, Any]) -> bool:
    state["yolo"] = not state["yolo"]
    click.echo(f"(yolo = {state['yolo']})")
    return False


def _slash_tools(args: str, state: dict[str, Any]) -> bool:
    """Inline summary of implemented toolsets — full taxonomy via `tools` subcommand."""
    from deepagent_hermes.tools.toolsets import IMPLEMENTED_TOOLSETS, TOOLSETS

    click.echo(click.style("Toolsets (implemented):", fg="cyan"))
    for ts in sorted(IMPLEMENTED_TOOLSETS):
        names = TOOLSETS.get(ts, [])
        click.echo(f"  ● {ts:<18}  {', '.join(names)}")
    stubbed = len(TOOLSETS) - len(IMPLEMENTED_TOOLSETS)
    click.echo(
        click.style(
            f"  (+{stubbed} declared but stubbed — run `deepagent-hermes tools` for the full taxonomy)",
            fg="bright_black",
        )
    )
    return False


def _slash_toolsets(args: str, state: dict[str, Any]) -> bool:
    """Shows which toolsets are currently enabled vs disabled in the loaded config."""
    cfg = state["cfg"]
    from deepagent_hermes.tools.toolsets import IMPLEMENTED_TOOLSETS, resolve_enabled

    disabled = set(cfg.agent_disabled_toolsets or [])
    enabled = resolve_enabled(disabled_toolsets=disabled, platform=os.getenv("HERMES_PLATFORM", "cli"))

    click.echo(click.style("Toolsets (this session):", fg="cyan"))
    for ts in sorted(IMPLEMENTED_TOOLSETS):
        if ts in enabled:
            click.echo(click.style(f"  ✓ {ts}", fg="green"))
        else:
            click.echo(click.style(f"  ✗ {ts}  (disabled in config)", fg="bright_black"))
    if disabled:
        click.echo(click.style(f"\n  agent.disabled_toolsets = {sorted(disabled)}", fg="bright_black"))
    return False


def _slash_skills(args: str, state: dict[str, Any]) -> bool:
    """List skills, or `/skills <query>` to filter, or `/skills show <name>` to view."""
    lib = state.get("skill_lib") or _skill_library()
    state["skill_lib"] = lib
    parts = args.strip().split(maxsplit=1)
    if parts and parts[0] == "show" and len(parts) == 2:
        skill = lib.get(parts[1].strip())
        if not skill:
            click.echo(click.style(f"No skill named {parts[1]!r}.", fg="yellow"))
            return False
        click.echo(click.style(f"# {skill.name}", fg="cyan", bold=True))
        click.echo(click.style(f"  {skill.category or '(uncategorized)'}", fg="bright_black"))
        click.echo()
        click.echo(skill.description)
        click.echo()
        click.echo(click.style("─" * 50, fg="bright_black"))
        click.echo(skill.body[:1500])
        if len(skill.body) > 1500:
            extra = len(skill.body) - 1500
            click.echo(
                click.style(
                    f"\n... (+{extra} more chars — full body via `deepagent-hermes skills show {skill.name}`)",
                    fg="bright_black",
                )
            )
        return False

    query = parts[0] if parts else ""
    items = lib.list()
    if query:
        q = query.lower()
        items = [s for s in items if q in s.name.lower() or q in s.description.lower()]
    if not items:
        click.echo(f"No skills match {query!r}.")
        return False
    click.echo(click.style(f"Skills ({len(items)}):", fg="cyan"))
    for s in sorted(items, key=lambda x: (x.category or "", x.name))[:25]:
        desc = (s.description or "").replace("\n", " ").strip()
        if len(desc) > 60:
            desc = desc[:57] + "..."
        click.echo(f"  · {s.name:<32}  {desc}")
    if len(items) > 25:
        click.echo(click.style(f"  (+{len(items) - 25} more — `/skills <query>` to filter)", fg="bright_black"))
    return False


def _slash_cron(args: str, state: dict[str, Any]) -> bool:
    """List currently-scheduled cron jobs."""
    from deepagent_hermes.cron.jobs import list_jobs

    items = list_jobs()
    if not items:
        click.echo("No cron jobs scheduled. Add one with `deepagent-hermes cron create --prompt ... --schedule ...`.")
        return False
    click.echo(click.style(f"Cron jobs ({len(items)}):", fg="cyan"))
    for job in items:
        flag = "▶" if job.get("enabled", True) and job.get("state") != "paused" else "⏸"
        click.echo(
            f"  {flag} {job['id'][:10]}  {(job.get('name') or '?')[:24]:<24}  "
            f"[{(job.get('schedule_display') or '?')[:16]:<16}]  next={job.get('next_run_at') or '—'}"
        )
    return False


def _slash_curator(args: str, state: dict[str, Any]) -> bool:
    """Inline curator state — same data as `curator status` subcommand."""
    try:
        cstate = _curator_state_get()
    except Exception as e:
        click.echo(click.style(f"(curator state unavailable: {e})", fg="yellow"))
        return False
    cfg = state["cfg"]
    last_run = float(cstate.get("last_run_at") or 0.0)
    last_act = float(cstate.get("last_user_activity") or 0.0)
    paused = bool(cstate.get("paused", False))

    click.echo(click.style("Curator:", fg="cyan"))
    click.echo(f"  enabled={cfg.curator_enabled}  paused={paused}  interval={cfg.curator_interval_hours}h")
    click.echo(f"  last run:      {_fmt_ts(last_run)}")
    click.echo(f"  last activity: {_fmt_ts(last_act)}")
    lib = state.get("skill_lib") or _skill_library()
    state["skill_lib"] = lib
    pinned = [s.name for s in lib.list() if (s.metadata or {}).get("hermes", {}).get("pinned")]
    if pinned:
        click.echo(f"  pinned ({len(pinned)}): {', '.join(pinned[:5])}{'...' if len(pinned) > 5 else ''}")
    return False


def _slash_memory(args: str, state: dict[str, Any]) -> bool:
    """Show MEMORY.md + USER.md current contents (live disk, not snapshot)."""
    from deepagent_hermes.config import hermes_home

    home = hermes_home()
    memory_md = home / "memories" / "MEMORY.md"
    user_md = home / "memories" / "USER.md"
    shown_anything = False
    for label, path in (("USER.md", user_md), ("MEMORY.md", memory_md)):
        if path.exists() and path.stat().st_size > 0:
            shown_anything = True
            content = path.read_text(encoding="utf-8")
            click.echo(click.style(f"{label}  ({path.stat().st_size} bytes)", fg="cyan", bold=True))
            click.echo(click.style(f"  {path}", fg="bright_black"))
            click.echo(content)
            click.echo()
        else:
            click.echo(click.style(f"{label}: empty", fg="bright_black"))
    if not shown_anything:
        click.echo(
            click.style(
                "  Memory grows as the reflection subagent decides things are worth saving,",
                fg="bright_black",
            )
        )
        click.echo(
            click.style(
                '  or by user request ("remember that I prefer X").',
                fg="bright_black",
            )
        )
    return False


def _slash_compress(args: str, state: dict[str, Any]) -> bool:
    """Note: compression is automatic at 50% context — manual trigger is a v0.2 item."""
    click.echo(click.style("Manual compression isn't yet wired (auto-fires at 50% context). v0.2 task.", fg="yellow"))
    return False


def _slash_stop(args: str, state: dict[str, Any]) -> bool:
    click.echo("(No in-flight turn to stop — REPL is synchronous in v0.1.)")
    return False


def _slash_reload(args: str, state: dict[str, Any]) -> bool:
    state["cfg"] = _load_config()
    click.echo("(config reloaded)")
    return False


_SLASH_HANDLERS: dict[str, Callable[[str, dict[str, Any]], bool]] = {
    "quit": _slash_quit,
    "exit": _slash_quit,
    "help": _slash_help,
    "new": _slash_new,
    "reset": _slash_reset,
    "model": _slash_model,
    "config": _slash_config,
    "verbose": _slash_verbose,
    "yolo": _slash_yolo,
    "tools": _slash_tools,
    "toolsets": _slash_toolsets,
    "skills": _slash_skills,
    "cron": _slash_cron,
    "curator": _slash_curator,
    "memory": _slash_memory,
    "compress": _slash_compress,
    "stop": _slash_stop,
    "reload": _slash_reload,
}


# ── agent turn ─────────────────────────────────────────────────────


def _run_agent_turn(agent: Any, user_text: str, state: dict[str, Any]) -> None:
    """Send ``user_text`` through ``agent.stream(...)`` and print via the parser.

    Threads the session id from REPL state so the FTS5 recorder + reflection
    counters cohere across turns. Wraps the parser's ``PrintAdapter`` with a
    prettifier so skill / memory / compression events surface as callouts
    instead of raw ``extracted_type: {...}`` JSON.
    """
    try:
        from langgraph_stream_parser import StreamParser
        from langgraph_stream_parser.adapters import PrintAdapter
        from langgraph_stream_parser.events import ToolExtractedEvent
    except ImportError:
        click.echo("(langgraph-stream-parser missing; printing raw response.)")
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": user_text}]})
            for msg in (result or {}).get("messages", []):
                content = getattr(msg, "content", None) or msg.get("content", "")
                if content:
                    click.echo(content)
        except Exception as e:
            click.echo(click.style(f"Agent invoke failed: {e}", fg="red"))
        return

    parser = StreamParser()
    adapter = PrintAdapter()

    def _pretty_extraction(event: ToolExtractedEvent) -> bool:
        """Render hermes-specific extractor events as callouts. Returns True
        when we consumed the event so the default adapter doesn't double-print.
        """
        et = event.extracted_type
        data = event.data if isinstance(event.data, dict) else {}
        if et == "skill_event":
            sub = data.get("extracted_subtype") or et
            name = data.get("name") or "?"
            verb = {
                "skill_created": "created",
                "skill_updated": "updated",
                "skill_deleted": "deleted",
            }.get(sub, sub)
            click.echo(click.style(f"  ◆ skill {verb}: ", fg="magenta") + click.style(name, fg="bright_magenta", bold=True))
            return True
        if et == "skill_loaded":
            chars = data.get("body_chars", "?")
            click.echo(click.style(f"  ◆ skill loaded into context  ({chars} chars)", fg="magenta"))
            return True
        if et == "memory_updated":
            sub = data.get("extracted_subtype") or et
            target = data.get("target") or "?"
            verb = {
                "memory_added": "added",
                "memory_replaced": "replaced",
                "memory_removed": "removed",
                "memory_read": "read",
            }.get(sub, sub)
            click.echo(click.style(f"  ◆ {target} memory {verb}", fg="cyan"))
            return True
        if et == "compression_summary":
            before = data.get("before_tokens", "?")
            after = data.get("after_tokens", "?")
            ratio = data.get("ratio")
            tail = f"  ({ratio:.1f}x)" if isinstance(ratio, (int, float)) else ""
            click.echo(click.style(f"  ◆ context compressed: {before} → {after}{tail}", fg="yellow"))
            return True
        return False

    try:
        stream = agent.stream(
            {
                "messages": [{"role": "user", "content": user_text}],
                "session_id": state.get("session_id"),
                "model_override": state.get("model_override"),
                "iteration_budget_remaining": state["cfg"].agent_max_iterations,
            },
            config={"configurable": {"thread_id": state.get("session_id")}},
            stream_mode="updates",
        )
        for event in parser.parse(stream):
            if isinstance(event, ToolExtractedEvent) and _pretty_extraction(event):
                continue
            adapter.handle(event)
    except Exception as e:
        click.echo(click.style(f"Agent stream failed: {e}", fg="red"))


# ── tools ──────────────────────────────────────────────────────────


@cli.command()
@click.option("--toolset", "toolset_filter", default=None, help="Show only one toolset.")
@click.option(
    "--implemented-only",
    is_flag=True,
    help="Hide stubbed toolsets (those declared by SPEC §11 but not yet implemented).",
)
def tools(toolset_filter: str | None, implemented_only: bool) -> None:
    """List declared toolsets and the tool names each ships.

    Shows the toolset taxonomy from SPEC §11 — both implemented sets
    (filesystem, skills, memory, etc.) and stubbed-but-declared sets
    (homeassistant, spotify, etc.). For live check-status info, run
    ``deepagent-hermes doctor``.
    """
    from deepagent_hermes.tools.toolsets import IMPLEMENTED_TOOLSETS, TOOLSETS

    declared = sorted(TOOLSETS.keys())
    shown = 0
    for ts in declared:
        if toolset_filter and ts != toolset_filter:
            continue
        is_impl = ts in IMPLEMENTED_TOOLSETS
        if implemented_only and not is_impl:
            continue
        mark = "●" if is_impl else "○"
        names = TOOLSETS[ts]
        header_color = "cyan" if is_impl else "bright_black"
        suffix = "" if is_impl else "  (declared, not implemented in v0.1)"
        click.echo(
            click.style(f"  {mark} {ts:<20}", fg=header_color) + click.style(f" {len(names)} tool(s){suffix}", fg="bright_black")
        )
        for name in names:
            click.echo(click.style(f"      · {name}", fg="bright_black"))
        shown += 1
    if shown == 0:
        if toolset_filter:
            click.echo(f"No toolset named {toolset_filter!r}. Try `deepagent-hermes tools` for all.")
        else:
            click.echo("No toolsets declared.")
        return
    click.echo("")
    click.echo(click.style(f"  ● implemented  ({len(IMPLEMENTED_TOOLSETS)} of {len(TOOLSETS)})", fg="cyan"))
    click.echo(click.style("  ○ declared but stubbed", fg="bright_black"))


# ── skills ─────────────────────────────────────────────────────────


@cli.group()
def skills() -> None:
    """Inspect / install / audit bundled and user skills."""


def _skill_library(*, with_audit: bool = True) -> Any:
    """Build a SkillLibrary from defaults (bundled + user + project).

    By default attaches the per-home audit log so CLI mutations land in
    the same ``skill_mutations`` table the agent writes to. Read-only
    commands (``list``, ``show``) pass ``with_audit=False`` to skip
    SQLite startup.
    """
    from deepagent_hermes.config import hermes_home
    from deepagent_hermes.skills.library import SkillLibrary

    dirs: list[Path] = []
    bundled = Path(__file__).resolve().parent / "_bundled_skills"
    if bundled.is_dir():
        dirs.append(bundled)
    dirs.append(hermes_home() / "skills")
    project = Path.cwd() / ".deepagent-hermes" / "skills"
    if project.is_dir():
        dirs.append(project)
    audit_log = None
    if with_audit:
        from deepagent_hermes.skills.audit import SkillAuditLog

        db_path = hermes_home() / "state.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        audit_log = SkillAuditLog(db_path=str(db_path))
    lib = SkillLibrary(dirs=dirs, audit_log=audit_log)
    if audit_log is not None:
        lib.set_mutation_context(source="cli")
    return lib


def _audit_log() -> Any:
    """Open the per-home skill-mutation log."""
    from deepagent_hermes.config import hermes_home
    from deepagent_hermes.skills.audit import SkillAuditLog

    db_path = hermes_home() / "state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SkillAuditLog(db_path=str(db_path))


@skills.command("list")
@click.option("--category", default=None, help="Filter by category.")
@click.option("--query", default="", help="Substring match against name or description.")
def skills_list(category: str | None, query: str) -> None:
    """List discovered skills (bundled + user)."""
    lib = _skill_library()
    items = lib.list()
    if category:
        items = [s for s in items if (s.category or "") == category]
    if query:
        q = query.lower()
        items = [s for s in items if q in s.name.lower() or q in s.description.lower()]
    if not items:
        click.echo("No skills match.")
        return
    # Group by category.
    by_cat: dict[str, list[Any]] = {}
    for s in items:
        by_cat.setdefault(s.category or "", []).append(s)
    for cat in sorted(by_cat):
        click.echo(click.style(f"\n  {cat or '(uncategorized)'}", fg="cyan"))
        for s in sorted(by_cat[cat], key=lambda x: x.name):
            desc = (s.description or "").replace("\n", " ").strip()
            if len(desc) > 80:
                desc = desc[:77] + "..."
            click.echo(f"    {s.name:<32}  {desc}")
    click.echo(click.style(f"\n  {len(items)} skill(s).", fg="bright_black"))


@skills.command("show")
@click.argument("name")
def skills_show(name: str) -> None:
    """Show full SKILL.md body for ``name``."""
    lib = _skill_library()
    skill = lib.get(name)
    if skill is None:
        click.echo(click.style(f"No skill named {name!r}.", fg="yellow"))
        sys.exit(1)
    click.echo(click.style(f"# {skill.name}", fg="cyan", bold=True))
    click.echo(click.style(f"  category: {skill.category or '(uncategorized)'}", fg="bright_black"))
    click.echo(click.style(f"  path: {skill.path}", fg="bright_black"))
    if skill.version:
        click.echo(click.style(f"  version: {skill.version}", fg="bright_black"))
    click.echo()
    click.echo(skill.description)
    click.echo()
    click.echo(click.style("─" * 60, fg="bright_black"))
    click.echo(skill.body)


@skills.command("install")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def skills_install(path: Path) -> None:
    """Install a skill directory into ``<HERMES_HOME>/skills/``.

    PATH may be either a SKILL.md file or a directory containing one.
    """
    import shutil

    from deepagent_hermes.config import hermes_home
    from deepagent_hermes.skills.validator import validate as validate_frontmatter

    src_dir = path if path.is_dir() else path.parent
    skill_md = src_dir / "SKILL.md"
    if not skill_md.is_file():
        click.echo(click.style(f"No SKILL.md found at {src_dir}.", fg="yellow"))
        sys.exit(2)

    import frontmatter

    post = frontmatter.load(skill_md)
    fm = dict(post.metadata)
    errs = validate_frontmatter(fm, parent_dir_name=src_dir.name)
    if errs:
        click.echo(click.style("SKILL.md frontmatter is invalid:", fg="red"))
        for e in errs:
            click.echo(f"  - {e}")
        sys.exit(2)

    target = hermes_home() / "skills" / src_dir.name
    if target.exists():
        click.echo(click.style(f"Already installed at {target} — won't overwrite.", fg="yellow"))
        sys.exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, target)
    click.echo(click.style(f"Installed {fm.get('name', '?')} → {target}", fg="green"))


@skills.command("audit")
def skills_audit() -> None:
    """Validate every skill against agentskills.io rules."""
    lib = _skill_library()
    errs_by_skill = lib.validate_all()
    if not errs_by_skill:
        n = len(lib.list())
        click.echo(click.style(f"All {n} skill(s) pass validation.", fg="green"))
        return
    n_bad = 0
    for skill_name, errs in errs_by_skill.items():
        if not errs:
            continue
        n_bad += 1
        click.echo(click.style(f"  ✗ {skill_name}", fg="red"))
        for e in errs:
            click.echo(f"      {e}")
    if n_bad == 0:
        click.echo(click.style("All skills pass validation.", fg="green"))
    else:
        click.echo(click.style(f"\n  {n_bad} skill(s) failed validation.", fg="red"))
        sys.exit(1)


# ── audit ──────────────────────────────────────────────────────────


@cli.group()
def audit() -> None:
    """Inspect / roll back the skill mutation log.

    Every ``skill_manage`` action the agent runs and every CLI skill
    mutation appends a row to ``<HERMES_HOME>/state.db``. Use this group
    to see what changed and to revert.
    """


def _format_mutation_row(row: Any, *, full: bool = False) -> str:
    """Render a :class:`MutationRow` for terminal output."""
    from datetime import datetime

    ts = datetime.fromtimestamp(row.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    source = row.source or "?"
    head = f"#{row.id}  {ts}  {row.action:<10s}  {row.skill_name}  ({source})"
    if not full:
        return head
    lines = [head]
    if row.skill_path:
        lines.append(f"    path: {row.skill_path}")
    if row.session_id:
        lines.append(f"    session: {row.session_id}")
    if row.tool_call_id:
        lines.append(f"    tool_call_id: {row.tool_call_id}")
    if row.before_hash:
        lines.append(f"    before sha256: {row.before_hash[:12]}…")
    if row.after_hash:
        lines.append(f"    after  sha256: {row.after_hash[:12]}…")
    return "\n".join(lines)


@audit.command("log")
@click.option("--skill", "skill_name", default=None, help="Restrict to one skill.")
@click.option("--limit", type=int, default=20, show_default=True, help="Max rows to show.")
@click.option("--full", is_flag=True, help="Show full row details (path, hashes, session).")
def audit_log_cmd(skill_name: str | None, limit: int, full: bool) -> None:
    """Print recent skill mutations, most recent first."""
    log = _audit_log()
    rows = log.list(skill_name=skill_name, limit=limit)
    if not rows:
        if skill_name:
            click.echo(f"No mutations recorded for {skill_name!r}.")
        else:
            click.echo("No skill mutations recorded yet.")
        return
    for r in rows:
        click.echo(_format_mutation_row(r, full=full))


@audit.command("show")
@click.argument("mutation_id", type=int)
@click.option("--diff", "show_diff", is_flag=True, help="Show unified diff of before vs after.")
def audit_show_cmd(mutation_id: int, show_diff: bool) -> None:
    """Show full content (or diff) of a single mutation."""
    import difflib

    log = _audit_log()
    row = log.get(mutation_id)
    if row is None:
        click.echo(click.style(f"mutation #{mutation_id} not found", fg="red"))
        sys.exit(1)
    click.echo(_format_mutation_row(row, full=True))
    click.echo("")
    if show_diff:
        before_text = (row.before_content or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        after_text = (row.after_content or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                before_text,
                after_text,
                fromfile=f"{row.skill_name}@before",
                tofile=f"{row.skill_name}@after",
            )
        )
        click.echo(diff if diff else "(no textual change)")
        return
    if row.before_content is not None:
        click.echo(click.style("--- before ---", fg="yellow"))
        click.echo(row.before_content.decode("utf-8", errors="replace"))
    if row.after_content is not None:
        click.echo(click.style("--- after ---", fg="green"))
        click.echo(row.after_content.decode("utf-8", errors="replace"))


@audit.command("diff")
@click.argument("skill_name")
@click.argument("mutation_id", type=int)
def audit_diff_cmd(skill_name: str, mutation_id: int) -> None:
    """Diff the SKILL.md on disk against the recorded ``after_content`` of MUTATION_ID."""
    from deepagent_hermes.skills.audit import RollbackError

    log = _audit_log()
    try:
        diff_text = log.diff_against_disk(skill_name, mutation_id)
    except RollbackError as exc:
        click.echo(click.style(str(exc), fg="red"))
        sys.exit(1)
    if not diff_text:
        click.echo(f"No drift — disk matches mutation #{mutation_id}.")
        return
    click.echo(diff_text)


@audit.command("rollback")
@click.argument("skill_name")
@click.argument("mutation_id", type=int)
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt (use in scripts).")
def audit_rollback_cmd(skill_name: str, mutation_id: int, yes: bool) -> None:
    """Restore SKILL_NAME to the state recorded *before* MUTATION_ID.

    The rollback itself is logged as a new mutation (action=``rollback``),
    so the audit history remains complete. Get the MUTATION_ID from
    ``hermes audit log``.
    """
    from deepagent_hermes.skills.audit import RollbackError

    log = _audit_log()
    target = log.get(mutation_id)
    if target is None:
        click.echo(click.style(f"mutation #{mutation_id} not found", fg="red"))
        sys.exit(1)
    if target.skill_name != skill_name:
        click.echo(
            click.style(
                f"mutation #{mutation_id} is for skill {target.skill_name!r}, not {skill_name!r}",
                fg="red",
            )
        )
        sys.exit(1)
    if target.before_content is None:
        click.echo(
            click.style(
                f"mutation #{mutation_id} was a {target.action} with no before-state — nothing to roll back to.",
                fg="red",
            )
        )
        sys.exit(1)
    if not yes:
        click.echo(_format_mutation_row(target, full=True))
        click.echo(
            click.style(
                f"\nThis will rewrite {target.skill_path} with its pre-mutation content.",
                fg="yellow",
            )
        )
        if not click.confirm("Proceed?", default=False):
            click.echo("Cancelled.")
            return
    try:
        path = log.rollback_to(skill_name, mutation_id, source="cli")
    except RollbackError as exc:
        click.echo(click.style(f"rollback failed: {exc}", fg="red"))
        sys.exit(1)
    click.echo(click.style(f"Rolled back {skill_name} → {path}", fg="green"))


# ── cron ───────────────────────────────────────────────────────────


@cli.group()
def cron() -> None:
    """Manage scheduled cron jobs (SPEC §14)."""


@cron.command("list")
def cron_list() -> None:
    """List all scheduled cron jobs."""
    from deepagent_hermes.cron.jobs import list_jobs

    items = list_jobs()
    if not items:
        click.echo("No cron jobs scheduled.")
        return
    for job in items:
        click.echo(
            f"  {job['id']}  {job.get('name', '?'):<30} "
            f"[{job.get('schedule_display', '?'):<20}] "
            f"state={job.get('state', '?'):<10} "
            f"next={job.get('next_run_at') or '—'}"
        )


@cron.command("create")
@click.option("--prompt", "prompt", required=True, help="Prompt to run on schedule.")
@click.option("--schedule", "schedule_expr", required=True, help="Schedule expression.")
@click.option("--name", default=None, help="Friendly name (defaults to first 50 chars of prompt).")
@click.option("--model", default=None, help="Per-job model override.")
def cron_create(prompt: str, schedule_expr: str, name: str | None, model: str | None) -> None:
    """Create a new cron job."""
    from deepagent_hermes.cron.jobs import create_job

    try:
        job = create_job(prompt, schedule_expr, name=name, model=model)
    except ValueError as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(2)
    click.echo(f"Created cron job {job['id']} ({job['name']}); next run {job['next_run_at']}.")


@cron.command("delete")
@click.argument("id")
def cron_delete(id: str) -> None:
    """Delete a cron job by ID."""
    from deepagent_hermes.cron.jobs import delete_job

    click.echo("Deleted." if delete_job(id) else f"No cron job with id {id!r}.")


@cron.command("pause")
@click.argument("id")
@click.option("--reason", default="", help="Optional reason recorded in paused_reason.")
def cron_pause(id: str, reason: str) -> None:
    """Pause a cron job (disables without deleting)."""
    from deepagent_hermes.cron.jobs import pause_job

    click.echo("Paused." if pause_job(id, reason) else f"No cron job with id {id!r}.")


@cron.command("resume")
@click.argument("id")
def cron_resume(id: str) -> None:
    """Resume a paused cron job."""
    from deepagent_hermes.cron.jobs import resume_job

    click.echo("Resumed." if resume_job(id) else f"No cron job with id {id!r}.")


@cron.command("run-due")
def cron_run_due() -> None:
    """Run a single tick — execute every due job and exit."""
    from deepagent_hermes.cron.scheduler import HermesCron

    results = HermesCron().tick()
    click.echo(f"Tick complete: {len(results)} job(s) run.")
    for r in results:
        click.echo(f"  {r['job_id']}: {'ok' if r['success'] else 'error'}")


@cron.command("daemon")
def cron_daemon() -> None:
    """Run the cron daemon forever (alias for ``python -m deepagent_hermes.cron``)."""
    from deepagent_hermes.cron.__main__ import main as cron_main

    sys.exit(cron_main())


# ── curator ────────────────────────────────────────────────────────


@cli.group()
def curator() -> None:
    """Skill curator lifecycle controls (SPEC §9 + §10)."""


def _curator_store() -> Any:
    """Open the SQLite store at ``<HERMES_HOME>/state.db``."""
    from deepagent_hermes.config import hermes_home
    from deepagent_hermes.store.sqlite_fts import SqliteFtsStore

    db_path = hermes_home() / "state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteFtsStore(db_path=str(db_path))


def _curator_state_get() -> dict[str, Any]:
    from deepagent_hermes.curator import _load_curator_state

    return _load_curator_state(_curator_store())


def _curator_state_save(state: dict[str, Any]) -> None:
    from deepagent_hermes.curator import _save_curator_state

    _save_curator_state(_curator_store(), state)


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "—"
    import datetime as dt

    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


@curator.command("status")
def curator_status() -> None:
    """Print curator state (last run, last activity, paused flag, pinned skills)."""
    state = _curator_state_get()
    cfg = _load_config()
    last_run = float(state.get("last_run_at") or 0.0)
    last_act = float(state.get("last_user_activity") or 0.0)
    paused = bool(state.get("paused", False))
    interval_s = cfg.curator_interval_hours * 3600

    click.echo(click.style("Curator state", fg="cyan", bold=True))
    click.echo(f"  enabled:           {cfg.curator_enabled}")
    click.echo(f"  paused:            {paused}")
    click.echo(f"  interval:          {cfg.curator_interval_hours} h ({cfg.curator_interval_hours / 24:.1f} d)")
    click.echo(f"  min idle:          {cfg.curator_min_idle_hours} h")
    click.echo(f"  stale after:       {cfg.curator_stale_after_days} d")
    click.echo(f"  archive after:     {cfg.curator_archive_after_days} d")
    click.echo(f"  last_run_at:       {_fmt_ts(last_run)}")
    click.echo(f"  last_user_act.:    {_fmt_ts(last_act)}")

    import time

    now = time.time()
    if last_run > 0:
        next_run = last_run + interval_s
        delta_h = (next_run - now) / 3600
        if delta_h > 0:
            click.echo(click.style(f"  next eligible:     {_fmt_ts(next_run)}  (~{delta_h:.1f}h)", fg="bright_black"))
        else:
            click.echo(click.style(f"  next eligible:     overdue by {-delta_h:.1f}h", fg="yellow"))

    # Pinned skills
    lib = _skill_library()
    pinned = [s for s in lib.list() if (s.metadata or {}).get("hermes", {}).get("pinned")]
    if pinned:
        click.echo(click.style(f"\nPinned skills ({len(pinned)}):", fg="cyan"))
        for s in sorted(pinned, key=lambda x: x.name):
            click.echo(f"  · {s.name}")
    else:
        click.echo(click.style("\nPinned skills: none", fg="bright_black"))


@curator.command("run")
@click.option("--dry-run", is_flag=True, help="Print proposed actions without writing.")
def curator_run(dry_run: bool) -> None:
    """Manually run a curator lifecycle pass now (mark stale + archive).

    Bypasses the interval + idle gates — useful for one-off cleanup.
    """
    from deepagent_hermes.curator import mark_stale_and_archive

    cfg = _load_config()
    lib = _skill_library()
    if dry_run:
        # Run against a no-op library wrapper so nothing persists.
        class _Wrap:
            def __init__(self, inner: Any) -> None:
                self._i = inner

            def list(self) -> Any:
                return self._i.list()

            def write(self, *a: Any, **k: Any) -> Any:
                return None

            def delete(self, *a: Any, **k: Any) -> Any:
                return True

        result = mark_stale_and_archive(
            _Wrap(lib),
            stale_days=cfg.curator_stale_after_days,
            archive_days=cfg.curator_archive_after_days,
        )
        click.echo(click.style("(dry-run — no changes written)", fg="yellow"))
    else:
        result = mark_stale_and_archive(
            lib,
            stale_days=cfg.curator_stale_after_days,
            archive_days=cfg.curator_archive_after_days,
        )
        import time

        state = _curator_state_get()
        state["last_run_at"] = time.time()
        _curator_state_save(state)

    click.echo(click.style("Curator pass", fg="cyan", bold=True))
    for key in ("marked_stale", "archived", "skipped_pinned"):
        names = result.get(key, [])
        label = key.replace("_", " ")
        if names:
            click.echo(f"  {label} ({len(names)}):")
            for n in names:
                click.echo(f"    · {n}")
        else:
            click.echo(click.style(f"  {label}: none", fg="bright_black"))


@curator.command("pause")
def curator_pause() -> None:
    """Pause the curator's scheduled runs (status flag only)."""
    state = _curator_state_get()
    state["paused"] = True
    _curator_state_save(state)
    click.echo("Curator paused. Resume with `curator resume`.")


@curator.command("resume")
def curator_resume() -> None:
    """Resume the curator's scheduled runs."""
    state = _curator_state_get()
    state["paused"] = False
    _curator_state_save(state)
    click.echo("Curator resumed.")


def _set_pinned(name: str, value: bool) -> int:
    import frontmatter

    lib = _skill_library()
    skill = lib.get(name)
    if skill is None:
        click.echo(click.style(f"No skill named {name!r}.", fg="yellow"))
        return 1
    post = frontmatter.load(skill.path)
    fm = dict(post.metadata)
    hermes_meta = dict(fm.get("hermes") or {})
    if value:
        hermes_meta["pinned"] = True
    else:
        hermes_meta.pop("pinned", None)
    if hermes_meta:
        fm["hermes"] = hermes_meta
    elif "hermes" in fm:
        del fm["hermes"]
    post.metadata = fm
    skill.path.write_text(frontmatter.dumps(post), encoding="utf-8")
    verb = "pinned" if value else "unpinned"
    click.echo(click.style(f"{verb}: {name}", fg="green"))
    return 0


@curator.command("pin")
@click.argument("name")
def curator_pin(name: str) -> None:
    """Pin skill ``name`` so the curator never archives it."""
    sys.exit(_set_pinned(name, True))


@curator.command("unpin")
@click.argument("name")
def curator_unpin(name: str) -> None:
    """Unpin skill ``name`` so it's eligible for curator lifecycle."""
    sys.exit(_set_pinned(name, False))


# ── plugins ────────────────────────────────────────────────────────


@cli.group()
def plugins() -> None:
    """Discover / enable / disable plugins (SPEC §15)."""


@plugins.command("list")
def plugins_list() -> None:
    """List discovered plugins from all four sources."""
    from deepagent_hermes.plugins.loader import HermesPluginLoader

    cfg = _load_config()
    loader = HermesPluginLoader(
        enabled=cfg.plugins_enabled or None,
        disabled=cfg.plugins_disabled,
    )
    loaded = loader.discover()
    if not loaded:
        click.echo("No plugins discovered.")
        return
    for p in loaded:
        flag = "on " if p.enabled else "off"
        click.echo(
            f"  [{flag}] {p.name:<24} ({p.source:<11}) "
            f"v{p.version or '?':<8} {p.description or ''}" + (f"  — {p.error}" if p.error else "")
        )


@plugins.command("enable")
@click.argument("name")
def plugins_enable(name: str) -> None:
    """Add ``name`` to ``[plugins.enabled]`` in user TOML (TBD)."""
    click.echo(f"(Plugin enable for {name!r} TBD — edit deepagent-hermes.toml.)")


@plugins.command("disable")
@click.argument("name")
def plugins_disable(name: str) -> None:
    """Add ``name`` to ``[plugins.disabled]`` in user TOML (TBD)."""
    click.echo(f"(Plugin disable for {name!r} TBD — edit deepagent-hermes.toml.)")


# ── doctor ─────────────────────────────────────────────────────────


@cli.command()
@click.option("--model", "model_id", default=None, help="Override model for this verify run.")
def verify(model_id: str | None) -> None:
    """End-to-end smoke: one live model round-trip + write the side effects.

    Catches the kinds of fresh-install problems ``doctor`` misses — packaging
    bugs that hide bundled prompts or skills, an API key that the SDK can't
    actually authenticate with, FTS5 init failing, etc. If this passes, the
    chat REPL will work.

    Honest about cost: this makes a single model call (one prompt, ≤20 tokens
    of output). On the default Anthropic Sonnet 4.5 that's a fraction of a
    cent; on gpt-4o-mini via OpenRouter it's effectively free.
    """
    import tempfile
    import time
    from pathlib import Path

    from deepagent_hermes.config import HermesConfig, hermes_home

    click.echo(click.style("deepagent-hermes verify — live end-to-end smoke", fg="cyan", bold=True))
    click.echo()

    # ── (1) packaging / bundled assets ───────────────────────────────────
    pkg_dir = Path(__file__).resolve().parent
    prompts_dir = pkg_dir / "_prompts"
    skills_dir = pkg_dir / "_bundled_skills"
    needed_prompts = [
        "default_identity.md",
        "combined_review.md",
        "skill_review.md",
        "memory_review.md",
        "compression_summary.md",
    ]
    missing = [p for p in needed_prompts if not (prompts_dir / p).is_file()]
    if missing:
        click.echo(click.style(f"  ✗ bundled prompts missing: {missing}", fg="red"))
        click.echo(click.style(f"    expected under {prompts_dir}", fg="bright_black"))
        click.echo(click.style("    this is the v0.1.0/v0.1.1 packaging bug — upgrade to v0.1.2+", fg="yellow"))
        sys.exit(2)
    click.echo(click.style(f"  ✓ bundled prompts ({len(needed_prompts)} critical) present", fg="green"))

    n_bundled = sum(1 for _ in skills_dir.rglob("SKILL.md")) if skills_dir.is_dir() else 0
    if n_bundled == 0:
        click.echo(click.style("  ⚠ no bundled SKILL.md files found", fg="yellow"))
        click.echo(
            click.style(
                f"    expected at {skills_dir} — agent will still run with an empty library",
                fg="bright_black",
            )
        )
    else:
        click.echo(click.style(f"  ✓ bundled skills: {n_bundled} SKILL.md files", fg="green"))

    # ── (2) HERMES_HOME writability ──────────────────────────────────────
    home = hermes_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".verify_write_test"
        probe.write_text("ok")
        probe.unlink()
        click.echo(click.style(f"  ✓ HERMES_HOME writable ({home})", fg="green"))
    except OSError as e:
        click.echo(click.style(f"  ✗ HERMES_HOME not writable ({home}): {e}", fg="red"))
        sys.exit(2)

    # ── (3) model API key sanity ─────────────────────────────────────────
    cfg = HermesConfig.resolve()
    model_for_run = model_id or cfg.model_default
    click.echo(click.style(f"  · model:  {model_for_run}", fg="bright_black"))
    if model_for_run.startswith("anthropic:") and not os.getenv("ANTHROPIC_API_KEY"):
        click.echo(click.style("  ✗ model is anthropic:* but ANTHROPIC_API_KEY not set", fg="red"))
        sys.exit(2)
    if model_for_run.startswith("openai:") and not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")):
        click.echo(click.style("  ✗ model is openai:* but neither OPENAI_API_KEY nor OPENROUTER_API_KEY set", fg="red"))
        sys.exit(2)

    # ── (4) build the agent in an isolated workspace ─────────────────────
    workspace = Path(tempfile.mkdtemp(prefix="dah-verify-"))
    click.echo(click.style(f"  · isolated workspace: {workspace}", fg="bright_black"))

    overrides: dict[str, Any] = {}
    if model_id:
        overrides["model_default"] = model_id

    t0 = time.perf_counter()
    try:
        cfg_for_run = HermesConfig.resolve(overrides=overrides) if overrides else cfg
        from deepagent_hermes import create_hermes_agent

        agent = create_hermes_agent(cfg_for_run, workspace=workspace, session_id="verify-001")
    except Exception as e:
        click.echo(click.style(f"  ✗ agent build failed: {type(e).__name__}: {e}", fg="red"))
        sys.exit(2)
    build_s = time.perf_counter() - t0
    click.echo(click.style(f"  ✓ agent built in {build_s:.1f}s", fg="green"))

    # ── (5) one real round-trip ──────────────────────────────────────────
    click.echo(click.style("  · invoking model with 'reply yes/no: are you working?' ...", fg="bright_black"))
    t0 = time.perf_counter()
    try:
        result = agent.invoke(
            {
                "messages": [{"role": "user", "content": "Reply with one word, either YES or NO: are you working?"}],
                "session_id": "verify-001",
                "iteration_budget_remaining": 5,
            },
            config={"configurable": {"thread_id": "verify-001"}},
        )
    except Exception as e:
        click.echo(click.style(f"  ✗ model invoke failed: {type(e).__name__}: {e}", fg="red"))
        click.echo(click.style("    (auth error? rate limit? check the API key + model id)", fg="bright_black"))
        sys.exit(2)
    invoke_s = time.perf_counter() - t0

    msgs = result.get("messages", [])
    last_ai = next((m for m in reversed(msgs) if getattr(m, "type", None) == "ai"), None)
    content = getattr(last_ai, "content", "")
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    if not content:
        click.echo(click.style("  ✗ model returned an empty response", fg="red"))
        sys.exit(2)
    click.echo(click.style(f"  ✓ model round-trip in {invoke_s:.1f}s: {content.strip()[:80]!r}", fg="green"))

    # ── (6) FTS5 store wrote the turn ────────────────────────────────────
    import sqlite3

    db = home / "state.db"
    if not db.exists():
        click.echo(click.style("  ✗ FTS5 store wasn't created", fg="red"))
        sys.exit(2)
    conn = sqlite3.connect(str(db))
    try:
        n_sess = conn.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", ("verify-001",)).fetchone()[0]
        n_msgs = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", ("verify-001",)).fetchone()[0]
    finally:
        conn.close()
    if n_sess == 0:
        click.echo(click.style("  ⚠ session row not recorded in FTS5 store", fg="yellow"))
    else:
        click.echo(click.style(f"  ✓ FTS5 store: {n_sess} session(s), {n_msgs} message(s) recorded", fg="green"))

    click.echo()
    click.echo(click.style("VERIFY: PASS — runtime is wired end-to-end.", fg="green", bold=True))
    click.echo(
        click.style(
            "  next: `deepagent-hermes chat`  (or set DEEPAGENT_AGENT_SPEC=deepagent_hermes.agent:graph in a host)",
            fg="bright_black",
        )
    )


@cli.command()
def doctor() -> None:
    """Sanity check: Python version, deps, env vars, HERMES_HOME writability."""
    from deepagent_hermes.config import hermes_home

    click.echo("deepagent-hermes doctor:")
    click.echo(f"  python: {sys.version.split()[0]} (need >= 3.11)")
    py_ok = sys.version_info >= (3, 11)
    click.echo(f"    {'OK' if py_ok else 'FAIL'}")

    try:
        import langgraph_stream_parser  # noqa: F401

        click.echo("  langgraph-stream-parser: installed")
    except ImportError as e:
        click.echo(f"  langgraph-stream-parser: MISSING ({e})")

    if os.getenv("ANTHROPIC_API_KEY"):
        click.echo("  ANTHROPIC_API_KEY: set")
    else:
        click.echo("  ANTHROPIC_API_KEY: not set (required for the default anthropic:* model)")

    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if os.getenv(var):
            click.echo(f"  {var}: set")

    home = hermes_home()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".doctor_write_test"
        probe.write_text("ok")
        probe.unlink()
        click.echo(f"  HERMES_HOME ({home}): writable")
    except OSError as e:
        click.echo(f"  HERMES_HOME ({home}): NOT writable — {e}")

    cron_dir = home / "cron"
    click.echo(f"  cron dir: {'exists' if cron_dir.exists() else 'absent (will be created on first use)'}")
    click.echo(f"  shutil.which('bash'): {shutil.which('bash') or 'not on PATH (no_agent shell scripts will fail)'}")


# ── entry point ────────────────────────────────────────────────────


def main() -> None:
    """Console-script entry point (``deepagent-hermes`` in pyproject)."""
    # On Windows the default stdout codec is cp1252; that chokes on the
    # unicode characters bundled skills + agent responses routinely contain
    # (em-dashes, arrows, the section sign). Reconfigure to UTF-8 with a
    # `replace` errors handler so we never crash the CLI on a bad byte.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    cli(prog_name="deepagent-hermes")


if __name__ == "__main__":
    main()


__all__ = ["BUILTIN_SLASH_COMMANDS", "cli", "main"]
