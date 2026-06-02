"""``deepagent-hermes`` CLI — subcommands + chat REPL + slash-command dispatch.

SPEC §16 reproduction. Built on ``click`` for the same hyphenated subcommand
grammar Hermes ships (``deepagent-hermes chat``, ``... cron list``, ...).

Subcommands:

  - ``chat``             — interactive REPL with slash-command dispatch
  - ``tools``            — list registered toolsets + check status
  - ``skills``           — list / show / install / audit
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
from pathlib import Path
from typing import Any, Callable

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
        return None, (
            "Agent module not yet integrated "
            f"(import error: {e}). Run 'pytest' to verify subsystems work."
        )
    factory = getattr(agent_mod, "create_hermes_agent", None) or getattr(
        agent_mod, "graph", None
    )
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

    # Session-mutable state surfaced to slash commands.
    state: dict[str, Any] = {
        "messages": [],
        "model_override": model_id,
        "verbose": False,
        "yolo": False,
        "cfg": cfg,
    }

    click.echo("deepagent-hermes chat — type /help for commands, /quit to exit.")
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
    state["messages"].clear()
    click.echo("(new session — messages cleared)")
    return False


def _slash_reset(args: str, state: dict[str, Any]) -> bool:
    state["messages"].clear()
    state["model_override"] = None
    state["verbose"] = False
    state["yolo"] = False
    click.echo("(state reset)")
    return False


def _slash_model(args: str, state: dict[str, Any]) -> bool:
    if not args.strip():
        click.echo(
            "Current model: "
            f"{state.get('model_override') or state['cfg'].model_default}"
        )
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
    click.echo("(Tool list TBD — see `deepagent-hermes tools` subcommand.)")
    return False


def _slash_toolsets(args: str, state: dict[str, Any]) -> bool:
    click.echo("(Toolset toggling TBD — configure via deepagent-hermes.toml.)")
    return False


def _slash_skills(args: str, state: dict[str, Any]) -> bool:
    click.echo("(Use `deepagent-hermes skills list` for the bundled skills.)")
    return False


def _slash_cron(args: str, state: dict[str, Any]) -> bool:
    click.echo("(Use `deepagent-hermes cron list` for scheduled jobs.)")
    return False


def _slash_curator(args: str, state: dict[str, Any]) -> bool:
    click.echo("(Use `deepagent-hermes curator status` for curator info.)")
    return False


def _slash_memory(args: str, state: dict[str, Any]) -> bool:
    click.echo("(MEMORY.md / USER.md viewing TBD in v0.1.)")
    return False


def _slash_compress(args: str, state: dict[str, Any]) -> bool:
    click.echo("(Manual /compress TBD — auto-fires at 50% context.)")
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
    """Send ``user_text`` through ``agent.stream(...)`` and print via the parser."""
    try:
        from langgraph_stream_parser import StreamParser
        from langgraph_stream_parser.adapters import PrintAdapter
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
    try:
        stream = agent.stream(
            {
                "messages": [{"role": "user", "content": user_text}],
                "model_override": state.get("model_override"),
            },
            stream_mode="updates",
        )
        for event in parser.parse(stream):
            adapter.handle(event)
    except Exception as e:
        click.echo(click.style(f"Agent stream failed: {e}", fg="red"))


# ── tools ──────────────────────────────────────────────────────────


@cli.command()
def tools() -> None:
    """List registered toolsets + their check status (cached 30s)."""
    try:
        from deepagent_hermes.tools.registry import HermesToolRegistry  # noqa: F401

        click.echo("(Tool registry inspection not yet wired in v0.1.)")
    except ImportError:
        click.echo("(deepagent_hermes.tools.registry not yet built — see SPEC §11.)")


# ── skills ─────────────────────────────────────────────────────────


@cli.group()
def skills() -> None:
    """Inspect / install / audit bundled and user skills."""


@skills.command("list")
def skills_list() -> None:
    """List discovered skills."""
    click.echo("(Skill library not yet wired in v0.1 — see SPEC §10.)")


@skills.command("show")
@click.argument("name")
def skills_show(name: str) -> None:
    """Show full SKILL.md body for ``name``."""
    click.echo(f"(Skill show for {name!r} TBD — see SPEC §10.)")


@skills.command("install")
@click.argument("path", type=click.Path(exists=True))
def skills_install(path: str) -> None:
    """Install a skill directory into ``<HERMES_HOME>/skills/``."""
    click.echo(f"(Skill install from {path!r} TBD — see SPEC §10.)")


@skills.command("audit")
def skills_audit() -> None:
    """Validate every skill against agentskills.io rules."""
    click.echo("(Skill audit TBD — see SPEC §10.)")


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
            f"  {job['id']}  {job.get('name','?'):<30} "
            f"[{job.get('schedule_display','?'):<20}] "
            f"state={job.get('state','?'):<10} "
            f"next={job.get('next_run_at') or '—'}"
        )


@cron.command("create")
@click.option("--prompt", "prompt", required=True, help="Prompt to run on schedule.")
@click.option("--schedule", "schedule_expr", required=True, help="Schedule expression.")
@click.option("--name", default=None, help="Friendly name (defaults to first 50 chars of prompt).")
@click.option("--model", default=None, help="Per-job model override.")
def cron_create(
    prompt: str, schedule_expr: str, name: str | None, model: str | None
) -> None:
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


@curator.command("status")
def curator_status() -> None:
    """Print curator state (last run, next run, pinned skills)."""
    click.echo("(Curator status TBD — see SPEC §9.4.)")


@curator.command("run")
def curator_run() -> None:
    """Manually run a curator pass now."""
    click.echo("(Manual curator run TBD — see SPEC §9.4.)")


@curator.command("pause")
def curator_pause() -> None:
    """Pause the curator's scheduled runs."""
    click.echo("(Curator pause TBD.)")


@curator.command("resume")
def curator_resume() -> None:
    """Resume the curator's scheduled runs."""
    click.echo("(Curator resume TBD.)")


@curator.command("pin")
@click.argument("name")
def curator_pin(name: str) -> None:
    """Pin skill ``name`` so the curator never archives it."""
    click.echo(f"(Pin {name!r} TBD.)")


@curator.command("unpin")
@click.argument("name")
def curator_unpin(name: str) -> None:
    """Unpin skill ``name`` so it's eligible for curator lifecycle."""
    click.echo(f"(Unpin {name!r} TBD.)")


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
            f"v{p.version or '?':<8} {p.description or ''}"
            + (f"  — {p.error}" if p.error else "")
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
        click.echo("  ANTHROPIC_API_KEY: not set (required for anthropic:* models)")

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
    click.echo(
        f"  cron dir: {'exists' if cron_dir.exists() else 'absent (will be created on first use)'}"
    )
    click.echo(
        f"  shutil.which('bash'): {shutil.which('bash') or 'not on PATH (no_agent shell scripts will fail)'}"
    )


# ── entry point ────────────────────────────────────────────────────


def main() -> None:
    """Console-script entry point (``deepagent-hermes`` in pyproject)."""
    cli(prog_name="deepagent-hermes")


if __name__ == "__main__":
    main()


__all__ = ["BUILTIN_SLASH_COMMANDS", "cli", "main"]
