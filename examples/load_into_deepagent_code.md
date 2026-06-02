# Running `deepagent-hermes` inside `deepagent-code`

The four `deepagent-*` hosts (`code`, `lab`, `vscode`, `cowork-dash`) all read
the agent factory from the `DEEPAGENT_AGENT_SPEC` environment variable. Point
that at `deepagent-hermes`'s exported graph and the host loads it unchanged —
no host code edits.

## One-liner

```bash
# from the deepagent-code repo
export DEEPAGENT_AGENT_SPEC="deepagent_hermes.agent:graph"
deepagent-code chat
```

PowerShell equivalent:

```powershell
$env:DEEPAGENT_AGENT_SPEC = "deepagent_hermes.agent:graph"
deepagent-code chat
```

## What this does

1. The host imports `deepagent_hermes.agent` and looks for the symbol named
   `graph` (a compiled `StateGraph` built by `create_hermes_agent(HermesConfig.resolve())`).
2. `deepagent-code`'s CLI wires its existing `StreamParser` + adapters around
   that graph — no host-side changes needed.
3. All `deepagent-hermes` middleware (reflection, skill loader, memory tools,
   cron tool, etc.) runs inside the host's chat loop.

## Configuration precedence

Both projects share the `DEEPAGENT_*` env-var prefix; `deepagent-hermes` adds
its own `DEEPAGENT_HERMES_*` prefix. The resolution chain is:

```
defaults
  < deepagents.toml
  < ~/.deepagent-hermes/config.toml
  < ./deepagent-hermes.toml
  < DEEPAGENT_* env
  < DEEPAGENT_HERMES_* env
  < CLI flags passed to the host
```

So you can set the model globally with:

```bash
export DEEPAGENT_HERMES_MODEL_DEFAULT="anthropic:claude-haiku-4-5-20251001"
```

and the agent loaded inside `deepagent-code` will pick it up.

## Persistent install (recommended)

In `deepagent-code`'s venv:

```bash
pip install -e "C:/Users/Kedar/Documents/Code/deepagent-hermes[dev]"
```

Then either set the env var per-session as above, or bake it into your shell
profile so every `deepagent-code` invocation defaults to the Hermes agent.

## Switching back

```bash
unset DEEPAGENT_AGENT_SPEC          # bash/zsh
Remove-Item Env:\DEEPAGENT_AGENT_SPEC  # PowerShell
```

`deepagent-code` falls back to its own bundled default agent when the env
var is unset.

## Smoke test from this repo

```bash
python examples/cli_smoke.py
```

If that prints `Hello.` (or similar) you have a working Hermes agent that
will load into any of the four hosts.
