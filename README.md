# deepagent-hermes

A faithful reproduction of [Nous Research's Hermes Agent](https://github.com/nousresearch/hermes-agent) on top of LangGraph + [`deepagents`](https://github.com/langchain-ai/deepagents) + [`langgraph-stream-parser`](https://github.com/dkedar7/langgraph-stream-parser).

**Status: pre-alpha (v0.1.0a0).** Spec at [SPEC.md](./SPEC.md). Code is scaffolded; most subsystems work end-to-end but the bundled-skills library is empty and 5 of 6 terminal backends are stubs.

## What it is

A `deepagents`-built agent with a **closed reflection→skill-creation loop**:

- After ~10 tool-using iterations, a review subagent runs in the background, writes/patches a `SKILL.md` capturing the pattern it just exercised, and ships it to a skill library.
- Next session, the agent reads the library at startup, sees the new skill's description in its system prompt, and can `skill_view(name)` to load the full body on demand (progressive disclosure per the [agentskills.io spec](https://agentskills.io/specification.md)).
- A weekly **curator** consolidates skills into umbrellas and archives stale ones.
- A **frozen-snapshot memory** (`MEMORY.md` + `USER.md`) preserves prefix-cache hits for the entire session.
- **FTS5 session search** indexes every past conversation in a local SQLite DB.
- Optional **Honcho user model** for dialectic cross-session user profiling.

Designed to be loaded into the existing `deepagent-*` host family ([`deepagent-code`](https://github.com/dkedar7/deepagent-code), [`deepagent-lab`](https://github.com/dkedar7/deepagent-lab), [`cowork-dash`](https://github.com/dkedar7/cowork-dash), [`deepagent-vscode`](https://github.com/dkedar7/deepagent-vscode)) without UI changes — set `DEEPAGENT_AGENT_SPEC=deepagent_hermes.agent:graph` in any of them.

## Installation

```bash
# central venv per the deepagent-* convention
uv venv "$env:USERPROFILE\.venvs\deepagent-hermes"
. "$env:USERPROFILE\.venvs\deepagent-hermes\Scripts\Activate.ps1"

# while langgraph-stream-parser v0.2.0 is unreleased, install editable
uv pip install -e "..\langgraph-stream-parser"

uv pip install -e .

# optional extras
uv pip install -e ".[honcho]"     # Honcho user-model provider
uv pip install -e ".[modal]"      # Modal sandbox backend
uv pip install -e ".[daytona]"    # Daytona sandbox backend
uv pip install -e ".[dev]"        # tests + lint
```

## Quick start

```bash
# show resolved config + sources
deepagent-hermes --show-config

# interactive chat
deepagent-hermes chat

# from inside chat:
#   /skills            list available skills
#   /model anthropic:claude-haiku-4-5-20251001    switch models
#   /memory            dump current memory snapshot
#   /compress          force context compression
#   /quit
```

## Load into an existing host

Any `deepagent-*` host with `langgraph-stream-parser>=0.2` host conventions can run this agent:

```bash
# deepagent-code
DEEPAGENT_AGENT_SPEC="deepagent_hermes.agent:graph" deepagent-code

# deepagent-lab — set the same in deepagents.toml under [agent]
echo 'spec = "deepagent_hermes.agent:graph"' >> deepagents.toml
deepagent-lab
```

## Configuration

`deepagent-hermes.toml` (project) or `~/.deepagent-hermes/config.toml` (global). Layered resolution: `defaults < TOML < DEEPAGENT_HERMES_* env < CLI overrides`. See [SPEC §2](./SPEC.md#2-configuration) for every field; `deepagent-hermes --show-config` prints the resolved value + source of each.

## Architecture

See [SPEC.md](./SPEC.md) for the full 21-section requirements doc. Top-level layout:

- `src/deepagent_hermes/agent.py` — the compiled graph (entry point for hosts)
- `src/deepagent_hermes/config.py` — `HermesConfig(HostConfig)` resolver
- `src/deepagent_hermes/state.py` — `HermesState` (extends `AgentState`)
- `src/deepagent_hermes/reflection.py` — closed-loop middleware + review subagent
- `src/deepagent_hermes/skills/` — SkillLibrary, loader, tools
- `src/deepagent_hermes/memory/` — frozen-snapshot memory + provider ABC
- `src/deepagent_hermes/store/sqlite_fts.py` — `BaseStore` with FTS5
- `src/deepagent_hermes/search/session_search.py` — `session_search` tool
- `src/deepagent_hermes/compression.py` — `HermesCompressionMiddleware`
- `src/deepagent_hermes/caching.py` — `AnthropicCachingS3Middleware`
- `src/deepagent_hermes/budget.py` — `IterationBudgetMiddleware`
- `src/deepagent_hermes/tools/` — registry + 33 toolsets + 6 terminal envs
- `src/deepagent_hermes/cron/` — daemon + `cronjob` tool
- `src/deepagent_hermes/plugins/` — discovery + lifecycle hooks
- `src/deepagent_hermes/cli.py` — `deepagent-hermes` entry point
- `prompts/` — verbatim/paraphrased system-prompt building blocks

## Status by subsystem

| Subsystem | Status |
|---|---|
| Config + state + agent factory | working |
| Reflection loop (10-iter trigger, subagent review) | working |
| Skill library + agentskills.io validator | working |
| Skill loader (`@dynamic_prompt`) | working |
| `skill_view` / `skill_manage` / `skills_list` tools | working |
| Frozen-snapshot memory (MEMORY.md / USER.md) | working |
| SQLite FTS5 store + `session_search` (3 modes) | working |
| Honcho provider | stub (extras-gated) |
| Iteration budget middleware | working |
| Compression middleware (13-section template) | working |
| Anthropic `system_and_3` caching strategy | working |
| Tool registry + 33-toolset enum | working |
| `LocalEnvironment` terminal backend | working |
| Docker / SSH / Daytona / Modal / Singularity backends | stubs (`NotImplementedError`) |
| Cron daemon + `cronjob` tool | basic (local delivery only) |
| Plugin loader (4 discovery sources) | working (5 of 17 hooks) |
| CLI + v1-essentials slash commands | working |
| Curator (skill lifecycle) | basic |
| Self-evolution integration | docs only (separate offline repo) |
| Bundled skills | none — ship your own under `~/.deepagent-hermes/skills/` |

## License

MIT. See [LICENSE](./LICENSE). This project is a faithful reproduction of the design ideas in Nous Research's Hermes Agent — see [NOTICE](./NOTICE) for attribution.
