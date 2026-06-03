# Changelog

All notable changes to `deepagent-hermes` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-02

Initial public release. A faithful reproduction of [Nous Research's Hermes Agent](https://github.com/nousresearch/hermes-agent) on top of LangGraph + [`deepagents`](https://github.com/langchain-ai/deepagents) + [`langgraph-stream-parser`](https://github.com/dkedar7/langgraph-stream-parser).

### Highlights

- **Closed reflection→skill-creation loop.** After ~10 tool-using turns the review subagent fires, inspects the conversation, and writes/patches a `SKILL.md` to the user's library. Verified live against Anthropic.
- **Frozen-snapshot memory.** `MEMORY.md` + `USER.md` loaded once at session start; mid-session writes hit disk but don't change the system prompt, so the prefix cache stays warm.
- **FTS5 session search.** SQLite-backed store at `<HERMES_HOME>/state.db` with `messages_fts` (unicode61) + `messages_fts_trigram` (CJK) virtual tables; `session_search` tool with DISCOVERY / SCROLL / BROWSE modes.
- **agentskills.io spec compliance.** Bundled library validator enforces the spec verbatim; `skill_view` does progressive disclosure.
- **`langgraph-stream-parser` host-family compatible.** Adopt this agent in any `deepagent-*` host (cowork-dash / deepagent-lab / deepagent-code / deepagent-vscode) with `DEEPAGENT_AGENT_SPEC=deepagent_hermes.agent:graph`. No host code changes.

### What ships

- **Agent factory** `create_hermes_agent()` wiring a 14-middleware stack: `PluginEventBus`, `IterationBudgetMiddleware`, `PromptAssemblyMiddleware`, `SkillLoaderMiddleware`, `MemoryToolMiddleware`, `HermesStateRecorderMiddleware`, `ReflectionMiddleware`, `CuratorMiddleware`, deepagents' `TodoListMiddleware` / `FilesystemMiddleware` / `SubAgentMiddleware`, `HermesCompressionMiddleware`, `AnthropicCachingS3Middleware`, `PatchToolCallsMiddleware`.
- **Three-layer system prompt** (stable / context / volatile) with a byte-stable date-only line so the prefix cache survives the whole day.
- **`system_and_3` Anthropic prompt caching** strategy: system + last 3 messages, total 4 breakpoints (under the per-request cap).
- **13-section compression summary template** with anti-thrash skip when consecutive passes yield < 10%.
- **Iteration budget** (default 90 parent, 50 subagent), with `execute_code` as a refund-tool by default.
- **Three review prompts** (memory / skills / combined) for the reflection fork plus a curator prompt for the weekly consolidation pass.
- **33-toolset registry** with 30s `check_fn` TTL cache.
- **Six terminal-environment backends:**
  - `LocalEnvironment` — full subprocess impl (Windows-aware: detects Git Bash on PATH).
  - `DockerEnvironment` — container-per-session via `docker run --rm`.
  - `SshEnvironment` — paramiko-based with reconnect-on-broken-pipe.
  - `SingularityEnvironment` — `singularity` / `apptainer` auto-detected.
  - `DaytonaEnvironment`, `ModalEnvironment` — lazy SDK imports with defensive attribute probing; raise informative `ImportError` when the SDK is missing.
- **Cron daemon** (`python -m deepagent_hermes.cron`) with the 30-field Hermes job JSON shape, three deliverers (`local`, `stdout`, `agentmail`).
- **Plugin loader** with 4 discovery sources (bundled / user / project / pip entry-points) and a `PluginEventBus` middleware that wires 13 of 17 documented lifecycle hooks.
- **CLI** with v1-essential slash commands: `/new` / `/reset` / `/compress` / `/stop` / `/help` / `/quit` / `/model` / `/config` / `/skills` / `/cron` / `/curator` / `/memory` / `/tools` / `/toolsets` / `/verbose` / `/yolo` / `/reload`.
- **26 bundled skills** copied from `nousresearch/hermes-agent` (MIT, attributed in [`NOTICE`](./NOTICE)) covering software-development, github, research, data-science, mlops, productivity, note-taking.

### Stream-parser extractors (upstreamed)

Four new built-in extractors landed in `langgraph-stream-parser` v0.2.x:

- `SkillManageExtractor` (`skill_event`)
- `SkillViewExtractor` (`skill_loaded`)
- `CompressionExtractor` (`compression_summary`)
- `MemoryExtractor` (`memory_updated`)

Hosts surface these as inline events alongside agent text.

### Configuration

Layered resolver: `defaults < deepagent-hermes.toml < DEEPAGENT_HERMES_* env < CLI overrides`. Inherits the `langgraph-stream-parser` `HostConfig` for the cross-host keys. `deepagent-hermes --show-config` dumps every value with its source.

### Test posture

398 tests pass, 3 skipped (Docker / Singularity gated by binary presence; one Honcho test redundant when the SDK is installed). Live smokes cover a single-turn round-trip, a 5-turn reflection-trigger trace, a 12-turn substantive arc (memory writes verified end-to-end across 3 distinct topics), and host-adoption through `deepagent-code`'s config.

### Platform-forced divergences from Hermes

(documented in [`SPEC.md` §1](./SPEC.md))

- `langgraph-checkpoint-sqlite` added as required dep (langgraph only ships `InMemorySaver`).
- `BaseStore` has no FTS5 — we implement `SqliteFtsStore(BaseStore)` ourselves with Hermes's verbatim schema.
- `deepagents.create_deep_agent` is bypassed because it always prepends `BASE_AGENT_PROMPT` and appends user middleware *after* the defaults — we own the stack via `langchain.agents.create_agent` directly.

### Bugs caught + fixed during pre-release dogfood

Each of these surfaced live and the fix is in tree:

- **Anthropic `cache_control` per-request cap.** The parent `AnthropicPromptCachingMiddleware.model_settings["cache_control"]` nudge causes langchain-anthropic to tag the tools block; combined with our explicit system + last-3 tags that's 5 breakpoints, over the cap. Dropped the model_settings line.
- **Middleware state-update silently dropped.** Returning `{"session_id": ...}` from `before_agent` had no effect because the field wasn't in any merged `state_schema`. Fixed by per-middleware `state_schema` TypedDict extensions on the recorder, reflection, budget, and skills-loader middleware.
- **Parallel-write `InvalidUpdateError`.** Once the schemas were declared, parent + subagent writes in the same superstep crashed without an explicit reducer. Fixed by annotating counter/string fields with `Annotated[T, last_write_wins]` and the skills-loader lists/dicts with union/merge reducers.
- **`LangGraph` coerces `NotRequired[int]` to 0 at schema-merge time.** The budget seed's `if current is None: seed` was a no-op; budget started exhausted. Fixed with `if not current: seed`.
- **Workspace virtual-mode layering.** Agent wrote `/workspace/foo.py` which resolved to the literal `C:\workspace\foo.py` outside the sandbox. Fixed by `FilesystemBackend(virtual_mode=True)`.
- **Review subagent had no tools.** Even when reflection fired, the subagent couldn't act because its tool list was empty. Fixed by passing `skill_tools + memory tool` through `build_review_subagent(tools=...)`.

### Known limitations / deferred to v0.2.x

- **Honcho user-model provider** is implemented against the real `honcho-ai>=2.0,<3` SDK but ships as an `[honcho]` optional dep; needs an account + key to actually wire up.
- **Daytona / Modal backends** are lazy-SDK with explicit `TODO(*-api-verify)` markers on every uncertain SDK shape; should be verified against real accounts before v0.2.
- **4 of 17 plugin lifecycle hooks** are middleware-unreachable in v0.1 (`on_session_reset`, `subagent_stop`, `pre_gateway_dispatch`, gateway-only hooks) — documented in `PluginEventBus.__doc__`.
- **`mypy --strict`** not run end-to-end (would surface trivial Annotated quirks + incomplete langchain stubs). `ruff check` is clean.
- **Self-evolution integration** is docs-only — the offline DSPy/GEPA → PR pipeline ships separately under `deepagent-hermes-self-evolve` when needed.

### Acknowledgments

Nous Research is the originator of the design ideas reproduced here. Their `hermes-agent` (MIT) is the source of truth for the architecture, prompt structures, configuration defaults, file formats, and the 26 bundled SKILL.md files — see [`NOTICE`](./NOTICE) for full attribution.

[0.1.0]: https://github.com/dkedar7/deepagent-hermes/releases/tag/v0.1.0
