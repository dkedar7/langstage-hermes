# `deepagent-hermes` — Requirements Specification

**Version**: 0.1 (draft)
**Date**: 2026-06-01
**Goal**: Reproduce Nous Research's Hermes Agent (v0.15.1) on LangGraph + `deepagents` + `langgraph-stream-parser`, as faithfully as the platform allows. Divergences from Hermes are explicitly enumerated as **PLATFORM-FORCED** or **OUT-OF-SCOPE**.

**Naming**: project = `deepagent-hermes` (fits the `deepagent-*` family — `lab`, `code`, `vscode`).

**Repo layout target**: `C:\Users\Kedar\Documents\Code\deepagent-hermes`, central venv at `C:\Users\Kedar\.venvs\deepagent-hermes`, private GH remote, pins `langgraph-stream-parser>=0.2,<0.3` and `deepagents`.

**Source-of-truth references**:
- Hermes repo: `https://github.com/NousResearch/hermes-agent` (clone at `C:\Users\Kedar\Documents\Code\hermes-agent` for source consultation)
- agentskills.io spec: `https://agentskills.io/specification.md`
- `langgraph-stream-parser` v0.2.0 (substrate)
- `deepagents` 0.6+ + `langchain.agents.middleware`

---

## 0. Scope

**In scope:**
- §1 Agent loop & system-prompt assembly
- §2 Three-layer system prompt (stable / context / volatile) with prefix-cache discipline
- §3 Context compression
- §4 Iteration budget + retry/failover
- §5 Multi-provider model routing + transports
- §6 Reflection → skill creation (closed loop) + curator
- §7 Skill format, library layout, retrieval, slash commands
- §8 Toolset taxonomy and registry
- §9 Terminal-environment backends (6)
- §10 Persistent memory — MEMORY.md / USER.md + Honcho user model
- §11 FTS5 cross-session recall (`session_search`)
- §12 Cron + scheduled jobs
- §13 Plugin discovery + lifecycle hooks
- §14 CLI/TUI surface (slash commands, `/model`, etc.)
- §15 Self-evolution integration (offline PR pipeline only)

**Out of scope:**
- Gateway / 20+ messaging platforms (Telegram, Slack, Discord, WhatsApp, Signal, Matrix, Mattermost, Email, SMS, DingTalk, Feishu, WeCom, BlueBubbles, Home Assistant, Teams, Google Chat). The four existing `deepagent-*` hosts cover this surface.
- The `acp_adapter` / `acp_registry` (Agent Communication Protocol) — separate concern.
- A full TUI clone (`ui-tui/`) — the spec covers the CLI slash-command grammar but defers the prompt_toolkit UI; existing `deepagent-code` is the host.

---

## 1. Platform-forced divergences (read first)

These are gaps in `deepagents`/`langgraph`/`langchain` where Hermes does something the platform doesn't natively support. **Every one of these requires code Hermes already has.**

| # | Hermes feature | langgraph/deepagents gap | Required workaround |
|---|---|---|---|
| D1 | FTS5 session search on SQLite | `BaseStore.search()` is vector + filter only; no FTS | Implement `SqliteFtsStore(BaseStore)` with `messages_fts` + `messages_fts_trigram` virtual tables (verbatim Hermes schema) |
| D2 | SQLite checkpointer | Only `InMemorySaver` ships; SQLite/Postgres are separate uninstalled packages | Add `langgraph-checkpoint-sqlite` as required dep |
| D3 | 6 terminal backends (local/Docker/SSH/Daytona/Modal/Singularity) | Only `BaseSandbox` abstract ships in deepagents; no concrete impls | Subclass `BaseSandbox` six times; each implements `_run_bash` + `cleanup` per Hermes's `tools/environments/base.py` protocol |
| D4 | Plugin discovery (pip entry-points + user/project/bundled dirs) | No plugin discovery in deepagents | Implement `HermesPluginLoader` scanning `~/.deepagent-hermes/plugins/`, `./.deepagent-hermes/plugins/`, plus `importlib.metadata.entry_points(group="deepagent_hermes.plugins")` |
| D5 | Cron scheduler (per-job JSON, 60s tick, multi-deliver) | No scheduler in langgraph | Either: (a) bundle a `HermesCron` daemon process using `croniter` + a `.tick.lock` file (faithful), or (b) document Windows Task Scheduler / cron / systemd as the trigger and ship a `deepagent-hermes cron run-due` CLI. **Recommend (a)** for fidelity. |
| D6 | Memory provider plugin slot (single-select: honcho/mem0/byterover/…) | No plugin slot concept in deepagents | `HermesConfig.memory_provider` + a `MemoryProvider` ABC; bundled `HonchoProvider` (only one implemented in v1) |
| D7 | Reflection fork inherits prefix cache | LangGraph subagents (`SubAgentMiddleware`) re-build state but use the same compiled subgraph — cache discipline is the framework's | Build the review subagent with the **same `system_prompt` byte-identical** to the parent (volatile-layer date stays the same → prefix cache hits naturally); no special hook needed |
| D8 | `create_deep_agent` always prepends `BASE_AGENT_PROMPT` and appends user middleware AFTER the defaults | Can't remove or reorder defaults | Drop `create_deep_agent` — build our own factory on top of `langchain.agents.create_agent` so we own the middleware list. Use deepagents middleware classes directly. |
| D9 | Iteration budget = 90 (parent) / 50 (subagent) | LangGraph `recursion_limit` is per-graph not per-task | Implement `IterationBudgetMiddleware` with state-tracked counter; jump to `"end"` on exhaustion |
| D10 | Auxiliary client for summarization (separate model + credentials) | `SummarizationMiddleware` takes one model | Either pass a second `init_chat_model(...)` instance, or subclass `SummarizationMiddleware` to read `request.runtime.context.aux_model` |
| D11 | Three API modes (`chat_completions`, `codex_responses`, `anthropic_messages`) | langchain abstracts away wire format | Mostly transparent — `init_chat_model("openai:gpt-5", ...)` vs `("openai:codex-mini", ...)` vs `("anthropic:claude-sonnet-4-6", ...)` already route correctly. The `bedrock_converse` case needs `langchain-aws`. |
| D12 | Cache breakpoints on system + last 3 messages | `AnthropicPromptCachingMiddleware` (in `langchain-anthropic`) handles system but not "last 3 messages" — its strategy is its own | Either accept langchain-anthropic's strategy as a divergence, OR subclass it to match Hermes's `"system_and_3"` strategy exactly |

These workarounds dominate the build budget. **Items D1, D3, D5 are individually larger than the closed-loop reflection itself.**

---

## 2. Configuration

`deepagent-hermes.toml` (project) or `~/.deepagent-hermes/config.toml` (global), resolved through `langgraph-stream-parser`'s `HostConfig`. Subclass `HermesConfig(HostConfig)`. Field defaults match Hermes verbatim.

```toml
[model]
default = "anthropic:claude-sonnet-4-6"
provider = "auto"
# context_length = 200000
# max_tokens = 8192
aux_model = "anthropic:claude-haiku-4-5-20251001"  # for summarization (Hermes auxiliary_client)

[agent]
api_max_retries = 3
max_iterations = 90              # IterationBudgetMiddleware
delegation_max_iterations = 50
task_completion_guidance = true
environment_probe = true
tool_use_enforcement = "auto"
disabled_toolsets = []

[memory]
memory_enabled = true
user_profile_enabled = true
nudge_interval = 10              # turns since last memory review
memory_char_limit = 2200
user_char_limit = 1375
provider = ""                    # "honcho" or empty

[skills]
creation_nudge_interval = 10     # tool iterations since last skill_manage
external_dirs = []
disabled = []
[skills.platform_disabled]
# telegram = ["foo-skill"]

[compression]
enabled = true
threshold = 0.50                 # of context_length
target_ratio = 0.20
protect_first_n = 3
protect_last_n = 20
abort_on_summary_failure = false

[delegation]
max_concurrent_children = 4
max_spawn_depth = 3
max_iterations = 50

[curator]
enabled = true
interval_hours = 168             # weekly
min_idle_hours = 2
stale_after_days = 30
archive_after_days = 90
prune_builtins = true

[cron]
tick_seconds = 60

[plugins]
enabled = []
disabled = []
```

**Acceptance**: `deepagent-hermes --show-config` prints every field with its source (defaults < TOML < env < CLI) per the parser's existing `HostConfig.describe()` printer.

---

## 3. State schema

Custom `AgentState` TypedDict (registered via `HermesMiddleware.state_schema`):

```python
class HermesState(AgentState):  # extends langchain's base
    # Iteration tracking — drives reflection triggers
    iters_since_skill: int           # incremented per tool-using turn; reset on skill_manage call
    turns_since_memory: int          # incremented per user turn; reset on memory call
    iteration_budget_remaining: int  # decremented by IterationBudgetMiddleware

    # Skill state
    active_skills: list[str]         # names currently loaded via skill_view
    loaded_skill_bodies: dict[str, str]  # cached SKILL.md bodies (token cost)

    # Compression state
    last_compression_at: int         # message index; for anti-thrash
    consecutive_low_yield_compressions: int

    # Background-review coordination
    pending_review_kind: Literal["memory", "skills", "combined", None]
    last_review_started_at: float    # UNIX ts; sentinel for "review already running"

    # Cost / budget
    estimated_cost_usd: float
    actual_cost_usd: float | None

    # Session lineage
    session_id: str
    parent_session_id: str | None
    rewind_count: int
```

Hermes tracks all of these as `agent._*` instance attrs; on `deepagents` we use state because middleware is stateless and state IS the per-thread persistence boundary. **Annotate counters with `PrivateStateAttr`** so they don't pollute input/output JSON schema.

---

## 4. Agent loop (Hermes §1)

**Hermes**: `AIAgent` class, `run_conversation()` in `run_agent.py` → delegates to `conversation_loop.run_conversation()` (~4750 LOC). Each iteration: budget consume → message sanitize → transport.build_kwargs → API call (3 retries) → response normalize → tool dispatch → repeat. Returns `{messages, token_usage, cost, exit_reason}`.

**Reproduction**: `create_hermes_agent(config: HermesConfig) -> CompiledStateGraph`. Internally calls `langchain.agents.create_agent(...)` with our middleware stack (NOT `deepagents.create_deep_agent` — see D8):

```
middleware = [
    HermesMiddleware(),               # state schema + lifecycle nudges
    IterationBudgetMiddleware(...),   # before_model: check budget; after_model: decrement
    PromptAssemblyMiddleware(...),    # @dynamic_prompt: build stable/context/volatile
    SkillLoaderMiddleware(...),       # adds skills to dynamic prompt + skill_view tool
    SkillToolsMiddleware(...),        # registers skills_list / skill_view / skill_manage
    MemoryToolMiddleware(...),        # MEMORY.md + USER.md frozen-snapshot
    SessionSearchMiddleware(...),     # session_search tool, FTS5
    ReflectionMiddleware(...),        # after_model: spawn background review
    ToolCountingMiddleware(...),      # wrap_tool_call: bump iters_since_skill
    CuratorMiddleware(...),           # before_agent: maybe_run_curator()
    HonchoMiddleware(...),            # before_agent/after_agent: peer chat
    TodoListMiddleware(),             # write_todos tool
    FilesystemMiddleware(backend=...),
    SubAgentMiddleware(...),
    HermesCompressionMiddleware(...), # replaces SummarizationMiddleware, uses aux_model
    AnthropicCachingS3Middleware(...),# subclass: "system_and_3" strategy
    PatchToolCallsMiddleware(),
    HumanInTheLoopMiddleware(interrupt_on=...),
]
```

Order is significant. Wrap-style middleware (outer→inner): `IterationBudget`, `PromptAssembly`, `SkillLoader`, then compression/caching. Hooks are evaluated in declared order.

**Acceptance**: a single turn against `init_chat_model("anthropic:claude-sonnet-4-6")` with no skills/memory configured produces a non-empty `AIMessage`; `iters_since_skill` and `turns_since_memory` are incremented by 1.

---

## 5. System prompt assembly (Hermes §1.3)

**Hermes**: `agent/system_prompt.py::build_system_prompt_parts` returns `{stable, context, volatile}`, joined with `"\n\n"`, cached on `agent._cached_system_prompt`, rebuilt only on compression.

**Reproduction**: `PromptAssemblyMiddleware` using `@dynamic_prompt`. Returns the three layers concatenated. **Layer contents map 1:1 to Hermes**:

- **stable**: `SOUL.md` (if `~/.deepagent-hermes/SOUL.md` exists) else `DEFAULT_AGENT_IDENTITY` (copy from `prompt_builder.py`); `HERMES_AGENT_HELP_GUIDANCE`; `TASK_COMPLETION_GUIDANCE`; tool-aware guidance (concatenate `MEMORY_GUIDANCE` / `SESSION_SEARCH_GUIDANCE` / `SKILLS_GUIDANCE` / `KANBAN_GUIDANCE` based on which toolsets are enabled); `COMPUTER_USE_GUIDANCE` if computer_use enabled; `TOOL_USE_ENFORCEMENT_GUIDANCE` gated by `TOOL_USE_ENFORCEMENT_MODELS = ("gpt","codex","gemini","gemma","grok","glm","qwen","deepseek")`; Google/OpenAI execution-discipline blocks; skills index from `build_skills_system_prompt`; environment hints; env probe; active-profile hint; platform hint.
- **context**: user `system_message` override; `build_context_files_prompt(cwd)` — load AGENTS.md / .cursorrules / HERMES.md walking up from cwd, scan with `tools/threat_patterns.py` (port verbatim), replace on hit with `"[BLOCKED: ...]"`.
- **volatile**: `MEMORY.md`, `USER.md`, external memory-provider block, then **date-only** line `"Conversation started: <Weekday, Month DD, YYYY>"` (NOT timestamp — byte-stable for prefix cache), then optional `Session ID:` / `Model:` / `Provider:`.

**Acceptance**: two consecutive turns within the same day produce byte-identical system prompts; the line `Conversation started: Monday, June 01, 2026` appears.

---

## 6. Prompt caching (Hermes §1.4)

**Hermes**: `apply_anthropic_cache_control(api_messages, cache_ttl="5m" | "1h", native_anthropic=False)`. Places 4 `cache_control` breakpoints: system prompt + last 3 non-system messages. Strategy literal: `"system_and_3"`.

**Reproduction**: `AnthropicCachingS3Middleware(AnthropicPromptCachingMiddleware)` subclass that overrides `wrap_model_call` to set cache control on system + last 3 messages instead of the upstream strategy. Same `ttl` arg (`"5m"` default, `"1h"` opt-in). No-op for non-Anthropic models.

**Acceptance**: outbound API request to Anthropic carries exactly 4 `cache_control` blocks on a 5+ message conversation, 1 block on a 1-message conversation.

---

## 7. Context compression (Hermes §1.5)

**Hermes**: `ContextCompressor(ContextEngine)` in `agent/context_compressor.py`. Defaults: `threshold_percent=0.50`, `protect_first_n=3`, `protect_last_n=20`, `summary_target_ratio=0.20`. Anti-thrash skip after 2 consecutive <10% yields. Five-step algorithm; summary template enforces 13 specific sections; `SUMMARY_PREFIX` explicitly tells the model the summary is reference, not active instructions, and that latest user message wins on contradiction.

**Reproduction**: `HermesCompressionMiddleware` — own implementation, NOT `SummarizationMiddleware`. Reasons: (a) need `protect_first_n` (deepagents only has `messages_to_keep`); (b) need tool-call/tool-result pruning (truncate `arguments` head-200 chars, dedupe identical results, summarize tool outputs); (c) need the 13-section template verbatim; (d) need the auxiliary-client routing.

Implementation: `before_model` hook checks `_estimate_tokens(state.messages) > threshold_tokens`; if yes, run the 5-step pipeline; replace middle messages with a single `SystemMessage` containing the summary prefixed with `SUMMARY_PREFIX`.

**Acceptance**: a synthetic conversation of 1000 dummy tool calls + 50 user messages compresses to ≤20% of original tokens; first 3 and last 20 messages preserved verbatim; summary contains all 13 section headers.

---

## 8. Iteration budget (Hermes §1.6)

**Hermes**: `IterationBudget` class. Parent default 90, subagent default 50. `consume()` decrements, `refund()` for programmatic `execute_code` so it doesn't count. Exit with `"budget_exhausted"`.

**Reproduction**: `IterationBudgetMiddleware` with `state_schema={"iteration_budget_remaining": int}`. `before_model` hook returns `{"jump_to": "end"}` (decorated `@hook_config(can_jump_to=["end"])`) when remaining ≤ 0. `wrap_tool_call` decrements, but skips decrement for `execute_code` tool (refund equivalent). Subagent middleware seeds remaining=50; parent seeds 90.

**Acceptance**: a synthetic agent that loops a no-op tool exits after exactly 90 calls with state field `iteration_budget_remaining == 0`.

---

## 9. Reflection / skill creation (Hermes §2, the differentiator)

**Hermes** (verified from source):
- Counters: `_iters_since_skill` (per tool-using iteration, reset on `skill_manage`), `_turns_since_memory` (per user turn, reset on `memory`). Defaults both = 10.
- Trigger: after the assistant produces `final_response` AND not interrupted AND (`_should_review_memory OR _should_review_skills`).
- Mechanism: `_spawn_background_review(messages_snapshot, review_memory, review_skills)` → forks an `AIAgent` daemon thread, inherits provider/model/base_url/system prompt (same prefix cache), narrows tool whitelist to memory + skill tools, sets `_memory_nudge_interval=0` and `_skill_nudge_interval=0` to prevent recursion.
- Prompts: `_MEMORY_REVIEW_PROMPT`, `_SKILL_REVIEW_PROMPT`, `_COMBINED_REVIEW_PROMPT` in `agent/background_review.py`.
- Self-evaluation: NOT a separate LLM call — the review-fork IS the self-evaluation. The library shape is enforced by the prompt ("CLASS-LEVEL skills, each with a rich SKILL.md and a `references/` directory"; "Preference Order: update currently-loaded skill → existing umbrella → support file → create new umbrella"; "Be ACTIVE — most sessions produce at least one skill update").

**Reproduction**:

1. **`ReflectionMiddleware`** with `state_schema={"iters_since_skill", "turns_since_memory", "pending_review_kind"}`. Hooks:
   - `wrap_tool_call`: increments `iters_since_skill` if `skills` toolset enabled AND tool != `skill_manage`. Resets on `skill_manage` call.
   - On user turn boundary (detected in `before_model` by checking last message role): increment `turns_since_memory`. Reset when `memory` tool called.
   - `after_model`: if last AIMessage has no pending tool calls (= final_response) AND either counter ≥ threshold, set `pending_review_kind` in state.
   - `after_agent`: if `pending_review_kind` is set, spawn the review.

2. **Background review spawn**: two implementation choices, both viable:
   - **(A) Background `threading.Thread`** (faithful to Hermes). Compile a second `CompiledStateGraph` via `create_hermes_agent` with the same config but `enabled_toolsets=["memory","skills"]`, `iters_since_skill=0`, `turns_since_memory=0`, recursion_limit=20. Invoke synchronously on the thread with a single user message = the appropriate review prompt. **Pro**: matches Hermes's "spawn and forget" behavior; **Con**: thread management, race conditions, no observability via the parser's event stream.
   - **(B) `SubAgentMiddleware` review subagent** (deepagents-native). Register a `review` subagent at agent-build time with the review prompt as `system_prompt`. Call it from `after_agent` via `Command(goto="review_subagent")` or via the `task` tool. **Pro**: stays inside the LangGraph compiled graph, events flow through the parser, observable. **Con**: not asynchronous — the user waits for review to finish before the next turn (acceptable trade for fidelity; mitigated by aux_model).

   **Recommend (B)** because the parser's `ToolExtractedEvent` can surface "🧠 skill updated: pdf-merging" inline in every host UI. Hermes's threading is an implementation detail, not a feature — (B) is the langgraph-idiomatic equivalent.

3. **Three review prompts**: ported verbatim from `agent/background_review.py`. Store as `prompts/memory_review.md`, `prompts/skill_review.md`, `prompts/combined_review.md`.

4. **Curator** (separate concern): `CuratorMiddleware.before_agent` checks `~/.deepagent-hermes/skills/.curator_state`. If `last_run_at + interval_hours < now` AND `last_user_activity > min_idle_hours ago`, spawn a curator subagent with `CURATOR_REVIEW_PROMPT`. Skill lifecycle states: active → stale (30 days inactive) → archived (90 days). Pinned skills immune. Report at `~/.deepagent-hermes/logs/curator/{YYYYMMDD-HHMMSS}/{run.json, REPORT.md}`.

**Acceptance**:
- After exactly 10 tool-using turns where no `skill_manage` was called, the next `after_model` sets `pending_review_kind="skills"`.
- The skill_review run consumes the review prompt verbatim from `prompts/skill_review.md`.
- Curator state file exists and updates on each run.

---

## 10. Skill format and library (Hermes §3)

**Hermes**: agentskills.io–compatible. Frontmatter: `name` (≤64), `description` (≤1024), optional `version`, `license`, `platforms`, `prerequisites`, `compatibility`, `metadata.hermes.{tags, related_skills}`. Directory: `skills/<category>/<name>/SKILL.md` + `references/`, `templates/`, `assets/`. Retrieval: filesystem scan by `build_skills_system_prompt`, two-layer cache (LRU + disk snapshot `.skills_prompt_snapshot.json`), groups by category, filters by platform + disabled + condition fields (`requires_tools/toolsets`, `fallback_for_tools/toolsets`). Renders as `## Skills (mandatory)` block with `- <name>: <description>` per skill. Full body loaded on demand via `skill_view(name)`.

**Reproduction**:

1. **`SkillLibrary` class** at `deepagent_hermes/skills/library.py`. Methods: `list()`, `get(name) -> Skill`, `write(skill)`, `delete(name)`, `validate(frontmatter)` per agentskills.io spec. File ops + frontmatter validation via `python-frontmatter` or `pyyaml`.
2. **Default skill dirs**: `~/.deepagent-hermes/skills/` (user) + `./.deepagent-hermes/skills/` (project shadow) + bundled `<repo>/skills/` + any in `config.skills.external_dirs`. Resolution: project > user > bundled (later wins on name collision).
3. **`SkillLoaderMiddleware`** uses `@dynamic_prompt` to inject the skills index into the system prompt. Cache identical to Hermes (LRU in-process + disk snapshot validated by mtime/size manifest). Verbatim preface text from `agent/prompt_builder.py` lines 1236-1262.
4. **Tools registered by `SkillToolsMiddleware`**: `skills_list(query?, category?)`, `skill_view(name)`, `skill_manage(action, ...)`. Schemas mirror Hermes. `skill_view` appends body to `state.loaded_skill_bodies[name]` (so a later `wrap_model_call` can inject loaded bodies into prompt — Hermes does this; the alternative is to return the body as the tool result and let the model re-read each turn, costing more tokens).
5. **`skill_manage` actions**: `create`, `patch`, `write_file`, `delete`, `pin`, `unpin`. `patch` takes diff-like edit; `write_file` overwrites the whole file; `delete` archives to `_archived/` rather than rm. On any successful action, **reset `state.iters_since_skill = 0`**.

**Acceptance**:
- A SKILL.md with `name: "PDF-Processing"` is rejected at write (uppercase not allowed by agentskills.io spec).
- `skills_list()` returns only skills compatible with current platform (per `HERMES_PLATFORM` env var, default `"cli"`).
- After `skill_view("foo")`, the next `wrap_model_call` injects the SKILL.md body into the system prompt; the loaded list is shown in volatile layer.

---

## 11. Toolset taxonomy (Hermes §4)

**Hermes**: `tools/registry.py` `ToolRegistry` singleton; AST-scan of `tools/*.py` for `registry.register(...)` calls. 30s TTL on `check_fn` to avoid re-probing Docker/Modal/playwright per turn. 33 toolsets enumerated in `toolsets.py`. Per-platform overrides via `platform_toolsets.<platform>` in config.

**Reproduction**:

1. **`HermesToolRegistry`** at `deepagent_hermes/tools/registry.py`. Mirrors Hermes API: `register(tool, *, toolset, check_fn?, requires_env?)`. The 30s TTL cache wraps `check_fn` results to avoid expensive probes.
2. **`toolsets.py`**: define every toolset Hermes defines (33), each mapping to a list of `BaseTool` constructors. Implementations:
   - **Must build from scratch**: `web` (search + extract — use `tavily` or `serper`), `vision_analyze` (Anthropic vision), `image_generate`, `terminal/process`, `skill_manage/skill_view/skills_list`, `memory`, `session_search`, `cronjob`, `delegate_task`, `clarify`, `execute_code`, `todo` (deepagents `write_todos` is the equivalent — alias).
   - **Wrap existing**: `file` toolset = deepagents' `FilesystemMiddleware` tools (`read_file`, `write_file`, `edit_file`, `glob`, `grep`, `ls`); just rename to Hermes's set (`read_file`, `write_file`, `patch`, `search_files`).
   - **Out of scope for v1**: `discord`, `discord_admin`, `yuanbao`, `feishu_doc`, `feishu_drive`, `spotify`, `homeassistant`, `tts`, `video_analyze`, `video_generate`, `x_search`, `messaging`, `kanban` (defer to v2).
3. **`enabled_toolsets`** per session: filter at agent-build time. Config: `[agent].disabled_toolsets`, `[platform_toolsets].<platform>` overrides, plus session-level via runtime `context_schema`.

**Acceptance**: `deepagent-hermes tools` CLI lists every registered toolset with its tools and the `check_fn` status (cached 30s).

---

## 12. Terminal environments (Hermes §4) — 6 backends

**Hermes**: `tools/environments/base.py::BaseEnvironment` ABC. Unified spawn-per-call model. Session snapshot at `/tmp/hermes-snap-{session_id}.sh` (env vars + functions + aliases + shellopts). CWD tracking via `/tmp/hermes-cwd-{session_id}.txt` (local) or stdout marker `__HERMES_CWD_{session_id}__<path>__HERMES_CWD_{session_id}__` (remote). Stdin modes: `pipe` / `heredoc`. ProcessHandle duck type (`subprocess.Popen` natively; SDK backends use `_ThreadedProcessHandle`).

**Reproduction**: implement `BaseEnvironment` matching Hermes's protocol. Six concrete subclasses:

1. **`LocalEnvironment`** — `subprocess.Popen`. Snapshot in `%TEMP%/deepagent-hermes-snap-{session}.sh` (Windows-aware path). **Acceptance**: `pwd` then `cd ..` then `pwd` shows the directory changed across calls.
2. **`DockerEnvironment`** — `docker run --rm -v <bind> <image> bash -c ...`. Required env var `DEEPAGENT_HERMES_DOCKER_IMAGE` (default `python:3.13-slim`). **Out of scope without Docker installed on dev box.**
3. **`SshEnvironment`** — `paramiko` for SSH; `rsync` or `scp` for file sync. Config: `[ssh].host`, `.user`, `.key_path`.
4. **`DaytonaEnvironment`** — `daytona-sdk` (optional dep extra `[daytona]`).
5. **`ModalEnvironment`** — `modal` SDK (optional dep extra `[modal]`). `_ThreadedProcessHandle` wraps blocking exec.
6. **`SingularityEnvironment`** — `singularity exec --bind <bind> <image>`. Optional extra `[singularity]`.

**Plug-in protocol**: same shape as Hermes. `_run_bash(cmd_string, *, login, timeout, stdin_data) -> ProcessHandle` + `cleanup()`. Universal `_wait_for_process` poll loop with 10s heartbeat, interrupt checking, process-group kill on signal exit.

**Acceptance**: each concrete backend implements the protocol and `pytest -k terminal_<backend>` passes the same suite (CWD persists, snapshot loads, command output captured).

---

## 13. Persistent memory (Hermes §5)

### 13.1 MEMORY.md + USER.md (bundled)

**Hermes**: `tools/memory_tool.py`. Two files under `HERMES_HOME/memories/`. Single `memory` tool, actions `add | replace | remove | read`. Entry delimiter `\n§\n`. Char limits: `memory_char_limit=2200`, `user_char_limit=1375`. **Frozen-snapshot pattern**: loaded into `_system_prompt_snapshot` at session start; mid-session writes update disk but NOT the prompt → prefix cache preserved for the entire session. Threat-pattern scan on every entry on load.

**Reproduction**: `MemoryToolMiddleware`. Identical: snapshot at `before_agent`, expose `memory` tool that writes disk but does NOT mutate the snapshot. Volatile-layer prompt builder reads `state.memory_snapshot` / `state.user_snapshot` only. Acceptance: changing MEMORY.md mid-session does not change subsequent system prompts (verify by hashing).

### 13.2 Honcho user model (plugin)

**Hermes**: `plugins/memory/honcho/`. Uses official `honcho` SDK. Config resolution: `$HERMES_HOME/honcho.json` → `~/.hermes/honcho.json` → `~/.honcho/config.json` → env vars `HONCHO_API_KEY`, `HONCHO_ENVIRONMENT`. Host key `hermes` (or `hermes_<profile>`). Recall modes: `hybrid` (default), `context`, `tools`. Plug-in is single-select via `memory.provider` in config.

**Reproduction**:

1. **`MemoryProvider` ABC** at `deepagent_hermes/memory/provider.py`. Methods: `setup_session(session_id, user_id)`, `recall(query, mode) -> list[str]`, `record_turn(role, content)`, `teardown()`.
2. **`HonchoProvider(MemoryProvider)`** at `plugins/honcho_provider.py`. Pip dep: `honcho-sdk`. Config resolution identical to Hermes (same precedence chain).
3. **`HonchoMiddleware`**: `before_agent` → `provider.setup_session()`; `after_agent` → `provider.teardown()`; `after_model` → `provider.record_turn(role, content)` for the latest message. Inject `provider.recall(...)` output into volatile system-prompt layer.
4. **Other providers** (`mem0`, `byterover`, `hindsight`, `holographic`, `openviking`, `retaindb`, `supermemory`): **OUT OF SCOPE for v1**. Document the ABC so contributors can add them.

**Acceptance**: with `memory.provider = "honcho"` and Honcho credentials, asking "what do you know about me?" returns content the user has previously taught the agent across sessions.

### 13.3 FTS5 session search

**Hermes**: `hermes_state.py`. SQLite at `HERMES_HOME/state.db`, WAL mode. Schema: `sessions`, `messages`, `state_meta`, `compression_locks`. Two FTS5 virtual tables: `messages_fts USING fts5(content)` (unicode61) and `messages_fts_trigram USING fts5(content, tokenize='trigram')` (CJK). Indexed value = `content || ' ' || tool_name || ' ' || tool_calls`. Triggers keep FTS sync'd. `session_search` tool: 3 modes — DISCOVERY (FTS5 BM25 + lineage dedupe + snippet + ±5 window + bookend), SCROLL (anchor + ±window), BROWSE (recent chronologically).

**Reproduction**:

1. **`SqliteFtsStore(BaseStore)`** at `deepagent_hermes/store/sqlite_fts.py`. Implements `get/put/delete/search/list_namespaces/batch`. Schema verbatim from Hermes. WAL mode + 20-150ms retry jitter.
2. **`HermesStateRecorder`**: a `wrap_tool_call` + `after_model` listener that writes every message to `messages` table. Triggers handle FTS sync.
3. **`session_search` tool** at `deepagent_hermes/tools/session_search.py`. Three modes match Hermes argument schema exactly: `(query?, session_id?, around_message_id?, window?, sources_exclude?)`.
4. **`langgraph-checkpoint-sqlite`** added as required dep — same SQLite file holds both checkpoints and search index. Tag rows with `source` column matching `HERMES_SESSION_SOURCE` env var (default `"user"`; reflection forks tag `"tool"` and are hidden by default).

**Acceptance**:
- 1000 dummy messages indexed; FTS5 query returns BM25-ranked top 10 in <100ms.
- CJK query routes through trigram table (test with `"日本語"` content).
- DISCOVERY mode returns sessions with `bookend_start` (first 3 user+assistant) and `bookend_end` (last 3).

---

## 14. Cron (Hermes §6)

**Hermes**: `cron/` (jobs.py + scheduler.py). Storage: `~/.hermes/cron/jobs.json` (mode 0600, dir 0700). Tick lock `~/.hermes/cron/.tick.lock`. Gateway calls `tick()` every 60s. Job JSON: ~30 fields (see source). Schedules: interval, cron, once. Skills first-class field (preloaded into spawned agent's system prompt). Scripts can inject stdout as context (`no_agent=False`) or BE the job (`no_agent=True`). Cron platform hint tells the spawned agent it's running headless. Restricted toolset (always strips `cronjob`, `messaging`, `clarify`). `SILENT_MARKER = "[SILENT]"` suppresses delivery.

**Reproduction**:

1. **`HermesCron` daemon** at `deepagent_hermes/cron/scheduler.py`. Long-running `python -m deepagent_hermes.cron` process; 60s tick; lock file. Uses `croniter` for cron expressions, `parse_duration` for intervals. Storage at `~/.deepagent-hermes/cron/jobs.json`.
2. **`cronjob` tool** registered to the agent: actions `create`, `list`, `show`, `delete`, `pause`, `resume`, `run-now`. Job JSON shape: **verbatim copy of Hermes's 30 fields** including `skills`, `script`, `no_agent`, `context_from`, `enabled_toolsets`, `workdir`, `profile`.
3. **Job execution**: spawn `create_hermes_agent(config_with_overrides)` with `platform="cron"`, `enabled_toolsets=resolve_cron_disabled_toolsets()`, system prompt augmented with `PLATFORM_HINTS["cron"]` block (verbatim).
4. **Output**: append to `~/.deepagent-hermes/cron/output/{job_id}/{timestamp}.md`. Deliver per `deliver` field (`origin | local | telegram | …`). v1 supports `local` only (writes to disk); other deliverers OUT OF SCOPE.
5. **Windows note**: the long-running daemon process is cross-platform but Kedar's primary platform is Windows — document `Start-Process` + `Register-ScheduledTask` snippet for keeping it alive on logon, and document the Linux/macOS `systemd --user` / `launchd` equivalents.

**Acceptance**: create a job with `schedule="1m"` + `prompt="say hi"`; daemon ticks; first output appears in cron output dir within 70s.

---

## 15. Plugins (Hermes §7)

**Hermes**: `hermes_cli/plugins.py`. Discovery order: bundled → user (`~/.hermes/plugins/<name>/`) → project (`./.hermes/plugins/<name>/`, opt-in via env) → pip entry-points (`hermes_agent.plugins`). Each directory plugin has `plugin.yaml` + `__init__.py` with `register(ctx)`. `PluginContext.register_tool()` delegates to `tools.registry.register()`. 17 lifecycle hooks (pre_tool_call, post_tool_call, transform_terminal_output, transform_tool_result, transform_llm_output, pre_llm_call, post_llm_call, pre_api_request, post_api_request, on_session_start, on_session_end, on_session_finalize, on_session_reset, subagent_stop, pre_gateway_dispatch, pre_approval_request, post_approval_response). Single-select slots: memory provider, context engine.

**Reproduction**:

1. **`HermesPluginLoader`** scans bundled (`<repo>/plugins/`), user (`~/.deepagent-hermes/plugins/`), project (`./.deepagent-hermes/plugins/`, env-gated `DEEPAGENT_HERMES_ENABLE_PROJECT_PLUGINS=1`), and `importlib.metadata.entry_points(group="deepagent_hermes.plugins")`. Each directory plugin: `plugin.yaml` manifest + `__init__.py` with `register(ctx)`.
2. **`PluginContext`**: methods `register_tool(tool, *, toolset)`, `register_memory_provider(provider)`, `register_context_engine(engine)`, `register_hook(name, fn)`, `register_slash_command(name, fn)`.
3. **17 lifecycle hooks**: map to deepagents middleware. Most have direct equivalents (pre_tool_call → `wrap_tool_call`; pre_llm_call/post_llm_call → `wrap_model_call`; on_session_start → `before_agent`; on_session_end → `after_agent`). A few have no direct equivalent and require custom event bus: `transform_terminal_output`, `transform_tool_result`, `transform_llm_output` — implement as a `PluginEventBus` middleware emits events that subscribed plugins can transform synchronously.
4. **Single-select slots**: memory provider (`HonchoProvider` already bundled), context engine (default `HermesCompressionMiddleware`; alternative implementations can replace it).
5. **Plugin config**: `[plugins.enabled]` allow-list, `[plugins.disabled]` deny-list (always wins).

**Acceptance**: a third-party pip package `deepagent-hermes-mem0` exposing `[project.entry-points."deepagent_hermes.plugins"]` gets discovered and its `register(ctx)` runs at startup.

---

## 16. CLI / TUI (Hermes §8)

**Hermes**: `hermes_cli/` (subcommands: chat, setup, model, doctor, update, tools, skills, cron, curator, kanban, plugins, gateway, acp). `ui-tui/` is prompt_toolkit-based. Slash commands enumerated in `COMMAND_REGISTRY`.

**Reproduction**:

1. **CLI** at `deepagent_hermes/cli.py`. Subcommands: `chat`, `--show-config`, `tools`, `skills` (list/show/install/audit), `cron` (list/create/run-due), `curator` (status/run/pause/resume/pin), `plugins` (list/enable/disable), `doctor`.
2. **Slash commands** (in-session, during `chat`): port the full `COMMAND_REGISTRY` from `hermes_cli/commands.py`. v1 essentials (must ship): `/new`, `/reset`, `/compress`, `/stop`, `/help`, `/quit`, `/model`, `/config`, `/skills`, `/cron`, `/curator`, `/memory`, `/tools`, `/toolsets`, `/verbose`, `/yolo`, `/reload`. v2 nice-to-haves: `/rollback`, `/snapshot`, `/queue`, `/steer`, `/voice`, `/skin`, `/insights`.
3. **`/model` switching**: implemented as a `@wrap_model_call` middleware reading from `state.model_override`; slash command sets the state field. Re-uses `init_chat_model` for new provider strings.
4. **TUI**: **OUT OF SCOPE** for v1. `deepagent-code` is the existing host with its own CLI. Define the slash-command grammar so a future TUI can plug in.

**Acceptance**: `deepagent-hermes chat` opens an interactive prompt; `/model anthropic:claude-haiku-4-5-20251001` switches the model for subsequent turns; `/skills` lists available skills.

---

## 17. Self-evolution (Hermes §9)

**Hermes**: `NousResearch/hermes-agent-self-evolution` — separate repo. DSPy + GEPA. Phases 1-4 (phase 1 implemented: SKILL.md evolution). Offline runs; produces PRs against main repo; human merge before deployment. **No runtime hook.**

**Reproduction**:

1. **Document the integration shape** in README — pip package `deepagent-hermes-self-evolve` points at `DEEPAGENT_HERMES_REPO` env var, evolves SKILL.md files using DSPy + GEPA traces from `~/.deepagent-hermes/logs/`, produces a branch + PR.
2. **No runtime code** in v1. The runtime is fine with offline-evolved skills loaded normally — they're just SKILL.md files.
3. **Trace logging**: ensure all model/tool calls log to `~/.deepagent-hermes/logs/{session_id}.jsonl` in a shape DSPy can consume. This is needed regardless (for the curator and for debugging).

**Acceptance**: trace log produced per session; format documented; future self-evolve repo can consume.

---

## 18. Stream events / observability

**This part is FREE from `langgraph-stream-parser`** — confirming it works without changes:

- All Hermes runtime events map onto existing `langgraph-stream-parser` event types:
  - Agent text → `ContentEvent`
  - Tool calls → `ToolCallStartEvent` / `ToolCallEndEvent`
  - Reflection (`think_tool` equivalent) → `ReasoningEvent` via `ThinkToolExtractor`
  - Skill creation → **new** `ToolExtractedEvent(extracted_type="skill_created", data={name, action})` via a new `SkillManageExtractor` upstreamed as a PR to `langgraph-stream-parser`
  - Compression event → **new** `ToolExtractedEvent(extracted_type="compression_summary", data={ratio, sections})` via `CompressionExtractor`
  - Memory update → `ToolExtractedEvent(extracted_type="memory_updated")`
  - Cron schedule/fire → `CustomEvent` via `get_stream_writer()`
  - Curator run → `CustomEvent`
  - Interrupts → `InterruptEvent` (already covered)

- Upstream PR to `langgraph-stream-parser`: add three new built-in extractors (`SkillManageExtractor`, `CompressionExtractor`, `MemoryExtractor`). All hosts pick them up automatically through `compat.stream_graph_updates`.

**Acceptance**: running `deepagent-hermes` agent in `deepagent-code` CLI surfaces "🧠 skill created: pdf-merging" inline; same agent in `deepagent-lab` surfaces it via JupyterDisplay; in `cowork-dash` via the FastAPI session adapter.

---

## 19. Repository layout

```
deepagent-hermes/
├── pyproject.toml                # pins langgraph-stream-parser>=0.2,<0.3, deepagents, langchain, langchain-anthropic, langgraph-checkpoint-sqlite, croniter, pyyaml, python-frontmatter
├── deepagent-hermes.toml         # canonical config example
├── README.md
├── LICENSE                       # MIT
├── prompts/
│   ├── default_identity.md       # DEFAULT_AGENT_IDENTITY verbatim
│   ├── memory_guidance.md
│   ├── session_search_guidance.md
│   ├── skills_guidance.md
│   ├── kanban_guidance.md
│   ├── tool_use_enforcement.md
│   ├── task_completion.md
│   ├── computer_use.md
│   ├── openai_execution.md
│   ├── google_execution.md
│   ├── memory_review.md          # _MEMORY_REVIEW_PROMPT
│   ├── skill_review.md           # _SKILL_REVIEW_PROMPT
│   ├── combined_review.md
│   ├── curator_review.md
│   ├── compression_summary.md    # 13-section template + SUMMARY_PREFIX
│   └── platform_hints/
│       ├── cli.md
│       ├── cron.md
│       └── ...
├── src/deepagent_hermes/
│   ├── __init__.py               # exports create_hermes_agent, HermesConfig
│   ├── agent.py                  # graph = create_hermes_agent(HermesConfig.resolve())
│   ├── config.py                 # HermesConfig(HostConfig)
│   ├── state.py                  # HermesState TypedDict
│   ├── prompts.py                # PromptAssemblyMiddleware + helpers
│   ├── caching.py                # AnthropicCachingS3Middleware
│   ├── compression.py            # HermesCompressionMiddleware
│   ├── budget.py                 # IterationBudgetMiddleware
│   ├── reflection.py             # ReflectionMiddleware + review subagent
│   ├── curator.py                # CuratorMiddleware
│   ├── skills/
│   │   ├── library.py            # SkillLibrary
│   │   ├── validator.py          # agentskills.io frontmatter rules
│   │   ├── loader.py             # SkillLoaderMiddleware (@dynamic_prompt)
│   │   ├── tools.py              # skills_list / skill_view / skill_manage
│   │   └── prompt.py             # build_skills_system_prompt + cache
│   ├── memory/
│   │   ├── tool.py               # MemoryToolMiddleware (MEMORY.md + USER.md)
│   │   ├── provider.py           # MemoryProvider ABC
│   │   └── threat_patterns.py    # port from Hermes
│   ├── store/
│   │   └── sqlite_fts.py         # SqliteFtsStore(BaseStore) + schema
│   ├── search/
│   │   └── session_search.py     # session_search tool (3 modes)
│   ├── tools/
│   │   ├── registry.py           # HermesToolRegistry
│   │   ├── toolsets.py           # 33-toolset enumeration
│   │   ├── web.py
│   │   ├── vision.py
│   │   ├── image_gen.py
│   │   ├── code_execution.py
│   │   ├── clarify.py
│   │   ├── delegate.py
│   │   ├── todo.py               # alias of deepagents write_todos
│   │   ├── file.py               # alias of deepagents FilesystemMiddleware tools
│   │   └── environments/
│   │       ├── base.py           # BaseEnvironment protocol
│   │       ├── local.py
│   │       ├── docker.py
│   │       ├── ssh.py
│   │       ├── daytona.py
│   │       ├── modal.py
│   │       └── singularity.py
│   ├── cron/
│   │   ├── jobs.py               # job CRUD + storage
│   │   ├── scheduler.py          # 60s tick daemon
│   │   └── tool.py               # cronjob tool
│   ├── plugins/
│   │   ├── loader.py             # HermesPluginLoader
│   │   ├── context.py            # PluginContext + hook registration
│   │   └── builtin/
│   │       └── honcho_provider/
│   │           ├── plugin.yaml
│   │           └── __init__.py
│   ├── extractors.py             # SkillManageExtractor + CompressionExtractor + MemoryExtractor (to upstream)
│   └── cli.py                    # subcommands + slash-command dispatch
├── skills/                       # bundled skills (curated subset; ship maybe 20-30, not 176)
│   ├── software-development/
│   ├── research/
│   └── ...
├── tests/
│   ├── test_state_schema.py
│   ├── test_prompt_assembly.py
│   ├── test_caching_strategy.py
│   ├── test_compression.py
│   ├── test_iteration_budget.py
│   ├── test_reflection_trigger.py
│   ├── test_curator_lifecycle.py
│   ├── test_skill_validator.py
│   ├── test_skill_loader_progressive_disclosure.py
│   ├── test_skill_manage_actions.py
│   ├── test_memory_snapshot_frozen.py
│   ├── test_honcho_provider.py
│   ├── test_sqlite_fts_store.py
│   ├── test_session_search_modes.py
│   ├── test_cron_daemon.py
│   ├── test_plugin_loader.py
│   ├── test_terminal_local.py
│   └── test_with_existing_hosts.py
└── examples/
    ├── cli_smoke.py
    ├── load_into_deepagent_code.md
    └── deepagent-hermes.toml.example
```

---

## 20. Acceptance criteria summary

A v1 release is shippable when:

1. `deepagent-hermes chat` opens an interactive loop running a deepagents-built agent.
2. All §3 state fields are tracked correctly across turns (verified by test).
3. Reflection fires at exactly 10 tool iterations / 10 user turns (verified by test).
4. Skill library scans + injects index into system prompt; `skill_view` loads body; `skill_manage` writes valid agentskills.io files (verified by test against the agentskills.io reference validator).
5. SQLite FTS5 session search returns BM25-ranked results across multiple sessions.
6. Honcho provider plugin integrates and recalls cross-session.
7. Cron daemon ticks every 60s, fires jobs on schedule, writes output.
8. Local terminal backend works end-to-end (Docker/SSH/etc. tested if SDK available, else skipped).
9. CLI slash-command parity with Hermes v1 essentials list.
10. Existing hosts (`deepagent-code`, `deepagent-lab`, `cowork-dash`, `deepagent-vscode`) load this agent unchanged via `DEEPAGENT_AGENT_SPEC=deepagent_hermes.agent:graph` (verified by smoke test).
11. All three new parser extractors merged upstream to `langgraph-stream-parser`.
12. Test suite covers ~70%+ lines; agentskills.io validator passes on bundled skills.

---

## 21. Open decisions

Spec is complete enough to build from; these are choices that don't block but should be resolved before Phase 0:

1. **Bundled-skill curation**: which subset of Hermes's 176 to ship in this repo? Recommend: ship 20-30 that align with Kedar's actual usage (data-science, software-development, github, research, devops); let the rest be `pip install deepagent-hermes-skills-extra`.
2. **Honcho dep — required or extras?** Recommend `extras = ["honcho-sdk"]` so the base install is light; `pip install deepagent-hermes[honcho]` to enable.
3. **Terminal backends — bundle all 6 or just local?** Recommend bundle Local + LightDocker (subprocess-based, no Docker SDK) in core; ship Modal/Daytona/SSH/Singularity as extras.
4. **Reflection spawn — threading vs. subagent (decision (A) vs (B) in §9)?** Recommend (B) for observability.
5. **Curator persistence** — same SQLite DB as session search or separate? Recommend same DB, separate table `curator_state`.
