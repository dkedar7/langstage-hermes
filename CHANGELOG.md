# Changelog

All notable changes to `langstage-hermes` (formerly `deepagent-hermes`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] — 2026-07-02

### Fixed
- **Bare `pip install langstage-hermes` couldn't run a turn.** 0.4.0 made AG-UI the
  only render path but left the AG-UI runtime (`ag-ui-langgraph[fastapi]`) in the
  optional `[agui]` extra, so a default install hit an ImportError on the first
  message. The runtime is now a base dependency (via `langstage-core[agui]`); the
  `[agui]` extra is a redundant no-op alias.

## [0.4.0] — 2026-07-02

### Changed
- **Repointed to `langstage-core` 1.0; AG-UI is now the only render path.** The
  dependency `langgraph-stream-parser` was renamed to `langstage-core` (imports
  `langstage_core`), and its event layer was retired. `chat` now always streams
  through the in-process AG-UI adapter (previously behind `LANGSTAGE_HERMES_AGUI`);
  the four extractors ride the core's `extractors=` param, so the skill / memory /
  compression callouts are on by default. The StreamParser fallback + the env toggle
  are gone.

## [0.3.13] — 2026-07-02

### Added
- **Experimental AG-UI render path (`LANGSTAGE_HERMES_AGUI=1`).** `chat` can stream
  through the in-process `ag-ui-langgraph` adapter (via the core's
  `agui.iter_event_frames`) instead of the built-in `StreamParser`, and hermes' four
  tool-result extractors (skill/skill-view/compression/memory) ride the core's new
  `extractors=` param so the domain callouts surface as `extraction` frames. hermes'
  richer input (`session_id` / `model_override` / `iteration_budget_remaining`) rides
  `state=`. Requires the `agui` extra. Default path untouched.

### Fixed
- **Extractor callouts were dead code.** The four extractors were defined but never
  registered (`chat` used a bare `StreamParser()`), so skill/memory/compression
  callouts never fired. The AG-UI path wires them via `extractors=` — they now fire
  for the first time. (The legacy path remains unwired; it retires with the event layer.)

## [0.3.12] — 2026-06-30

### Fixed
- **`doctor` green-lit an `openai:*` model with the `[openai]` extra missing.**
  On a plain `pip install langstage-hermes` (no extras), `langchain-openai` is
  absent, so an `openai:*` model cannot build — yet `doctor` reported an all-green
  exit-0 bill of health while `verify` correctly failed (exit 2) naming
  `pip install "langstage-hermes[openai]"`. `doctor` advertises that it checks
  "deps", and the configured provider package is the dep that most often breaks a
  fresh install on the documented OpenAI/OpenRouter path. `doctor` now checks the
  configured model's provider package is importable, prints a `✗` line naming the
  same extra `verify` does, and exits non-zero — the full diagnostic still prints
  first. (Found by the dogfood routine, gh #41.)

## [0.3.11] — 2026-06-29

### Added
- **`skills remove` (alias `uninstall`).** `install` had no inverse, and
  `audit rollback` refuses on an install's `create` mutation, so the only way to
  remove a skill was a manual `rm` that desynced the audit log. `remove` archives
  the skill under `skills/_archived/` and lands a rollback-able `delete` audit row,
  so `audit rollback <name> <id>` restores it. (gh #39.)

## [0.3.10] — 2026-06-28

### Fixed
- **`memory.provider="markdown"` crashed instead of working.** The bundled
  `MarkdownProvider` self-registers at import, but the agent factory never imported
  it (only the `plugins` CLI did), so selecting it `KeyError`'d at agent build —
  chat wouldn't start and `verify` exited 2. The factory now registers the builtin
  providers and degrades an unknown provider name to noop with a warning instead of
  crashing. (gh #37.)

## [0.3.9] — 2026-06-27

### Fixed
- **`doctor` was not model-aware.** Unlike `verify`, it unconditionally checked
  `ANTHROPIC_API_KEY` ("required for the default anthropic:* model") even when an
  `openai:*` model was configured, and it printed the OpenAI/OpenRouter lines
  only when those keys were *set* — so on the README's OpenRouter path it cited a
  key the user doesn't need and stayed silent about the one they do (exiting 0
  with a clean bill of health). `doctor` now resolves the configured model and
  checks the key that model actually needs, printing the resolved model and
  flagging a missing OpenAI/OpenRouter key — matching `verify`. (Found by the
  dogfood routine, gh #35.)

## [0.3.8] — 2026-06-26

### Fixed
- **The documented `OPENROUTER_API_KEY` path didn't work.** The README advertises
  `OPENROUTER_API_KEY` as a drop-in alternative to `OPENAI_API_KEY`, but the
  runtime built `openai:*` models with a bare `init_chat_model`, and `ChatOpenAI`
  only reads `OPENAI_API_KEY` — so following the README verbatim failed at agent
  build with `OpenAIError: Missing credentials`. Worse, `verify`/`doctor`
  accepted `OPENROUTER_API_KEY` as a satisfied key, giving a false-positive
  preflight. When an `openai:*` model is selected with only `OPENROUTER_API_KEY`
  set, hermes now aliases it to `OPENAI_API_KEY` and defaults `OPENAI_BASE_URL`
  to the OpenRouter endpoint — so the documented path works and the
  `verify`/`doctor` acceptance becomes correct. (Found by the dogfood routine,
  gh #33.)

## [0.3.7] — 2026-06-25

### Fixed
- **`skills install` was invisible to the audit log.** The `audit` group's help
  promises "every CLI skill mutation appends a row… to see what changed and to
  revert," but `skills install` copied the skill with a raw `shutil.copytree`
  and never recorded a mutation — so `audit log` stayed empty after an install
  and `audit rollback` reported "mutation not found." It now lands a `create`
  row via the audit-aware library (new `SkillLibrary.record_install`), so the
  install shows up in `audit log` and behaves like any other create (rollback
  points you at `delete`, since a create has no prior state). (Found by the
  dogfood routine, gh #31.)

## [0.3.6] — 2026-06-22

### Fixed
- **Stale `langgraph-stream-parser` floor stranded hermes on core 0.4.x.** The
  pin was `>=0.3,<0.5`, two minor versions behind the rest of the family — so a
  clean install missed every 0.6.x core fix: BOM-safe `langstage.toml` parsing,
  the keyless stub compiling without a checkpointer, the *visible* legacy-env
  deprecation notice, dict-form message rendering, and the `tool_end` name
  backfill. Bumped to `>=0.6.10,<0.7` (and the `agui` extra likewise). The full
  suite passes on modern core (the visible `DEEPAGENT_HERMES_*` notice now
  reaches console-script users too). (Found by the dogfood routine.)
- **`verify` now points at the `[openai]` extra** when an OpenAI-compatible
  model fails to build for lack of `langchain-openai`, instead of surfacing only
  langchain's raw "install langchain-openai" message.

## [0.3.5] — 2026-06-22

### Fixed
- **Stale `~/.deepagent-hermes` paths in shipped text.** The `markdown-provider`
  plugin description (and the bundled skills README) still advertised the
  pre-rename `~/.deepagent-hermes/...` location, though the runtime correctly uses
  `HERMES_HOME` (`~/.langstage-hermes`). Updated both to the canonical path.
  (gh #-dogfood, cosmetic)

## [0.3.4] — 2026-06-21

### Fixed
- **`skills install <dir>` now installs under the skill's frontmatter `name`**,
  regardless of the source directory's name. Previously it required the directory
  to be named exactly after the skill (it validated/installed by dir name), so
  pointing `install` at any working dir failed with `name: must match parent
  directory name`. (gh #-dogfood)
- **`--show-config`'s "no TOML found" message** named only the cross-host
  `langstage.toml`/`deepagents.toml`; it now lists the real search order, leading
  with the documented `langstage-hermes.toml` / `~/.langstage-hermes/config.toml`.

## [0.3.3] — 2026-06-21

### Fixed
- **`uv venv .venv` install (the README's own command) silently dropped ALL bundled
  skills (gh #-dogfood).** The skill scanner's `_EXCLUDED_DIR_NAMES` (`.venv`, `venv`,
  `node_modules`, …) was matched against each SKILL.md's **absolute** path, so a package
  installed into a venv named `.venv` had `.venv` in every bundled-skill path → 0 loaded
  and `verify` failed (exit 2). Exclusions are now matched **relative to the search
  directory**, so junk dirs *inside* a skill tree are still skipped but the install prefix
  no longer collides. (Regression-sibling of the 0.3.2 bundled-skills fix — different root
  cause, exposed by the documented `uv venv .venv`.)

## [0.3.2] — 2026-06-20

### Fixed

- **Zero of the 26 bundled skills loaded (gh #-dogfood).** `_bundled_skills_dir()` resolved `parents[3] / "skills"` — a nonexistent repo-root `skills/` dir — instead of the in-package `langstage_hermes/_bundled_skills/`, so a clean install loaded **no** bundled skills (`skills list` → "No skills match", `SkillLibrary().list()` → 0). It now resolves relative to the package (`parent.parent / "_bundled_skills"`), working identically in source checkouts and installed wheels. Bundled skills load again (23 on the `cli` platform; 26 shipped).
- **`verify` reported a false green for bundled skills.** It counted SKILL.md files with a raw glob (26) from a *different* path than the runtime loaded, so it printed `✓ 26` while the agent had 0 — exactly the false-positive `verify` exists to prevent. It now counts what `SkillLibrary` actually **loads** and fails (red, exit 2) if files ship but none load.
- **Keyless model fallback disagreed with the configured default.** `agent.py`'s `_init_chat_model(None)` hard-coded `anthropic:claude-sonnet-4-5-20250929` while `HermesConfig.model_default` is `anthropic:claude-sonnet-4-6`; aligned both. README's documented default corrected to match.

### Tests

- New runtime-path tests that load bundled skills through `_bundled_skills_dir()` / `SkillLibrary` (the existing tests only validated the SKILL.md files via a hard-coded path, so the loader bug was invisible to them).

## [0.3.1] — 2026-06-20

### Fixed

- **Canonical `LANGSTAGE_HERMES_*` (and legacy core `DEEPAGENT_*`) env vars were silently ignored (gh #24).** `HermesConfig.resolve()` overrode the base resolver and read env vars by their raw declared (legacy) name, so the canonical `LANGSTAGE_HERMES_*` names it advertises — and that `--show-config`/`describe()` print — had no effect, and the legacy `DEEPAGENT_*` fallback for inherited core vars was dead under Hermes. This broke the README's documented OpenRouter setup *silently* (wrong value, no warning → confusing downstream `ANTHROPIC_API_KEY not set`). The override now routes every env read through the base's `_env_pair()`/`_warn_legacy_env()`: canonical wins, legacy resolves as a deprecated fallback with a `DeprecationWarning`. Added regression tests, and switched `examples/dogfood_openrouter.py` to the canonical names (the legacy-only examples were why this went unnoticed).

## [0.3.0] — 2026-06-14

### Added

- Adopt AG-UI: widen the `langgraph-stream-parser` ceiling to `<0.5` and add an `[agui]` extra so this surface's agent can be served over AG-UI via `langstage-agui`. Additive; no runtime changes.

## [0.2.1] — 2026-06-13

### Fixed

- `langstage-hermes --version` (and the CLI banner) reported `0.1.4` regardless of the installed version — a hand-maintained `__version__` constant in `__init__.py` that had been stuck since before 0.1.5. It now reads from installed package metadata (`importlib.metadata.version`), so it can never drift from `pyproject.toml` again.

## [0.2.0] — 2026-06-12

**deepagent-hermes is now `langstage-hermes`** — the reference agent of the LangStage family ("every stage for your LangGraph agent").

### Changed

- Distribution `deepagent-hermes` → **`langstage-hermes`**; module `deepagent_hermes` → **`langstage_hermes`**. A deprecated alias package keeps `import deepagent_hermes` — and crucially the documented host spec string `deepagent_hermes.agent:graph` — working with a `DeprecationWarning`. The `deepagent-hermes` command remains as an alias of `langstage-hermes`.
- Canonical env vocabulary: `LANGSTAGE_HERMES_*` (and `LANGSTAGE_AGENT_SPEC` for the chat spec), with every `DEEPAGENT_HERMES_*` name still resolving as a fallback — both through `HermesConfig` and at the raw `os.environ` call sites (terminal backends, home resolution, plugins).
- Hermes home: `~/.langstage-hermes` is the new default, but **existing `~/.deepagent-hermes` installs keep winning** when present, so no skills/memories/state are orphaned by the upgrade. `LANGSTAGE_HERMES_HOME` > `DEEPAGENT_HERMES_HOME` > `HERMES_HOME` env overrides. Project config `langstage-hermes.toml` (legacy `deepagent-hermes.toml` still read; new name wins per directory). Six modules that duplicated home resolution now delegate to `config.hermes_home()`.
- Parser pinned `>=0.3,<0.4`.

## [0.1.5] — 2026-06-10

### Added

- **`chat -a/--agent <spec>`** — pick the chat agent explicitly, with the same flag spelling and spec format (`module:attr` / `path/to/file.py:attr`) as `deepagent-code -a` and `cowork-dash run -a`. The flag wins over `DEEPAGENT_AGENT_SPEC`.
- **README: "One agent, every surface"** family table cross-linking all six deep-agent repos.

## [0.1.4] — 2026-06-08

### Added

- **`create_hermes_agent(model=...)` accepts LangChain model instances directly** (#11) — bring-your-own-model instead of only `provider:model` id strings.
- **FIGlet banner** on the chat REPL and bare CLI invocation (#12).
- **`chat` consumes `DEEPAGENT_AGENT_SPEC`** (#13) — the spec env var now actually drives which agent the REPL runs (it was display-only "advisory" before).

## [0.1.3] — 2026-06-04

### Fixed

- **v0.1.2 wheel shipped no Python code** (#10). The explicit `[tool.hatch.build.targets.sdist]` include list was *restrictive*: release CI's `python -m build` (sdist → wheel-from-sdist) produced a wheel with prompts and skills but **zero `.py` files**, so every install crashed with `ModuleNotFoundError: deepagent_hermes.cli`. Removed the restrictive include so hatch ships all tracked files.

## [0.1.2] — 2026-06-04

### Fixed — fresh-install ship-blockers

A first-time-user audit caught three bugs that made v0.1.0 / v0.1.1 unusable straight off PyPI. Every fresh install starting today should land cleanly.

- **Bundled prompts were missing from the wheel.** `pyproject.toml` used `[tool.hatch.build.targets.wheel.shared-data]` to ship `prompts/` — which puts files in `share/` at install time, **not inside the package**. Every `deepagent-hermes chat` died with `prompt not found: combined_review.md`. Moved `prompts/` → `src/deepagent_hermes/_prompts/` so they ship via the normal package-data path.
- **Bundled 26 SKILL.md files were also missing.** Same root cause + `agent.py:_default_skill_dirs` walked `Path(__file__).parent.parent.parent` to find them — which resolved to `Lib/` on PyPI installs. Moved `skills/` → `src/deepagent_hermes/_bundled_skills/` and updated the resolver to look at `Path(__file__).parent / "_bundled_skills"`.
- **`MarkdownProvider` plugin failed to load on every fresh install.** The bundled plugin self-registered via the memory-provider registry on import, but the plugin loader expected a `register(ctx)` function and logged `"Plugin 'markdown-provider' has no callable register()"` whenever you ran `plugins list`. Added a no-op `register(ctx)` that confirms the import side-effect happened.

### Added

- **`deepagent-hermes verify`** — single command that does an end-to-end smoke: checks bundled prompts + skills are packaged, HERMES_HOME is writable, the API key matches the model, builds the agent, makes one real model call, confirms the FTS5 store persisted the turn. Run this first on any fresh install — if it passes, `chat` will work. ~3-5s + ~1¢ on gpt-4o-mini.
- **`[openai]` extras dependency** — `pip install "deepagent-hermes[openai]"` pulls in `langchain-openai>=0.2` for OpenAI / OpenRouter / any OpenAI-wire provider. Previously you got a langchain-internal `ChatOpenAI requires the langchain-openai package` error and had to figure it out from the traceback.
- **README section "Picking a model"** with explicit OpenAI / OpenRouter instructions and a pointer to `verify`.
- **`doctor` now mentions `OPENAI_API_KEY` / `OPENROUTER_API_KEY`** when set, instead of pretending only Anthropic keys count.

### Carried forward from unreleased work on `main`

The v0.1.1 → v0.1.2 window also picked up the UI hookup work that landed on `main` after v0.1.1 shipped (PR #5):

- `skills list / show / install / audit`, `tools`, `curator status / run / pause / resume / pin / unpin` — all real now (were `TBD` stubs in v0.1.1).
- Inline slash commands `/skills /tools /toolsets /cron /curator /memory` work without redirecting to subcommands.
- `session_id` threaded properly through the chat REPL (was being manufactured fresh every turn — broke cross-turn FTS5 lineage).
- Pretty `◆` callouts for `skill_event` / `memory_updated` / `compression_summary` in the chat stream.
- Real bug fixed in the store: `SqliteFtsStore._do_put` silently dropped any non-`messages`/`sessions` namespace — meaning **every curator state write since shipping had been a no-op**. Added an `_KV_NAMESPACES` allow-list (currently `curator_state`).

### Validation

- 423 tests pass / 3 skipped (Docker / Singularity binaries absent, real-model eval skipped without `OPENROUTER_API_KEY`).
- Ruff clean.
- Built local wheel + installed into a fresh `uv` venv + ran `verify` against gpt-4o-mini through OpenRouter: full pass.

[0.1.2]: https://github.com/dkedar7/deepagent-hermes/releases/tag/v0.1.2

## [0.1.1] — 2026-06-03

### Changed — bundled memory provider

- **Replaced `HonchoProvider` with `MarkdownProvider`** as the bundled `MemoryProvider`. The `MemoryProvider` ABC and plug-in slot are unchanged; out-of-tree providers (mem0 / Honcho / embeddings-backed / etc.) can still register via the `deepagent_hermes.plugins` entry-point group.
  - `MarkdownProvider` recalls relevant sections from `<HERMES_HOME>/memories/notes/*.md` via keyword overlap. Splits each `.md` at H1/H2/H3 boundaries; ranks results by matching-token count then by section length (shorter wins on ties for focus); returns top-N sections with a `_From <file>:_` prefix.
  - Pure Python, zero external dependencies. ~60 lines of provider + a ~30-line pure-function recall helper (`search_notes`) that's directly callable from tooling or tests without instantiating the provider.
  - 20 new tests in `tests/test_markdown_provider.py`.
  - Rationale: the bundled `MEMORY.md` / `USER.md` (frozen-snapshot, ≤2200 + ≤1375 chars) already covers the "user model" surface in-prompt. The interesting unmet need was long-form context too big for the system prompt — exactly what hand-authored or agent-written `notes/*.md` solves. A service dependency for that surface didn't pay rent.

### Removed

- `honcho_provider` builtin plug-in directory.
- `tests/test_honcho_provider.py`, `examples/honcho_verify.py`, `examples/dogfood_honcho.py`.
- `[honcho]` optional dependency from `pyproject.toml`.
- `needs_honcho` pytest marker.
- `honcho.json` from `.gitignore` (no longer a known config path).

### Fixed

- Workspace virtual-mode bug surfaced in the v0.1.0 dogfood — `FilesystemBackend` now uses `virtual_mode=True` so the agent's `/workspace/foo.py` paths resolve inside the configured root instead of the literal filesystem `/workspace/`.
- Tools returning `Command` (`skill_view`, `skill_manage`) now inject `tool_call_id` via `Annotated[str, InjectedToolCallId]` instead of hard-coding `""`; LangGraph's ToolNode requires every tool call to produce a matching `ToolMessage`.
- Parallel-write `InvalidUpdateError` on counter state — added `Annotated[T, reducer]` to `iters_since_skill` / `turns_since_memory` / `iteration_budget_remaining` / `memory_snapshot` / `session_id` and friends. Parent + subagent writes in the same superstep now compose cleanly.
- `IterationBudgetMiddleware.before_agent` now seeds when the budget is None **or** 0 (LangGraph coerces `NotRequired[int]` to 0 at schema-merge time, which previously made the seed a no-op and every agent invocation immediately exhausted).

### Verified

- 393 tests pass / 2 skipped (Docker / Singularity gated by binary presence).
- `ruff check` clean across `src/` `tests/` `examples/`.
- Live model round-trips:
  - Single-turn smoke (`examples/live_smoke.py`).
  - 5-turn reflection-trigger trace.
  - 12-turn substantive dogfood — agent autonomously wrote 702 bytes of USER.md across 3 distinct topics and self-refined its own memory at the final turn.
  - 8-turn procedural dogfood — agent autonomously wrote a `SKILL.md` (`python-performance-investigation`).
  - Host-adoption smoke through `deepagent-code`'s `CodeConfig` (full `DEEPAGENT_AGENT_SPEC=deepagent_hermes.agent:graph` round-trip).

[0.1.1]: https://github.com/dkedar7/deepagent-hermes/releases/tag/v0.1.1

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
