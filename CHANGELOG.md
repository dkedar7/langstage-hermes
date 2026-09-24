# Changelog

All notable changes to `langstage-hermes` (formerly `deepagent-hermes`) will be documented in this file.

## [0.4.33] - 2026-09-24

### Fixed
- **The cron daemon recovers from a stale `.tick.lock` (gh #136).** After a SIGTERM, OOM kill or reboot, the
  fallback `O_EXCL` lockfile was left behind and every later `cron daemon` start failed with "Stale or active
  lockfile ... Remove it". `filelock` is now a declared dependency, so the default lock is an OS lock that the
  kernel releases when the process dies. If `filelock` is ever missing, the fallback reads the PID it recorded:
  a dead PID, or an empty or garbage file, is reclaimed once. Only a lock held by a live process refuses the start.
- **A failed one-shot cron job is kept, with its error (gh #137).** A `once at` job (or any repeat-limited job)
  whose final run failed was deleted exactly like a successful one, taking `last_status` / `last_error` with it.
  It now stays in `cron list` as a disabled job with `state: "error"` and its `last_error`. `cron delete` removes
  it. A successful final run is still removed.
- **Legacy `DEEPAGENT_HERMES_HOME` no longer outranks `HERMES_HOME`, and it warns (gh #145).** The home precedence
  is now `LANGSTAGE_HERMES_HOME` > `HERMES_HOME` > legacy `DEEPAGENT_HERMES_HOME`. A stale legacy value left over
  from the old package used to silently redirect every skill, memory and `state.db` write. When the legacy
  variable is the one used, it prints the same one-time deprecation note as every other `DEEPAGENT_*` variable
  (core's `_warn_legacy_env`, silenced by `LANGSTAGE_SUPPRESS_LEGACY_NOTICE=1`). Every home resolver in the
  package now goes through `config.hermes_home_override()`, and `demo` redirects its throwaway home through
  `LANGSTAGE_HERMES_HOME` + `HERMES_HOME` instead of the legacy variable.
- **Relative paths in `langstage-hermes.toml` / `deepagents.toml` resolve against the TOML file's directory
  (gh #162).** `HermesConfig.resolve()` re-implements TOML layering and bypassed langstage-core 1.0.36's rule, so
  `[workspace] root`, a file-path `[agent] spec` and `[skills] external_dirs` resolved against the cwd. They now
  follow core: a relative value from a TOML file is rebased onto that file's directory, `~` is expanded from every
  source, and env / CLI values stay cwd-relative. `toml_dir_for()` works for hermes configs, and `chat` passes it
  to `load_agent_spec` so a dotted `module:attr` spec from TOML imports relative to its file.
- **`search --session <id> --around <missing message>` and `tools --toolset <unknown>` exit 1 (gh #134).** Both
  printed a not-found message and exited 0. They now match the CLI's named-target-not-found norm. The `tools`
  help no longer advertises a `filesystem` toolset (the real name is `file`).
- **`search` rejects `--session` or `--around` given alone (gh #135).** Outside SCROLL mode they were silently
  dropped, so `search "q" --session <id>` returned unscoped results, even for a session id that doesn't exist,
  with exit 0. Given without its partner, each is now a usage error (exit 2) that names the missing flag.
- **`verify --json` no longer reports `round_trip: "skipped — no key"` when only the aux key is missing
  (gh #146).** "no key" is now reserved for a missing primary key. With the primary key present and the aux key
  missing, it says `skipped — preflight failed (aux model key missing; see model_key_aux)`.
- **`doctor` checks the aux model's provider package (gh #158).** It only probed the main model's package, so an
  `anthropic:*` main + `openai:*` aux config without the `[openai]` extra passed `doctor` and crashed the
  reflection subagent at runtime. When the aux model uses a different provider, `doctor` now reports its package
  (`provider_pkg_aux` in `--json`) and a missing one exits 2 with the `pip install "langstage-hermes[openai]"` hint.
- **`skill_manage(create)` over an existing skill is logged as `write_file` (gh #148).** It forced
  `action="create"`, so an overwrite looked like a first-time create in `audit log`. `library.write` now picks the
  label (`create` for a new skill, `write_file` when it replaces one). Overwriting is still allowed: the reflection
  loop refining a skill it wrote earlier is the normal case (SPEC §9).
- **`audit diff` / `audit show --diff` are readable for agent-authored skills (gh #153).** `library.write` now
  ends every SKILL.md with a newline, and the diff renderer emits git's `\ No newline at end of file` marker
  instead of running the last removed line into the next added one. This also fixes rows recorded before this
  release.

### Changed
- New dependency: `filelock>=3.0` (see gh #136 above).

### Tests
- **The test suite can no longer touch a developer's real hermes home.** Some suites (`test_cli_subcommands.py`,
  `test_rename_shim.py`) wrote `state.db` into whatever `HERMES_HOME` the shell exported. `tests/conftest.py` now
  points `HERMES_HOME`, `LANGSTAGE_HERMES_HOME`, `DEEPAGENT_HERMES_HOME` and `HOME` / `USERPROFILE` at a session
  temp dir when it is imported, and at a fresh per-test dir through an autouse fixture. A test that sets its own
  home with monkeypatch still wins. `test_home_sandbox_guard.py` runs a suite file in a child pytest with every
  home variable pointed at a "real" directory and asserts that directory is left untouched.

## [0.4.32] - 2026-09-24

### Security
- **`skills install` copies only the skill, never its parent directory (gh #159).** Installing from a FILE
  (`skills install ./SKILL.md`, or a differently-named draft) ran `shutil.copytree` on the file's whole parent
  directory, so installing from a repo root or `~/Downloads` copied `.env` secrets, `.git/`, `node_modules/` and
  unrelated binaries into `<HERMES_HOME>/skills/<name>/`, which feeds the agent's context. The rule is now:
  - **File install:** only the named file is installed, as `SKILL.md`. Nothing next to it is copied. To include
    support files (`references/`, `templates/`, `assets/`, `scripts/`), install the skill's directory instead.
  - **Directory install:** the skill directory's contents are copied, except hidden entries (`.git`, `.env*`,
    `.venv`, `.DS_Store`, ...), symlinks (which could point outside the skill, e.g. at `~/.ssh`),
    `node_modules`, `__pycache__`, `venv`, any directory holding a `pyvenv.cfg`, `*.egg-info` and `*.pyc`/`*.pyo`.
    Skipped paths are listed after the `Installed ...` line.
  The `skills install` help no longer tells you to "point this at any working directory". The install audit row
  still records the installed `SKILL.md`, which is now exactly the file that was validated.

## [0.4.31] - 2026-09-24

### Fixed
- **`--show-config` reports a present-but-malformed config file as MALFORMED, not "no config found" (gh #151).**
  When `langstage-hermes.toml` or `$HERMES_HOME/config.toml` existed but failed to parse, stderr said
  `note: ignoring malformed config <path>` while the footer said `TOML: no config found (looked for ...)`, which
  sent users looking for a misplaced file instead of a syntax error. `HermesConfig.resolve` now loads both TOML
  stacks the way langstage-core 1.0.36 does (core's `_load_toml_layers` for `langstage.toml`, and a hermes twin for
  the hermes files) and hands core the found-but-rejected files. The footer now reads
  `TOML: <path> is MALFORMED and was ignored entirely (<error>)`. `HermesConfig.malformed_toml()`,
  `config_dict()["toml"]` (`found` / `malformed` / `malformed_files`) and `config_issues()` report the same thing,
  and `config_issues()` also lists values that hermes' own resolve had to degrade. A missing file still shows the
  hermes `no config found (looked for ...)` line.
- **Cron failure tracebacks follow the resolved `debug` setting.** `cron/scheduler.py` read `LANGSTAGE_DEBUG`
  straight from the environment, so the legacy `DEEPAGENT_DEBUG` and `debug = true` in a TOML file were ignored.
  It now uses `HermesConfig.resolve().debug`, resolved only when a failure is being logged, and falls back to the
  env switch if resolution itself fails.

### Changed
- Requires `langstage-core>=1.0.36`. With it, an agent spec that names a `str` attribute is rejected by core's
  loader ("resolved to a str ..., not an agent") before hermes' own invokable check runs.

## [0.4.30] - 2026-09-23

### Fixed
- **`skills remove <bundled-skill>` no longer deletes the skill out of the installed package (gh #154).** A bundled
  skill with no user shadow resolved to the packaged `_bundled_skills/` tree, and `SkillLibrary.delete` then
  `shutil.move`d it out of `site-packages` into `<wheel>/_bundled_skills/_archived/` — a per-home command silently
  removing the skill for every `HERMES_HOME` and every user on that interpreter (and misreporting the archive as
  under `skills/_archived/`). `delete` now raises `BundledSkillError` for a bundled skill; `skills remove` prints
  why and how to hide it instead (`skills.disabled` / `LANGSTAGE_HERMES_SKILLS_DISABLED`) and exits 1. Removing a
  user skill that shadows a bundled one still archives under `HERMES_HOME` and re-exposes the bundled copy. The
  curator could reach the same path: its lifecycle pass walked bundled skills too, so an install older than
  `archive_after_days` would have its bundled skills archived out of the package by `curator run` / the weekly
  `CuratorMiddleware`. The lifecycle now skips bundled skills entirely, and the new in-place frontmatter writer
  (`SkillLibrary.update_frontmatter`, used by the stale marker and pin/unpin) refuses them too. The agent's
  content edits had the same flaw: `skill_manage` `patch` / `write_file` on a bundled skill rewrote its SKILL.md
  inside the package. They now **copy-on-write** (`SkillLibrary.shadow_bundled`): the whole skill directory is
  copied into the user skills dir and edited there, where it shadows the bundled copy per the documented
  project > user > bundled precedence (SPEC §10.2). A patch or write that fails validation leaves no stray copy.
  `skill_manage` `pin` / `delete` on a bundled skill return a clear tool error, and `audit rollback` refuses a
  pre-0.4.30 row that points inside the package.
- **The curator ages skills by real usage, not SKILL.md mtime (gh #141).** `skill_last_used:<name>` had a reader
  but no writer — and `SqliteFtsStore` silently dropped every `state_meta` put anyway — so the "30/90 days
  inactive" lifecycle was really a file-edit timer that archived skills the agent `skill_view`ed every session.
  The agent's `skill_view` and `skill_manage` tools now stamp `skill_last_used:<name>` in `state.db`
  (`make_skill_tools(library, store=...)`), the store persists the `state_meta` namespace, and both the in-agent
  `CuratorMiddleware` and `curator run` read it. The mtime fallback remains for skills never used via the agent.
- **Pinned skills are exempt from the curator again (gh #119).** The curator read `pinned` from a top-level
  `hermes` block while the agent's `skill_manage(pin)` (and SPEC §9, the validator, `Skill.pinned`) use
  `metadata.hermes.pinned`, so agent-pinned skills were silently archived. The curator, `curator status`, and the
  `/curator` slash command now read pins through one accessor (`is_pinned`, also behind `Skill.pinned`), and
  `curator pin` / `unpin` now go through the agent's pin writer, so both paths write `metadata.hermes.pinned` (with
  an audit row). Pins written by older `curator pin` (top-level `hermes.pinned`) are still honored, and are
  migrated to the nested key on the next pin/unpin.
- **`curator run` actually marks skills `stale` (gh #120).** The stale transition called `library.write(skill)`,
  which matches no real signature; the `TypeError` was swallowed, so nothing was ever marked and the command
  still exited 0. It now rewrites the frontmatter in place via `SkillLibrary.update_frontmatter` (body and location
  preserved, audit action `curator-stale`), putting `lifecycle: stale` under `metadata.hermes`, where `Skill` reads
  it.
- **`demo` no longer writes a skill and a fabricated user preference into a real `HERMES_HOME` (gh #114).** With
  `HERMES_HOME` set (the README's "try it" flow), the demo ran the whole loop against the real home, leaving a
  `profile-slow-python` skill in the user's library and an invented "Prefers a profiling-driven investigation…"
  note in `USER.md` that the frozen memory snapshot then fed to every later real session. The loop now always runs
  in a throwaway home; with `HERMES_HOME` set, only the demo *session* is copied into `<HERMES_HOME>/state.db` so
  `search` still reads it back (gh #88). Re-running `demo` replaces its previous session instead of duplicating it,
  and no other session is touched.
- **A type-mismatched value in `langstage-hermes.toml` degrades with a note instead of crashing (gh #122).** e.g.
  `[memory] nudge_interval = 3.5` or `= "ten"` crashed `--show-config`, `verify` and `skills list` with a
  traceback, while the same typo via an env var already degraded cleanly (gh #83). The TOML layer now keeps the
  value resolved so far and prints core's one-line `note: ignoring malformed …`. Contributed by
  @ethanhawkes-gif in #123 — thank you!

## [0.4.29] - 2026-08-08

### Fixed
- **`cron run-due` / the daemon no longer dump a ~126-line Python traceback on every failed job (gh #111).** When
  a scheduled job's agent invoke failed (an expired/missing key, a rate limit, a network blip), the scheduler
  logged it with `logger.exception(...)`, spilling a full stack trace to stderr — on both the one-shot `cron
  run-due` and the long-running `cron daemon`. The failure is *already* captured cleanly in the job's
  `last_status: error` / `error` field (and in `run-due --json`), so the traceback was pure duplicate noise on an
  automation surface, and inconsistent with the "clean guidance, not raw tracebacks" norm #76 established for
  `chat`. Failures now log **one clean line** (job id + concise cause) by default; the full traceback is preserved
  under `LANGSTAGE_DEBUG` (the family-wide debug switch). The run is still recorded as failed and exit/return
  behavior is unchanged — only the console noise. The top-level job-crash, deliverer-failure, and daemon-tick
  handlers get the same treatment.

## [0.4.28] - 2026-08-06

### Fixed
- **`skills validate` / `skills install` with a file PATH now act on THAT file, never a sibling `SKILL.md` (gh
  #107).** Both commands resolved a file argument to `PATH.parent / "SKILL.md"`, discarding the filename you
  passed — so pointing `validate` at an invalid draft that happened to sit next to a valid `SKILL.md` returned a
  false `✓ valid` / exit 0 about the *sibling*, defeating a command the docstring markets as a CI/pre-commit gate,
  and `install ./draft.md` silently installed the sibling skill instead. Path resolution now honors the exact
  argument: a **directory** resolves to its `SKILL.md` (the documented convention), a **file** is used as-is. An
  invalid draft beside a valid `SKILL.md` is now reported invalid (validate: exit 1; install: rejected, exit 2),
  and `--json`'s `path` field names the file you actually passed. A validator advertised for CI gating can no
  longer return a false PASS about a different file.

### Added
- **`--json` on `doctor` and `verify` — the two "run this first" readiness checks are now machine-readable (gh
  #108).** `doctor` and `verify` are the commands the README tells a fresh adopter to run first, and the two
  you'd gate a setup script / CI job / Dockerfile healthcheck on — yet they were the only user-facing commands
  with no `--json`, leaving a script grepping human prose and inferring cause from an exit code. Both now accept
  `--json`, emitting a top-level `ok` plus a per-check `checks` list (mirroring the `skills audit --json` shape).
  `doctor --json` carries `python` / `langstage-core` / `model` / `api_key` / `provider_pkg` / `hermes_home` /
  `cron_dir` / `bash`; `verify --json` carries `bundled_prompts` / `bundled_skills` / `hermes_home` / `model_key`
  (+ `model_key_aux` on the mixed-provider path, the natural home for the #96 aux preflight) / `round_trip`, plus
  top-level `model` / `model_aux`. `.ok` equals `exit code == 0`, so `doctor --json | jq -e .ok` is a one-liner
  readiness gate; exit codes and the human render are unchanged. `verify --json` reports the live round-trip as
  `skipped` when a required key is absent, so a keyless CI run never triggers a paid model call.
- **`--json` on the `cron` subcommands — the scheduler is now scriptable like every other surface (gh #109).**
  `cron` is the subsystem you'd actually build automation *around*, yet monitoring which jobs exist, their state /
  next run, or what a tick executed meant scraping a fixed-width text table. `cron list` / `create` / `run-due` /
  `delete` / `pause` / `resume` now accept `--json`, each emitting one object with stable keys: `list` →
  `{"jobs":[{id,name,schedule,state,next_run,last_run,last_status,model}],"count"}`; `create` →
  `{id,name,schedule,next_run}`; `run-due` → `{"ran":[{id,name,status,error}],"count"}` so an external tick loop
  can tell what fired and whether anything failed; the mutations → `{id,action,ok[,error]}`. The not-found exit
  codes a prior fix established (gh #97) are preserved: `delete` / `pause` / `resume` still exit 1 on a missing id.

## [0.4.27] - 2026-08-03

### Added
- **`langstage-hermes memory show` — a keyless, offline reader for the frozen-snapshot memory (`MEMORY.md` +
  `USER.md`), gh #101.** The frozen-snapshot memory is the README's *first* headline bullet, yet its only reader
  was the `/memory` slash command inside a keyed `chat`: offline there was no way to see "what has the agent
  learned about me?" (`USER.md`) or "what's in the session snapshot?" (`MEMORY.md`) short of `cat`-ing the files
  by hand. This closes the last agent-only-data gap the keyless-preview line of work (`search` #79, `memory
  notes` #94, `skills validate` #98) was built for. `memory show` reads `<HERMES_HOME>/memories/{USER.md,MEMORY.md}`
  from the same home `doctor` / `--show-config` report — no key, no model, no side effects — and prints each layer
  with its char count vs the configured truncation budget (`memory_char_limit` = 2200, `memory_user_char_limit` =
  1375), the near/over-budget feedback loop `/memory` in chat never gave. `--user` / `--session` narrow to one
  layer; `--json` emits stable per-layer keys (`path`, `exists`, `chars`, `limit`, `over_limit`, `content`)
  mirroring the other `--json` surfaces; `memory dump` is an alias. A missing/empty layer prints a clear one-line
  message (and never creates the files as a read side effect), never a traceback.

### Fixed
- **`demo`'s showcase session no longer records failed `ls` tool calls (gh #102).** The keyless `demo` — the
  headline "watch the reflection→skill loop close" onboarding path — drove its scripted agent with `ls` calls
  carrying `args: {}`. The bundled `ls` tool requires a `path`, so each call failed schema validation and recorded
  an `Error invoking tool 'ls' … path: Field required` ToolMessage into the `demo-001` session — which the README
  then sends brand-new users to inspect with `search` (both SCROLL and DISCOVERY). A new user's first hands-on
  look at the flagship loop was a session full of failures. The scripted `ls` now carries a valid `{"path": "."}`
  (a read-only, side-effect-free cwd listing), so the recorded trace is clean and `search` surfaces success.
- **`verify`'s aux-model key preflight now names the aux model (gh #103, follow-up to #96).** On the documented
  mixed-provider path (an `openai:*` main model with its key set + the default `anthropic:*` aux missing
  `ANTHROPIC_API_KEY`), `verify` printed a bare `model is anthropic:* but ANTHROPIC_API_KEY not set` — two lines
  under an `openai:*` main-model row — reading as if the (fine) main model were the anthropic one at fault. The
  shared `_preflight_model_key` now threads a qualifier so the aux failure reads `aux model is anthropic:* but
  ANTHROPIC_API_KEY not set`, mirroring `doctor`'s `model (aux)` labelling so the two commands can't drift.
- **`doctor` now exits non-zero when the configured model's required API key is missing (gh #104).** `doctor`
  printed the required-key failure (`ANTHROPIC_API_KEY: not set (required …)`) but still exited 0 — disagreeing
  with `verify` (exit 2) and with `doctor`'s own missing-provider-package path (exit 2, #41), so a `✗` coexisted
  with a clean bill of health and the exit code was useless as a CI/scripted health gate. `doctor` now exits 2
  when the main model's — or a distinct-scheme aux model's — required key is missing, deferring the exit so the
  full diagnostic still prints. The sibling of #41 (missing package) it never covered.

## [0.4.26] - 2026-07-31

### Added
- **`langstage-hermes skills validate PATH [--json]` — a keyless, non-mutating pre-install check for a
  `SKILL.md` (gh #98).** Skill creation is the project's headline activity, yet it was the one authoring
  loop with no offline red/green: the only ways to run the agentskills.io validator were `skills install`
  (which copies the skill into `<HERMES_HOME>/skills` **and** appends a `create` audit row) or `skills audit`
  (which only checks already-installed skills). `validate` runs the **exact same** validator against an
  arbitrary working tree and writes nothing — no store copy, no audit row, no model, no key — so an author
  or a CI gate for a skills repo can ask "did I get the frontmatter right?" without side effects. PATH
  mirrors `skills install` (a `SKILL.md` file or a directory containing one) and the effective name is
  resolved exactly as `install` resolves it, so `validate`'s verdict predicts `install`'s. Exit 0 valid,
  1 invalid; `--json` emits `{"path", "name", "valid", "errors"}` mirroring the other `--json` surfaces.
  This closes the keyless-preview set — `search` (#79), `memory notes` (#94), and now skills.

### Fixed
- **`verify` / `doctor` now preflight `model_aux`, not only the main model (gh #96).** The reflection review
  subagent — the project's headline closed-loop feature — runs on `model_aux`. On the documented mixed-provider
  path (an `openai:*` main model + the default `anthropic:*` aux), a user who set only the main model and
  `OPENAI_API_KEY` sailed through `verify` green, then hit an Anthropic auth error at the first reflection
  (~iteration 10). Both commands now run the same provider-aware key preflight against the aux model too:
  `verify` fails preflight (exit 2) before building the agent, and `doctor` prints a `model (aux):` row plus
  its distinct key requirement — surfaced only when the aux model uses a different provider scheme than the
  main model, so the default all-anthropic config is unchanged.
- **`cron delete` / `cron pause` / `cron resume` now exit 1 on a non-existent job id (gh #97).** They printed
  `No cron job with id 'X'.` but exited 0, so a CI/script wrapper around the cron lifecycle could not tell a
  typo'd or stale id from a real success (`cron pause $ID && echo ok` printed `ok` even when nothing was
  paused). They now exit 1 like every other not-found path in the CLI (`skills remove`, `audit rollback`, …),
  keeping the same message.
- **`search --session <missing> --around N` (SCROLL) now exits 1 when the named session doesn't exist
  (gh #99).** It printed `scroll: session_id 'X' not found` (and the `--json` body carried a matching `error`
  key) but exited 0 — the one not-found path in `search` that broke the CLI's exit-1 norm (a sibling of #97
  in a different subsystem). A missing *message* inside an *existing* session still exits 0 (the session was
  found; only the anchor id was out of range), as does an empty discovery result.

## [0.4.25] - 2026-07-26

### Added
- **A keyless `langstage-hermes memory notes "<query>"` CLI previews what the bundled
  `MarkdownProvider` will recall — offline, no model (gh #94).** The MarkdownProvider is a headline
  memory feature (drop hand-authored notes in `<HERMES_HOME>/memories/notes/*.md` and the agent
  surfaces relevant sections on demand), but its only reader was the live agent — needing an API key
  and a model turn to see what a note would surface. This is exactly the gap the keyless `search` CLI
  (gh #79) closed for the FTS5 session store; the notes store now gets the same treatment. The new
  nested `memory notes` subcommand resolves `<HERMES_HOME>/memories/notes` from the same config path
  `doctor` / `--show-config` report, calls the already-exported pure `search_notes(...)` function
  (near-zero new retrieval logic), and prints the source file + matching section so hits are
  actionable. `--limit N` caps the sections returned; `--json` emits `{"query", "count", "results"}`
  mirroring `search --json` (each result carries `file` / `section` / `snippet`, with `file` parsed
  back out of the `_From <file>:_` prefix). A missing/empty notes dir or a no-match query prints a
  clear message (and never creates the dir as a read-side effect), never a traceback. Note authors
  finally have an offline feedback loop. Implemented as Option A (a nested `memory notes` command),
  which fit the existing Click group structure (`skills`, `audit`, `cron`, ...) cleanly.

## [0.4.24] - 2026-07-25

### Fixed
- **An unrecognized boolean `LANGSTAGE_HERMES_*` env value no longer silently flips a default off
  (gh #92).** A value like `LANGSTAGE_HERMES_MEMORY_ENABLED=enabled` — a natural way to try to
  *enable* the memory subsystem — coerced silently to `False` (disabling it), with `--show-config`
  even crediting `[env:...]` as if the value was honored. This contradicted the policy already
  established for malformed *numeric* env vars in #83 (a `note:` + fallback to the field default).
  The boolean env fields now use langstage-core's strict boolean caster (`_env_bool_strict`, added
  in **langstage-core 1.0.29**), which raises on an unrecognized value so `HermesConfig.resolve()`'s
  guard emits the same one-line `note:` and keeps the field default — booleans and numbers now
  degrade consistently. Recognized values (`1/true/yes/on`, `0/false/no/off`) are unaffected. The
  lenient `_env_bool` is still used for the direct suppress-notice flag reads, where a typo should
  mean "off", not warn. Requires **langstage-core >= 1.0.29**.

## [0.4.23] - 2026-07-25

### Fixed
- **`--json` is now honored on `skills list`, `skills audit`, and `audit log`, so the README's
  cross-command scripting/CI claim is true instead of a crash (gh #90).** The keyless `search` CLI
  (gh #79) shipped a `--json` mode, and its README section advertised that flag as a *cross-command*
  surface — "`--json` emits structured output for scripting/CI, mirroring `audit`/`skills`" — but the
  implementation wired `--json` onto `search` alone. A scripting/CI user who followed the docs and ran
  `langstage-hermes skills list --json` (or `skills audit --json`, or `audit log --json`) hit a hard
  `click` `No such option: '--json'` and exit 2 — a crash-out, not a graceful "unsupported". The
  phrasing traces straight to #79's proposal, which described `--json` as mirroring how `audit`/`skills`
  "already print structured state"; that wording landed in the README, but the wiring didn't. Rather
  than walk the docs back, we honored them: the three commands the reporter reached for while
  dogfooding all already computed clean structured state, so each now grows a `--json` flag that dumps
  exactly that state as one JSON object on stdout, matching `search --json`'s conventions (stable keys,
  a `count`, `default=str` serialization). `skills list --json` → `{"skills": [{name, category,
  description, version, path}...], "count", "load_errors"}` (full untruncated descriptions; skills
  dropped for broken frontmatter surface under `load_errors` inside the payload instead of as stderr
  noise, so stdout stays a single pure JSON object). `skills audit --json` → `{"ok", "skill_count",
  "failed_count", "results": [{name, ok, errors}...]}`. `audit log --json` → `{"mutations": [{id,
  timestamp, skill_name, action, source, session_id, tool_call_id, skill_path, before_hash,
  after_hash}...], "count"}` (honors `--skill`/`--limit`; carries every scalar field regardless of the
  human-only `--full` toggle, but not the SKILL.md blobs — those stay in `audit show`). `--json`
  changes only the rendering, never the contract: each command's human output is byte-for-byte
  unchanged, and `skills audit --json` keeps the human command's exit code (1 when any skill fails,
  0 otherwise) so CI gets the pass/fail signal without parsing. The README's `search` section was
  rewritten to name exactly these commands, so the docs and the actual `--help`/behavior now agree
  exactly. Verified keyless end-to-end: all three repro commands from the issue now emit valid JSON
  and exit cleanly.

## [0.4.22] - 2026-07-23

### Fixed
- **`demo` now populates the store `search` reads, so the documented keyless `demo` → `search`
  loop actually closes (gh #88).** The keyless `search` CLI (gh #79) reads `<HERMES_HOME>/state.db`,
  and both the README *and* `search`'s own empty-store hint tell you to run `langstage-hermes demo`
  to fill it. But `demo` always wrote to a throwaway `tempfile.mkdtemp(prefix="dah-demo-")` home and
  `rmtree`d it on exit — so the demo store and the search store were structurally in different
  directories and could never coincide. A user who followed the docs verbatim (`export HERMES_HOME=…`;
  `demo`; `search "python"`) got an empty store forever, and the empty-store hint looped them straight
  back to `demo`. (This is the exact trap where gh #79's own verification only found the demo data by
  pointing `HERMES_HOME` at the *kept* `/tmp/dah-demo-…` dir — a path no doc mentions.) `demo` now
  **honours an explicitly-set `HERMES_HOME`**: when `LANGSTAGE_HERMES_HOME` / legacy
  `DEEPAGENT_HERMES_HOME` / `HERMES_HOME` is set it records the session into that real
  `<HERMES_HOME>/state.db` (and never removes it), so `search` reads the demo session straight back —
  exactly what the docs promise. With *no* home env var set, `demo` still uses a throwaway home and
  cleans up on exit, so a bare `demo` never litters the default `~/.langstage-hermes` (the gh #69
  no-pollution property is preserved for the case that wanted it). `search`'s empty-store hint and the
  README's "populate a store" line were rewritten to match: they now name the `HERMES_HOME`
  requirement instead of pointing an unset user at a `demo` that can't populate their default store.
  Verified end-to-end keyless: `HERMES_HOME=<dir> langstage-hermes demo` then
  `HERMES_HOME=<dir> langstage-hermes search "python"` returns the demo's `demo-001` session.

## [0.4.21] - 2026-07-23

### Fixed
- **A malformed numeric `LANGSTAGE_HERMES_*` env var no longer crashes the CLI with an uncaught
  `ValueError` (gh #83).** A single bad char — `LANGSTAGE_HERMES_MEMORY_NUDGE_INTERVAL=10x`,
  `COMPRESSION_THRESHOLD=high` — made `HermesConfig.resolve()` raise straight out and dump a
  traceback (exit 1), taking down `--show-config`, `verify`, `skills list`, `curator status`, and
  `plugins list` — including the exact commands the README tells a new user to run first. This is
  the env-side mirror of the malformed-TOML handling that already degraded gracefully. `HermesConfig`
  copies the base env-casting loop (it layers two TOML sources, so it can't just inherit
  `HostConfig.resolve()`); that copy is now guarded the same way the base was in **langstage-core
  1.0.23** (#104), reusing core's `_warn_malformed_env_value` helper so the wording can't drift: the
  bad value is caught, a one-line `note:` goes to stderr, and resolution keeps the value from the
  layer beneath (a `langstage-hermes.toml` value if set, else the default) rather than crashing.
  Requires **langstage-core >= 1.0.23**.

## [0.4.20] - 2026-07-23

### Added
- **`langstage-hermes search` — a keyless, offline CLI over the FTS5 session store (gh #79).** The
  README sells FTS5 session search as a headline capability, but the only reader was the in-chat
  `session_search` *agent tool* — reachable only from `chat`, i.e. behind a live model + API key. Every
  other SQLite-backed subsystem here (`audit`, `skills`, `cron`, `curator`) has a read surface; the one
  *headline* store did not, and it was exactly the keyless/offline audience the project courts who could
  populate `state.db` (via `demo`/`verify`) but never read it back. `search` closes that gap with the same
  three modes SPEC §13.3 documents, printing `session_id` + `message_id` so hits are actionable:
  - **DISCOVERY** — `langstage-hermes search "profile slow python" [--limit N] [--json]`: BM25 top-N with a
    highlighted snippet, deduped by session lineage.
  - **SCROLL** — `langstage-hermes search --session <sid> --around <msg_id> [--window N]`: a ±window view
    centred on an anchor message (clamped 1–20).
  - **BROWSE** — `langstage-hermes search --browse [--limit N]` (also the default when no query is given):
    recent sessions, newest first.
  It reuses the existing retrieval in `search/session_search.py` (a new `search_sessions_structured`
  twin of `run_session_search` that returns plain dicts) against `<HERMES_HOME>/state.db` — no model, no
  key, no network — and honours `HERMES_HOME` exactly as `doctor`/`--show-config` report it. A `--json`
  mode mirrors how `audit`/`skills` already print structured state, for scripting/CI. Unlike the in-chat
  tool (which hides tool-role rows the agent already has), the human CLI searches **all** roles within a
  real session — so the crystallised-skill / session-summary rows land in results (the `demo` store's
  "profile slow python" skill is one) — while still excluding internal reflection-fork *sessions*
  (`source=tool`). An absent or empty store prints a clear one-line message pointing at `demo`/`chat`,
  never a traceback, and never creates an empty DB as a side effect of a read.

### Fixed
- **`chat` now honours `[agent] spec` from a TOML config — it was silently dropped while `--show-config`
  reported it as active (gh #85).** `chat` loaded a custom agent only from the `-a/--agent` flag or the
  `LANGSTAGE_AGENT_SPEC` (legacy `DEEPAGENT_AGENT_SPEC`) env var. The equivalent `[agent] spec` TOML key —
  a first-class `HostConfig` field that `--show-config` fully resolves and attributes to the file
  (`toml: agent.spec`) — was never consulted at runtime: `chat` fell back to the built-in hermes agent as
  if nothing were configured. That was an advertised-vs-honored gap on the exact command the README points
  users to for trust, made worse because the **sibling console script in the same wheel**,
  `langstage-agui`, *does* honor `[agent] spec` — so two entry points shipped by one package disagreed on
  a documented key, and the primary diagnostic (`--show-config`) green-lit a setting the runtime dropped
  (the same trust-in-diagnostics theme as #61/#83). Root cause: `_resolve_agent` read the env var only and
  was never handed `cfg.agent_spec`, even though `chat` loaded the config a few lines later. The fix
  threads the resolved `cfg.agent_spec` into agent resolution as the layer **below** the flag and env var,
  mirroring `langstage-agui` (which passes `--agent` as an override then loads `cfg.agent_spec`): precedence
  is now `-a` flag > `LANGSTAGE_AGENT_SPEC` env > `[agent] spec` TOML > built-in default — matching how
  every other config field layers. The chat context block also surfaces a TOML-sourced spec as `(active)`,
  so the diagnostic no longer goes dark just because the spec came from a file rather than an env var.
  Proven keyless with the issue's `echo_agent.py`: a `[agent] spec` in `langstage-hermes.toml` now runs the
  custom graph (`MARKER-custom-agent-ran`) with no API key, instead of hitting the built-in hermes
  ANTHROPIC preflight.
- **A syntactically valid but unrecognized key in `langstage-hermes.toml` now warns on stderr instead of
  vanishing without a trace (gh #84).** It was the last silent config-failure path: the CLI already emits a
  `note:` for a malformed file (#61) and for deprecated env vars, but a mistyped key produced no stdout, no
  stderr, no `--show-config` signal — the value simply never took effect. It is an easy mistake because the
  accepted key names are internally inconsistent within a single section (`memory.memory_enabled` keeps the
  prefix while `memory.nudge_interval`/`memory.provider` drop it; `model_aux` is spelled `model.aux_model`),
  so the *natural* first guess (`[memory] enabled`, `[model] aux`) silently falls back to the default. Now,
  after resolution, each hermes TOML file's leaf keys are diffed against the resolver's own
  `field → toml-key` map and each genuinely-unknown key gets a one-line `note:` naming it and (via a
  difflib near-match) the closest accepted key: `note: unknown config key '[memory] enabled' in
  langstage-hermes.toml (ignored). Did you mean 'memory_enabled'?`. Care was taken to avoid false positives:
  the recognized set is built from the **live** `HermesConfig._toml_map()` (base `HostConfig` keys like
  `agent.spec` included) so it can never drift from what actually resolves; `[section]` headers are never
  flagged (only leaves are); and free-form tables whose sub-keys are *data* — the dict-valued
  `skills.platform_disabled` (keyed by arbitrary platform names) and the `configurable` passthrough — are
  skipped. It covers **both** hermes config files (the global `$HERMES_HOME/config.toml` and the project
  `langstage-hermes.toml`, plus the legacy `deepagent-hermes.toml`), but not the shared cross-host
  `langstage.toml` (which legitimately carries keys for *other* hosts). It is a **warning only** — never a
  hard error, never a changed exit code (a config file that suddenly failed to load would be a far worse
  regression); ASCII-only for cp1252 Windows consoles; deduped per (file, key); suppressed under pytest and
  via `LANGSTAGE_SUPPRESS_UNKNOWN_KEY_NOTICE=1`.

## [0.4.19] - 2026-07-19

### Fixed
- **A `SKILL.md` with unparseable YAML frontmatter was dropped from the library in total silence — gone
  from `skills list` *and* from the live agent, with no warning, no error, exit 0 (gh #81).** One missing
  quote, one bad indent, one unterminated `[` and the skill simply ceased to exist: `skills list` printed
  an unchanged count with empty stderr, and — because the agent factory routes through the *same*
  `SkillLibrary.list()` loader — `chat` started up without it and never said why. The only trace was a
  `logger.debug("skipping unparseable skill at %s: %s", ...)` in `SkillLibrary._scan_directory`, which sits
  below the CLI's default log level and so reached nobody. The intent there was right (warn and keep going
  — one bad file must not take down the other 26 skills); the *level* was wrong. This was the mirror image
  of every other skill error surface: `skills install` rejects invalid frontmatter loudly (exit 2),
  `skills audit` reports it as `__error__/<dir>: parse failure: ...`, and only the two surfaces a user
  actually reaches — "did my skill load?" and the running agent — stayed quiet, leaving a
  *why-won't-my-skill-load* trap that could only be escaped by already knowing to run `skills audit`.
  Both surfaces now name the offending directory, the file and the parse error on **stderr**, and point at
  `skills audit` for the full report. Behaviour is otherwise unchanged: still a warning, never a failure —
  `skills list` still exits 0 and still lists every skill that *does* parse, and one broken skill still
  cannot stop the agent from starting.
- **Root cause of the silence, and why the fix did not add a second copy.** `_scan_directory` (the loader)
  and `validate_all` (behind `skills audit`) each carried their own `try: frontmatter.load(...) / except`
  with their own wording — the same detect-and-phrase logic in two places, which is exactly the shape #78
  had to unwind for the provider→package table. Rather than grow a third, both are now built from shared
  primitives in `library.py`: `_error_key()` (the `__error__/<dir>` key), `_parse_failure()` (the canonical
  `parse failure: <exc>` wording, byte-identical between a warning and its audit row), a `SkillLoadError`
  record, and `format_load_error()` for presentation. Every scan collects the drops onto
  `SkillLibrary.load_errors`, and the two production callers surface that list on their own channel — the
  CLI via `click.echo(..., err=True)`, the agent via `logging` — so the diagnostic stays off `skills list`'s
  stdout, which is a listing that gets piped and grepped (the command has no `--json` mode to keep clean).
  `format_load_error()` collapses whitespace, because a `yaml.scanner.ScannerError` stringifies to four
  lines, and stays ASCII-only for Windows cp1252 consoles.
- **The agent got the warning at *build* time, not once per model call.** `SkillLoaderMiddleware` re-scans
  the library on every model call, so warning from inside the scan would have repeated the same line for a
  whole REPL session. The scan therefore stays quiet (it only records), and `create_hermes_agent` scans once
  and warns once while the graph is being built — before the `chat` banner, so a user who never runs
  `skills list` still learns their skill vanished. Regression tests are built on the issue's clean-room
  repro (a valid skill plus one neighbor with a deliberate YAML typo) and pin all of it: the warning's
  content, that it lands on stderr and never on stdout, that it fires even when `--query`/`--category`
  filters hide everything, that `skills list` still exits 0 and the valid neighbor still lists, that the
  agent still builds and warns exactly once, that the per-model-call scan never logs at WARNING, that a
  healthy library raises no false alarm, and that the warning and the `skills audit` row are generated from
  the same detection. They fail before the fix and pass after.

## [0.4.18] - 2026-07-18

### Fixed
- **`chat`'s build failure for a missing provider package named the WRONG package — `pip install
  langchain-openai` instead of the documented `[openai]` extra (gh #78).** On a plain
  `pip install langstage-hermes` (no extras), a user following the README's OpenAI/OpenRouter Quick
  start (`langstage-hermes chat --model openai:openai/gpt-4o-mini`) hit
  `Failed to build agent: Initializing ChatOpenAI requires the langchain-openai package. Please install
  it with 'pip install langchain-openai'` — because `chat`'s agent-build `except` handler printed
  `Failed to build agent: {e}` verbatim, leaking langchain's raw `ImportError`. That message points
  **away** from `pip install "langstage-hermes[openai]"`, which the README documents as *the* supported
  way to add OpenAI support, and it diverged from `verify`/`doctor` — the exact commands the README's
  Verify section says to run first — which had already been taught to name the hermes extra (#33, #41).
  This was the last un-fixed member of that preflight-consistency family: #76 added `chat`'s missing-*key*
  gate, but the missing-*provider-package* hint on the same build path was never given the same
  treatment. `chat` now appends the same guidance `verify` gives
  (`for OpenAI-compatible models install: pip install "langstage-hermes[openai]"`), keeping the existing
  failure shape (exit 2 with a message, no crash) — only the content changed.
- **Root cause of the drift: the provider→package mapping existed in three places.** `verify` string-matched
  `"langchain-openai" in str(e)`, `doctor` carried its own `{"openai:": ("langchain_openai", …)}` table, and
  `chat` had nothing — which is precisely how #78 survived the #41 and #76 fixes. All three now read from a
  single module-level `_PROVIDER_PACKAGES` table (model-id prefix → importable module, install command,
  provider label) via shared `_provider_package()` / `_missing_provider_install_line()` helpers, so a fourth
  divergent copy can't appear. The table covers the same provider set `doctor` already knew — `openai:*`
  (behind the `[openai]` extra, the only model-provider extra `pyproject.toml` declares) and `anthropic:*`
  (a base dependency, so its hint stays the plain package; there is no hermes extra to name) — and the
  guidance is no longer OpenAI-shaped: any prefix in the table gets its own line. Detection also stopped
  being a bare substring match on one hard-coded distribution name: an `ImportError` qualifies when it names
  the provider's module (either spelling) **or** when that module is genuinely not importable, so a reworded
  langchain message still gets the hint while an unrelated build failure (or an unknown/custom provider like
  `ollama:*`) gets none. Regression tests assert `chat` names the extra for `openai:*` and the plain package
  for `anthropic:*`, that an unrelated `RuntimeError` gets no misleading install hint, that `verify`/`doctor`
  still emit their unchanged gold-standard lines from the shared table, and pin the helper's contract. The
  `ImportError` is simulated rather than depending on `langchain_openai` being absent, so the tests hold on a
  machine that has the extra installed; they fail before the fix and pass after.

## [0.4.17] - 2026-07-16

### Fixed
- **`chat` — the headline Quick Start command — had NO provider-aware API-key preflight, so a missing or
  invalid provider key leaked a raw provider exception instead of the clean guidance `verify`/`doctor`
  already give (gh #76).** `verify` and `doctor` both gate on the *configured model's* key (the
  #33 / #35 / #41 preflight-consistency work), but `chat` was never given the gate: on the default
  `anthropic:*` path the REPL opened, accepted the user's first message, and only *then* surfaced a bare
  `TypeError: Could not resolve authentication method …` mid-session; on the `openai:*` path it leaked a
  bare `Missing credentials …` at build. `chat` now runs the **same** preflight **before** building the
  agent / entering the REPL and exits 2 with the same clean, colored message (e.g.
  `✗ model is anthropic:* but ANTHROPIC_API_KEY not set`) plus a pointer to run `verify`/`doctor`. The
  gate `verify` already implemented was factored into a shared `_preflight_model_key(model_for_run)`
  helper now used by both `verify` and `chat`, so the two can't drift again. The gate runs only for the
  built-in factory (which builds the model from `model.default`); a BYO agent spec owns its own model, so
  `chat -a <spec>` is left untouched (no #33-style false positive). Regression tests assert `chat` with a
  missing anthropic/openai key exits 2 with the clean guidance and no raw traceback, that a valid key (or
  `OPENROUTER_API_KEY` for `openai:*`, or a spec graph) still proceeds to the REPL, and pin the shared
  helper's contract.

## [0.4.16] - 2026-07-15

### Fixed
- **`skills.disabled` / `skills.platform_disabled` was a silent no-op — a disabled skill still loaded
  into the agent and still showed in `skills list` (gh #74).** The value resolved correctly through
  env / TOML and `--show-config` advertised it as active, but `SkillLibrary.list()`'s
  `config["disabled"]` / `config["platform_disabled"]` filter was dead code: **no production caller
  ever passed `config=`**, so `SkillLibrary.config` was always `{}`. A user who disabled a skill to
  keep it out of the agent's context found it silently still injected into the toolset and the
  `skills_list` tool. Every runtime construction site now threads the resolved config in via a new
  `HermesConfig.skills_filter_config()` helper (`{"disabled": …, "platform_disabled": …}`): the agent
  runtime (`agent.py`), the CLI's `_skill_library` backing `skills list`/`show` (`cli.py`), and the
  module-level `skills_list`/`skill_view` tool library (`skills/tools.py`). A skill named in
  `skills.disabled` (or `skills.platform_disabled[<session_platform>]`) is now excluded from both the
  loaded set and `skills list`; a non-disabled skill is unaffected. The bundled-count health check in
  `check`/`doctor` deliberately keeps counting the full bundled set (it must catch a packaging bug
  regardless of a user's disabled list). Regression tests assert the exclusion through each production
  caller (agent, CLI, default tool library) and cover both the `disabled` and `platform_disabled`
  knobs; they fail before the wiring and pass after.

## [0.4.15] - 2026-07-14

### Fixed
- **A bare-duration schedule (`"30m"`, `"2h"`, `"1d"`) parsed as a one-shot instead of a recurring
  interval (gh #71).** The module docstring, the `cronjob` tool example, and the invalid-schedule hint
  all document a bare duration as a recurring interval identical to `"every 30m"`, but `parse_schedule`
  fell through to the one-shot fallback (`kind: "once"`, `display: "once in 30m"`). A "run my morning
  brief" job created as `30m`/`1d` therefore fired exactly once and retired itself to `completed` — the
  opposite of the documented contract — with no warning. A bare duration now parses to
  `kind: "interval"`, so `"30m"` means exactly the same thing as `"every 30m"` and the job keeps
  re-firing on schedule; `"once at <ts>"` / an ISO timestamp remain the way to request a genuine
  one-shot. Regression tests assert the interval kind, alias-equivalence with `"every …"`, and that a
  bare-duration job reschedules (stays `scheduled`, is not retired) after it runs.
- **A failed scheduled agent invoke was recorded as `last_status: ok` and its error text delivered as
  the job result (gh #72).** When a cron job's agent invoke raised (expired/rotated key, rate-limit,
  provider outage, bad model name, …), `_build_cron_response` swallowed the exception and *returned it
  as a normal string* (`[agent invoke failed: …]`). `run_job`'s agent branch — unlike the `no_agent`
  script branch — never revisited `success`, so `mark_job_run(success=True, error=None)` fired: the
  tick summary printed `<id>: ok`, `jobs.json` recorded `last_status: "ok"` / `last_error: None`, and
  the configured deliverer (`local` / `stdout` / `agentmail`) sent the literal error string in place of
  the brief/digest. Failures were invisible to the operator and any success-keyed retry/alerting never
  fired. `_build_cron_response` now returns `(ok, body)` exactly like `_run_script`, so the agent path
  mirrors the script path: a raised invoke sets `success=False` / `last_status="error"` /
  `last_error=<msg>`, delivery is suppressed on failure (gated on `ok`), and the error is still saved
  to the output doc as an audit trail. Regression tests drive `run_job` with a raising agent seam and
  assert the failure surfaces and nothing is delivered.

## [0.4.14] - 2026-07-12

### Added
- **`langstage-hermes demo` — a keyless / offline way to watch the reflection→skill-creation loop
  close (gh #69).** The headline feature is the closed reflection→skill-creation loop, but every
  documented way to see it close needed a paid API key and a live multi-turn session (`verify` does a
  single ≤20-token round-trip and deliberately never runs enough iterations to trigger the loop;
  `examples/dogfood*.py` and `chat` all require a live model). So the one thing a new adopter most wants
  to confirm — *does the skill loop really work, and what does a generated `SKILL.md` look like?* — was
  exactly the thing they couldn't try before committing a key. The new `demo` subcommand drives the
  **real** shipped machinery — `create_hermes_agent` with the genuine `ReflectionMiddleware`, the
  `task` review-subagent dispatch, the real `skill_manage` / `memory` tools, `SkillLibrary.write()`,
  the audit log and the FTS5 store — against a **scripted fake model** (threaded in via the
  bring-your-own-model `model=` / `aux_model=` kwargs `create_hermes_agent` already accepts). No
  network, no API key, fully deterministic. It writes a real `SKILL.md` + memory note under a throwaway
  `HERMES_HOME`, prints the generated skill's frontmatter and body, then cleans up (`--keep-workspace`
  keeps it for inspection). Reusable programmatically via `langstage_hermes.demo.run_demo(...)`, so CI
  (and downstream adopters) can smoke-test the genuine loop with no secrets — exercising the real
  `SkillLibrary` loader + reflection middleware, not a file-glob stand-in.

## [0.4.13] - 2026-07-12

### Fixed
- **`verify` leaked its `/tmp/dah-verify-*` isolated workspace on every run (gh #68).** The command
  created the workspace with `tempfile.mkdtemp(prefix="dah-verify-")` but never removed it — there was
  no `try/finally`, no `shutil.rmtree`, and no `TemporaryDirectory` — so **every** invocation that got
  past the model-key gate leaked one dir: on the agent-build-failure path, the model-invoke-failure
  path, and the `VERIFY: PASS` success path alike. Because the README tells users to "run this first on
  any fresh install" (exactly the command re-run while debugging a key/model setup), the leaked dirs
  accumulated one-per-run precisely when the command is used as intended. The workspace is now wrapped
  in a `try/finally` that removes it on **every** exit path — success and failure. A new
  `--keep-workspace` flag is the opt-out for anyone who wants to inspect the workspace post-mortem (it
  prints where the dir was kept). Regression tests drive the real `verify` command (build-failure and
  success paths) and assert nothing is left behind; run against the pre-fix code they fail.

## [0.4.12] - 2026-07-10

### Fixed
- **README "Load into an existing host" snippet wrote a bare `spec =` with no `[agent]`
  table, so the agent was silently not loaded (gh #66).** The block told the reader to set
  the spec "under `[agent]`", but the command it gave — `echo 'spec = "…"' >> langstage.toml`
  — appended a **top-level** `spec` key. The resolver reads `agent.spec` (i.e. `spec` under
  `[agent]`), so a top-level `spec` is ignored and `agent_spec` stays `None` — a user
  copy-pasting the block got the default agent with no error to tell them why. The snippet
  now writes the table header: `printf '[agent]\nspec = "langstage_hermes.agent:graph"\n' >>
  langstage.toml`. A new test runs exactly the documented command and asserts it configures
  the agent, so a revert to the bare form fails CI.

## [0.4.11] - 2026-07-09

### Fixed
- **The `--show-config` "no config found" list omitted `deepagent-hermes.toml`, though that
  file IS searched and honored (gh #64).** The diagnostic's "looked for" string was
  hardcoded and had drifted from the resolver, which reads the legacy
  `deepagent-hermes.toml` (new `langstage-hermes.toml` wins per directory) exactly as the
  CHANGELOG promises. A migrating user with a legacy project TOML was told by the built-in
  debugger that their working file wasn't in the search path. The list is now built from the
  same `HERMES_PROJECT_TOML` / `LEGACY_HERMES_PROJECT_TOML` constants the resolver iterates,
  so it can't claim a narrower search than what runs. A test asserts every honored filename
  appears in the line. Follow-up to #57.

## [0.4.10] - 2026-07-08

### Changed
- **`--show-config` and the REPL `/config` are locked to the single config renderer
  (config-diagnostic consolidation).** Both already called `cfg.describe()`, but the
  `HermesConfig.describe()` override silently dropped the new `omit_keys` / `configurable`
  arguments; it now forwards them to the base renderer (**langstage-core 1.0.14**, now the
  minimum pin), and a parity test asserts the REPL `/config` renders byte-for-byte what
  `--show-config` prints — so the two config views can't drift, closing the seam behind the
  recurring `--show-config` bugs (#55/#57/#61).

### Fixed
- **`--show-config` no longer reports a malformed `langstage-hermes.toml` as "TOML read
  from: <it>", and its warning prints once (gh #61).** A `langstage-hermes.toml` with a
  syntax error is correctly ignored (config falls back to env + defaults), but `--show-config`
  still listed the file as read — contradicting its own "ignoring malformed config" note —
  and the note printed twice (the loader plus the source-labeling re-read each warned). The
  underlying `_read_toml` fix landed in **langstage-core 1.0.13** (now the minimum pin, which
  records malformed paths + dedupes the warning); `load_hermes_toml_config` now skips listing
  a malformed file too.

### Fixed
- **The `/compress` REPL command actually compresses now, instead of a stub (gh #59).**
  The README Quick start advertises `/compress` ("force context compression"), but the
  handler was an unwired stub that printed "not yet wired… v0.2 task" and did nothing. It
  now builds a `HermesCompressionMiddleware` from the resolved config — the same way the
  agent wires it — and force-runs the compression pipeline on the current session's history
  in place (head/tail protection kept, the middle summarised), reporting the before→after
  token estimate and message count. Falls back to a non-model summary when the summariser
  can't run, so it works even keyless (unless `compression.abort_on_summary_failure` is set).

### Fixed
- **`--show-config` now names the *resolved* global config path, so it doesn't
  misdirect under a custom `HERMES_HOME` (gh #57).** The global config lives at
  `$HERMES_HOME/config.toml` and moves with `HERMES_HOME` — but the "no config found"
  diagnostic (and the docs) hardcoded `~/.langstage-hermes/config.toml`. A user who set a
  custom `HERMES_HOME` and placed their global config at the *documented* path found it
  silently ignored, while `--show-config` claimed it looked at `~/.langstage-hermes` — which
  it did not. The diagnostic now prints the real `$HERMES_HOME/config.toml`; the README and
  docstrings state that the global config moves with `HERMES_HOME`. (Resolution itself was
  already correct — config placed at `$HERMES_HOME/config.toml` always loaded.)

## [0.4.6] - 2026-07-05

### Fixed
- **`--show-config` now attributes each value to the file it actually came from
  (gh #55).** With a global `~/.langstage-hermes/config.toml` and a project
  `langstage-hermes.toml` both present, every TOML-resolved field was labeled with the
  *last file read* (`langstage-hermes.toml`) — so a value living only in the global
  config was misreported as coming from the project file, defeating the whole point of
  `--show-config` (telling you where a value came from). It now names the
  highest-precedence file that actually sets each key. Runtime resolution was already
  correct; this was a diagnostic-label bug only.

## [0.4.5] - 2026-07-04

### Fixed
- **`skills list`/`show`/`audit` now search `config.skills.external_dirs` (gh #52).**
  The CLI's `_skill_library` hardcoded `[bundled, hermes_home/skills, project]` and
  never added `cfg.skills_external_dirs`, while the runtime agent (`_default_skill_dirs`)
  does — so `audit`, a validation gate, silently skipped external skills and reported a
  false green for ones the agent will happily load. `_skill_library` now appends the
  configured external dirs too, matching the runtime resolution.

## [0.4.4] - 2026-07-03

### Fixed
- **`chat` now roots the agent at the resolved workspace, not the launch dir
  (ADR 0005).** The `chat` path built the factory as `create_hermes_agent(cfg)`
  without forwarding a workspace, so a resolved `--workspace` / `langstage-hermes.toml`
  / env `workspace_root` was silently dropped and the agent's filesystem operated in
  the launch cwd (only `verify` forwarded it). `chat` now calls
  `core.apply_workspace(cfg.workspace_root)` before building, and the factory
  defaults its workspace to `core.workspace_root()` instead of `cwd` — so the
  resolved root reaches the agent's `FilesystemBackend`. A standalone
  `create_hermes_agent()` is unchanged (`workspace_root()` falls back to cwd). The
  chat header now shows the actual resolved workspace. Requires `langstage-core>=1.0.7`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3] — 2026-07-03

### Fixed
- **`doctor` reported "langgraph-stream-parser: installed" (gh #49)** — a retired
  package name. The check imports the renamed `langstage_core`, so the label now
  reads `langstage-core`. Also scrubbed the remaining `langgraph-stream-parser`
  mentions from the agent-spec error message and internal docstrings.

## [0.4.2] — 2026-07-02

### Fixed
- **`chat` crashed on every message (gh #43).** The 0.4.0 AG-UI migration made the
  host drive the graph *asynchronously* (`astream`), but three parts of hermes were
  sync-only and had never been exercised through a real turn:
  1. **Checkpointer** — the graph used the sync `SqliteSaver`, which has no async
     methods (`"SqliteSaver does not support async methods"`). Switched to
     `InMemorySaver` (async-safe + loop-agnostic; hermes' turn loop opens a fresh
     event loop per turn, which rules out a long-lived `AsyncSqliteSaver`).
     Conversation/interrupt state still persists across turns within a session;
     hermes' durable memory (FTS5 store + snapshots) is unaffected. Durable
     cross-restart checkpointing is a follow-up.
  2. **Middleware** — `SkillLoaderMiddleware` defined only sync `wrap_model_call`,
     so the async path failed with `"awrap_model_call is not available"`. Added the
     async pair (the caching / prompt / event-bus middleware already had theirs).
     A new structural test asserts every model-call middleware has the async pair.
  3. **Recursion limit** — hermes compiles with `.with_config(recursion_limit=1000)`
     (its multi-middleware graph burns past the default 25 per turn), but the AG-UI
     adapter built its own run config and dropped it, so a normal tool-using turn
     died with `"Recursion limit of 25 reached"`. `build_session_agent` now forwards
     the graph's config to the adapter.

  Verified end-to-end against a live model: a plain reply and a multi-step
  tool-using turn (write a file → confirm) both complete cleanly.

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
