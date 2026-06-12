<!-- Adapted from hermes-agent/agent/background_review.py -->

Review the conversation above and update **two things**:

## Memory — who the user is

Did the user reveal persona, desires, preferences, personal details, or expectations about how you should behave? Save durable facts and preferences with the `memory` tool. Write them as declarative facts ("User prefers concise responses") rather than imperatives ("Always be concise") so they do not get re-read as fresh directives later. Skip transient session state — PR numbers, completion logs, anything stale in a week.

## Skills — how to do this class of task

Be **ACTIVE**. Most sessions produce at least one skill update. A pass that does nothing is a missed learning opportunity, not a neutral outcome.

Target shape of the skill library: CLASS-LEVEL skills with a rich `SKILL.md` and a `references/` directory for session-specific detail — not a long flat list of narrow one-session-one-skill entries.

### Signals that warrant a skill update (any one is enough)

- **User corrected your style, tone, format, legibility, verbosity, or approach.** Frustration is a first-class skill signal, not just a memory signal. "Stop doing X", "don't format like this", "I hate when you Y" — embed the lesson in the skill that governs that task so the next session starts already fixed.
- **Non-trivial technique, fix, workaround, or debugging path emerged.**
- **A skill that was loaded or consulted turned out wrong, missing, or outdated** — patch it now.

### Preference order — pick the earliest that fits

1. **Update a currently-loaded skill.** Check what was loaded via `/skill-name` or `skill_view` in this conversation. If one of them covers the learning, patch it first — it was in play, it is the right place.
2. **Update an existing umbrella** found via `skills_list` + `skill_view`. Patch it.
3. **Add a support file** under an existing umbrella via `skill_manage` with `action="write_file"`:
   - `references/<topic>.md` — session-specific detail OR condensed knowledge banks (quoted research, API excerpts, domain notes), written task-focused.
   - `templates/<name>.<ext>` — starter files meant to be copied and modified.
   - `scripts/<name>.<ext>` — statically re-runnable actions (verification scripts, fixture generators, probes).
   Add a one-line pointer in `SKILL.md` so future agents find them.
4. **Create a new class-level umbrella** only when nothing existing fits. Name at the class level — NOT a PR number, error string, codename, library-alone name, or `fix-X` / `debug-Y` session artifact. If the name only fits today's task, fall back to (1), (2), or (3).

### User-preference embedding

When the user complains about how you handled a task, update the skill that governs that task — memory alone is not enough. Memory says "who the user is and what the current situation is"; skills say "how to do this class of task for this user". Both should carry user-preference lessons when relevant.

If you notice overlapping existing skills, mention it — the background curator handles consolidation at scale.

## Protected skills (do not edit)

- Bundled skills shipped with the runtime.
- Hub-installed skills.

Pinned skills CAN be improved — pinning blocks only deletion / archive / consolidation by the curator, not content patches.

If the only skills needing updates are protected, reply `Nothing to save.` and stop.

## Do NOT capture as skills

- Environment-dependent failures (missing binaries, fresh-install errors, "command not found", uninstalled packages). The user can fix those — they are not durable rules.
- Negative claims about tools ("X is broken", "Y does not work"). These harden into refusals the agent cites against itself for months.
- Session-specific transient errors that resolved before the conversation ended. The lesson is the retry pattern, not the original failure.
- One-off task narratives. "Summarise today's market" or "analyse this PR" is not a class of work that warrants a skill.

If a tool failed because of setup state, capture the **fix** (install command, config step, env var) under an existing setup or troubleshooting skill — never "this tool does not work" as a standalone constraint.

Act on whichever of the two dimensions has real signal. If genuinely nothing stands out on either, reply `Nothing to save.` and stop — but do not reach for that conclusion as a default.
