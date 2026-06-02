<!-- Adapted from hermes-agent/agent/background_review.py -->

Review the conversation above and update the skill library.

Be **ACTIVE**. Most sessions produce at least one skill update — even a small one. A pass that does nothing is a missed learning opportunity, not a neutral outcome. "Nothing to save" is a real option, but it should not be the default.

## Target shape of the library

CLASS-LEVEL skills, each with a rich `SKILL.md` and a `references/` directory for session-specific detail. Not a long flat list of narrow one-session-one-skill entries. This shapes HOW you update, not WHETHER you update.

## Signals that warrant an update (any one is enough)

- **User corrected your style, tone, format, legibility, verbosity, or approach.** Frustration signals like "stop doing X", "this is too verbose", "don't format like that", "why are you explaining", "just give me the answer", "you always do Y and I hate it", or an explicit "remember this" are **first-class skill signals**, not just memory signals. Embed the lesson in the relevant skill so the next session starts already corrected.
- **User corrected your workflow, approach, or sequence of steps.** Encode the correction as a pitfall or explicit step in the skill that governs that class of task.
- **Non-trivial technique, fix, workaround, debugging path, or tool-usage pattern emerged.** Capture it.
- **A skill that got loaded this session turned out wrong, missing a step, or outdated.** Patch it NOW.

## Preference order — pick the earliest action that fits

1. **Update a currently-loaded skill.** Look back through the conversation for skills loaded via `/skill-name` or `skill_view`. If any covers the territory of the new learning, patch that one first — it was in play, so it is the right place to extend.
2. **Update an existing umbrella** found via `skills_list` + `skill_view`. Add a subsection, a pitfall, or broaden a trigger.
3. **Add a support file** under an existing umbrella:
   - `references/<topic>.md` — session-specific detail or a condensed knowledge bank (quoted research, API excerpts, provider quirks, reproduction recipes). Task-focused; not a mirror of upstream docs.
   - `templates/<name>.<ext>` — starter files meant to be copied and modified.
   - `scripts/<name>.<ext>` — statically re-runnable actions (verification scripts, fixture generators, probes).
   Add via `skill_manage` with `action="write_file"` and a path under one of those directories, then add a one-line pointer in the umbrella's `SKILL.md` so future agents find it.
4. **Create a new class-level umbrella** only when nothing existing fits. The name MUST be at the class level — not a PR number, error string, feature codename, library-alone name, or `fix-X` / `debug-Y` / `audit-Z` session artifact. If the name only makes sense for today's task, fall back to (1), (2), or (3).

## User-preference embedding

When the user expresses a style, format, or workflow preference, the lesson belongs in the SKILL.md body — not only in memory. Memory captures "who the user is and what the current situation is"; skills capture "how to do this class of task for this user". When the user complains about how you handled a task, the skill that governs that task needs to carry the lesson.

If you notice two existing skills that overlap, mention it in your reply — the background curator handles consolidation at scale.

## Protected skills (do not edit)

- Bundled skills shipped with the runtime.
- Hub-installed skills.

Pinned skills CAN be improved — pinning blocks only deletion / archive / consolidation by the curator, not content patches. Patch them when a pitfall or missing step turns up.

If the only skills needing updates are protected, reply `Nothing to save.` and stop.

## Do NOT capture (these become self-imposed constraints that bite later)

- Environment-dependent failures: missing binaries, fresh-install errors, post-migration path mismatches, "command not found", unconfigured credentials, uninstalled packages. The user can fix these — they are not durable rules.
- Negative claims about tools or features ("browser tools do not work", "X tool is broken"). These harden into refusals that the agent cites against itself for months after the actual problem was fixed.
- Session-specific transient errors that resolved before the conversation ended. If retrying worked, the lesson is the retry pattern, not the original failure.
- One-off task narratives. "Summarise today's market" or "analyse this PR" is not a class of work that warrants a skill.

If a tool failed because of setup state, capture the **fix** (install command, config step, env var) under an existing setup or troubleshooting skill — never "this tool does not work" as a standalone constraint.
