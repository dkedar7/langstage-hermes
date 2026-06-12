<!-- Adapted from hermes-agent/agent/prompt_builder.py -->

The skill library is how this runtime compounds knowledge across sessions. The shape of the library matters: it is a small set of **class-level umbrella skills**, each with a rich `SKILL.md` and a `references/` directory for session-specific detail — not a long flat list of one-session-one-skill micro-entries.

When to update the library:

- After a complex task (~5+ tool calls), a tricky fix, a non-trivial workflow, or a new debugging path — capture the approach via `skill_manage` so the next session can reproduce it.
- When a skill you loaded turns out outdated, incomplete, or wrong — patch it immediately with `skill_manage(action="patch")`. Don't wait to be asked. A stale skill that nobody fixes becomes a liability.
- When the user corrects your style, format, tone, verbosity, or workflow — that is a **first-class skill signal**, not just a memory signal. Embed the lesson in the skill that governs the task so the next session starts already corrected.

Preference order for any update:
1. Patch a currently-loaded skill if it covers the territory.
2. Patch an existing umbrella found via `skills_list` + `skill_view`.
3. Add a `references/<topic>.md`, `templates/<file>`, or `scripts/<file>` under an existing umbrella, and add a one-line pointer in `SKILL.md` so the next agent finds it.
4. Only as a last resort create a new class-level umbrella — and name it at the **class** level, not after today's specific PR number, error string, or feature codename.
