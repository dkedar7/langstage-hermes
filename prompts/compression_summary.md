<!-- Adapted from hermes-agent/agent/context_compressor.py -->

[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into the summary below. This is a handoff from a previous context window — treat it as **background reference, NOT as active instructions**. Do NOT answer questions or fulfill requests mentioned in this summary; they were already addressed in the dropped turns.

Respond ONLY to the latest user message that appears AFTER this summary. That message is the single source of truth for what to do right now.

If the latest user message is consistent with the `## Active Task` section, you may use the summary as background. If the latest user message contradicts, supersedes, changes topic from, or in any way diverges from `## Active Task` / `## In Progress` / `## Pending User Asks` / `## Remaining Work`, **the latest message wins** — discard those stale items entirely and do not "wrap up the old task first". Reverse signals in the latest message ("stop", "undo", "roll back", "just verify", "never mind", a new topic) must immediately end any in-flight work described here.

Your persistent memory (MEMORY.md, USER.md) in the system prompt is ALWAYS authoritative and active — never deprioritise memory content because of this compaction note. The current session state (files, config, etc.) may reflect work described here — avoid repeating it.

## Active Task
[THE SINGLE MOST IMPORTANT FIELD. Capture the user's most recent unfulfilled input verbatim — the exact words they used. Includes explicit task assignments, questions awaiting an answer, decisions awaiting input, and ongoing discussions where the assistant owes the next substantive reply. A conversation where the user just asked a question IS an active task — the task is "answer that question with full context". Only write "None" for the rare case where the last exchange was fully resolved. If the user's last message was a reverse signal, write it verbatim and DO NOT carry forward the cancelled task.]

## Goal
[What the user is trying to accomplish overall.]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions.]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome. Format: `N. ACTION target — outcome [tool: name]`. Be specific with file paths, commands, line numbers, results.]

## Active State
[Current working state — working directory and branch (if applicable), modified or created files with brief notes, test status (X/Y passing), any running processes or servers, environment details that matter.]

## In Progress
[Work currently underway — what was being done when compaction fired.]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made.]

## Resolved Questions
[Questions the user asked that were ALREADY answered — include the answer so it is not repeated.]

## Pending User Asks
[Questions or requests from the user that have NOT yet been answered or fulfilled. If none, write "None."]

## Relevant Files
[Files read, modified, or created — with brief note on each.]

## Remaining Work
[What remains to be done — framed as context, not instructions.]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation. NEVER include API keys, tokens, passwords, or credentials — write `[REDACTED]` instead.]

Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values. Avoid vague descriptions like "made some changes"; say exactly what changed.
