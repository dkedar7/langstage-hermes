<!-- Adapted from hermes-agent/agent/background_review.py -->

Review the conversation above and consider whether anything is worth saving to memory.

Focus on:

1. Has the user revealed things about themselves — persona, desires, preferences, recurring patterns, or personal details that will still matter in a week?
2. Has the user expressed expectations about how you should behave, their working style, or ways they want you to operate?

If something stands out, save it using the `memory` tool. Write it as a declarative fact, not as an instruction to yourself ("User prefers concise responses" — yes; "Always respond concisely" — no), so it does not get re-read as a fresh directive in a later session.

Do **not** save: completed task narratives, PR numbers, commit SHAs, "today I fixed X", file counts, transient TODO state, or any artifact that will be stale in seven days. Those belong in `session_search`, not memory.

If nothing is worth saving, reply with exactly `Nothing to save.` and stop.
