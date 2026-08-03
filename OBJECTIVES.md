# Objectives & scope — langstage-hermes

*What this repo is for, who it serves, and what it deliberately is **not** — the yardstick
for deciding whether a proposed change or filed issue belongs here. When triaging an issue,
start here.*

## Objective

A batteries-included, **opinionated agent** whose differentiators are **durable memory** and
**skills**. Unlike the other family repos, this is not a surface or an SDK — it is a
ready-to-run agent product:

- **Memory** — an FTS5 session store (`search`) and a `MarkdownProvider` for hand-authored
  long-form notes (`memory notes`), previewable keyless and offline.
- **Skills** — `SKILL.md` loading, plus keyless `skills validate` / `skills audit`.
- A rich CLI around them (`search`, `memory`, `skills`, `cron`, `curator`, `doctor`/`verify`).

## Who it's for

Someone who wants a capable agent with memory and skills out of the box — not to assemble one
from primitives.

## In scope

- The memory and skills subsystems and their **keyless preview/validation** CLIs.
- Provider-aware preflight that checks every model actually used (main **and** `model_aux`).
- Curator, cron, and plugins **in service of** the memory/skills mission.
- Consistent CLI ergonomics — exit codes, `--json`, honest not-found paths.

## Out of scope (anti-scope)

- Drifting into a general agent **framework** or a plugin marketplace. It is one opinionated
  agent; memory and skills are the point, not arbitrary extensibility.
- Bridge/surface concerns (the AG-UI wire, config resolution) — those belong in
  **langstage-core**.
- Architecture/config knobs added for their own sake rather than a memory/skills need.

## How this fits the family

langstage-hermes is the family's **opinionated agent product**, built on langstage-core — the
one repo that is a *thing you run*, not a surface you drive or an SDK you import. Shared wire or
config behavior belongs in core; the surfaces (web, cli, jupyter, vscode) are separate.

## Using this to triage

Before acting on an issue or PR: does it serve the objective above? Is it in scope or
anti-scope? Weigh its value — **security > correctness > advertised-≠-honored > DX/docs >
polish > net-new feature** — against the cost of a manual release. Then **fix, defer, or
decline with a reason.** Not every filed issue is worth acting on.
