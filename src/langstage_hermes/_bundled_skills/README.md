# Bundled skills

v0.1.0a1 ships **26 SKILL.md files** curated from [Nous Research's Hermes Agent](https://github.com/nousresearch/hermes-agent) bundle (MIT licensed; attribution in [`/NOTICE`](../NOTICE)). The set is aimed at this project owner's data-science / software-development / research / devops usage shape.

Categories:

- `software-development/` — `systematic-debugging`, `test-driven-development`, `plan`, `writing-plans`, `python-debugpy`, `spike`, `requesting-code-review`, `hermes-agent-skill-authoring`
- `github/` — `github-auth`, `github-code-review`, `github-issues`, `github-pr-workflow`, `github-repo-management`, `codebase-inspection`
- `research/` — `arxiv`, `blogwatcher`, `research-paper-writing`, `llm-wiki`
- `data-science/` — `jupyter-live-kernel`
- `mlops/` — `huggingface-hub`, `evaluation` (with `evaluating-llms-harness` sub-skill), `vector-databases`
- `productivity/` — `notion`, `google-workspace`, `powerpoint`
- `note-taking/` — `obsidian`

Every bundled SKILL.md validates against the [agentskills.io spec](https://agentskills.io/specification.md) — enforced by `tests/test_bundled_skills.py`.

## Layering with your own

The library resolves dirs in this order (later wins on name collision):

1. **Bundled** — this directory
2. **User** — `~/.deepagent-hermes/skills/<name>/SKILL.md`
3. **Project shadow** — `./.deepagent-hermes/skills/<name>/SKILL.md`
4. **Extras** — anything in `config.skills.external_dirs`

So you can override any bundled skill by dropping a same-named SKILL.md into your user or project dir.

## Autonomous skill creation

The whole point of this agent: when the reflection middleware fires (default every 10 tool iterations), the review subagent reads the conversation and writes a new SKILL.md to your user library — or patches an existing one. The bundled set is the *starter library*, not the ceiling.

See [SPEC §9](../SPEC.md#9-reflection--skill-creation-spec-2-the-differentiator) for the reflection loop, [SPEC §10](../SPEC.md#10-skill-format-and-library-spec-3) for the library layout.
