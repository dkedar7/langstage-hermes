# Bundled skills

v0.1.0a0 ships **no** bundled skills (decision per SPEC §21.1 and the autonomous-build directive).

Drop your own SKILL.md files into `~/.deepagent-hermes/skills/<name>/SKILL.md` per the [agentskills.io specification](https://agentskills.io/specification.md), or let the agent autonomously create them during its reflection passes (every ~10 tool iterations by default).

A curated subset of Hermes Agent's bundled skills (data-science, software-development, github, research, devops) will land in a future `deepagent-hermes-skills` companion package.
