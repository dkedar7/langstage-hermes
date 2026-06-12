"""Skill library, loader, validator, and tools.

Public surface:

- :class:`Skill`, :class:`SkillLibrary` — filesystem model
- :func:`validate`, :func:`is_valid` — agentskills.io frontmatter validation
- :func:`build_skills_system_prompt` — prompt-block renderer (cached)
- :class:`SkillLoaderMiddleware` — appends the block to the system prompt
- :func:`make_skill_tools` — bind tools to a specific library
- :func:`skills_list`, :func:`skill_view`, :func:`skill_manage` — default-library
  tool variants (for ad-hoc use / docs)
"""

from __future__ import annotations

from langstage_hermes.skills.library import Skill, SkillLibrary
from langstage_hermes.skills.loader import SkillLoaderMiddleware
from langstage_hermes.skills.prompt import build_skills_system_prompt, clear_prompt_cache
from langstage_hermes.skills.tools import (
    make_skill_tools,
    skill_manage,
    skill_view,
    skills_list,
)
from langstage_hermes.skills.validator import is_valid, validate

__all__ = [
    "Skill",
    "SkillLibrary",
    "SkillLoaderMiddleware",
    "build_skills_system_prompt",
    "clear_prompt_cache",
    "is_valid",
    "make_skill_tools",
    "skill_manage",
    "skill_view",
    "skills_list",
    "validate",
]
