"""Prompt-injection / promptware / exfiltration pattern scanner.

Port of Hermes's ``tools/threat_patterns.py`` — single source of truth for the
patterns that the memory snapshot, context-file scanner, and tool-result
delimiter system all want to share.

Patterns are organized by **attack class**, not by source file. Each pattern is
a ``(regex, pattern_id, scope)`` triple where ``scope`` controls which scanners
apply it:

- ``"all"`` — applied everywhere (classic prompt injection, exfiltration).
- ``"context"`` — applied to context files + memory snapshot + tool results
  (promptware / C2 / behavioral hijack).
- ``"strict"`` — applied to user-mediated writes (memory tool, skill installs)
  where the user can resolve false positives interactively.

Our SPEC §13.1 calls these scopes ``"memory"`` / ``"context"`` / ``"tool_result"``.
We accept both vocabularies — ``"memory"`` is aliased to ``"strict"`` (the
broadest set, which is correct for memory entries because they are user-curated
and persist into the frozen snapshot for the entire session), and
``"tool_result"`` is aliased to ``"context"``.

Design note (DO NOT regress): patterns are anchored on **attack-specific
vocabulary** (C2 verbs, named frameworks, persistence paths), not on generic
"bossy English". A pattern like ``you\\s+must\\s+...`` matches AGENTS.md,
CLAUDE.md, and most legitimate instruction-writing, so we anchor "you must" to
C2 verbs (register/connect/report/beacon) instead. Same logic for "ignore" —
``ignore\\s+previous\\s+instructions`` is the attack, ``ignore the previous
suggestion`` is legitimate disagreement.
"""

from __future__ import annotations

import re
from typing import Optional

# Each entry: (regex, pattern_id, scope). Patterns compiled lazily at module load.
# scope ∈ {"all", "context", "strict"}.
_PATTERNS: list[tuple[str, str, str]] = [
    # ── Classic prompt injection (applies everywhere) ────────────────
    (
        r"ignore\s+(?:\w+\s+)*(previous|all|above|prior)\s+(?:\w+\s+)*instructions",
        "prompt_injection",
        "all",
    ),
    (r"system\s+prompt\s+override", "sys_prompt_override", "all"),
    (
        r"disregard\s+(?:\w+\s+)*(your|all|any|the\s+above|previous|prior)"
        r"\s+(?:\w+\s+)*(instructions|rules|guidelines)",
        "disregard_rules",
        "all",
    ),
    (
        r"act\s+as\s+(if|though)\s+(?:\w+\s+)*you\s+(?:\w+\s+)*(have\s+no|don\'t\s+have)"
        r"\s+(?:\w+\s+)*(restrictions|limits|rules)",
        "bypass_restrictions",
        "all",
    ),
    (
        r"<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->",
        "html_comment_injection",
        "all",
    ),
    (r'<\s*div\s+style\s*=\s*["\'][\s\S]*?display\s*:\s*none', "hidden_div", "all"),
    (
        r"<\s*\|?\s*(?:system|im_start|im_end)\s*\|?\s*>",
        "system_tag_injection",
        "all",
    ),
    (r"do\s+not\s+(?:\w+\s+)*tell\s+(?:\w+\s+)*the\s+user", "deception_hide", "all"),

    # ── Role-play / identity hijack ──────────────────────────────────
    (r"you\s+are\s+(?:\w+\s+)*now\s+(?:a|an|the)\s+", "role_hijack", "context"),
    (r"pretend\s+(?:\w+\s+)*(you\s+are|to\s+be)\s+", "role_pretend", "context"),
    (
        r"output\s+(?:\w+\s+)*(system|initial)\s+prompt",
        "leak_system_prompt",
        "context",
    ),
    (
        r"(respond|answer|reply)\s+without\s+(?:\w+\s+)*(restrictions|limitations|filters|safety)",
        "remove_filters",
        "context",
    ),
    (r"\bname\s+yourself\s+\w+", "identity_override", "context"),

    # ── C2 / promptware vocabulary (context scope) ───────────────────
    (
        r"you\s+must\s+(?:\w+\s+){0,3}(register|connect|report|beacon)\b",
        "forced_action",
        "context",
    ),
    (r"only\s+use\s+one[\s\-]?liners?\b", "anti_forensic_oneliner", "context"),
    (
        r"never\s+(?:\w+\s+)*(?:create|write)\s+(?:\w+\s+)*(?:script|file)\s+(?:\w+\s+)*disk",
        "anti_forensic_disk",
        "context",
    ),
    (
        r"\b(?:cobalt\s*strike|sliver|havoc|mythic|brainworm)\b",
        "known_c2_framework",
        "context",
    ),

    # ── Exfiltration (applies everywhere) ────────────────────────────
    (
        r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)",
        "exfil_curl",
        "all",
    ),
    (
        r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)",
        "exfil_wget",
        "all",
    ),
    (
        r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)",
        "read_secrets",
        "all",
    ),

    # ── Persistence / strict-only patterns ───────────────────────────
    (r"authorized_keys", "ssh_backdoor", "strict"),
    (
        r"(?:api[_-]?key|token|secret|password)\s*[=:]\s*[\"'][A-Za-z0-9+/=_-]{20,}",
        "hardcoded_secret",
        "strict",
    ),
]

# Invisible / bidirectional unicode characters used in injection attacks.
INVISIBLE_CHARS: frozenset[str] = frozenset(
    {
        "​",  # zero-width space
        "‌",  # zero-width non-joiner
        "‍",  # zero-width joiner
        "⁠",  # word joiner
        "⁢",  # invisible times
        "⁣",  # invisible separator
        "⁤",  # invisible plus
        "﻿",  # zero-width no-break space (BOM)
        "‪",  # left-to-right embedding
        "‫",  # right-to-left embedding
        "‬",  # pop directional formatting
        "‭",  # left-to-right override
        "‮",  # right-to-left override
        "⁦",  # left-to-right isolate
        "⁧",  # right-to-left isolate
        "⁨",  # first strong isolate
        "⁩",  # pop directional isolate
    }
)

# Combining-mark density threshold for obfuscation detection.
# A normal latin string has ~0% combining marks; Zalgo/heavy-stack attacks
# routinely run 30%+. We trip at 25% to leave room for legitimate non-Latin
# scripts that use combining marks naturally (Vietnamese, Sanskrit) without
# tripping a single accent in an English sentence.
_COMBINING_MARK_DENSITY_THRESHOLD = 0.25
_COMBINING_MARK_MIN_LENGTH = 20  # don't bother flagging short strings

_SCOPE_ALIASES = {
    "memory": "strict",
    "tool_result": "context",
    "all": "all",
    "context": "context",
    "strict": "strict",
}

# Compiled pattern sets, indexed by canonical scope name.
_COMPILED: dict[str, list[tuple[re.Pattern[str], str]]] = {}


def _compile() -> None:
    """Compile patterns once at import time, grouped by canonical scope.

    A pattern with scope ``"all"`` lands in every set. ``"context"`` lands in
    context + strict (context implies strict — strict scanners want everything
    a context scanner catches plus more). ``"strict"`` lands in strict only.
    """
    global _COMPILED
    if _COMPILED:
        return

    all_set: list[tuple[re.Pattern[str], str]] = []
    context_set: list[tuple[re.Pattern[str], str]] = []
    strict_set: list[tuple[re.Pattern[str], str]] = []

    for pattern, pid, scope in _PATTERNS:
        compiled = re.compile(pattern, re.IGNORECASE)
        entry = (compiled, pid)
        if scope == "all":
            all_set.append(entry)
            context_set.append(entry)
            strict_set.append(entry)
        elif scope == "context":
            context_set.append(entry)
            strict_set.append(entry)
        elif scope == "strict":
            strict_set.append(entry)
        else:
            raise ValueError(f"threat_patterns: unknown scope {scope!r} for pattern {pid!r}")

    _COMPILED = {"all": all_set, "context": context_set, "strict": strict_set}


_compile()


def _combining_mark_hits(text: str) -> Optional[str]:
    """Return an attack id if combining-mark density is suspiciously high."""
    if len(text) < _COMBINING_MARK_MIN_LENGTH:
        return None
    import unicodedata

    marks = sum(1 for ch in text if unicodedata.category(ch) == "Mn")
    density = marks / len(text)
    if density >= _COMBINING_MARK_DENSITY_THRESHOLD:
        return f"combining_mark_density_{int(density * 100)}pct"
    return None


def scan_for_threats(text: str, *, scope: str = "memory") -> list[str]:
    """Return a list of matched pattern IDs in ``text`` at the given scope.

    Scope vocabulary (both Hermes and SPEC §13.1 names accepted):

    - ``"all"`` — classic injection + exfil only.
    - ``"context"`` / ``"tool_result"`` — adds promptware / C2 / role-play.
    - ``"strict"`` / ``"memory"`` — adds persistence / hardcoded-secret patterns.

    Also reports invisible unicode characters (as
    ``"invisible_unicode_U+XXXX"``) and abnormal combining-mark density
    (as ``"combining_mark_density_<pct>pct"``).
    """
    if not text:
        return []

    canonical = _SCOPE_ALIASES.get(scope)
    if canonical is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")

    findings: list[str] = []

    # Invisible unicode (set intersection — one O(n) pass, not 17 ``in`` checks).
    invisible_hits = set(text) & INVISIBLE_CHARS
    for ch in sorted(invisible_hits):  # sorted for stable test output
        findings.append(f"invisible_unicode_U+{ord(ch):04X}")

    # Combining-mark density (Zalgo / obfuscation)
    density_hit = _combining_mark_hits(text)
    if density_hit:
        findings.append(density_hit)

    # Regex patterns
    for compiled, pid in _COMPILED[canonical]:
        if compiled.search(text):
            findings.append(pid)

    return findings


def scan(text: str, *, scope: str = "memory") -> Optional[str]:
    """Return a single human-readable reason if ``text`` matches, else ``None``.

    Convenience wrapper for the most common case — first-hit blocking. The
    memory snapshot scanner uses this to decide whether to replace an entry
    with ``"[BLOCKED: <reason>]"``.
    """
    findings = scan_for_threats(text, scope=scope)
    if not findings:
        return None
    pid = findings[0]
    if pid.startswith("invisible_unicode_"):
        codepoint = pid.replace("invisible_unicode_", "")
        return f"contains invisible unicode character {codepoint}"
    if pid.startswith("combining_mark_density_"):
        pct = pid.replace("combining_mark_density_", "")
        return f"abnormal combining-mark density ({pct})"
    return f"matches threat pattern '{pid}'"


__all__ = [
    "INVISIBLE_CHARS",
    "scan",
    "scan_for_threats",
]
