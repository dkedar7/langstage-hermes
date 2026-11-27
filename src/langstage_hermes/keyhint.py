"""Name an unset provider key in Hermes' own error lines (gh #240).

langstage-core 1.0.69 appends ``(hint: ANTHROPIC_API_KEY is not set)`` to the
connection error a provider client raises when its key is missing (core #253),
but only on the lines core formats. Hermes formats its own for cron results and
``verify``, so they go through these two wrappers. With an older core (no
``langstage_core.keyhint``) they return the plain text. Only the variable's name
is ever printed, never a value.
"""

from __future__ import annotations

__all__ = ["describe_error", "with_key_hint"]


def describe_error(exc: BaseException) -> str:
    """``"Type: message"``, plus core's missing-key hint when one applies."""
    try:
        from langstage_core.keyhint import describe_error as _describe
    except ImportError:  # langstage-core < 1.0.69
        return f"{type(exc).__name__}: {exc}"
    return _describe(exc)


def with_key_hint(message: str, error: BaseException | str | None = None) -> str:
    """``message`` with core's missing-key hint appended on the same line, once."""
    try:
        from langstage_core.keyhint import with_key_hint as _with
    except ImportError:  # langstage-core < 1.0.69
        return message
    return _with(message, error)
