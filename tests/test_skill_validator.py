"""Tests for ``langstage_hermes.skills.validator``.

Covers every agentskills.io rule plus the Hermes extensions.
"""

from __future__ import annotations

from langstage_hermes.skills.validator import (
    MAX_COMPATIBILITY_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    is_valid,
    validate,
)

# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------


def test_minimal_valid_frontmatter():
    fm = {"name": "pdf-processing", "description": "Process PDFs."}
    assert validate(fm) == []
    assert is_valid(fm)


def test_full_valid_frontmatter():
    fm = {
        "name": "pdf-processing",
        "description": "Extract PDF text, fill forms, merge files.",
        "license": "Apache-2.0",
        "compatibility": "Requires Python 3.14+",
        "allowed-tools": "Bash(git:*) Read",
        "metadata": {
            "author": "example-org",
            "version": "1.0",
            "hermes": {"tags": ["pdf"], "related_skills": ["docx"]},
        },
        "version": "1.0",
        "platforms": ["macos", "linux"],
        "prerequisites": {"env_vars": ["PDFTK"], "commands": ["pdftk"]},
    }
    assert validate(fm) == []


def test_parent_dir_must_match_name():
    fm = {"name": "pdf-processing", "description": "x"}
    assert validate(fm, parent_dir_name="pdf-processing") == []
    errs = validate(fm, parent_dir_name="wrong-dir")
    assert any("parent directory" in e for e in errs)


# ---------------------------------------------------------------------------
# Name rules
# ---------------------------------------------------------------------------


def test_name_required():
    errs = validate({"description": "x"})
    assert any("name" in e and "required" in e for e in errs)


def test_name_must_be_string():
    errs = validate({"name": 123, "description": "x"})
    assert any("name" in e and "string" in e for e in errs)


def test_name_uppercase_rejected():
    errs = validate({"name": "PDF-Processing", "description": "x"})
    assert any("lowercase" in e for e in errs)


def test_name_underscore_rejected():
    errs = validate({"name": "pdf_processing", "description": "x"})
    assert any("lowercase" in e or "hyphens" in e for e in errs)


def test_name_leading_hyphen_rejected():
    errs = validate({"name": "-pdf", "description": "x"})
    assert any("hyphen" in e for e in errs)


def test_name_trailing_hyphen_rejected():
    errs = validate({"name": "pdf-", "description": "x"})
    assert any("hyphen" in e for e in errs)


def test_name_consecutive_hyphens_rejected():
    errs = validate({"name": "pdf--processing", "description": "x"})
    assert any("consecutive" in e for e in errs)


def test_name_too_long_rejected():
    long_name = "a" * (MAX_NAME_LENGTH + 1)
    errs = validate({"name": long_name, "description": "x"})
    assert any("characters" in e for e in errs)


def test_name_empty_rejected():
    errs = validate({"name": "", "description": "x"})
    assert any("name" in e and ("required" in e or "missing" in e) for e in errs)


def test_name_at_exact_max_accepted():
    name = "a" * MAX_NAME_LENGTH
    assert validate({"name": name, "description": "x"}) == []


# ---------------------------------------------------------------------------
# Description rules
# ---------------------------------------------------------------------------


def test_description_required():
    errs = validate({"name": "valid-name"})
    assert any("description" in e and "required" in e for e in errs)


def test_description_must_be_string():
    errs = validate({"name": "valid-name", "description": 123})
    assert any("description" in e and "string" in e for e in errs)


def test_description_empty_rejected():
    errs = validate({"name": "valid-name", "description": "   "})
    assert any("description" in e for e in errs)


def test_description_too_long_rejected():
    desc = "x" * (MAX_DESCRIPTION_LENGTH + 1)
    errs = validate({"name": "valid-name", "description": desc})
    assert any("description" in e and "characters" in e for e in errs)


def test_description_at_exact_max_accepted():
    desc = "x" * MAX_DESCRIPTION_LENGTH
    assert validate({"name": "valid-name", "description": desc}) == []


# ---------------------------------------------------------------------------
# Optional agentskills.io fields
# ---------------------------------------------------------------------------


def test_license_must_be_string():
    errs = validate({"name": "n", "description": "d", "license": 123})
    assert any("license" in e for e in errs)


def test_compatibility_too_long_rejected():
    fm = {
        "name": "n",
        "description": "d",
        "compatibility": "x" * (MAX_COMPATIBILITY_LENGTH + 1),
    }
    errs = validate(fm)
    assert any("compatibility" in e for e in errs)


def test_compatibility_empty_rejected():
    fm = {"name": "n", "description": "d", "compatibility": ""}
    errs = validate(fm)
    assert any("compatibility" in e for e in errs)


def test_metadata_must_be_dict():
    errs = validate({"name": "n", "description": "d", "metadata": "not a dict"})
    assert any("metadata" in e and "mapping" in e for e in errs)


def test_metadata_arbitrary_keys_allowed():
    fm = {
        "name": "n",
        "description": "d",
        "metadata": {"author": "me", "anything": "goes"},
    }
    assert validate(fm) == []


def test_allowed_tools_must_be_string():
    errs = validate({"name": "n", "description": "d", "allowed-tools": ["Read"]})
    assert any("allowed-tools" in e for e in errs)


# ---------------------------------------------------------------------------
# Hermes extensions
# ---------------------------------------------------------------------------


def test_platforms_must_be_list():
    errs = validate({"name": "n", "description": "d", "platforms": "macos"})
    assert any("platforms" in e for e in errs)


def test_platforms_invalid_value_rejected():
    errs = validate({"name": "n", "description": "d", "platforms": ["plan9"]})
    assert any("platforms" in e and "plan9" in e for e in errs)


def test_platforms_valid():
    fm = {"name": "n", "description": "d", "platforms": ["macos", "linux", "windows"]}
    assert validate(fm) == []


def test_prerequisites_must_be_dict():
    errs = validate({"name": "n", "description": "d", "prerequisites": ["foo"]})
    assert any("prerequisites" in e for e in errs)


def test_prerequisites_env_vars_must_be_list():
    fm = {"name": "n", "description": "d", "prerequisites": {"env_vars": "API_KEY"}}
    errs = validate(fm)
    assert any("env_vars" in e for e in errs)


def test_metadata_hermes_tags_must_be_listish():
    fm = {
        "name": "n",
        "description": "d",
        "metadata": {"hermes": {"tags": 42}},
    }
    errs = validate(fm)
    assert any("tags" in e for e in errs)


def test_version_must_be_string():
    errs = validate({"name": "n", "description": "d", "version": 1.0})
    assert any("version" in e for e in errs)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_none_frontmatter_yields_error():
    errs = validate(None)
    assert len(errs) == 1
    assert "frontmatter" in errs[0]


def test_non_dict_frontmatter_yields_error():
    errs = validate("not a dict")
    assert len(errs) == 1
    assert "mapping" in errs[0]


def test_multiple_errors_returned():
    errs = validate({"name": "BAD-NAME!"})  # bad name + missing description
    assert len(errs) >= 2


def test_digits_in_name_allowed():
    assert validate({"name": "pdf-2", "description": "d"}) == []
