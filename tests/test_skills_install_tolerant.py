"""`skills install` installs under the skill's frontmatter `name`, regardless of
the source directory's name (gh #-dogfood), and `--show-config`'s no-TOML message
names the documented langstage-hermes.toml.
"""

from click.testing import CliRunner

from langstage_hermes.cli import cli
from langstage_hermes.config import HermesConfig

_SKILL = "---\nname: csv-profiler\ndescription: Profile a CSV.\nversion: 1.0.0\nplatforms: [linux, macos, windows]\n---\nbody\n"


def test_install_uses_frontmatter_name_not_dir_name(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("LANGSTAGE_HERMES_HOME", raising=False)
    monkeypatch.delenv("DEEPAGENT_HERMES_HOME", raising=False)

    # Source dir is named 'authored' but the skill's name is 'csv-profiler'.
    src = tmp_path / "authored"
    src.mkdir()
    (src / "SKILL.md").write_text(_SKILL, encoding="utf-8")

    r = CliRunner().invoke(cli, ["skills", "install", str(src)])
    assert r.exit_code == 0, r.output
    assert (home / "skills" / "csv-profiler" / "SKILL.md").is_file()
    # Not installed under the source dir name.
    assert not (home / "skills" / "authored").exists()


def test_show_config_no_toml_message_names_hermes_toml():
    text = HermesConfig.resolve(use_toml=False).describe()
    assert "langstage-hermes.toml" in text
