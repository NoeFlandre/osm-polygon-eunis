from __future__ import annotations

import argparse
import re
from pathlib import Path

from osm_polygon_eunis import cli

ROOT = Path(__file__).resolve().parents[2]
OPERATIONS = (ROOT / "docs" / "operations.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
ENV_READ = re.compile(r"""environ(?:\.get\(|\[)\s*["']([A-Z][A-Z0-9_]+)["']""")
README_ENV_VARS = (
    "OSM_EUNIS_WORKDIR",
    "EUNIS_SIDECAR_DIR",
    "EUNIS_SOURCE_DIR",
    "EUNIS_REFERENCE_DIR",
)


def _subcommand_options() -> dict[str, set[str]]:
    parser = cli._parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    global_flags = {o for a in parser._actions for o in a.option_strings}
    return {
        name: {
            option
            for sub_action in sub._actions
            for option in sub_action.option_strings
            if option.startswith("--") and option not in global_flags
        }
        for name, sub in action.choices.items()
    }


def _command_reference_rows() -> set[tuple[str, str]]:
    rows: set[tuple[str, str]] = set()
    for line in OPERATIONS.split("## Command reference", 1)[1].splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[1].startswith("`--"):
            continue
        option = cells[1].strip("`").split()[0]
        for command in re.findall(r"`([a-z-]+)`", cells[0]):
            rows.add((command, option))
    return rows


def test_command_reference_documents_exactly_the_cli_options() -> None:
    # Global flags (--quiet, --verbose, --debug, --version) are documented separately.
    actual = {(c, o) for c, options in _subcommand_options().items() for o in options}
    assert actual == _command_reference_rows()


def test_every_environment_variable_read_by_the_package_is_in_operations_guide() -> None:
    names = {
        name
        for path in (ROOT / "src").rglob("*.py")
        for name in ENV_READ.findall(path.read_text(encoding="utf-8"))
    }
    assert "OSM_EUNIS_WORKDIR" in names
    assert {name for name in names if name not in OPERATIONS} == set()


def test_readme_mentions_every_path_environment_variable() -> None:
    assert [name for name in README_ENV_VARS if name not in README] == []
