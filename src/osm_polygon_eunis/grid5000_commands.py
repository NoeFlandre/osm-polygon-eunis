"""Argument-array command helpers shared by the Grid'5000 submission modules."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from typing import Final

Command = tuple[str, ...]
CommandRunner = Callable[[Command], str]

_TOKEN_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def validate_host(value: str, field_name: str) -> None:
    """Raise ValueError unless value is one safe host, site, cluster or queue token."""

    if not _TOKEN_PATTERN.fullmatch(value):
        suffix = " host name" if field_name == "frontend" else " value"
        raise ValueError(f"{field_name} must be a non-empty{suffix}")


def build_ssh_command(frontend: str, command: Command) -> Command:
    """Serialize a remote command so SSH's remote shell preserves its arguments."""

    if not frontend or any(character.isspace() for character in frontend):
        raise ValueError("frontend must be a non-empty host name")
    if not command:
        raise ValueError("remote command must not be empty")
    return ("ssh", frontend, shlex.join(command))
