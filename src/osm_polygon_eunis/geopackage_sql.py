"""Safe identifier quoting shared by GeoPackage SQL readers."""

from __future__ import annotations


def _sql_identifier(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("invalid GeoPackage identifier")
    return '"' + value.replace('"', '""') + '"'
