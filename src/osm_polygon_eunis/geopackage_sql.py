"""Safe identifier quoting shared by GeoPackage SQL readers."""

from __future__ import annotations


def sql_identifier(value: str) -> str:
    """Quote a GeoPackage identifier for SQL, rejecting empty values and NUL bytes."""
    if not value or "\x00" in value:
        raise ValueError("invalid GeoPackage identifier")
    return '"' + value.replace('"', '""') + '"'
