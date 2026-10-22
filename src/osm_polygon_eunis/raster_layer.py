"""Stable raster-layer identity and official EUNIS filename parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_LAYER_CODE = re.compile(r"^Prob_(?P<code>[A-Z][A-Z0-9.]+)_\d+m\.tif$")


@dataclass(frozen=True, slots=True)
class RasterLayer:
    """One EEA score raster and its stable EUNIS metadata."""

    code: str
    name: str
    path: Path
    source_version: str
    asset_filename: str | None = None


def parse_layer_code(filename: str) -> str:
    """Extract the EUNIS code from an official probability-map filename."""

    match = _LAYER_CODE.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"could not parse reference layer code from {filename!r}")
    return match.group("code")
