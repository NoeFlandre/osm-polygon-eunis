"""Official EEA raster reference access with exact cell intersections."""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window, WindowError, from_bounds
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .domain import EunisResult, OverlapCandidate
from .matching import choose_winner

_LAYER_CODE = re.compile(r"^Prob_(?P<code>[A-Z][A-Z0-9.]+)_\d+m\.tif$")


@dataclass(frozen=True, slots=True)
class RasterLayer:
    """One EEA probability raster and its stable EUNIS metadata."""

    code: str
    name: str
    path: Path
    source_version: str


def parse_layer_code(filename: str) -> str:
    """Extract the EUNIS code from an official probability-map filename."""

    match = _LAYER_CODE.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"could not parse reference layer code from {filename!r}")
    return match.group("code")


def load_label_table(path: Path) -> dict[str, str]:
    """Load a compact code-to-name JSON table and reject incomplete values."""

    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("EUNIS label table must be a JSON object")
    labels: dict[str, str] = {}
    for code, name in value.items():
        if not isinstance(code, str) or not isinstance(name, str) or not name.strip():
            raise ValueError("EUNIS label table contains a non-string or empty name")
        labels[code] = name
    return labels


def raster_layer_from_file(
    path: Path,
    labels: dict[str, str],
    source_version: str,
) -> RasterLayer:
    """Create a layer only when its parsed code has a verified name."""

    code = parse_layer_code(path.name)
    try:
        name = labels[code]
    except KeyError as error:
        raise ValueError(f"missing EUNIS name for reference code {code}") from error
    return RasterLayer(code, name, path, source_version)


class RasterReference:
    """Read only the raster windows needed for a projected input polygon."""

    def __init__(self, layers: tuple[RasterLayer, ...], threshold: int = 0) -> None:
        versions = {layer.source_version for layer in layers}
        if len(versions) > 1:
            raise ValueError("all reference layers must use one source version")
        if threshold < 0:
            raise ValueError("raster threshold must be non-negative")
        self._layers = layers
        self._threshold = threshold
        self._source_version = next(iter(versions), None)

    def overlap(self, polygon: BaseGeometry | None) -> EunisResult:
        """Return the label with the largest actual intersection area."""

        if polygon is None or polygon.is_empty or not polygon.is_valid:
            return choose_winner(polygon, (), source_version=self._source_version)

        candidates: list[OverlapCandidate] = []
        with ExitStack() as stack:
            for layer in self._layers:
                dataset = stack.enter_context(rasterio.open(layer.path))
                self._validate_crs(dataset, layer.path)
                cell_geometry = self._positive_cell_geometry(dataset, polygon)
                if cell_geometry is not None:
                    candidates.append(OverlapCandidate(layer.code, layer.name, cell_geometry))
        return choose_winner(polygon, candidates, source_version=self._source_version)

    @staticmethod
    def _validate_crs(dataset: rasterio.DatasetReader, path: Path) -> None:
        if dataset.crs is None or dataset.crs.to_epsg() != 3035:
            raise ValueError(f"reference raster {path} is not EPSG:3035")

    def _positive_cell_geometry(
        self,
        dataset: rasterio.DatasetReader,
        polygon: BaseGeometry,
    ) -> BaseGeometry | None:
        raster_bounds = box(*dataset.bounds)
        if not polygon.intersects(raster_bounds):
            return None
        try:
            window = from_bounds(*polygon.bounds, transform=dataset.transform)
            full_window = Window.from_slices(
                (0, dataset.height),
                (0, dataset.width),
            )
            window = window.intersection(full_window).round_offsets().round_lengths()
        except WindowError:
            return None
        if window.width <= 0 or window.height <= 0:
            return None

        data = dataset.read(1, window=window, masked=True)
        values = np.asarray(data)
        valid = (~np.ma.getmaskarray(data)) & (values > self._threshold)
        if not valid.any():
            return None
        cells = [
            shape(geometry)
            for geometry, _ in shapes(
                valid.astype("uint8"),
                mask=valid,
                transform=dataset.window_transform(window),
            )
        ]
        merged = unary_union(cells)
        return None if merged.is_empty else merged


def resolve_eea_layers(config_path: Path, workspace: Path) -> tuple[RasterLayer, ...]:
    """Load a run-resolved local layer manifest produced from the EEA catalog."""

    config: Any = json.loads(config_path.read_text(encoding="utf-8"))
    layer_specs = config.get("layers") if isinstance(config, dict) else None
    if not isinstance(layer_specs, list) or not layer_specs:
        raise ValueError(
            "reference config has no resolved layers; run the EEA catalog resolver first",
        )
    layers: list[RasterLayer] = []
    for spec in layer_specs:
        if not isinstance(spec, dict):
            raise ValueError("reference layer specification must be an object")
        relative_path = spec.get("path")
        code = spec.get("code")
        name = spec.get("name")
        if not all(isinstance(value, str) and value for value in (relative_path, code, name)):
            raise ValueError("reference layer specification is missing path, code, or name")
        path = workspace / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        layers.append(
            RasterLayer(
                code=code,
                name=name,
                path=path,
                source_version=str(config["source_version"]),
            )
        )
    return tuple(layers)
