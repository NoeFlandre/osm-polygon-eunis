"""Raster layer classes and small grouping helpers for shared grids."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .domain import EunisTaxonomy
from .grid_overlap import TILE_SIZE, is_square_north_up, tile_window
from .raster_geometry import wgs84_envelope

_LAYER_CODE = re.compile(r"^Prob_(?P<code>[A-Z][A-Z0-9.]+)_\d+m\.tif$")


@dataclass(frozen=True, slots=True)
class RasterLayer:
    """One EEA probability raster and its stable EUNIS metadata."""

    code: str
    name: str
    path: Path
    source_version: str


@dataclass(frozen=True, slots=True)
class RasterClass:
    """One hierarchy node and its source layers, counted as a pixel union."""

    code: str
    name: str
    layer_indexes: tuple[int, ...]


def parse_layer_code(filename: str) -> str:
    """Extract the EUNIS code from an official probability-map filename."""
    match = _LAYER_CODE.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"could not parse reference layer code from {filename!r}")
    return match.group("code")


def shares_square_grid(
    datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
) -> bool:
    """Return whether all datasets share one square, north-up grid."""
    grids = {
        (tuple(dataset.transform)[:6], dataset.width, dataset.height) for _, dataset in datasets
    }
    return len(grids) == 1 and is_square_north_up(datasets[0][1].transform)


def classes_on_shared_grid(
    datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
    taxonomy: EunisTaxonomy | None,
) -> tuple[RasterClass, ...] | None:
    """Group same-code layers and their verified taxonomy ancestors."""
    if not shares_square_grid(datasets):
        return None
    grouped_layers = _layers_by_code(datasets)
    if grouped_layers is None:
        return None
    indexes_by_code, names_by_code = grouped_layers
    indexes_by_node = _indexes_by_taxonomy_node(indexes_by_code, names_by_code, taxonomy)
    if indexes_by_node is None:
        return None
    return _classes_from_nodes(indexes_by_node, names_by_code, taxonomy)


def _layers_by_code(
    datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
) -> tuple[dict[str, list[int]], dict[str, set[str]]] | None:
    indexes_by_code: dict[str, list[int]] = {}
    names_by_code: dict[str, set[str]] = {}
    for index, (layer, _dataset) in enumerate(datasets):
        if not layer.code or not layer.name:
            return None
        indexes_by_code.setdefault(layer.code, []).append(index)
        names_by_code.setdefault(layer.code, set()).add(layer.name)
    return indexes_by_code, names_by_code


def _indexes_by_taxonomy_node(
    indexes_by_code: dict[str, list[int]],
    names_by_code: dict[str, set[str]],
    taxonomy: EunisTaxonomy | None,
) -> dict[str, set[int]] | None:
    indexes_by_node: dict[str, set[int]] = {}
    for code, indexes in indexes_by_code.items():
        if taxonomy is None and _has_conflicting_names(names_by_code[code]):
            return None
        for node in _taxonomy_nodes(code, taxonomy):
            indexes_by_node.setdefault(node, set()).update(indexes)
    return indexes_by_node


def _has_conflicting_names(names: set[str]) -> bool:
    return len(names) != 1


def _taxonomy_nodes(code: str, taxonomy: EunisTaxonomy | None) -> tuple[str, ...]:
    if taxonomy is None:
        return (code,)
    return taxonomy.ancestors_including_self(code)


def _classes_from_nodes(
    indexes_by_node: dict[str, set[int]],
    names_by_code: dict[str, set[str]],
    taxonomy: EunisTaxonomy | None,
) -> tuple[RasterClass, ...] | None:
    classes: list[RasterClass] = []
    for code, indexes in indexes_by_node.items():
        name = _class_label(code, names_by_code, taxonomy)
        if name is None:
            return None
        classes.append(RasterClass(code, name, tuple(sorted(indexes))))
    return tuple(sorted(classes, key=lambda class_: class_.code))


def _class_label(
    code: str,
    names_by_code: dict[str, set[str]],
    taxonomy: EunisTaxonomy | None,
) -> str | None:
    if taxonomy is not None:
        return taxonomy.labels.get(code)
    names = names_by_code.get(code, set())
    if _has_conflicting_names(names):
        return None
    return next(iter(names))


def layer_mask_stack(
    datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
    row: int,
    column: int,
    read_positive_mask: Callable[[rasterio.DatasetReader, Window], np.ndarray | None],
) -> np.ndarray:
    """Read each raster layer's positive mask for one cache tile."""
    layer_stack = np.zeros((len(datasets), TILE_SIZE, TILE_SIZE), dtype=bool)
    for index, (_, dataset) in enumerate(datasets):
        valid = read_positive_mask(dataset, tile_window(dataset, row, column))
        if valid is not None:
            layer_stack[index, : valid.shape[0], : valid.shape[1]] = valid
    return layer_stack


def union_class_masks(
    layer_stack: np.ndarray,
    classes: tuple[RasterClass, ...],
) -> np.ndarray | None:
    """Union each class's source-layer masks and return no stack when empty."""
    stack = np.zeros((len(classes), TILE_SIZE, TILE_SIZE), dtype=bool)
    for index, class_ in enumerate(classes):
        stack[index] = np.any(layer_stack[list(class_.layer_indexes)], axis=0)
    return stack if stack.any() else None


def wgs84_dataset_extent(
    datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
) -> tuple[float, float, float, float] | None:
    """Return the WGS84 bounds covering every raster dataset."""
    iterator = iter(datasets)
    first = next(iterator, None)
    if first is None:
        return None
    bounds = first[1].bounds
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    for _, dataset in iterator:
        bounds = dataset.bounds
        left = min(left, bounds.left)
        bottom = min(bottom, bounds.bottom)
        right = max(right, bounds.right)
        top = max(top, bounds.top)
    return wgs84_envelope((left, bottom, right, top))
