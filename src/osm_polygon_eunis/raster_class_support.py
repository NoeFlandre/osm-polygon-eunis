"""Highest-source-score class areas for independent EUNIS hierarchy decisions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Protocol, TypeGuard

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry.base import BaseGeometry

from .eunis_asset_manifest import EunisRasterAssetManifest
from .geometry import is_usable
from .grid_overlap import (
    TILE_SIZE,
    WeightedCells,
    band_row_ranges,
    tile_window,
    weighted_cells,
)
from .raster_reference_helpers import RasterLayerLike
from .raster_reference_helpers import shares_square_grid as _shares_square_grid

_GDAL_CACHE_MAX_MB = 256
_SUPPORT_CLASS_KEY_FIELDS = 2
_EEA_SCORE_MIN = 0.0
_EEA_SCORE_MAX = 1.0
_EEA_SCORE_SCALE = 1.0
_EEA_SCORE_OFFSET = 0.0


@dataclass(frozen=True, slots=True)
class ScoreDiagnostics:
    """Compact numeric diagnostics for one polygon's derived classification."""

    unknown_area_m2: float
    zero_max_score_area_m2: float
    ambiguous_area_m2_by_level: Mapping[str, float]
    maximum_source_score_min: float | None
    maximum_source_score_max: float | None
    maximum_source_score_area_weighted_mean: float | None

    def as_dict(self) -> dict[str, object]:
        """Return stable JSON-compatible diagnostics without changing source scores."""

        return {
            "method": "highest-source-score-derived-classification",
            "score_range": [_EEA_SCORE_MIN, _EEA_SCORE_MAX],
            "unknown_area_m2": self.unknown_area_m2,
            "zero_max_score_area_m2": self.zero_max_score_area_m2,
            "ambiguous_area_m2_by_level": dict(sorted(self.ambiguous_area_m2_by_level.items())),
            "maximum_source_score_min": self.maximum_source_score_min,
            "maximum_source_score_max": self.maximum_source_score_max,
            "maximum_source_score_area_weighted_mean": (
                self.maximum_source_score_area_weighted_mean
            ),
            "score_mean_scope": "diagnostic only; scores are not calibrated across habitats",
        }


@dataclass(frozen=True, slots=True)
class ClassificationAreas:
    """Categorical winning areas and the score/uncertainty diagnostics behind them."""

    areas_m2: Mapping[tuple[str, str], float]
    diagnostics: ScoreDiagnostics


@dataclass(slots=True)
class _ClassificationStats:
    areas_m2: dict[tuple[str, str], float]
    unknown_area_m2: float = 0.0
    zero_max_score_area_m2: float = 0.0
    ambiguous_area_m2_by_level: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(("L1", "L2", "L3"), 0.0)
    )
    maximum_score_area_m2: float = 0.0
    maximum_score_weighted_sum: float = 0.0
    maximum_score_min: float | None = None
    maximum_score_max: float | None = None
    covered_area_m2: float = 0.0


@dataclass(frozen=True, slots=True)
class _ScoreTile:
    known: np.ndarray
    levels: Mapping[str, _LevelScores]


@dataclass(slots=True)
class _LevelScores:
    best: np.ndarray
    winner: np.ndarray
    tied: np.ndarray

    def update(
        self,
        values: np.ndarray,
        source_mask: np.ndarray,
        finite: np.ndarray,
        class_index: int,
    ) -> None:
        height, width = values.shape
        best = self.best[:height, :width]
        winner = self.winner[:height, :width]
        tied = self.tied[:height, :width]
        valid_score = ~source_mask & finite
        higher = valid_score & (values > best)
        best[higher] = values[higher]
        winner[higher] = class_index
        tied[higher] = False
        equal_positive = valid_score & (values == best) & (values > 0.0)
        tied |= equal_positive & (winner != class_index)


class _Reference(Protocol):
    @property
    def _stack(self) -> ExitStack | None: ...

    @property
    def _datasets(self) -> tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...]: ...

    def _open_datasets(
        self,
        stack: ExitStack,
    ) -> Iterable[tuple[RasterLayerLike, rasterio.DatasetReader]]: ...

    def _covering_window(
        self,
        dataset: rasterio.DatasetReader,
        polygon: BaseGeometry,
    ) -> Window | None: ...

    def validate_classification_inventory(self) -> EunisRasterAssetManifest: ...


def support_areas(
    reference: _Reference,
    polygon: BaseGeometry,
    classes: Mapping[tuple[str, str], Iterable[str]],
) -> dict[tuple[str, str], float]:
    """Measure highest-source-score categorical area for each hierarchy class.

    This compatibility-shaped helper returns only the class areas. Use
    :func:`classification_areas` when the per-polygon uncertainty and score
    diagnostics are also needed.
    """

    return dict(classification_areas(reference, polygon, classes).areas_m2)


def classification_areas(
    reference: _Reference,
    polygon: BaseGeometry,
    classes: Mapping[tuple[str, str], Iterable[str]],
) -> ClassificationAreas:
    """Classify known cells by maximal source score and measure their areas.

    EEA metadata documents suitability values in the range 0 to 1. Raster scale
    and offset must be identity because this implementation does not invent a
    decoding. Every competitor must have a valid score for a cell to be
    classified; raster masks, NoData, and uncovered polygon area stay unknown
    in the unchanged full-polygon denominator. A zero maximum has no winner.
    Ties are resolved independently at each hierarchy level only when all tied
    direct classes share that level's explicit ancestor.
    """

    if not is_usable(polygon):
        raise ValueError("support query requires a usable projected polygon")
    polygon_area = float(polygon.area)
    if not math.isfinite(polygon_area) or polygon_area <= 0.0:
        raise ValueError("support query requires a positive full polygon area")
    normalized = _normalize_class_groups(classes)
    reference.validate_classification_inventory()
    if reference._stack is not None:
        return _classification_areas_open(
            reference, polygon, reference._datasets, normalized, polygon_area
        )
    with ExitStack() as stack:
        stack.enter_context(rasterio.Env(GDAL_CACHEMAX=_GDAL_CACHE_MAX_MB))
        datasets = tuple(reference._open_datasets(stack))
        return _classification_areas_open(reference, polygon, datasets, normalized, polygon_area)


def _classification_areas_open(
    reference: _Reference,
    polygon: BaseGeometry,
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
    classes: tuple[tuple[tuple[str, str], tuple[str, ...]], ...],
    polygon_area: float,
) -> ClassificationAreas:
    first = _support_grid_dataset(datasets)
    code_to_class_indexes, class_codes = _class_membership(classes, datasets)
    _validate_score_decoding(datasets)
    window = reference._covering_window(first, polygon)
    stats = _ClassificationStats(
        areas_m2={(level, code): 0.0 for level, codes in class_codes.items() for code in codes}
    )
    if window is None:
        return _classification_result(stats, polygon_area)
    rows = (int(window.row_off), int(window.row_off + window.height))
    cols = (int(window.col_off), int(window.col_off + window.width))
    for band in band_row_ranges(rows[0], rows[1], cols[0], cols[1]):
        cells = weighted_cells(polygon, first.transform, band, cols)
        stats.covered_area_m2 += float(cells.areas.sum())
        _accumulate_classification_cells(stats, code_to_class_indexes, class_codes, cells, datasets)
    return _classification_result(stats, polygon_area)


def _support_grid_dataset(
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
) -> rasterio.DatasetReader:
    if not datasets or not _shares_square_grid(datasets):
        raise ValueError("score reference layers must share one square-cell grid")
    return datasets[0][1]


def _validate_support_areas(areas: Mapping[tuple[str, str], float]) -> None:
    if any(not math.isfinite(area) or area < 0.0 for area in areas.values()):
        raise ArithmeticError("support query produced a non-finite or negative area")


def _accumulate_classification_cells(
    stats: _ClassificationStats,
    code_to_class_indexes: Mapping[str, Mapping[str, int]],
    class_codes: Mapping[str, tuple[str, ...]],
    cells: WeightedCells,
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
) -> None:
    if not len(cells.rows):
        return
    tile_rows = cells.rows // TILE_SIZE
    tile_cols = cells.cols // TILE_SIZE
    tile_ids = tile_rows * 1_000_003 + tile_cols
    order = np.argsort(tile_ids, kind="stable")
    cuts = np.flatnonzero(np.diff(tile_ids[order])) + 1
    indexes_by_code = {layer.code: index for index, (layer, _dataset) in enumerate(datasets)}
    for group in np.split(order, cuts):
        tile_row = int(tile_rows[group[0]])
        tile_column = int(tile_cols[group[0]])
        tile = _read_score_tile(
            datasets, indexes_by_code, code_to_class_indexes, tile_row, tile_column
        )
        _accumulate_tile_areas(stats, tile, class_codes, cells, group)


def _read_score_tile(
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
    indexes_by_code: Mapping[str, int],
    code_to_class_indexes: Mapping[str, Mapping[str, int]],
    tile_row: int,
    tile_column: int,
) -> _ScoreTile:
    tile_known = np.ones((TILE_SIZE, TILE_SIZE), dtype=bool)
    levels = {
        level: _LevelScores(
            np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float64),
            np.full((TILE_SIZE, TILE_SIZE), -1, dtype=np.int32),
            np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool),
        )
        for level in ("L1", "L2", "L3")
    }
    for code, index in sorted(indexes_by_code.items()):
        dataset = datasets[index][1]
        data = dataset.read(1, window=tile_window(dataset, tile_row, tile_column), masked=True)
        values = np.asarray(np.ma.getdata(data), dtype=np.float64)
        source_mask = np.ma.getmaskarray(data)
        finite = np.isfinite(values)
        if np.any(
            (~source_mask) & finite & ((values < _EEA_SCORE_MIN) | (values > _EEA_SCORE_MAX))
        ):
            raise ValueError(f"EEA score raster {code} has values outside the documented 0-1 range")
        height, width = values.shape
        tile_known[:height, :width] &= ~source_mask & finite
        for level in ("L1", "L2", "L3"):
            levels[level].update(
                values,
                source_mask,
                finite,
                code_to_class_indexes[level][code],
            )
    return _ScoreTile(tile_known, levels)


def _accumulate_tile_areas(
    stats: _ClassificationStats,
    tile: _ScoreTile,
    class_codes: Mapping[str, tuple[str, ...]],
    cells: WeightedCells,
    group: np.ndarray,
) -> None:
    local_rows = cells.rows[group] % TILE_SIZE
    local_cols = cells.cols[group] % TILE_SIZE
    weights = cells.areas[group]
    known = tile.known[local_rows, local_cols]
    stats.unknown_area_m2 += float((~known).astype(np.float64) @ weights)
    max_score = tile.levels["L1"].best[local_rows, local_cols]
    stats.zero_max_score_area_m2 += float((known & (max_score <= 0.0)).astype(np.float64) @ weights)
    positive_known = known & (max_score > 0.0)
    _accumulate_score_summary(stats, max_score, positive_known, weights)
    for level in ("L1", "L2", "L3"):
        best = tile.levels[level].best[local_rows, local_cols]
        winner = tile.levels[level].winner[local_rows, local_cols]
        level_tied = tile.levels[level].tied[local_rows, local_cols]
        ambiguous = level_tied & known & (best > 0.0)
        stats.ambiguous_area_m2_by_level[level] += float(ambiguous.astype(np.float64) @ weights)
        accepted = known & (winner >= 0) & ~level_tied & (best > 0.0)
        for class_index, code in enumerate(class_codes[level]):
            matches = accepted & (winner == class_index)
            stats.areas_m2[(level, code)] += float(matches.astype(np.float64) @ weights)


def _accumulate_score_summary(
    stats: _ClassificationStats,
    scores: np.ndarray,
    selected: np.ndarray,
    weights: np.ndarray,
) -> None:
    if not selected.any():
        return
    selected_scores = scores[selected]
    selected_weights = weights[selected]
    stats.maximum_score_area_m2 += float(selected_weights.sum())
    stats.maximum_score_weighted_sum += float(selected_scores @ selected_weights)
    local_min, local_max = float(selected_scores.min()), float(selected_scores.max())
    stats.maximum_score_min = (
        local_min if stats.maximum_score_min is None else min(stats.maximum_score_min, local_min)
    )
    stats.maximum_score_max = (
        local_max if stats.maximum_score_max is None else max(stats.maximum_score_max, local_max)
    )


def _class_membership(
    classes: tuple[tuple[tuple[str, str], tuple[str, ...]], ...],
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
) -> tuple[dict[str, dict[str, int]], dict[str, tuple[str, ...]]]:
    by_level = _class_groups_by_level(classes)
    dataset_codes = {layer.code for layer, _dataset in datasets}
    _validate_competitor_members(by_level, dataset_codes)
    return _level_class_indexes(by_level, dataset_codes)


def _class_groups_by_level(
    classes: tuple[tuple[tuple[str, str], tuple[str, ...]], ...],
) -> dict[str, dict[str, set[str]]]:
    by_level: dict[str, dict[str, set[str]]] = {level: {} for level in ("L1", "L2", "L3")}
    for (level, class_code), member_codes in classes:
        if level not in by_level:
            raise ValueError(f"unsupported EUNIS score hierarchy level {level!r}")
        by_level[level][class_code] = set(member_codes)
    return by_level


def _validate_competitor_members(
    by_level: Mapping[str, Mapping[str, set[str]]],
    dataset_codes: set[str],
) -> None:
    _validate_competitor_code_coverage(by_level, dataset_codes)
    _require_explicit_hierarchy(by_level)


def _validate_competitor_code_coverage(
    by_level: Mapping[str, Mapping[str, set[str]]],
    dataset_codes: set[str],
) -> None:
    all_members = set().union(*(codes for groups in by_level.values() for codes in groups.values()))
    missing = sorted(all_members - dataset_codes)
    if missing:
        raise ValueError(f"support class references missing raster codes: {missing}")
    unmapped = sorted(dataset_codes - all_members)
    if unmapped:
        raise ValueError(f"classification omits competing raster codes: {unmapped}")


def _require_explicit_hierarchy(by_level: Mapping[str, Mapping[str, set[str]]]) -> None:
    if any(not by_level[level] for level in ("L1", "L2", "L3")):
        raise ValueError("classification needs explicit L1, L2, and L3 ancestry")


def _level_class_indexes(
    by_level: Mapping[str, Mapping[str, set[str]]],
    dataset_codes: set[str],
) -> tuple[dict[str, dict[str, int]], dict[str, tuple[str, ...]]]:
    code_to_class_indexes: dict[str, dict[str, int]] = {}
    class_codes: dict[str, tuple[str, ...]] = {}
    for level, groups in by_level.items():
        class_codes[level], code_to_class_indexes[level] = _indexes_by_level(
            level, groups, dataset_codes
        )
    _validate_direct_l3_codes(dataset_codes, code_to_class_indexes["L3"], class_codes["L3"])
    return code_to_class_indexes, class_codes


def _indexes_by_level(
    level: str,
    groups: Mapping[str, set[str]],
    dataset_codes: set[str],
) -> tuple[tuple[str, ...], dict[str, int]]:
    ordered_codes = tuple(sorted(groups))
    inverse: dict[str, int] = {}
    for class_index, class_code in enumerate(ordered_codes):
        for direct_code in groups[class_code]:
            _record_class_index(inverse, direct_code, class_index, level)
    _require_index_coverage(inverse, dataset_codes, level)
    return ordered_codes, inverse


def _record_class_index(
    inverse: dict[str, int],
    direct_code: str,
    class_index: int,
    level: str,
) -> None:
    previous = inverse.get(direct_code)
    if previous is not None and previous != class_index:
        raise ValueError(f"direct raster code {direct_code} has conflicting {level} ancestors")
    inverse[direct_code] = class_index


def _require_index_coverage(
    inverse: Mapping[str, int],
    dataset_codes: set[str],
    level: str,
) -> None:
    if set(inverse) != dataset_codes:
        raise ValueError(f"classification does not map every competing raster code at {level}")


def _validate_direct_l3_codes(
    dataset_codes: set[str],
    direct_indexes: Mapping[str, int],
    l3_codes: tuple[str, ...],
) -> None:
    for direct_code in dataset_codes:
        if l3_codes[direct_indexes[direct_code]] != direct_code:
            raise ValueError(f"direct raster code {direct_code} must be its own L3 class")


def _validate_score_decoding(
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
) -> None:
    if len({layer.code for layer, _dataset in datasets}) != len(datasets):
        raise ValueError("classification raster codes must be unique")
    for layer, dataset in datasets:
        _validate_score_band(layer, dataset)


def _validate_score_band(
    layer: RasterLayerLike,
    dataset: rasterio.DatasetReader,
) -> None:
    _require_single_band(layer, dataset)
    _require_real_numeric_band(layer, dataset)
    _require_identity_score_decoding(layer, dataset)


def _require_single_band(layer: RasterLayerLike, dataset: rasterio.DatasetReader) -> None:
    if dataset.count != 1:
        raise ValueError(f"EEA score raster {layer.code} must have exactly one band")


def _require_real_numeric_band(layer: RasterLayerLike, dataset: rasterio.DatasetReader) -> None:
    dtype = np.dtype(dataset.dtypes[0])
    if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"EEA score raster {layer.code} must use a real numeric band")


def _require_identity_score_decoding(
    layer: RasterLayerLike, dataset: rasterio.DatasetReader
) -> None:
    identity_scale = tuple(dataset.scales) == (_EEA_SCORE_SCALE,)
    identity_offset = tuple(dataset.offsets) == (_EEA_SCORE_OFFSET,)
    if not identity_scale or not identity_offset:
        raise ValueError(
            f"EEA score raster {layer.code} has an undocumented scale or offset; "
            "only identity decoding is supported"
        )


def _classification_result(stats: _ClassificationStats, polygon_area: float) -> ClassificationAreas:
    _validate_support_areas(stats.areas_m2)
    tolerance = max(1e-7, polygon_area * 1e-10)
    unknown_area = stats.unknown_area_m2 + max(0.0, polygon_area - stats.covered_area_m2)
    if unknown_area < -tolerance or unknown_area > polygon_area + tolerance:
        raise ArithmeticError("unknown classification area falls outside the full polygon area")
    unknown_area = min(polygon_area, max(0.0, unknown_area))
    return ClassificationAreas(
        dict(stats.areas_m2),
        ScoreDiagnostics(
            unknown_area,
            stats.zero_max_score_area_m2,
            dict(stats.ambiguous_area_m2_by_level),
            stats.maximum_score_min,
            stats.maximum_score_max,
            (
                stats.maximum_score_weighted_sum / stats.maximum_score_area_m2
                if stats.maximum_score_area_m2
                else None
            ),
        ),
    )


def _normalize_class_groups(
    classes: Mapping[tuple[str, str], Iterable[str]],
) -> tuple[tuple[tuple[str, str], tuple[str, ...]], ...]:
    if not classes:
        raise ValueError("at least one support class must be requested")
    normalized = []
    for class_key, raw_codes in classes.items():
        if not _is_support_class_key(class_key):
            raise ValueError("support class keys must be non-empty (level, code) pairs")
        normalized.append((class_key, _normalize_class_codes(class_key, raw_codes)))
    return tuple(sorted(normalized))


def _is_support_class_key(value: object) -> TypeGuard[tuple[str, str]]:
    return (
        isinstance(value, tuple)
        and len(value) == _SUPPORT_CLASS_KEY_FIELDS
        and all(isinstance(part, str) and part.strip() for part in value)
    )


def _normalize_class_codes(
    class_key: tuple[str, str],
    raw_codes: Iterable[str],
) -> tuple[str, ...]:
    codes = tuple(sorted(set(raw_codes)))
    if not codes or any(not isinstance(code, str) or not code.strip() for code in codes):
        raise ValueError(f"support class {class_key} needs non-empty raster codes")
    return codes
