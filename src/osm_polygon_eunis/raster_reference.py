"""Read exact habitat overlaps from EEA probability rasters."""

from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import shapely
from rasterio.windows import Window, WindowError, from_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .domain import EunisResult
from .geometry import is_usable
from .grid_overlap import (
    TILE_SIZE,
    WeightedCells,
    band_row_ranges,
    is_square_north_up,
    stack_bytes,
    tile_window,
    weighted_cells,
)
from .matching import choose_winner
from .raster_geometry import (
    is_epsg_3035,
    wgs84_envelope,
)

logger = logging.getLogger(__name__)

_LAYER_CODE = re.compile(r"^Prob_(?P<code>[A-Z][A-Z0-9.]+)_\d+m\.tif$")
_DEFAULT_RASTER_TILE_CACHE_BYTES = 1536 * 1024 * 1024
# GDAL's block cache per open raster reference, in MiB (``GDAL_CACHEMAX``).
# Sized independently of ``_DEFAULT_RASTER_TILE_CACHE_BYTES`` above.
_GDAL_CACHE_MAX_MB = 256


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


class RasterReference:
    """Read only the raster windows needed for a projected input polygon."""

    def __init__(
        self,
        layers: tuple[RasterLayer, ...],
        threshold: int = 0,
        *,
        tile_cache_bytes: int = _DEFAULT_RASTER_TILE_CACHE_BYTES,
    ) -> None:
        """Prepare raster layers for exact overlap queries.

        Args:
            layers: Probability rasters that share one source version.
            threshold: Minimum raster value counted as habitat presence.
            tile_cache_bytes: Maximum estimated memory retained for tile geometries.

        Raises:
            ValueError: If layer versions disagree, the threshold is negative, or the cache
                budget is not positive.
        """
        versions = {layer.source_version for layer in layers}
        if len(versions) > 1:
            raise ValueError("all reference layers must use one source version")
        if threshold < 0:
            raise ValueError("raster threshold must be non-negative")
        if tile_cache_bytes <= 0:
            raise ValueError("raster tile cache byte budget must be positive")
        self._layers = layers
        self._threshold = threshold
        self._tile_cache_byte_budget = tile_cache_bytes
        self._source_version = next(iter(versions), None)
        self.intersection_errors = 0
        self._stack: ExitStack | None = None
        self._datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...] = ()
        self._tile_cache_bytes = 0
        self._tile_cache_hits = 0
        self._tile_cache_misses = 0
        self._source_extent_wgs84: tuple[float, float, float, float] | None = None
        self._stack_cache: OrderedDict[tuple[int, int], np.ndarray | None] = OrderedDict()
        self._tile_cache_bytes = 0
        self._grid_order: tuple[int, ...] | None = None
        self._grid_checked = False

    @property
    def tile_cache_hits(self) -> int:
        """Return the number of tile geometry cache hits since construction."""

        return self._tile_cache_hits

    @property
    def tile_cache_misses(self) -> int:
        """Return the number of tile geometry cache misses since construction."""

        return self._tile_cache_misses

    @property
    def tile_cache_miss_rate(self) -> float:
        """Return the fraction of tile lookups that missed since construction."""

        lookups = self._tile_cache_hits + self._tile_cache_misses
        return self._tile_cache_misses / lookups if lookups else 0.0

    @property
    def tile_cache_bytes(self) -> int:
        """Return the conservative estimated bytes retained by the tile cache."""

        return self._tile_cache_bytes

    @property
    def tile_cache_byte_budget(self) -> int:
        """Return the configured estimated-byte budget for cached tile geometries."""

        return self._tile_cache_byte_budget

    @property
    def source_extent_wgs84(self) -> tuple[float, float, float, float] | None:
        """Conservative WGS84 box covering every open layer, or None if closed."""

        if not self._datasets:
            return None
        if self._source_extent_wgs84 is None:
            self._source_extent_wgs84 = _wgs84_dataset_extent(self._datasets)
        return self._source_extent_wgs84

    def __enter__(self) -> RasterReference:
        """Open raster datasets and return this reference for overlap queries."""
        if self._stack is not None:
            raise RuntimeError("raster reference is already open")
        stack = ExitStack()
        stack.enter_context(rasterio.Env(GDAL_CACHEMAX=_GDAL_CACHE_MAX_MB))
        datasets: list[tuple[RasterLayer, rasterio.DatasetReader]] = []
        for layer in self._layers:
            dataset = stack.enter_context(rasterio.open(layer.path))
            self._validate_crs(dataset, layer.path)
            datasets.append((layer, dataset))
        self._stack = stack
        self._datasets = tuple(datasets)
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the datasets opened by this reference."""
        self.close()

    def close(self) -> None:
        """Close raster datasets and clear this reference's tile cache."""
        if self._stack is not None:
            self._stack.close()
            self._stack = None
            self._datasets = ()
            self._stack_cache.clear()
            self._tile_cache_bytes = 0
            self._grid_order = None
            self._grid_checked = False

    def _count_intersection_error(self) -> None:
        self.intersection_errors += 1

    def overlap(self, polygon: BaseGeometry | None) -> EunisResult:
        """Return the label with the largest actual intersection area."""

        if not is_usable(polygon):
            return choose_winner(polygon, (), source_version=self._source_version)

        if self._stack is not None:
            return self._overlap_open(polygon, self._datasets)
        return self._overlap_closed(polygon)

    def _overlap_closed(self, polygon: BaseGeometry) -> EunisResult:
        with ExitStack() as stack:
            stack.enter_context(rasterio.Env(GDAL_CACHEMAX=_GDAL_CACHE_MAX_MB))
            datasets = self._open_datasets(stack)
            return self._overlap_open(polygon, tuple(datasets))

    def _open_datasets(
        self,
        stack: ExitStack,
    ) -> list[tuple[RasterLayer, rasterio.DatasetReader]]:
        datasets: list[tuple[RasterLayer, rasterio.DatasetReader]] = []
        for layer in self._layers:
            dataset = stack.enter_context(rasterio.open(layer.path))
            self._validate_crs(dataset, layer.path)
            datasets.append((layer, dataset))
        return datasets

    def _overlap_open(
        self,
        polygon: BaseGeometry,
        datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
    ) -> EunisResult:
        order = self._shared_grid_order(datasets)
        if order is None:
            raise ValueError(
                "raster reference layers must share one square-cell grid and unique codes"
            )
        try:
            return self._overlap_grid(polygon, datasets, order)
        except (ValueError, RuntimeError, shapely.errors.GEOSException) as error:
            logger.warning("skipping EUNIS polygon after grid overlap error: %s", error)
            self._count_intersection_error()
            return EunisResult(None, None, None, None)

    def _shared_grid_order(
        self,
        datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
    ) -> tuple[int, ...] | None:
        """Return layer indices sorted by code when every layer shares one grid."""

        if not self._grid_checked:
            self._grid_order = _code_order_on_shared_grid(datasets) if datasets else None
            self._grid_checked = True
        return self._grid_order

    def _overlap_grid(
        self,
        polygon: BaseGeometry,
        datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
        order: tuple[int, ...],
    ) -> EunisResult:
        first = datasets[0][1]
        window = self._covering_window(first, polygon)
        polygon_area = float(polygon.area)
        if window is None or not polygon_area > 0.0:
            return EunisResult(None, None, None, None)
        rows = (int(window.row_off), int(window.row_off + window.height))
        cols = (int(window.col_off), int(window.col_off + window.width))
        areas = np.zeros(len(datasets), dtype=np.float64)
        for band in band_row_ranges(rows[0], rows[1], cols[0], cols[1]):
            cells = weighted_cells(polygon, first.transform, band, cols)
            self._accumulate_cells(areas, cells, datasets)
        ordered = areas[list(order)]
        best = int(np.argmax(ordered))
        if not ordered[best] > 0.0:
            return EunisResult(None, None, None, None)
        layer = datasets[order[best]][0]
        percentage = max(0.0, min(100.0, 100.0 * float(ordered[best]) / polygon_area))
        return EunisResult(layer.code, layer.name, percentage, self._source_version)

    def _accumulate_cells(
        self,
        areas: np.ndarray,
        cells: WeightedCells,
        datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
    ) -> None:
        if not len(cells.rows):
            return
        tile_rows = cells.rows // TILE_SIZE
        tile_cols = cells.cols // TILE_SIZE
        tile_ids = tile_rows * 1_000_003 + tile_cols
        order = np.argsort(tile_ids, kind="stable")
        cuts = np.flatnonzero(np.diff(tile_ids[order])) + 1
        for group in np.split(order, cuts):
            stack = self._tile_stack(datasets, int(tile_rows[group[0]]), int(tile_cols[group[0]]))
            if stack is None:
                continue
            local_rows = cells.rows[group] % TILE_SIZE
            local_cols = cells.cols[group] % TILE_SIZE
            areas += stack[:, local_rows, local_cols].astype(np.float64) @ cells.areas[group]

    def _tile_stack(
        self,
        datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
        row: int,
        column: int,
    ) -> np.ndarray | None:
        key = (row, column)
        if key in self._stack_cache:
            self._stack_cache.move_to_end(key)
            self._tile_cache_hits += 1
            return self._stack_cache[key]
        self._tile_cache_misses += 1
        stack = np.zeros((len(datasets), TILE_SIZE, TILE_SIZE), dtype=bool)
        for index, (_, dataset) in enumerate(datasets):
            valid = self._positive_mask(dataset, tile_window(dataset, row, column))
            if valid is not None:
                stack[index, : valid.shape[0], : valid.shape[1]] = valid
        result = stack if stack.any() else None
        self._remember_stack(key, result)
        return result

    def _remember_stack(self, key: tuple[int, int], stack: np.ndarray | None) -> None:
        size = stack_bytes(stack)
        if size > self._tile_cache_byte_budget:
            return
        while self._tile_cache_bytes + size > self._tile_cache_byte_budget:
            evicted = self._stack_cache.pop(next(iter(self._stack_cache)))
            self._tile_cache_bytes -= stack_bytes(evicted)
        self._stack_cache[key] = stack
        self._tile_cache_bytes += size

    def _covering_window(
        self,
        dataset: rasterio.DatasetReader,
        polygon: BaseGeometry,
    ) -> Window | None:
        if not polygon.intersects(box(*dataset.bounds)):
            return None
        return self._window(dataset, polygon)

    @staticmethod
    def _validate_crs(dataset: rasterio.DatasetReader, path: Path) -> None:
        try:
            crs = dataset.crs
            valid = is_epsg_3035(crs)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"reference raster {path} is not EPSG:3035")

    @staticmethod
    def _window(
        dataset: rasterio.DatasetReader,
        polygon: BaseGeometry,
    ) -> Window | None:
        try:
            window = from_bounds(*polygon.bounds, transform=dataset.transform)
        except WindowError as error:
            logger.warning("raster window fallback after WindowError: %s", error)
            return None
        # Round outwards. Rounding to nearest can shrink the window inside the
        # polygon's bounding box and silently drop the partially covered cells
        # on its edges, which undercounts the overlap area.
        col_off = max(0, math.floor(window.col_off))
        row_off = max(0, math.floor(window.row_off))
        col_end = min(dataset.width, math.ceil(window.col_off + window.width))
        row_end = min(dataset.height, math.ceil(window.row_off + window.height))
        if col_end <= col_off or row_end <= row_off:
            return None
        return Window.from_slices((row_off, row_end), (col_off, col_end))

    def _positive_mask(
        self,
        dataset: rasterio.DatasetReader,
        window: Window,
    ) -> np.ndarray | None:
        data = dataset.read(1, window=window, masked=True)
        values = np.asarray(data)
        valid = (~np.ma.getmaskarray(data)) & (values > self._threshold)
        return valid if valid.any() else None


def _shares_square_grid(
    datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
) -> bool:
    grids = {(tuple(d.transform)[:6], d.width, d.height) for _, d in datasets}
    return len(grids) == 1 and is_square_north_up(datasets[0][1].transform)


def _code_order_on_shared_grid(
    datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
) -> tuple[int, ...] | None:
    codes = [layer.code for layer, _ in datasets]
    if len(set(codes)) != len(codes) or not _shares_square_grid(datasets):
        return None
    return tuple(sorted(range(len(datasets)), key=codes.__getitem__))


def _wgs84_dataset_extent(
    datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
) -> tuple[float, float, float, float] | None:
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
