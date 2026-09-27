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
from rasterio.windows import Window, WindowError, from_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .domain import EunisResult, OverlapCandidate
from .geometry import is_usable
from .matching import choose_winner
from .raster_geometry import EPSG_LAEA_EUROPE as _EPSG_LAEA_EUROPE
from .raster_geometry import (
    _is_epsg_3035,
    _mask_geometry,
    _raster_tile_indices,
    _raster_tile_window,
    _tile_geometry_cache_size,
)
from .raster_geometry import (
    wgs84_envelope as _raster_wgs84_envelope,
)
from .reference_geometry import (
    _has_disjoint_components,
    _merge_tile_cells,
)

logger = logging.getLogger(__name__)

_LAYER_CODE = re.compile(r"^Prob_(?P<code>[A-Z][A-Z0-9.]+)_\d+m\.tif$")
_DEFAULT_RASTER_TILE_CACHE_BYTES = 512 * 1024 * 1024
EPSG_LAEA_EUROPE = _EPSG_LAEA_EUROPE


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
        self._tile_cache: OrderedDict[tuple[str, int, int], BaseGeometry | None] = OrderedDict()
        self._tile_cache_sizes: dict[tuple[str, int, int], int] = {}
        self._tile_cache_bytes = 0
        self._tile_cache_hits = 0
        self._tile_cache_misses = 0
        self._source_extent_wgs84: tuple[float, float, float, float] | None = None

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
        stack.enter_context(rasterio.Env(GDAL_CACHEMAX=256))
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
            self._tile_cache.clear()
            self._tile_cache_sizes.clear()
            self._tile_cache_bytes = 0

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
            stack.enter_context(rasterio.Env(GDAL_CACHEMAX=256))
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
        candidates: list[OverlapCandidate] = []
        # Every EEA layer in a group shares one grid, so the covering window and
        # the raster-bounds test depend on the grid, not the layer. Resolving
        # them once per grid keeps identical windows while avoiding tens of
        # repeated coordinate computations for each polygon.
        windows: dict[tuple[object, ...], Window | None] = {}
        for layer, dataset in datasets:
            window = self._shared_window(windows, dataset, polygon)
            if window is None:
                continue
            cell_geometry = _merge_tile_cells(self._raster_tile_cells(layer, dataset, window))
            if cell_geometry is not None:
                candidates.append(
                    OverlapCandidate(
                        layer.code,
                        layer.name,
                        cell_geometry,
                        components_are_disjoint=_has_disjoint_components(cell_geometry),
                    )
                )
        return choose_winner(
            polygon,
            candidates,
            source_version=self._source_version,
            on_error=self._count_intersection_error,
        )

    def _shared_window(
        self,
        windows: dict[tuple[object, ...], Window | None],
        dataset: rasterio.DatasetReader,
        polygon: BaseGeometry,
    ) -> Window | None:
        key = (tuple(dataset.transform)[:6], dataset.width, dataset.height)
        if key in windows:
            return windows[key]
        window = self._covering_window(dataset, polygon)
        windows[key] = window
        return window

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
            valid = _is_epsg_3035(crs)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"reference raster {path} is not EPSG:3035")

    def _raster_tile_cells(
        self,
        layer: RasterLayer,
        dataset: rasterio.DatasetReader,
        window: Window,
    ) -> list[BaseGeometry]:
        return [
            geometry
            for row in _raster_tile_indices(window.row_off, window.height)
            for column in _raster_tile_indices(window.col_off, window.width)
            if (geometry := self._cached_tile_geometry(layer, dataset, row, column)) is not None
        ]

    def _cached_tile_geometry(
        self,
        layer: RasterLayer,
        dataset: rasterio.DatasetReader,
        row: int,
        column: int,
    ) -> BaseGeometry | None:
        key = (layer.code, row, column)
        if key in self._tile_cache:
            geometry = self._tile_cache[key]
            self._tile_cache.move_to_end(key)
            self._tile_cache_hits += 1
            return geometry
        self._tile_cache_misses += 1
        window = _raster_tile_window(dataset, row, column)
        valid = self._positive_mask(dataset, window)
        geometry = (
            None if valid is None else _mask_geometry(valid, dataset.window_transform(window))
        )
        self._cache_tile_geometry(key, geometry)
        return geometry

    def _cache_tile_geometry(
        self,
        key: tuple[str, int, int],
        geometry: BaseGeometry | None,
    ) -> None:
        estimated_size = _tile_geometry_cache_size(geometry)
        if estimated_size > self._tile_cache_byte_budget:
            return
        while self._tile_cache_bytes + estimated_size > self._tile_cache_byte_budget:
            oldest = next(iter(self._tile_cache))
            self._evict_tile_geometry(oldest)
        self._tile_cache[key] = geometry
        self._tile_cache_sizes[key] = estimated_size
        self._tile_cache_bytes += estimated_size

    def _evict_tile_geometry(self, key: tuple[str, int, int]) -> None:
        self._tile_cache.pop(key)
        self._tile_cache_bytes -= self._tile_cache_sizes.pop(key)

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


def wgs84_envelope(
    bounds: tuple[float, float, float, float],
    source_crs: str = "EPSG:3035",
    pad: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return a conservative WGS84 envelope for a projected extent.

    ``transform_bounds`` densifies the edges, so the box covers the curved
    projected boundary rather than only its corners, and the pad keeps it a
    strict superset. It prunes polygons that cannot reach a reference at all;
    every surviving polygon is still projected and intersected exactly.
    """

    return _raster_wgs84_envelope(bounds, source_crs=source_crs, pad=pad)


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
