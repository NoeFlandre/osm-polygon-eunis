"""One-pass adapters over the complete raw polygon inventory.

These iterators intentionally accept no document, relevance, or evaluation
filters. They emit one attempt receipt for every input row and retain only the
current polygon's support result. The shared pool builder/exporter still
materializes a full pool, so production execution waits for its bounded API.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol

from osm_polygon_eunis.pool_contract import ReferencePin

from .common import PoolPolygonInput, PoolPolygonLabelRecords
from .lucas import (
    LucasFeatureQuery,
    LucasGeoPackageReader,
    LucasTaxonomy,
    measure_lucas_supports,
)
from .worldcover import (
    WorldCoverEvidence,
    worldcover_supports,
)


class WorldCoverSourceAttempt(Protocol):
    """Record protocol emitted by ``osm-worldcover``'s raw-row adapter."""

    source_polygon_id: str
    class_fractions: tuple[tuple[int, float], ...]
    attempted_tiles: tuple[str, ...]
    missing_tiles: tuple[str, ...]
    extraction_error: str | None


def iter_worldcover_attempts(
    raw_polygons: Iterable[PoolPolygonInput],
    evidence_for: Callable[[PoolPolygonInput], WorldCoverEvidence],
    reference_pin: ReferencePin,
) -> Iterator[PoolPolygonLabelRecords]:
    """Attempt the canonical raster evidence provider once for every raw row."""

    for polygon in _unique_raw_polygons(raw_polygons):
        if polygon.geometry is None or polygon.geometry.is_empty or not polygon.geometry.is_valid:
            evidence = WorldCoverEvidence(
                source_polygon_id=polygon.source_polygon_id,
                class_fractions=(),
                extraction_error="geometry_unusable",
            )
        else:
            try:
                evidence = evidence_for(polygon)
            except (OSError, RuntimeError, ValueError) as error:
                evidence = WorldCoverEvidence(
                    source_polygon_id=polygon.source_polygon_id,
                    class_fractions=(),
                    extraction_error=f"worldcover_reference_query_failed:{type(error).__name__}",
                )
        yield worldcover_supports(polygon, evidence, reference_pin)


def iter_worldcover_source_attempts(
    raw_polygons: Iterable[PoolPolygonInput],
    source_attempts: Iterable[WorldCoverSourceAttempt],
    reference_pin: ReferencePin,
) -> Iterator[PoolPolygonLabelRecords]:
    """Join canonical ``osm-worldcover`` evidence to the same full raw order."""

    evidence_iterator = iter(source_attempts)
    for polygon in _unique_raw_polygons(raw_polygons):
        try:
            evidence = next(evidence_iterator)
        except StopIteration as error:
            raise ValueError("WorldCover raw polygon and attempt counts differ") from error
        adapted = WorldCoverEvidence(
            source_polygon_id=evidence.source_polygon_id,
            class_fractions=evidence.class_fractions,
            attempted_tiles=evidence.attempted_tiles,
            missing_tiles=evidence.missing_tiles,
            extraction_error=evidence.extraction_error,
        )
        yield worldcover_supports(polygon, adapted, reference_pin)
    if next(evidence_iterator, None) is not None:
        raise ValueError("WorldCover raw polygon and attempt counts differ")


def iter_lucas_attempts(
    raw_polygons: Iterable[PoolPolygonInput],
    reader: LucasGeoPackageReader,
    taxonomy: LucasTaxonomy,
    reference_pin: ReferencePin,
) -> Iterator[PoolPolygonLabelRecords]:
    """Query the pinned survey layer once per raw row without upstream filters."""

    if not reader.is_open:
        raise RuntimeError("LUCAS runner requires an open, verified GeoPackage reader")
    if reader.taxonomy.sha256 != taxonomy.sha256:
        raise ValueError("LUCAS runner taxonomy and reader taxonomy do not match")
    for polygon in _unique_raw_polygons(raw_polygons):
        if polygon.geometry is None or polygon.geometry.is_empty or not polygon.geometry.is_valid:
            query = LucasFeatureQuery((), (), "geometry_unusable")
        else:
            try:
                query = reader.query(polygon.geometry)
            except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
                query = LucasFeatureQuery(
                    (), (), f"lucas_reference_query_failed:{type(error).__name__}"
                )
        yield measure_lucas_supports(polygon, query, taxonomy, reference_pin)


def _unique_raw_polygons(
    raw_polygons: Iterable[PoolPolygonInput],
) -> Iterator[PoolPolygonInput]:
    """Reject duplicate keys rather than silently dropping source rows."""

    seen_native_keys = set()
    seen_inventory_ids: set[str] = set()
    for polygon in raw_polygons:
        native_key = polygon.inventory.native_key
        inventory_id = polygon.inventory.inventory_row_id
        if native_key in seen_native_keys:
            raise ValueError(f"raw polygon inventory repeats native key {native_key!r}")
        if inventory_id in seen_inventory_ids:
            raise ValueError(f"raw polygon inventory repeats row ID {inventory_id!r}")
        seen_native_keys.add(native_key)
        seen_inventory_ids.add(inventory_id)
        yield polygon
