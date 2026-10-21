"""One-pass adapters over the complete raw polygon inventory.

These iterators intentionally accept no document, relevance, or evaluation
filters. They emit one attempt receipt for every input row and retain only the
current polygon's label result. ``export_polygon_attempts`` hands bounded
batches to the shared disk-backed exporter. A large individual LUCAS query can
still hold many footprint geometries in memory, so production waits for bounded
per-polygon aggregation and verified reference pins.
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
    WORLDCOVER_AREA_MEASUREMENT,
    WorldCoverEvidence,
    worldcover_supports,
)


class WorldCoverSourceAttempt(Protocol):
    """Record protocol emitted by ``osm-worldcover``'s raw-row adapter."""

    @property
    def source_polygon_id(self) -> str:
        """Stable source-polygon identifier for one raw input row."""

        ...

    @property
    def polygon_area_m2(self) -> float | None:
        """Full source polygon area in WGS84 ellipsoidal equal-area square metres."""

        ...

    @property
    def class_areas_m2(self) -> tuple[tuple[int, float], ...]:
        """Class support areas in square metres on the full polygon denominator."""

        ...

    @property
    def class_evidence_ids(self) -> tuple[tuple[int, tuple[str, ...]], ...]:
        """Raster tile IDs that contribute positive support to each class."""

        ...

    @property
    def area_measurement(self) -> str:
        """Pinned area algorithm used for both numerator and denominator."""

        ...

    @property
    def attempted_tiles(self) -> tuple[str, ...]:
        """Tile identifiers visited for this polygon."""

        ...

    @property
    def missing_tiles(self) -> tuple[str, ...]:
        """Tile identifiers not published by the reference."""

        ...

    @property
    def extraction_error(self) -> str | None:
        """Terminal extractor failure, if the attempt did not complete."""

        ...


def iter_worldcover_attempts(
    raw_polygons: Iterable[PoolPolygonInput],
    evidence_for: Callable[[PoolPolygonInput], WorldCoverEvidence],
    reference_pin: ReferencePin,
) -> Iterator[PoolPolygonLabelRecords]:
    """Attempt the canonical raster evidence provider once for every raw row."""
    for _polygon, records in iter_worldcover_rows(raw_polygons, evidence_for, reference_pin):
        yield records


def iter_worldcover_rows(
    raw_polygons: Iterable[PoolPolygonInput],
    evidence_for: Callable[[PoolPolygonInput], WorldCoverEvidence],
    reference_pin: ReferencePin,
) -> Iterator[tuple[PoolPolygonInput, PoolPolygonLabelRecords]]:
    """Yield each raw polygon beside its label result in one source pass."""

    for polygon in _raw_polygons_in_order(raw_polygons):
        if polygon.geometry is None or polygon.geometry.is_empty or not polygon.geometry.is_valid:
            evidence = WorldCoverEvidence(
                source_polygon_id=polygon.source_polygon_id,
                polygon_area_m2=None,
                class_areas_m2=(),
                area_measurement=WORLDCOVER_AREA_MEASUREMENT,
                extraction_error="geometry_unusable",
            )
        else:
            try:
                evidence = evidence_for(polygon)
            except (OSError, RuntimeError, ValueError) as error:
                evidence = WorldCoverEvidence(
                    source_polygon_id=polygon.source_polygon_id,
                    polygon_area_m2=None,
                    class_areas_m2=(),
                    area_measurement=WORLDCOVER_AREA_MEASUREMENT,
                    extraction_error=f"worldcover_reference_query_failed:{type(error).__name__}",
                )
        yield polygon, worldcover_supports(polygon, evidence, reference_pin)


def iter_worldcover_source_attempts(
    raw_polygons: Iterable[PoolPolygonInput],
    source_attempts: Iterable[WorldCoverSourceAttempt],
    reference_pin: ReferencePin,
) -> Iterator[PoolPolygonLabelRecords]:
    """Join canonical ``osm-worldcover`` evidence to the same full raw order."""
    for _polygon, records in iter_worldcover_source_rows(
        raw_polygons, source_attempts, reference_pin
    ):
        yield records


def iter_worldcover_source_rows(
    raw_polygons: Iterable[PoolPolygonInput],
    source_attempts: Iterable[WorldCoverSourceAttempt],
    reference_pin: ReferencePin,
) -> Iterator[tuple[PoolPolygonInput, PoolPolygonLabelRecords]]:
    """Join source evidence and raw polygons once, yielding bounded-runner rows."""

    evidence_iterator = iter(source_attempts)
    for polygon in _raw_polygons_in_order(raw_polygons):
        try:
            evidence = next(evidence_iterator)
        except StopIteration as error:
            raise ValueError("WorldCover raw polygon and attempt counts differ") from error
        adapted = WorldCoverEvidence(
            source_polygon_id=evidence.source_polygon_id,
            polygon_area_m2=evidence.polygon_area_m2,
            class_areas_m2=evidence.class_areas_m2,
            area_measurement=evidence.area_measurement,
            attempted_tiles=evidence.attempted_tiles,
            missing_tiles=evidence.missing_tiles,
            extraction_error=evidence.extraction_error,
            class_evidence_ids=getattr(evidence, "class_evidence_ids", ()),
        )
        yield polygon, worldcover_supports(polygon, adapted, reference_pin)
    if next(evidence_iterator, None) is not None:
        raise ValueError("WorldCover raw polygon and attempt counts differ")


def iter_lucas_attempts(
    raw_polygons: Iterable[PoolPolygonInput],
    reader: LucasGeoPackageReader,
    taxonomy: LucasTaxonomy,
    reference_pin: ReferencePin,
) -> Iterator[PoolPolygonLabelRecords]:
    """Query the pinned survey layer once per raw row without upstream filters."""
    for _polygon, records in iter_lucas_rows(raw_polygons, reader, taxonomy, reference_pin):
        yield records


def iter_lucas_rows(
    raw_polygons: Iterable[PoolPolygonInput],
    reader: LucasGeoPackageReader,
    taxonomy: LucasTaxonomy,
    reference_pin: ReferencePin,
) -> Iterator[tuple[PoolPolygonInput, PoolPolygonLabelRecords]]:
    """Yield each raw polygon beside its LUCAS decision in one source pass."""

    if not reader.is_open:
        raise RuntimeError("LUCAS runner requires an open, verified GeoPackage reader")
    if reader.taxonomy.sha256 != taxonomy.sha256:
        raise ValueError("LUCAS runner taxonomy and reader taxonomy do not match")
    for polygon in _raw_polygons_in_order(raw_polygons):
        if polygon.geometry is None or polygon.geometry.is_empty or not polygon.geometry.is_valid:
            query = LucasFeatureQuery((), (), "geometry_unusable")
        else:
            try:
                query = reader.query(polygon.geometry)
            except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
                query = LucasFeatureQuery(
                    (), (), f"lucas_reference_query_failed:{type(error).__name__}"
                )
        yield polygon, measure_lucas_supports(polygon, query, taxonomy, reference_pin)


def _raw_polygons_in_order(
    raw_polygons: Iterable[PoolPolygonInput],
) -> Iterator[PoolPolygonInput]:
    """Preserve every raw row in order without a source-wide in-memory set.

    The shared disk-backed pool exporter checks global native-key and row-ID
    uniqueness after all attempts have streamed into its SQLite index.
    """
    yield from raw_polygons
