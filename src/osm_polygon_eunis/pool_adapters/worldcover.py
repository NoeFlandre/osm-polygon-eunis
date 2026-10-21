"""WorldCover evidence adaptation from its canonical raster extractor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from osm_polygon_eunis.pool_contract import AssignmentKind, LabelLevel, ReferencePin

from .common import (
    PoolPolygonInput,
    PoolPolygonLabelRecords,
    SupportArea,
    SupportAttempt,
    label_polygon,
)

WORLDCOVER_CODES: Final[frozenset[int]] = frozenset({10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100})
WORLDCOVER_LEVELS: Final[tuple[LabelLevel, ...]] = (LabelLevel("class", supported=True),)
WORLDCOVER_AREA_MEASUREMENT: Final[str] = (
    "pyproj.Geod(ellps=WGS84); boundary-cell intersections geodesic; v1"
)


@dataclass(frozen=True, slots=True)
class WorldCoverEvidence:
    """Per-polygon physical-area evidence from ``osm-worldcover``'s pool adapter.

    Numerators and denominator use WGS84 ellipsoidal square metres. NoData and
    unavailable-tile pixels remain unattributed in the full-polygon denominator.
    Existing published degree-based fractions are legacy evidence and must not
    be passed through this pool adapter.
    """

    source_polygon_id: str
    polygon_area_m2: float | None
    class_areas_m2: tuple[tuple[int, float], ...]
    area_measurement: str
    attempted_tiles: tuple[str, ...] = ()
    missing_tiles: tuple[str, ...] = ()
    extraction_error: str | None = None


def worldcover_supports(
    polygon: PoolPolygonInput,
    evidence: WorldCoverEvidence,
    reference_pin: ReferencePin,
) -> PoolPolygonLabelRecords:
    """Turn geodesic class areas into full-area support candidates for one polygon."""

    if polygon.source_polygon_id != evidence.source_polygon_id:
        raise ValueError("WorldCover evidence does not match the raw polygon key")
    if reference_pin.support_unit != "m^2":
        raise ValueError("WorldCover pool support must use physical square metres")
    if evidence.area_measurement != WORLDCOVER_AREA_MEASUREMENT:
        raise ValueError("WorldCover support area measurement is not the pinned geodesic method")
    if (
        polygon.snapshot is not None
        and polygon.geometry is not None
        and not polygon.geometry.is_empty
        and polygon.geometry.is_valid
    ):
        if polygon.snapshot.crs != "EPSG:4326":
            raise ValueError("WorldCover source polygon snapshots must use EPSG:4326")
        if polygon.snapshot.area_unit != "m^2":
            raise ValueError("WorldCover snapshot area must use physical square metres")

    if evidence.extraction_error is not None:
        attempt = SupportAttempt(
            (),
            f"worldcover_extraction_failed:{evidence.extraction_error}",
            attempted_tile_ids=evidence.attempted_tiles,
            missing_tile_ids=evidence.missing_tiles,
        )
    elif (
        polygon.snapshot is None
        or polygon.snapshot.original_area is None
        or not math.isfinite(polygon.snapshot.original_area)
        or polygon.snapshot.original_area <= 0.0
        or evidence.polygon_area_m2 is None
        or not math.isfinite(evidence.polygon_area_m2)
        or evidence.polygon_area_m2 <= 0.0
    ):
        attempt = SupportAttempt(
            (),
            "geometry_area_unusable",
            attempted_tile_ids=evidence.attempted_tiles,
            missing_tile_ids=evidence.missing_tiles,
        )
    else:
        if not math.isclose(
            polygon.snapshot.original_area,
            evidence.polygon_area_m2,
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise ValueError("WorldCover snapshot area differs from full geodesic polygon area")
        attempt = SupportAttempt(
            _area_supports(
                evidence.class_areas_m2,
                polygon.snapshot.original_area,
                evidence.attempted_tiles,
            ),
            attempted_tile_ids=evidence.attempted_tiles,
            missing_tile_ids=evidence.missing_tiles,
        )
    return label_polygon(polygon, attempt, WORLDCOVER_LEVELS, reference_pin)


def _area_supports(
    class_areas_m2: tuple[tuple[int, float], ...],
    original_area: float,
    evidence_ids: tuple[str, ...],
) -> tuple[SupportArea, ...]:
    if not math.isfinite(original_area) or original_area <= 0.0:
        raise ValueError("WorldCover evidence needs a positive full polygon area")
    seen: set[int] = set()
    supports: list[SupportArea] = []
    area_total = 0.0
    for code, area in class_areas_m2:
        if code not in WORLDCOVER_CODES:
            raise ValueError(f"unknown WorldCover class code {code}")
        if code in seen:
            raise ValueError(f"WorldCover code {code} must be unioned before adaptation")
        seen.add(code)
        if not math.isfinite(area) or area < 0.0 or area > original_area:
            raise ValueError(f"WorldCover class area for {code} must be within polygon area")
        area_total += area
        if area > 0.0:
            supports.append(
                SupportArea("class", str(code), area, AssignmentKind.DIRECT, evidence_ids)
            )
    if area_total > original_area * (1.0 + 1e-9):
        raise ValueError("WorldCover class areas exceed the full geodesic denominator")
    return tuple(supports)
