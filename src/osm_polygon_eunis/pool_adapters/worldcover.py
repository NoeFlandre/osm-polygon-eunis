"""WorldCover evidence adaptation from its canonical raster extractor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import shapely

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


@dataclass(frozen=True, slots=True)
class WorldCoverEvidence:
    """Per-polygon class fractions from ``osm-worldcover``'s raster extractor.

    Fractions use the polygon's full planar area in the raster CRS. The existing
    WorldCover extractor omits NoData and unavailable-tile pixels, so their
    share remains unattributed in the denominator.
    """

    source_polygon_id: str
    class_fractions: tuple[tuple[int, float], ...]
    attempted_tiles: tuple[str, ...] = ()
    missing_tiles: tuple[str, ...] = ()
    extraction_error: str | None = None


def worldcover_supports(
    polygon: PoolPolygonInput,
    evidence: WorldCoverEvidence,
    reference_pin: ReferencePin,
) -> PoolPolygonLabelRecords:
    """Turn class fractions into full-area support candidates for one polygon."""

    if polygon.source_polygon_id != evidence.source_polygon_id:
        raise ValueError("WorldCover evidence does not match the raw polygon key")
    if reference_pin.support_unit != "degree^2":
        raise ValueError("WorldCover v200 support must use its EPSG:4326 planar area unit")
    if (
        polygon.snapshot is not None
        and polygon.geometry is not None
        and not polygon.geometry.is_empty
        and polygon.geometry.is_valid
    ):
        if polygon.snapshot.crs != "EPSG:4326":
            raise ValueError("WorldCover source polygon snapshots must use EPSG:4326")
        geometry_area = float(shapely.area(polygon.geometry))
        if (
            polygon.snapshot.original_area is not None
            and math.isfinite(polygon.snapshot.original_area)
            and polygon.snapshot.original_area > 0.0
            and not math.isclose(
                geometry_area,
                polygon.snapshot.original_area,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("WorldCover snapshot area must be the full original polygon area")

    if evidence.extraction_error is not None:
        attempt = SupportAttempt(
            (),
            "worldcover_extraction_failed",
            attempted_tile_ids=evidence.attempted_tiles,
            missing_tile_ids=evidence.missing_tiles,
        )
    elif (
        polygon.snapshot is None
        or polygon.snapshot.original_area is None
        or not math.isfinite(polygon.snapshot.original_area)
        or polygon.snapshot.original_area <= 0.0
    ):
        attempt = SupportAttempt(
            (),
            "geometry_area_unusable",
            attempted_tile_ids=evidence.attempted_tiles,
            missing_tile_ids=evidence.missing_tiles,
        )
    else:
        attempt = SupportAttempt(
            _area_supports(
                evidence.class_fractions,
                polygon.snapshot.original_area,
                evidence.attempted_tiles,
            ),
            attempted_tile_ids=evidence.attempted_tiles,
            missing_tile_ids=evidence.missing_tiles,
        )
    return label_polygon(polygon, attempt, WORLDCOVER_LEVELS, reference_pin)


def _area_supports(
    class_fractions: tuple[tuple[int, float], ...],
    original_area: float,
    evidence_ids: tuple[str, ...],
) -> tuple[SupportArea, ...]:
    if not math.isfinite(original_area) or original_area <= 0.0:
        raise ValueError("WorldCover evidence needs a positive full polygon area")
    seen: set[int] = set()
    supports: list[SupportArea] = []
    fraction_total = 0.0
    for code, fraction in class_fractions:
        if code == 0:
            if not math.isfinite(fraction) or fraction < 0.0:
                raise ValueError("WorldCover NoData fraction must be finite and non-negative")
            continue
        if code not in WORLDCOVER_CODES:
            raise ValueError(f"unknown WorldCover class code {code}")
        if code in seen:
            raise ValueError(f"WorldCover code {code} must be unioned before adaptation")
        seen.add(code)
        if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError(f"WorldCover class fraction for {code} must be within [0, 1]")
        fraction_total += fraction
        numerator = fraction * original_area
        if numerator > 0.0:
            supports.append(
                SupportArea("class", str(code), numerator, AssignmentKind.DIRECT, evidence_ids)
            )
    if fraction_total > 1.0 + 1e-9:
        raise ValueError("WorldCover class fractions exceed the full-area denominator")
    return tuple(supports)
