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
    "WGS84 ellipsoidal CEA (lat_ts=30); geographic edges densified <=0.01deg; v2"
)
WORLDCOVER_REFERENCE_ID_NAMESPACE: Final[str] = "worldcover-tile-id"


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
    class_evidence_ids: tuple[tuple[int, tuple[str, ...]], ...] = ()


def worldcover_supports(
    polygon: PoolPolygonInput,
    evidence: WorldCoverEvidence,
    reference_pin: ReferencePin,
) -> PoolPolygonLabelRecords:
    """Turn equal-area class measurements into full-area support candidates."""

    if polygon.source_polygon_id != evidence.source_polygon_id:
        raise ValueError("WorldCover evidence does not match the raw polygon key")
    if reference_pin.support_unit != "m^2":
        raise ValueError("WorldCover pool support must use physical square metres")
    if evidence.area_measurement != WORLDCOVER_AREA_MEASUREMENT:
        raise ValueError("WorldCover support area measurement is not the pinned equal-area method")
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
        reason = (
            evidence.extraction_error
            if evidence.extraction_error in {"geometry_unusable", "source_polygon_id_missing"}
            else f"worldcover_extraction_failed:{evidence.extraction_error}"
        )
        attempt = SupportAttempt(
            (),
            reason,
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
            raise ValueError("WorldCover snapshot area differs from full equal-area polygon area")
        attempt = SupportAttempt(
            _area_supports(
                evidence.class_areas_m2,
                polygon.snapshot.original_area,
                evidence.class_evidence_ids,
            ),
            attempted_tile_ids=evidence.attempted_tiles,
            missing_tile_ids=evidence.missing_tiles,
        )
    return label_polygon(
        polygon,
        attempt,
        WORLDCOVER_LEVELS,
        reference_pin,
        reference_id_namespace=WORLDCOVER_REFERENCE_ID_NAMESPACE,
    )


def _area_supports(
    class_areas_m2: tuple[tuple[int, float], ...],
    original_area: float,
    class_evidence_ids: tuple[tuple[int, tuple[str, ...]], ...],
) -> tuple[SupportArea, ...]:
    if not math.isfinite(original_area) or original_area <= 0.0:
        raise ValueError("WorldCover evidence needs a positive full polygon area")
    seen: set[int] = set()
    evidence_by_class = _evidence_by_class(class_evidence_ids)
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
            evidence_ids = evidence_by_class.get(code, ())
            if not evidence_ids:
                raise ValueError(f"WorldCover class {code} is missing contributing tile IDs")
            supports.append(
                SupportArea("class", str(code), area, AssignmentKind.DIRECT, evidence_ids)
            )
    extra_evidence = set(evidence_by_class) - seen
    if extra_evidence:
        raise ValueError(
            f"WorldCover support evidence has no class area for {sorted(extra_evidence)}"
        )
    if area_total > original_area * (1.0 + 1e-9):
        raise ValueError("WorldCover class areas exceed the full equal-area denominator")
    return tuple(supports)


def _evidence_by_class(
    class_evidence_ids: tuple[tuple[int, tuple[str, ...]], ...],
) -> dict[int, tuple[str, ...]]:
    """Index exact contributing tile IDs by raster class."""
    evidence_by_class: dict[int, tuple[str, ...]] = {}
    for code, evidence_ids in class_evidence_ids:
        if code not in WORLDCOVER_CODES:
            raise ValueError(f"WorldCover evidence uses unknown class code {code}")
        if code in evidence_by_class:
            raise ValueError(f"WorldCover code {code} repeats its support evidence")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"WorldCover code {code} repeats a support tile ID")
        evidence_by_class[code] = evidence_ids
    return evidence_by_class
