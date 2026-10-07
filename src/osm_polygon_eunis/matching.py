"""Pure deterministic selection of the largest actual geometry overlap."""

import logging
from collections.abc import Callable, Iterable
from typing import TypeGuard

from shapely import area as shapely_area
from shapely import get_parts
from shapely import intersection as shapely_intersection
from shapely.geometry.base import BaseGeometry

from .domain import EunisResult, OverlapCandidate
from .geometry import is_usable

logger = logging.getLogger(__name__)


def _ignore_error() -> None:
    """Default intersection-error hook: drop the candidate without counting it."""


def _empty_result() -> EunisResult:
    return EunisResult(None, None, None, None)


def choose_winner(
    polygon: BaseGeometry | None,
    candidates: Iterable[OverlapCandidate],
    *,
    source_version: str | None,
    on_error: Callable[[], None] = _ignore_error,
    extra_areas: Iterable[tuple[float, str, str]] = (),
) -> EunisResult:
    """Select the candidate with the largest actual intersection area.

    ``on_error`` is called once per candidate dropped because the exact GEOS
    intersection raised; the selected result is the same either way.
    ``extra_areas`` are precomputed ``(area, code, name)`` overlaps that compete
    with the candidates under the same ordering.
    """

    if not _usable_polygon(polygon):
        return _empty_result()

    overlaps = [
        (area, candidate.code, candidate.name)
        for area, candidate in _positive_overlaps(polygon, candidates, on_error)
    ]
    overlaps.extend(extra_areas)
    if not overlaps:
        return _empty_result()
    area, code, name = min(overlaps, key=lambda item: (-item[0], item[1]))
    return EunisResult(code, name, _percentage(area, polygon.area), source_version)


def _usable_polygon(polygon: BaseGeometry | None) -> TypeGuard[BaseGeometry]:
    return is_usable(polygon) and polygon.area > 0


def _positive_overlaps(
    polygon: BaseGeometry,
    candidates: Iterable[OverlapCandidate],
    on_error: Callable[[], None],
) -> list[tuple[float, OverlapCandidate]]:
    overlaps: list[tuple[float, OverlapCandidate]] = []
    for candidate in candidates:
        area = _intersection_area(polygon, candidate, on_error)
        if area is not None:
            overlaps.append((area, candidate))
    return overlaps


def _intersection_area(
    polygon: BaseGeometry,
    candidate: OverlapCandidate,
    on_error: Callable[[], None],
) -> float | None:
    if not is_usable(candidate.geometry):
        return None
    try:
        area = _exact_intersection_area(polygon, candidate)
    except (ValueError, RuntimeError) as error:
        logger.warning(
            "skipping EUNIS candidate %s after intersection error: %s",
            candidate.code,
            error,
        )
        on_error()
        return None
    return area if area > 0.0 else None


def _exact_intersection_area(polygon: BaseGeometry, candidate: OverlapCandidate) -> float:
    if not candidate.components_are_disjoint:
        return float(polygon.intersection(candidate.geometry).area)
    components = get_parts(candidate.geometry)
    if not len(components):
        return 0.0
    return float(shapely_area(shapely_intersection(polygon, components)).sum())


def _percentage(area: float, polygon_area: float) -> float:
    return max(0.0, min(100.0, 100.0 * area / float(polygon_area)))


def prefer_result(current: EunisResult, candidate: EunisResult) -> EunisResult:
    """Keep the larger exact-overlap result across streamed reference groups."""

    if current.is_empty:
        return candidate
    if candidate.is_empty:
        return current
    _require_same_source_version(current, candidate)
    return candidate if _outranks(_rank(candidate), _rank(current)) else current


def _outranks(candidate: tuple[float, str], current: tuple[float, str]) -> bool:
    if candidate[0] != current[0]:
        return candidate[0] > current[0]
    return candidate[1] < current[1]


def _rank(result: EunisResult) -> tuple[float, str]:
    if result.code is None or result.overlap_percentage is None:
        raise ValueError("EUNIS rank requires a code and overlap percentage")
    return result.overlap_percentage, result.code


def _require_same_source_version(current: EunisResult, candidate: EunisResult) -> None:
    if current.source_version != candidate.source_version:
        raise ValueError("cannot merge EUNIS results from different source versions")
