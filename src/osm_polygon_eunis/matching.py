"""Pure deterministic selection of the largest actual geometry overlap."""

from collections.abc import Iterable

from shapely.geometry.base import BaseGeometry

from .domain import EunisResult, OverlapCandidate


def _empty_result() -> EunisResult:
    return EunisResult(None, None, None, None)


def choose_winner(
    polygon: BaseGeometry | None,
    candidates: Iterable[OverlapCandidate],
    *,
    source_version: str | None,
) -> EunisResult:
    """Select the candidate with the largest actual intersection area."""

    if not _usable_polygon(polygon):
        return _empty_result()
    assert polygon is not None

    overlaps = _positive_overlaps(polygon, candidates)
    if not overlaps:
        return _empty_result()
    area, winner = min(overlaps, key=lambda item: (-item[0], item[1].code))
    return EunisResult(winner.code, winner.name, _percentage(area, polygon.area), source_version)


def _usable_polygon(polygon: BaseGeometry | None) -> bool:
    return polygon is not None and not polygon.is_empty and polygon.is_valid and polygon.area > 0


def _positive_overlaps(
    polygon: BaseGeometry,
    candidates: Iterable[OverlapCandidate],
) -> list[tuple[float, OverlapCandidate]]:
    overlaps: list[tuple[float, OverlapCandidate]] = []
    for candidate in candidates:
        area = _intersection_area(polygon, candidate)
        if area is not None:
            overlaps.append((area, candidate))
    return overlaps


def _intersection_area(
    polygon: BaseGeometry,
    candidate: OverlapCandidate,
) -> float | None:
    if candidate.geometry.is_empty or not candidate.geometry.is_valid:
        return None
    try:
        area = float(polygon.intersection(candidate.geometry).area)
    except (ValueError, RuntimeError):
        return None
    return area if area > 0.0 else None


def _percentage(area: float, polygon_area: float) -> float:
    return max(0.0, min(100.0, 100.0 * area / float(polygon_area)))


def prefer_result(current: EunisResult, candidate: EunisResult) -> EunisResult:
    """Keep the larger exact-overlap result across streamed reference groups."""

    if current.is_empty:
        return candidate
    if candidate.is_empty:
        return current
    _require_same_source_version(current, candidate)
    assert current.overlap_percentage is not None
    assert candidate.overlap_percentage is not None
    return _prefer_non_empty(current, candidate)


def _require_same_source_version(current: EunisResult, candidate: EunisResult) -> None:
    if current.source_version != candidate.source_version:
        raise ValueError("cannot merge EUNIS results from different source versions")


def _prefer_non_empty(current: EunisResult, candidate: EunisResult) -> EunisResult:
    assert current.overlap_percentage is not None
    assert candidate.overlap_percentage is not None
    if candidate.overlap_percentage != current.overlap_percentage:
        return _higher_percentage(current, candidate)
    return _prefer_code(current, candidate)


def _higher_percentage(current: EunisResult, candidate: EunisResult) -> EunisResult:
    assert current.overlap_percentage is not None
    assert candidate.overlap_percentage is not None
    return candidate if candidate.overlap_percentage > current.overlap_percentage else current


def _prefer_code(current: EunisResult, candidate: EunisResult) -> EunisResult:
    assert current.code is not None
    assert candidate.code is not None
    return candidate if candidate.code < current.code else current
