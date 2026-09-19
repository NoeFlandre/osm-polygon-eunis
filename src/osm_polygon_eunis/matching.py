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

    if polygon is None or polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        return _empty_result()

    overlaps: list[tuple[float, OverlapCandidate]] = []
    for candidate in candidates:
        if candidate.geometry.is_empty or not candidate.geometry.is_valid:
            continue
        try:
            area = float(polygon.intersection(candidate.geometry).area)
        except (ValueError, RuntimeError):
            continue
        if area > 0.0:
            overlaps.append((area, candidate))

    if not overlaps:
        return _empty_result()
    area, winner = sorted(overlaps, key=lambda item: (-item[0], item[1].code))[0]
    percentage = max(0.0, min(100.0, 100.0 * area / float(polygon.area)))
    return EunisResult(winner.code, winner.name, percentage, source_version)
