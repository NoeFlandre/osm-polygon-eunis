"""Select labels from exact intersections with per-code support unions."""

import logging
import math
from collections.abc import Callable, Iterable
from typing import TypeGuard

from shapely import area as shapely_area
from shapely import get_parts
from shapely import intersection as shapely_intersection
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .domain import EunisResult, EunisTaxonomy, OverlapCandidate
from .geometry import is_usable

logger = logging.getLogger(__name__)

_MIN_ASSIGNED_PERCENTAGE = 80.0
_MAX_PERCENTAGE = 100.0
_BOUNDARY_TOLERANCE_PERCENTAGE = 1e-9


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
    taxonomy: EunisTaxonomy | None = None,
) -> EunisResult:
    """Select one exact code from same-code support unions.

    The reported percentage uses the full OSM polygon area. A code is assigned
    only when exactly one per-code union covers at least 80% of that area.
    ``on_error`` is called for each support candidate dropped because it is
    unusable or its class union or exact intersection failed. ``extra_areas``
    must contain at most one pre-unioned ``(area, code, name)`` value per code.
    """

    if not _usable_polygon(polygon):
        return _empty_result()

    overlaps = _positive_overlaps(polygon, candidates, on_error, taxonomy)
    extra_areas = tuple(extra_areas)
    if taxonomy is not None and extra_areas:
        raise ValueError("scalar support areas cannot be rolled up through an EUNIS hierarchy")
    overlaps.extend(extra_areas)
    return result_from_class_areas(overlaps, polygon.area, source_version, taxonomy=taxonomy)


def result_from_class_areas(
    overlaps: Iterable[tuple[float, str, str]],
    polygon_area: float,
    source_version: str | None,
    *,
    taxonomy: EunisTaxonomy | None = None,
) -> EunisResult:
    """Select one exact class from pre-unioned full-polygon support areas.

    A class is eligible only when its union coverage is at least 80% of the
    original polygon area. If multiple distinct class codes qualify, the label
    stays null and the maximum class-union percentage remains in the result.
    """
    if polygon_area <= 0.0:
        return _empty_result()
    class_coverage = _class_coverages(overlaps, polygon_area)
    if not class_coverage:
        return _empty_result()
    return _result_from_class_coverages(class_coverage, source_version, taxonomy)


def _result_from_class_coverages(
    class_coverage: dict[str, tuple[float, str]],
    source_version: str | None,
    taxonomy: EunisTaxonomy | None,
) -> EunisResult:
    best_code = min(class_coverage, key=lambda code: (-class_coverage[code][0], code))
    qualifying = _qualifying_codes(class_coverage)
    if taxonomy is not None:
        qualifying = _deepest_qualifying_codes(qualifying, taxonomy)
    return _result_for_qualifying_codes(
        class_coverage,
        best_code,
        qualifying,
        source_version,
        taxonomy,
    )


def _result_for_qualifying_codes(
    class_coverage: dict[str, tuple[float, str]],
    best_code: str,
    qualifying: list[str],
    source_version: str | None,
    taxonomy: EunisTaxonomy | None,
) -> EunisResult:
    if not qualifying:
        return EunisResult(None, None, class_coverage[best_code][0], source_version)
    if len(qualifying) > 1:
        maximum = max(class_coverage[code][0] for code in qualifying)
        return EunisResult(None, None, maximum, source_version)
    return _assigned_class_result(class_coverage, qualifying[0], source_version, taxonomy)


def _assigned_class_result(
    class_coverage: dict[str, tuple[float, str]],
    code: str,
    source_version: str | None,
    taxonomy: EunisTaxonomy | None,
) -> EunisResult:
    coverage, name = class_coverage[code]
    if taxonomy is not None:
        name = taxonomy.labels.get(code, name)
    return EunisResult(code, name, coverage, source_version)


def _deepest_qualifying_codes(
    qualifying: list[str],
    taxonomy: EunisTaxonomy,
) -> list[str]:
    qualifying_set = set(qualifying)
    return [
        code
        for code in qualifying
        if not any(
            code in taxonomy.ancestors_including_self(other)[1:]
            for other in qualifying_set
            if other != code
        )
    ]


def _class_coverages(
    overlaps: Iterable[tuple[float, str, str]],
    polygon_area: float,
) -> dict[str, tuple[float, str]]:
    by_code: dict[str, tuple[float, str]] = {}
    for area, code, name in overlaps:
        if area <= 0.0:
            continue
        if code in by_code:
            raise ValueError(f"class support areas must be unioned before selection: {code}")
        percentage = _percentage(area, polygon_area)
        if abs(percentage - _MIN_ASSIGNED_PERCENTAGE) <= _BOUNDARY_TOLERANCE_PERCENTAGE:
            percentage = _MIN_ASSIGNED_PERCENTAGE
        by_code[code] = (percentage, name)
    return by_code


def _qualifying_codes(class_coverage: dict[str, tuple[float, str]]) -> list[str]:
    return [
        code
        for code, (percentage, _name) in class_coverage.items()
        if percentage >= _MIN_ASSIGNED_PERCENTAGE
    ]


def _usable_polygon(polygon: BaseGeometry | None) -> TypeGuard[BaseGeometry]:
    return is_usable(polygon) and polygon.area > 0


def _positive_overlaps(
    polygon: BaseGeometry,
    candidates: Iterable[OverlapCandidate],
    on_error: Callable[[], None],
    taxonomy: EunisTaxonomy | None = None,
) -> list[tuple[float, str, str]]:
    by_code = _group_candidates(candidates, on_error)
    by_node = _union_supports_by_node(by_code, on_error, taxonomy)
    return _overlap_areas_by_node(polygon, by_node, on_error)


def _union_supports_by_node(
    by_code: dict[str, list[OverlapCandidate]],
    on_error: Callable[[], None],
    taxonomy: EunisTaxonomy | None,
) -> dict[str, list[OverlapCandidate]]:
    by_node: dict[str, list[OverlapCandidate]] = {}
    for code, supports in by_code.items():
        candidate = _class_union_candidate(code, supports, on_error)
        if candidate is not None:
            _append_support_to_taxonomy_nodes(by_node, candidate, taxonomy)
    return by_node


def _append_support_to_taxonomy_nodes(
    by_node: dict[str, list[OverlapCandidate]],
    candidate: OverlapCandidate,
    taxonomy: EunisTaxonomy | None,
) -> None:
    nodes = (
        taxonomy.ancestors_including_self(candidate.code)
        if taxonomy is not None
        else (candidate.code,)
    )
    for node in nodes:
        name = taxonomy.labels[node] if taxonomy is not None else candidate.name
        by_node.setdefault(node, []).append(
            OverlapCandidate(
                node,
                name,
                candidate.geometry,
                error_count=candidate.error_count,
            )
        )


def _overlap_areas_by_node(
    polygon: BaseGeometry,
    by_node: dict[str, list[OverlapCandidate]],
    on_error: Callable[[], None],
) -> list[tuple[float, str, str]]:
    overlaps: list[tuple[float, str, str]] = []
    for node, supports in by_node.items():
        candidate = _class_union_candidate(node, supports, on_error)
        if candidate is None:
            continue
        area = _intersection_area(
            polygon,
            candidate,
            on_error,
            error_count=sum(support.error_count for support in supports),
        )
        if area is not None:
            overlaps.append((area, candidate.code, candidate.name))
    return overlaps


def _group_candidates(
    candidates: Iterable[OverlapCandidate],
    on_error: Callable[[], None],
) -> dict[str, list[OverlapCandidate]]:
    by_code: dict[str, list[OverlapCandidate]] = {}
    for candidate in candidates:
        if not is_usable(candidate.geometry):
            logger.warning("skipping unusable EUNIS candidate %s", candidate.code)
            _report_errors(on_error, candidate.error_count)
            continue
        by_code.setdefault(candidate.code, []).append(candidate)
    return by_code


def _class_union_candidate(
    code: str,
    supports: list[OverlapCandidate],
    on_error: Callable[[], None],
) -> OverlapCandidate | None:
    name = _class_name(code, supports)
    if len(supports) == 1:
        return supports[0]
    geometry = _union_class_geometry(code, supports, on_error)
    if geometry is None:
        return None
    return OverlapCandidate(
        code,
        name,
        geometry,
        error_count=sum(support.error_count for support in supports),
    )


def _class_name(code: str, supports: list[OverlapCandidate]) -> str:
    names = {support.name for support in supports}
    if len(names) != 1:
        raise ValueError(f"EUNIS class {code} has conflicting names")
    return names.pop()


def _union_class_geometry(
    code: str,
    supports: list[OverlapCandidate],
    on_error: Callable[[], None],
) -> BaseGeometry | None:
    try:
        geometry = unary_union([support.geometry for support in supports])
    except (GEOSException, ValueError, RuntimeError) as error:
        return _failed_class_union(code, supports, on_error, error)
    if not is_usable(geometry):
        return _failed_class_union(code, supports, on_error)
    return geometry


def _failed_class_union(
    code: str,
    supports: list[OverlapCandidate],
    on_error: Callable[[], None],
    error: Exception | None = None,
) -> None:
    if error is None:
        logger.warning("skipping unusable EUNIS support union for %s", code)
    else:
        logger.warning("skipping EUNIS class %s after support union error: %s", code, error)
    _report_errors(on_error, sum(support.error_count for support in supports))


def _intersection_area(
    polygon: BaseGeometry,
    candidate: OverlapCandidate,
    on_error: Callable[[], None],
    *,
    error_count: int = 1,
) -> float | None:
    if not is_usable(candidate.geometry):
        logger.warning("skipping unusable EUNIS candidate %s", candidate.code)
        _report_errors(on_error, error_count)
        return None
    try:
        area = _exact_intersection_area(polygon, candidate)
    except (GEOSException, ValueError, RuntimeError) as error:
        logger.warning(
            "skipping EUNIS candidate %s after intersection error: %s",
            candidate.code,
            error,
        )
        _report_errors(on_error, error_count)
        return None
    return area if area > 0.0 else None


def _report_errors(on_error: Callable[[], None], count: int) -> None:
    for _ in range(count):
        on_error()


def _exact_intersection_area(polygon: BaseGeometry, candidate: OverlapCandidate) -> float:
    if not candidate.components_are_disjoint:
        return float(polygon.intersection(candidate.geometry).area)
    components = get_parts(candidate.geometry)
    if not len(components):
        return 0.0
    return float(shapely_area(shapely_intersection(polygon, components)).sum())


def _percentage(area: float, polygon_area: float) -> float:
    if not math.isfinite(area) or not math.isfinite(polygon_area) or polygon_area <= 0.0:
        raise ValueError(
            "class support and polygon areas must be finite, with a positive polygon area"
        )
    percentage = 100.0 * area / float(polygon_area)
    if not math.isfinite(percentage):
        raise ValueError("class overlap percentage must be finite")
    if percentage > _MAX_PERCENTAGE:
        if percentage - _MAX_PERCENTAGE <= _BOUNDARY_TOLERANCE_PERCENTAGE:
            return _MAX_PERCENTAGE
        raise ValueError("class support area exceeds polygon area")
    return max(0.0, percentage)


def prefer_result(current: EunisResult, candidate: EunisResult) -> EunisResult:
    """Keep the larger exact-overlap result across streamed reference groups."""

    if current.is_empty:
        return candidate
    if candidate.is_empty:
        return current
    _require_same_source_version(current, candidate)
    if _qualifying_classes_conflict(current, candidate):
        return _ambiguous_result(current, candidate)
    if _outranks(_rank(candidate), _rank(current)):
        return candidate
    return current


def _ambiguous_result(current: EunisResult, candidate: EunisResult) -> EunisResult:
    percentage = max(
        value
        for value in (current.overlap_percentage, candidate.overlap_percentage)
        if value is not None
    )
    return EunisResult(None, None, percentage, current.source_version)


def _qualifying_classes_conflict(current: EunisResult, candidate: EunisResult) -> bool:
    if _is_ambiguous(current) or _is_ambiguous(candidate):
        return True
    return (
        current.code != candidate.code
        and _meets_assignment_threshold(current)
        and _meets_assignment_threshold(candidate)
    )


def _is_ambiguous(result: EunisResult) -> bool:
    return result.code is None and _meets_assignment_threshold(result)


def _meets_assignment_threshold(result: EunisResult) -> bool:
    percentage = result.overlap_percentage
    return percentage is not None and percentage >= _MIN_ASSIGNED_PERCENTAGE


def _outranks(candidate: tuple[float, str], current: tuple[float, str]) -> bool:
    if candidate[0] != current[0]:
        return candidate[0] > current[0]
    return candidate[1] < current[1]


def _rank(result: EunisResult) -> tuple[float, str]:
    if result.overlap_percentage is None:
        raise ValueError("EUNIS rank requires an overlap percentage")
    return result.overlap_percentage, result.code or ""


def _require_same_source_version(current: EunisResult, candidate: EunisResult) -> None:
    if current.source_version != candidate.source_version:
        raise ValueError("cannot merge EUNIS results from different source versions")
