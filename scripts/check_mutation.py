"""Fail CI when mutation results fall below the measured quality baseline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Each mutated module ratchets independently: its score may not fall below its entry.
MODULE_MINIMUM_SCORES: dict[str, float] = {
    "geometry": 93.02,
    "grid_overlap": 93.67,
    "matching": 99.2,
    "release_plan": 100.0,
    "transform": 98.3,
}


def _equivalent(reason: str, module: str, *mutants: str) -> dict[str, str]:
    return {f"osm_polygon_eunis.{module}.{mutant}": reason for mutant in mutants}


ALLOWED_MUTANTS: dict[str, str] = {
    **_equivalent(
        "Equivalent: the caller checks unequal percentages before comparing with >.",
        "matching",
        "x__outranks__mutmut_5",
    ),
    **_equivalent(
        "Equivalent: valid polygonal geometries cannot have zero area.",
        "geometry",
        "x__is_positive_polygonal__mutmut_10",
    ),
    **_equivalent(
        "Equivalent: make_valid preserves the areal set of an already valid polygon.",
        "geometry",
        "x__polygon_parts_from_polygon__mutmut_2",
    ),
    **_equivalent(
        "Equivalent: collapsed polygons repair to non-areal parts before this area guard.",
        "geometry",
        "x__polygon_parts_from_polygon__mutmut_6",
        "x__polygon_parts_from_polygon__mutmut_7",
    ),
    **_equivalent(
        "Equivalent: pyproj resolves lowercase EPSG identifiers to the same CRS.",
        "geometry",
        "x_to_equal_area__mutmut_2",
        "x_to_equal_area__mutmut_4",
    ),
    **_equivalent(
        "Equivalent: every column of one Arrow table has the same length, so a strict zip "
        "never sees unequal inputs.",
        "transform",
        *(f"x__table_results__mutmut_{index}" for index in (6, 9, 10, 11, 13, 14)),
    ),
    **_equivalent(
        "Equivalent: Parquet compression names are case-insensitive and read back as ZSTD.",
        "transform",
        "x__writer__mutmut_8",
        "x_update_label_sidecar__mutmut_30",
    ),
    **_equivalent(
        "Equivalent: None is falsy, so all_touched=None burns like all_touched=False.",
        "grid_overlap",
        "x_weighted_cells__mutmut_24",
    ),
    **_equivalent(
        "Equivalent: a cell that all_touched marks but whose centre is outside the polygon is "
        "crossed by the boundary, so it is a boundary cell and never counts as a full cell. "
        "Checked against 500 random polygons.",
        "grid_overlap",
        "x_weighted_cells__mutmut_29",
    ),
    **_equivalent(
        "Equivalent: np.empty defaults to float64, the dtype of the explicit argument.",
        "grid_overlap",
        "x__empty_cells__mutmut_18",
        "x__empty_cells__mutmut_20",
    ),
    **_equivalent(
        "Equivalent: rasterize fills with 0 by default, any non-zero burn value is True after "
        "the boolean cast, and the output dtype or copy flag does not change the mask.",
        "grid_overlap",
        "x__burn__mutmut_11",
        "x__burn__mutmut_15",
        "x__burn__mutmut_17",
        "x__burn__mutmut_18",
        "x__burn__mutmut_23",
        "x__burn__mutmut_25",
        "x__burn__mutmut_26",
    ),
    **_equivalent(
        "Equivalent: np.zeros defaults to float64, the dtype of the explicit argument.",
        "grid_overlap",
        "x__boundary_areas__mutmut_3",
        "x__boundary_areas__mutmut_5",
        "x__strip_areas__mutmut_3",
        "x__strip_areas__mutmut_5",
        "x__cell_areas__mutmut_11",
        "x__cell_areas__mutmut_13",
    ),
    **_equivalent(
        "Equivalent: asarray keeps a float64 array, so the explicit dtype is redundant.",
        "grid_overlap",
        "x__cell_areas__mutmut_50",
        "x__cell_areas__mutmut_52",
    ),
    **_equivalent(
        "Equivalent: grouping by float block keys only splits blocks further, so every cell "
        "gets the same area.",
        "grid_overlap",
        "x__strip_areas__mutmut_16",
    ),
    **_equivalent(
        'Equivalent: numpy accepts "STABLE" as the stable sort kind.',
        "grid_overlap",
        "x__groups__mutmut_7",
    ),
    **_equivalent(
        "Equivalent: prepare() and the contains() predicate only speed up whole-cell detection; "
        "the clip fallback returns the same area for a cell that the polygon contains.",
        "grid_overlap",
        "x__cell_areas__mutmut_14",
        "x__cell_areas__mutmut_16",
        "x__cell_areas__mutmut_17",
    ),
    **_equivalent(
        "Equivalent: clipping a polygon to a cell it does not intersect yields an empty "
        "geometry with zero area, which is the value already in place.",
        "grid_overlap",
        "x__cell_areas__mutmut_30",
    ),
}
_NAME_PARTS = 3  # osm_polygon_eunis.<module>.<mutant>
_RESULT_LINE = re.compile(r"^\s*([\w.]+):\s*(.+?)\s*$")
_STAT_FIELDS = (
    "killed",
    "survived",
    "no_tests",
    "skipped",
    "suspicious",
    "timeout",
    "segfault",
)
_RUN_MARKER_FIELDS = ("check_was_interrupted_by_user",)
_STATUS_FIELDS = {
    "killed": "killed",
    "survived": "survived",
    "no tests": "no_tests",
    "skipped": "skipped",
    "suspicious": "suspicious",
    "timeout": "timeout",
    "segfault": "segfault",
}


def evaluate(stats: dict[str, Any], results: str) -> list[str]:
    """Return gate failures for exported mutmut statistics and result rows."""

    counts, total, errors = _validated_stats(stats)
    reported, result_errors = _parse_results(results)
    errors.extend(result_errors)
    if counts is None or total is None:
        return errors
    errors.extend(_check_inventory(counts, total, reported))
    errors.extend(_check_quality(counts, reported))
    return errors


def _validated_stats(stats: dict[str, Any]) -> tuple[dict[str, int] | None, int | None, list[str]]:
    errors: list[str] = []
    counts: dict[str, int] = {}
    for field in _STAT_FIELDS:
        value = stats.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"invalid mutation statistic {field!r}: {value!r}")
            continue
        counts[field] = value
    total = stats.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        errors.append(f"invalid mutation statistic 'total': {total!r}")
        return None, None, errors
    if len(counts) != len(_STAT_FIELDS):
        return None, total, errors
    for field in _RUN_MARKER_FIELDS:
        value = stats.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"invalid mutation statistic {field!r}: {value!r}")
        elif value:
            errors.append(f"mutation run reported {field}={value}")
    return counts, total, errors


def _parse_results(results: str) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    reported: dict[str, str] = {}
    for line in results.splitlines():
        match = _RESULT_LINE.fullmatch(line)
        if match is None:
            if line.strip():
                errors.append(f"unrecognized mutation result row: {line.strip()!r}")
            continue
        name, status = match.groups()
        if name in reported:
            errors.append(f"duplicate mutation result row for {name}")
        reported[name] = status
    return reported, errors


def _check_inventory(counts: dict[str, int], total: int, reported: dict[str, str]) -> list[str]:
    errors: list[str] = []
    outcome_total = sum(counts.values())
    if outcome_total != total:
        errors.append(
            f"mutation inventory is inconsistent: outcomes total {outcome_total}, total {total}"
        )
    if len(reported) != total:
        errors.append(
            f"mutation result inventory has {len(reported)} rows, but statistics report {total}"
        )
    for status, field in _STATUS_FIELDS.items():
        observed = sum(row_status == status for row_status in reported.values())
        if observed != counts[field]:
            errors.append(
                f"mutation result inventory reports {observed} {field}, "
                f"but statistics report {counts[field]}"
            )
    return errors


def _check_quality(counts: dict[str, int], reported: dict[str, str]) -> list[str]:
    errors: list[str] = []
    errors.extend(
        f"mutation run reported {counts[field]} {field} mutant(s)"
        for field in ("no_tests", "skipped", "suspicious", "timeout", "segfault")
        if counts[field]
    )
    unexpected_statuses = sorted(set(reported.values()) - set(_STATUS_FIELDS))
    if unexpected_statuses:
        errors.append(f"unexpected mutation result status(es): {', '.join(unexpected_statuses)}")
    errors.extend(_check_module_scores(reported))
    for name, status in reported.items():
        if status == "survived" and name not in ALLOWED_MUTANTS:
            errors.append(f"unapproved survivor: {name}")
    return errors


def module_of(name: str) -> str:
    """Return the module part of ``osm_polygon_eunis.<module>.<mutant>``."""

    parts = name.split(".")
    return parts[1] if len(parts) >= _NAME_PARTS else ""


def module_scores(reported: dict[str, str]) -> dict[str, float]:
    """Return the percentage of killed mutants per module."""

    totals: dict[str, int] = {}
    killed: dict[str, int] = {}
    for name, status in reported.items():
        module = module_of(name)
        totals[module] = totals.get(module, 0) + 1
        killed[module] = killed.get(module, 0) + (status == "killed")
    return {module: 100.0 * killed[module] / totals[module] for module in totals}


def _check_module_scores(reported: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for module, score in sorted(module_scores(reported).items()):
        minimum = MODULE_MINIMUM_SCORES.get(module)
        if minimum is None:
            errors.append(f"no minimum mutation score is configured for module {module!r}")
        elif score < minimum:
            errors.append(
                f"mutation score {score:.2f}% for {module} is below its {minimum:.2f}% baseline"
            )
    return errors


def main() -> int:
    """Validate the exported mutation inventory and configured score baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("mutants/mutmut-cicd-stats.json"))
    parser.add_argument("--results", type=Path, default=Path("mutation-results.txt"))
    args = parser.parse_args()
    try:
        stats = json.loads(args.stats.read_text(encoding="utf-8"))
        results = args.results.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read mutation report: {error}", file=sys.stderr)
        return 2
    if not isinstance(stats, dict):
        print("mutation statistics must be a JSON object", file=sys.stderr)
        return 2

    errors = evaluate(stats, results)
    killed = stats.get("killed", 0)
    total = stats.get("total", 0)
    score = (
        100.0 * killed / total
        if isinstance(killed, int) and isinstance(total, int) and total > 0
        else 0.0
    )
    print(f"mutation score: {score:.2f}% ({stats.get('killed')}/{stats.get('total')})")
    for module, module_score in sorted(module_scores(_parse_results(results)[0]).items()):
        print(f"  {module}: {module_score:.2f}%")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for name, reason in ALLOWED_MUTANTS.items():
        print(f"allowlisted equivalent: {name} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
