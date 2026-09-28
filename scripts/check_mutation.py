"""Fail CI when mutation results fall below the measured quality baseline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

MINIMUM_SCORE = 91.9
ALLOWED_MUTANTS = {
    "osm_polygon_eunis.matching.x__outranks__mutmut_5": (
        "Equivalent: the caller checks unequal percentages before comparing with >."
    ),
    "osm_polygon_eunis.geometry.x__is_positive_polygonal__mutmut_10": (
        "Equivalent: valid polygonal geometries cannot have zero area."
    ),
    "osm_polygon_eunis.geometry.x__polygon_parts_from_polygon__mutmut_2": (
        "Equivalent: make_valid preserves the areal set of an already valid polygon."
    ),
    "osm_polygon_eunis.geometry.x__polygon_parts_from_polygon__mutmut_6": (
        "Equivalent: collapsed polygons repair to non-areal parts before this area guard."
    ),
    "osm_polygon_eunis.geometry.x__polygon_parts_from_polygon__mutmut_7": (
        "Equivalent: collapsed polygons repair to non-areal parts before this area guard."
    ),
    "osm_polygon_eunis.geometry.x_to_equal_area__mutmut_2": (
        "Equivalent: pyproj resolves lowercase EPSG:4326 to the same CRS."
    ),
    "osm_polygon_eunis.geometry.x_to_equal_area__mutmut_4": (
        "Equivalent: pyproj resolves lowercase EPSG:3035 to the same CRS."
    ),
}
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
    errors.extend(_check_quality(counts, total, reported))
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


def _check_quality(counts: dict[str, int], total: int, reported: dict[str, str]) -> list[str]:
    errors: list[str] = []
    errors.extend(
        f"mutation run reported {counts[field]} {field} mutant(s)"
        for field in ("no_tests", "skipped", "suspicious", "timeout", "segfault")
        if counts[field]
    )
    unexpected_statuses = sorted(set(reported.values()) - set(_STATUS_FIELDS))
    if unexpected_statuses:
        errors.append(f"unexpected mutation result status(es): {', '.join(unexpected_statuses)}")
    score = 100.0 * counts["killed"] / total
    if score < MINIMUM_SCORE:
        errors.append(f"mutation score {score:.2f}% is below the {MINIMUM_SCORE:.1f}% baseline")
    for name, status in reported.items():
        if status == "survived" and name not in ALLOWED_MUTANTS:
            errors.append(f"unapproved survivor: {name}")
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
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for name, reason in ALLOWED_MUTANTS.items():
        print(f"allowlisted equivalent: {name} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
