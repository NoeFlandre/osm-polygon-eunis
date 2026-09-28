"""Report per-function CRAP scores using line and branch coverage.

The threshold of 6.0 caps a fully covered function at cyclomatic complexity 5;
coverage lowers scores for more complex functions but does not waive missing
module measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radon.complexity import cc_visit
from radon.visitors import Class

CRAP_THRESHOLD = 6.0
ROOT = Path(__file__).parents[1]


def _load_coverage(coverage_path: Path) -> Mapping[str, Any] | None:
    if not coverage_path.is_file():
        return None
    return json.loads(coverage_path.read_text(encoding="utf-8")).get("files", {})


def _coverage_fraction(file_coverage: Mapping[str, Any], start: int, end: int) -> float:
    def in_span(key: str) -> set[int]:
        return {line for line in file_coverage.get(key, ()) if start <= line <= end}

    def branches_in_span(key: str) -> set[tuple[int, int]]:
        return {
            (branch[0], branch[1])
            for branch in file_coverage.get(key, ())
            if start <= branch[0] <= end
        }

    executed = in_span("executed_lines")
    missing = in_span("missing_lines")
    executed_branches = branches_in_span("executed_branches")
    missing_branches = branches_in_span("missing_branches")
    covered = len(executed) + len(executed_branches)
    measured = covered + len(missing) + len(missing_branches)
    return covered / measured if measured else 0.0


def _missing_coverage_files(root: Path, coverage: Mapping[str, Any]) -> list[str]:
    source_root = root / "src" / "osm_polygon_eunis"
    return [
        str(path.relative_to(root))
        for path in sorted(source_root.glob("*.py"))
        if str(path.relative_to(root)) not in coverage
    ]


def _block_scores(root: Path, coverage: Mapping[str, Any]) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for path in sorted((root / "src" / "osm_polygon_eunis").glob("*.py")):
        relative = str(path.relative_to(root))
        file_coverage = coverage.get(relative, {})
        for block in cc_visit(path.read_text(encoding="utf-8")):
            # radon also yields each method as its own block, so skipping the
            # enclosing class does not hide any function from the gate.
            if isinstance(block, Class):
                continue
            fraction = _coverage_fraction(file_coverage, block.lineno, block.endline)
            crap = block.complexity**2 * (1.0 - fraction) ** 3 + block.complexity
            rows.append((crap, f"{relative}:{block.lineno} {block.name}"))
    return rows


def _report(rows: list[tuple[float, str]]) -> int:
    for crap, name in sorted(rows, reverse=True):
        print(f"{crap:.2f} {name}")
    failures = [
        f"CRAP {crap:.2f} >= {CRAP_THRESHOLD:g}: {name}"
        for crap, name in rows
        if crap >= CRAP_THRESHOLD
    ]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Print CRAP scores and fail if a function exceeds the configured limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-json",
        type=Path,
        help="coverage JSON path (defaults to coverage.json in the repository root)",
    )
    args = parser.parse_args(argv)
    coverage_path = args.coverage_json or ROOT / "coverage.json"
    coverage = _load_coverage(coverage_path)
    if coverage is None:
        print("run pytest with coverage before check_crap.py", file=sys.stderr)
        return 1
    missing = _missing_coverage_files(ROOT, coverage)
    if missing:
        print(
            f"coverage is missing source module entries: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1
    return _report(_block_scores(ROOT, coverage))


if __name__ == "__main__":
    raise SystemExit(main())
