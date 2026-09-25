"""Report cyclomatic complexity and per-function CRAP scores."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from radon.complexity import cc_visit
from radon.visitors import Class

CRAP_THRESHOLD = 6.0
ROOT = Path(__file__).parents[1]


def _load_coverage(root: Path) -> Mapping[str, Any] | None:
    coverage_path = root / "coverage.json"
    if not coverage_path.is_file():
        return None
    return json.loads(coverage_path.read_text(encoding="utf-8")).get("files", {})


def _coverage_fraction(file_coverage: Mapping[str, Any], start: int, end: int) -> float:
    def in_span(key: str) -> set[int]:
        return {line for line in file_coverage.get(key, ()) if start <= line <= end}

    executed = in_span("executed_lines")
    missing = in_span("missing_lines")
    statements = len(executed) + len(missing)
    return len(executed) / statements if statements else 0.0


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


def main() -> int:
    coverage = _load_coverage(ROOT)
    if coverage is None:
        print("run pytest with coverage before check_crap.py", file=sys.stderr)
        return 1
    return _report(_block_scores(ROOT, coverage))


if __name__ == "__main__":
    raise SystemExit(main())
