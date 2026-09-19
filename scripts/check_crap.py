"""Report cyclomatic complexity and conservative CRAP scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from radon.complexity import cc_visit
from radon.visitors import Class


def main() -> int:
    root = Path(__file__).parents[1]
    coverage_path = root / "coverage.json"
    if not coverage_path.is_file():
        print("run pytest with coverage before check_crap.py", file=sys.stderr)
        return 1
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")).get("files", {})
    failures: list[str] = []
    rows: list[tuple[float, str]] = []
    for path in sorted((root / "src" / "osm_polygon_eunis").glob("*.py")):
        relative = str(path.relative_to(root))
        file_percent = float(
            coverage.get(relative, {}).get("summary", {}).get("percent_covered", 0)
        )
        coverage_fraction = file_percent / 100.0
        for block in cc_visit(path.read_text(encoding="utf-8")):
            if isinstance(block, Class):
                continue
            crap = block.complexity**2 * (1.0 - coverage_fraction) ** 3 + block.complexity
            rows.append((crap, f"{relative}:{block.lineno} {block.name}"))
            if crap >= 6.0:
                failures.append(f"CRAP {crap:.2f} >= 6: {relative}:{block.lineno} {block.name}")
    for crap, name in sorted(rows, reverse=True):
        print(f"{crap:.2f} {name}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
