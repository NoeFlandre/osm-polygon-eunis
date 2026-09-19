"""Fail when module dependencies cross the declared direction."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = "osm_polygon_eunis"
MODULES = {
    path.stem: path
    for path in (Path(__file__).parents[1] / "src" / PACKAGE).glob("*.py")
    if path.stem != "__init__"
}
FORBIDDEN = {
    "domain": {
        "matching",
        "geometry",
        "reference",
        "transform",
        "sources",
        "publish",
        "runner",
        "cli",
    },
    "matching": {"reference", "transform", "sources", "publish", "runner", "cli"},
    "geometry": {"reference", "transform", "sources", "publish", "runner", "cli"},
    "reference": {"transform", "sources", "publish", "runner", "cli"},
    "transform": {"sources", "publish", "runner", "cli"},
    "sources": {"publish", "runner", "cli"},
    "publish": {"runner", "cli"},
    "eea": {"sources", "publish", "runner", "cli"},
    "runner": {"cli"},
    "cli": set(),
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = (node.module,)
        else:
            continue
        for name in names:
            prefix = f"{PACKAGE}."
            if name.startswith(prefix):
                result.add(name[len(prefix) :].split(".", 1)[0])
    return result


def main() -> int:
    errors: list[str] = []
    graph: dict[str, set[str]] = {}
    for module, path in MODULES.items():
        dependencies = _imports(path) & MODULES.keys()
        graph[module] = dependencies
        for dependency in sorted(dependencies & FORBIDDEN.get(module, set())):
            errors.append(f"{module} imports forbidden higher-level module {dependency}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str, trail: tuple[str, ...]) -> None:
        if module in visiting:
            errors.append(f"circular import: {' -> '.join((*trail, module))}")
            return
        if module in visited:
            return
        visiting.add(module)
        for dependency in graph[module]:
            visit(dependency, (*trail, module))
        visiting.remove(module)
        visited.add(module)

    for module in MODULES:
        visit(module, ())
    if errors:
        print("\n".join(sorted(set(errors))), file=sys.stderr)
        return 1
    print("architecture: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
