"""Fail when module dependencies cross the declared direction."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = "osm_polygon_eunis"
PREFIX = f"{PACKAGE}."
MODULES = {
    path.stem: path
    for path in (Path(__file__).parents[1] / "src" / PACKAGE).glob("*.py")
    if path.stem != "__init__"
}
# Lowest layer first; a module may import only modules listed before it.
LAYERS = (
    "domain",
    "fileio",
    "geometry",
    "matching",
    "reference",
    "eea",
    "transform",
    "cards",
    "sources",
    "publish",
    "runner",
    "cli",
)
FORBIDDEN = {module: set(LAYERS[index + 1 :]) for index, module in enumerate(LAYERS)}


def _imported_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return ()
    if node.level:
        if node.module:
            return (PREFIX + node.module,)
        return tuple(PREFIX + alias.name for alias in node.names)
    return (node.module,) if node.module else ()


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        name[len(PREFIX) :].split(".", 1)[0]
        for node in ast.walk(tree)
        for name in _imported_names(node)
        if name.startswith(PREFIX)
    }


def _cycles(graph: dict[str, set[str]]) -> list[str]:
    errors: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str, trail: tuple[str, ...]) -> None:
        if module in visiting:
            errors.append(f"circular import: {' -> '.join((*trail, module))}")
            return
        if module in visited:
            return
        visiting.add(module)
        for dependency in sorted(graph[module]):
            visit(dependency, (*trail, module))
        visiting.remove(module)
        visited.add(module)

    for module in sorted(graph):
        visit(module, ())
    return errors


def main() -> int:
    errors = [
        f"{module} is not declared in LAYERS"
        for module in sorted(MODULES.keys() - FORBIDDEN.keys())
    ]
    graph: dict[str, set[str]] = {}
    for module, path in MODULES.items():
        dependencies = _imports(path) & MODULES.keys()
        graph[module] = dependencies
        for dependency in sorted(dependencies & FORBIDDEN.get(module, set())):
            errors.append(f"{module} imports forbidden higher-level module {dependency}")
    errors.extend(_cycles(graph))
    if errors:
        print("\n".join(sorted(set(errors))), file=sys.stderr)
        return 1
    print("architecture: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
