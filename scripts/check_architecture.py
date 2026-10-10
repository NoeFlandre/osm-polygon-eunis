"""Fail when modules cross the declared layers or use private names across modules."""

from __future__ import annotations

import ast
import functools
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
    "_protocols",
    "domain",
    "options",
    "fileio",
    "geometry",
    "matching",
    "raster_geometry",
    "geopackage_sql",
    "grid_overlap",
    "geopackage_tiles",
    "geopackage_reference",
    "raster_reference",
    "eea",
    "reference_cache",
    "transform",
    "reference_staging",
    "cards",
    "sources",
    "publish",
    "release_plan",
    "geometry_checkpoints",
    "geometry_chunks",
    "geometry_workers",
    "geometry_jobs",
    "manifest_state",
    "shard_processing",
    "card_publishing",
    "release_orchestration",
    "grid5000",
    "run_analysis",
    "cli",
)
FORBIDDEN = {module: set(LAYERS[index + 1 :]) for index, module in enumerate(LAYERS)}


@functools.cache
def _nodes(path: Path) -> tuple[ast.AST, ...]:
    # Every check reads the same nodes, so parse and walk each module once per run.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(ast.walk(tree))


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
    return {
        name[len(PREFIX) :].split(".", 1)[0]
        for node in _nodes(path)
        for name in _imported_names(node)
        if name.startswith(PREFIX)
    }


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _is_package_import(node: ast.ImportFrom) -> bool:
    if node.level:
        return True
    return node.module is not None and (node.module == PACKAGE or node.module.startswith(PREFIX))


def _private_imports(path: Path) -> set[tuple[str, str]]:
    """Return (source module, name) pairs for underscore names imported from the package."""
    return {
        ("." * node.level + (node.module or ""), alias.name)
        for node in _nodes(path)
        if isinstance(node, ast.ImportFrom) and _is_package_import(node)
        for alias in node.names
        if _is_private(alias.name)
    }


def _is_sibling_module_import(node: ast.ImportFrom) -> bool:
    # Covers `from . import x` and `from osm_polygon_eunis import x`.
    if node.level:
        return node.module is None
    return node.module == PACKAGE


def _package_module_aliases(nodes: tuple[ast.AST, ...]) -> dict[str, str]:
    """Map local names bound to sibling package modules onto their module names."""
    return {
        alias.asname or alias.name: alias.name
        for node in nodes
        if isinstance(node, ast.ImportFrom) and _is_sibling_module_import(node)
        for alias in node.names
        if alias.name in MODULES
    }


def _private_attributes(path: Path) -> set[tuple[str, str]]:
    """Return (module name, attribute) pairs for underscore attributes read from a sibling."""
    nodes = _nodes(path)
    aliases = _package_module_aliases(nodes)
    return {
        (aliases[node.value.id], node.attr)
        for node in nodes
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
        and _is_private(node.attr)
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
    """Return a failure status when package imports break the layer graph or use private names."""
    errors = [
        f"{module} is not declared in LAYERS"
        for module in sorted(MODULES.keys() - FORBIDDEN.keys())
    ]
    graph: dict[str, set[str]] = {}
    for module, path in MODULES.items():
        dependencies = _imports(path) & MODULES.keys()
        graph[module] = dependencies
        errors.extend(
            f"{module} imports forbidden higher-level module {dependency}"
            for dependency in sorted(dependencies & FORBIDDEN.get(module, set()))
        )
        errors.extend(
            f"{module} imports private name {name} from {source}"
            for source, name in sorted(_private_imports(path))
        )
        errors.extend(
            f"{module} uses private name {attribute} of {sibling}"
            for sibling, attribute in sorted(_private_attributes(path))
        )
    errors.extend(_cycles(graph))
    if errors:
        print("\n".join(sorted(set(errors))), file=sys.stderr)
        return 1
    print("architecture: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
