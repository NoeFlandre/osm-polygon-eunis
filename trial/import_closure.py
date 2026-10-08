import ast
import pathlib

ROOT = pathlib.Path("<eunis-repo>")
PKG = ROOT / "src" / "osm_polygon_eunis"
known = {p.stem for p in PKG.glob("*.py") if p.stem != "__init__"}

def local_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level >= 1 and node.module:
                found.add(node.module.split(".")[0])
            elif node.level >= 1 and node.module is None:
                for alias in node.names:
                    found.add(alias.name)
            elif node.module and node.module.startswith("osm_polygon_eunis"):
                parts = node.module.split(".")
                if len(parts) >= 2:
                    found.add(parts[1])
                else:
                    for alias in node.names:
                        found.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("osm_polygon_eunis."):
                    found.add(alias.name.split(".")[1])
    return {name for name in found if name in known}

seeds = ["manifest_state", "publish"]
closure: set[str] = set()
queue = list(seeds)
while queue:
    name = queue.pop()
    if name in closure:
        continue
    closure.add(name)
    queue.extend(local_imports(PKG / f"{name}.py") - closure)

# Test imports (test files also load these modules)
for test in ("tests/unit/test_manifest_state.py", "tests/unit/test_publish.py"):
    tree = ast.parse((ROOT / test).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("osm_polygon_eunis"):
            parts = node.module.split(".")
            if len(parts) >= 2 and parts[1] in known:
                closure.add(parts[1])
            for alias in node.names:
                if alias.name in known:
                    closure.add(alias.name)

# Closure over the test-import additions
queue = list(closure)
while queue:
    name = queue.pop()
    for dep in local_imports(PKG / f"{name}.py") - closure:
        closure.add(dep)
        queue.append(dep)

print("CLOSURE:", sorted(closure))
print("MISSING_FROM_ALSO_COPY_AND_SOURCE:")
existing = {"__init__", "_protocols", "domain", "fileio", "publish", "sources", "matching", "geometry", "grid_overlap", "transform", "release_plan"}
for name in sorted(closure - existing - {"manifest_state"}):
    print(f"  src/osm_polygon_eunis/{name}.py")
