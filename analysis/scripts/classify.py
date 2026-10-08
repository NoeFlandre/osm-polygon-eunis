import ast
import pathlib
import sys

ROOT = pathlib.Path("<eunis-repo>")
PKG = ROOT / "src" / "osm_polygon_eunis"
known = {p.stem for p in PKG.glob("*.py") if p.stem != "__init__"}


def targets(node, prefix_pkg=True):
    """Return set of local module stems referenced by an import node."""
    out = set()
    if isinstance(node, ast.ImportFrom):
        if node.level >= 1:
            if node.module:
                out.add(node.module.split(".")[0])
            else:
                for a in node.names:
                    out.add(a.name)
        elif node.module and node.module.startswith("osm_polygon_eunis"):
            parts = node.module.split(".")
            if len(parts) >= 2:
                out.add(parts[1])
            else:
                for a in node.names:
                    out.add(a.name)
    elif isinstance(node, ast.Import):
        for a in node.names:
            if a.name.startswith("osm_polygon_eunis."):
                out.add(a.name.split(".")[1])
    return {x for x in out if x in known}


def scan(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top, nested = set(), set()
    # module-level: statements directly in tree.body, or within top-level if/try blocks (not functions/classes)
    def walk_top(stmts):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # class bodies execute at import; include class-level imports (rare)
                if isinstance(s, ast.ClassDef):
                    walk_top(s.body)
                continue
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                top.update(targets(s))
            elif isinstance(s, (ast.If, ast.Try, ast.With)):
                for field in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(s, field, [])
                    for h in sub:
                        if isinstance(h, ast.ExceptHandler):
                            walk_top(h.body)
                        else:
                            walk_top([h])
    walk_top(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            nested.update(targets(node))
    return top, nested - top


if __name__ == "__main__":
    for rel in sys.argv[1:]:
        p = ROOT / rel
        top, fn_only = scan(p)
        print(f"== {rel}")
        print("  module-level:", sorted(top))
        print("  function-level only:", sorted(fn_only))
