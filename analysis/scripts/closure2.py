import sys, pathlib
sys.path.insert(0, "<scratchpad>/scratchpad/eunis108-analysis")
from classify import ROOT, PKG, scan, known

seeds = {
    "manifest_state": PKG / "manifest_state.py",
    "publish": PKG / "publish.py",
}
test_files = {
    "tests/unit/test_manifest_state.py": ROOT / "tests/unit/test_manifest_state.py",
    "tests/unit/test_publish.py": ROOT / "tests/unit/test_publish.py",
}
# Module-level closure, with the importer recorded
mod_closure = {}
fn_only = {}
queue = []
for name in ("manifest_state", "publish"):
    mod_closure[name] = "SEED"
    queue.append(name)
for label, p in test_files.items():
    top, fo = scan(p)
    for t in top:
        if t not in mod_closure:
            mod_closure[t] = f"module-level import in {label}"
            queue.append(t)
    for t in fo:
        fn_only.setdefault(t, set()).add(label)
while queue:
    name = queue.pop()
    top, fo = scan(PKG / f"{name}.py")
    for t in top:
        if t not in mod_closure:
            mod_closure[t] = f"module-level from {name}.py"
            queue.append(t)
    for t in fo:
        fn_only.setdefault(t, set()).add(f"{name}.py")
print("MODULE-LEVEL CLOSURE (", len(mod_closure), "):")
for k in sorted(mod_closure):
    print(f"  {k}: {mod_closure[k]}")
print("FUNCTION-LEVEL-ONLY refs (not in module-level closure):")
for k in sorted(fn_only):
    if k not in mod_closure:
        print(f"  {k}: from {sorted(fn_only[k])}")
print("ALL KNOWN NOT IN CLOSURE:", sorted(known - set(mod_closure)))
