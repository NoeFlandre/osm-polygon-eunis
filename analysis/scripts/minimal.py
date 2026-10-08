import sys
sys.path.insert(0, "<scratchpad>/scratchpad/eunis108-analysis")
from classify import ROOT, PKG, scan, known

def closure(seed_mods, test_files):
    seen = {}
    q = []
    for m in seed_mods:
        seen[m] = "seed"; q.append(m)
    for t in test_files:
        top, _ = scan(ROOT / t)
        for m in top:
            if m not in seen:
                seen[m] = f"import in {t}"; q.append(m)
    while q:
        n = q.pop()
        top, _ = scan(PKG / f"{n}.py")
        for m in top:
            if m not in seen:
                seen[m] = f"from {n}"; q.append(m)
    return seen

trial = {"__init__","_protocols","card_publishing","cards","domain","eea","fileio","geometry_checkpoints","geometry_chunks","geometry_jobs","geometry_workers","geopackage_reference","geopackage_sql","geopackage_tiles","options","publish","raster_geometry","raster_reference","reference_cache","reference_staging","release_orchestration","shard_processing","sources","matching","geometry","grid_overlap","transform","release_plan","manifest_state"}
full = closure(["manifest_state","publish"], ["tests/unit/test_manifest_state.py","tests/unit/test_publish.py"])
print("closure size (no __init__):", len(full), "| not in trial:", sorted(set(full)-trial))
print("in trial but not in closure:", sorted(trial-set(full)-{"__init__"}))
# minimal: drop the release_orchestration import from test_manifest_state (hypothetical)
mini = closure(["manifest_state","publish"], ["tests/unit/test_publish.py"])
print("closure without test_manifest_state imports:", sorted(mini))
print("count:", len(mini))
print("only-from-release_orchestration chain:", sorted(set(full)-set(mini)))
