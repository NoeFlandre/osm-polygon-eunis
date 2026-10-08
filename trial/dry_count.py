"""Generate mutants only (no stats run, no tests) and list mutant names per module.

Run from the repo root so mutmut's relative `mutants/` paths resolve there.
"""

import collections
import os
import sys

import mutmut.__main__ as mm


def main() -> int:
    os.makedirs("mutants", exist_ok=True)
    mm.copy_src_dir()
    mm.copy_also_copy_files()
    mm.setup_source_paths()
    stats = mm.create_mutants(2)
    print(f"GENERATED mutated={stats.mutated} ignored={stats.ignored} unmodified={stats.unmodified}")

    per_module: collections.Counter[str] = collections.Counter()
    names_by_module: dict[str, list[str]] = collections.defaultdict(list)
    for path in mm.walk_mutatable_files():
        data = mm.SourceFileMutationData(path=path)
        data.load()
        for key in data.exit_code_by_key:
            module = key.split(".")[1] if "." in key else "?"
            per_module[module] += 1
            names_by_module[module].append(key)

    for module in sorted(per_module):
        print(f"MODULE {module}: {per_module[module]} mutants")
    out_dir = "<scratchpad>/scratchpad/eunis108"
    for module in ("manifest_state", "publish"):
        with open(f"{out_dir}/names-{module}.txt", "w", encoding="utf-8") as handle:
            handle.write("\n".join(sorted(names_by_module.get(module, []))) + "\n")
    print("SAMPLE", (names_by_module.get("manifest_state") or ["none"])[:3])
    print("SAMPLE", (names_by_module.get("publish") or ["none"])[:3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
