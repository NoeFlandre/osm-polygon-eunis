import re, sys, difflib, pathlib
base = pathlib.Path("<eunis-repo>/mutants/src/osm_polygon_eunis")
mod, funcs = sys.argv[1], sys.argv[2:]
text = (base / f"{mod}.py").read_text(encoding="utf-8").splitlines()
# collect top-level function blocks: start at "def name(" at column 0, end before next column-0 non-blank statement
blocks = {}
starts = [i for i, l in enumerate(text) if re.match(r"^(def |class |@)", l)]
starts.append(len(text))
for a, b in zip(starts, starts[1:]):
    m = re.match(r"^def (\w+)\(", text[a])
    if m:
        blocks[m.group(1)] = [l.rstrip() for l in text[a:b] if l.strip() != ""]
for f in funcs:
    orig = blocks.get(f"x_{f}__mutmut_orig") or blocks.get(f"x__{f}__mutmut_orig")
    if orig is None:
        print("NO ORIG", f); continue
    names = sorted([k for k in blocks if re.match(rf"x_{{1,2}}{f}__mutmut_\d+$", k)], key=lambda k: int(k.rsplit("_", 1)[1]))
    print(f"### {f}: {len(names)} variants")
    for k in names:
        d = [l for l in difflib.unified_diff(orig, blocks[k], lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        print(f"{k.rsplit('__mutmut_',1)[1]}: " + " | ".join(x.strip()[:110] for x in d))
