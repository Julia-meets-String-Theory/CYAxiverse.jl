"""Build the G2 polytope manifest of the thesis sample from select_polytopes.py output.

Every h11 in [H_MIN, H_MAX] contributes its first TARGET selected polytopes, in the legacy order; the 1-based position in
that order is the unified position, and G2 (run with --seed BASE_SEED) seeds that polytope as BASE_SEED + position.
With REFERENCE given (the committed references/shoroq/data/polytope_manifest_h11_004_050.json), the result must reproduce it exactly
(vertices, order, normal forms, seeds); the historical tranche label is copied from the reference.
Usage: python build_g2_manifest.py SELECTION_JSON OUT_JSON [TARGET] [H_MIN] [H_MAX] [BASE_SEED] [--reference REF_JSON]
"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("selection"); ap.add_argument("out")
ap.add_argument("target", nargs="?", type=int, default=100); ap.add_argument("h_min", nargs="?", type=int, default=4)
ap.add_argument("h_max", nargs="?", type=int, default=50); ap.add_argument("base_seed", nargs="?", type=int, default=20260930)
ap.add_argument("--reference", default=None)
a = ap.parse_args()
sel = json.load(open(a.selection))
ref = json.load(open(a.reference)) if a.reference else None
ref_by = {(p["h11"], p["unified_position"]): p for p in ref["polytopes"]} if ref else {}
polys = []
for h in range(a.h_min, a.h_max + 1):
    d = sel["by_h11"][str(h)]
    if d["selected"] < a.target:
        sys.exit(f"only {d['selected']} distinct favorable polytopes at h11={h}")
    for s in d["selection"][: a.target]:
        r = ref_by.get((h, s["position"]), {})
        polys.append({"h11": h, "unified_position": s["position"], "vertices": s["vertices"], "normal_form_sha256": s["normal_form_sha256"],
                      "md5_order_key": s["md5"], "seed": a.base_seed + s["position"], "historical_tranche": r.get("historical_tranche")})
out = {"source": (ref or {}).get("source", f"KS mirror rev {sel['revision']}; legacy md5 order"), "revision": sel["revision"],
       "per_h11": a.target, "h11_range": [a.h_min, a.h_max], "base_seed": a.base_seed,
       "seed_rule": "G2 seeds the polytope at 1-based manifest position p within its h11 block as base_seed + p",
       "polytopes": polys}
if ref:
    for k in ("historical_tranche_note",):
        out[k] = ref[k]
    same = [(p["h11"], p["unified_position"], p["vertices"], p["normal_form_sha256"], p["seed"]) for p in polys] == \
           [(p["h11"], p["unified_position"], p["vertices"], p["normal_form_sha256"], p["seed"]) for p in ref["polytopes"]]
    print("reproduces reference manifest:", same)
    if not same:
        sys.exit(1)
json.dump(out, open(a.out, "w"), separators=(",", ":"))
print("polytopes:", len(polys))
