"""Population test of producers/geometry/flag_qcd_edge.py on the frozen audits of all 4491 accepted geometries.

The frozen audits use tranche-local file names; they are remapped to unified positions (unified_polytopes.json), passed
through flag_qcd_edge.py unchanged, and the result must equal the frozen qcd_edge_flags.json (the list of geometries,
the beyond-tolerance subset and every deficit).
Usage: python check_qcd_flags_frozen.py FROZEN_G2_ROOT OUT_DIR
       FROZEN_G2_ROOT = thesis_g2_adaptive (tranche 1 at the root, tranche 2 in tranche2/, unified view in unified_h11_004_050/)
"""
import glob, json, os, subprocess, sys

root, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
U = json.load(open(os.path.join(root, "unified_h11_004_050", "manifests", "unified_polytopes.json")))["polytopes"]
pos = {(p["tranche"], p["h11"], p["local_index"]): p["position"] for p in U if p["status"] == "accepted"}
remapped, n = os.path.join(out, "frozen_audits_unified_positions.jsonl"), 0
with open(remapped, "w") as fo:
    for t, d in ((1, root), (2, os.path.join(root, "tranche2"))):
        for p in sorted(glob.glob(os.path.join(d, "validation", "materialization", "audit_with_materialized_h11_*.jsonl"))):
            for line in open(p):
                r = json.loads(line); h = int(r["h11"]); li = int(r["file"].split("/")[1][3:])
                r["file"] = f"h11_{h:03d}/np_{pos[(t, h, li)]:07d}/" + "/".join(r["file"].split("/")[2:])
                fo.write(json.dumps(r) + "\n"); n += 1
new = os.path.join(out, "qcd_edge_flags_recomputed.json")
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "..", "..", "producers", "geometry", "flag_qcd_edge.py"), remapped, new],
               check=True, stdout=subprocess.DEVNULL)
A = json.load(open(new)); F = json.load(open(os.path.join(root, "unified_h11_004_050", "manifests", "qcd_edge_flags.json")))
fa = {(f["h11"], f["unified_position"]): f for f in A["flags"]}; ff = {(f["h11"], f["unified_position"]): f for f in F["flags"]}
res = {"audited_geometries": n, "flags_recomputed": A["count"], "flags_frozen": F["count"],
       "same_geometries": set(fa) == set(ff),
       "same_beyond_tolerance": {k for k, f in fa.items() if f["beyond_gate_tolerance_1e-5_rel"]} == {k for k, f in ff.items() if f["beyond_gate_tolerance_1e-5_rel"]},
       "max_abs_deficit_difference": max((abs(fa[k]["deficit_below_25"] - ff[k]["deficit_below_25"]) for k in ff if k in fa), default=None)}
res["pass"] = n == 4491 and res["same_geometries"] and res["same_beyond_tolerance"] and res["max_abs_deficit_difference"] == 0.0
json.dump(res, open(os.path.join(out, "qcd_flags_check.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if res["pass"] else 1)
