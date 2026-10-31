"""Population completeness of a stage 2-3 run against the frozen dataset, in both directions.

compare_geometry_subset.py and compare_vacuum_repair.py check every record the new run produced against the frozen one.
This script checks the converse: every frozen record has a new counterpart, the populations are the same sets, and the
population totals agree.
  * G2 status of every attempted polytope equals the frozen status (accepted / rejected_*), no generation errors
  * theta = 0 inertia-scan records, tachyonic list, steps 01/02/03/04/06 and the 05 classification: same key sets;
    02 (vacua CLI groups) and 06 (end-to-end check) records equal the frozen ones apart from run times and paths
  * relic-input manifest: same geometry set, same status counts and mode totals
  * QCD-edge flags: same geometries and every numerical field identical; same counts
With --full the absolute population numbers of the frozen dataset are also required (4700 polytopes, 4491 accepted,
209 rejected, 46 tachyonic, 16,250 theta = 0 modes >= H0, 16,313 final modes, 4445 + 46 statuses, 539 QCD-edge flags).
Without --full the frozen populations are restricted to the attempted (h11, position) pairs (subset runs).
Usage: python check_full_population.py WORK FROZEN_ROOT OUT_JSON [--full]
"""
import csv, json, os, sys
from collections import Counter

work, fr, out = sys.argv[1:4]
FULL = "--full" in sys.argv[4:]
V = os.path.join(work, "vacuum_repair")
FU = os.path.join(fr, "thesis_g2_adaptive", "unified_h11_004_050", "manifests")
FRES = os.path.join(fr, "tachyonic_vacua_audit", "results")
SKIP = {"seconds", "file", "output"}     # run times and file locations only
res, ok = {}, True


def check(name, cond, **info):
    global ok
    res[name] = dict(info, pass_=bool(cond))
    ok = ok and bool(cond)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def keyed(rows):
    return {(int(r["h11"]), int(r["position"])): r for r in rows}


def gkey(g):
    """(h11, position) of a geometry id h11_XXX/np_POSITION/cy_0000001"""
    return tuple(int(x.split("_")[1]) for x in g.split("/")[:2])


# G2 statuses
new_status = {(int(r["h11"]), int(r["polytope_index"])): r["status"]
              for r in csv.DictReader(open(os.path.join(work, "validation", "status", "polytope_status.csv")))}
attempted = set(new_status)
inside = lambda k: FULL or k in attempted
old_status = {(p["h11"], p["position"]): p["status"] for p in json.load(open(os.path.join(FU, "unified_polytopes.json")))["polytopes"]
              if inside((p["h11"], p["position"]))}
diff = sorted(k for k in set(new_status) | set(old_status) if new_status.get(k) != old_status.get(k))
check("g2_status_identical", not diff, attempted=len(new_status), frozen=len(old_status), differences=[str(k) for k in diff[:20]],
      new_counts=dict(Counter(new_status.values())), frozen_counts=dict(Counter(old_status.values())))
check("no_generation_errors", "generation_error" not in new_status.values())
accepted = {k for k, s in old_status.items() if s == "accepted"}

# stage 3 key sets
scan_new = keyed(jl(os.path.join(V, "00_theta0_inertia_scan.jsonl")))
scan_old = {k: r for k, r in keyed(jl(os.path.join(fr, "spectrum_count_audit", "results", "accounting_all.jsonl"))).items() if inside(k)}
check("inertia_scan_same_population", set(scan_new) == set(scan_old) == accepted, new=len(scan_new), frozen=len(scan_old), accepted=len(accepted))
tl = lambda p: {tuple(map(int, l.split("\t"))) for l in open(p).read().splitlines()[1:] if l.strip()}
t_new, t_old = tl(os.path.join(V, "tachyonic_list.tsv")), {k for k in tl(os.path.join(FRES, "tachyonic_list.tsv")) if inside(k)}
check("tachyonic_list_identical", t_new == t_old, new=len(t_new), frozen=len(t_old))
for step in ("01_theta0_structure", "02_vacua_auto_groups", "03_minima_validation", "04_vacuum_spectra", "06_verify_corrected"):
    a = keyed(jl(os.path.join(V, f"{step}.jsonl"))) if step != "06_verify_corrected" else \
        {gkey(r["geometry_id"]): r for r in jl(os.path.join(V, f"{step}.jsonl"))}
    b = keyed(jl(os.path.join(FRES, f"{step}.jsonl"))) if step != "06_verify_corrected" else \
        {gkey(r["geometry_id"]): r for r in jl(os.path.join(FRES, f"{step}.jsonl"))}
    b = {k: r for k, r in b.items() if inside(k)}
    info = {"new": len(a), "frozen": len(b)}
    same_set = set(a) == set(b) == t_old
    if step in ("02_vacua_auto_groups", "06_verify_corrected"):    # records not compared field by field elsewhere
        d = {str(k): sorted(f for f in set(a[k]) | set(b[k]) if f not in SKIP and a[k].get(f) != b[k].get(f)) for k in set(a) & set(b)}
        d = {k: v for k, v in d.items() if v}
        info["field_differences"] = d
        same_set = same_set and not d
    check(f"{step}_same_population", same_set, **info)
cls_new = {r["geometry_id"] for r in json.load(open(os.path.join(V, "05_vacuum_classification.json")))}
gid = lambda k: f"h11_{k[0]:03d}/np_{k[1]:07d}/cy_0000001"
cls_old = {r["geometry_id"] for r in json.load(open(os.path.join(FRES, "05_vacuum_classification.json")))
           if inside(gkey(r["geometry_id"]))}
check("05_classification_same_population", cls_new == cls_old == {gid(k) for k in t_old}, new=len(cls_new), frozen=len(cls_old))

# relic-input manifest
M_new = json.load(open(os.path.join(work, "relic_input", "relic_abundance_input_manifest.json")))["geometries"]
M_old = [r for r in json.load(open(os.path.join(fr, "relic_abundance_input", "relic_abundance_input_manifest.json")))["geometries"]
         if inside((r["h11"], r["position"]))]
tot = lambda M: {"geometries": len(M), "status": dict(sorted(Counter(r["status"] for r in M).items())),
                 "n_modes_above_H0": sum(r["n_modes_above_H0"] for r in M),
                 "theta0_stored_modes_above_H0": sum(r["theta0_stored_modes_above_H0"] for r in M)}
check("manifest_same_population_and_totals", {r["geometry_id"] for r in M_new} == {r["geometry_id"] for r in M_old} and tot(M_new) == tot(M_old),
      new=tot(M_new), frozen=tot(M_old))

# QCD-edge flags
FIELDS = ("qcd_volume_kappa_reconstruction", "deficit_below_25", "beyond_gate_tolerance_1e-5_rel", "point_kind", "tau_max_over_min")
# qcd_volume_exact_rational: the frozen file holds an exact-rational cross-check for the beyond-tolerance flags only;
# flag_qcd_edge.py does not recompute it (documented, written as null). Required null here; frozen values listed as not reproduced
EXACT = "qcd_volume_exact_rational"
Q_new = json.load(open(os.path.join(work, "validation", "qcd_edge_flags.json")))
Q_old = json.load(open(os.path.join(FU, "qcd_edge_flags.json")))
qa = {(f["h11"], f["unified_position"]): f for f in Q_new["flags"]}
qb = {(f["h11"], f["unified_position"]): f for f in Q_old["flags"] if inside((f["h11"], f["unified_position"]))}
qd = {str(k): [f for f in FIELDS if qa[k].get(f) != qb[k].get(f)] for k in set(qa) & set(qb)}
qd = {k: v for k, v in qd.items() if v}
not_reproduced = {str(k): qb[k][EXACT] for k in sorted(qb) if qb[k].get(EXACT) is not None}
check("qcd_edge_flags_identical", set(qa) == set(qb) and not qd and all(f.get(EXACT) is None for f in qa.values()),
      new=len(qa), frozen=len(qb), compared_fields=FIELDS, field_differences=qd,
      frozen_exact_rational_not_recomputed=not_reproduced,
      new_beyond_tolerance=sum(f["beyond_gate_tolerance_1e-5_rel"] for f in qa.values()),
      frozen_beyond_tolerance=sum(f["beyond_gate_tolerance_1e-5_rel"] for f in qb.values()))

if FULL:
    t = tot(M_new)
    check("frozen_population_numbers", len(new_status) == 4700 and len(accepted) == 4491 and len(new_status) - len(accepted) == 209
          and len(t_new) == 46 and t["theta0_stored_modes_above_H0"] == 16250 and t["n_modes_above_H0"] == 16313
          and t["status"] == {"ORIGINAL_SPECTRUM": 4445, "VACUUM_CORRECTED": 46} and Q_new["count"] == 539,
          polytopes=len(new_status), accepted=len(accepted), tachyonic=len(t_new), totals=t, qcd_flags=Q_new["count"])

res = {"full_population": FULL, "checks": {k: {kk.rstrip("_"): vv for kk, vv in v.items()} for k, v in res.items()}, "pass": ok}
json.dump(res, open(out, "w"), indent=1, default=str)
print(json.dumps({k: v["pass"] for k, v in res["checks"].items()}, indent=1))
print("POPULATION", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
