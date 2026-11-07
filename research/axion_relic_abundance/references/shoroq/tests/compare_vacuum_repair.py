"""Compare a (subset) stage-3 run of this repository with the frozen vacuum-repair evidence.

  * theta = 0 inertia scan records (00) equal the frozen mode accounting, apart from run times
  * saddle structure (01) and full-potential validation (03) records equal the frozen ones, apart from run times
  * corrected spectra: spectrum/physical/* (excluding run-time metadata) and vacuum_correction/theta_min identical
  * vacuum classification (05) and relic-input manifest records equal the frozen ones in every scientific field
    (status, mode counts, validation status, QCD-edge and sub-H0 tachyon flags, tranche, normal form); only paths and
    file checksums (which include run-time metadata) may differ
Usage: python compare_vacuum_repair.py WORK FROZEN_ACCOUNTING_JSONL FROZEN_REPAIR_RESULTS_DIR FROZEN_CORRECTED_SPECTRA_DIR
                                       FROZEN_RELIC_INPUT_MANIFEST OUT_JSON
"""
import json, os, sys

import h5py
import numpy as np

work, facc, fres, fcorr, fman, out = sys.argv[1:7]
V = os.path.join(work, "vacuum_repair")
SKIP = {"seconds", "file", "output"}
key = lambda r: (r["h11"], r["position"])


def load(p):
    return {key(r): r for r in map(json.loads, open(p))}


def cmp_records(new, old):
    bad = {}
    for k, r in new.items():
        o = old.get(k)
        if o is None:
            bad[str(k)] = "missing in frozen"; continue
        d = [f for f in set(r) | set(o) if f not in SKIP and r.get(f) != o.get(f)]
        if d:
            bad[str(k)] = sorted(d)
    return {"records": len(new), "identical": not bad, "differences": bad}


res = {"00_inertia_scan": cmp_records(load(os.path.join(V, "00_theta0_inertia_scan.jsonl")), load(facc)),
       "01_theta0_structure": cmp_records(load(os.path.join(V, "01_theta0_structure.jsonl")), load(os.path.join(fres, "01_theta0_structure.jsonl"))),
       "03_minima_validation": cmp_records(load(os.path.join(V, "03_minima_validation.jsonl")), load(os.path.join(fres, "03_minima_validation.jsonl")))}
spec = {}
for k, r in load(os.path.join(V, "04_vacuum_spectra.jsonl")).items():
    gid = r["geometry_id"]
    new, old = os.path.join(V, "corrected_spectra", gid, "cyax.h5"), os.path.join(fcorr, gid, "cyax.h5")
    diffs = []
    with h5py.File(new, "r") as a, h5py.File(old, "r") as b:
        for grp in ("spectrum/physical", "vacuum_correction"):
            names = []
            a[grp].visit(lambda n: names.append(n) if isinstance(a[grp][n], h5py.Dataset) else None)
            for n in names:
                if n.startswith("metadata/") or n in ("theta0_spectrum_source",):
                    continue
                x, y = a[grp][n][()], b[grp][n][()] if n in b[grp] else None
                if y is None or (np.asarray(x).dtype.kind in "SUO" and np.asarray(x).tolist() != np.asarray(y).tolist()) or \
                   (np.asarray(x).dtype.kind not in "SUO" and not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)):
                    diffs.append(f"{grp}/{n}")
    spec[gid] = {"identical": not diffs, "different": diffs}
res["04_corrected_spectra"] = {"geometries": len(spec), "identical": all(v["identical"] for v in spec.values()), "per_geometry": spec}
# vacuum classification and relic-input manifest records
cls_new = {r["geometry_id"]: r for r in json.load(open(os.path.join(V, "05_vacuum_classification.json")))}
cls_old = {r["geometry_id"]: r for r in json.load(open(os.path.join(fres, "05_vacuum_classification.json")))}
cdiff = {g: sorted(k for k in set(r) | set(cls_old.get(g, {})) if r.get(k) != cls_old.get(g, {}).get(k)) for g, r in cls_new.items()}
res["05_vacuum_classification"] = {"records": len(cls_new), "identical": not any(cdiff.values()), "differences": {g: d for g, d in cdiff.items() if d}}
FIELDS = ("status", "n_modes_above_H0", "theta0_stored_modes_above_H0", "validation_status", "exclusion_reason", "qcd_edge_flag",
          "qcd_edge_beyond_gate_tolerance", "subH0_tachyonic_leading_directions_at_theta0", "tranche", "normal_form_sha256",
          "theta0_tachyonic_abs_m_ge_H0", "h11", "position", "frst")
man_new = {r["geometry_id"]: r for r in json.load(open(os.path.join(work, "relic_input", "relic_abundance_input_manifest.json")))["geometries"]}
man_old = {r["geometry_id"]: r for r in json.load(open(fman))["geometries"]}
mdiff = {g: [f for f in FIELDS if r.get(f) != man_old.get(g, {}).get(f)] if g in man_old else ["missing in frozen"] for g, r in man_new.items()}
res["relic_input_manifest_records"] = {"records": len(man_new), "fields": list(FIELDS), "identical": not any(mdiff.values()),
                                       "differences": {g: d for g, d in mdiff.items() if d}}
res["all_identical"] = all(v["identical"] for v in res.values() if isinstance(v, dict))
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1)[:4000])
sys.exit(0 if res["all_identical"] else 1)
