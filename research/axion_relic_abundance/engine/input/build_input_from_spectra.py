"""Relic input from existing CYAxiverse physical spectra (no geometry generation, no spectrum recomputation).

Writes the same relic-input layout as the vacuum-repair producer (manifest, geometries CSV, spectra/ view, SHA256SUMS.txt),
so stages 4-5 run unchanged. Every spectrum file must satisfy the spectrum contract of DATA_CONTRACTS.md; A1
(engine/relic/01_build_mode_table.py) re-checks every file and every number.

Vacuum validation (explicit, per geometry; never silently accepted):
  "VERIFY"               the theta = 0 inertia scan (producers/vacuum/00_theta0_inertia_scan.jl, run by
                         run_00_prepare_from_spectra.sh on the potential stored in the file) decides:
                         VERIFIED_THETA0_INERTIA  no tachyonic direction with |m| >= H0 and inertia count = stored count
                         TACHYONIC_AT_THETA0      tachyonic directions with |m| >= H0 (repair with the vacuum producer)
                         UNVERIFIED               scan error (e.g. no potential data in the file) or count mismatch
  "DOCUMENTED_EXTERNAL"  vacuum established elsewhere: method, reference and evidence files (sha256 given, re-hashed
                         here and again by A1). Not re-verified by this tool; reported as such in every run.
  anything else          UNVERIFIED
A geometry whose state is not usable stops the build (vacuum_policy.on_unusable = "refuse", default) or is excluded
with status EXCLUDED_<state> ("exclude"); excluded geometries are listed in the manifest and never enter the tables.
Usage: python build_input_from_spectra.py SPECTRA_LIST OUT_DIR [--inertia-scan SCAN_JSONL]
"""
import argparse, csv, hashlib, json, math, os, re, sys
from collections import Counter, defaultdict

import h5py

ap = argparse.ArgumentParser()
ap.add_argument("spectra_list"); ap.add_argument("out_dir"); ap.add_argument("--inertia-scan", default=None)
a = ap.parse_args()
try:
    L = json.load(open(a.spectra_list))
except (OSError, ValueError) as e:
    sys.exit(f"spectra list {a.spectra_list}: not readable JSON ({e})")
isinstance(L, dict) and L.get("schema") == "cyaxiverse_relic_spectra/1" or sys.exit("spectra list: schema must be cyaxiverse_relic_spectra/1")
BASE = os.path.dirname(os.path.abspath(a.spectra_list))
POLICY = L.get("vacuum_policy", {}).get("on_unusable", "refuse")
POLICY in ("refuse", "exclude") or sys.exit("vacuum_policy.on_unusable must be 'refuse' or 'exclude'")
OUT = os.path.abspath(a.out_dir)
os.path.exists(OUT) and sys.exit(f"{OUT} exists: refusing to overwrite")
THR = math.log10(2.13 * 0.7e-33)                     # H0 threshold of the VMM physical spectrum (log10 eV)
GID = re.compile(r"h11_(\d{3})/np_(\d{7})/cy_(\d{7})")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


absp = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(BASE, p))
scan = {}
if a.inertia_scan:
    for line in open(a.inertia_scan):
        r = json.loads(line); scan[(r["h11"], r["position"], r.get("frst", 1))] = r

records, seen, refused = [], set(), []
isinstance(L.get("spectra"), list) and L["spectra"] or sys.exit("spectra list: 'spectra' must be a non-empty list")
for k, e in enumerate(L["spectra"]):
    isinstance(e, dict) and isinstance(e.get("path"), str) or sys.exit(f"spectra[{k}]: entry needs a 'path' string")
    path = absp(e["path"])
    gid = e.get("geometry_id") or "/".join(path.split("/")[-4:-1])
    m = GID.fullmatch(gid) or sys.exit(f"{e['path']}: geometry_id {gid!r} must be h11_XXX/np_XXXXXXX/cy_XXXXXXX (CYAxiverse layout)")
    h, pos, frst = (int(x) for x in m.groups())
    gid in seen and sys.exit(f"duplicate geometry_id {gid}")
    seen.add(gid)
    os.path.isfile(path) or sys.exit(f"{path}: no such spectrum file")
    try:
        f = h5py.File(path, "r")
    except OSError as err:
        sys.exit(f"{path}: not a readable HDF5 file ({str(err)[:120]})")
    with f:
        "spectrum/physical/m" in f and "spectrum/physical/metadata" in f or sys.exit(f"{path}: no spectrum/physical group (compute the spectrum first)")
        n = len(f["spectrum/physical/m"][()])
        md = f["spectrum/physical/metadata"]
        ep = md["evaluation_point"][()] if "evaluation_point" in md else b""
        ep = ep.decode() if isinstance(ep, bytes) else str(ep)
        corrected = ep == "validated_leading_branch_minimum" or "vacuum_correction" in f
    status = "VACUUM_CORRECTED" if corrected else "ORIGINAL_SPECTRUM"
    v = dict(e.get("vacuum_validation") or {})
    req = v.get("state", "UNVERIFIED")
    reason = None
    if req == "VERIFY":
        s = scan.get((h, pos, frst))
        if corrected:
            state, reason = "UNVERIFIED", "VERIFY applies to theta = 0 spectra; a corrected spectrum needs DOCUMENTED_EXTERNAL evidence"
        elif s is None:
            state, reason = "UNVERIFIED", "no inertia-scan record (run run_00_prepare_from_spectra.sh)"
        elif "error" in s:
            state, reason = "UNVERIFIED", f"inertia scan failed: {s['error']}"
        elif s["inertia_below_minus_H0"] > 0:
            state, reason = "TACHYONIC_AT_THETA0", f"{s['inertia_below_minus_H0']} tachyonic direction(s) with |m| >= H0 at theta = 0"
        elif not (s["inertia_above_H0"] == s["stored_physical_count"] == n):
            state, reason = "UNVERIFIED", f"inertia count {s['inertia_above_H0']} != stored count {n}"
        else:
            state = "VERIFIED_THETA0_INERTIA"
        v = {"state": state, "method": "theta = 0 production-Hessian inertia at +-H0 (00_theta0_inertia_scan.jl, prec 500)",
             "reference": "this run: vacuum_check/inertia_scan.jsonl", "evidence": []}
        if s is not None and "error" not in s:
            # the certificate covers |m| >= H0 only; below H0 the full-Hessian inertia is resolved only if the leading-scale span
            # fits the working precision (500 digits) and every PQ eigenvalue was resolved
            span = s.get("pq_eig_dynamic_range_dex"); res_all = s.get("pq_modes_within_prec") == s.get("n_leading")
            v["diagnostics"] = {"inertia_negative_full_hessian": s.get("inertia_negative"), "pq_modes_within_prec": s.get("pq_modes_within_prec"),
                                "n_leading": s.get("n_leading"), "pq_eig_dynamic_range_dex": span, "working_precision_digits": 500,
                                "subH0_inertia_resolved": bool(res_all and span is not None and span < 500),
                                "note": "VERIFIED_THETA0_INERTIA certifies directions with |m| >= H0 only (METHODOLOGY.md)"}
    elif req == "DOCUMENTED_EXTERNAL":
        ev = []
        for x in v.get("evidence", []):
            isinstance(x, dict) and isinstance(x.get("path"), str) or sys.exit(f"{gid}: every evidence entry needs 'path' and 'sha256'")
            p = absp(x["path"])
            os.path.isfile(p) or sys.exit(f"{gid}: evidence file {x['path']} not found")
            str(x.get("sha256", "")).lower() == sha(p) or sys.exit(f"{gid}: evidence file {x['path']}: sha256 differs from the declared value")
            x = dict(x, sha256=sha(p))
            ev.append({"path": p, "sha256": x["sha256"]})
        (v.get("method") and v.get("reference") and ev) or sys.exit(f"{gid}: DOCUMENTED_EXTERNAL needs method, reference and evidence files")
        state = "DOCUMENTED_EXTERNAL"; v["evidence"] = ev
    else:
        state, reason = "UNVERIFIED", f"vacuum_validation.state {req!r}: no verification or documented evidence"
        v = {"state": state}
    v["state"] = state
    if reason:
        v["reason"] = reason
    pv = e.get("provenance", {})
    rec = {"geometry_id": gid, "h11": h, "position": pos, "frst": frst, "tranche": pv.get("tranche", "NA"),
           "local_index": pv.get("local_index", pos), "normal_form_sha256": pv.get("normal_form_sha256", "NA"),
           "qcd_edge_flag": pv.get("qcd_edge_flag", "NA"), "qcd_edge_beyond_gate_tolerance": pv.get("qcd_edge_beyond_gate_tolerance", "NA"),
           "theta0_stored_modes_above_H0": n if status == "ORIGINAL_SPECTRUM" else pv.get("theta0_stored_modes_above_H0", "NA"),
           "subH0_tachyonic_leading_directions_at_theta0": "NA", "vacuum_validation": v}
    usable = state in ("VERIFIED_THETA0_INERTIA", "DOCUMENTED_EXTERNAL")
    if usable:
        rec.update(status=status, spectrum_source=path, spectrum_source_sha256=sha(path), n_modes_above_H0=n,
                   vacuum_source=v.get("reference", ""), validation_status=state, exclusion_reason=None)
    else:
        refused.append((gid, state, reason))
        rec.update(status=f"EXCLUDED_{state}", spectrum_source=None, n_modes_above_H0=0, vacuum_source=v.get("reference", ""),
                   validation_status=state, exclusion_reason=reason)
    records.append(rec)

if refused and POLICY == "refuse":
    sys.exit("vacuum not usable for %d geometr%s (vacuum_policy.on_unusable = refuse):\n  " % (len(refused), "y" if len(refused) == 1 else "ies")
             + "\n  ".join(f"{g}: {s} ({r})" for g, s, r in refused))
records.sort(key=lambda r: (r["h11"], r["geometry_id"]))
usable = [r for r in records if r["spectrum_source"]]
usable or sys.exit("no usable geometry")
os.makedirs(f"{OUT}/spectra")
for r in usable:
    link = f"{OUT}/spectra/{r['geometry_id']}/cyax.h5"
    os.makedirs(os.path.dirname(link), exist_ok=True)
    os.symlink(r["spectrum_source"], link)
by_h = defaultdict(Counter)
for r in records:
    b = by_h[r["h11"]]; b["geometries"] += 1; b[r["status"]] += 1; b["modes_final"] += r["n_modes_above_H0"]
summary = {"final_usable_geometries": len(usable), "final_usable_modes": sum(r["n_modes_above_H0"] for r in usable),
           "excluded_geometries": len(records) - len(usable),
           "by_status": dict(Counter(r["status"] for r in records)),
           "by_vacuum_validation": dict(Counter(r["vacuum_validation"]["state"] for r in records))}
manifest = {
    "name": "relic_abundance_input_manifest",
    "created": "built by engine/input/build_input_from_spectra.py",
    "purpose": L.get("description", "relic input from existing CYAxiverse physical spectra"),
    "population_rule": "Only CYAxiverse stored physical modes (m >= H0 = 10^%.4f eV) enter; the unit of exclusion is the geometry." % THR,
    "spectrum_source_base": "",
    "source_list": {"path": os.path.abspath(a.spectra_list), "sha256": sha(a.spectra_list)},
    "vacuum_policy": {"on_unusable": POLICY},
    "inertia_scan": {"path": os.path.abspath(a.inertia_scan), "sha256": sha(a.inertia_scan)} if a.inertia_scan else None,
    "spectra_view": "spectra/ (absolute symlinks; one cyax.h5 per usable geometry; read spectrum/physical/*)",
    "summary": summary,
    "per_h11": {h: dict(by_h[h]) for h in sorted(by_h)},
    "geometries": records,
}
json.dump(manifest, open(f"{OUT}/relic_abundance_input_manifest.json", "w"), indent=1)
with open(f"{OUT}/relic_abundance_input_geometries.csv", "w", newline="") as f:
    cols = ["geometry_id", "h11", "position", "tranche", "local_index", "status", "n_modes_above_H0",
            "theta0_stored_modes_above_H0", "spectrum_source", "spectrum_source_sha256", "validation_status",
            "exclusion_reason", "qcd_edge_flag", "qcd_edge_beyond_gate_tolerance", "subH0_tachyonic_leading_directions_at_theta0", "normal_form_sha256"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in records]
with open(f"{OUT}/SHA256SUMS.txt", "w") as f:
    for name in ("relic_abundance_input_manifest.json", "relic_abundance_input_geometries.csv"):
        f.write(f"{sha(f'{OUT}/{name}')}  {name}\n")
print(json.dumps(summary, indent=1))
for g, s, r in refused:
    print(f"EXCLUDED {g}: {s} ({r})")
