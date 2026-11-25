"""Independent re-read of relic_abundance_input_manifest.json against the files it points to (read-only).
Usage: python 07_verify_manifest.py MANIFEST OUT_JSON"""
import hashlib, json, math, os, sys
from collections import Counter
import h5py

man, out = sys.argv[1], sys.argv[2]
M = json.load(open(man))
THR = math.log10(2.13 * 0.7e-33)
root = os.path.dirname(man)
base = os.path.normpath(os.path.join(root, M.get("spectrum_source_base", "")))   # spectrum_source is relative to WORK
bad = []; tot = Counter(); seen = set()
for r in M["geometries"]:
    g = r["geometry_id"]; seen.add(g)
    assert g not in tot, g
    tot["records"] += 1; tot[r["status"]] += 1
    if r["status"].startswith("EXCLUDED_"):
        if os.path.lexists(f"{root}/spectra/{g}/cyax.h5"): bad.append((g, "excluded geometry present in view"))
        continue
    p = r["spectrum_source"] if os.path.isabs(r["spectrum_source"]) else os.path.join(base, r["spectrum_source"])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != r["spectrum_source_sha256"]: bad.append((g, "sha256"))
    link = f"{root}/spectra/{g}/cyax.h5"
    if os.path.realpath(link) != os.path.realpath(p): bad.append((g, "view link"))
    with h5py.File(p, "r") as f:
        m = f["spectrum/physical/m"][()]
        n = len(m)
        md = f["spectrum/physical/metadata"]
        if int(md["prec"][()]) != 500 or abs(float(md["threshold_log10"][()]) - THR) > 1e-12: bad.append((g, "settings"))
        if (r["status"] == "VACUUM_CORRECTED") != ("vacuum_correction" in f): bad.append((g, "vacuum group"))
        for k in ("m", "fK_log10", "fpert_log10", "lambda_self_log10", "lambda_self_sign"):
            if k not in f["spectrum/physical"]: bad.append((g, f"missing {k}"))
    if n != r["n_modes_above_H0"]: bad.append((g, f"count {n} != {r['n_modes_above_H0']}"))
    if n and min(m) < THR: bad.append((g, "mass below H0"))
    tot["modes"] += n
    if r["status"] == "VACUUM_CORRECTED" and "/vacuum_repair/corrected_spectra/" not in os.path.realpath(p):
        bad.append((g, "corrected source outside vacuum_repair/corrected_spectra"))
    if r["status"] == "ORIGINAL_SPECTRUM" and "/corrected_spectra/" in os.path.realpath(p):
        bad.append((g, "original source inside corrected_spectra"))
nview = sum(1 for d, _, fs in os.walk(f"{root}/spectra") for x in fs if x == "cyax.h5")
res = {"records": tot["records"], "by_status": {k: v for k, v in tot.items() if k not in ("records", "modes")},
       "modes_total": tot["modes"], "view_files": nview, "problems": bad,
       "summary_matches": tot["modes"] == M["summary"]["final_usable_modes"] and
                          tot["ORIGINAL_SPECTRUM"] + tot["VACUUM_CORRECTED"] == M["summary"]["final_usable_geometries"]}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
