"""Acceptance of the stage-2 integrity sample (producers/geometry/integrity_sample.py + integrity_spectra.jl) with the
criteria the production dataset was accepted on (thesis_g2_adaptive run manifests: "exact chain + independent spectrum
recompute identical"; all of them hold for every sampled geometry of both production tranches).

Pass criteria, for every sampled geometry:
  tip_materialized_bitwise_raw     Kaehler point read by Julia's input equals the G2 output bit for bit
  Q_materialized_equals_exact      instanton charge matrix equals the one rebuilt from the rays
  L_sign_materialized_equals_exact signs of the instanton scales equal the exact-rational signs
  julia_Q_exact, julia_L_equal, julia_tip_bitwise   Julia reads Q, L and the tip exactly as stored
  integrity_spectra.tsv: independent spectrum recompute has the same mode count, masses and f_pert as the batch output
The sample must not be empty and every sampled geometry must have a spectrum recompute row.

Recorded as diagnostics only (no pass/fail threshold, as in production):
  L_materialized_vs_exact_max_dex, L_materialized_vs_exact_leading300_max_dex, tau/Kinv_materialized_vs_exact_rel,
  julia_qKq_rel.
  Why L dex is not a rejection criterion: the materialized L are float64 log10 values of the instanton scales and the
  exact values are 60-digit mpmath values of Eq. 9 at the float64-stored Kaehler point. For very anisotropic geometries
  (V ~ 1e9-1e12) the leading terms differ by up to 6.7e-4 dex in the accepted production data (all terms: 5.8e-3),
  typically < 1e-6; production documented this as a precision note, not a criterion. Whether the regenerated L equal the
  frozen L is decided bit for bit by compare_geometry_subset.py, and |qKq| agreement by the final gate.
Usage: python check_integrity_sample.py INTEGRITY_DIR OUT_JSON
"""
import csv, json, os, sys

d, out = sys.argv[1:3]
EXACT = ("tip_materialized_bitwise_raw", "Q_materialized_equals_exact", "L_sign_materialized_equals_exact",
         "julia_Q_exact", "julia_L_equal", "julia_tip_bitwise")
DIAG = ("L_materialized_vs_exact_max_dex", "L_materialized_vs_exact_leading300_max_dex", "tau_materialized_vs_exact_rel",
        "Kinv_materialized_vs_exact_rel", "julia_qKq_rel")
s = json.load(open(os.path.join(d, "integrity_sample.json")))["sample"]
t = {r["file"]: r for r in csv.DictReader(open(os.path.join(d, "integrity_spectra.tsv")), delimiter="\t")}
fail = []
for r in s:
    bad = [k for k in EXACT if r.get(k) is not True]
    sp = t.get(r["file"])
    if sp is None:
        bad.append("no spectrum recompute row")
    elif not (sp["masses_identical"] == "true" and sp["fpert_identical"] == "true" and sp["recomputed_modes"] == sp["batch_modes"]):
        bad.append("spectrum recompute differs")
    if bad:
        fail.append({"file": r["file"], "failed": bad})
res = {"sampled": len(s), "spectrum_rows": len(t), "criteria": list(EXACT) + ["spectrum recompute identical"],
       "failures": fail, "diagnostics_max": {k: max((r[k] for r in s), default=None) for k in DIAG},
       "diagnostics_per_geometry": [{"file": r["file"], "stratum": r["stratum"], **{k: r[k] for k in DIAG}} for r in s]}
res["pass"] = bool(s) and not fail and len(t) == len(s)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("sampled", "spectrum_rows", "failures", "diagnostics_max", "pass")}, indent=1))
sys.exit(0 if res["pass"] else 1)
