"""Classify the tachyonic geometries from the evidence files of steps 00-04, then build the relic-abundance input
manifest (every accepted geometry, one record each) and a spectra view of relative symlinks. Nothing under WORK/spectra
is written. Paths stored in the manifest are relative to WORK (manifest key spectrum_source_base = "..").
Usage: python 05_build_relic_input.py WORK POLYTOPE_MANIFEST [--expected-tachyonic N]
  WORK               pipeline output root (validation/status, validation/qcd_edge_flags.json, spectra/, vacuum_repair/)
  POLYTOPE_MANIFEST  references/shoroq/data/polytope_manifest_h11_004_050.json (unified positions, historical tranche, normal forms)
"""
import argparse, csv, glob, hashlib, json, math, os, sys
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("work"); ap.add_argument("polytope_manifest")
ap.add_argument("--expected-tachyonic", type=int, default=None, help="optional gate on the number of theta = 0 tachyonic geometries (reference cases)")
args = ap.parse_args()
WORK = os.path.abspath(args.work)
W = f"{WORK}/vacuum_repair"
OUT = f"{WORK}/relic_input"
THR = math.log10(2.13 * 0.7e-33)


def jl(p):
    return [json.loads(l) for l in open(p)]


def key(r):
    return (r["h11"], r["position"])


def gid(h, p):
    return f"h11_{h:03d}/np_{p:07d}/cy_0000001"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


acc = {key(r): r for r in jl(f"{W}/00_theta0_inertia_scan.jsonl")}
s01 = {key(r): r for r in jl(f"{W}/01_theta0_structure.jsonl")}
vac = {key(r): r for r in jl(f"{W}/02_vacua_auto_groups.jsonl")}
val = {key(r): r for r in jl(f"{W}/03_minima_validation.jsonl")}
spc = {key(r): r for r in jl(f"{W}/04_vacuum_spectra.jsonl")}
# accepted geometries: G2 status ledger (single unified run: polytope_index = unified position) joined to the sample manifest
status = {(int(r["h11"]), int(r["polytope_index"])): r["status"] for r in csv.DictReader(open(f"{WORK}/validation/status/polytope_status.csv"))}
uni = [{"h11": p["h11"], "position": p["unified_position"], "tranche": p.get("historical_tranche", "NA"), "local_index": p["unified_position"],
        "normal_form_sha256": p.get("normal_form_sha256", "NA"), "status": status.get((p["h11"], p["unified_position"]), "not_attempted")}
       for p in json.load(open(args.polytope_manifest))["polytopes"]]
qcd = {(f["h11"], f["unified_position"]): f for f in json.load(open(f"{WORK}/validation/qcd_edge_flags.json"))["flags"]}
tach46 = sorted(s01)
assert args.expected_tachyonic is None or len(tach46) == args.expected_tachyonic, f"{len(tach46)} tachyonic geometries, expected {args.expected_tachyonic}"


def classify(k):
    """Evidence-based vacuum status for one tachyonic geometry: (status, reasons, chosen-copy record)."""
    v, s, a, s1 = val.get(k), spc.get(k), vac.get(k), s01[k]
    reasons = []
    if v is None or "error" in v:
        return "SOLVER_FAILURE", [f"validation error: {None if v is None else v.get('error')}"], None
    if a is None or a.get("status") != "completed":
        reasons.append("repository vacua pipeline (--method auto) did not complete")
    if s is None or "error" in s:
        return "INVALID_RESULT", [f"spectrum error: {None if s is None else s.get('error')}"], None
    c = next(c for c in v["copies"] if c["copy"] == s["chosen_copy"])
    tol = v["gradient_tolerance_log10"]
    checks = {
        "repository_vacua_auto_verified": a is not None and a.get("verification_status") == "verified",
        "exact_theta_matches_repository_branch": v["exact_vs_repository_theta_max_torus_distance"] < 1e-9,
        "newton_converged": c["correction_status"] == "converged",
        "gradient_numerical_le_tol": c["p1661_gradient_residual_inf_log10"] <= tol and c["p3322_gradient_residual_inf_log10"] <= tol,
        "gradient_analytic_le_tol": c["analytic_gradient_bound_log10"] <= tol,
        "no_resolved_negative_full_hessian_both_precisions": c["p1661_neg_resolved"] == 0 and c["p3322_neg_resolved"] == 0,
        "no_negative_abs_m_ge_H0_full_hessian": c["p1661_neg_ge_H0"] == 0 and c["p3322_neg_ge_H0"] == 0,
        "full_hessian_H0_count_precision_stable": c["p1661_pos_ge_H0"] == c["p3322_pos_ge_H0"],
        # exact: all effective leading amplitudes positive and Qtilde nonsingular => W_eff = C' Qt D Qt' C positive definite;
        # production inertia at -lambda(H0) confirms numerically (a prec-500 eigen-solver noise value is not a criterion)
        "production_matrix_positive_definite": c["W_eff_leading_negative_signs"] == 0 and c["W_eff_inertia_below_minus_H0"] == 0,
        "potential_lowered": c["V_min_sign"] < 0,
        "spectrum_not_provisional": not s["provisional"],
        "theta0_control_reproduces_stored": s["theta0_recomputed_equals_stored"],
        "spectrum_count_equals_inertia": s["new_count"] == c["W_eff_inertia_above_H0"],
        "degenerate_copies_identical_spectrum": s["copies_spectrum_max_abs_dm"] < 1e-9,
    }
    failed = [n for n, ok in checks.items() if not ok]
    if not failed:
        return "MINIMUM_FOUND", [], (c, checks)
    if not checks["no_resolved_negative_full_hessian_both_precisions"] or not checks["no_negative_abs_m_ge_H0_full_hessian"]:
        return "SADDLE_ONLY", failed, (c, checks)
    if not checks["newton_converged"] or not checks["gradient_numerical_le_tol"]:
        return "NO_CONVERGENCE", failed, (c, checks)
    return "INVALID_RESULT", failed, (c, checks)


os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/spectra", exist_ok=True)
records, per_geom_vac = [], []
tot = Counter()
for u in sorted((r for r in uni if r["status"] == "accepted"), key=lambda r: (r["h11"], r["position"])):
    h, p = u["h11"], u["position"]
    k = (h, p)
    frozen = f"spectra/{gid(h, p)}/cyax.h5"                    # theta = 0 production spectrum, relative to WORK
    a = acc[k]
    rec = {"geometry_id": gid(h, p), "h11": h, "position": p, "frst": 1, "tranche": u["tranche"],
           "local_index": u["local_index"], "normal_form_sha256": u["normal_form_sha256"],
           "qcd_edge_flag": k in qcd, "qcd_edge_beyond_gate_tolerance": bool(qcd.get(k, {}).get("beyond_gate_tolerance_1e-5_rel", False)),
           "theta0_stored_modes_above_H0": a["stored_physical_count"],
           "subH0_tachyonic_leading_directions_at_theta0": sum(1 for m, sg in zip(a["pq_log10_m_eV"], a["pq_sign"]) if sg < 0 and m < THR)}
    if k not in s01:
        rec.update(status="ORIGINAL_SPECTRUM", spectrum_source=frozen, n_modes_above_H0=a["stored_physical_count"],
                   vacuum_source="theta = 0 (frozen production spectrum; no leading direction with |m| >= H0 is tachyonic)",
                   validation_status="FROZEN_PRODUCTION_VALIDATED", exclusion_reason=None)
        tot["ORIGINAL_SPECTRUM"] += 1; tot["modes_original"] += rec["n_modes_above_H0"]
    else:
        st, reasons, cc = classify(k)
        s = spc.get(k, {})
        vrow = {"geometry_id": gid(h, p), "h11": h, "position": p, "vacuum_class": st, "failed_checks": reasons,
                "det_Qtilde": s01[k]["det_Qtilde"], "n_negative_leading": s01[k]["n_negative_leading"],
                "theta0_neg_abs_m_ge_H0": s01[k]["eig_negative_abs_m_ge_H0"], "theta0_stored": s01[k]["stored_physical_count"]}
        if st == "MINIMUM_FOUND":
            c, checks = cc
            rec.update(status="VACUUM_CORRECTED", spectrum_source=os.path.relpath(os.path.realpath(s["output"]), os.path.realpath(WORK)),
                       n_modes_above_H0=s["new_count"],
                       vacuum_source=f"vacuum repair: leading-branch minimum copy {s['chosen_copy']}/{s['n_copies']} "
                                     "(theta_min stored in the spectrum file, group vacuum_correction/); validation "
                                     "vacuum_repair/03_minima_validation.jsonl",
                       validation_status="VACUUM_VALIDATED (" + ", ".join(n for n in checks) + ")",
                       exclusion_reason=None,
                       theta0_tachyonic_abs_m_ge_H0=s01[k]["eig_negative_abs_m_ge_H0"])
            tot["VACUUM_CORRECTED"] += 1; tot["modes_corrected"] += s["new_count"]; tot["modes_corrected_old"] += s01[k]["stored_physical_count"]
            vrow.update(new_count=s["new_count"], gradient_residual_log10=c["p1661_gradient_residual_inf_log10"],
                        gradient_tolerance_log10=val[k]["gradient_tolerance_log10"], min_eig_sign=c["p1661_min_eig_sign"],
                        min_eig_abs_m_log10=c["p1661_min_eig_abs_mass_log10"], noise_floor_m_log10=c["p1661_noise_floor_mass_log10"])
        else:
            rec.update(status="EXCLUDED_VACUUM_UNRESOLVED", spectrum_source=None, n_modes_above_H0=0,
                       vacuum_source="vacuum_repair/03_minima_validation.jsonl",
                       validation_status=st, exclusion_reason="; ".join(reasons) or st)
            tot["EXCLUDED_VACUUM_UNRESOLVED"] += 1; tot["modes_excluded_old"] += s01[k]["stored_physical_count"]
        per_geom_vac.append(vrow)
    if rec["spectrum_source"]:
        target = f"{WORK}/{rec['spectrum_source']}"
        rec["spectrum_source_sha256"] = sha(target)
        link = f"{OUT}/spectra/{gid(h, p)}/cyax.h5"
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.relpath(target, os.path.dirname(link)), link)
    records.append(rec)

N_geom = tot["ORIGINAL_SPECTRUM"] + tot["VACUUM_CORRECTED"]
N_modes = tot["modes_original"] + tot["modes_corrected"]
by_h = defaultdict(lambda: Counter())
for r in records:
    b = by_h[r["h11"]]; b["geometries_frozen"] += 1; b[r["status"]] += 1; b["modes_final"] += r["n_modes_above_H0"]
    b["modes_theta0_stored"] += r["theta0_stored_modes_above_H0"]
summary = {
    "tachyonic_geometries_attempted": len(tach46),          # geometries of this run with theta = 0 tachyons |m| >= H0
    "minima_found_and_validated": tot["VACUUM_CORRECTED"],
    "unresolved_excluded": tot["EXCLUDED_VACUUM_UNRESOLVED"],
    "original_physical_modes": sum(r["theta0_stored_modes_above_H0"] for r in records),
    "unaffected_geometries": tot["ORIGINAL_SPECTRUM"], "unaffected_modes": tot["modes_original"],
    "repaired_geometries": tot["VACUUM_CORRECTED"], "repaired_modes_theta0_stored": tot["modes_corrected_old"],
    "repaired_modes_at_theta_min": tot["modes_corrected"],
    "excluded_geometries": tot["EXCLUDED_VACUUM_UNRESOLVED"], "excluded_old_modes": tot["modes_excluded_old"],
    "final_usable_geometries": N_geom, "final_usable_modes": N_modes,
    "geometries_with_subH0_tachyonic_direction_only_kept_unchanged": sum(1 for r in records if r["status"] == "ORIGINAL_SPECTRUM" and r["subH0_tachyonic_leading_directions_at_theta0"] > 0),
}
manifest = {
    "name": "relic_abundance_input_manifest",
    "created": "built by producers/vacuum/05_build_relic_input.py",
    "purpose": "Cosmology input population for the relic-abundance analysis: the frozen thesis G2 adaptive dataset "
               "(h11 4-50, 4491 geometries) with the 46 geometries whose theta = 0 production mass matrix has tachyonic "
               "directions with |m| >= H0 re-evaluated at a validated axion minimum.",
    "population_rule": "Only CYAxiverse stored physical modes (m >= H0, H0 = 2.13*0.7e-33 eV = 10^%.4f eV) enter; "
                       "sub-H0 modes are out of scope by decision (2026-09-30). The unit of exclusion is the geometry." % THR,
    "spectrum_source_base": "..",
    "frozen_dataset": {"root": "WORK (spectra/, geometry/, materialized/)", "code": "CYAxiverse.jl vmm a5eacd8",
                       "spectra_settings": "batch_physical_spectrum.jl --prec 500 --quartics (mixed quartics), threshold log10(H0)"},
    "vacuum_correction": {"workspace": "WORK/vacuum_repair",
                          "method": "repository leading-branch minimum (generate.leading_critical_branches, negative_mode_range 0:0; "
                                    "counted by batch_vacua_pipeline.jl --method auto as square_reduced_potential_determinant_count), "
                                    "validated on the full potential with CYAxiverse.inflation_points at ~500 and ~1000 digits; spectrum by the "
                                    "production engine and writer on the potential re-expanded about theta_min"},
    "spectra_view": "spectra/ (relative symlinks; one cyax.h5 per usable geometry; read spectrum/physical/*)",
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
json.dump(per_geom_vac, open(f"{W}/05_vacuum_classification.json", "w"), indent=1)
with open(f"{OUT}/SHA256SUMS.txt", "w") as f:                       # the input directory's own checksums (read by A1)
    for name in ("relic_abundance_input_manifest.json", "relic_abundance_input_geometries.csv"):
        f.write(f"{sha(f'{OUT}/{name}')}  {name}\n")
print(json.dumps(summary, indent=1))
print(Counter(v["vacuum_class"] for v in per_geom_vac))
