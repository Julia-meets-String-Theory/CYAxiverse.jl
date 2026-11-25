"""Acceptance gate for a G2 → materializer → Julia → spectrum stage (smoke test or full dataset).

Inputs are the outputs of the validation tools (audit_geometry.py JSONL, the materializer driver log, compare_julia.py
JSONL, batch_physical_spectrum.jl summary CSVs + logs). Every criterion has an explicit tolerance; the result is
written as JSON with PASS/FAIL per criterion, the worst observed values, and the list of offending geometries.
Usage: python validate_stage.py OUT_JSON --audit A.jsonl[,B.jsonl] --materialize M.jsonl[,..] --julia J.jsonl
                                 --spectra-csv 'glob' --spectra-log 'glob' [--expected-geometries N]
"""
import argparse, csv, glob, json, math

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--audit", required=True); ap.add_argument("--materialize", required=True)
ap.add_argument("--julia", required=True); ap.add_argument("--spectra-csv", required=True); ap.add_argument("--spectra-log", required=True)
ap.add_argument("--expected-geometries", type=int, default=None)
a = ap.parse_args()
rd = lambda paths: [json.loads(l) for p in paths.split(",") for l in open(p) if l.strip()]
aud, jul = rd(a.audit), rd(a.julia)
mat = [json.loads(l) for p in a.materialize.split(",") for l in open(p) if l.strip().startswith("{")]
spec = [r for p in sorted(glob.glob(a.spectra_csv)) for r in csv.DictReader(open(p))]
logs = "".join(open(p).read() for p in sorted(glob.glob(a.spectra_log)))

TOL = {"engine_rel": 1e-6, "kkinv": 1e-6, "slack": 1e-6, "divisor": 1e-8, "qcd_rel": 1e-5, "L_rel_to_Lmin": 1e-9,
       "julia_qkq_rel": 1e-6, "chol": 1e-12}
C = {}


def crit(name, rows, pred, worst=None):
    bad = [r.get("file") for r in rows if not pred(r)]
    C[name] = {"pass": not bad and len(rows) > 0, "checked": len(rows), "failed": len(bad), "failed_examples": bad[:5]}
    if worst is not None and rows:
        C[name]["worst"] = max(worst(r) for r in rows)


ok = [r for r in aud if "audit_error" not in r]
C["audit_ran_without_error"] = {"pass": len(ok) == len(aud) and len(aud) > 0, "checked": len(aud), "failed": len(aud) - len(ok),
                                "failed_examples": [(r["file"], r["audit_error"]) for r in aud if "audit_error" in r][:5]}
if a.expected_geometries is not None:
    C["geometry_count"] = {"pass": len(aud) == a.expected_geometries, "checked": len(aud), "expected": a.expected_geometries}
# --- geometry / adaptive point
crit("policy_is_adaptive_fast", ok, lambda r: r["moduli_policy"] == "adaptive" and r["sampler"] == "fast")
crit("cy_rebuild_identity", ok, lambda r: all(r[k] for k in ("rebuild_triangulation_id_match", "rebuild_basis_match", "rebuild_glsm_match", "rebuild_kappa_match", "rebuild_hyperplanes_match")))
crit("adaptive_m_and_qcd_replica", ok, lambda r: r["replica_matches_stored"])
crit("single_final_point_t_equals_m_tpre", ok, lambda r: r["tip_equals_m_times_pre"] == 0.0, lambda r: r["tip_equals_m_times_pre"])
crit("finite_positive_volume", ok, lambda r: math.isfinite(r["V_final"]) and r["V_final"] > 0)
crit("positive_basis_divisor_volumes", ok, lambda r: r["tau_min"] > 0)
crit("stretched_cone_slack", ok, lambda r: r["final_min_slack"] >= 1 - TOL["slack"], lambda r: -r["final_min_slack"])
crit("curve_volumes_positive", ok, lambda r: r["final_min_curve"] > 0)
crit("prime_divisor_volumes_ge_1", ok, lambda r: r["final_min_prime"] >= 1 - TOL["divisor"])
crit("effective_divisor_volumes_ge_1", ok, lambda r: r["final_min_eff"] >= 1 - TOL["divisor"])
crit("qcd_window_25_40", ok, lambda r: 25 * (1 - TOL["qcd_rel"]) <= r["qcd_volume_final"] <= 40 * (1 + TOL["qcd_rel"]),
     lambda r: max(25 - r["qcd_volume_final"], r["qcd_volume_final"] - 40))
crit("eq21_repo_criterion_fresh_at_final", ok, lambda r: r["eq21_final_fresh_onesided"])
crit("kinv_positive_definite", ok, lambda r: r["kinv_eig_min"] > 0)
# --- final-point consistency (three engines)
crit("tau_engines_agree", ok, lambda r: r["tau_cytools_vs_exact"] <= TOL["engine_rel"], lambda r: r["tau_cytools_vs_exact"])
crit("V_engines_agree", ok, lambda r: r["V_cytools_vs_exact"] <= TOL["engine_rel"] and r["V_stored_vs_exact"] <= TOL["engine_rel"], lambda r: max(r["V_cytools_vs_exact"], r["V_stored_vs_exact"]))
crit("kinv_engines_agree", ok, lambda r: r["kinv_cytools_vs_exact"] <= TOL["engine_rel"], lambda r: r["kinv_cytools_vs_exact"])
crit("K_times_Kinv_is_identity", ok, lambda r: r["KKinv_minus_I_exact"] <= TOL["kkinv"], lambda r: r["KKinv_minus_I_exact"])
crit("scaling_tau_m2_kinv_m4_V_m3", ok, lambda r: max(r["tau_scaling_m2"], r["kinv_scaling_m4"], r["V_scaling_check"]) <= TOL["engine_rel"],
     lambda r: max(r["tau_scaling_m2"], r["kinv_scaling_m4"], r["V_scaling_check"]))
# --- Q/L
crit("Q_equals_independent_eq9", ok, lambda r: r["Q_equal_independent"] and r["N_terms"] == r["N_expected"] and r["Q_zero_columns"] == 0)
crit("L_finite_and_matches_independent", ok, lambda r: r["L_finite"] and r["L_sign_equal_independent"]
     and r["L_log_maxdiff_independent"] <= TOL["L_rel_to_Lmin"] * max(1.0, abs(r["L_min"])), lambda r: r["L_log_maxdiff_independent"] / max(1.0, abs(r["L_min"])))
# --- materialization
matok = [m for m in mat if m.get("status") == "materialized"]
C["materializer_status"] = {"pass": len(matok) == len(mat) == len(aud), "checked": len(mat), "materialized": len(matok),
                            "failed_examples": [(m.get("source"), m.get("error")) for m in mat if m.get("status") != "materialized"][:5]}
crit("materialized_Q_L_Kinv_tip_V_equal_G2", ok, lambda r: all(r.get(k) is True for k in ("mat_Q_equal_G2", "mat_L_bitwise_G2", "mat_kinv_vs_exact_bitwise", "mat_tip_bitwise", "mat_V_stored_bitwise")))
crit("materialized_shapes", ok, lambda r: tuple(r["mat_Q_disk_shape"]) == (r["N_terms"], r["h11"]) and tuple(r["mat_L_disk_shape"]) == (r["N_terms"], 2))
# --- Julia
jok = [j for j in jul if "julia_error" not in j]
C["julia_reads_without_error"] = {"pass": len(jok) == len(jul) == len(aud), "checked": len(jul), "failed": len(jul) - len(jok),
                                  "failed_examples": [(j["file"], j["julia_error"]) for j in jul if "julia_error" in j][:5]}
crit("julia_orientation_and_values", jok, lambda j: j["Q_columns_exact"] and j["L_sign_equal"] and j["L_columns_maxdiff"] == 0.0 and j["tip_bitwise"]
     and j["tau_rel"] <= 1e-15 and j["V_julia_equals_stored"] and j["direct_columns_equal_effective_cone"] and j["Kinv_symmetric"]
     and tuple(j["python_Q_shape"]) == tuple(j["julia_reader_Q"]) == tuple(j["julia_raw_shapes"]["Q"]) and tuple(j["h5py_disk_shapes"]["Q"]) == tuple(j["python_Q_shape"][::-1]))
crit("julia_qKq_matches_python", jok, lambda j: j["qKq_rel_maxdiff"] <= TOL["julia_qkq_rel"], lambda j: j["qKq_rel_maxdiff"])
crit("julia_cholesky", jok, lambda j: j["cholesky_residual"] <= TOL["chol"], lambda j: j["cholesky_residual"])
# --- spectrum
C["spectrum_success"] = {"pass": len(spec) == len(aud) and all(s["status"] == "success" for s in spec), "checked": len(spec),
                         "failed": sum(s["status"] != "success" for s in spec),
                         "failed_examples": [(s["h11"], s["polytope"], s["frst"], s["error"]) for s in spec if s["status"] != "success"][:5]}
C["spectrum_not_provisional"] = {"pass": all(s["provisional"] == "false" for s in spec), "checked": len(spec),
                                 "failed": sum(s["provisional"] != "false" for s in spec)}
bad_lines = [l for l in logs.splitlines() if any(w in l.lower() for w in ("warning", "error", "failed:", "provisional"))]
C["spectrum_logs_clean"] = {"pass": not bad_lines, "offending_lines": bad_lines[:10]}
C["spectrum_physical_modes"] = {"pass": True, "min": min(int(s["physical_count"]) for s in spec) if spec else None,
                                "max": max(int(s["physical_count"]) for s in spec) if spec else None,
                                "total": sum(int(s["physical_count"]) for s in spec), "note": "informational"}
groups = {"G2 generation / adaptive point": ["audit_ran_without_error", "policy_is_adaptive_fast", "cy_rebuild_identity", "adaptive_m_and_qcd_replica",
                                             "finite_positive_volume", "positive_basis_divisor_volumes", "stretched_cone_slack", "curve_volumes_positive",
                                             "prime_divisor_volumes_ge_1", "effective_divisor_volumes_ge_1", "qcd_window_25_40",
                                             "eq21_repo_criterion_fresh_at_final", "kinv_positive_definite"] + (["geometry_count"] if a.expected_geometries is not None else []),
          "final-point consistency": ["single_final_point_t_equals_m_tpre", "tau_engines_agree", "V_engines_agree", "kinv_engines_agree",
                                      "K_times_Kinv_is_identity", "scaling_tau_m2_kinv_m4_V_m3", "Q_equals_independent_eq9", "L_finite_and_matches_independent"],
          "materialization": ["materializer_status", "materialized_Q_L_Kinv_tip_V_equal_G2", "materialized_shapes"],
          "Python/Julia orientation": ["julia_reads_without_error", "julia_orientation_and_values", "julia_qKq_matches_python", "julia_cholesky"],
          "physical spectrum": ["spectrum_success", "spectrum_not_provisional", "spectrum_logs_clean"]}
summary = {g: ("PASS" if all(C[c]["pass"] for c in cs) else "FAIL") for g, cs in groups.items()}
overall = "PASS" if all(v == "PASS" for v in summary.values()) else "FAIL"
json.dump({"overall": overall, "groups": summary, "tolerances": TOL, "criteria": C}, open(a.out, "w"), indent=1)
for g, cs in groups.items():
    print(f"{summary[g]:4s}  {g}")
    for c in cs:
        v = C[c]
        print(f"      {'ok ' if v['pass'] else 'BAD'} {c}: checked={v.get('checked')} failed={v.get('failed', 0)}" + (f" worst={v['worst']:.3g}" if "worst" in v else ""))
print("OVERALL", overall)
