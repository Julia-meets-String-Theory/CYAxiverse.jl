"""Independent cross-check of the Julia theta analysis (Python used for VERIFICATION ONLY; it draws nothing).

Recomputes from the relic tables, with independent code paths:
  * theta_90 / theta_100 per geometry, exact N_TIGHT maxima and intervals on (0, 0.5] (theta_core.py),
    class counts at the representative angle, minimum theta_90;
  * the reheating census against the relic run's own global summary and, for a reference case, the approved counts;
  * PRE c = Omega/(f^2 theta^2) equals the configured k(T_R).
(The representative-geometry and figure-data checks of the original cross-check belong to the excluded presentation stage.)
Run: WORK=... python 02_crosscheck_theta.py
"""
import json, math, sys

import numpy as np
import pandas as pd

import os
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RELIC = os.environ.get("SHOROQ_RELIC_DIR", os.path.join(os.environ.get("WORK", ""), "relic"))
FA = os.environ.get("SHOROQ_FINAL_DIR", os.path.join(os.environ.get("WORK", ""), "final_analysis")); PT = f"{RELIC}/tables"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "engine", "relic"))
from theta_core import profile  # noqa: E402
from common import check_run_location, read_table  # noqa: E402

cfg = json.load(open(os.environ.get("SHOROQ_THETA_CONFIG", f"{FA}/theta_config.json"))); ODM = cfg["dark_matter"]["Omega_DM_h2"]; TR = cfg["theta"]["theta_ref"]; HI = cfg["theta"]["theta_max"]
XLO, XHI = cfg["dark_matter"]["TIGHT_DM"]
pcfg = json.load(open(f"{RELIC}/run_config.json")); check_run_location(pcfg, RELIC)
LABS = [s["label"] for s in pcfg["reheating"]["scenarios"]]
GSUM = json.load(open(f"{PT}/global_summary.json"))
# population (approved 2026-10-31): PRIMARY (complete abundances) or SENSITIVITY (primary + PARTIAL_VALIDITY partial sums)
POPN = os.environ.get("THETA_POPULATION", "PRIMARY"); assert POPN in ("PRIMARY", "SENSITIVITY"), POPN
TDIR, XOUT, KEEP = (("theta", "crosscheck_python.json", ("PRIMARY",)) if POPN == "PRIMARY" else
                    ("theta_sensitivity_partial_validity", "crosscheck_python_sensitivity_partial_validity.json", ("PRIMARY", "SENSITIVITY_ONLY")))
summ = json.load(open(f"{FA}/{TDIR}/theta_summary_final.json"))
_tgp = f"{FA}/{TDIR}/theta_geometry_final.csv"                                                # "Inf" for a geometry with Omega = 0
tg = (pd.read_csv(_tgp, dtype={"theta_90": float}) if os.path.getsize(_tgp) > 1 else          # empty population: empty table
      pd.DataFrame({"scenario": pd.Series(dtype=str), "geometry_id": pd.Series(dtype=str), "theta_90": pd.Series(dtype=float)}))
same = lambda a, b: a == b or abs(a / b - 1) < 1e-13                                          # interval endpoints (0 allowed)
out = {"theta": {}, "census": {}, "new6": {}}; ok = True


def theta_check(lab, om, g):
    """exact maxima, intervals, counts at theta_rep and theta_90 of one population, independently of the Julia code"""
    with np.errstate(divide="ignore"):                       # Omega = 0: thresholds +inf (never in the band)
        t90 = TR * np.sqrt(XLO * ODM / om); t100 = TR * np.sqrt(XHI * ODM / om)
    A = profile(om, XLO, ODM, TR, 0.0, HI, x_hi=XHI); s = summ[lab]
    if A["N_max"] == 0:                                      # NO_MATCH: no maximising interval and no representative angle
        A["intervals"] = []
    same_iv = len(A["intervals"]) == len(s["maximal_intervals"]) and all(same(a, b) for x, y in zip(A["intervals"], s["maximal_intervals"]) for a, b in zip(x, y))
    rep = s["theta_rep"]
    if rep is None:
        c_ok = A["N_max"] == 0 and s["counts_at_theta_rep"] is None and s["upper_boundary"]["status"] == "NO_MATCH"
    else:
        cnt = {"TIGHT_DM": int(((t90 <= rep) & (rep <= t100)).sum()), "OVERABUNDANT": int((rep > t100).sum()), "ALLOWED_UNDERABUNDANT": int((rep < t90).sum())}
        c_ok = all(cnt[k] == s["counts_at_theta_rep"][k] for k in cnt)
    J = tg[tg.scenario == lab].set_index("geometry_id").loc[g.geometry_id]
    jt = J.theta_90.to_numpy(float); fin = np.isfinite(t90)
    rel = float(np.max(np.abs(jt[fin] / t90[fin] - 1))) if fin.any() else 0.0
    inf_ok = bool((np.isinf(jt) == ~fin).all())
    mn = float(t90[fin].min()) if fin.any() else None
    min_ok = (mn is None and s["min_theta_90"] is None) or (mn is not None and s["min_theta_90"] is not None and abs(mn / s["min_theta_90"] - 1) < 1e-13)
    r = {"N_max_python": A["N_max"], "N_max_julia": s["N_tight_max"], "intervals_agree": same_iv, "counts_agree": c_ok,
         "min_theta_90_python": mn, "min_theta_90_julia": s["min_theta_90"], "theta_90_max_rel_diff": rel}
    if not fin.all():                                         # reported only when such geometries exist (reference outputs unchanged)
        r["Omega_zero_geometries_theta_90_inf"] = inf_ok
    good = A["N_max"] == s["N_tight_max"] and same_iv and c_ok and rel < 1e-12 and min_ok and inf_ok and len(J) == len(g) == int((tg.scenario == lab).sum())
    return r, good
for lab in LABS:
    g = read_table(f"{PT}/geometry_summary_{lab}.csv", "geometry_summary_<label>.csv")
    V = read_table(f"{PT}/geometry_abundance_validity_{lab}.csv", "geometry_abundance_validity_<label>.csv")
    assert list(V.geometry_id) == list(g.geometry_id)
    sel = V.theta_population.isin(KEEP).to_numpy()
    om = np.where(V.theta_population == "PRIMARY", V.Omega_complete_h2_at_theta_ref, V.Omega_valid_modes_partial_h2_at_theta_ref).astype(float)[sel]
    g = g[sel].reset_index(drop=True)
    if summ[lab].get("status") == "EMPTY_POPULATION" or len(g) == 0:
        good = len(g) == 0 and summ[lab].get("status") == "EMPTY_POPULATION"
        out["theta"][lab] = {"empty_population": good}; ok &= good
    else:
        r, good = theta_check(lab, om, g); out["theta"][lab] = r; ok &= good
    m = read_table(f"{PT}/master_modes_{lab}.csv", "master_modes_<label>.csv")         # schema-typed (F2)
    cen = m.category.value_counts().to_dict(); cen["excluded"] = int((~m.inflation_valid).sum())
    gc = GSUM[lab]["census"]                                   # the relic driver's own census
    c3 = all(cen.get(c, 0) == gc["by_category"][c] for c in ("FROZEN", "LATE", "POST_REHEATING", "PRE_REHEATING")) and \
        cen["excluded"] == gc["by_abundance_state"]["INFLATIONARY_VALIDITY_EXCLUDED"]
    if "expected_census_before_inflation_cut" in pcfg:         # reference case: the approved counts
        n3 = pcfg["expected_census_before_inflation_cut"][lab]; nx = pcfg["validity"]["inflation_cut"]["expected_modes"]
        c3 = c3 and (cen.get("FROZEN", 0) == n3["FROZEN"] and cen.get("LATE", 0) == n3["LATE"] and cen.get("POST_REHEATING", 0) == n3["POST_REHEATING"]
                     and cen.get("PRE_REHEATING", 0) == n3["PRE_REHEATING"] and cen["excluded"] == nx)
    out["census"][lab] = {"python": {k: int(v) for k, v in cen.items()}, "agree": bool(c3)}; ok &= bool(c3)
    pre = m[(m.category == "PRE_REHEATING") & m.inflation_valid]
    k = (pre.Omega_h2_at_theta_ref.astype(float) / (pre.f_pert_GeV.astype(float) ** 2 * TR ** 2)).to_numpy()   # float even if every value is integral
    kc = next(s_["Omega_pre_h2_per_fGeV2_theta2"] for s_ in pcfg["reheating"]["scenarios"] if s_["label"] == lab)
    out["new6"][lab] = {"PRE_c_over_k_max_rel": float(np.max(np.abs(k / kc - 1))) if len(k) else 0.0}; ok &= out["new6"][lab]["PRE_c_over_k_max_rel"] < 1e-12

out["all_checks_pass"] = bool(ok)
os.makedirs(f"{FA}/verification", exist_ok=True)
out["population"] = POPN
json.dump(out, open(f"{FA}/verification/{XOUT}", "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
print("ALL CHECKS PASS:", ok)
