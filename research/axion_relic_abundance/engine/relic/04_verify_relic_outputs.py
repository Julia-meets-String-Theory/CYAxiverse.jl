"""Independent verification of the production outputs.

  (a) C, the previously validated independent float64 implementation, re-derives classification, validity, T_osc,
      1+z_osc and Omega h^2 for EVERY production row (16,313 modes x 3 scenarios) and every per-geometry sum.
  (b) B, the 50-digit analytic evaluation, re-derives a stratified sample (scenario x state x h11 bin x vacuum status,
      plus extremes and boundary-nearest modes) and complete per-geometry sums for a stratified set of geometries.
The pre-production benchmark reproduction (part (c) of the original verifier) compared the run with dry-run artefacts
that are not part of this repository and is omitted; (a) and (b) are unchanged.
Usage: SHOROQ_RELIC_CONFIG=RUN_CONFIG python 04_verify_relic_outputs.py RELIC_OUT_DIR   (tables in RELIC_OUT_DIR/tables)
"""
import json, math, os, sys

import mpmath as mp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import check_run_location, dump_json, load_config, read_table
import reference_mp as Bm
import independent_relic as C

OUT = os.path.abspath(sys.argv[1]); T = f"{OUT}/tables"; V = f"{OUT}/validation"
os.makedirs(V, exist_ok=True)
cfg = load_config(); check_run_location(cfg, OUT); TH = cfg["axion"]["theta_reference"]; HI3 = cfg["validity"]["inflation_cut"]["three_H_I_max_GeV"]
SC = cfg["reheating"]["scenarios"]
tab = C.GTab(); c = C.Cosmo("vmm", tab); H0c, Heqc = c.H(C.T0), c.H(c.T_eq())
rd = lambda a, b: abs(a / b - 1)
res = {"C_full": {}, "B_sample": {}, "B_geometries": {}}
mx = lambda x: float(x.max()) if x.notna().any() else 0.0     # largest relative difference; 0.0 when a class has no row in this sample
M = {s["label"]: read_table(f"{T}/master_modes_{s['label']}.csv", "master_modes_<label>.csv") for s in SC}       # schema-typed (F2)
G = {s["label"]: read_table(f"{T}/geometry_summary_{s['label']}.csv", "geometry_summary_<label>.csv") for s in SC}

# A mass within 1e-12 dex of a class threshold (3H0, 3H(T_eq), 3H_R, 3H_I,max) is a tie: the side is decided by the last bit of
# each implementation's threshold value, so independent implementations may legitimately disagree there. Such rows are listed
# under "threshold_ties" (key present only when ties occur) and are the only category disagreements accepted.
def is_tie(lm, HRc):
    return any(abs((lm - 9) - math.log10(t)) < 1e-12 for t in (3 * H0c, 3 * Heqc, 3 * HRc, HI3))


# ---------------- (a) C on every row
tosc_cache = {}; DIS = {}
for s in SC:
    lab, TR = s["label"], s["T_reh_GeV"]; A = M[lab]; HRc = C.fuzzy_HR(tab, TR)
    cat, val, Om, Tos, opz = [], [], [], [], []
    for lm, lf in zip(A.log10_mass_eV.values, A.log10_f_pert_GeV.values):
        m, f = 10 ** (lm - 9), 10 ** lf
        k = "FROZEN" if m / 3 <= H0c else "PRE_REHEATING" if m > 3 * HRc else "LATE" if m / 3 <= Heqc else "POST_REHEATING"
        v = not (m > HI3); o = t = z = np.nan
        if v and k in ("LATE", "POST_REHEATING"):
            t = tosc_cache.get(lm)
            if t is None:
                t = tosc_cache[lm] = c.T_osc(m)
            o = c.omega_h2(m, f, TH, t); z = C.fuzzy_onepz_R(tab, t)
        elif v and k == "PRE_REHEATING":
            o, z = C.fuzzy_pre_omega_h2(tab, m, f, TH, TR)
        cat.append(k); val.append(v); Om.append(o); Tos.append(t); opz.append(z)
    X = A.assign(cat_C=cat, valid_C=val, Om_C=Om, Tosc_C=Tos, opz_C=opz)
    aom = X.Omega_h2_at_theta_ref.where(X.category != "LATE", X.Omega_h2_late_census_only_at_theta_ref)
    post, pre, late = X.category == "POST_REHEATING", (X.category == "PRE_REHEATING") & X.inflation_valid, X.category == "LATE"
    rel = (aom / X.Om_C - 1).abs()
    # per-geometry sums from C
    gC = X[X.included_in_cdm_sum].groupby("geometry_id").Om_C.sum()
    gA = G[lab].set_index("geometry_id").Omega_geometry_h2_at_theta_ref
    gpre = X[X.included_in_cdm_sum & (X.category == "PRE_REHEATING")].groupby("geometry_id").Om_C.sum()
    gpost = X[X.included_in_cdm_sum & (X.category == "POST_REHEATING")].groupby("geometry_id").Om_C.sum()
    Gi = G[lab].set_index("geometry_id")
    # geometries without CDM-eligible mode: Omega_geometry must be exactly 0 and C must find no included mode (kept apart from
    # C_full so that the reference-case comparison of C_full is unaffected when no such geometry exists)
    # geometries without included mode: Omega 0 (evaluated empty sum) or NA (a mode not evaluated by the inflation cut);
    # C must find no included mode for either (kept apart from C_full: unchanged when no such geometry exists)
    gz = gA[gA == 0].index; gn = gA[gA.isna()].index
    if len(gz) or len(gn):
        xi = Gi.number_inflation_excluded.astype(int)
        res.setdefault("geometries_without_included_mode", {})[lab] = {
            "Omega_zero": int(len(gz)), "Omega_not_evaluated_NA": int(len(gn)),
            "with_C_included_mode": int(gz.isin(gC.index).sum() + gn.isin(gC.index).sum()),
            "zero_with_excluded_mode": int((xi.loc[gz] > 0).sum()), "NA_without_excluded_mode": int((xi.loc[gn] == 0).sum())}
    gA = gA[(gA != 0) & gA.notna()]
    dis = X[(X.category != X.cat_C) | (X.inflation_valid != X.valid_C)]
    ties = [[g, int(i)] for g, i, lm in zip(dis.geometry_id, dis.mode_index, dis.log10_mass_eV) if is_tie(lm, HRc)]
    DIS[("C", lab)] = (len(dis), len(ties))
    if ties:
        res.setdefault("threshold_ties", {})[lab] = {"C_full": ties}
    res["C_full"][lab] = {
        "rows": len(X), "category_agree": int((X.category == X.cat_C).sum()), "validity_agree": int((X.inflation_valid == X.valid_C).sum()),
        "NA_pattern_agree": int(((aom.isna()) == X.Om_C.isna()).sum()),
        "POST_n": int(post.sum()), "POST_max_rel_Omega": mx(rel[post]), "POST_max_rel_Tosc": mx((X.T_osc_GeV / X.Tosc_C - 1).abs()[post]),
        "POST_max_rel_1pz": mx(((X.z_osc + 1) / X.opz_C - 1).abs()[post]),
        "LATE_n": int(late.sum()), "LATE_max_rel_Omega": mx(rel[late]),
        "PRE_valid_n": int(pre.sum()), "PRE_max_rel_Omega": mx(rel[pre]), "PRE_max_rel_1pz": mx(((X.z_osc + 1) / X.opz_C - 1).abs()[pre]),
        "geometries": int(len(gA)), "geometry_total_max_rel": mx((gA / gC.reindex(gA.index) - 1).abs()),
        "geometry_pre_max_rel": mx((Gi.Omega_pre_reheating_h2_at_theta_ref.dropna() / gpre.reindex(Gi.Omega_pre_reheating_h2_at_theta_ref.dropna().index) - 1).abs()),
        "geometry_post_max_rel": mx((Gi.Omega_post_reheating_h2_at_theta_ref.dropna() / gpost.reindex(Gi.Omega_post_reheating_h2_at_theta_ref.dropna().index) - 1).abs()),
    }

# ---------------- (b) B on a stratified sample and on complete geometries
R = Bm.Ref()
def b_eval(row, TR):
    b = R.evaluate(row.log10_mass_eV, row.log10_f_pert_GeV, TH, TR, HI3)
    return b
rng = np.random.default_rng(20261001)
hbin = lambda h: "4-10" if h <= 10 else "11-20" if h <= 20 else "21-30" if h <= 30 else "31-40" if h <= 40 else "41-50"
for s in SC:
    lab, TR = s["label"], s["T_reh_GeV"]; A = M[lab].assign(hb=M[lab].h11.map(hbin))
    idx = []
    for _, d in A.groupby(["abundance_state", "hb", "vacuum_status"]):
        idx += list(rng.choice(d.index.values, size=min(2, len(d)), replace=False))
    for st in ("INCLUDED_POST_REHEATING", "INCLUDED_PRE_REHEATING"):
        d = A[A.abundance_state == st]
        if d.empty:
            continue
        idx += [d.Omega_h2_at_theta_ref.idxmin(), d.Omega_h2_at_theta_ref.idxmax(), d.log10_mass_eV.idxmin(), d.log10_mass_eV.idxmax(),
                d.log10_f_pert_GeV.idxmin(), d.log10_f_pert_GeV.idxmax()]
    S = A.loc[sorted(set(idx))]
    rows = []
    for _, r in S.iterrows():
        b = b_eval(r, TR)
        aom = r.Omega_h2_at_theta_ref if r.category != "LATE" else r.Omega_h2_late_census_only_at_theta_ref
        rows.append({"scenario": lab, "geometry_id": r.geometry_id, "mode_index": int(r.mode_index), "state": r.abundance_state,
                     "cat_agree": b["category"] == r.category, "valid_agree": b["inflation_valid"] == bool(r.inflation_valid),
                     "rel_Omega": rd(aom, float(b["Omega_h2"])) if b["Omega_h2"] is not None else (0.0 if math.isnan(aom) else np.inf),
                     "rel_Tosc": rd(r.T_osc_GeV, float(b["T_osc_GeV"])) if b["T_osc_GeV"] is not None else (0.0 if math.isnan(r.T_osc_GeV) else np.inf),
                     "rel_1pz": rd(r.z_osc + 1, float(b["one_plus_z_osc"])) if b["one_plus_z_osc"] is not None else (0.0 if math.isnan(r.z_osc) else np.inf)})
    Sd = pd.DataFrame(rows, columns=["scenario", "geometry_id", "mode_index", "state", "cat_agree", "valid_agree", "rel_Omega", "rel_Tosc", "rel_1pz"])
    Sd = Sd.astype({"cat_agree": bool, "valid_agree": bool, "rel_Omega": float, "rel_Tosc": float, "rel_1pz": float}); Sd.to_csv(f"{V}/B_stratified_sample_{lab}.csv", index=False, float_format="%.6g")
    HRc_b = C.fuzzy_HR(tab, TR)
    bt = [[g, int(i)] for g, i, a_, lm in zip(Sd.geometry_id, Sd.mode_index, Sd.cat_agree & Sd.valid_agree,
                                              [A.loc[(A.geometry_id == g) & (A.mode_index == i), "log10_mass_eV"].iloc[0] for g, i in zip(Sd.geometry_id, Sd.mode_index)])
          if not a_ and is_tie(lm, HRc_b)]
    DIS[("B", lab)] = (int((~(Sd.cat_agree.astype(bool) & Sd.valid_agree.astype(bool))).sum()), len(bt))
    if bt:
        res.setdefault("threshold_ties", {}).setdefault(lab, {})["B_sample"] = bt
    res["B_sample"][lab] = {"rows": len(Sd), "strata_states": Sd.state.value_counts().to_dict(), "category_agree": int(Sd.cat_agree.sum()),
                            "validity_agree": int(Sd.valid_agree.sum()),
                            "POST_max_rel_Omega": mx(Sd[Sd.state == "INCLUDED_POST_REHEATING"].rel_Omega),
                            "PRE_max_rel_Omega": mx(Sd[Sd.state == "INCLUDED_PRE_REHEATING"].rel_Omega),
                            "LATE_max_rel_Omega": mx(Sd[Sd.state == "NOT_INCLUDED_LATE"].rel_Omega),
                            "max_rel_Tosc": mx(Sd.rel_Tosc), "max_rel_1pz": mx(Sd.rel_1pz),
                            "NA_pattern_ok": bool(np.isfinite(Sd.rel_Omega).all())}
    # complete geometries: two per h11 bin, one vacuum-corrected, one with inflation-excluded modes
    Gl = G[lab].assign(hb=G[lab].h11.map(hbin))
    gids = []
    for _, d in Gl.groupby("hb"):
        gids += list(rng.choice(d.geometry_id.values, size=min(2, len(d)), replace=False))
    gids += [d.geometry_id.iloc[0] for d in (Gl[Gl.vacuum_status == "VACUUM_CORRECTED"], Gl[Gl.number_inflation_excluded > 0]) if not d.empty]
    gr = []
    for gid in dict.fromkeys(gids):
        modes = A[A.geometry_id == gid]; tot = mp.mpf(0); pre = mp.mpf(0); post = mp.mpf(0)
        for _, r in modes.iterrows():
            b = b_eval(r, TR)
            if b["Omega_h2"] is not None and b["category"] in ("POST_REHEATING", "PRE_REHEATING"):
                tot += b["Omega_h2"]; pre += b["Omega_h2"] if b["category"] == "PRE_REHEATING" else 0; post += b["Omega_h2"] if b["category"] == "POST_REHEATING" else 0
        g = Gl.set_index("geometry_id").loc[gid]
        og = g.Omega_geometry_h2_at_theta_ref           # no included mode: 0 (evaluated) or NA (a mode not evaluated)
        rel_total = rd(og, float(tot)) if tot > 0 else (0.0 if og == 0 or (math.isnan(og) and int(g.number_inflation_excluded) > 0) else np.inf)
        gr.append({"geometry_id": gid, "modes": len(modes), "rel_total": rel_total,
                   "rel_pre": rd(g.Omega_pre_reheating_h2_at_theta_ref, float(pre)) if pre > 0 else 0.0,
                   "rel_post": rd(g.Omega_post_reheating_h2_at_theta_ref, float(post)) if post > 0 else 0.0})
    gd = pd.DataFrame(gr)
    res["B_geometries"][lab] = {"geometries": len(gd), "modes": int(gd.modes.sum()), "max_rel_total": float(gd.rel_total.max()),
                                "max_rel_pre": float(gd.rel_pre.max()), "max_rel_post": float(gd.rel_post.max()), "ids": gd.geometry_id.tolist()}

# ---------------- verdict
TOL = {"POST": 1e-10, "PRE": 1e-12, "geometry": 1e-10}
chk = {}
for lab, v in res["C_full"].items():
    ndis, nt = DIS[("C", lab)]                    # every category/validity disagreement must be a listed threshold tie
    chk[f"C_{lab}_categories"] = ndis == nt and v["NA_pattern_agree"] == v["rows"]
    chk[f"C_{lab}_POST"] = v["POST_max_rel_Omega"] <= TOL["POST"] and v["POST_max_rel_Tosc"] <= TOL["POST"] and v["LATE_max_rel_Omega"] <= TOL["POST"]
    chk[f"C_{lab}_PRE"] = v["PRE_max_rel_Omega"] <= TOL["PRE"]
    chk[f"C_{lab}_geometry_sums"] = max(v["geometry_total_max_rel"], v["geometry_pre_max_rel"], v["geometry_post_max_rel"]) <= TOL["geometry"]
for lab, v in res.get("geometries_without_included_mode", {}).items():
    chk[f"C_{lab}_zero_or_NA_geometries_have_no_included_mode"] = v["with_C_included_mode"] == 0 and v["zero_with_excluded_mode"] == 0 and v["NA_without_excluded_mode"] == 0
for lab, v in res["B_sample"].items():
    ndis, nt = DIS[("B", lab)]
    chk[f"B_{lab}_sample"] = ndis == nt and v["NA_pattern_ok"] and \
        v["POST_max_rel_Omega"] <= TOL["POST"] and v["PRE_max_rel_Omega"] <= TOL["PRE"] and v["LATE_max_rel_Omega"] <= TOL["POST"]
    chk[f"B_{lab}_geometries"] = res["B_geometries"][lab]["max_rel_total"] <= TOL["geometry"]
res["tolerances"] = TOL; res["checks"] = chk; res["all_pass"] = all(chk.values())
dump_json(res, f"{V}/production_verification.json")
print(json.dumps(res, indent=1))
sys.exit(0 if res["all_pass"] else 1)
