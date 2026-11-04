"""Validate production-format outputs against config/output_schema.json and the methodology invariants.

Used now on the benchmark outputs (schema test of the eventual production run) and, unchanged, later on the production
outputs. Reads every cell as text so that NA, booleans and numbers are checked exactly as written.
Checks: column names/order, types, enums, the NA rule of every column, the state/inclusion/NA invariants, per-geometry
aggregates recomputed from the master table, per-h11 and global summaries recomputed from the per-geometry table,
identical mode sets across scenarios, and flags/m_KK identical to the A1 canonical records for real modes.
Usage: SHOROQ_RELIC_CONFIG=RUN_CONFIG python 06_validate_outputs.py OUTPUT_DIR [EXPECTED_MODES]
"""
import json, math, os, re, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SCHEMA, canonical_modes, check_run_location, dump_json, load_config, methodology_version, scenario_w_R, schema_path

OUT = os.path.abspath(sys.argv[1]); EXPECTED = int(sys.argv[2]) if len(sys.argv) > 2 else None
cfg = load_config(); check_run_location(cfg, os.path.dirname(OUT)); MV = methodology_version(cfg); sch = json.load(open(schema_path(cfg))); NA = sch["null_token"]
LABS = [s["label"] for s in cfg["reheating"]["scenarios"]]; TREH = {s["label"]: s["T_reh_GeV"] for s in cfg["reheating"]["scenarios"]}
WR = {s["label"]: scenario_w_R(cfg, s) for s in cfg["reheating"]["scenarios"]}      # 0.0 for every scenario in v1
errors = []
def err(msg):
    if len(errors) < 200:
        errors.append(msg)

isna = lambda v: v == NA
f = lambda v: float(v)
b = lambda v: v == "true"
RULES = {   # schema null_when text -> predicate on the (text) row; unknown text is an error (keeps schema and validator in sync)
    "never": lambda r: False,
    "category not in {LATE, POST_REHEATING} or not inflation_valid": lambda r: r["category"] not in ("LATE", "POST_REHEATING") or not b(r["inflation_valid"]),
    "category == FROZEN or not inflation_valid": lambda r: r["category"] == "FROZEN" or not b(r["inflation_valid"]),
    "not included_in_cdm_sum": lambda r: not b(r["included_in_cdm_sum"]),
    "category != LATE or not inflation_valid": lambda r: r["category"] != "LATE" or not b(r["inflation_valid"]),
    "inflation_valid": lambda r: b(r["inflation_valid"]),
    "number_included == 0": lambda r: int(r["number_included"]) == 0,
    "no included POST_REHEATING mode": lambda r: int(r["number_post_reheating"]) == 0,
    "no included PRE_REHEATING mode": lambda r: int(r["number_pre_reheating_valid"]) == 0,
    "number_late == 0": lambda r: int(r["number_late"]) == 0,
    "number_included_above_mKK == 0": lambda r: int(r["number_included_above_mKK"]) == 0,
    "number_included_above_fpert == 0": lambda r: int(r["number_included_above_fpert"]) == 0,
    "geometries_with_included == 0": lambda r: int(r["geometries_with_included"]) == 0,
    "number_included == 0 and number_inflation_excluded > 0": lambda r: int(r["number_included"]) == 0 and int(r["number_inflation_excluded"]) > 0,
    "abundance_validity == EVALUATED_POSITIVE": lambda r: r["abundance_validity"] == "EVALUATED_POSITIVE",
    "abundance_validity not in {EVALUATED_POSITIVE, EVALUATED_ZERO}": lambda r: r["abundance_validity"] not in ("EVALUATED_POSITIVE", "EVALUATED_ZERO"),
    "abundance_validity != PARTIAL_VALIDITY": lambda r: r["abundance_validity"] != "PARTIAL_VALIDITY",
}


def check_table(path, spec, lab):
    T = pd.read_csv(path, dtype=str, keep_default_na=False)
    names = [c["name"] for c in spec]
    if list(T.columns) != names:
        err(f"{path}: columns {list(T.columns)} != schema")
        return T
    for c in spec:
        rule = RULES.get(c["null_when"])
        if rule is None:
            err(f"schema null_when not implemented: {c['null_when']}"); continue
        for i, r in T.iterrows():
            v = r[c["name"]]; should_na = rule(r)
            if isna(v) != should_na:
                err(f"{os.path.basename(path)} row {i} {c['name']}: NA={isna(v)} expected NA={should_na} (state {r.get('abundance_state', '')})"); continue
            if isna(v):
                continue
            t = c["type"]
            if t == "int" and not re.fullmatch(r"-?\d+", v):
                err(f"{path} row {i} {c['name']}: not int {v}")
            elif t == "float":
                try:
                    x = float(v)
                    if not math.isfinite(x):
                        err(f"{path} row {i} {c['name']}: non-finite")
                except ValueError:
                    err(f"{path} row {i} {c['name']}: not float {v}")
            elif t == "bool" and v not in sch["bool_tokens"]:
                err(f"{path} row {i} {c['name']}: not bool {v}")
            elif t.startswith("enum:") and v not in t[5:].split("|"):
                err(f"{path} row {i} {c['name']}: {v} not in enum")
    return T


res = {"output_dir": OUT, "tables": {}}
masters = {}
A1 = pd.read_csv(canonical_modes(), float_precision="round_trip", keep_default_na=False,
                 dtype={"log10_mass_eV": float, "log10_f_pert_GeV": float, "m_KK_GeV": float}).set_index(["geometry_id", "mode_index"])
# the geometry population of the run: every usable geometry of the A1 canonical geometry table, also without any mode
A1G = pd.read_csv(cfg["input"]["canonical_geometry_table"], keep_default_na=False, dtype=str) if "canonical_geometry_table" in cfg["input"] else None
for lab in LABS:
    M = check_table(f"{OUT}/master_modes_{lab}.csv", sch["tables"]["master_modes_<label>.csv"], lab)
    G = check_table(f"{OUT}/geometry_summary_{lab}.csv", sch["tables"]["geometry_summary_<label>.csv"], lab)
    H = check_table(f"{OUT}/h11_summary_{lab}.csv", sch["tables"]["h11_summary_<label>.csv"], lab)
    E = check_table(f"{OUT}/geometry_abundance_validity_{lab}.csv", sch["tables"]["geometry_abundance_validity_<label>.csv"], lab)
    masters[lab] = M
    st, inc, val, cat = M.abundance_state, M.included_in_cdm_sum.map(b).astype(bool), M.inflation_valid.map(b).astype(bool), M.category   # bool also when empty
    om = M.Omega_h2_at_theta_ref.map(lambda v: np.nan if isna(v) else float(v)).astype(float)
    inv = {
        "included_iff_state_INCLUDED": bool((inc == st.str.startswith("INCLUDED_")).all()),
        "included_iff_Omega_present": bool((inc == om.notna()).all()),
        "Omega_positive_finite": bool((om.dropna() > 0).all() and np.isfinite(om.dropna()).all()),
        "excluded_iff_not_valid": bool(((st == "INFLATIONARY_VALIDITY_EXCLUDED") == ~val).all()),
        "FROZEN_state": bool((st[cat == "FROZEN"] == "NOT_INCLUDED_FROZEN").all()),
        "LATE_state": bool((st[(cat == "LATE") & val] == "NOT_INCLUDED_LATE").all()),
        "PRE_T_osc_NA": bool((M.T_osc_GeV[cat == "PRE_REHEATING"] == NA).all()),
        "branch_matches_state": bool((M.abundance_branch == st.map({"INCLUDED_POST_REHEATING": "post_reheating_full_H", "INCLUDED_PRE_REHEATING": "fuzzy_wR0_eq2.10_2.15" if WR[lab] == 0 else "fuzzy_wR_general_eq2.10_2.15",
                                       "NOT_INCLUDED_LATE": "late_full_H_census_only", "NOT_INCLUDED_FROZEN": "frozen_not_oscillating",
                                       "INFLATIONARY_VALIDITY_EXCLUDED": "not_evaluated_inflation_invalid"})).all()),
        "scenario_and_T_reh_constant": bool((M.scenario == lab).all() and (M.T_reh_GeV.astype(float) == TREH[lab]).all()),
        "theta_pi_over_2": bool((M.theta_ref.astype(float) == math.pi / 2).all()),
        "mass_and_f_consistent_with_logs": bool(np.allclose(M.mass_eV.astype(float), 10 ** M.log10_mass_eV.astype(float), rtol=1e-15, atol=0) and
                                                np.allclose(M.f_pert_GeV.astype(float), 10 ** M.log10_f_pert_GeV.astype(float), rtol=1e-15, atol=0)),
        "no_zero_written_for_excluded": bool((M.Omega_h2_at_theta_ref[st == "INFLATIONARY_VALIDITY_EXCLUDED"] == NA).all()),
    }
    if MV == 2:        # w_R identity of the scenario in every table, in range, and in the summaries (v2 schema)
        wcol = lambda T: T["w_R"].astype(float)
        inv["w_R_equals_scenario_in_all_four_tables"] = bool(all((wcol(T) == WR[lab]).all() for T in (M, G, H, E)))
        inv["w_R_in_supported_domain"] = bool(-1 < WR[lab] <= 1)
        gsj = json.load(open(f"{OUT}/global_summary.json"))[lab]
        inv["w_R_in_global_summary"] = gsj.get("w_R") == WR[lab]
    # flags identical to A1 for real modes
    real = M[M.vacuum_status != "SYNTHETIC_BENCHMARK"]
    a = A1.loc[list(zip(real.geometry_id, real.mode_index.astype(int)))]
    inv["flags_identical_to_A1"] = bool((real.inflation_valid.map(b).values == a.inflation_valid.values).all() and
                                        (real.above_mKK.map(b).values == a.above_mKK.values).all() and (real.above_fpert.map(b).values == a.above_fpert.values).all())
    inv["mKK_identical_to_A1_rel_1e-12"] = bool(np.allclose(real.m_KK_GeV.astype(float).values, a.m_KK_GeV.values, rtol=1e-12, atol=0))
    inv["log10_mass_and_f_bit_identical_to_A1"] = bool((real.log10_mass_eV.astype(float).values == a.log10_mass_eV.values).all() and
                                                       (real.log10_f_pert_GeV.astype(float).values == a.log10_f_pert_GeV.values).all())
    # per-geometry recomputation
    Mg = M.assign(om=om, inc=inc, val=val, mKK=M.above_mKK.map(b), fp=M.above_fpert.map(b))
    agg = Mg.groupby("geometry_id", sort=False).agg(n=("mode_index", "size"), fr=("category", lambda s: int((s == "FROZEN").sum())),
                                                    la=("category", lambda s: int((s == "LATE").sum())), po=("category", lambda s: int((s == "POST_REHEATING").sum())),
                                                    pr=("category", lambda s: int((s == "PRE_REHEATING").sum())), ex=("val", lambda s: int((~s).sum())),
                                                    ni=("inc", "sum"), tot=("om", lambda s: s.sum(min_count=1)))
    Gi = G.set_index("geometry_id")
    gcmp = agg.join(Gi, how="outer")
    tot_g = gcmp.Omega_geometry_h2_at_theta_ref.map(lambda v: np.nan if isna(v) else float(v))
    zero_mode = Gi.number_of_modes.astype(int) == 0
    inv["geometry_set_equals_master_plus_zero_mode_geometries"] = len(gcmp) == len(Gi) and set(agg.index) == set(Gi.index[~zero_mode])
    if A1G is not None:
        inv["geometry_population_equals_A1"] = bool(list(Gi.index) == list(A1G.geometry_id) and
                                                    (Gi.number_of_modes.astype(int).values == A1G.n_modes.astype(int).values).all())
    gcmp = gcmp[gcmp.n.notna()]                   # geometries with modes (zero-mode rows are checked below)
    tot_g = tot_g[gcmp.index]
    inv["geometry_counts_recomputed"] = bool((gcmp.n == gcmp.number_of_modes.astype(int)).all() and (gcmp.fr == gcmp.number_frozen.astype(int)).all() and
        (gcmp.la == gcmp.number_late.astype(int)).all() and (gcmp.po == gcmp.number_post_reheating.astype(int)).all() and
        (gcmp.pr == gcmp.number_pre_reheating.astype(int)).all() and (gcmp.ex == gcmp.number_inflation_excluded.astype(int)).all() and
        (gcmp.ni == gcmp.number_included.astype(int)).all())
    inv["geometry_counts_sum_to_modes"] = bool((Gi.number_frozen.astype(int) + Gi.number_late.astype(int) + Gi.number_post_reheating.astype(int) +
                                                Gi.number_pre_reheating.astype(int) == Gi.number_of_modes.astype(int)).all())
    inv["geometry_Omega_at_theta_ref_recomputed_rel_1e-14"] = bool(((gcmp.tot.isna() & ((tot_g == 0) | tot_g.isna())) | (abs(tot_g / gcmp.tot - 1) <= 1e-14)).all())
    # Omega_geometry: > 0 iff a mode is included; 0 iff nothing included and nothing excluded (evaluated empty sum);
    # NA iff nothing included and a mode is not evaluated (inflation cut)
    gom = Gi.Omega_geometry_h2_at_theta_ref.map(lambda v: np.nan if isna(v) else float(v)); ni = Gi.number_included.astype(int)
    nx = Gi.number_inflation_excluded.astype(int)
    inv["Omega_geometry_positive_zero_NA_rule"] = bool(((gom > 0) == (ni > 0)).all() and ((gom == 0) == ((ni == 0) & (nx == 0))).all()
                                                       and (gom.isna() == ((ni == 0) & (nx > 0))).all())
    zr = Gi[zero_mode]
    inv["zero_mode_geometries_counts_zero"] = bool(all((zr[c].astype(int) == 0).all() for c in ("number_frozen", "number_late", "number_post_reheating",
        "number_pre_reheating", "number_inflation_excluded", "number_included", "number_above_mKK", "number_above_fpert")))
    Ei = E.set_index("geometry_id")
    inv["validity_rows_equal_geometry_rows"] = bool(list(Ei.index) == list(Gi.index) and
        all((Ei[c].values == Gi[c].values).all() for c in ("h11", "number_of_modes", "number_included", "number_inflation_excluded")))
    nm = Gi.number_of_modes.astype(int)
    cls = np.where(nx == 0, np.where(ni > 0, "EVALUATED_POSITIVE", "EVALUATED_ZERO"), np.where(nx == nm, "INFLATION_INVALID_ALL", "PARTIAL_VALIDITY"))
    rsn = np.where(cls == "EVALUATED_POSITIVE", NA, np.where(cls == "EVALUATED_ZERO", np.where(nm == 0, "NO_MODES_ABOVE_THRESHOLD", "ALL_MODES_FROZEN_OR_LATE"),
                   np.where(cls == "PARTIAL_VALIDITY", "SOME_MODES_INFLATION_EXCLUDED", "ALL_MODES_INFLATION_EXCLUDED")))
    pop = np.where(np.char.startswith(cls.astype(str), "EVALUATED_"), "PRIMARY", np.where(cls == "PARTIAL_VALIDITY", "SENSITIVITY_ONLY", "EXCLUDED_NOT_EVALUATED"))
    gtxt = Gi.Omega_geometry_h2_at_theta_ref.values
    comp = np.where(np.char.startswith(cls.astype(str), "EVALUATED_"), gtxt, NA)
    part = np.where(cls == "PARTIAL_VALIDITY", np.where(gtxt == NA, "0", gtxt), NA)
    inv["validity_class_reason_population_recomputed"] = bool((Ei.abundance_validity.values == cls).all() and (Ei.reason.values == rsn).all() and
                                                              (Ei.theta_population.values == pop).all())
    inv["validity_complete_and_partial_Omega_text_identical_to_geometry_table"] = bool((Ei.Omega_complete_h2_at_theta_ref.values == comp).all() and
                                                                                       (Ei.Omega_valid_modes_partial_h2_at_theta_ref.values == part).all())
    # per-h11 and global from geometry table
    hh = Gi.reset_index().groupby(Gi.h11.astype(int).values).agg(geometries=("geometry_id", "size"), modes=("number_of_modes", lambda s: s.astype(int).sum()),
                                                               inc=("number_included", lambda s: s.astype(int).sum()))
    Hh = H.set_index(H.h11.astype(int))
    inv["h11_summary_recomputed"] = bool((hh.geometries == Hh.geometries.astype(int)).all() and (hh.modes == Hh.modes.astype(int)).all() and (hh.inc == Hh.n_included.astype(int)).all())
    gs = json.load(open(f"{OUT}/global_summary.json"))[lab]["census"]
    inv["global_summary_recomputed"] = gs["modes"] == len(M) and gs["geometries"] == len(G) and gs["included_in_cdm_sum"] == int(inc.sum()) and \
        all(gs["by_category"][k] == int((cat == k).sum()) for k in sch["categories"]) and all(gs["by_abundance_state"][k] == int((st == k).sum()) for k in sch["abundance_states"])
    if EXPECTED is not None:
        inv["row_count_expected"] = len(M) == EXPECTED
    res["tables"][lab] = {"rows_master": len(M), "rows_geometry": len(G), "rows_h11": len(H), "invariants": inv}
keys = {lab: set(zip(masters[lab].geometry_id, masters[lab].mode_index)) for lab in LABS}
res["same_modes_in_every_scenario"] = all(keys[l] == keys[LABS[0]] for l in LABS)
res["errors"] = errors; res["n_errors"] = len(errors)
res["all_pass"] = not errors and res["same_modes_in_every_scenario"] and all(all(v["invariants"].values()) for v in res["tables"].values())
dump_json(res, f"{OUT}/output_validation.json")
print(json.dumps({k: v for k, v in res.items() if k != "errors"}, indent=1)); print("errors:", errors[:10])
sys.exit(0 if res["all_pass"] else 1)
