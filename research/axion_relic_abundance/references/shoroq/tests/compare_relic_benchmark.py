"""Restored part (c) of the relic verification: the pre-production benchmark (real + synthetic boundary modes).

  1. The relic script of this repository, rerun on the frozen benchmark mode table (BENCH_RERUN), reproduces the frozen
     benchmark output (FROZEN_BENCH/out_A_julia) byte for byte (all tables; run_metadata only in its scientific fields).
     One documented change (2026-10-30/31, approved): a geometry without included mode and without inflation-excluded mode
     has Omega_geometry = 0 instead of NA; the benchmark's synthetic pseudo-geometries with a single FROZEN/LATE mode carry it. The frozen geometry summaries are
     compared after exactly this cell mapping; every other byte must be identical (count reported).
  2. The production rows of the benchmark geometries (PROD_TABLES) equal the benchmark rows (same code, same inputs),
     and agree with the 50-digit implementation B to 1e-9 (B was stored with 10 significant digits).
  3. The full-precision A-vs-B agreement of the benchmark (benchmark_summary.json) is within the production tolerances.
Usage: python compare_relic_benchmark.py BENCH_RERUN FROZEN_BENCH FROZEN_BENCH_SUMMARY_JSON PROD_TABLES OUT_JSON
"""
import filecmp, json, os, sys

import pandas as pd

rerun, fb, fsum, prod, out = sys.argv[1:6]
A = os.path.join(fb, "out_A_julia")
LABS = ["5MeV", "1e4GeV", "1e10GeV"]
res = {"rerun_identical": {}, "production_vs_benchmark": {}}
files = [f"{k}_{l}.csv" for k in ("master_modes", "geometry_summary", "h11_summary") for l in LABS] + ["global_summary.json"]


def zero_omega_mapped(path):
    """Frozen geometry summary with the one documented 2026-10-30/31 change applied: a geometry without included mode and
    without inflation-excluded mode (an evaluated empty sum, EVALUATED_ZERO) has Omega_geometry_h2_at_theta_ref = 0 (frozen: NA);
    with an inflation-excluded mode it stays NA (not evaluated). Every other byte must stay identical."""
    L = open(path).read().split("\n"); h = L[0].split(","); io, ii = h.index("Omega_geometry_h2_at_theta_ref"), h.index("number_included")
    ix = h.index("number_inflation_excluded")
    n = 0
    for k in range(1, len(L)):
        x = L[k].split(",")
        if len(x) == len(h) and x[ii] == "0" and x[ix] == "0" and x[io] == "NA":
            x[io] = "0"; L[k] = ",".join(x); n += 1
    return "\n".join(L), n


res["zero_omega_rows_mapped"] = {}
for f in files:
    if f.startswith("geometry_summary_"):
        txt, n = zero_omega_mapped(os.path.join(A, f))
        res["rerun_identical"][f] = txt == open(os.path.join(rerun, f)).read(); res["zero_omega_rows_mapped"][f] = n
    else:
        res["rerun_identical"][f] = filecmp.cmp(os.path.join(A, f), os.path.join(rerun, f), shallow=False)
ma, mb = json.load(open(os.path.join(A, "run_metadata.json"))), json.load(open(os.path.join(rerun, "run_metadata.json")))
sci = ("rows", "is_full_population", "constants_checked_against_config", "derived_checked_against_config", "scenarios", "cosmology", "methodology",
       "gstar_table_sha256", "mode_table_sha256")
res["rerun_identical"]["run_metadata.json (scientific fields)"] = all(ma.get(k) == mb.get(k) for k in sci)

TW = pd.read_csv(os.path.join(fb, "benchmark_three_way.csv"))
for lab in LABS:
    Bp = pd.read_csv(os.path.join(A, f"master_modes_{lab}.csv"), dtype=str, keep_default_na=False)
    Pp = pd.read_csv(os.path.join(prod, f"master_modes_{lab}.csv"), dtype=str, keep_default_na=False)
    Bp = Bp[Bp.vacuum_status != "SYNTHETIC_BENCHMARK"]
    J = Bp.merge(Pp, on=["geometry_id", "mode_index"], suffixes=("_b", "_p"))
    cols = [c for c in Bp.columns if c not in ("geometry_id", "mode_index")]
    ident = all((J[f"{c}_b"] == J[f"{c}_p"]).all() for c in cols)
    Bg = pd.read_csv(os.path.join(A, f"geometry_summary_{lab}.csv"), dtype=str, keep_default_na=False)
    Pg = pd.read_csv(os.path.join(prod, f"geometry_summary_{lab}.csv"), dtype=str, keep_default_na=False)
    Bg = Bg[Bg.vacuum_status != "SYNTHETIC_BENCHMARK"]
    Jg = Bg.merge(Pg, on="geometry_id", suffixes=("_b", "_p"))
    gident = all((Jg[f"{c}_b"] == Jg[f"{c}_p"]).all() for c in Bg.columns if c != "geometry_id")
    M = pd.read_csv(os.path.join(prod, f"master_modes_{lab}.csv"), float_precision="round_trip", na_values=["NA"], keep_default_na=False)
    tw = TW[(TW.scenario == lab) & (TW.benchmark_kind == "real_full_geometry")].merge(
        M[["geometry_id", "mode_index", "Omega_h2_at_theta_ref", "Omega_h2_late_census_only_at_theta_ref", "category"]], on=["geometry_id", "mode_index"])
    pom = tw.Omega_h2_at_theta_ref.where(tw.category != "LATE", tw.Omega_h2_late_census_only_at_theta_ref)
    res["production_vs_benchmark"][lab] = {"benchmark_real_modes": len(Bp), "matched": len(J), "all_columns_bit_identical": bool(ident and len(J) == len(Bp)),
                                           "benchmark_geometries": len(Bg), "geometry_rows_bit_identical": bool(gident and len(Jg) == len(Bg)),
                                           "max_rel_vs_B_10_significant_digits": float((pom / tw.Omega_B - 1).abs().max())}
BS = json.load(open(fsum))
res["full_precision_A_vs_B"] = {"POST": BS["POST"]["max_rel_Omega_A_B"], "PRE": BS["PRE_valid"]["max_rel_Omega_A_B_eq210_215"],
                                "benchmark_all_checks_pass": BS["all_checks_pass"]}
TOL = {"POST": 1e-10, "PRE": 1e-12}
chk = {f"rerun_{k}": v for k, v in res["rerun_identical"].items()}
for lab, v in res["production_vs_benchmark"].items():
    chk[f"benchmark_{lab}_identical"] = v["all_columns_bit_identical"] and v["geometry_rows_bit_identical"] and v["max_rel_vs_B_10_significant_digits"] <= 1e-9
chk["full_precision_A_vs_B"] = res["full_precision_A_vs_B"]["POST"] <= TOL["POST"] and res["full_precision_A_vs_B"]["PRE"] <= TOL["PRE"] and BS["all_checks_pass"]
res["checks"] = chk; res["all_pass"] = all(chk.values())
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if res["all_pass"] else 1)
