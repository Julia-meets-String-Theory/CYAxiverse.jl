"""A1 input adapter: relic_abundance_input (manifest + spectra/) -> canonical per-mode and per-geometry records.

Canonical identifiers are the manifest geometry_id and the stored mode_index; file-internal polytope labels are read
only to document the known label anomaly and are never used as keys. No cosmology is evaluated here.
Per mode it attaches: mass (log10 eV and eV), f_relic = f_pert (C = 1), theta_reference = pi/2, vacuum status, the
inflationary validity flag (Fuzzy footnote 16, per mode), and the diagnostic flags above_mKK / above_fpert with ratios.
Every check result is recorded in OUT_DIR/a1_input_verification.json; the script exits non-zero if any check fails.
General checks (every sample): per-file spectrum contract, unique geometry and mode keys, per-h11 and total counts equal
the manifest, vacuum validation usable for every geometry kept. Population checks against expected counts (geometries,
modes, statuses, h11 range, flag counts) run only when the template carries expectations (a reference case).
Vacuum validation (per geometry, manifest field vacuum_validation.state; see DATA_CONTRACTS.md):
  VERIFIED_THETA0_INERTIA  theta = 0 inertia scan of this tool: no tachyonic direction with |m| >= H0  (ORIGINAL_SPECTRUM)
  VERIFIED_VACUUM_REPAIR   spectrum about a minimum validated by the vacuum-repair producer           (VACUUM_CORRECTED)
  DOCUMENTED_EXTERNAL      external evidence (method, reference, evidence files re-hashed here); not re-verified
  TACHYONIC_AT_THETA0, UNVERIFIED  not usable: the geometry must be excluded (manifest vacuum_policy.on_unusable =
                           "exclude", status EXCLUDED_*) or the run stops
Manifests written by the vacuum-repair producer (producers/vacuum/05_build_relic_input.py) carry no field: their
state follows from the status (that producer runs the inertia scan on every geometry).
Usage: python 01_a1_input_adapter.py CONFIG_TEMPLATE RELIC_INPUT_DIR OUT_DIR [--expect-manifest-sha256 SHA]
  CONFIG_TEMPLATE   run template from 00_compose_run_config.py (constants, validity rules, optional expected counts)
  RELIC_INPUT_DIR   output of producers/vacuum/05_build_relic_input.py (manifest, SHA256SUMS.txt, spectra/)
  OUT_DIR           receives canonical_modes.csv, canonical_geometries.csv, a1_input_verification.json
"""
import argparse, hashlib, json, math, os, sys
from collections import Counter

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump_json, sha256

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("input_dir"); ap.add_argument("out_dir")
ap.add_argument("--expect-manifest-sha256", default=None, help="optional pin of the relic-input manifest (regression runs)")
args = ap.parse_args()
cfg = json.load(open(args.config))
I = cfg["input"]
input_dir = os.path.abspath(args.input_dir)
man_path = os.path.join(input_dir, "relic_abundance_input_manifest.json")
os.makedirs(args.out_dir, exist_ok=True)
CANONICAL_MODES = os.path.join(args.out_dir, "canonical_modes.csv")
CANONICAL_GEOMS = os.path.join(args.out_dir, "canonical_geometries.csv")
manifest_sha = sha256(man_path)
if args.expect_manifest_sha256 is not None:
    assert manifest_sha == args.expect_manifest_sha256, "manifest checksum mismatch"
for line in open(os.path.join(input_dir, "SHA256SUMS.txt")):                   # the input directory's own checksums
    s, name = line.split()
    assert sha256(os.path.join(input_dir, name)) == s, f"input file changed: {name}"
M = json.load(open(man_path))
# spectrum_source is absolute in the frozen 2026-09-30 manifest and relative to M["spectrum_source_base"] (itself relative
# to the manifest directory) in manifests written by the clean pipeline
SRC_BASE = os.path.normpath(os.path.join(input_dir, M.get("spectrum_source_base", "")))
resolve = lambda p: p if os.path.isabs(p) else os.path.join(SRC_BASE, p)

THR = I["spectrum_settings"]["threshold_log10_eV"]
TH = cfg["axion"]["theta_reference"]
V = cfg["validity"]
L3HI = math.log10(V["inflation_cut"]["three_H_I_max_GeV"])                       # log10 GeV
HI3 = V["inflation_cut"]["three_H_I_max_GeV"]
MPL_KK = V["mKK_flag"]["M_pl_GeV"]
fail, notes = [], []
modes, geoms, vac_rows, excluded = [], [], [], []
status_count = Counter()
PRODUCER = "vacuum_correction" in M and not any("vacuum_validation" in r for r in M["geometries"])
USABLE = {"ORIGINAL_SPECTRUM": {"VERIFIED_THETA0_INERTIA", "DOCUMENTED_EXTERNAL"},
          "VACUUM_CORRECTED": {"VERIFIED_VACUUM_REPAIR", "DOCUMENTED_EXTERNAL"}}
EXCLUDE_OK = M.get("vacuum_policy", {}).get("on_unusable") == "exclude"


def vacuum_validation(r):
    """(state, method, reference, evidence files, problem) of one manifest record"""
    v = r.get("vacuum_validation")
    if v is None:
        if PRODUCER:
            st = {"ORIGINAL_SPECTRUM": "VERIFIED_THETA0_INERTIA", "VACUUM_CORRECTED": "VERIFIED_VACUUM_REPAIR"}.get(r["status"], "TACHYONIC_AT_THETA0")
            return st, "vacuum-repair producer (theta = 0 inertia scan of every geometry)", r.get("vacuum_source", ""), [], None
        return "UNVERIFIED", "", "", [], None
    st, ev, prob = v.get("state"), v.get("evidence", []), None
    if st == "DOCUMENTED_EXTERNAL":
        if not (v.get("method") and v.get("reference") and ev):
            prob = "DOCUMENTED_EXTERNAL needs method, reference and at least one evidence file"
        for e in ev:
            p = e["path"] if os.path.isabs(e["path"]) else os.path.join(input_dir, e["path"])
            if not os.path.isfile(p) or sha256(p) != str(e.get("sha256", "")).lower():
                prob = f"evidence file missing or changed: {e['path']}"
    return st, v.get("method", ""), v.get("reference", ""), [e["path"] for e in ev], prob


META_KEYS = ("h11", "polytope", "frst", "prec", "quartics", "threshold_log10", "status", "terminal_status", "provisional", "schema_version")
ARRAYS = {"m": "f", "mode_indices": "i", "fpert_log10": "f", "lambda_self_log10": "f", "lambda_self_sign": "i"}


def read_spectrum(path):
    """((m, mode_indices, fpert, lambda_self_log10, lambda_self_sign, metadata, CY volume, h11, has vacuum group), None) or
    (None, (check name, detail)) when the file is unreadable, a dataset is missing or an array has a non-numeric dtype"""
    try:
        f = h5py.File(path, "r")
    except OSError as e:
        return None, ("hdf5_unreadable", str(e)[:160])
    with f:
        for k in [f"spectrum/physical/{a}" for a in ARRAYS] + [f"spectrum/physical/metadata/{k}" for k in META_KEYS] + \
                 ["cytools/geometric/CY_volume", "cytools/geometric/h11"]:
            if k not in f:
                return None, ("missing_dataset", k)
        p = f["spectrum/physical"]; md = p["metadata"]
        for a, kind in ARRAYS.items():                # numeric storage only (no strings that numpy would silently convert)
            dk = p[a].dtype.kind
            if not (dk in "iu" or (kind == "f" and dk == "f")):
                return None, ("dataset_dtype", f"{a}: {p[a].dtype}")
        for a in ARRAYS:                              # one-dimensional arrays only (a 2-D or scalar dataset would be misread)
            if p[a].ndim != 1:
                return None, ("dataset_shape", f"{a}: expected a 1-D array, got shape {p[a].shape}")
        g = lambda k: (md[k][()].decode() if isinstance(md[k][()], bytes) else md[k][()]) if k in md else None
        meta = {k: g(k) for k in META_KEYS + ("evaluation_point",)}
        arrs = [np.asarray(p["m"][()], float), np.asarray(p["mode_indices"][()], int), np.asarray(p["fpert_log10"][()], float),
                np.asarray(p["lambda_self_log10"][()], float), np.asarray(p["lambda_self_sign"][()], int)]
        return (*arrs, meta, float(f["cytools/geometric/CY_volume"][()]), int(f["cytools/geometric/h11"][()]), "vacuum_correction" in f), None


for r in M["geometries"]:
    gid, h, pos, st = r["geometry_id"], r["h11"], r["position"], r["status"]
    status_count[st] += 1
    vst, vmeth, vref, vev, vprob = vacuum_validation(r)
    dg = (r.get("vacuum_validation") or {}).get("diagnostics", {})
    vac_rows.append({"geometry_id": gid, "status": st, "vacuum_validation": vst, "method": vmeth, "reference": vref, "evidence": ";".join(vev),
                     "subH0_inertia_resolved": dg.get("subH0_inertia_resolved", "NA"), "inertia_negative_full_hessian": dg.get("inertia_negative_full_hessian", "NA"),
                     "pq_eig_dynamic_range_dex": dg.get("pq_eig_dynamic_range_dex", "NA")})
    if vprob:
        fail.append((gid, "vacuum_validation_evidence", vprob))
    if st.startswith("EXCLUDED_"):
        excluded.append(gid)
        if not EXCLUDE_OK:
            fail.append((gid, "excluded_geometry_present", "manifest vacuum_policy.on_unusable is not 'exclude'"))
        continue
    if vst not in USABLE.get(st, ()):
        fail.append((gid, "vacuum_validation_not_usable", f"{st}/{vst}"))
        continue
    src = r["spectrum_source"]
    view = os.path.join(input_dir, "spectra", gid, "cyax.h5")
    if gid != f"h11_{h:03d}/np_{pos:07d}/cy_{r['frst']:07d}":
        fail.append((gid, "geometry_id_format", ""))
    if os.path.realpath(view) != os.path.realpath(resolve(src)):
        fail.append((gid, "view_symlink_target", os.path.realpath(view)))
    fsha = sha256(resolve(src))
    if fsha != r["spectrum_source_sha256"]:
        fail.append((gid, "spectrum_sha256", src))
    spec, problem = read_spectrum(view)
    if problem:                                   # unreadable file, missing dataset or wrong dtype: named failure, no traceback
        fail.append((gid, problem[0], problem[1]))
        continue
    m, mi, fp, ls, lss, meta, vol, h11_geo, has_vac = spec; n = len(m)
    S = I["spectrum_settings"]
    chk = {
        "n_modes_equals_manifest": n == r["n_modes_above_H0"],
        "h11_matches_manifest": meta["h11"] == h and h11_geo == h and meta["frst"] == r["frst"],
        "settings_production": meta["prec"] == S["prec"] and bool(int(meta["quartics"])) == S["quartics"] and abs(meta["threshold_log10"] - THR) < 1e-12,
        "completed_not_provisional": meta["status"] == "completed" and meta["terminal_status"] == "completed" and int(meta["provisional"]) == 0,
        "schema_version": meta["schema_version"] == S["schema_version"],
        "array_lengths": len(mi) == n and len(fp) == n and len(ls) == n and len(lss) == n,
        "mode_indices_unique_top_block": sorted(mi.tolist()) == list(range(h - n, h)),
        "finite": bool(np.all(np.isfinite(m)) and np.all(np.isfinite(fp))),
        "all_m_ge_H0": bool(np.all(m >= THR)),
        "fpert_equals_m_over_sqrt_lambda": bool(np.all(np.abs(fp - (m - 9 - 0.5 * ls)) < 1e-9)),
        "cy_volume_positive_finite": math.isfinite(vol) and vol > 0,
        "lambda_self_sign_is_plus_or_minus_one": bool(np.all(np.isin(lss, (-1, 1)))),
        "vacuum_group_iff_corrected": has_vac == (st == "VACUUM_CORRECTED"),
        # vmm spectra about theta = 0 carry no evaluation_point; repaired spectra carry exactly validated_leading_branch_minimum
        "evaluation_point_known": meta["evaluation_point"] in (None, "validated_leading_branch_minimum"),
        "evaluation_point_iff_corrected": (meta["evaluation_point"] == "validated_leading_branch_minimum") == (st == "VACUUM_CORRECTED"),
        "status_known": st in ("ORIGINAL_SPECTRUM", "VACUUM_CORRECTED"),
    }
    for k, ok in chk.items():
        if not ok:
            fail.append((gid, k, ""))
    if not (chk["cy_volume_positive_finite"] and chk["finite"] and chk["array_lengths"]):
        continue                                  # m_KK and the per-mode quantities need a positive volume and finite logs
    label = int(meta["polytope"])
    label_kind = "local_index" if label == r["local_index"] else ("unified_position" if label == pos else "OTHER")
    if label_kind == "OTHER" or (label_kind == "unified_position" and label != r["local_index"] and st != "VACUUM_CORRECTED"):
        fail.append((gid, "file_label_unexpected", label))
    if label != r["local_index"]:
        notes.append({"geometry_id": gid, "status": st, "file_polytope_label": label, "manifest_local_index": r["local_index"], "manifest_position": pos})
    order = np.argsort(-m, kind="stable")
    rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
    log10_mKK_GeV = math.log10(MPL_KK) - (2 / 3) * math.log10(vol)
    for i in range(n):
        lm, lf = float(m[i]), float(fp[i])
        m_gev = 10.0 ** (lm - 9)
        valid = not (m_gev > HI3)
        modes.append({
            "geometry_id": gid, "h11": h, "mode_index": int(mi[i]), "mode_rank_desc": int(rank[i]), "n_modes_geometry": n,
            "log10_mass_eV": lm, "mass_eV": 10.0 ** lm, "log10_f_pert_GeV": lf, "f_pert_GeV": 10.0 ** lf,
            "f_relic_over_f_pert": 1.0, "theta_reference": TH,
            "vacuum_status": st, "spectrum_source": src,
            "cy_volume": vol, "m_KK_GeV": 10.0 ** log10_mKK_GeV, "log10_m_KK_GeV": log10_mKK_GeV,
            "mass_over_mKK": 10.0 ** (lm - 9 - log10_mKK_GeV), "above_mKK": bool(m_gev > 10.0 ** log10_mKK_GeV),
            "mass_over_fpert": 10.0 ** (lm - 9 - lf), "above_fpert": bool(m_gev > 10.0 ** lf),
            "mass_over_3HImax": m_gev / HI3, "inflation_margin_dex": L3HI - (lm - 9),
            "inflation_valid": valid,
            "inflation_exclusion_reason": "NA" if valid else "m_a>3H_I_max(Fuzzy_fn16;1.4e14_GeV)",
            "lambda_self_sign": int(lss[i]), "lambda_self_log10": float(ls[i]),
        })
    geoms.append({"geometry_id": gid, "h11": h, "position": pos, "tranche": r["tranche"], "local_index": r["local_index"],
                  "vacuum_status": st, "n_modes": n, "cy_volume": vol, "m_KK_GeV": 10.0 ** log10_mKK_GeV,
                  "file_polytope_label": label, "file_label_kind": label_kind,
                  "spectrum_source": src, "spectrum_view": view, "spectrum_source_sha256": fsha,
                  "validation_status": r["validation_status"], "qcd_edge_flag": r["qcd_edge_flag"],
                  "qcd_edge_beyond_gate_tolerance": r["qcd_edge_beyond_gate_tolerance"]})

if not geoms or fail:
    dump_json({"manifest": man_path, "all_checks_pass": False, "per_file_failures": fail, "excluded_geometries": excluded},
              os.path.join(args.out_dir, "a1_input_verification.json"))
    sys.exit(f"{len(fail)} per-file failure(s): " + "; ".join(f"{g} {k} {d}" for g, k, d in fail[:5]) if fail else "no usable geometry in the input")
# A geometry whose validated spectrum has no mode above the threshold is kept (n_modes = 0): it belongs to the population
# (Omega_geometry = 0, reason NO_MODES_ABOVE_THRESHOLD in the relic run). The mode table may therefore be empty.
MODE_COLS = ["geometry_id", "h11", "mode_index", "mode_rank_desc", "n_modes_geometry", "log10_mass_eV", "mass_eV", "log10_f_pert_GeV", "f_pert_GeV",
             "f_relic_over_f_pert", "theta_reference", "vacuum_status", "spectrum_source", "cy_volume", "m_KK_GeV", "log10_m_KK_GeV", "mass_over_mKK",
             "above_mKK", "mass_over_fpert", "above_fpert", "mass_over_3HImax", "inflation_margin_dex", "inflation_valid", "inflation_exclusion_reason",
             "lambda_self_sign", "lambda_self_log10"]
D = pd.DataFrame(modes, columns=MODE_COLS).sort_values(["h11", "geometry_id", "mode_index"], kind="stable").reset_index(drop=True)
D = D.astype({"inflation_valid": bool, "above_mKK": bool, "above_fpert": bool})
G = pd.DataFrame(geoms).sort_values(["h11", "geometry_id"], kind="stable").reset_index(drop=True)
assert not D.astype(str).apply(lambda col: col.str.contains(",")).any().any(), "comma inside a mode-table field"

# ---- population-level checks
exp = cfg["run"]
pc = {
    "manifest_geometries": len(M["geometries"]),
    "geometries_read": len(G), "modes_read": len(D),
    "geometries_by_status": dict(Counter(G.vacuum_status)), "modes_by_status": dict(Counter(D.vacuum_status)),
    "manifest_status_counts": dict(status_count),
    "h11_range": [int(G.h11.min()), int(G.h11.max())], "h11_values_present": sorted(int(x) for x in G.h11.unique()),
    "geometries_with_zero_modes_above_threshold": int((G.n_modes == 0).sum()),
    "unique_mode_keys": int(D[["geometry_id", "mode_index"]].drop_duplicates().shape[0]),
    "unique_geometry_ids": int(G.geometry_id.nunique()),
}
pc["excluded_geometries"] = len(excluded)
pc["vacuum_validation"] = dict(Counter(v["vacuum_validation"] for v in vac_rows))
per_h11_ok = all(int((D.h11 == int(h)).sum()) == v["modes_final"] for h, v in M["per_h11"].items())
checks = {
    "at_least_one_geometry": len(G) > 0,
    "unique_geometry_and_mode_keys": pc["geometries_read"] == pc["unique_geometry_ids"] and pc["modes_read"] == pc["unique_mode_keys"],
    "excluded_geometries_only_by_declared_policy": not excluded or EXCLUDE_OK,
    "per_h11_mode_counts_match_manifest": per_h11_ok,
    "manifest_summary_total": M["summary"]["final_usable_modes"] == len(D) and M["summary"]["final_usable_geometries"] == len(G),
    "no_per_file_failures": len(fail) == 0,
    # the frozen dataset has 19 such files (tranche-2 vacuum-corrected geometries); a single unified run has none
    "label_anomaly_corrected_files_only": all(x["status"] == "VACUUM_CORRECTED" and x["file_polytope_label"] == x["manifest_position"] for x in notes),
}
if "expected_geometries" in exp:                                        # reference case (expectations given)
    checks.update({
        "geometries_4491": pc["geometries_read"] == exp["expected_geometries"] == pc["unique_geometry_ids"],
        "modes_16313": pc["modes_read"] == exp["expected_modes"] == pc["unique_mode_keys"],
        "status_geometries": pc["geometries_by_status"] == exp["expected_geometries_by_vacuum_status"],
        "status_modes": pc["modes_by_status"] == exp["expected_modes_by_vacuum_status"],
        "no_excluded_geometries": status_count.get("EXCLUDED_VACUUM_UNRESOLVED", 0) == 0,
        "h11_4_to_50_all_present": pc["h11_values_present"] == list(range(exp["h11_min"], exp["h11_max"] + 1)),
    })
# ---- flags (per mode; nothing is removed here)
MPL_RED = cfg["cosmology"]["M_pl_reduced_GeV"]
flags = {
    "f_pert_above_reduced_M_pl_modes": int((D.log10_f_pert_GeV > math.log10(MPL_RED)).sum()),   # diagnostic only (f > M_pl: outside the 4d EFT)
    "inflation_invalid_modes": int((~D.inflation_valid).sum()), "inflation_invalid_geometries": int(D[~D.inflation_valid].geometry_id.nunique()),
    "above_mKK_modes": int(D.above_mKK.sum()), "above_mKK_geometries": int(D[D.above_mKK].geometry_id.nunique()),
    "above_fpert_modes": int(D.above_fpert.sum()), "above_fpert_geometries": int(D[D.above_fpert].geometry_id.nunique()),
    "nearest_inflation_margin_dex_valid_side": float(D.loc[D.inflation_valid, "inflation_margin_dex"].min()) if D.inflation_valid.any() else None,
    "nearest_inflation_margin_dex_invalid_side": float(-D.loc[~D.inflation_valid, "inflation_margin_dex"].max()) if (~D.inflation_valid).any() else None,
}
if "expected_modes" in V["inflation_cut"]:
    checks["inflation_invalid_365_in_329"] = flags["inflation_invalid_modes"] == V["inflation_cut"]["expected_modes"] and flags["inflation_invalid_geometries"] == V["inflation_cut"]["expected_geometries"]
if "expected_modes_above" in V["mKK_flag"]:
    checks["above_mKK_226_in_212"] = flags["above_mKK_modes"] == V["mKK_flag"]["expected_modes_above"] and flags["above_mKK_geometries"] == V["mKK_flag"]["expected_geometries"]
if "expected_modes_above" in V["fpert_flag"]:
    checks["above_fpert_326_in_305"] = flags["above_fpert_modes"] == V["fpert_flag"]["expected_modes_above"] and flags["above_fpert_geometries"] == V["fpert_flag"]["expected_geometries"]
checks["f_relic_equals_f_pert"] = bool((D.f_relic_over_f_pert == 1.0).all())
checks["theta_reference_pi_over_2"] = bool((D.theta_reference == math.pi / 2).all())

pd.DataFrame(vac_rows).to_csv(os.path.join(args.out_dir, "vacuum_validation.csv"), index=False)
D.to_csv(CANONICAL_MODES, index=False, float_format="%.17g")
G.to_csv(CANONICAL_GEOMS, index=False, float_format="%.17g")
RB = pd.read_csv(CANONICAL_MODES, float_precision="round_trip", keep_default_na=False)
checks["canonical_csv_roundtrip_bit_exact"] = bool(all((RB[c].values == D[c].values).all() for c in
                                                       ("log10_mass_eV", "log10_f_pert_GeV", "cy_volume", "mode_index", "geometry_id")))
spectrum_set = hashlib.sha256("".join(f"{a}\t{b}\n" for a, b in sorted(zip(G.geometry_id, G.spectrum_source_sha256))).encode()).hexdigest()
out = {"manifest": man_path, "manifest_sha256": manifest_sha, "input_sha256sums_file_sha256": sha256(os.path.join(input_dir, "SHA256SUMS.txt")),
       "spectrum_set_sha256": spectrum_set,
       "spectrum_files_rehashed": len(G), "population": pc, "checks": checks, "all_checks_pass": all(checks.values()),
       "expectations_checked": "expected_geometries" in exp, "excluded_geometries": excluded,
       "flags": flags, "per_file_failures": fail, "file_label_anomaly": notes,
       "outputs": {"canonical_modes.csv": sha256(CANONICAL_MODES), "canonical_geometries.csv": sha256(CANONICAL_GEOMS)}}
dump_json(out, os.path.join(args.out_dir, "a1_input_verification.json"))
print(json.dumps({k: v for k, v in out.items() if k not in ("file_label_anomaly",)}, indent=1))
sys.exit(0 if out["all_checks_pass"] else 1)
