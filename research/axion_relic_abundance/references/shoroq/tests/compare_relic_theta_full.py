"""Compare stages 4-5 run on a REGENERATED relic input (full reproduction) with the frozen relic and theta results.

  * A1 canonical_modes.csv / canonical_geometries.csv: same rows in the same order, every cell identical as text, except
    the file-location columns (spectrum_source, spectrum_view) and the file checksum spectrum_source_sha256 (the HDF5
    files carry run-time metadata); for those the geometry id and the kind (theta = 0 or corrected spectrum) must agree;
    a frozen tranche-2 theta = 0 path differs only by its np_ directory (tranche-local index), accepted only through the
    exact unified_polytopes.json mapping
  * file-label columns of canonical_geometries.csv (local_index, file_polytope_label, file_label_kind): the unified run
    labels every file by its unified position; the frozen tranche-2 files carry their tranche-local index. Accepted only
    if new local_index = file_polytope_label = position with kind local_index, and the frozen row is self-consistent
    (label = its local_index for kind local_index, label = position for kind unified_position)
  * the ten relic tables (master_modes, geometry_summary, h11_summary per T_reh, global_summary.json) byte-identical
  * run_metadata.json: only script/config/mode-table/manifest locations and checksums may differ
  * independent verification (C on every row, B stratified) results identical to the frozen run
  * every theta result file byte-identical; Python theta cross-check results identical
Usage: python compare_relic_theta_full.py WORK FROZEN_ROOT OUT_JSON
"""
import csv, filecmp, json, os, sys

work, fr, out = sys.argv[1:4]
R, F = os.path.join(work, "relic"), os.path.join(fr, "relic_abundance_production")
LOC = {"spectrum_source", "spectrum_view", "spectrum_source_sha256"}
LABEL_COLS = {"local_index", "file_polytope_label", "file_label_kind"}
META_ALLOWED = {"script", "script_sha256", "config", "config_sha256", "mode_table", "mode_table_sha256", "manifest_sha256"}
res, ok = {}, True


def check(name, cond, **info):
    global ok
    res[name] = dict(info, passed=bool(cond))
    ok = ok and bool(cond)


def kind(path):
    return "corrected" if "corrected_spectra" in path else "theta0"


def tail(path):
    return "/".join(path.split("/")[-4:])


UP = {(p["h11"], p["position"]): p for p in
      json.load(open(os.path.join(fr, "thesis_g2_adaptive", "unified_h11_004_050", "manifests", "unified_polytopes.json")))["polytopes"]}


def label_only(u, v):
    """frozen tranche-2 theta = 0 spectra are stored under their tranche-local np_ index, the unified run under the unified
    position: accepted only if that index is the sole difference and it maps exactly through unified_polytopes.json"""
    a, b = u.split("/")[-4:], v.split("/")[-4:]
    if a[0] != b[0] or a[2:] != b[2:] or "/tranche2/" not in v or "corrected_spectra" in u + v:
        return False
    p = UP.get((int(a[0][4:]), int(a[1][3:])))
    return p is not None and p["tranche"] == 2 and int(b[1][3:]) == p["local_index"] != p["position"]


for name in ("canonical_modes.csv", "canonical_geometries.csv"):
    a = list(csv.reader(open(os.path.join(R, "a1", name))))
    b = list(csv.reader(open(os.path.join(fr, "relic_abundance_production_prep", "results", name))))
    hdr = a[0]
    cells, loc, labels, path_labels = [], [], 0, 0
    if hdr == b[0] and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a[1:], b[1:]), 1):
            X, Y = dict(zip(hdr, x)), dict(zip(hdr, y))
            if LABEL_COLS <= set(hdr) and any(X[c] != Y[c] for c in LABEL_COLS):
                new_ok = X["local_index"] == X["file_polytope_label"] == X["position"] and X["file_label_kind"] == "local_index"
                old_ok = (Y["file_label_kind"] == "local_index" and Y["file_polytope_label"] == Y["local_index"]) or \
                         (Y["file_label_kind"] == "unified_position" and Y["file_polytope_label"] == Y["position"])
                if new_ok and old_ok:
                    labels += 1
                    X.update({c: Y[c] for c in LABEL_COLS}); x = [X[c] for c in hdr]
            for c, u, v in zip(hdr, x, y):
                if c in LOC:
                    if c == "spectrum_source_sha256" or (tail(u) == tail(v) and kind(u) == kind(v)):
                        continue
                    if label_only(u, v):
                        path_labels += 1
                    else:
                        loc.append((i, c, u, v))
                elif u != v:
                    cells.append((i, c, u, v))
    check(f"a1_{name}", hdr == b[0] and len(a) == len(b) and not cells and not loc, rows=len(a) - 1, frozen_rows=len(b) - 1,
          same_header=hdr == b[0], label_differences=labels, path_label_differences=path_labels, cell_differences=cells[:20], n_cell_differences=len(cells), location_kind_differences=loc[:20],
          columns_excluded_from_text_equality=sorted(LOC & set(hdr)))

tables = [f"{k}_{l}.csv" for k in ("master_modes", "geometry_summary", "h11_summary") for l in ("5MeV", "1e4GeV", "1e10GeV")] + ["global_summary.json"]
tb = {f: filecmp.cmp(os.path.join(F, "tables", f), os.path.join(R, "tables", f), shallow=False) for f in tables}
check("relic_tables_byte_identical", all(tb.values()), files=tb)
ma, mb = (json.load(open(os.path.join(d, "tables", "run_metadata.json"))) for d in (R, F))
md = sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))
check("run_metadata_only_locations_and_checksums_differ", set(md) <= META_ALLOWED, differing_keys=md)
va, vb = (json.load(open(os.path.join(d, "validation", "production_verification.json"))) for d in (R, F))
check("independent_verification_identical", all(va[k] == vb[k] for k in ("C_full", "B_sample", "B_geometries")) and va.get("all_pass") is True)

FT, NT = os.path.join(fr, "relic_abundance_final_analysis", "results", "theta"), os.path.join(work, "final_analysis", "theta")
th = {}
for d, _, fs in os.walk(FT):
    for f in fs:
        if f.endswith((".csv", ".json")):
            rel = os.path.relpath(os.path.join(d, f), FT)
            n = os.path.join(NT, rel)
            th[rel] = os.path.exists(n) and filecmp.cmp(os.path.join(FT, rel), n, shallow=False)
check("theta_results_byte_identical", th and all(th.values()), files=len(th), different=sorted(k for k, v in th.items() if not v))
ca = json.load(open(os.path.join(work, "final_analysis", "verification", "crosscheck_python.json")))
cb = json.load(open(os.path.join(fr, "relic_abundance_final_analysis", "results", "verification", "crosscheck_python.json")))
check("theta_crosscheck_identical", ca["theta"] == cb["theta"] and ca["new6"] == cb["new6"] and all(v["agree"] for v in ca["census"].values()))

json.dump({"checks": res, "pass": ok}, open(out, "w"), indent=1, default=str)
print(json.dumps({k: v["passed"] for k, v in res.items()}, indent=1))
print("RELIC/THETA", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
