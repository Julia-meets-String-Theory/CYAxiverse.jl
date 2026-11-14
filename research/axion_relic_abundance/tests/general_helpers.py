"""Helpers of tests/test_general.sh (general samples; no population count of any reference case is used).

  make-list FROZEN_ROOT OUT_LIST KIND [POLICY]   spectra list from frozen Shoroq spectra (read in place, never written):
      one       one theta = 0 geometry (VERIFY)
      subset    two tranche-1 theta = 0 geometries for every 4th h11 of tranche 1 (VERIFY), three vacuum-corrected geometries
                (DOCUMENTED_EXTERNAL, evidence: the frozen minima-validation records), one theta = 0 tachyonic
                geometry (VERIFY: must come out TACHYONIC_AT_THETA0) and one entry without vacuum information (UNVERIFIED)
  compare-rows WORK FROZEN_ROOT OUT_JSON       every master / geometry-summary row of every scenario label that also exists in
      the frozen run, and every canonical mode row, must equal the frozen row exactly (per-mode and per-geometry independence)
  tamper-evidence LIST OUT_LIST                copy of LIST with a wrong evidence checksum
"""
import csv, json, os, sys

cmd = sys.argv[1]


def frozen_manifest(fr):
    return json.load(open(os.path.join(fr, "relic_abundance_input", "relic_abundance_input_manifest.json")))["geometries"]


if cmd == "make-list":
    fr, out, kind = sys.argv[2:5]
    policy = sys.argv[5] if len(sys.argv) > 5 else "exclude"
    G = frozen_manifest(fr)
    t1 = [r for r in G if r["tranche"] == 1 and r["status"] == "ORIGINAL_SPECTRUM"]
    ev = os.path.join(fr, "tachyonic_vacua_audit", "results", "03_minima_validation.jsonl")
    import hashlib
    evs = hashlib.sha256(open(ev, "rb").read()).hexdigest()
    verify = lambda r: {"path": r["spectrum_source"], "geometry_id": r["geometry_id"], "vacuum_validation": {"state": "VERIFY"}}
    if kind == "one":
        S = [verify(t1[0])]
    else:
        S = []
        hs = sorted({r["h11"] for r in t1})[::4]                       # every 4th h11 present in tranche 1 (4 ... 40)
        for h in hs:
            S += [verify(r) for r in [r for r in t1 if r["h11"] == h][:2]]
        corr = [r for r in G if r["status"] == "VACUUM_CORRECTED"]
        chosen = (corr[0], corr[len(corr) // 2], corr[-1])
        for r in chosen:
            S.append({"path": r["spectrum_source"], "geometry_id": r["geometry_id"],
                      "vacuum_validation": {"state": "DOCUMENTED_EXTERNAL",
                                            "method": "leading-branch minimum validated on the full potential at ~500 and ~1000 digits",
                                            "reference": "Shoroq tachyonic_vacua_audit (frozen 2026-09-30)",
                                            "evidence": [{"path": ev, "sha256": evs}]}})
        # theta = 0 spectrum of another tachyonic tranche-1 geometry: the scan must find it TACHYONIC_AT_THETA0
        tach = next(r for r in corr if r["tranche"] == 1 and r not in chosen)
        S.append({"path": os.path.join(fr, "thesis_g2_adaptive", "spectra", tach["geometry_id"], "cyax.h5"),
                  "geometry_id": tach["geometry_id"], "vacuum_validation": {"state": "VERIFY"}})
        u = next(r for r in t1 if r["h11"] not in hs)
        S.append({"path": u["spectrum_source"], "geometry_id": u["geometry_id"]})              # no vacuum information: UNVERIFIED
    json.dump({"schema": "cyaxiverse_relic_spectra/1", "description": f"general test ({kind}) on frozen Shoroq spectra",
               "vacuum_policy": {"on_unusable": policy}, "spectra": S}, open(out, "w"), indent=1)
    print(out, len(S), "entries")

elif cmd == "compare-rows":
    work, fr, out = sys.argv[2:5]
    FT = os.path.join(fr, "relic_abundance_production", "tables")
    res, ok = {}, True
    labs = [s["label"] for s in json.load(open(os.path.join(work, "relic", "run_config.json")))["reheating"]["scenarios"]]
    for lab in labs:
        for kind, key in (("master_modes", (0, 2)), ("geometry_summary", (0,))):
            fp = os.path.join(FT, f"{kind}_{lab}.csv")
            if not os.path.exists(fp):
                continue                                        # custom scenario: no frozen counterpart
            a = list(csv.reader(open(os.path.join(work, "relic", "tables", f"{kind}_{lab}.csv"))))
            b = list(csv.reader(open(fp)))
            B = {tuple(r[i] for i in key): r for r in b[1:]}
            diff = [r[:3] for r in a[1:] if B.get(tuple(r[i] for i in key)) != r]
            res[f"{kind}_{lab}"] = {"rows": len(a) - 1, "header_identical": a[0] == b[0], "rows_different": len(diff), "examples": diff[:5]}
            ok &= a[0] == b[0] and not diff and len(a) > 1
    a = list(csv.reader(open(os.path.join(work, "relic", "a1", "canonical_modes.csv"))))
    b = list(csv.reader(open(os.path.join(fr, "relic_abundance_production_prep", "results", "canonical_modes.csv"))))
    B = {(r[0], r[2]): r for r in b[1:]}
    diff = [r[:3] for r in a[1:] if B.get((r[0], r[2])) != r]
    res["canonical_modes"] = {"rows": len(a) - 1, "header_identical": a[0] == b[0], "rows_different": len(diff), "examples": diff[:5]}
    ok &= a[0] == b[0] and not diff
    res["pass"] = bool(ok)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: (v if k == "pass" else {kk: v[kk] for kk in ("rows", "rows_different")}) for k, v in res.items()}, indent=1))
    sys.exit(0 if ok else 1)

elif cmd == "tamper-evidence":
    L = json.load(open(sys.argv[2]))
    for s in L["spectra"]:
        for e in s.get("vacuum_validation", {}).get("evidence", []):
            e["sha256"] = "0" * 64
    json.dump(L, open(sys.argv[3], "w"), indent=1)
