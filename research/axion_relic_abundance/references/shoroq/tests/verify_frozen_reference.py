"""Verify that a copy of the frozen reference (the accepted 2026-09-30/10-01 dataset and results) is complete and intact
before any regression comparison uses it. Every file is checked against the reference's own SHA256SUMS; files without
a checksum list are checked against pinned checksums.

Expected layout of FROZEN_ROOT (directory names of the original research workspace):
  relic_abundance_input/                      SHA256SUMS.txt, relic_abundance_input_manifest.json, spectra/ (symlink view)
  relic_abundance_production/                 SHA256SUMS.txt, tables/
  relic_abundance_production_prep/            SHA256SUMS.txt, benchmarks/, results/benchmark_summary.json
  relic_abundance_final_analysis/             SHA256SUMS.txt, results/theta/
  thesis_g2_adaptive/unified_h11_004_050/     manifests/SHA256SUMS_unified.txt, geometry/, materialized/, spectra/
  thesis_g2_adaptive/ (+ tranche2/)           validation/materialization/audit_with_materialized_h11_*.jsonl
  spectrum_count_audit/results/accounting_all.jsonl
  tachyonic_vacua_audit/                      results/SHA256SUMS_audit.txt, results/*.jsonl, corrected_spectra/
Usage: python verify_frozen_reference.py FROZEN_ROOT OUT_JSON [relic|subset|all]
"""
import hashlib, json, os, sys

root, out = sys.argv[1:3]
part = sys.argv[3] if len(sys.argv) > 3 else "all"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
PINNED = {  # frozen files that have no checksum list of their own
    "spectrum_count_audit/results/accounting_all.jsonl": "a84f855c9b0f0f2d63b914ea1a4e3e83202c35a7ff114662f2ededc6567aef48",
    "relic_abundance_input/relic_abundance_input_manifest.json": "4a5d3270216d0748cfd49f3806f65963362fc8ba92a18c4f125cce6852ecc6a2",
    "tachyonic_vacua_audit/results/05_vacuum_classification.json": "a29b762d44467418dc0a2c5b01aa24fb4deb3e6158eb504993a1bd0dee4e7104",
    "thesis_g2_adaptive/unified_h11_004_050/manifests/unified_polytopes.json": "3b749a229fb5331635dd503bfb9f5bcfbe61d65188ef2df8d67f451a7e011ea4",
}
LISTS = {  # (checksum list, base directory of its paths)
    "relic": [("relic_abundance_input/SHA256SUMS.txt", "relic_abundance_input"),
              ("relic_abundance_production/SHA256SUMS.txt", "relic_abundance_production"),
              ("relic_abundance_production_prep/SHA256SUMS.txt", "relic_abundance_production_prep"),
              ("relic_abundance_final_analysis/SHA256SUMS.txt", "relic_abundance_final_analysis")],
    "subset": [("thesis_g2_adaptive/unified_h11_004_050/manifests/SHA256SUMS_unified.txt", "thesis_g2_adaptive/unified_h11_004_050/manifests"),
               ("tachyonic_vacua_audit/results/SHA256SUMS_audit.txt", "tachyonic_vacua_audit"),
               ("tachyonic_vacua_audit/results/SHA256SUMS_corrected_spectra.txt", "tachyonic_vacua_audit/corrected_spectra")],
}
lists = LISTS["relic"] + LISTS["subset"] if part == "all" else LISTS[part]
pins = {k: v for k, v in PINNED.items() if part != "relic" or k.startswith("relic_abundance_input/")}
res = {"frozen_root": "<argument>", "part": part, "checksum_lists": {}, "pinned": {}, "missing": []}
for lst, base in lists:
    p = os.path.join(root, lst)
    if not os.path.exists(p):
        res["missing"].append(lst); continue
    ok, bad = 0, []
    for line in open(p):
        if not line.strip():
            continue
        h, f = line.split(None, 1); f = f.strip().lstrip("*")
        q = os.path.normpath(os.path.join(root, base, f))
        if os.path.exists(q) and sha(q) == h:
            ok += 1
        else:
            bad.append(f)
    res["checksum_lists"][lst] = {"ok": ok, "failed": bad[:10], "n_failed": len(bad)}
for f, h in pins.items():
    p = os.path.join(root, f)
    res["pinned"][f] = os.path.exists(p) and sha(p) == h
res["pass"] = not res["missing"] and all(v["n_failed"] == 0 and v["ok"] > 0 for v in res["checksum_lists"].values()) and all(res["pinned"].values())
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if res["pass"] else 1)
