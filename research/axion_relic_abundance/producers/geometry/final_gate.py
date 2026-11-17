"""Final acceptance of the geometry/spectrum stage: validate_stage.py plus the two documented resolutions.

The dataset is accepted iff every criterion of validate_stage.py passes, with exactly two documented exceptions:
  * julia_qKq_matches_python: a plain relative tolerance is ill-posed for strongly cancelling contractions; the stage
    passes if qkq_scale_aware.py passes (|julia - python| <= 1e-12 sum|q||Kinv||q|), as adopted for the frozen dataset;
  * qcd_window_25_40: every violating geometry must be a lower-edge case listed in flag_qcd_edge.py's output (owner
    decision 2026-09-30: keep and flag); any violation above 40 or unflagged fails the stage.
Usage: python final_gate.py GATE_JSON QKQ_JSON QCD_FLAGS_JSON AUDIT_JSONL[,...] OUT_JSON
"""
import json, sys

gate, qkq, qcd = (json.load(open(p)) for p in sys.argv[1:4])
aud = [json.loads(l) for p in sys.argv[4].split(",") for l in open(p) if l.strip()]
out = sys.argv[5]
C = gate["criteria"]
flagged = {f["file"] for f in qcd["flags"] if f["beyond_gate_tolerance_1e-5_rel"]}
qcd_bad = [r["file"] for r in aud if "qcd_volume_final" in r and not (25 * (1 - 1e-5) <= r["qcd_volume_final"] <= 40 * (1 + 1e-5))]
resolved = {
    "julia_qKq_matches_python": {"original_pass": C["julia_qKq_matches_python"]["pass"], "scale_aware_pass": qkq["pass"],
                                 "worst_scaled": qkq["worst_scaled"], "pass": qkq["pass"]},
    "qcd_window_25_40": {"original_pass": C["qcd_window_25_40"]["pass"], "violations": len(qcd_bad),
                         "all_violations_flagged_lower_edge": set(qcd_bad) <= flagged, "pass": set(qcd_bad) <= flagged},
}
failed = [k for k, v in C.items() if not v["pass"] and k not in resolved] + [k for k, v in resolved.items() if not v["pass"]]
res = {"overall": "PASS" if not failed else "FAIL", "failed_criteria": failed, "resolved_criteria": resolved,
       "validate_stage_overall": gate["overall"], "qcd_edge_flags": qcd["count"], "qcd_edge_beyond_tolerance": qcd["count_beyond_gate_tolerance"]}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if not failed else 1)
