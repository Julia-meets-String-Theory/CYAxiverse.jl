"""QCD-edge flags of the accepted geometries (bookkeeping only: no geometry is removed).

Owner decision (2026-09-30): keep every G2-accepted geometry and flag the QCD-edge cases. G2 accepts the QCD window
[25, 40] using CYTools divisor volumes, which lose precision at very anisotropic points; the audit re-evaluates the QCD
divisor volume at the stored final point with the kappa reconstruction that the materializer uses.
  flag                          qcd_volume_final < 25 - 1e-8
  beyond_gate_tolerance_1e-5_rel qcd_volume_final < 25 (1 - 1e-5)   (the tolerance of validate_stage.py)
This rule reproduces the 539 flags (6 beyond tolerance) of the accepted dataset exactly. The exact-rational volume
(qcd_volume_exact_rational) is not recomputed here and is written as null.
Usage: python flag_qcd_edge.py AUDIT_JSONL[,AUDIT_JSONL...] OUT_JSON
"""
import json, sys

audits, out = sys.argv[1].split(","), sys.argv[2]
flags = []
for p in audits:
    for line in open(p):
        r = json.loads(line)
        if "qcd_volume_final" not in r or r["qcd_volume_final"] >= 25 - 1e-8:
            continue
        h11, pos = int(r["h11"]), int(r["file"].split("/")[1][3:])          # file = h11_XXX/np_POSITION/cy_0000001/cyax.h5
        flags.append({"h11": h11, "unified_position": pos, "file": r["file"],
                      "qcd_volume_kappa_reconstruction": r["qcd_volume_final"], "qcd_volume_exact_rational": None,
                      "deficit_below_25": 25 - r["qcd_volume_final"],
                      "beyond_gate_tolerance_1e-5_rel": r["qcd_volume_final"] < 25 * (1 - 1e-5),
                      "point_kind": r.get("point_kind"), "tau_max_over_min": r["tau_max"] / r["tau_min"]})
flags.sort(key=lambda f: (f["h11"], f["unified_position"]))
res = {"decision": "user 2026-09-30: keep all G2-accepted geometries, flag QCD-edge cases",
       "definition": __doc__.split("\n\n")[1].strip(), "count": len(flags),
       "count_beyond_gate_tolerance": sum(f["beyond_gate_tolerance_1e-5_rel"] for f in flags),
       "max_deficit": max((f["deficit_below_25"] for f in flags), default=0.0), "flags": flags}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "flags"}, indent=1))
