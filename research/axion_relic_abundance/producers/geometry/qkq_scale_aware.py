"""Scale-aware Python/Julia comparison of q_i' Kinv q_j (supplements validate_stage.py's julia_qKq_matches_python).

A plain relative tolerance on q_i' Kinv q_j is ill-posed when the contraction cancels strongly (sum of |terms| up to
1e11 x |result| at h11 >= 30): both languages then carry only float64 rounding relative to the term magnitudes.
Criterion: |julia - python| <= 1e-12 * sum_ik |q_i| |Kinv_ik| |q_k| (i.e. agreement to ~1e4 ulp of the summands);
an orientation/basis error would produce O(1) differences. Reports both the plain relative and the scale-aware values.
Usage: python qkq_scale_aware.py MATERIALIZED_ROOT JULIA_CROSSCHECK_JSONL OUT_JSON
"""
import json, sys
import numpy as np, h5py

root, jl, out = sys.argv[1:4]
res, worst_scaled, worst_rel, fails = [], 0.0, 0.0, []
for line in open(jl):
    j = json.loads(line)
    if "error" in j:
        fails.append((j["file"], "julia error")); continue
    with h5py.File(f"{root}/{j['file']}", "r") as f:
        Q = np.asarray(f["cytools/potential/Q"][()]).T; K = np.asarray(f["cytools/geometric/Kinv"][()])
    for (a, b), vj in zip(j["sample_pairs"], j["sample_qKq"]):
        qa, qb = Q[:, a - 1], Q[:, b - 1]
        vp = float(qa @ K @ qb); mag = float(np.abs(qa) @ np.abs(K) @ np.abs(qb))
        scaled = abs(vj - vp) / mag if mag else 0.0; rel = abs(vj - vp) / max(abs(vp), 1e-300)
        worst_scaled = max(worst_scaled, scaled); worst_rel = max(worst_rel, rel)
        if scaled > 1e-12:
            fails.append((j["file"], [a, b], scaled))
S = {"criterion": "|julia-python| <= 1e-12 * sum|q||Kinv||q|", "geometries": sum(1 for _ in open(jl)), "pass": not fails,
     "failures": fails[:10], "worst_scaled": worst_scaled, "worst_plain_relative": worst_rel}
json.dump(S, open(out, "w"), indent=1)
print(json.dumps(S))
