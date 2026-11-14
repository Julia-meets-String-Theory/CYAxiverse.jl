"""Phase 11: independent end-to-end integrity checks on a seeded, stratified random sample of the final dataset.

Chain per geometry: t_final (stored) -> exact rational tau, V, Kinv (from stored integer kappa) -> Q (from rays), L
(mpmath, Eq. 9) -> compare with the materialized arrays Julia reads -> Julia-interpreted values (julia_compare.jsonl)
-> spectrum (batch output vs an independent recompute, done by integrity_spectra.jl) and vs the audited run.
Usage: python integrity_sample.py FINAL_ROOT OUT_JSON [per_stratum]
"""
import csv, glob, itertools, json, math, os, random, sys
from fractions import Fraction
import numpy as np, h5py, mpmath

N = sys.argv[1]; out = sys.argv[2]; k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
mpmath.mp.dps = 60
aud = {json.loads(l)["file"]: json.loads(l) for p in glob.glob(f"{N}/validation/materialization/audit_with_materialized_h11_*.jsonl") for l in open(p)}
jul = {json.loads(l)["file"]: json.loads(l) for l in open(f"{N}/validation/julia/julia_compare.jsonl")}
strata = {
    "h11_41_50": lambda r: 41 <= r["h11"] <= 50, "qcd_edge_flagged": lambda r: r["qcd_volume_final"] < 25 * (1 - 1e-5), "mid_h11_15_25": lambda r: 15 <= r["h11"] <= 25, "high_h11_31_40": lambda r: 31 <= r["h11"] <= 40,
    "canonical_tip": lambda r: r["point_kind"] == "canonical_tip", "projected_point": lambda r: r["point_kind"] == "randomized_projection",
    "eq21_dilation": lambda r: r["m_eq21_g2_sorted"] > 1 and r["binding_constraint"] == "eq21",
    "qcd_dilation_m_gt_2": lambda r: r["binding_constraint"] == "qcd_lower_edge" and r["m_stored"] > 2,
}
rng = random.Random(20260930)
chosen = {}
for name, pred in strata.items():
    pool = sorted(f for f, r in aud.items() if pred(r) and f not in chosen)
    for f in (pool if name == "qcd_edge_flagged" else rng.sample(pool, min(k, len(pool)))):
        chosen[f] = name


def exact(kappa, t):
    h = len(t); tf = [Fraction(float(v)) for v in t]
    V = Fraction(0); tau = [Fraction(0)] * h; kab = [[Fraction(0)] * h for _ in range(h)]
    for i, j, kk, v in kappa:
        i, j, kk = int(i), int(j), int(kk); v = Fraction(int(round(v)))
        perms = set(itertools.permutations((i, j, kk)))
        V += len(perms) * v * tf[i] * tf[j] * tf[kk] / 6
        for a, b, c in perms:
            kab[a][b] += v * tf[c]; tau[a] += v * tf[b] * tf[c] / 2
    return V, tau, [[4 * (tau[a] * tau[b] - V * kab[a][b]) for b in range(h)] for a in range(h)]


mp = lambda q: mpmath.mpf(q.numerator) / q.denominator
res = []
for f, stratum in chosen.items():
    r = aud[f]
    raw = next(p for p in glob.glob(f"{N}/geometry/run_h11_*/{f}"))
    with h5py.File(raw, "r") as fr:
        g = fr["cytools/geometric"]; kappa = g["kappa"][()]; t = g["tip"][()]; rays = np.asarray(g["effective_cone"][()], np.int64)
    with h5py.File(f"{N}/materialized/{f}", "r") as fm:
        gm = fm["cytools/geometric"]; Qm = np.asarray(fm["cytools/potential/Q"][()]).T; Lm = np.asarray(fm["cytools/potential/L"][()]).T
        tau_m = gm["divisor_volumes"][()]; K_m = gm["Kinv"][()]; tip_m = gm["tip"][()]
    V, tau, K = exact(kappa, t)
    tau_x = np.asarray([float(v) for v in tau]); K_x = np.asarray([[float(v) for v in row] for row in K])
    nq = len(rays); x = [sum(int(rays[a][c]) * tau[c] for c in range(len(tau))) for a in range(nq)]
    Qx = [rays[a] for a in range(nq)]; sx = [1] * nq
    Vm = mp(V); pi = mpmath.pi; l10e = mpmath.log10(mpmath.e)
    lx = [mpmath.log10(8 * pi * mp(x[a]) / Vm ** 2) - 2 * pi * mp(x[a]) * l10e for a in range(nq)]
    for i in range(nq - 1):
        for j in range(i + 1, nq):
            qkq = sum(int(rays[i][p]) * K[p][q] * int(rays[j][q]) for p in range(len(tau)) for q in range(len(tau)) if rays[i][p] and rays[j][q])
            s = mp(x[i]) + mp(x[j]); amp = 8 * pi / Vm ** 2 * (pi * mp(qkq) + s)
            Qx.append(rays[j] - rays[i]); sx.append(int(mpmath.sign(amp))); lx.append(mpmath.log10(abs(amp)) - 2 * pi * s * l10e)
    Qx = np.asarray(Qx, np.int64).T; lx = np.asarray([float(v) for v in lx]); sx = np.asarray(sx, float)
    lead = lx >= lx.max() - 300
    j = jul.get(f, {})
    res.append({"file": f, "stratum": stratum, "h11": r["h11"], "point_kind": r["point_kind"], "m": r["m_stored"], "binding": r["binding_constraint"],
                "tip_materialized_bitwise_raw": bool(np.array_equal(tip_m, t)),
                "tau_materialized_vs_exact_rel": float(np.max(np.abs(tau_m - tau_x)) / np.max(np.abs(tau_x))),
                "Kinv_materialized_vs_exact_rel": float(np.max(np.abs(K_m - K_x)) / np.max(np.abs(K_x))),
                "V_exact": float(V), "Q_materialized_equals_exact": bool(np.array_equal(Qm, Qx)),
                "L_sign_materialized_equals_exact": bool(np.array_equal(Lm[0], sx)),
                "L_materialized_vs_exact_max_dex": float(np.max(np.abs(Lm[1] - lx))),
                "L_materialized_vs_exact_leading300_max_dex": float(np.max(np.abs(Lm[1] - lx)[lead])),
                "julia_Q_exact": j.get("Q_columns_exact"), "julia_L_equal": j.get("L_columns_maxdiff") == 0.0,
                "julia_qKq_rel": j.get("qKq_rel_maxdiff"), "julia_tip_bitwise": j.get("tip_bitwise")})
    print(f, stratum, res[-1]["Q_materialized_equals_exact"], "%.1e" % res[-1]["L_materialized_vs_exact_max_dex"], flush=True)
json.dump({"seed": 20260930, "per_stratum": k, "sample": res}, open(out, "w"), indent=1)
with open(out.replace(".json", "_files.txt"), "w") as fh:
    fh.write("\n".join(chosen) + "\n")
