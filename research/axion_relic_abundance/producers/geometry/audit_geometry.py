"""Independent per-geometry audit of G2 (generate_geometric_data_multitriangulation.py) schema-1.1 outputs.

For every cyax.h5 under RUN_ROOT this script
  * rebuilds the CY from the stored lattice points + simplices with CYTools and checks topology identity
    (basis, GLSM, kappa, Kähler hyperplanes, effective-cone rays) against the stored arrays;
  * reconstructs the Kähler-point path from construction metadata and re-derives it (canonical tip /
    projected point from the recorded seed, solver dependence of the projection);
  * re-derives the adaptive radial scale m = max(m_Eq21, m_prime, m_QCD) with an exact replica of G2's
    criterion and search, and identifies the binding constraint;
  * evaluates every quantity at the stored final point t_final with three engines
    (CYTools fan, exact COO kappa reconstruction = G2/materializer engine, and an independent kappa formula)
    and checks K Kinv = I, symmetry, positivity, cone slack, divisor bounds, QCD window, Eq. 21;
  * rebuilds Q and L with an independent implementation of the cited Eq. 9 construction and compares with
    G2's _geometry_potential_terms (and, when present, the materialized datasets).
Nothing under RUN_ROOT is modified (files are opened read-only).
Usage: python audit_geometry.py REPO_SCRIPTS RUN_ROOT OUT_JSONL [MATERIALIZED_ROOT] [--workers N]
"""
import itertools, json, math, os, sys, time, traceback
import numpy as np, h5py

REPO_SCRIPTS = sys.argv[1]
sys.path.insert(0, REPO_SCRIPTS)
import generate_geometric_data_multitriangulation as mg  # noqa: E402
from cytools import Polytope  # noqa: E402

LOG10E = math.log10(math.e)


# ----------------------------------------------------------------------------------------------- Eq. 21 replica
def eq21_ok(G0, x0, m, il, jl, two_sided=False):
    """Exact replica of G2 adaptive prefactor_is_valid (lines 2576-2594), optionally two-sided."""
    m2 = m * m
    x = m2 * x0
    with np.errstate(divide="ignore", invalid="ignore"):
        lhs = np.abs(np.log(np.abs(math.pi * m2 ** 2 * G0[il, jl])) - 2 * math.pi * (x[il] + x[jl]))
        rhs = np.abs(np.log(np.abs(x)) - 2 * math.pi * x)
    ok = bool(np.all(lhs > rhs[il]))
    if two_sided:
        ok = ok and bool(np.all(lhs > rhs[jl]))
    return ok


def g2_search(G0, x0, il, jl, max_m=1e6, two_sided=False):
    """Exact replica of G2's doubling + bisection search (lines 2599-2612)."""
    ok = lambda m: eq21_ok(G0, x0, m, il, jl, two_sided)
    lo, up = 1.0, 1.0
    if not ok(up):
        while up < max_m:
            lo = up
            up = min(2.0 * up, max_m)
            if ok(up):
                break
        else:
            return None
        if not ok(up):
            return None
        while up - lo > 1e-2:
            mid = 0.5 * (lo + up)
            if ok(mid):
                up = mid
            else:
                lo = mid
    return up


def grid_first_pass(G0, x0, il, jl, m_hi, step=1e-3):
    """Smallest m on a fine grid in [1, m_hi] that passes (monotonicity check of the bisection)."""
    for m in np.arange(1.0, m_hi + step, step):
        if eq21_ok(G0, x0, float(m), il, jl):
            return float(m)
    return None


# ----------------------------------------------------------------------------------------------- geometry engines
def kappa_dense(kappa, h11):
    K = np.zeros((h11, h11, h11))
    for i, j, k, v in kappa:
        for p in set(itertools.permutations((int(i), int(j), int(k)))):
            K[p] = v
    return K


def independent_K(kd, t):
    """K_{i jbar} = d^2(-2 ln V)/dT_i dTbar_j = 1/4 [ -A^{-1}/V + t t^T/(2 V^2) ], A_ij = kappa_ijk t^k."""
    A = np.einsum("ijk,k->ij", kd, t)
    V = np.einsum("ijk,i,j,k->", kd, t, t, t) / 6.0
    tau = 0.5 * np.einsum("ijk,j,k->i", kd, t, t)
    Kmat = 0.25 * (-np.linalg.inv(A) / V + np.outer(t, t) / (2 * V * V))
    return Kmat, V, tau


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def first_occurrence_unique(rows):
    seen, out = set(), []
    for r in rows:
        key = tuple(int(v) for v in r)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return np.asarray(out, dtype=np.int64)


# ----------------------------------------------------------------------------------------------- Q/L replica
def independent_QL(rays, tau, Kinv, V):
    nq = rays.shape[0]
    x = rays @ tau
    pref = 8 * math.pi / V ** 2
    Q = [rays[a] for a in range(nq)]
    sign = [1.0] * nq
    logl = [math.log10(pref * x[a]) - 2 * math.pi * x[a] * LOG10E for a in range(nq)]
    for i in range(nq - 1):
        for j in range(i + 1, nq):
            amp = pref * (math.pi * float(rays[i] @ Kinv @ rays[j]) + x[i] + x[j])
            Q.append(rays[j] - rays[i])
            sign.append(float(np.sign(amp)))
            logl.append(math.log10(abs(amp)) - 2 * math.pi * (x[i] + x[j]) * LOG10E)
    return np.asarray(Q, np.int64).T, np.vstack([sign, logl])


# ----------------------------------------------------------------------------------------------- main audit
def audit(path, run_root, mat_root):
    t_start = time.time()
    r = {"file": os.path.relpath(path, run_root)}
    f = h5py.File(path, "r")
    meta = json.loads(f.attrs["construction_metadata_json"])
    g = f["cytools/geometric"]
    rd = lambda n: g[n][()]
    h11 = int(rd("h11")); r["h11"] = h11
    pts, simp = np.asarray(rd("points")), np.asarray(rd("simplices"))
    tip = np.asarray(rd("tip"), float); tip_pre = np.asarray(rd("tip_pre_normalization"), float)
    pref = np.asarray(rd("tip_prefactor"), float); m_stored = float(pref[1]); ang = float(pref[0])
    kappa = np.asarray(rd("kappa"), float); glsm = np.asarray(rd("glsm"), np.int64)
    basis = np.asarray(rd("basis")); rays = np.asarray(rd("effective_cone"), np.int64)
    Hst = np.asarray(rd("kahler_hyperplanes"), float); mori_st = np.asarray(rd("mori_cone"), float)
    V_stored = float(rd("CY_volume")); V_pre_stored = float(rd("CY_volume_pre_normalization"))
    ev = g["divisor_volume_evidence"]
    ev_names = list(ev.keys())
    r["evidence_datasets"] = ev_names
    scan = meta["kaehler_point_scan"]
    sel = int(scan["selected_attempt_index"])
    seld = [d for d in scan["diagnostics"] if int(d.get("attempt_index", -1)) == sel and d.get("point_status") == "accepted"]
    r.update(moduli_policy=meta["moduli_policy"], sampler=meta["sampling"]["scheme"], selected_attempt=sel,
             point_kind=seld[0]["point_kind"] if seld else None, solver=seld[0].get("solver") if seld else None,
             n_points_tried=int(scan["attempted_point_count"]), status_counts=scan["status_counts"],
             tip_solver=meta["mosek_license"]["tip_solver"], projection_solvers=meta["mosek_license"]["projection_solvers"],
             m_stored=m_stored, angular_scale=ang, qcd_index_stored=meta.get("qcd_divisor_index"),
             qcd_window=meta.get("qcd_divisor_volume_window"), point_seed=scan["point_seed"],
             polytope_id=meta["polytope_id"], triangulation_id=meta["triangulation_id"], nq=int(rays.shape[0]))

    # ---- single authoritative final point
    r["tip_equals_m_times_pre"] = rel(tip, m_stored * tip_pre)

    # ---- rebuild the CY (Phase 13 identity + independent engine)
    poly = Polytope(pts, deterministic_glsm_basis=True)
    tri = poly.triangulate(simplices=simp, check_input_simplices=False)
    cy = tri.get_cy()
    kc = cy.toric_kahler_cone(); H = np.asarray(kc.hyperplanes(), float)
    r["rebuild_triangulation_id_match"] = mg.topology_identity(meta["polytope_id"], tri, mg.extract_topology(cy, tri))[0] == meta["triangulation_id"]
    r["rebuild_basis_match"] = bool(np.array_equal(np.asarray(cy.divisor_basis()), basis))
    r["rebuild_glsm_match"] = bool(np.array_equal(np.asarray(cy.glsm_charge_matrix(include_origin=False)), glsm))
    kc_coo = np.asarray(cy.intersection_numbers(in_basis=True, format="coo"), float)
    r["rebuild_kappa_match"] = bool(kc_coo.shape == kappa.shape and np.array_equal(kc_coo, kappa))
    r["rebuild_hyperplanes_match"] = bool(H.shape == Hst.shape and np.array_equal(H, Hst))
    raw_rays = np.rint(np.asarray(cy.toric_effective_cone().rays(), float)).astype(np.int64)
    r["raw_ray_count"] = int(raw_rays.shape[0]); r["duplicate_rays_removed"] = int(raw_rays.shape[0] - rays.shape[0])
    r["rays_equal_np_unique_of_raw"] = bool(np.array_equal(np.unique(raw_rays, axis=0), rays))
    r["rays_already_sorted_in_raw_order"] = bool(np.array_equal(first_occurrence_unique(raw_rays), rays))
    # basis check: every effective ray is a GLSM column (prime divisor class) or a nonneg combination
    glsm_cols = {tuple(c) for c in glsm.T.tolist()}
    r["rays_that_are_prime_classes"] = int(sum(tuple(q) in glsm_cols for q in rays.tolist()))

    # ---- Kähler-point provenance (Phase 2/4)
    ref_tip_now = np.asarray(kc.tip_of_stretched_cone(1.0), float)
    r["min_slack_tip_now"] = float(np.min(H @ ref_tip_now))
    cand = tip_pre / ang
    if r["point_kind"] == "canonical_tip":
        r["canonical_tip_reproduced_rel"] = rel(cand, ref_tip_now)
    else:
        gen = mg.sample_stretched_kaehler_points(kc, ref_tip_now, None, sel, lambda *_: None, [],
                                                 point_seed=int(scan["point_seed"]), include_metadata=True)
        prop = None
        for p in gen:
            if int(p["attempt_index"]) == sel:
                prop = p
        r["projected_point_reproduced_rel"] = None if prop is None else rel(cand, prop["point"])
        r["projected_point_reproduced_solver"] = None if prop is None else prop["solver"]
        # solver dependence of the same projection target
        from qpsolvers import solve_qp
        from scipy.sparse import csc_matrix, eye as speye
        seed = mg.stable_seed("kaehler-point-attempt", int(scan["point_seed"]), sel)
        rng = np.random.default_rng(seed)
        d = rng.normal(size=h11); d /= max(float(np.linalg.norm(d)), np.finfo(float).tiny)
        target = max(float(np.linalg.norm(ref_tip_now)), 1.0) * (2.0 ** rng.uniform(-1.0, 4.0)) * d
        sol = {}
        for s in ("cvxopt", "osqp", "highs"):
            try:
                x = solve_qp(speye(h11, format="csc"), -target, G=-csc_matrix(H), h=-np.ones(H.shape[0]), solver=s, verbose=False)
                sol[s] = None if x is None else np.asarray(x, float)
            except Exception:
                sol[s] = None
        r["projection_solver_rel_diff"] = {s: (None if v is None else rel(v, cand)) for s, v in sol.items()}
        r["projection_solver_min_slack"] = {s: (None if v is None else float(np.min(H @ v))) for s, v in sol.items()}
        r["projection_target_norm_over_tip"] = float(np.linalg.norm(target) / np.linalg.norm(ref_tip_now))

    # ---- evaluate at the pre-normalization point (m decomposition, Phase 2/3/5)
    tau0 = np.asarray(cy.compute_divisor_volumes(tip_pre, in_basis=True), float)
    prime0 = np.asarray(cy.compute_divisor_volumes(tip_pre), float)
    Kraw0 = np.asarray(cy.compute_inverse_kahler_metric(tip_pre), float); K0 = 0.5 * (Kraw0 + Kraw0.T)
    x0 = rays @ tau0; G0 = rays @ K0 @ rays.T
    il, jl = np.tril_indices(rays.shape[0], -1)
    r["pre_min_slack"] = float(np.min(H @ tip_pre)); r["pre_min_prime_vol"] = float(prime0.min())
    r["pre_min_eff_vol"] = float(x0.min())
    m21 = g2_search(G0, x0, il, jl)
    r["m_eq21_g2_sorted"] = m21
    r["m_eq21_grid_first_pass"] = grid_first_pass(G0, x0, il, jl, m21) if m21 is not None else None
    raw_u = first_occurrence_unique(raw_rays)
    ilr, jlr = np.tril_indices(raw_u.shape[0], -1)
    r["m_eq21_raw_cytools_order"] = g2_search(raw_u @ K0 @ raw_u.T, raw_u @ tau0, ilr, jlr)
    r["m_eq21_two_sided"] = g2_search(G0, x0, il, jl, two_sided=True)
    m_prime = math.sqrt(1.0 / float(prime0.min()))
    r["m_prime"] = m_prime
    qmin, qmax = [float(v) for v in meta["qcd_divisor_volume_window"]]
    chosen = None
    if r["moduli_policy"] == "canonical_qcd":
        # replica of select_canonical_qcd_candidate (default m >= 1 branch, no explicit index)
        cands = sorted([(c, float(v)) for c, v in enumerate(prime0) if np.isfinite(v) and v > 0 and math.sqrt(40.0 / v) >= 1.0],
                       key=lambda cv: (-cv[1], cv[0]))
        for c, v in cands:
            mm = math.sqrt(40.0 / v)
            if (mm * mm * prime0).min() >= 1 - 1e-8 and (rays @ (mm * mm * tau0)).min() >= 1 - 1e-8:
                chosen = (c, mm); break
    for c, v in (enumerate(prime0) if r["moduli_policy"] != "canonical_qcd" else []):
        lower = max(math.sqrt(qmin / v), m21, m_prime); upper = math.sqrt(qmax / v)
        if lower <= upper and lower <= 1e6 and eq21_ok(G0, x0, lower, il, jl):
            chosen = (c, lower); break
    r["replica_qcd_index"] = None if chosen is None else chosen[0]
    r["replica_m"] = None if chosen is None else chosen[1]
    r["replica_matches_stored"] = bool(chosen is not None and chosen[0] == r["qcd_index_stored"]
                                       and abs(chosen[1] - m_stored) <= 1e-12 * m_stored)
    if chosen is not None and r["moduli_policy"] == "canonical_qcd":
        r["binding_constraint"] = "qcd_target_40"
        r["m_without_qcd"] = max(1.0, m_prime)
        r["qcd_extra_dilation"] = m_stored / r["m_without_qcd"]
        r["eq21_satisfied_at_final_if_checked"] = eq21_ok(G0, x0, m_stored, il, jl)
        r["m_eq21_exceeds_final_m"] = bool(m21 is not None and m21 > m_stored * (1 + 1e-12))
        r["qcd_index_is_first_prime"] = bool(chosen[0] == 0)
        r["qcd_tip_volume_rank_desc"] = int(np.argsort(-prime0).tolist().index(chosen[0]))
    elif chosen is not None:
        c = chosen[0]
        terms = {"eq21": m21, "prime_bound": m_prime, "qcd_lower_edge": math.sqrt(qmin / prime0[c])}
        r["binding_constraint"] = max(terms, key=terms.get)
        r["m_without_qcd"] = max(1.0, m21, m_prime)
        r["qcd_extra_dilation"] = m_stored / r["m_without_qcd"]
        r["n_prime_reachable_window"] = int(sum(
            max(math.sqrt(qmin / v), m21, m_prime) <= math.sqrt(qmax / v) for v in prime0))
        r["qcd_index_is_first_prime"] = bool(c == 0)
        r["qcd_tip_volume_rank_desc"] = int(np.argsort(-prime0).tolist().index(c))

    # ---- final point: three engines (Phase 6/7)
    V_c = float(cy.compute_cy_volume(tip)); tau_c = np.asarray(cy.compute_divisor_volumes(tip, in_basis=True), float)
    prime_c = np.asarray(cy.compute_divisor_volumes(tip), float)
    Kraw = np.asarray(cy.compute_inverse_kahler_metric(tip), float)
    r["cytools_kinv_raw_asym"] = rel(Kraw, Kraw.T) if np.any(Kraw) else None
    Kc = 0.5 * (Kraw + Kraw.T)
    ex = mg._reconstruct_intersection_geometry(kappa, tip)
    V_e, tau_e, Kinv_e = ex["cy_volume"], np.asarray(ex["divisor_volumes"]), np.asarray(ex["inverse_metric"])
    kd = kappa_dense(kappa, h11)
    Kind, V_i, tau_i = independent_K(kd, tip)
    r.update(V_final=V_e, V_stored_vs_exact=abs(V_stored - V_e) / V_e, V_cytools_vs_exact=abs(V_c - V_e) / V_e,
             V_pre_stored_vs_exact=abs(V_pre_stored - mg._reconstruct_intersection_geometry(kappa, tip_pre)["cy_volume"]) / V_pre_stored,
             V_scaling_check=abs(V_e / V_pre_stored - m_stored ** 3) / m_stored ** 3,
             tau_cytools_vs_exact=rel(tau_c, tau_e), tau_indep_vs_exact=rel(tau_i, tau_e),
             kinv_cytools_vs_exact=rel(Kc, Kinv_e), kinv_scaling_m4=rel(Kinv_e, m_stored ** 4 * K0),
             tau_scaling_m2=rel(tau_e, m_stored ** 2 * tau0),
             KKinv_minus_I_exact=float(np.max(np.abs(Kind @ Kinv_e - np.eye(h11)))),
             KKinv_minus_I_cytools=float(np.max(np.abs(Kind @ Kc - np.eye(h11)))),
             kinv_exact_asym=rel(Kinv_e, Kinv_e.T), kinv_eig_min=float(np.linalg.eigvalsh(Kinv_e).min()),
             kinv_cond=float(np.linalg.cond(Kinv_e)), tau_min=float(tau_e.min()), tau_max=float(tau_e.max()))
    prime_e = glsm.T @ tau_e; eff_e = rays @ tau_e
    r["prime_glsm_vs_cytools_direct"] = rel(prime_e, prime_c)
    r.update(final_min_slack=float(np.min(H @ tip)), final_min_curve=float(np.min(mori_st @ tip)),
             final_min_prime=float(prime_e.min()), final_min_eff=float(eff_e.min()),
             qcd_volume_final=float(prime_e[r["qcd_index_stored"]]) if r["qcd_index_stored"] is not None else None,
             n_prime_in_window_final=int(np.sum((prime_e >= qmin - 1e-8) & (prime_e <= qmax + 1e-8))))
    # divisor_volume_evidence vs recomputation
    for name in ev_names:
        arr = np.asarray(ev[name][()])
        if arr.dtype.kind == "f" and arr.ndim == 1:
            if arr.size == prime_e.size and "prime" in name:
                r[f"evidence_{name}_vs_recomputed"] = rel(arr, prime_e if "final" in name or "pre" not in name else m_stored ** -2 * prime_e)
            elif arr.size == eff_e.size and "effective" in name:
                r[f"evidence_{name}_vs_recomputed"] = rel(arr, eff_e if "final" in name or "pre" not in name else m_stored ** -2 * eff_e)
    # Eq. 21 at the final point with FRESH quantities (stale-state test)
    Gf = rays @ Kinv_e @ rays.T; xf = rays @ tau_e
    r["eq21_final_fresh_onesided"] = eq21_ok(Gf, xf, 1.0, il, jl)
    r["eq21_final_fresh_twosided"] = eq21_ok(Gf, xf, 1.0, il, jl, two_sided=True)
    if rays.shape[0] > 1:
        # log space: exp(-2 pi x) underflows for x > ~113
        with np.errstate(divide="ignore"):
            log10C = np.log10(math.pi * np.abs(Gf[il, jl])) - 2 * math.pi * (xf[il] + xf[jl]) * LOG10E
        log10A = np.log10(xf) - 2 * math.pi * xf * LOG10E
        margin_i = log10A[il] - log10C
        margin_j = log10A[jl] - log10C
        r["eq21_pairs_with_C_ge_1"] = int(np.sum(log10C >= 0))
        r["eq21_min_margin_dex_onesided"] = float(margin_i.min()); r["eq21_min_margin_dex_otherside"] = float(margin_j.min())
        r["eq21_pairs_violating_otherside"] = int(np.sum(margin_j <= 0))
        r["eq21_worst_otherside_violation_dex"] = float(-margin_j.min()) if margin_j.min() < 0 else 0.0

    # ---- Q/L (Phase 8): G2 routine vs independent implementation vs CYTools-engine inputs
    ref = {"h11": h11, "effective_cone": rays, "kappa": kappa, "tip": tip, "basis_matrix": np.asarray(rd("basis_matrix")),
           "glsm": glsm, "prime_toric_divisors": np.asarray(rd("prime_toric_divisors"))}
    terms = mg._geometry_potential_terms(ref)
    qg, lg = np.asarray(terms["q"], np.int64), np.asarray(terms["l"], float)
    qi, li = independent_QL(rays, tau_e, Kinv_e, V_e)
    qc, lc = independent_QL(rays, tau_c, Kc, V_c)
    nq = rays.shape[0]
    r.update(N_terms=int(qg.shape[1]), N_expected=nq + nq * (nq - 1) // 2, Q_shape=list(qg.shape),
             Q_equal_independent=bool(np.array_equal(qg, qi)), L_sign_equal_independent=bool(np.array_equal(lg[0], li[0])),
             L_log_maxdiff_independent=float(np.max(np.abs(lg[1] - li[1]))),
             L_sign_equal_cytools_engine=bool(np.array_equal(lg[0], lc[0])),
             L_log_maxdiff_cytools_engine=float(np.max(np.abs(lg[1] - lc[1]))),
             Q_zero_columns=int(np.sum(~np.any(qg != 0, axis=0))),
             Q_distinct_columns=int(np.unique(qg.T, axis=0).shape[0]),
             L_negative_terms=int(np.sum(lg[0] < 0)), L_finite=bool(np.all(np.isfinite(lg))),
             L_max=float(lg[1].max()), L_min=float(lg[1].min()),
             L_leading5=[round(v, 3) for v in sorted(lg[1], reverse=True)[:5]])
    # cancellation inside the pair amplitude pi q_i Kinv q_j + (q_i+q_j).tau (explains engine-to-engine dex differences)
    if nq > 1:
        pi_, pj_ = np.triu_indices(nq, 1)
        met = math.pi * np.einsum("ai,ij,aj->a", rays[pi_], Kinv_e, rays[pj_]); ts = (rays[pi_] + rays[pj_]) @ tau_e
        canc = np.abs(met + ts) / (np.abs(met) + np.abs(ts))
        r["pair_amplitude_min_cancellation_ratio"] = float(canc.min())
        k = int(np.argmax(np.abs(lg[1] - lc[1]))); r["L_engine_maxdiff_term"] = k
        r["L_engine_maxdiff_term_cancellation"] = float(canc[k - nq]) if k >= nq else 1.0
    # stale-state sensitivity: L built at the pre-normalization point differs by this much (dex)
    if abs(m_stored - 1) > 1e-12:
        exp = mg._reconstruct_intersection_geometry(kappa, tip_pre)
        _, lpre = independent_QL(rays, np.asarray(exp["divisor_volumes"]), np.asarray(exp["inverse_metric"]), exp["cy_volume"])
        r["L_if_stale_pre_point_maxdiff_dex"] = float(np.max(np.abs(lpre[1] - lg[1])))
    f.close()

    # ---- materialized datasets (Phase 9)
    if mat_root:
        mp = os.path.join(mat_root, r["file"])
        if os.path.exists(mp):
            with h5py.File(mp, "r") as fm:
                gm = fm["cytools/geometric"]; pm = fm["cytools/potential"]
                Qm = np.asarray(pm["Q"][()]); Lm = np.asarray(pm["L"][()])
                r.update(mat_Q_disk_shape=list(Qm.shape), mat_L_disk_shape=list(Lm.shape),
                         mat_Q_equal_G2=bool(np.array_equal(Qm.T, qg)), mat_L_bitwise_G2=bool(np.array_equal(Lm.T, lg)),
                         mat_L_vs_independent_dex=float(np.max(np.abs(Lm.T[1] - li[1]))),
                         mat_tau_vs_cytools=rel(np.asarray(gm["divisor_volumes"][()]), tau_c),
                         mat_kinv_vs_cytools=rel(np.asarray(gm["Kinv"][()]), Kc),
                         mat_kinv_vs_exact_bitwise=bool(np.array_equal(np.asarray(gm["Kinv"][()]), Kinv_e)),
                         mat_prime_vs_recomputed=rel(np.asarray(gm["prime_divisor_volumes"][()]), prime_e),
                         mat_tip_bitwise=bool(np.array_equal(np.asarray(gm["tip"][()]), tip)),
                         mat_V_stored_bitwise=bool(float(gm["CY_volume"][()]) == V_stored))
        else:
            r["mat_missing"] = True
    r["seconds"] = round(time.time() - t_start, 2)
    return r


def worker(args):
    path, run_root, mat_root = args
    try:
        return audit(path, run_root, mat_root)
    except Exception as exc:
        return {"file": os.path.relpath(path, run_root), "audit_error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:]}


if __name__ == "__main__":
    run_root, out = sys.argv[2], sys.argv[3]
    rest = sys.argv[4:]
    workers = 1
    if "--workers" in rest:
        k = rest.index("--workers"); workers = int(rest[k + 1]); rest = rest[:k] + rest[k + 2:]
    mat_root = rest[0] if rest else None
    files = sorted(os.path.join(dp, fn) for dp, _, fns in os.walk(run_root) for fn in fns if fn == "cyax.h5")
    jobs = [(p, run_root, mat_root) for p in files]
    with open(out, "w") as fh:
        if workers > 1:
            from multiprocessing import Pool
            with Pool(workers) as pool:
                for n, res in enumerate(pool.imap_unordered(worker, jobs, chunksize=2)):
                    fh.write(json.dumps(res, default=float) + "\n"); fh.flush()
                    if n % 100 == 0:
                        print(n, res.get("file"), res.get("seconds"), res.get("audit_error", ""), flush=True)
        else:
            for n, j in enumerate(jobs):
                res = worker(j)
                fh.write(json.dumps(res, default=float) + "\n"); fh.flush()
                print(n, res.get("file"), res.get("seconds"), res.get("audit_error", ""), flush=True)
    print("done", len(jobs))
