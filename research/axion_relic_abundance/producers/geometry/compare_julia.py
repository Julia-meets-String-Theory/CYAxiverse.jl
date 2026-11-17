"""Phase 10 comparator: Julia-interpreted values (julia_crosscheck.jl) vs Python canonical G2 values.

For each geometry: Python canonical Q (h11 x N) and L (2 x N) from G2's _geometry_potential_terms at the stored
final tip, Kinv from G2's _reconstruct_intersection_geometry, h5py on-disk shapes of the materialized file, and the
Julia raw/reader shapes.  Sampled Q/L columns and q_i' Kinv q_j contractions must agree (Q exactly).
Usage: python compare_julia.py REPO_SCRIPTS SRC_ROOT[,SRC_ROOT2] MAT_ROOT JULIA_JSONL OUT_JSONL
"""
import json, math, os, sys
import numpy as np, h5py

sys.path.insert(0, sys.argv[1])
import generate_geometric_data_multitriangulation as mg  # noqa: E402

src_roots, mat_root, jl, out = sys.argv[2:6]
src_roots = src_roots.split(",")  # one or more raw G2 roots (comma-separated)
find_src = lambda rel: next(os.path.join(r, rel) for r in src_roots if os.path.exists(os.path.join(r, rel)))
rows = []
for line in open(jl):
    j = json.loads(line)
    r = {"file": j["file"], "h11": j["h11"]}
    if "error" in j:
        r["julia_error"] = j["error"]; rows.append(r); continue
    with h5py.File(find_src(j["file"]), "r") as f:
        g = f["cytools/geometric"]
        ref = {k: g[k][()] for k in ("effective_cone", "kappa", "tip", "basis_matrix", "glsm", "prime_toric_divisors")}
        ref["h11"] = int(g["h11"][()]); V_stored = float(g["CY_volume"][()])
    terms = mg._geometry_potential_terms(ref)
    q, l = np.asarray(terms["q"], np.int64), np.asarray(terms["l"], float)
    geo = terms["geometry"]; K = np.asarray(geo["inverse_metric"]); tau = np.asarray(geo["divisor_volumes"])
    with h5py.File(os.path.join(mat_root, j["file"]), "r") as f:
        disk = {"Q": list(f["cytools/potential/Q"].shape), "L": list(f["cytools/potential/L"].shape),
                "Kinv": list(f["cytools/geometric/Kinv"].shape), "effective_cone": list(f["cytools/geometric/effective_cone"].shape)}
    cols = [c - 1 for c in j["sample_columns"]]
    qj = np.asarray(j["sample_Q"], np.int64).T; lj = np.asarray(j["sample_L"], float).T
    pairs = [(a - 1, b - 1) for a, b in j["sample_pairs"]]
    qkq_py = np.asarray([q[:, a] @ K @ q[:, b] for a, b in pairs])
    qkq_jl = np.asarray(j["sample_qKq"])
    # a transposed Kinv or a transposed/misread Q would change these contractions
    qkq_py_KT = np.asarray([q[:, a] @ K.T @ q[:, b] for a, b in pairs])
    r.update(python_Q_shape=list(q.shape), python_L_shape=list(l.shape), h5py_disk_shapes=disk,
             julia_raw_shapes=j["julia_raw_shapes"], julia_reader_Q=j["reader_Q_size"], julia_reader_L=j["reader_L_size"],
             Q_columns_exact=bool(np.array_equal(qj, q[:, cols])),
             L_columns_maxdiff=float(np.max(np.abs(lj - l[:, cols]))), L_sign_equal=bool(np.array_equal(lj[0], l[0, cols])),
             qKq_rel_maxdiff=float(np.max(np.abs(qkq_jl - qkq_py) / np.maximum(np.abs(qkq_py), 1e-300))),
             qKq_KT_vs_K_rel=float(np.max(np.abs(qkq_py_KT - qkq_py) / np.maximum(np.abs(qkq_py), 1e-300))),
             tip_bitwise=bool(np.array_equal(np.asarray(j["tip"]), np.asarray(ref["tip"], float))),
             tau_rel=float(np.max(np.abs(np.asarray(j["divisor_volumes"]) - tau) / np.abs(tau))),
             V_julia_equals_stored=bool(j["CY_volume"] == V_stored),
             direct_columns_equal_effective_cone=j["direct_columns_equal_effective_cone"],
             Kinv_symmetric=j["Kinv_raw_symmetric"], cholesky_residual=j["cholesky_residual"],
             oriented_equals_raw=bool(j["oriented_Q_equals_raw"] and j["oriented_L_equals_raw"]),
             geometry_Kinv_equals_factored=j["geometry_Kinv_equals_factored_Kinv"])
    rows.append(r)
with open(out, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
ok = [r for r in rows if "julia_error" not in r]
print("geometries", len(rows), "julia errors", len(rows) - len(ok))
if ok:
    print("Q columns exact", sum(r["Q_columns_exact"] for r in ok), "/", len(ok))
    print("L sign equal", sum(r["L_sign_equal"] for r in ok), "; L max diff", max(r["L_columns_maxdiff"] for r in ok))
    print("qKq max rel diff", max(r["qKq_rel_maxdiff"] for r in ok), "; tip bitwise", sum(r["tip_bitwise"] for r in ok),
          "; tau max rel", max(r["tau_rel"] for r in ok), "; V equal stored", sum(r["V_julia_equals_stored"] for r in ok))
    print("direct cols == effective cone", sum(r["direct_columns_equal_effective_cone"] for r in ok),
          "; Kinv symmetric", sum(r["Kinv_symmetric"] for r in ok), "; chol resid max", max(r["cholesky_residual"] for r in ok),
          "; oriented==raw", sum(r["oriented_equals_raw"] for r in ok))
    shapes = {(tuple(r["python_Q_shape"]) == tuple(r["julia_reader_Q"]), tuple(r["h5py_disk_shapes"]["Q"]) == tuple(r["python_Q_shape"][::-1]),
               tuple(r["julia_raw_shapes"]["Q"]) == tuple(r["python_Q_shape"])) for r in ok}
    print("shape pattern (py==julia reader, disk==py^T, julia raw==py):", shapes)
