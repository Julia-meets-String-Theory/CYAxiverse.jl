#!/usr/bin/env python3
"""Make schema-1.1 cyax.h5 files from generate_geometric_data_multitriangulation.py readable by CYAxiverse.jl.

This driver contains NO physics.  All materialized quantities come from the repository's own bridge,
``scripts/build_orientifold_axion_database.py::materialize_potential_and_kinetic_data``, which reconstructs
``divisor_volumes``, ``Kinv``, ``prime_divisor_volumes`` and ``potential/{Q,L}`` from the stored ``kappa`` and final
``tip`` with the generator's own ``_reconstruct_intersection_geometry`` and ``_geometry_potential_terms``, and writes
Q/L transposed on disk so that CYAxiverse.jl's raw readers see Q as h11 x N and L as 2 x N.

What this driver adds: it never touches the source file (it materializes a copy), refuses to overwrite, checks the
input is a genuine reconstruct-on-demand artifact, verifies afterwards that every pre-existing dataset and attribute
is unchanged and that the new arrays are well formed, and records provenance in a root ``julia_materialization`` group.

Usage:
  python materialize_cyax_for_julia.py --repo /path/to/CYAxiverse.jl SRC_ROOT DST_ROOT [--allow-dirty-repo]
SRC_ROOT and DST_ROOT use the h11_XXX/np_YYYYYYY/cy_ZZZZZZZ/cyax.h5 layout; DST_ROOT is what CYAXIVERSE_DATA_DIR points to.
"""
import argparse, datetime, glob, hashlib, importlib.metadata, json, os, platform, shutil, subprocess, sys

import h5py
import numpy as np

DRIVER_SCHEMA = "cyax-julia-materialization-1.0"
REQUIRED = ("kappa", "tip", "glsm", "prime_toric_divisors", "effective_cone", "basis_matrix", "h11", "h21",
            "CY_volume", "basis", "tip_prefactor", "points", "simplices")
NEW_GEOMETRIC = ("divisor_volumes", "Kinv", "prime_divisor_volumes")
NEW_POTENTIAL = ("Q", "L")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout.strip()


def _canon(value):
    a = np.asarray(value)
    if a.dtype == object or a.dtype.kind in "SUO":
        return repr(a.tolist()).encode()
    return a.tobytes() + repr((a.shape, str(a.dtype))).encode()


def snapshot(path):
    """Every dataset value and attribute in the file, keyed by HDF5 path, for the preservation check."""
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            for k, v in obj.attrs.items():
                out[f"{name}@{k}"] = _canon(v)
            if isinstance(obj, h5py.Dataset):
                out[name] = _canon(obj[()])
        for k, v in f.attrs.items():
            out[f"/@{k}"] = _canon(v)
        f.visititems(visit)
    return out


def preflight(path, h11_from_path):
    with h5py.File(path, "r") as f:
        if "schema_version" not in f.attrs:
            raise RuntimeError("not a generator artifact: missing root schema_version attribute")
        pot = f.get("cytools/potential")
        if pot is None or pot.attrs.get("storage_schema") != "reconstruct_on_demand":
            raise RuntimeError("cytools/potential is not a reconstruct_on_demand group")
        g = f["cytools/geometric"]
        missing = [k for k in REQUIRED if k not in g]
        if missing:
            raise RuntimeError(f"missing reconstruction references: {missing}")
        present = [k for k in NEW_GEOMETRIC if k in g] + [f"potential/{k}" for k in NEW_POTENTIAL if k in pot]
        if present:
            raise RuntimeError(f"already materialized fields present: {present}")
        h11 = int(g["h11"][()])
        if h11 != h11_from_path:
            raise RuntimeError(f"h11 dataset {h11} disagrees with directory h11 {h11_from_path}")
        rays = np.asarray(g["effective_cone"][()])
        tip = np.asarray(g["tip"][()], float)
        if rays.ndim != 2 or rays.shape[1] != h11 or tip.shape != (h11,) or not np.all(np.isfinite(tip)):
            raise RuntimeError(f"inconsistent shapes: effective_cone {rays.shape}, tip {tip.shape}, h11 {h11}")
        if np.unique(rays, axis=0).shape[0] != rays.shape[0]:
            raise RuntimeError("effective_cone contains duplicate rays")
        return dict(schema_version=str(f.attrs["schema_version"]), h11=h11, nq=int(rays.shape[0]))


def postcheck(path, info):
    h11, nq = info["h11"], info["nq"]
    nterms = nq + nq * (nq - 1) // 2
    with h5py.File(path, "r") as f:
        g, p = f["cytools/geometric"], f["cytools/potential"]
        tau = np.asarray(g["divisor_volumes"][()]); K = np.asarray(g["Kinv"][()])
        Qd = np.asarray(p["Q"][()]); Ld = np.asarray(p["L"][()])
        pdv = np.asarray(g["prime_divisor_volumes"][()])
    problems = []
    if tau.shape != (h11,): problems.append(f"divisor_volumes shape {tau.shape}")
    if K.shape != (h11, h11): problems.append(f"Kinv shape {K.shape}")
    if Qd.shape != (nterms, h11): problems.append(f"Q on-disk shape {Qd.shape}, expected {(nterms, h11)}")
    if Ld.shape != (nterms, 2): problems.append(f"L on-disk shape {Ld.shape}, expected {(nterms, 2)}")
    for name, a in (("divisor_volumes", tau), ("Kinv", K), ("Q", Qd), ("L", Ld), ("prime_divisor_volumes", pdv)):
        if not np.all(np.isfinite(a)): problems.append(f"{name} has NaN/Inf")
    if not problems:
        if not np.array_equal(K, K.T): problems.append("Kinv not exactly symmetric")
        if np.linalg.eigvalsh(K).min() <= 0: problems.append("Kinv not positive definite")
        if not np.array_equal(Qd, np.rint(Qd)): problems.append("Q not integral")
        if not np.all(np.abs(Ld[:, 0]) == 1): problems.append("L signs not +-1")
    if problems:
        raise RuntimeError("post-materialization check failed: " + "; ".join(problems))
    return dict(nterms=nterms, Q_disk_shape=list(Qd.shape), L_disk_shape=list(Ld.shape))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, help="CYAxiverse.jl checkout providing scripts/build_orientifold_axion_database.py")
    ap.add_argument("--allow-dirty-repo", action="store_true", help="permit uncommitted changes in the repository (recorded)")
    ap.add_argument("src_root"); ap.add_argument("dst_root")
    a = ap.parse_args()

    repo = os.path.abspath(a.repo); scripts = os.path.join(repo, "scripts")
    commit = git(repo, "rev-parse", "HEAD"); dirty = bool(git(repo, "status", "--porcelain", "--", "scripts", "src"))
    if dirty and not a.allow_dirty_repo:
        sys.exit(f"repository {repo} has uncommitted changes in scripts/ or src/; refusing (use --allow-dirty-repo to override)")
    sys.path.insert(0, scripts)
    import build_orientifold_axion_database as bridge  # noqa: E402  (the repository's canonical materializer)

    driver_path = os.path.abspath(__file__)
    common = {
        "schema": DRIVER_SCHEMA,
        "materializer_function": "scripts/build_orientifold_axion_database.py::materialize_potential_and_kinetic_data",
        "reconstruction_backend": "generate_geometric_data_multitriangulation._reconstruct_intersection_geometry(kappa, tip) "
                                  "and _geometry_potential_terms(reference)",
        "on_disk_orientation": "Q stored (N, h11) and L stored (N, 2) by the repository bridge; CYAxiverse.jl raw readers see h11 x N and 2 x N",
        "repo_commit": commit, "repo_dirty": dirty,
        "materializer_module_sha256": sha256(bridge.__file__), "generator_module_sha256": sha256(bridge.mg.__file__),
        "driver_path": driver_path, "driver_sha256": sha256(driver_path),
        "python": platform.python_version(), "numpy": np.__version__, "h5py": h5py.__version__,
        "cytools": importlib.metadata.version("cytools"),
    }
    sources = sorted(glob.glob(os.path.join(a.src_root, "h11_*", "np_*", "cy_*", "cyax.h5")))
    if not sources:
        sys.exit(f"no h11_*/np_*/cy_*/cyax.h5 files under {a.src_root}")
    log = []
    for src in sources:
        rel = os.path.relpath(src, a.src_root)
        dst = os.path.join(a.dst_root, rel)
        rec = {"source": os.path.abspath(src), "destination": os.path.abspath(dst)}
        try:
            if os.path.exists(dst):
                raise RuntimeError("destination exists; refusing to overwrite")
            h11 = int(rel.split(os.sep)[0][4:])
            info = preflight(src, h11)
            src_hash = sha256(src)
            before = snapshot(src)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".materializing"
            shutil.copyfile(src, tmp)
            try:
                bridge.materialize_potential_and_kinetic_data(tmp)
                after = snapshot(tmp)
                changed = sorted(k for k in before if after.get(k) != before[k])
                if changed:
                    raise RuntimeError(f"pre-existing HDF5 content changed: {changed[:5]}")
                shapes = postcheck(tmp, info)
                with h5py.File(tmp, "r+") as f:
                    grp = f.create_group("julia_materialization")
                    for k, v in {**common, "source_file": os.path.abspath(src), "source_sha256": src_hash,
                                 "source_schema_version": info["schema_version"],
                                 "materialized_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}.items():
                        grp.attrs[k] = v
                    for name in NEW_GEOMETRIC:
                        f[f"cytools/geometric/{name}"].attrs["materialized_by"] = "/julia_materialization"
                    for name in NEW_POTENTIAL:
                        f[f"cytools/potential/{name}"].attrs["materialized_by"] = "/julia_materialization"
                if os.path.exists(dst):
                    raise RuntimeError("destination appeared during materialization; refusing to overwrite")
                os.replace(tmp, dst)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            if sha256(src) != src_hash:
                raise RuntimeError("SOURCE FILE CHANGED during materialization")
            rec.update(status="materialized", source_sha256=src_hash, destination_sha256=sha256(dst), **info, **shapes)
        except Exception as exc:
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        log.append(rec)
        print(json.dumps(rec), flush=True)
    failed = [r for r in log if r["status"] != "materialized"]
    if failed:
        sys.exit(f"{len(failed)} of {len(log)} files failed")


if __name__ == "__main__":
    main()
