"""Compare a (subset) stage-2 run of this repository with the frozen dataset, dataset by dataset.

For every geometry produced under WORK/geometry/run_*/ the same geometry id must exist in the frozen unified view and
  * raw G2 file      : every HDF5 dataset under cytools/ identical (values and dtype)
  * materialized file: divisor_volumes, Kinv, prime_divisor_volumes, potential/Q, potential/L identical
  * spectrum file    : every dataset under spectrum/physical/ except run-time metadata identical
Polytopes rejected by G2 must be the same as in the frozen run. HDF5 attributes and run-time metadata (timestamps,
paths, runtimes, code paths) are reported but not required to match.
File label spectrum/physical/metadata/polytope: batch_physical_spectrum.jl stores the np_ directory index. The frozen
tranche-2 spectra were computed in tranche-local directories, the unified run uses unified positions; a difference is
accepted only if the new value equals the unified position AND the frozen value equals the tranche-local index of
unified_polytopes.json. Such cases are listed under label_differences; any other difference is a problem.
Usage: python compare_geometry_subset.py WORK FROZEN_UNIFIED_VIEW OUT_JSON
       FROZEN_UNIFIED_VIEW = directory with geometry/, materialized/, spectra/ in the h11_XXX/np_POSITION/cy_0000001 layout
"""
import glob, json, os, sys

import h5py
import numpy as np

work, frozen, out = sys.argv[1:4]
RUNTIME = {"runtime_seconds", "created_utc", "timestamp", "code_git_revision", "julia_version", "hostname", "data_dir", "source_path"}


def datasets(path, prefix):
    out = {}
    with h5py.File(path, "r") as f:
        if prefix not in f:
            return out
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[f"{prefix}/{name}"] = obj[()]
        f[prefix].visititems(visit)
    return out


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "SUO" or b.dtype.kind in "SUO":
        return a.tolist() == b.tolist()
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)


rows, problems, labels = [], [], []
LABEL = "spectrum/physical/metadata/polytope"
UP = {(p["h11"], p["position"]): p for p in json.load(open(os.path.join(frozen, "manifests", "unified_polytopes.json")))["polytopes"]}
raws = sorted(glob.glob(os.path.join(work, "geometry", "run_*", "h11_*", "np_*", "cy_*", "cyax.h5")))
for raw in raws:
    gid = os.path.relpath(raw, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(raw)))))
    rec = {"geometry_id": os.path.dirname(gid)}
    for kind, new, old, prefixes in (("raw", raw, os.path.join(frozen, "geometry", gid), ["cytools"]),
                                     ("materialized", os.path.join(work, "materialized", gid), os.path.join(frozen, "materialized", gid),
                                      ["cytools/geometric", "cytools/potential"]),
                                     ("spectrum", os.path.join(work, "spectra", gid), os.path.join(frozen, "spectra", gid), ["spectrum/physical"])):
        if not os.path.exists(old):
            problems.append((gid, kind, "missing in frozen dataset")); continue
        if not os.path.exists(new):
            problems.append((gid, kind, "missing in new run")); continue
        A = {k: v for p in prefixes for k, v in datasets(new, p).items()}
        B = {k: v for p in prefixes for k, v in datasets(old, p).items()}
        if kind == "materialized":
            keep = ("divisor_volumes", "Kinv", "prime_divisor_volumes", "potential/Q", "potential/L")
            A = {k: v for k, v in A.items() if k.endswith(keep)}; B = {k: v for k, v in B.items() if k.endswith(keep)}
        meta = lambda k: k.split("/")[-1] in RUNTIME
        diff = sorted(k for k in set(A) | set(B) if not meta(k) and (k not in A or k not in B or not same(A[k], B[k])))
        if LABEL in diff and LABEL in A and LABEL in B:
            h, pos = (int(x.split("_")[1]) for x in gid.split("/")[:2])
            if int(A[LABEL]) == pos and int(B[LABEL]) == UP[(h, pos)]["local_index"] != pos:
                diff.remove(LABEL); labels.append({"geometry_id": rec["geometry_id"], "new": int(A[LABEL]), "frozen": int(B[LABEL])})
        rec[kind] = {"datasets_compared": len([k for k in A if not meta(k)]), "identical": not diff, "different": diff[:20]}
        if diff:
            problems.append((gid, kind, diff[:5]))
    rows.append(rec)

# rejected polytopes: same set as the frozen run for the same (h11, position) range
new_ids = {r["geometry_id"].rsplit("/", 1)[0] for r in rows}
h11s = {g.split("/")[0] for g in new_ids}
status = [l for l in open(os.path.join(work, "validation", "status", "polytope_status.csv"))][1:]
attempted = {f"h11_{int(l.split(',')[0]):03d}/np_{int(l.split(',')[1]):07d}" for l in status}
frozen_acc = {os.path.relpath(d, os.path.join(frozen, "geometry")) for h in h11s for d in glob.glob(os.path.join(frozen, "geometry", h, "np_*"))}
acc_same = new_ids == (frozen_acc & attempted)
res = {"geometries_compared": len(rows), "accepted_set_identical_to_frozen": acc_same,
       "all_identical": not problems and acc_same, "problems": problems[:20],
       "label_differences_count": len(labels), "label_differences": labels, "geometries": rows}
json.dump(res, open(out, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("geometries", "label_differences")}, indent=1, default=str))
sys.exit(0 if res["all_identical"] else 1)
