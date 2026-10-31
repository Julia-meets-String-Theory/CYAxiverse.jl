"""Ordered scan of the Kreuzer-Skarke HuggingFace mirror: for every h11 keep the first K rows in the legacy order.

Memory-bounded and resumable: streams 50k-row batches, works on chunks of CHUNK row groups (checkpointed as
partial_<file>__rg<a>-<b>.json) and reuses completed whole-file checkpoints (partial_<file>.json) when present.
Ordering key per h11 = (md5(DuckDB VARCHAR of the vertex list), VARCHAR). Network: HTTP range reads (about 1 GB of
checkpoints for K = 1000, h11 4-50).
Usage: python ks_ordered_scan.py OUT_DIR [K] [H_MIN] [H_MAX] [WORKERS] [CHUNK]
"""
import hashlib, heapq, json, os, sys, time
from multiprocessing import Pool
import numpy as np, requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_range import open_parquet, REVISION_API  # noqa: E402

OUT = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
HMIN = int(sys.argv[3]) if len(sys.argv) > 3 else 4; HMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 50
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 8; CHUNK = int(sys.argv[6]) if len(sys.argv) > 6 else 4
FILES = [f"polytopes-4d-{v:02d}-vertices.parquet" for v in list(range(5, 34)) + [36]]


def vc(verts):
    return "[" + ", ".join("[" + ", ".join(str(c) for c in v) + "]" for v in verts) + "]"


class Neg:
    __slots__ = ("k",)

    def __init__(self, k):
        self.k = k

    def __lt__(self, o):
        return self.k > o.k


def unit(args):
    name, a, b = args
    part = os.path.join(OUT, f"partial_{name}__rg{a:04d}-{b:04d}.json")
    if os.path.exists(part):
        return part, "cached"
    t0 = time.time(); s = requests.Session()
    pf, f = open_parquet(name, session=s)
    heaps = {h: [] for h in range(HMIN, HMAX + 1)}; counts = {h: 0 for h in range(HMIN, HMAX + 1)}
    for batch in pf.iter_batches(batch_size=50_000, row_groups=list(range(a, b)), columns=["h12", "vertices"]):
        h12 = batch.column(0).to_numpy()
        idx = np.nonzero((h12 >= HMIN) & (h12 <= HMAX))[0]
        if not idx.size:
            continue
        verts = batch.column(1).take(idx).to_pylist()
        for h, v in zip(h12[idx].tolist(), verts):
            sv = vc(v); key = (hashlib.md5(sv.encode()).hexdigest(), sv); counts[h] += 1
            hp = heaps[h]
            if len(hp) < K:
                heapq.heappush(hp, (Neg(key), key))
            elif key < hp[0][1]:
                heapq.heapreplace(hp, (Neg(key), key))
        del verts
    res = {"file": name, "row_groups": [a, b], "seconds": time.time() - t0, "bytes_fetched": f.bytes_fetched, "counts": counts,
           "best": {h: sorted(k for _, k in hp) for h, hp in heaps.items()}}
    json.dump(res, open(part + ".tmp", "w")); os.replace(part + ".tmp", part)
    return part, f"{res['seconds']:.0f}s {f.bytes_fetched / 1e9:.2f}GB"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    s = requests.Session(); rev = s.get(REVISION_API, timeout=60).json().get("sha")
    units, done_files = [], []
    for name in FILES:
        if os.path.exists(os.path.join(OUT, f"partial_{name}.json")):
            done_files.append(name); continue            # completed by v1 (whole file)
        n = open_parquet(name, session=s)[0].metadata.num_row_groups
        units += [(name, a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]
    units.sort(key=lambda u: -(u[2] - u[1]))
    json.dump({"revision": rev, "K": K, "h_range": [HMIN, HMAX], "files": FILES, "v1_complete_files": done_files, "units": len(units),
               "ordering": "md5(DuckDB VARCHAR of vertices), then VARCHAR"}, open(os.path.join(OUT, "scan_meta_v2.json"), "w"), indent=1)
    print(f"{len(done_files)} files from v1 checkpoints; {len(units)} row-group units to scan", flush=True)
    with Pool(WORKERS, maxtasksperchild=4) as pool:
        for i, (part, msg) in enumerate(pool.imap_unordered(unit, units), 1):
            print(f"[{i}/{len(units)}] {os.path.basename(part)} {msg}", flush=True)
    merged = {h: [] for h in range(HMIN, HMAX + 1)}; counts = {h: 0 for h in range(HMIN, HMAX + 1)}
    parts = [os.path.join(OUT, f"partial_{n}.json") for n in done_files] + \
            [os.path.join(OUT, f"partial_{n}__rg{a:04d}-{b:04d}.json") for n, a, b in units]
    for p in parts:
        r = json.load(open(p))
        for h in merged:
            merged[h].extend(tuple(k) for k in r["best"][str(h)][:K]); counts[h] += r["counts"][str(h)]
        for h in merged:
            merged[h] = sorted(merged[h])[:K]
    json.dump({"revision": rev, "K": K, "total_rows_per_h11": counts, "sources": [os.path.basename(p) for p in parts],
               "ordered_candidates": {h: [k[1] for k in v] for h, v in merged.items()}},
              open(os.path.join(OUT, "ordered_candidates.json"), "w"))
    print("done; rows at h11 41/45/50:", counts.get(41), counts.get(45), counts.get(50), flush=True)
