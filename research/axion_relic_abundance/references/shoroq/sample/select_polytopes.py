"""Apply the legacy selection rule to the ordered KS candidates and verify it reproduces tranche 1 exactly.

Legacy rule (01_fetch_and_triangulate.jl): walk candidates in (md5, VARCHAR) order; skip if h11(N) != H, if not
N-favorable, or if the native normal form (affine_transform=False) was already seen; keep the first TARGET.
Output: selection.json with, per h11, the ordered list of the first TARGET distinct favorable polytopes
(position, vertices, normal-form sha256, md5 key, tranche-1 membership) plus skip statistics, and a verification
that positions 1..n1 (n1 = tranche-1 count at that h11) equal the tranche-1 manifest polytopes in the same order.
Usage: python select_polytopes.py ORDERED_CANDIDATES_JSON TRANCHE1_MANIFEST OUT_JSON [TARGET] [H_MIN] [H_MAX]
"""
import hashlib, json, sys
import numpy as np
from cytools import Polytope

cand_path, t1_path, out = sys.argv[1:4]
TARGET = int(sys.argv[4]) if len(sys.argv) > 4 else 100
HMIN = int(sys.argv[5]) if len(sys.argv) > 5 else 4; HMAX = int(sys.argv[6]) if len(sys.argv) > 6 else 50
cand = json.load(open(cand_path)); t1 = json.load(open(t1_path))["polytopes"]


def nf_key(verts):
    p = Polytope(np.asarray(verts, int))
    nf = np.asarray(p.normal_form(affine_transform=False, backend="native"), int)
    return p, nf, hashlib.sha256(",".join(str(x) for x in nf.flatten(order="F")).encode()).hexdigest()


t1_by_h = {}
for e in t1:
    t1_by_h.setdefault(e["h11"], []).append(e)
res = {"target_per_h11": TARGET, "revision": cand["revision"], "rule": __doc__.split("\n\n")[1].strip(), "by_h11": {}}
for h in range(HMIN, HMAX + 1):
    seen, sel = set(), []
    skipped = {"wrong_h11": 0, "not_favorable": 0, "duplicate_normal_form": 0, "error": 0}
    lst = cand["ordered_candidates"][str(h)]
    examined = 0
    for rank, sv in enumerate(lst):
        examined += 1
        verts = json.loads(sv)
        try:
            p, nf, key = nf_key(verts)
            if int(p.h11(lattice="N")) != h:
                skipped["wrong_h11"] += 1; continue
            if not bool(p.is_favorable(lattice="N")):
                skipped["not_favorable"] += 1; continue
            if key in seen:
                skipped["duplicate_normal_form"] += 1; continue
        except Exception:
            skipped["error"] += 1; continue
        seen.add(key)
        sel.append({"position": len(sel) + 1, "candidate_rank": rank, "md5": hashlib.md5(sv.encode()).hexdigest(),
                    "vertices": np.asarray(p.vertices(), int).tolist(), "normal_form_sha256": key})
        if len(sel) == TARGET:
            break
    # verify tranche 1 = first n1 selected, in order (compare native normal forms)
    t1h = t1_by_h.get(h, [])
    t1_keys = [nf_key(e["vertices"])[2] for e in t1h]
    n1 = len(t1_keys)
    prefix_ok = [s["normal_form_sha256"] for s in sel[:n1]] == t1_keys
    for s in sel:
        s["tranche1"] = s["position"] <= n1 and prefix_ok
    res["by_h11"][h] = {"total_rows_at_h11": cand["total_rows_per_h11"][str(h)], "candidates_examined": examined,
                         "candidates_buffered": len(lst), "selected": len(sel), "enough_for_target": len(sel) >= TARGET,
                         "skipped": skipped, "tranche1_count": n1, "tranche1_reproduced_in_order": prefix_ok,
                         "selection": sel}
    print(h, "selected", len(sel), "of", examined, "examined; tranche1", n1, "reproduced:", prefix_ok, skipped, flush=True)
json.dump(res, open(out, "w"))
