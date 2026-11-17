"""Geometry selection of the producers (stage 3): which of the G2-accepted geometries enter the vacuum scan and the relic input.

G2 can accept more than one FRST (triangulation) per polytope: `--n` larger than the number of polytopes of an h11 block asks for
extra triangulations. The Shoroq reference asks for exactly one geometry per polytope (cy_0000001). The policy is explicit:
  all         every G2-accepted geometry (default)
  first_frst  only the first accepted FRST of each polytope (cy_0000001); the other accepted geometries are listed as not selected
Nothing is dropped silently: the selection report (vacuum_repair/05_selection_report.json) lists accepted, selected, not selected
and failed items with reasons. geometry_id = h11_XXX/np_XXXXXXX/cy_XXXXXXX is unique per geometry.

Usage: python selection.py list WORK SELECTION OUT_TSV    (OUT_TSV: columns h11, position[, frst]; frst only if some selected FRST != 1)
"""
import csv, os, re, sys

MODES = ("all", "first_frst")


def accepted_geometries(work):
    """[(h11, position, frst, output_path)] of the G2-accepted FRST proposals (validation/status/frst_proposal_status.csv)."""
    out = []
    seen = set()
    for r in csv.DictReader(open(f"{work}/validation/status/frst_proposal_status.csv")):
        if r["status"] != "accepted":
            continue
        m = re.search(r"/cy_(\d+)/", r["output_path"])
        if not m:
            raise SystemExit(f"ERROR: accepted proposal without a cy_XXXXXXX output path: {r}")
        key = (int(r["h11"]), int(r["polytope_index"]), int(m.group(1)))
        if key in seen:
            raise SystemExit(f"ERROR: geometry {key} accepted twice in the FRST ledger")
        seen.add(key); out.append(key + (r["output_path"],))
    return sorted(out)


def select(accepted, mode):
    """(selected, not_selected) as lists of (h11, position, frst, path)."""
    if mode not in MODES:
        raise SystemExit(f"ERROR: GEOMETRY_SELECTION must be one of {MODES}, got {mode!r}")
    if mode == "all":
        return list(accepted), []
    first = {}
    for g in accepted:
        first.setdefault((g[0], g[1]), g[2])          # accepted is sorted: the lowest FRST index of each polytope
    sel = [g for g in accepted if g[2] == first[(g[0], g[1])]]
    return sel, [g for g in accepted if g not in sel]


def gid(h, p, f):
    return f"h11_{h:03d}/np_{p:07d}/cy_{f:07d}"


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "list":
        raise SystemExit(__doc__)
    work, mode, out = sys.argv[2:]
    sel, rest = select(accepted_geometries(work), mode)
    three = any(g[2] != 1 for g in sel)
    open(out, "w").write("h11\tposition" + ("\tfrst" if three else "") + "\n" +
                         "".join(f"{g[0]}\t{g[1]}" + (f"\t{g[2]}" if three else "") + "\n" for g in sel))
    print(f"{len(sel)} geometries selected ({mode}); {len(rest)} accepted but not selected")
