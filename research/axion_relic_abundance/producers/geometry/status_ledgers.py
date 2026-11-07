"""Per-polytope and per-FRST-proposal status ledgers from G2's candidate_terminal_statuses.jsonl (read-only).

Mapping of G2 terminal statuses to the dataset's status vocabulary:
  accepted_geometry                              -> accepted
  kaehler_point_shortfall, volume_filter_rejection, prefactor/QCD/normalization failures -> rejected_moduli
  invalid_frst, duplicate_full_triangulation, duplicate_ntfe_identity                  -> rejected_triangulation
  numerical_geometry_failure, io/other exceptions, worker errors                        -> generation_error
A polytope whose proposals were all rejected ends with G2's geometry_target_shortfall record (budget exhausted); its
final status is the category of its rejections. Usage: python status_ledgers.py OUT_DIR RUN_ROOT [RUN_ROOT ...]
"""
import collections, csv, json, os, sys

out, roots = sys.argv[1], sys.argv[2:]
MODULI = {"kaehler_point_shortfall", "volume_filter_rejection", "prefactor_criterion_not_met", "qcd_normalization_failure",
          "kaehler_tip_failure", "kaehler_point_failure"}
TRI = {"invalid_frst", "duplicate_full_triangulation", "duplicate_ntfe_identity"}
cat = lambda s: "accepted" if s == "accepted_geometry" else "rejected_moduli" if s in MODULI else "rejected_triangulation" if s in TRI else "generation_error"
props, polys = [], {}
for root in roots:
    for line in open(os.path.join(root, "candidate_terminal_statuses.jsonl")):
        d = json.loads(line); key = (int(d["h11"]), int(d["polytope_index"]))
        polys.setdefault(key, {"h11": key[0], "polytope_index": key[1], "run_root": os.path.basename(root), "proposals": [], "terminal": None})
        if "proposal_index" in d and d.get("candidate_index") is not None:
            rec = {"h11": key[0], "polytope_index": key[1], "proposal_index": d["proposal_index"], "proposal_seed": d.get("proposal_seed"),
                   "g2_status": d["terminal_status"], "status": cat(d["terminal_status"]), "reason": str(d.get("terminal_reason", ""))[:160],
                   "triangulation_hash": d.get("full_triangulation_hash"), "output_path": d.get("output_path", "")}
            props.append(rec); polys[key]["proposals"].append(rec)
        else:
            polys[key]["terminal"] = {"g2_status": d["terminal_status"], "reason": d.get("terminal_reason")}
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "frst_proposal_status.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(props[0])); w.writeheader(); w.writerows(props)
rows = []
for key in sorted(polys):
    p = polys[key]; acc = [r for r in p["proposals"] if r["status"] == "accepted"]
    if acc:
        st, reason = "accepted", f"accepted on FRST proposal {acc[0]['proposal_index']}"
    else:
        cats = collections.Counter(r["status"] for r in p["proposals"])
        st = cats.most_common(1)[0][0] if cats else "generation_error"
        reason = f"{dict(collections.Counter(r['g2_status'] for r in p['proposals']))}; terminal={p['terminal']}"
    rows.append({"h11": key[0], "polytope_index": key[1], "run_root": p["run_root"], "status": st, "n_proposals": len(p["proposals"]),
                 "reason": reason[:300], "output_path": acc[0]["output_path"] if acc else ""})
with open(os.path.join(out, "polytope_status.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
S = {"polytopes_attempted": len(rows), "polytope_status": dict(collections.Counter(r["status"] for r in rows)),
     "frst_proposals": len(props), "frst_proposal_status": dict(collections.Counter(r["status"] for r in props)),
     "frst_proposal_g2_status": dict(collections.Counter(r["g2_status"] for r in props)),
     "polytope_terminal_g2": dict(collections.Counter((polys[k]["terminal"] or {}).get("g2_status") for k in polys if polys[k]["terminal"])),
     "accepted_by_h11": dict(sorted(collections.Counter(r["h11"] for r in rows if r["status"] == "accepted").items()))}
json.dump(S, open(os.path.join(out, "status_summary.json"), "w"), indent=1)
print(json.dumps(S, indent=1))
