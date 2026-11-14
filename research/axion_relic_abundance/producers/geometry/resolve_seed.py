"""Resolve and verify the G2 seed of a producer run (stage 2).

The upstream generator (scripts/generate_geometric_data_multitriangulation.py) takes ONE integer --seed and gives the polytope at
1-based position p within its h11 block the seed (seed + p); a proposal q of that polytope uses seed + p + q - 1. The polytope
manifest records the same rule (`base_seed`, `seed_rule`, per-entry `seed`). This module makes the choice of --seed explicit.

Precedence of the effective base seed (highest first):
  1. cli       an explicit seed: the producer environment variable G2_SEED (or --cli-seed)
  2. manifest  the manifest's `base_seed`; if absent, the common value (entry seed - position) of the per-entry `seed` fields
  3. default   DEFAULT_SEED = 20260930, the seed of the Shoroq reference recipe (v1)
The Shoroq manifest has base_seed 20260930, so the reference run is unchanged.

Consistency rules (a violation is an error, never silently ignored):
  * a seed must be an integer (not boolean, not a float or string), 0 <= seed < 2**31;
  * within every h11 block the polytopes must be listed with unified_position = 1, 2, 3, ... (G2 seeds by list position);
  * a per-entry `seed` must equal manifest base_seed + position (or the same common base when base_seed is absent).
An explicit seed that differs from the manifest overrides it; then the per-entry seed fields are informational only and the
override is recorded.

Usage:
  python resolve_seed.py resolve MANIFEST OUT_JSON [--cli-seed N]     prints the effective seed, writes the resolution record
  python resolve_seed.py verify  OUT_JSON FRST_PROPOSAL_STATUS_CSV    checks that the seeds G2 used are those resolved
  python resolve_seed.py defaults MANIFEST                            prints "H11_MIN H11_MAX N_PER_H11" of the manifest
"""
import argparse, csv, json, sys
from collections import defaultdict

DEFAULT_SEED = 20260930
MAX_SEED = 2 ** 31 - 1


def check_seed(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"ERROR: {what} must be an integer, got {value!r}")
    if not 0 <= value <= MAX_SEED:
        raise SystemExit(f"ERROR: {what} must satisfy 0 <= seed <= {MAX_SEED}, got {value}")
    return value


def parse_cli(text, what):
    try:
        return check_seed(int(str(text).strip(), 10), what)
    except ValueError:
        raise SystemExit(f"ERROR: {what} must be an integer, got {text!r}")


def resolve(manifest_path, cli_seed=None):
    try:
        M = json.load(open(manifest_path))
    except (OSError, ValueError) as e:
        raise SystemExit(f"ERROR: cannot read polytope manifest {manifest_path}: {e}")
    if not isinstance(M, dict) or not isinstance(M.get("polytopes"), list) or not M["polytopes"]:
        raise SystemExit("ERROR: polytope manifest must be an object with a non-empty 'polytopes' list")
    base = M.get("base_seed")
    if base is not None:
        base = check_seed(base, "manifest base_seed")
    pos_in_block = defaultdict(int)
    implied = set(); n_entry_seeds = 0
    for i, p in enumerate(M["polytopes"], start=1):
        h = p.get("h11")
        if isinstance(h, bool) or not isinstance(h, int):
            raise SystemExit(f"ERROR: manifest entry {i}: h11 must be an integer")
        pos_in_block[h] += 1
        pos = pos_in_block[h]
        up = p.get("unified_position", pos)
        if isinstance(up, bool) or not isinstance(up, int) or up != pos:
            raise SystemExit(f"ERROR: manifest entry {i} (h11 {h}): unified_position {up!r} is not its 1-based list position {pos} within the "
                             "h11 block; G2 seeds polytopes by list position, so the manifest must list each h11 block as 1, 2, 3, ...")
        if "seed" in p:
            n_entry_seeds += 1
            s = check_seed(p["seed"], f"manifest entry {i} (h11 {h}, position {pos}) seed")
            implied.add(s - pos)
            if base is not None and s != base + pos:
                raise SystemExit(f"ERROR: manifest entry {i} (h11 {h}, position {pos}): seed {s} != base_seed + position = {base + pos}")
    if base is None and implied:
        if len(implied) != 1:
            raise SystemExit("ERROR: manifest per-entry seeds do not share one base (seed - position differs between entries) and no base_seed is given")
        base = check_seed(next(iter(implied)), "base seed implied by the manifest entry seeds")
        manifest_source = "manifest.entry_seeds"
    else:
        manifest_source = "manifest.base_seed"
    if cli_seed is not None:
        eff, source = check_seed(cli_seed, "explicit seed"), "cli"
    elif base is not None:
        eff, source = base, manifest_source
    else:
        eff, source = DEFAULT_SEED, "default"
    return {"effective_seed": eff, "source": source, "explicit_seed": cli_seed, "manifest_base_seed": base,
            "default_seed": DEFAULT_SEED, "manifest_entries_with_seed": n_entry_seeds, "manifest_entries": len(M["polytopes"]),
            "override_of_manifest": cli_seed is not None and base is not None and cli_seed != base,
            "rule": "polytope at 1-based position p within its h11 block is seeded with effective_seed + p (G2); proposal q uses effective_seed + p + q - 1"}


def verify(resolution_path, status_csv):
    R = json.load(open(resolution_path)); eff = R["effective_seed"]
    rows = list(csv.DictReader(open(status_csv)))
    bad = []
    for r in rows:
        if r.get("proposal_seed") in (None, "", "None"):
            continue
        want = eff + int(r["polytope_index"]) + int(r["proposal_index"]) - 1
        if int(r["proposal_seed"]) != want:
            bad.append((r["h11"], r["polytope_index"], r["proposal_index"], r["proposal_seed"], want))
    R["verified_against_g2_ledger"] = {"proposals_checked": len(rows), "mismatches": len(bad)}
    json.dump(R, open(resolution_path, "w"), indent=1)
    if bad:
        raise SystemExit(f"ERROR: {len(bad)} G2 proposal seeds differ from effective_seed + position + proposal - 1; first: {bad[0]}")
    print(f"seed propagation verified: {len(rows)} proposals use effective seed {eff}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("resolve"); a.add_argument("manifest"); a.add_argument("out"); a.add_argument("--cli-seed", default=None)
    c = sub.add_parser("defaults"); c.add_argument("manifest")
    b = sub.add_parser("verify"); b.add_argument("resolution"); b.add_argument("status_csv")
    args = ap.parse_args()
    if args.cmd == "defaults":
        M = json.load(open(args.manifest)); cnt = defaultdict(int)
        for p in M["polytopes"]:
            cnt[p["h11"]] += 1
        print(min(cnt), max(cnt), max(cnt.values()))
    elif args.cmd == "resolve":
        cli = parse_cli(args.cli_seed, "G2_SEED") if args.cli_seed not in (None, "") else None
        R = resolve(args.manifest, cli)
        json.dump(R, open(args.out, "w"), indent=1)
        print(R["effective_seed"])
        print(f"seed {R['effective_seed']} (source: {R['source']}" + ("; overrides the manifest base_seed %s" % R["manifest_base_seed"] if R["override_of_manifest"] else "") + ")", file=sys.stderr)
    else:
        verify(args.resolution, args.status_csv)
