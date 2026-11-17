"""Seed resolution and geometry selection of the producers (external audit I-02, I-03).

Part 1 (no toolchain, seconds): unit tests of producers/geometry/resolve_seed.py and producers/vacuum/selection.py.
Part 2 (needs the producer toolchain: CYAX_REPO or the enclosing checkout, JULIA_VMM; SKIPPED otherwise, about 8 min):
  stage 2 on a two-polytope h11 = 5 manifest with
    manifest seed        base_seed 777000 -> G2 proposal seeds 777000 + position (+ proposal - 1)
    explicit G2_SEED     overrides the manifest, recorded as such
    default seed         manifest without seeds -> 20260930
    determinism          a rerun reproduces the triangulation hashes
  and stage 3 on a run with more than one accepted FRST of a polytope:
    all                  every accepted geometry reaches the relic input, with unique identifiers
    first_frst           only cy_0000001 of each polytope; the others are reported as not selected
Usage: SHOROQ_PATHS=paths.env python tests/test_producer_selection.py WORK
"""
import csv, json, os, subprocess, sys, tempfile

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, f"{REPO}/producers/geometry"); sys.path.insert(0, f"{REPO}/producers/vacuum")
import resolve_seed as RS   # noqa: E402
import selection as SEL     # noqa: E402

MAN = f"{REPO}/tests/producer_selection/manifest_h11_005_two_polytopes.json"
results = []


def record(name, ok, detail=""):
    results.append(ok); print(f"{'PASS' if ok else 'FAIL':7s} {name} {detail}", flush=True)


def raises(fn, *a):
    try:
        fn(*a)
    except SystemExit as e:
        return str(e)
    return None


def manifest(tmp, **kw):
    M = json.load(open(MAN))
    for k, v in kw.items():
        if v is None:
            M.pop(k, None)
        else:
            M[k] = v
    p = tempfile.mkstemp(suffix=".json", dir=tmp)[1]; json.dump(M, open(p, "w")); return p, M


with tempfile.TemporaryDirectory() as tmp:
    p, M = manifest(tmp)
    r = RS.resolve(p); record("seed: manifest base_seed used", r["effective_seed"] == 777000 and r["source"] == "manifest.base_seed")
    r = RS.resolve(p, 1234); record("seed: explicit seed overrides manifest and is recorded", r["effective_seed"] == 1234 and r["source"] == "cli" and r["override_of_manifest"])
    p2, M2 = manifest(tmp, base_seed=None); r = RS.resolve(p2)
    record("seed: base implied by the per-entry seeds", r["effective_seed"] == 777000 and r["source"] == "manifest.entry_seeds")
    p3, M3 = manifest(tmp, base_seed=None)
    for e in M3["polytopes"]:
        e.pop("seed")
    json.dump(M3, open(p3, "w")); r = RS.resolve(p3)
    record("seed: default 20260930 without any manifest seed", r["effective_seed"] == 20260930 and r["source"] == "default")
    shoroq = f"{REPO}/references/shoroq/data/polytope_manifest_h11_004_050.json"
    record("seed: Shoroq manifest resolves to 20260930", RS.resolve(shoroq)["effective_seed"] == 20260930)
    bad = {"bool": True, "float": 1.5, "string": "7", "negative": -1, "too large": 2 ** 31}
    for k, v in bad.items():
        pb, _ = manifest(tmp, base_seed=v); record(f"seed: invalid manifest base_seed rejected ({k})", raises(RS.resolve, pb) is not None)
        record(f"seed: invalid explicit seed rejected ({k})", raises(RS.check_seed, v, "explicit seed") is not None)
    record("seed: non-numeric G2_SEED text rejected", raises(RS.parse_cli, "abc", "G2_SEED") is not None)
    pe, ME = manifest(tmp); ME["polytopes"][0]["seed"] = 5; json.dump(ME, open(pe, "w"))
    record("seed: entry seed inconsistent with base_seed + position rejected", raises(RS.resolve, pe) is not None)
    pu, MU = manifest(tmp); MU["polytopes"][1]["unified_position"] = 7; json.dump(MU, open(pu, "w"))
    record("seed: unified_position that is not the list position rejected", raises(RS.resolve, pu) is not None)
    # verify against a ledger
    led = f"{tmp}/led.csv"; res = f"{tmp}/res.json"; json.dump(RS.resolve(p), open(res, "w"))
    rows = [{"h11": 5, "polytope_index": 1, "proposal_index": 1, "proposal_seed": 777001}, {"h11": 5, "polytope_index": 2, "proposal_index": 2, "proposal_seed": 777003}]
    for good in (True, False):
        rr = [dict(r) for r in rows]; rr[1]["proposal_seed"] = 777003 if good else 20260932
        with open(led, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
        e = raises(RS.verify, res, led); record(f"seed: ledger {'consistent accepted' if good else 'mismatch rejected'}", (e is None) == good)

acc = [(5, 1, 1, "a"), (5, 2, 1, "b"), (5, 2, 2, "c"), (6, 1, 1, "d")]
s_all, n_all = SEL.select(acc, "all"); s_f, n_f = SEL.select(acc, "first_frst")
record("selection: all keeps every accepted geometry", s_all == acc and n_all == [])
record("selection: first_frst keeps cy_0000001 and reports the rest", s_f == [acc[0], acc[1], acc[3]] and n_f == [acc[2]])
record("selection: unknown policy rejected", raises(SEL.select, acc, "second") is not None)
record("selection: geometry ids are unique and stable", SEL.gid(5, 2, 2) == "h11_005/np_0000002/cy_0000002")

# ---- Part 2: producer toolchain ----
ENVF = os.environ.get("SHOROQ_PATHS", f"{REPO}/configs/paths.env"); P = {}
for line in open(ENVF):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); P[k] = v.strip().strip('"')
cyax = P.get("CYAX_REPO") or os.path.normpath(f"{REPO}/../..")
if len(sys.argv) < 2 or not (os.path.exists(f"{cyax}/scripts/batch_physical_spectrum.jl") and P.get("JULIA_VMM")):
    print("SKIPPED producer end-to-end part: producer toolchain or WORK argument missing")
else:
    T = os.path.abspath(sys.argv[1]); os.makedirs(T)

    def run(script, W, extra=None, manifest_path=MAN):
        e = dict(os.environ, SHOROQ_PATHS=ENVF, WORK=W, POLYTOPE_MANIFEST=manifest_path); e.setdefault("ALLOW_OTHER_CYAX_COMMIT", "1"); e.update(extra or {})
        os.makedirs(W, exist_ok=True)
        r = subprocess.run(["bash", f"{REPO}/producers/{script}"], capture_output=True, text=True, env=e)
        open(f"{W}.{script}.log", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr); return r.returncode

    def ledger(W):
        return list(csv.DictReader(open(f"{W}/validation/status/frst_proposal_status.csv")))

    def seeds_ok(W, base):
        L = ledger(W); return bool(L) and all(int(r["proposal_seed"]) == base + int(r["polytope_index"]) + int(r["proposal_index"]) - 1 for r in L)

    W1 = f"{T}/manifest_seed"
    rc = run("run_geometry.sh", W1, {"N_PER_H11": "4"})
    R = json.load(open(f"{W1}/provenance/seed_resolution.json"))
    record("seed e2e: manifest base_seed reaches G2 and the provenance", rc == 0 and seeds_ok(W1, 777000) and R["source"] == "manifest.base_seed"
           and R["verified_against_g2_ledger"]["mismatches"] == 0, f"seeds {sorted({r['proposal_seed'] for r in ledger(W1)})}")
    W2 = f"{T}/explicit_seed"; rc = run("run_geometry.sh", W2, {"G2_SEED": "5000"})
    record("seed e2e: explicit G2_SEED overrides the manifest", rc == 0 and seeds_ok(W2, 5000) and json.load(open(f"{W2}/provenance/seed_resolution.json"))["override_of_manifest"])
    pd = f"{T}/manifest_noseed.json"; M = json.load(open(MAN)); M.pop("base_seed")
    for e in M["polytopes"]:
        e.pop("seed")
    json.dump(M, open(pd, "w")); W3 = f"{T}/default_seed"; rc = run("run_geometry.sh", W3, manifest_path=pd)
    record("seed e2e: default seed 20260930 without manifest seeds", rc == 0 and seeds_ok(W3, 20260930))
    W4 = f"{T}/manifest_seed_rerun"; rc = run("run_geometry.sh", W4, {"N_PER_H11": "4"})
    record("seed e2e: deterministic rerun (same triangulations)", rc == 0 and sorted(r["triangulation_hash"] for r in ledger(W1)) == sorted(r["triangulation_hash"] for r in ledger(W4)))
    # selection on the multi-FRST run
    n_acc = len(SEL.accepted_geometries(W1))
    multi = n_acc > len({(g[0], g[1]) for g in SEL.accepted_geometries(W1)})
    record("selection e2e: run has a polytope with two accepted FRSTs", multi, f"{n_acc} accepted geometries")
    import shutil
    Wf = f"{T}/first"; shutil.copytree(W1, Wf, symlinks=True)
    rc_all = run("run_vacuum_repair.sh", W1, {"GEOMETRY_SELECTION": "all"}); rc_first = run("run_vacuum_repair.sh", Wf, {"GEOMETRY_SELECTION": "first_frst"})
    ids = lambda W: [g["geometry_id"] for g in json.load(open(f"{W}/relic_input/relic_abundance_input_manifest.json"))["geometries"]]
    rep = lambda W: json.load(open(f"{W}/vacuum_repair/05_selection_report.json"))
    ia, i1 = ids(W1), ids(Wf)
    record("selection e2e: all -> every accepted geometry reaches the relic input, unique ids", rc_all == 0 and len(ia) == n_acc and len(set(ia)) == n_acc and rep(W1)["geometries_not_selected"] == 0)
    record("selection e2e: first_frst -> only cy_0000001, the rest reported", rc_first == 0 and all(i.endswith("cy_0000001") for i in i1)
           and rep(Wf)["geometries_not_selected"] == n_acc - len(i1) > 0 and set(i1) <= set(ia))
    # the same geometry has the same spectrum in both modes (no mixing of spectra between geometries)
    shared = set(i1)
    sha = lambda W: {g["geometry_id"]: g["spectrum_source_sha256"] for g in json.load(open(f"{W}/relic_input/relic_abundance_input_manifest.json"))["geometries"]}
    record("selection e2e: a geometry's spectrum is identical in both modes", all(sha(W1)[g] == sha(Wf)[g] for g in shared))
    # the relic engine accepts the multi-FRST input
    e = dict(os.environ, SHOROQ_PATHS=ENVF, WORK=W1, RELIC_RUN=f"{REPO}/examples/minimal/run.json")
    r1 = subprocess.run(["bash", f"{REPO}/engine/run_relic.sh"], capture_output=True, text=True, env=e)
    n_g = len(list(csv.DictReader(open(f"{W1}/relic/tables/geometry_abundance_validity_5MeV.csv")))) if r1.returncode == 0 else -1
    record("selection e2e: relic stage runs on the all-FRST input with every geometry", r1.returncode == 0 and n_g == n_acc, f"{n_g} geometries")

print(f"\nPRODUCER SELECTION/SEED TESTS: {sum(results)} passed, {len(results) - sum(results)} failed")
sys.exit(0 if all(results) else 1)
