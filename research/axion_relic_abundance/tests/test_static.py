"""Fast static checks of the repository (no pipeline run, no network). Run: python tests/test_static.py

  1. sample manifest: 4700 polytopes, 100 per h11 in 4..50, seed = 20260930 + position, distinct normal forms
  2. g* table checksum equals the approved value in the relic configuration
  3. relic script: everything from `using` onward is byte-identical to the frozen 2026-10-01 production script, and
     the physics block has the audited checksum
  4. theta script and interval machinery are the frozen ones
  5. approved methodology switches in both configurations
  6. derived cosmology recomputed by the 50-digit reference implementation B equals the configuration
  7. no machine-specific absolute paths in tracked files
"""
import hashlib, json, os, re, subprocess, sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "engine", "relic"))
FAILED = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


sha = lambda b: hashlib.sha256(b).hexdigest()
rd = lambda p: open(os.path.join(REPO, p), "rb").read()

# 1. sample
M = json.loads(rd("references/shoroq/data/polytope_manifest_h11_004_050.json"))
P = M["polytopes"]
per = {h: sorted(p["unified_position"] for p in P if p["h11"] == h) for h in range(4, 51)}
check("sample: 4700 polytopes, positions 1..100 for every h11 in 4..50", len(P) == 4700 and all(v == list(range(1, 101)) for v in per.values()))
check("sample: seed = 20260930 + unified position", all(p["seed"] == 20260930 + p["unified_position"] for p in P))
check("sample: distinct normal forms", len({p["normal_form_sha256"] for p in P}) == 4700)
check("sample: historical tranches 1600 / 3100", sum(p["historical_tranche"] == 1 for p in P) == 1600 and sum(p["historical_tranche"] == 2 for p in P) == 3100)

# 2. g* table
rc = json.loads(rd("references/shoroq/configs/relic_production_config.frozen.json"))
check("g* table sha256 = approved", sha(rd("data/standardmodel2018.dat")) == rc["cosmology"]["gstar_table_sha256"])

# 3. relic script
src = rd("engine/relic/compute_relic_abundance.jl").decode()
body = src[src.index("\nusing DelimitedFiles"):]
# documented driver changes (tests/driver_changes.json, tags [G1]..[G6]): reverting exactly these restores the frozen body
DC = json.loads(rd("tests/driver_changes.json")); rev = body; ok = True
for blk in DC["blocks"]:
    ok &= rev.count(blk["new"]) == 1; rev = rev.replace(blk["new"], blk["old"])
ok &= rev.count(DC["G1"]["new"]) == 1; rev = rev.replace(DC["G1"]["new"], DC["G1"]["old"])
check(f"relic script body = frozen production script except the documented driver changes G1-G7 ({len(DC['blocks']) + 1} blocks)",
      ok and sha(rev.encode()) == "9f73b9c63fd159a252a4cbd8582906c0148289123ccc272307fda7c6db877064")
a, b = src.index("# ============================== physics block"), src.index("# ============================== end of physics block")
check("relic physics block checksum", sha(src[a:b].encode()) == "a0c613f17cd8da72e93834396b0a1a0663d201454216e9427ef3895beace884a")

# 4. theta
# frozen final analysis: 3fffc5bde05cc6f3ead091ee053b292b3a5cd1700ef1c4948973a5a4df033d3d; generalised 2026-10-29 (scenario
# labels from the run, configurable primary smoothing / robustness thresholds, small-sample guards) and 2026-10-30 (Omega = 0
# geometries, NO_MATCH) and 2026-10-31 (primary and sensitivity populations; PROVENANCE.md). The binding check is the byte identity of every Shoroq theta output in
# references/shoroq/tests/regression_relic_theta.sh.
# 88488246... was the generalised version; c86e47a4... adds only `promote=true` to the per-scenario append (external audit I-01:
# a NO_MATCH scenario has all-missing angle columns). No mathematics changed; the Shoroq regressions must stay byte-identical.
# bd568493... = c86e47a4... plus the methodology-v2 w_R columns/keys (all conditional on METHOD_VERSION >= 2: v1 theta outputs are byte-identical,
# checked in tests/test_wR_general.py and references/shoroq/tests/regression_relic_theta.sh).
check("01_theta_final.jl is the reviewed generalised version",
      sha(rd("engine/theta/01_theta_final.jl")) == "bd5684931ecf2bb4c5e2f1837744227d1b4fa9dbdc09f087c3570ce5e2d9c707")
cj = rd("engine/theta/common.jl").decode()
check("theta interval machinery present", "function exact_profile" in cj and "function maximal_set" in cj and "function smoothed" in cj)

# 5. methodology switches
V = rc["validity"]
check("relic: inflation cut per mode at 3H_I,max = 1.4e14 GeV; m_KK and m > f_pert flags only; no 100 eV window",
      V["inflation_cut"]["enabled"] and V["inflation_cut"]["three_H_I_max_GeV"] == 1.4e14 and V["inflation_cut"]["unit"].startswith("mode")
      and not V["mKK_flag"]["exclusion"] and not V["fpert_flag"]["exclusion"] and not V["light_window_100eV"]["applied"])
cs = rc["categories_in_cdm_sum"]
check("relic: POST and PRE summed, FROZEN and LATE census only",
      cs["POST_REHEATING"]["in_cdm_sum"] and cs["PRE_REHEATING"]["in_cdm_sum"] and not cs["FROZEN"]["in_cdm_sum"] and not cs["LATE"]["in_cdm_sum"])
check("relic: f = f_pert (C = 1), w_R = 0, printed Fuzzy (2.16) not used, three T_reh",
      rc["axion"]["f_relic_over_f_pert"] == 1.0 and rc["reheating"]["w_R"] == 0.0 and rc["reheating"]["eq_2_16_printed_normalization_used"] is False
      and [s["T_reh_GeV"] for s in rc["reheating"]["scenarios"]] == [0.005, 1e4, 1e10])
# methodology v2 (constant w_R per scenario): identical to v1 except the documented keys; v1 itself is frozen
m1, m2 = json.loads(rd("configs/methodology_v1.json")), json.loads(rd("configs/methodology_v2.json"))
changed = sorted(k for k in set(m1) | set(m2) if m1.get(k) != m2.get(k))
check("methodology v2 differs from v1 only in schema/version/status/created, run/code provenance text, reheating, methodology_flags and outputs",
      changed == sorted(["schema", "methodology_version", "created", "status", "run", "code", "reheating", "methodology_flags", "outputs"]), ",".join(changed))
check("methodology v1 states w_R = 0 and has no methodology_version (frozen)", m1["reheating"]["w_R"] == 0.0 and "methodology_version" in m2 and "methodology_version" not in m1)
s1, s2 = json.loads(rd("configs/relic_output_schema.json")), json.loads(rd("configs/relic_output_schema_v2.json"))
check("output schema v2 = v1 plus the w_R column in every table (and the general-w branch label)",
      all([c["name"] for c in s2["tables"][t]] == [c["name"] for c in s1["tables"][t]] + ["w_R"] for t in s1["tables"])
      and "fuzzy_wR_general_eq2.10_2.15" in s2["abundance_branches"] and "fuzzy_wR_general_eq2.10_2.15" not in s1["abundance_branches"])
tc = json.loads(rd("references/shoroq/configs/theta_analysis_config.frozen.json"))
check("theta: domain 0 < theta <= 0.5, no lower cut, TIGHT 0.9 <= R <= 1, Omega_DM h^2 = 0.12",
      tc["theta"]["theta_max"] == 0.5 and tc["theta"]["theta_min"] is None and tc["dark_matter"]["TIGHT_DM"] == [0.9, 1.0]
      and tc["dark_matter"]["Omega_DM_h2"] == 0.12)

# 6. derived cosmology (implementation B, 50 digits)
import reference_mp as B  # noqa: E402
R = B.Ref(); D = rc["cosmology"]["derived"]; tol = rc["cosmology"]["derived_tolerance_rel"]
teq = R.T_eq()
vals = {"h": R.h, "Omega_m0": R.Omega_m0, "Omega_r0": R.Omega_r0, "Omega_L0": R.Omega_L0, "T_eq_GeV": teq,
        "three_H_Teq_eV": 3 * R.H(teq) * 1e9, "H_today_GeV": R.H(B.T0)}
worst = {k: abs(float(v) / D[k] - 1) for k, v in vals.items()}
check("derived cosmology: B equals configuration", all(e <= tol.get(k, tol["default"]) if k != "three_H_Teq_eV" else e <= tol["T_eq_GeV"] for k, e in worst.items()),
      json.dumps({k: f"{e:.1e}" for k, e in worst.items()}))

# 7. no machine-specific paths. Source tree only: tracked files when the package is in a git checkout (git ls-files), otherwise
# (an exported tree) every file outside disposable local directories. Virtual environments, caches and $WORK outputs are never scanned.
pat = re.compile(r"/Users/|/home/[a-z]|/private/tmp|scratchpad")
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".mypy_cache", ".pytest_cache", "work", ".ipynb_checkpoints"}


def source_files():
    try:
        r = subprocess.run(["git", "-C", REPO, "ls-files", "-z", "--", "."], capture_output=True, text=True, check=True)
        names = [n for n in r.stdout.split("\0") if n]
        if names:
            return [os.path.join(REPO, n) for n in names if os.path.exists(os.path.join(REPO, n))]
    except (OSError, subprocess.CalledProcessError):
        pass
    out = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        out += [os.path.join(root, f) for f in files]
    return out


hits = []
for p in source_files():
    if os.path.abspath(p) == os.path.abspath(__file__):
        continue
    if p.endswith((".py", ".jl", ".sh", ".json", ".toml", ".md", ".txt", ".example")) and pat.search(open(p, errors="ignore").read()):
        hits.append(os.path.relpath(p, REPO))
check("no machine-specific absolute paths in tracked files", not hits, ",".join(hits))
ds = [os.path.relpath(p, REPO) for p in source_files() if os.path.basename(p) == ".DS_Store"]
check("no .DS_Store files in the package", not ds, ",".join(ds))

print("\nALL STATIC CHECKS PASS" if not FAILED else f"\nFAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
