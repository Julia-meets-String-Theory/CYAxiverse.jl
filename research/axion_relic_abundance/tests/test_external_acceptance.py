"""External-researcher acceptance test: self-contained, no Shoroq data (fixtures built from tests/external_acceptance/data).

Each case reproduces a defect found by the 2026-10-30 acceptance tests or a documented contract, and states the required
behaviour. Engine stages run through the public drivers (engine/run_*.sh) exactly as a user runs them.
  genuine          two genuine non-Shoroq geometries, DOCUMENTED_EXTERNAL; relic + theta + first-principles recomputation
  genuine_verify   same with VERIFY (theta = 0 inertia scan; needs the producer toolchain, else SKIPPED); vacuum diagnostics
  zero_mixed       F1: a validated spectrum with 0 modes above H0 stays in every geometry table with Omega = 0
  zero_only        F1 + F6: only such geometries; theta reports NO_MATCH
  integer          F2: every log10 value integral
  nomatch          F6: no geometry can reach the TIGHT band; NO_MATCH, no representative angle, no division by zero
  noeligible       all-LATE and zero-mode geometries: EVALUATED_ZERO (Omega = 0) with distinct reasons, kept in theta
  populations      all four validity classes together: EVALUATED_POSITIVE / EVALUATED_ZERO in the primary theta population,
                   PARTIAL_VALIDITY only in the labelled sensitivity population, INFLATION_INVALID_ALL (Omega NA) in neither
  all_invalid      only an all-inflation-excluded geometry: relic stage completes, theta has an empty primary population
  boundary         F5: masses on the class thresholds (incl. 3H(T_eq) +-1 ulp)
  canonical_tamper F4: rows deleted from the canonical mode table after A1 must stop stage 4
  composer         F7/F14: unknown keys, case-colliding labels, boolean theta_max, malformed TIGHT window refused cleanly
  engine_only      F8: DOCUMENTED-only input runs without the CYAxiverse checkout and Julia 1.12
  relocation       F10: a moved or copied WORK is detected with a clear message
  invalid_inputs   F12/F13: contract violations refused with a named reason, no traceback, no truncated JSON
  repeat           byte-identical tables on a second run
Usage: SHOROQ_PATHS=paths.env python tests/test_external_acceptance.py WORK
"""
import csv, json, math, os, shutil, subprocess, sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
X = f"{REPO}/tests/external_acceptance"
T = os.path.abspath(sys.argv[1]); os.path.exists(T) and sys.exit(f"{T} exists")
os.makedirs(T)
ENVFILE = os.environ.get("SHOROQ_PATHS", f"{REPO}/configs/paths.env")
P = {}
for line in open(ENVFILE):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); P[k] = v.strip().strip('"')
PY = P["PYTHON"]
results = []


def record(name, status, detail=""):
    results.append({"case": name, "status": status, "detail": detail}); print(f"{status:7s} {name} {detail}", flush=True)


def sh(args, env=None, log=None):
    e = dict(os.environ); e["SHOROQ_PATHS"] = ENVFILE; e.update(env or {})
    r = subprocess.run(args, capture_output=True, text=True, env=e)
    if log:
        open(log, "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    return r


def pipeline(name, lst, run=None, paths=None, theta=True):
    """prepare -> relic -> theta through the public drivers; returns (exit codes, WORK)"""
    W = f"{T}/{name}"; env = {"WORK": W, "RELIC_RUN": run or f"{FX}/run.json"}
    if paths:
        env["SHOROQ_PATHS"] = paths
    codes = []
    for step in (["engine/run_prepare_from_spectra.sh", lst], ["engine/run_relic.sh"]) + ((["engine/run_theta.sh"],) if theta else ()):
        r = sh(["bash", f"{REPO}/{step[0]}"] + step[1:], env, log=f"{T}/{name}.{os.path.basename(step[0])}.log"); codes.append(r.returncode)
        if r.returncode:
            break
    return codes, W


def table(path):
    return list(csv.DictReader(open(path)))


def strict_json(path):
    def bad(x):
        raise ValueError(f"non-finite constant {x}")
    return json.load(open(path), parse_constant=bad)


def validators_pass(W):
    v = json.load(open(f"{W}/relic/tables/output_validation.json"))["all_pass"]
    c = json.load(open(f"{W}/relic/validation/production_verification.json"))["all_pass"]
    fp = sh([PY, f"{X}/first_principles.py", f"{W}/relic"], log=f"{W}.first_principles.log").returncode == 0
    return v and c and fp, f"schema={v} C/B={c} first_principles={fp}"


def theta_ok(W):
    x = json.load(open(f"{W}/final_analysis/verification/crosscheck_python.json"))["all_checks_pass"]
    s = strict_json(f"{W}/final_analysis/theta/theta_summary_final.json")
    return x, s


def case(name, fn):
    try:
        st, detail = fn()
        record(name, st, detail)
    except Exception as e:  # a crash of the check itself is a failure of the case
        record(name, "FAIL", f"{type(e).__name__}: {e}")


FX = f"{T}/fixtures"
r = sh([PY, f"{X}/make_fixtures.py", FX]); r.returncode == 0 or sys.exit(r.stderr)
LABS = [s["label"] for s in json.load(open(f"{FX}/run.json"))["scenarios"]]


def c_genuine():
    codes, W = pipeline("genuine", f"{FX}/list_genuine.json")
    if codes != [0, 0, 0]:
        return "FAIL", f"exit codes {codes}"
    ok, d = validators_pass(W); x, s = theta_ok(W)
    elig = all(r["abundance_validity"] == "EVALUATED_POSITIVE" and r["theta_population"] == "PRIMARY" for lab in LABS
               for r in table(f"{W}/relic/tables/geometry_abundance_validity_{lab}.csv"))
    return ("PASS" if ok and x and elig else "FAIL"), f"{d} theta_crosscheck={x} eligibility={elig}"


def c_genuine_verify():
    # an empty CYAX_REPO means the enclosing CYAxiverse.jl checkout (configs/paths.env.example)
    cyax = P.get("CYAX_REPO") or os.path.normpath(f"{REPO}/../..")
    if not (os.path.exists(f"{cyax}/scripts/batch_physical_spectrum.jl") and P.get("JULIA_VMM")):
        return "SKIPPED", "producer toolchain (CYAX_REPO or enclosing checkout, JULIA_VMM) not configured"
    codes, W = pipeline("genuine_verify", f"{FX}/list_genuine_verify.json", theta=False)
    if codes != [0, 0]:
        return "FAIL", f"exit codes {codes}"
    man = json.load(open(f"{W}/relic_input/relic_abundance_input_manifest.json"))
    st = {g["geometry_id"]: g["vacuum_validation"] for g in man["geometries"]}
    ok = all(v["state"] == "VERIFIED_THETA0_INERTIA" for v in st.values())
    diag = st["h11_015/np_0000001/cy_0000001"].get("diagnostics", {})
    flag = diag.get("subH0_inertia_resolved") is False
    vrows = table(f"{W}/relic/a1/vacuum_validation.csv")
    shown = any(r.get("subH0_inertia_resolved") == "False" for r in vrows)
    return ("PASS" if ok and flag and shown else "FAIL"), f"states_verified={ok} h11_015 sub-H0 inertia flagged unresolved={flag} in A1 table={shown}"


def validity_checks(W, gid, cls, reason, omega, complete, partial, pop):
    """geometry_summary Omega (None = NA), validity table class/reason/complete/partial/theta population, every scenario"""
    out = []
    num = lambda v: None if v in ("NA", "") else float(v)
    for lab in LABS:
        g = {r["geometry_id"]: r for r in table(f"{W}/relic/tables/geometry_summary_{lab}.csv")}
        e = {r["geometry_id"]: r for r in table(f"{W}/relic/tables/geometry_abundance_validity_{lab}.csv")}
        r = g.get(gid); q = e.get(gid)
        ok = r is not None and q is not None and len(g) == len(e) and q["abundance_validity"] == cls and q["reason"] == reason and q["theta_population"] == pop
        for got, want in ((num(r["Omega_geometry_h2_at_theta_ref"]) if r else None, omega), (num(q["Omega_complete_h2_at_theta_ref"]) if q else None, complete),
                          (num(q["Omega_valid_modes_partial_h2_at_theta_ref"]) if q else None, partial)):
            ok &= (got is None) if want is None else (got is not None and (got > 0 if want == ">0" else got == want))
        out.append(ok)
    return all(out)


def zero_checks(W, gid, reason):
    return validity_checks(W, gid, "EVALUATED_ZERO", reason, 0.0, 0.0, None, "PRIMARY")


def c_zero_mixed():
    codes, W = pipeline("zero_mixed", f"{FX}/list_zero_mixed.json")
    if codes[:2] != [0, 0]:
        return "FAIL", f"exit codes {codes}"
    gid = "h11_015/np_0000501/cy_0000001"
    z = zero_checks(W, gid, "NO_MODES_ABOVE_THRESHOLD")
    ok, d = validators_pass(W)
    if codes != [0, 0, 0]:
        return "FAIL", f"relic zero-mode row kept={z} {d}; theta exit {codes}"
    x, s = theta_ok(W)
    tg = [r for r in table(f"{W}/final_analysis/theta/theta_geometry_final.csv") if r["geometry_id"] == gid]
    t_ok = len(tg) == len(LABS) and all(math.isinf(float(r["theta_90"])) and r["class_at_theta_rep"] in ("ALLOWED_UNDERABUNDANT", "") for r in tg)
    pop = all(json.load(open(f"{W}/relic/tables/global_summary.json"))[lab]["census"]["geometries"] == 2 for lab in LABS)
    return ("PASS" if z and ok and x and t_ok and pop else "FAIL"), f"zero row kept with Omega=0={z} {d} theta_crosscheck={x} theta_rows={t_ok} census_geometries=2:{pop}"


def c_zero_only():
    codes, W = pipeline("zero_only", f"{FX}/list_zero_only.json")
    if codes != [0, 0, 0]:
        return "FAIL", f"exit codes {codes}"
    x, s = theta_ok(W)
    nm = all(s[lab]["N_tight_max"] == 0 and s[lab]["upper_boundary"]["status"] == "NO_MATCH" and s[lab]["theta_rep"] is None for lab in LABS)
    return ("PASS" if x and nm else "FAIL"), f"theta_crosscheck={x} NO_MATCH={nm}"


def c_integer():
    codes, W = pipeline("integer", f"{FX}/list_integer.json")
    if codes != [0, 0, 0]:
        return "FAIL", f"exit codes {codes}"
    ok, d = validators_pass(W); x, s = theta_ok(W)
    return ("PASS" if ok and x else "FAIL"), f"{d} theta_crosscheck={x}"


def c_nomatch():
    codes, W = pipeline("nomatch", f"{FX}/list_nomatch.json")
    if codes != [0, 0, 0]:
        return "FAIL", f"exit codes {codes}"
    x, s = theta_ok(W)
    nm = all(s[lab]["N_tight_max"] == 0 and s[lab]["upper_boundary"]["status"] == "NO_MATCH" and s[lab]["theta_rep"] is None and s[lab]["maximal_intervals"] == []
             and "NO_MATCH" in s[lab]["descriptor"] for lab in LABS)
    return ("PASS" if x and nm else "FAIL"), f"theta_crosscheck={x} NO_MATCH status/no representative angle/strict JSON={nm}"


def c_noeligible():
    codes, W = pipeline("noeligible", f"{FX}/list_noeligible.json")
    if codes[:2] != [0, 0]:
        return "FAIL", f"exit codes {codes}"
    z = (zero_checks(W, "h11_015/np_0000506/cy_0000001", "ALL_MODES_FROZEN_OR_LATE") and zero_checks(W, "h11_015/np_0000501/cy_0000001", "NO_MODES_ABOVE_THRESHOLD"))
    ok, d = validators_pass(W)
    x = codes == [0, 0, 0] and theta_ok(W)[0]
    return ("PASS" if z and ok and x else "FAIL"), f"reasons distinct, Omega=0={z} {d} theta={x}"


def theta_pop(W):
    s = strict_json(f"{W}/final_analysis/theta/theta_population_summary.json")
    rows = table(f"{W}/final_analysis/theta/theta_population.csv")
    return s, rows


def c_populations():
    codes, W = pipeline("populations", f"{FX}/list_populations.json")
    if codes[:2] != [0, 0]:
        return "FAIL", f"exit codes {codes}"
    g = lambda k: f"h11_{'009' if k == 'g9' else '015'}/np_{ {'g9': '0000001', 'zero': '0000501', 'late': '0000506', 'inv': '0000507', 'part': '0000509', 'plate': '0000510'}[k]}/cy_0000001"
    rel = (validity_checks(W, g("g9"), "EVALUATED_POSITIVE", "NA", ">0", ">0", None, "PRIMARY")
           and validity_checks(W, g("zero"), "EVALUATED_ZERO", "NO_MODES_ABOVE_THRESHOLD", 0.0, 0.0, None, "PRIMARY")
           and validity_checks(W, g("late"), "EVALUATED_ZERO", "ALL_MODES_FROZEN_OR_LATE", 0.0, 0.0, None, "PRIMARY")
           and validity_checks(W, g("inv"), "INFLATION_INVALID_ALL", "ALL_MODES_INFLATION_EXCLUDED", None, None, None, "EXCLUDED_NOT_EVALUATED")
           and validity_checks(W, g("part"), "PARTIAL_VALIDITY", "SOME_MODES_INFLATION_EXCLUDED", ">0", None, ">0", "SENSITIVITY_ONLY")
           and validity_checks(W, g("plate"), "PARTIAL_VALIDITY", "SOME_MODES_INFLATION_EXCLUDED", None, None, 0.0, "SENSITIVITY_ONLY"))
    ok, d = validators_pass(W)
    if codes != [0, 0, 0]:
        return "FAIL", f"relic classes={rel} {d}; theta exit {codes}"
    x, s = theta_ok(W)
    xs = json.load(open(f"{W}/final_analysis/verification/crosscheck_python_sensitivity_partial_validity.json"))["all_checks_pass"]
    ps, rows = theta_pop(W)
    den = all(ps["scenarios"][lab]["primary"]["n_geometries"] == 3 and ps["scenarios"][lab]["sensitivity_partial_validity"]["n_geometries"] == 5
              and ps["scenarios"][lab]["excluded_not_evaluated"]["n_geometries"] == 1 and ps["scenarios"][lab]["total_geometries"] == 6 for lab in LABS)
    tg = {r["geometry_id"] for r in table(f"{W}/final_analysis/theta/theta_geometry_final.csv")}
    ts = {r["geometry_id"] for r in table(f"{W}/final_analysis/theta_sensitivity_partial_validity/theta_geometry_final.csv")}
    sets = tg == {g("g9"), g("zero"), g("late")} and ts == tg | {g("part"), g("plate")}
    lbl = s["population"] == "PRIMARY" and strict_json(f"{W}/final_analysis/theta_sensitivity_partial_validity/theta_summary_final.json")["population"] == "SENSITIVITY_INCLUDING_PARTIAL_VALIDITY"
    listed = {r["geometry_id"]: r["theta_population"] for r in rows}.get(g("inv")) == "EXCLUDED_NOT_EVALUATED"
    good = rel and ok and x and xs and den and sets and lbl and listed
    return ("PASS" if good else "FAIL"), f"relic classes={rel} {d} crosscheck primary={x} sensitivity={xs} denominators={den} sets={sets} labels={lbl} invalid listed={listed}"


def c_all_invalid():
    codes, W = pipeline("all_invalid", f"{FX}/list_all_invalid.json")
    if codes != [0, 0, 0]:
        return "FAIL", f"exit codes {codes}"
    ok, d = validators_pass(W); ps, rows = theta_pop(W)
    e = all(ps["scenarios"][lab]["primary"]["n_geometries"] == 0 and ps["scenarios"][lab]["primary"]["status"] == "EMPTY_POPULATION" for lab in LABS)
    return ("PASS" if ok and e else "FAIL"), f"{d} empty primary population reported={e}"


def c_boundary():
    codes, W = pipeline("boundary", f"{FX}/list_boundary.json")
    if codes != [0, 0, 0]:
        return "FAIL", f"exit codes {codes}"
    ok, d = validators_pass(W); x, s = theta_ok(W)
    return ("PASS" if ok and x else "FAIL"), f"{d} theta_crosscheck={x}"


def c_canonical_tamper():
    W = f"{T}/canonical_tamper"; os.makedirs(W); shutil.copytree(f"{T}/genuine/relic_input", f"{W}/relic_input", symlinks=True)
    env = {"WORK": W, "RELIC_RUN": f"{FX}/run.json"}
    S = f"{REPO}/engine/relic"; R = f"{W}/relic"; os.makedirs(R)
    sh([PY, f"{S}/00_compose_run_config.py", "relic", f"{REPO}/configs/methodology_v1.json", f"{FX}/run.json", f"{R}/template.json"], env)
    sh([PY, f"{S}/01_build_mode_table.py", f"{R}/template.json", f"{W}/relic_input", f"{R}/a1"], env)
    sh([PY, f"{S}/02_write_run_config.py", f"{R}/template.json", f"{W}/relic_input", f"{R}/a1", R, f"{R}/run_config.json"], env)
    lines = open(f"{R}/a1/canonical_modes.csv").read().splitlines(); open(f"{R}/a1/canonical_modes.csv", "w").write("\n".join(lines[:-1]) + "\n")
    jl = [P["JULIA_ANALYSIS"], "--startup-file=no", f"--project={REPO}/environment/julia_analysis", f"{S}/compute_relic_abundance.jl",
          f"{R}/run_config.json", f"{R}/a1/canonical_modes.csv", f"{R}/tables", "abundance"]
    env2 = {"JULIA_DEPOT_PATH": P["JULIA_ANALYSIS_DEPOT"]} if P.get("JULIA_ANALYSIS_DEPOT") else {}
    r = sh(jl, env2, log=f"{W}.driver.log")
    refused = r.returncode != 0 and "canonical mode table" in (r.stdout + r.stderr)
    return ("PASS" if refused else "FAIL"), f"driver exit {r.returncode}; refusal names the canonical table checksum: {refused}"


def c_composer():
    base = {"schema": "cyaxiverse_relic_run/1", "scenarios": [{"label": "a", "T_reh_GeV": 1e4}]}
    bad = {"unknown_toplevel_key": dict(base, production_run_aproved=True),
           "unknown_scenario_key": dict(base, scenarios=[{"label": "a", "T_reh_GeV": 1e4, "w_R": 0.33}]),
           "unknown_theta_section": dict(base, theta_analyses={"theta": {"theta_max": 0.2}}),
           "case_colliding_labels": dict(base, scenarios=[{"label": "Tr", "T_reh_GeV": 1e4}, {"label": "TR", "T_reh_GeV": 1e9}]),
           "theta_max_bool": dict(base, theta_analysis={"theta": {"theta_max": True}}),
           "tight_three_elements": dict(base, theta_analysis={"dark_matter": {"TIGHT_DM": [0.9, 1.0, 1.1]}}),
           "omega_dm_string": dict(base, theta_analysis={"dark_matter": {"Omega_DM_h2": "0.12"}})}
    D = f"{T}/composer"; os.makedirs(D); fails = []
    for k, v in bad.items():
        json.dump(v, open(f"{D}/{k}.json", "w"))
        for kind, meth in (("relic", "methodology_v1.json"), ("theta", "theta_methodology_v1.json")):
            r = sh([PY, f"{REPO}/engine/relic/00_compose_run_config.py", kind, f"{REPO}/configs/{meth}", f"{D}/{k}.json", f"{D}/{k}.{kind}.out.json"])
            if r.returncode != 0 and "Traceback" not in r.stderr:
                break
        else:
            fails.append(k)
    open(f"{D}/truncated.json", "w").write('{"schema": "cyaxiverse_relic_run/1", "scen')
    r = sh([PY, f"{REPO}/engine/relic/00_compose_run_config.py", "relic", f"{REPO}/configs/methodology_v1.json", f"{D}/truncated.json", f"{D}/t.out.json"])
    if r.returncode == 0 or "Traceback" in r.stderr:
        fails.append("truncated_json")
    return ("PASS" if not fails else "FAIL"), f"accepted or traceback: {fails}"


def c_engine_only():
    pe = f"{T}/paths_engine_only.env"
    # values are quoted: a path containing spaces (the README's .venv inside a checkout under such a directory) must survive `source`
    open(pe, "w").write("".join(f'{k}="{v}"\n' for k, v in P.items() if k not in ("CYAX_REPO", "JULIA_VMM", "JULIA_VMM_DEPOT")))
    codes, W = pipeline("engine_only", f"{FX}/list_genuine.json", paths=pe)
    return ("PASS" if codes == [0, 0, 0] else "FAIL"), f"exit codes {codes} without CYAX_REPO/JULIA_VMM"


def c_relocation():
    src = f"{T}/genuine"
    for kind in ("moved", "copied"):
        W = f"{T}/relocation_{kind}"; shutil.copytree(src, W, symlinks=True); shutil.rmtree(f"{W}/final_analysis")
        if kind == "moved":
            os.rename(src, src + ".hidden")
        r = sh(["bash", f"{REPO}/engine/run_theta.sh"], {"WORK": W, "RELIC_RUN": f"{FX}/run.json"}, log=f"{W}.log")
        if kind == "moved":
            os.rename(src + ".hidden", src)
        if not (r.returncode != 0 and "moved or copied" in (r.stdout + r.stderr)):
            return "FAIL", f"{kind} WORK: exit {r.returncode}, no clear relocation message"
    return "PASS", "moved and copied WORK both stopped with a relocation message"


def c_invalid_inputs():
    out = {}
    for k in sorted(json.load(open(f"{FX}/cases.json"))):
        if not k.startswith("bad_"):
            continue
        codes, W = pipeline(k, f"{FX}/list_{k}.json", theta=False)
        logs = "".join(open(f"{T}/{k}.{s}.log").read() for s in ("run_prepare_from_spectra.sh", "run_relic.sh") if os.path.exists(f"{T}/{k}.{s}.log"))
        js_ok = True
        a1 = f"{W}/relic/a1/a1_input_verification.json"
        if os.path.exists(a1):
            try:
                json.load(open(a1))
            except ValueError:
                js_ok = False
        out[k] = {"refused": codes[-1] != 0, "traceback": "Traceback" in logs, "json_ok": js_ok}
    bad = {k: v for k, v in out.items() if not v["refused"] or v["traceback"] or not v["json_ok"]}
    return ("PASS" if not bad else "FAIL"), f"{len(out)} cases; not clean: {bad}"



# ---- regression: theta stage with mixed NO_MATCH / matched scenarios (external audit I-01) --------------------------------------
# The genuine h11 = 15 geometry has Omega(theta_ref) = 7.6e-4 at 5 MeV (no TIGHT match for theta <= 0.5) and 1.24 at 1e4 GeV and above (match).
SCEN = {"5MeV": 0.005, "10MeV": 0.01, "20MeV": 0.02, "1e4GeV": 1e4, "1e10GeV": 1e10}


def mixed_run(name, labels):
    p = f"{T}/run_{name}.json"
    json.dump({"schema": "cyaxiverse_relic_run/1", "scenarios": [{"label": l, "T_reh_GeV": SCEN[l]} for l in labels]}, open(p, "w"))
    return p


def mixed_case(name, labels, lst="h15only", sens_match=()):
    """theta stage must complete for any order of matched / NO_MATCH scenarios and report each scenario correctly"""
    run = mixed_run(name, labels)
    codes, W = pipeline(name, f"{FX}/list_{lst}.json", run=run)
    if codes != [0, 0, 0]:
        return False, f"exit codes {codes}", W
    ok = True
    for pop, d, cc in (("primary", "theta", "crosscheck_python.json"), ("sensitivity", "theta_sensitivity_partial_validity", "crosscheck_python_sensitivity_partial_validity.json")):
        s = strict_json(f"{W}/final_analysis/{d}/theta_summary_final.json")
        ok &= json.load(open(f"{W}/final_analysis/verification/{cc}"))["all_checks_pass"]
        for l in labels:
            want_nm = SCEN[l] < 0.1 and not (pop == "sensitivity" and l in sens_match)   # the partial sum of a PARTIAL_VALIDITY geometry can match
            nm = s[l]["N_tight_max"] == 0 and s[l]["theta_rep"] is None and s[l]["maximal_intervals"] == [] and "NO_MATCH" in s[l]["descriptor"]
            mt = s[l]["N_tight_max"] >= 1 and s[l]["theta_rep"] is not None and len(s[l]["maximal_intervals"]) >= 1
            ok &= nm if want_nm else mt
        rows = table(f"{W}/final_analysis/{d}/theta_geometry_final.csv")
        ok &= {r["scenario"] for r in rows} == set(labels)
        for r in rows:                                   # NO_MATCH scenarios carry no representative angle, matched ones do
            ok &= (r["theta_rep_scenario"] in ("", "NA")) == (SCEN[r["scenario"]] < 0.1 and not (pop == "sensitivity" and r["scenario"] in sens_match))
    return ok, "", W


def c_theta_mixed():
    sub = {"all_matched": ["1e4GeV", "1e10GeV"], "nomatch_first": ["5MeV", "1e4GeV"], "matched_first": ["1e4GeV", "5MeV"],
           "alternating": ["5MeV", "1e4GeV", "10MeV", "1e10GeV", "20MeV"]}
    bad = {}
    for k, labels in sub.items():
        ok, d, _ = mixed_case("mixed_" + k, labels)
        if not ok:
            bad[k] = d or "assertion"
    ok, d, _ = mixed_case("mixed_partial_pop", ["5MeV", "1e4GeV"], lst="h15_partial", sens_match=("5MeV",))   # populations differ at 5 MeV
    if not ok:
        bad["partial_pop"] = d or "assertion"
    return ("PASS" if not bad else "FAIL"), f"all matched, NO_MATCH first, NO_MATCH last, alternating, primary+sensitivity populations; failures: {bad}"


def c_theta_retry():
    """a failed theta stage leaves no blocking final_analysis/ (partial output kept under a new name); the documented rerun then succeeds"""
    codes, W = pipeline("retry", f"{FX}/list_h15only.json", run=mixed_run("retry_ok", ["5MeV", "1e4GeV"]), theta=False)
    if codes != [0, 0]:
        return "FAIL", f"relic exit codes {codes}"
    badrun = f"{T}/run_retry_bad.json"
    json.dump({"schema": "cyaxiverse_relic_run/1", "scenarios": [{"label": l, "T_reh_GeV": SCEN[l]} for l in ("5MeV", "1e4GeV")],
               "theta_analysis": {"theta": {"theta_max": 2.0}}}, open(badrun, "w"))
    r1 = sh(["bash", f"{REPO}/engine/run_theta.sh"], {"WORK": W, "RELIC_RUN": badrun}, log=f"{T}/retry.fail.log")
    ok1 = r1.returncode != 0 and "final_analysis" not in os.listdir(W)                 # refused configuration: nothing blocks the rerun
    # a failure AFTER output was written (a needed relic table is missing): the partial final_analysis/ is kept under a new name
    tab = f"{W}/relic/tables/master_modes_5MeV.csv"; os.rename(tab, tab + ".hidden")
    r2 = sh(["bash", f"{REPO}/engine/run_theta.sh"], {"WORK": W, "RELIC_RUN": f"{T}/run_retry_ok.json"}, log=f"{T}/retry.fail2.log")
    os.rename(tab + ".hidden", tab)
    names = os.listdir(W); kept = [x for x in names if x.startswith("final_analysis.failed.")]
    ok2 = r2.returncode != 0 and "final_analysis" not in names and len(kept) == 1
    r3 = sh(["bash", f"{REPO}/engine/run_theta.sh"], {"WORK": W, "RELIC_RUN": f"{T}/run_retry_ok.json"}, log=f"{T}/retry.rerun.log")
    x = r3.returncode == 0 and theta_ok(W)[0]
    return ("PASS" if ok1 and ok2 and x else "FAIL"), f"refused config leaves no blocker={ok1} failure after output keeps partial under a new name={ok2} ({kept}) rerun ok={x}"


def c_repeat():
    codes, W = pipeline("repeat", f"{FX}/list_genuine.json")
    same = codes == [0, 0, 0] and all(open(f"{W}/relic/tables/{f}", "rb").read() == open(f"{T}/genuine/relic/tables/{f}", "rb").read()
                                      for lab in LABS for f in (f"master_modes_{lab}.csv", f"geometry_summary_{lab}.csv", f"geometry_abundance_validity_{lab}.csv"))
    th = codes == [0, 0, 0] and open(f"{W}/final_analysis/theta/theta_summary_final.json", "rb").read() == open(f"{T}/genuine/final_analysis/theta/theta_summary_final.json", "rb").read()
    return ("PASS" if same and th else "FAIL"), f"tables identical={same} theta summary identical={th}"


for n, fn in [("genuine", c_genuine), ("genuine_verify", c_genuine_verify), ("zero_mixed", c_zero_mixed), ("zero_only", c_zero_only),
              ("integer", c_integer), ("nomatch", c_nomatch), ("noeligible", c_noeligible), ("populations", c_populations),
              ("all_invalid", c_all_invalid), ("boundary", c_boundary),
              ("canonical_tamper", c_canonical_tamper), ("composer", c_composer), ("engine_only", c_engine_only),
              ("relocation", c_relocation), ("invalid_inputs", c_invalid_inputs), ("theta_mixed", c_theta_mixed), ("theta_retry", c_theta_retry), ("repeat", c_repeat)]:
    case(n, fn)
summary = {"test": "external_acceptance", "cases": results, "n_pass": sum(r["status"] == "PASS" for r in results),
           "n_fail": sum(r["status"] == "FAIL" for r in results), "n_skipped": sum(r["status"] == "SKIPPED" for r in results)}
summary["verdict"] = "PASS" if summary["n_fail"] == 0 else "FAIL"
json.dump(summary, open(f"{T}/test_external_acceptance.json", "w"), indent=1)
print(f"EXTERNAL ACCEPTANCE {summary['verdict']}: {summary['n_pass']} passed, {summary['n_fail']} failed, {summary['n_skipped']} skipped")
sys.exit(0 if summary["n_fail"] == 0 else 1)
