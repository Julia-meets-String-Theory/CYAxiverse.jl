"""Methodology v2 validation: constant pre-reheating equation of state w_R per reheating scenario.

Every check recomputes the physics with a method that is NOT the engine's closed-form evaluation: the onset 3H(z_a) = m_a is found by
bisection in 50-digit arithmetic on H(z) of Fuzzy eq. (2.2), then Omega follows from eq. (2.10). Nothing is read from expected-value files
except the v1 byte-identity hashes (checksums of the frozen v1 example tables, computed before methodology v2 was introduced).

  A  physics (no pipeline)        independent root-finding vs closed form, mass exponents, normalisation, w -> 0, m = 3 H_R continuity, domain
  B  composer / engine refusal    unsupported or ambiguous inputs fail with a named reason (never a traceback, never a silent default)
  C  pipeline grid                synthetic mass scan, 3 T_R x 7 w_R in ONE run: schema, validators C/B, per-row independent recomputation,
                                  POST/class/validity invariance, geometry sums, w_R provenance in every table, theta analysis on nonzero w_R
  D  v1 compatibility             v1 example tables byte-identical to the frozen checksums; v2 at w_R = 0 == v1 apart from the w_R column
  E  reproducibility              the same v2 run twice gives byte-identical tables and theta outputs
  F  numerical range              strongly negative w_R: explicit stop (no clamp, no zero), continuity at w_R -> 0 through the engine
  G  domain edges                 w_R = 1, 0.999, -0.9, -0.85, -0.8, 1e-9 through the engine, per-row independent recomputation

Usage: SHOROQ_PATHS=paths.env python tests/test_wR_general.py WORK_DIR        (WORK_DIR must not exist)
"""
import csv, hashlib, json, math, os, shutil, subprocess, sys

import h5py
import mpmath as mp
import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "engine", "relic"))
import reference_mp as RM  # noqa: E402

T = os.path.abspath(sys.argv[1]); os.path.exists(T) and sys.exit(f"{T} exists")
os.makedirs(T)
ENVFILE = os.environ.get("SHOROQ_PATHS", f"{REPO}/configs/paths.env")
P = {}
for line in open(ENVFILE):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); P[k] = v.strip().strip('"')
PY = P["PYTHON"]
RESULTS = []
mp.mp.dps = 50
REF = RM.Ref()


def record(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail}); print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def sh(args, env=None, log=None):
    e = dict(os.environ); e["SHOROQ_PATHS"] = ENVFILE; e.update(env or {})
    r = subprocess.run(args, capture_output=True, text=True, env=e)
    if log:
        open(log, "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    return r


# ------------------------------------------------------------------ A: independent physics
def omega_by_root_finding(m, f, theta, TR, w):
    """Onset of the pre-reheating epoch by bisection on H(z) of Fuzzy (2.2), then (2.10). m, f in GeV (mpf)."""
    HR = REF.HR(TR); xR = mp.log(REF.onepz_R(TR)); w = mp.mpf(w)
    lnH = lambda x: mp.log(HR) + mp.mpf(3) / 2 * (1 + w) * (x - xR)            # H(z) = H_R ((1+z)/(1+z_R))^(3(1+w)/2), (2.2) for z >= z_R
    g = lambda x: lnH(x) - mp.log(m / 3)                                        # 3 H(z_a) = m_a
    lo, hi = xR, xR + 2000
    assert g(lo) < 0 < g(hi)
    for _ in range(400):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    x = (lo + hi) / 2
    return m**2 * f**2 * theta**2 * mp.exp(-3 * x) / (2 * 3 * RM.H100**2 * RM.MPL**2), mp.exp(x)


rel = lambda a, b: float(abs(mp.mpf(a) / mp.mpf(b) - 1))
THETA = mp.pi / 2
TRS = [mp.mpf("0.005"), mp.mpf("1e4"), mp.mpf("1e10")]
WS = ["-0.5", "-0.3333333333333333", "0", "0.3333333333333333", "0.17", "-0.42", "0.77", "1", "-0.9"]
worst = 0.0
for TR in TRS:
    HR = REF.HR(TR)
    for w in WS:
        for r in ("1.0000001", "10", "1e5", "1e12"):
            m = 3 * HR * mp.mpf(r); f = mp.mpf("1e16")
            om_root, opz_root = omega_by_root_finding(m, f, THETA, TR, w)
            om_ref, opz_ref = REF.omega_pre_eq210_215(m, f, THETA, TR, mp.mpf(w))
            worst = max(worst, rel(om_root, om_ref), rel(opz_root, opz_ref))
record("A1 onset found by root finding on H(z) of (2.2) equals closed form (2.15)+(2.10) [3 T_R x 9 w_R x 4 masses]", worst < 1e-35, f"max rel {worst:.1e}")

bad = []
for TR in TRS:
    HR = REF.HR(TR); f = mp.mpf("1e16")
    for w in WS:
        wm = mp.mpf(w); expo = 2 * wm / (1 + wm)
        m1, m2 = 3 * HR * mp.mpf("1e3"), 3 * HR * mp.mpf("1e9")
        o0 = [REF.omega_pre_eq210_215(m, f, THETA, TR, 0)[0] for m in (m1, m2)]
        ow = [REF.omega_pre_eq210_215(m, f, THETA, TR, wm)[0] for m in (m1, m2)]
        slope = mp.log(ow[1] / ow[0]) / mp.log(m2 / m1)                     # d ln Omega / d ln m at fixed f, theta, T_R
        bad += [("slope", str(TR), w)] if abs(slope - expo) > mp.mpf("1e-35") else []
        for m, a, b in zip((m1, m2), ow, o0):
            bad += [("ratio", str(TR), w)] if rel(a / b, (m / (3 * HR)) ** expo) > 1e-35 else []
record("A2 mass exponent 2w/(1+w) and Omega(w)/Omega(0) = (m/3H_R)^(2w/(1+w)); w=0:0, -1/3:-1, -1/2:-2, 1/3:+1/2",
       not bad and abs((2 * mp.mpf(1) / 3) / (1 + mp.mpf(1) / 3) - mp.mpf(1) / 2) < 1e-40 and abs(2 * mp.mpf(-1) / 3 / (1 - mp.mpf(1) / 3) + 1) < 1e-40
       and abs(2 * mp.mpf(-1) / 2 / (1 - mp.mpf(1) / 2) + 2) < 1e-40, str(bad[:2]))

bad = []
for TR in TRS:
    HR = REF.HR(TR); f = mp.mpf("1e16")
    for w in WS:
        m = 3 * HR
        om, opz = REF.omega_pre_eq210_215(m, f, THETA, TR, mp.mpf(w))
        # normalisation: at m = 3 H_R the onset is at reheating, (1+z_a) = (1+z_R), for every w_R; Omega = m^2 f^2 theta^2 / [2 (1+z_R)^3 3 M^2 H100^2]
        bad += [("opz", str(TR), w)] if rel(opz, REF.onepz_R(TR)) > 1e-40 else []
        bad += [("norm", str(TR), w)] if rel(om, REF.omega_pre_closed_form(f, THETA, TR)) > 1e-40 else []
record("A3 normalisation: at m = 3H_R every w_R gives 1+z_a = 1+z_R and Omega = (9/2) f^2 theta^2 H_R^2 s0/s(T_R) / (rho_c/h^2)", not bad, str(bad[:2]))

tiny = []
for TR in TRS:
    HR = REF.HR(TR); f = mp.mpf("1e16"); m = 3 * HR * mp.mpf("1e7"); w = mp.mpf("1e-12")
    o0 = REF.omega_pre_eq210_215(m, f, THETA, TR, 0)[0]; ow = REF.omega_pre_eq210_215(m, f, THETA, TR, w)[0]
    d = (ow / o0 - 1) / w                                                     # -> 2 ln(m/3H_R) as w -> 0
    tiny.append(rel(d, 2 * mp.log(m / (3 * HR))))
record("A4 w_R -> 0: d(Omega/Omega_0)/dw = 2 ln(m/3H_R)", max(tiny) < 1e-9, f"max rel {max(tiny):.1e}")

bad = []                       # continuity of the PRE and POST branches at m = 3H_R (w_R-independent threshold)
for TR in TRS:
    HR = REF.HR(TR); f = mp.mpf("1e16")
    m = 3 * HR * (1 - mp.mpf("1e-12"))                                       # POST just below the threshold
    Tosc = REF.T_osc(m); assert REF.classify(m, TR) == "POST_REHEATING"
    post = REF.omega_post(m, f, THETA, Tosc)
    m2 = 3 * HR * (1 + mp.mpf("1e-12"))
    assert REF.classify(m2, TR) == "PRE_REHEATING"
    for w in WS:
        pre = REF.omega_pre_eq210_215(m2, f, THETA, TR, mp.mpf(w))[0]
        # the only difference at the threshold: the full H (matter + Lambda) of the POST branch vs radiation-only H_R of (2.2): relative O(T_eq/T_R)
        tol = 4 * float(REF.T_eq() / TR) + 1e-9 * abs(float(2 * mp.mpf(w) / (1 + mp.mpf(w)))) + 1e-9
        bad += [(str(TR), w, rel(pre, post), tol)] if rel(pre, post) > max(tol, 1e-9) else []
record("A5 continuity PRE/POST at m = 3H_R for every w_R (difference only the matter-term O(T_eq/T_R))", not bad, str(bad[:2]))

# ------------------------------------------------------------------ B: refusals
COMPOSE = f"{REPO}/engine/relic/00_compose_run_config.py"


def compose(run, methodology="v2", kind="relic", name="c"):
    p = f"{T}/b_{name}.json"; json.dump(run, open(p, "w"))
    r = sh([PY, COMPOSE, kind, f"{REPO}/configs/methodology_{methodology}.json", p, f"{T}/b_{name}_out.json"])
    return r


def scen(**kw):
    return {"schema": "cyaxiverse_relic_run/2", "scenarios": [dict({"label": "a", "T_reh_GeV": 1e4}, **kw)]}


cases = {
    "w_R = -1": (scen(w_R=-1), "outside the supported domain"),
    "w_R = -1.5": (scen(w_R=-1.5), "outside the supported domain"),
    "w_R = 1.0000001": (scen(w_R=1.0000001), "outside the supported domain"),
    "w_R = 2": (scen(w_R=2), "outside the supported domain"),
    "w_R = true": (scen(w_R=True), "must be a finite number"),
    "w_R = '0.1'": (scen(w_R="0.1"), "must be a finite number"),
    "w_R = null": (scen(w_R=None), "must be a finite number"),
    "unknown scenario key": (scen(w_r=0.1), "unknown key"),
    "duplicate (T_reh, w_R)": ({"schema": "cyaxiverse_relic_run/2", "scenarios": [{"label": "a", "T_reh_GeV": 1e4, "w_R": 0.2},
                                                                                    {"label": "b", "T_reh_GeV": 1e4, "w_R": 0.2}]}, "same (T_reh_GeV, w_R)"),
    "v1 run file with v2 methodology": ({"schema": "cyaxiverse_relic_run/1", "scenarios": [{"label": "a", "T_reh_GeV": 1e4}]}, "schema must be cyaxiverse_relic_run/2"),
    "T_reh below T_eq": (scen(T_reh_GeV=1e-12, w_R=0.1), "violates"),
}
for nm, (run, msg) in cases.items():
    r = compose(run, name=nm.replace(" ", "_").replace("'", "").replace("=", ""))
    record(f"B composer refuses: {nm}", r.returncode != 0 and msg in (r.stdout + r.stderr) and "Traceback" not in r.stderr, (r.stderr or r.stdout).strip()[-120:])
r = compose({"schema": "cyaxiverse_relic_run/2", "scenarios": [{"label": "a", "T_reh_GeV": 1e4, "w_R": 1}, {"label": "b", "T_reh_GeV": 1e4, "w_R": -0.999}]}, name="edge_ok")
record("B composer accepts the domain edges w_R = 1 and w_R = -0.999 (the latter with warnings)", r.returncode == 0 and "WARNING" in r.stdout, r.stdout.strip()[-100:])
r = compose({"schema": "cyaxiverse_relic_run/1", "scenarios": [{"label": "a", "T_reh_GeV": 1e4, "w_R": 0.1}]}, methodology="v1", name="v1_w")
record("B methodology v1 refuses w_R", r.returncode != 0 and "w_R is fixed at 0" in (r.stdout + r.stderr), (r.stderr or r.stdout).strip()[-100:])
r = compose(scen(), methodology="v1", name="v1_schema2")
record("B methodology v1 with a v2 run file is refused", r.returncode != 0 and "schema must be cyaxiverse_relic_run/1" in (r.stdout + r.stderr))
r = compose({"schema": "cyaxiverse_relic_run/2", "scenarios": [{"label": "a", "T_reh_GeV": 1e4}]}, name="default")
rc = json.load(open(f"{T}/b_default_out.json"))
record("B documented default: a scenario without w_R is composed with w_R = 0.0 and the run records it",
       r.returncode == 0 and rc["reheating"]["scenarios"][0]["w_R"] == 0.0 and rc["methodology_version"] == 2 and rc["reheating"]["w_R_default"] == 0.0)

# ------------------------------------------------------------------ C: pipeline on a synthetic mass scan
BASE = f"{REPO}/tests/external_acceptance/data/h11_015/np_0000001/cy_0000001/cyax.h5"
FXD = f"{T}/fixtures"; os.makedirs(FXD)
HRs = {str(t): REF.HR(t) for t in TRS}
lg = lambda x: float(mp.log10(x * mp.mpf(10) ** 9))                        # GeV -> log10 eV
thresholds = []
for t in TRS:                                                              # masses just above and below 3 H_R of each reheating temperature
    for fac in ("1.000001", "0.999999"):
        thresholds.append(lg(3 * HRs[str(t)] * mp.mpf(fac)))
ladder = [-32.0, -29.0, -27.5, -26.0, -22.0, -15.0, -8.0, -3.0, 2.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 23.0, 23.5]       # FROZEN-free, LATE, POST, PRE, > 3 H_I,max
MA, MB = ladder[:15], ladder[15:] + thresholds                              # a h11 = 15 spectrum holds at most 15 modes
FA = [16.0 + 0.1 * (i % 7) for i in range(len(MA))]; FB = [15.5 + 0.15 * (i % 6) for i in range(len(MB))]
M2 = [-24.3, -17.2, -11.1, -5.5, 0.7, 4.4, 8.8, 13.3, 19.9]
F2 = [15.2 + 0.2 * (i % 5) for i in range(len(M2))]


def write_spec(name, np_idx, m, fp):
    d = f"{FXD}/spectra/{name}/h11_015/np_{np_idx:07d}/cy_0000001"; os.makedirs(d); p = f"{d}/cyax.h5"
    shutil.copy(BASE, p); m = np.asarray(m, float); fp = np.asarray(fp, float); n = len(m)
    with h5py.File(p, "r+") as f:
        g = f["spectrum/physical"]
        for k in ("m", "fpert_log10", "lambda_self_log10", "lambda_self_sign", "mode_indices"):
            del g[k]
        g["m"] = m; g["fpert_log10"] = fp; g["lambda_self_log10"] = 2 * (m - 9 - fp); g["lambda_self_sign"] = np.ones(n, np.int64)
        g["mode_indices"] = np.arange(15 - n, 15, dtype=np.int64); g["metadata/polytope"][()] = np_idx
    return p


s1, s2, s3 = write_spec("scan_a", 601, MA, FA), write_spec("scan_b", 602, MB, FB), write_spec("scan_c", 603, M2, F2)
ev = f"{FXD}/SYNTHETIC_NOTICE.txt"; open(ev, "w").write("SYNTHETIC test fixture: arrays rewritten, not a physical vacuum\n")
evsha = hashlib.sha256(open(ev, "rb").read()).hexdigest()


def spectra_list(path, paths):
    json.dump({"schema": "cyaxiverse_relic_spectra/1", "vacuum_policy": {"on_unusable": "refuse"},
               "spectra": [{"path": q, "vacuum_validation": {"state": "DOCUMENTED_EXTERNAL", "method": "SYNTHETIC fixture", "reference": "tests/test_wR_general.py",
                                                             "evidence": [{"path": ev, "sha256": evsha}]}} for q in paths]}, open(path, "w"), indent=1)


LIST = f"{FXD}/list.json"; spectra_list(LIST, [s1, s2, s3])
LISTMIN = f"{REPO}/examples/minimal/spectra_list.json"


def pipeline(name, lst, run, theta=True):
    W = f"{T}/{name}"; env = {"WORK": W, "RELIC_RUN": run}; codes = []
    for step in (["engine/run_prepare_from_spectra.sh", lst], ["engine/run_relic.sh"]) + ((["engine/run_theta.sh"],) if theta else ()):
        r = sh(["bash", f"{REPO}/{step[0]}"] + step[1:], env, log=f"{T}/{name}.{os.path.basename(step[0])}.log"); codes.append(r.returncode)
        if r.returncode:
            break
    return codes, W


def label(t, w):
    return "T%s-w%s" % (("%g" % float(t)).replace("+", ""), ("%g" % float(w)).replace("-", "m").replace(".", "p"))


GRID_W = [-0.5, -1 / 3, 0.0, 1 / 3, 0.17, -0.42, 0.77]
GRID_T = [0.005, 1e4, 1e10]
grid = [{"label": label(t, w), "T_reh_GeV": t, "w_R": w} for t in GRID_T for w in GRID_W]
GRUN = f"{T}/run_grid.json"
json.dump({"schema": "cyaxiverse_relic_run/2", "description": "3 T_R x 7 w_R in one run", "scenarios": grid, "production_run_approved": False,
           "theta_analysis": {"theta": {"theta_max": 0.5}, "optimisation": {"bootstrap_resamples": 30, "bootstrap_seed": 11}}}, open(GRUN, "w"), indent=1)
codes, WG = pipeline("grid", LIST, GRUN)
record("C1 21 scenarios (3 T_R x 7 w_R, incl. 0.17, -0.42, 0.77) in ONE run: prepare, relic, theta all succeed", codes == [0, 0, 0], str(codes))
if codes == [0, 0, 0]:
    tb = f"{WG}/relic/tables"
    val = json.load(open(f"{tb}/output_validation.json")); ver = json.load(open(f"{WG}/relic/validation/production_verification.json"))
    record("C2 schema/invariant validator (w_R in all four tables, branch labels, summaries) passes for all scenarios", val["all_pass"], f"errors={val['n_errors']}")
    record("C3 independent verifiers C (float64, every row) and B (50-digit sample, whole geometries) pass with per-scenario w_R",
           ver["all_pass"] and ver["scenario_w_R"] == {g["label"]: g["w_R"] for g in grid})
    master = {g["label"]: list(csv.DictReader(open(f"{tb}/master_modes_{g['label']}.csv"))) for g in grid}
    geom = {g["label"]: list(csv.DictReader(open(f"{tb}/geometry_summary_{g['label']}.csv"))) for g in grid}
    # --- per-row independent recomputation of every PRE abundance (root finding, 50 digits)
    worst = 0.0; npre = 0; cats = {}
    for g in grid:
        TR = mp.mpf(repr(g["T_reh_GeV"]))
        for r in master[g["label"]]:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
            if r["abundance_state"] == "INCLUDED_PRE_REHEATING":
                m = mp.mpf(r["mass_eV"]) * mp.mpf(10) ** -9; f = mp.mpf(r["f_pert_GeV"])
                om, opz = omega_by_root_finding(m, f, THETA, TR, repr(g["w_R"]))
                worst = max(worst, rel(r["Omega_h2_at_theta_ref"], om), rel(float(r["z_osc"]) + 1, opz)); npre += 1
                if float(r["w_R"]) != g["w_R"] or r["abundance_branch"] != ("fuzzy_wR0_eq2.10_2.15" if g["w_R"] == 0 else "fuzzy_wR_general_eq2.10_2.15"):
                    worst = 1.0
    record(f"C4 every PRE row ({npre}) equals the independent root-finding result; w_R and branch label recorded per row", worst < 1e-12 and npre > 100, f"max rel {worst:.1e}, categories {cats}")
    # --- invariance across w_R at fixed T_R: class, validity, state, POST/LATE/FROZEN values
    inv_bad = []
    for t in GRID_T:
        ref = master[label(t, 0.0)]
        for w in GRID_W:
            cur = master[label(t, w)]
            for a, b in zip(ref, cur):
                same = ("geometry_id", "mode_index", "category", "abundance_state", "inflation_valid", "included_in_cdm_sum", "mass_eV", "f_pert_GeV")
                if any(a[k] != b[k] for k in same):
                    inv_bad.append((t, w, "class/state")); break
                if a["category"] != "PRE_REHEATING" and any(a[k] != b[k] for k in ("Omega_h2_at_theta_ref", "T_osc_GeV", "z_osc", "Omega_h2_late_census_only_at_theta_ref")):
                    inv_bad.append((t, w, "POST/LATE/FROZEN value")); break
    record("C5 mode class, inflation validity, inclusion and all POST/LATE/FROZEN values are independent of w_R at fixed T_R", not inv_bad, str(inv_bad[:2]))
    # --- Omega(w)/Omega(0) per mode equals (m/3H_R)^(2w/(1+w)) (the validators check absolute values; this checks the relative law through the engine)
    ratio_bad = 0.0
    for t in GRID_T:
        HR = float(HRs[str(mp.mpf(repr(t)))]) if str(mp.mpf(repr(t))) in HRs else float(REF.HR(mp.mpf(repr(t))))
        for w in GRID_W:
            for a, b in zip(master[label(t, 0.0)], master[label(t, w)]):
                if b["abundance_state"] == "INCLUDED_PRE_REHEATING":
                    r_ = float(a["mass_eV"]) * 1e-9 / (3 * HR); expo = 2 * w / (1 + w)
                    ratio_bad = max(ratio_bad, abs(float(b["Omega_h2_at_theta_ref"]) / float(a["Omega_h2_at_theta_ref"]) / r_**expo - 1))
    record("C6 engine ratio Omega(w)/Omega(0) = (m/3H_R)^(2w/(1+w)) for every PRE mode of the grid", ratio_bad < 1e-11, f"max rel {ratio_bad:.1e}")
    # --- geometry sums
    gs_bad = 0.0
    for g in grid:
        tot = {}
        for r in master[g["label"]]:
            if r["included_in_cdm_sum"] == "true":
                tot[r["geometry_id"]] = tot.get(r["geometry_id"], 0.0) + float(r["Omega_h2_at_theta_ref"])
        for r in geom[g["label"]]:
            gs_bad = max(gs_bad, abs(float(r["Omega_geometry_h2_at_theta_ref"]) / tot[r["geometry_id"]] - 1)) if r["geometry_id"] in tot else gs_bad
            if float(r["w_R"]) != g["w_R"] or r["scenario"] != g["label"] or float(r["T_reh_GeV"]) != g["T_reh_GeV"]:
                gs_bad = 1.0
    record("C7 geometry totals = sum of included modes in every scenario; scenario, T_reh_GeV and w_R identify every geometry row", gs_bad < 1e-13, f"max rel {gs_bad:.1e}")
    # --- boundary continuity through the engine: the modes at 3H_R*(1 +- 1e-6): for each w the PRE mode just above equals the w=0 value to (1e-6 * exponent)
    cb = 0.0
    for t in GRID_T:
        HR = float(REF.HR(mp.mpf(repr(t))))
        for w in GRID_W:
            for a, b in zip(master[label(t, 0.0)], master[label(t, w)]):
                m = float(a["mass_eV"]) * 1e-9
                if b["abundance_state"] == "INCLUDED_PRE_REHEATING" and m < 3 * HR * (1 + 2e-6):
                    cb = max(cb, abs(float(b["Omega_h2_at_theta_ref"]) / float(a["Omega_h2_at_theta_ref"]) - 1) / (1e-5 + 2e-6 * abs(2 * w / (1 + w))))
    record("C8 continuity at m = 3H_R through the engine (PRE value within 2e-6 x exponent of the w=0 value)", cb <= 1.0, f"ratio to allowance {cb:.2f}")
    # --- theta analysis on nonzero w_R
    sm = json.load(open(f"{WG}/final_analysis/theta/theta_summary_final.json")); ps = json.load(open(f"{WG}/final_analysis/theta/theta_population_summary.json"))
    xc = json.load(open(f"{WG}/final_analysis/verification/crosscheck_python.json"))["all_checks_pass"]
    xs = json.load(open(f"{WG}/final_analysis/verification/crosscheck_python_sensitivity_partial_validity.json"))["all_checks_pass"]
    iv = json.load(open(f"{WG}/final_analysis/verification/input_verification.json"))
    tw = {g["label"]: g["w_R"] for g in grid}
    record("C9 theta analysis runs on all 21 scenarios: input verification, independent Python cross-check (primary and sensitivity) pass",
           xc and xs and iv["all_pass"], f"{iv['n_pass']}/{iv['n_checks']} input checks")
    record("C10 theta outputs carry w_R of each scenario (summary, population summary, geometry/interval tables)",
           all(sm[l]["w_R"] == tw[l] for l in tw) and all(ps["scenarios"][l]["w_R"] == tw[l] for l in tw) and
           all(float(r["w_R"]) == tw[r["scenario"]] for r in csv.DictReader(open(f"{WG}/final_analysis/theta/theta_geometry_final.csv"))))
    # --- independent common-angle matching: theta_90, theta_100 from Omega(theta_ref), scaling Omega ~ theta^2, N_max by sweep
    th_bad = []
    for g in grid:
        rows = list(csv.DictReader(open(f"{WG}/final_analysis/theta/theta_geometry_final.csv")))
        rows = [r for r in rows if r["scenario"] == g["label"]]
        iv_ = []
        for r in rows:
            om = float(r["Omega_geometry_h2_at_theta_ref"])
            if om <= 0:
                continue
            a, b = (math.pi / 2) * math.sqrt(0.9 * 0.12 / om), (math.pi / 2) * math.sqrt(1.0 * 0.12 / om)
            if abs(float(r["theta_90"]) / a - 1) > 1e-12 or abs(float(r["theta_100"]) / b - 1) > 1e-12:
                th_bad.append((g["label"], r["geometry_id"]))
            a, b = max(a, 0.0), min(b, 0.5)
            if a <= b:
                iv_.append((a, b))
        pts = sorted({x for a, b in iv_ for x in (a, b)})
        nmax = max([sum(a <= x <= b for a, b in iv_) for x in pts] + [0])
        if nmax != sm[g["label"]]["N_tight_max"]:
            th_bad.append((g["label"], "N_max", nmax, sm[g["label"]]["N_tight_max"]))
    record("C11 independent theta thresholds (Omega ~ theta^2, 0.9 <= R <= 1) and N_max sweep agree with the engine for all 21 scenarios", not th_bad, str(th_bad[:2]))

# ------------------------------------------------------------------ D: v1 compatibility
V1_SHA = {   # sha256 of the v1 example tables of the frozen engine (research/axion_relic_abundance/examples/minimal, 3 T_reh), computed before methodology v2
    "geometry_abundance_validity_1e10GeV.csv": None}
codes, W1 = pipeline("v1", LISTMIN, f"{REPO}/examples/minimal/run.json", theta=True)
codes0, W0 = pipeline("v2w0", LISTMIN, f"{REPO}/examples/wR_general/run_v2_w0_reproduces_v1.json", theta=True)
record("D1 v1 example and v2 at w_R = 0 (three T_R) both run end to end", codes == [0, 0, 0] and codes0 == [0, 0, 0], f"{codes} {codes0}")
if codes == [0, 0, 0] and codes0 == [0, 0, 0]:
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    pin = json.load(open(f"{REPO}/tests/v1_example_sha256.json")) if os.path.exists(f"{REPO}/tests/v1_example_sha256.json") else {}
    bad = [f for f, h in pin.get("relic_tables", {}).items() if sha(f"{W1}/relic/tables/{f}") != h]
    bad += [f for f, h in pin.get("theta", {}).items() if sha(f"{W1}/final_analysis/{f}") != h]
    record("D2 v1 example: relic tables and theta outputs byte-identical to the checksums recorded before methodology v2", pin and not bad, f"{len(pin.get('relic_tables', {}))}+{len(pin.get('theta', {}))} files, differing {bad[:3]}")
    diff = []
    for f in sorted(os.listdir(f"{W1}/relic/tables")):
        if f.endswith(".csv"):
            A = list(csv.reader(open(f"{W1}/relic/tables/{f}"))); B = list(csv.reader(open(f"{W0}/relic/tables/{f}")))
            if B[0][-1] != "w_R" or [r[:-1] for r in B] != A or {r[-1] for r in B[1:]} - {"0"}:
                diff.append(f)
    record("D3 v2 at w_R = 0 equals v1 in every relic table (text identical) once the w_R column is dropped", not diff, str(diff[:3]))
    tdiff = []
    for root, _, files in os.walk(f"{W1}/final_analysis/theta"):
        for f in files:
            if f.endswith(".csv") and f not in ("theta_bootstrap.csv",):
                a = list(csv.DictReader(open(f"{root}/{f}"))); b = list(csv.DictReader(open(f"{root.replace(W1, W0)}/{f}")))
                strip = lambda rows: [{k: v for k, v in r.items() if k != "w_R"} for r in rows]
                if strip(a) != strip(b):
                    tdiff.append(f)
    record("D4 v2 at w_R = 0 equals v1 in the theta tables once the w_R column is dropped", not tdiff, str(tdiff[:3]))

# ------------------------------------------------------------------ E: reproducibility
codes2, WG2 = pipeline("grid_repeat", LIST, GRUN)
if codes == [0, 0, 0] and codes2 == [0, 0, 0]:
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    bad = [f for f in sorted(os.listdir(f"{WG}/relic/tables")) if f.endswith((".csv", "population_validity_summary.json", "global_summary.json")) and
           sha(f"{WG}/relic/tables/{f}") != sha(f"{WG2}/relic/tables/{f}")]
    for root, _, files in os.walk(f"{WG}/final_analysis/theta"):
        bad += [f for f in files if f.endswith((".csv", ".json")) and sha(f"{root}/{f}") != sha(f"{root.replace(WG, WG2)}/{f}")]
    record("E1 repeating the 21-scenario v2 run gives byte-identical relic tables and theta outputs", not bad, str(bad[:3]))
else:
    record("E1 repeat run", False, f"{codes2}")

# ------------------------------------------------------------------ F: numerical range and engine-level refusal
def one_run(name, scenarios):
    run = f"{T}/{name}.json"
    json.dump({"schema": "cyaxiverse_relic_run/2", "scenarios": scenarios, "production_run_approved": False}, open(run, "w"))
    return pipeline(name, LIST, run, theta=False)


c, Wf = one_run("range_stop", [{"label": "extreme", "T_reh_GeV": 0.005, "w_R": -0.99}])
log = open(f"{T}/range_stop.run_relic.sh.log").read() if os.path.exists(f"{T}/range_stop.run_relic.sh.log") else ""
record("F1 w_R = -0.99 at 5 MeV: heavy PRE modes leave the float64 range, the run stops with a named reason, no table is written, nothing is clamped",
       c[-1] != 0 and "outside the float64 normal range" in log and not os.path.exists(f"{Wf}/relic/tables/master_modes_extreme.csv"), log.strip()[-160:])
tamper = json.load(open(f"{WG}/relic/run_config.json"))
tamper["reheating"]["scenarios"][0]["w_R"] = -1.0
os.makedirs(f"{T}/tamper", exist_ok=True); json.dump(tamper, open(f"{T}/tamper/run_config.json", "w"))
jl = P["JULIA_ANALYSIS"]; env = dict(os.environ)
if P.get("JULIA_ANALYSIS_DEPOT"):
    env["JULIA_DEPOT_PATH"] = P["JULIA_ANALYSIS_DEPOT"]
rj = subprocess.run([jl, "--startup-file=no", f"--project={REPO}/environment/julia_analysis", f"{REPO}/engine/relic/compute_relic_abundance.jl",
                     f"{T}/tamper/run_config.json", f"{WG}/relic/a1/canonical_modes.csv", f"{T}/tamper/out", "abundance"], capture_output=True, text=True, env=env)
record("F2 the engine itself refuses w_R = -1 in a hand-edited run configuration", rj.returncode != 0 and "-1 < w_R <= 1" in rj.stderr, rj.stderr.strip()[-160:])
tamper = json.load(open(f"{WG}/relic/run_config.json")); tamper["methodology_version"] = 1
json.dump(tamper, open(f"{T}/tamper/run_config_v1.json", "w"))
rj = subprocess.run([jl, "--startup-file=no", f"--project={REPO}/environment/julia_analysis", f"{REPO}/engine/relic/compute_relic_abundance.jl",
                     f"{T}/tamper/run_config_v1.json", f"{WG}/relic/a1/canonical_modes.csv", f"{T}/tamper/out1", "abundance"], capture_output=True, text=True, env=env)
record("F3 the engine refuses per-scenario w_R under methodology_version 1 (no silent inheritance of v2 semantics)", rj.returncode != 0 and "fixed at 0 in methodology v1" in rj.stderr, rj.stderr.strip()[-160:])
c, Wc = one_run("small_w", [{"label": "w0", "T_reh_GeV": 1e4, "w_R": 0.0}, {"label": "w1em12", "T_reh_GeV": 1e4, "w_R": 1e-12}, {"label": "w1em6", "T_reh_GeV": 1e4, "w_R": 1e-6}])
if c == [0, 0]:
    tb = f"{Wc}/relic/tables"; r0, r12, r6 = (list(csv.DictReader(open(f"{tb}/master_modes_{l}.csv"))) for l in ("w0", "w1em12", "w1em6"))
    worst = 0.0
    for a, b, d in zip(r0, r12, r6):
        if a["abundance_state"] == "INCLUDED_PRE_REHEATING":
            HR = float(REF.HR(mp.mpf("1e4"))); lr = math.log(float(a["mass_eV"]) * 1e-9 / (3 * HR))
            for x, w in ((b, 1e-12), (d, 1e-6)):
                exp_ratio = math.exp(2 * w / (1 + w) * lr)
                worst = max(worst, abs(float(x["Omega_h2_at_theta_ref"]) / float(a["Omega_h2_at_theta_ref"]) / exp_ratio - 1))
    record("F4 w_R -> 0 through the engine (w = 1e-12, 1e-6): logarithmic evaluation agrees with the w=0 branch times the exact ratio", worst < 1e-12, f"max rel {worst:.1e}")
else:
    record("F4 small-w run", False, str(c))

# ------------------------------------------------------------------ G: edges of the supported domain through the engine
EDGE = [{"label": "T1e10-w1", "T_reh_GeV": 1e10, "w_R": 1.0}, {"label": "T1e10-w0.999", "T_reh_GeV": 1e10, "w_R": 0.999},
        {"label": "T1e10-wm0.9", "T_reh_GeV": 1e10, "w_R": -0.9}, {"label": "T1e4-wm0.85", "T_reh_GeV": 1e4, "w_R": -0.85},
        {"label": "T5MeV-wm0.8", "T_reh_GeV": 0.005, "w_R": -0.8}, {"label": "T1e4-w1em9", "T_reh_GeV": 1e4, "w_R": 1e-9}]
c, We = one_run("edges", EDGE)
record("G1 domain edges run end to end with all validators: w_R = 1 (kination), 0.999, -0.9, -0.85, -0.8 (5 MeV), 1e-9", c == [0, 0], str(c))
if c == [0, 0]:
    tb = f"{We}/relic/tables"; worst = 0.0; n_ = 0
    for g in EDGE:
        TR = mp.mpf(repr(g["T_reh_GeV"]))
        for r in csv.DictReader(open(f"{tb}/master_modes_{g['label']}.csv")):
            if r["abundance_state"] == "INCLUDED_PRE_REHEATING":
                m = mp.mpf(r["mass_eV"]) * mp.mpf(10) ** -9
                om, _ = omega_by_root_finding(m, mp.mpf(r["f_pert_GeV"]), THETA, TR, repr(g["w_R"]))
                worst = max(worst, rel(r["Omega_h2_at_theta_ref"], om)); n_ += 1
    chk = json.load(open(f"{tb}/output_validation.json"))["all_pass"] and json.load(open(f"{We}/relic/validation/production_verification.json"))["all_pass"]
    record(f"G2 edge-of-domain PRE rows ({n_}) equal the independent root-finding result; validators pass", chk and worst < 1e-11 and n_ > 20, f"max rel {worst:.1e}")

nfail = [r for r in RESULTS if not r["pass"]]
json.dump(RESULTS, open(f"{T}/test_wR_general_results.json", "w"), indent=1)
print(f"\n{len(RESULTS) - len(nfail)} passed, {len(nfail)} failed of {len(RESULTS)}")
print("WR GENERAL VALIDATION PASS" if not nfail else "WR GENERAL VALIDATION FAIL: " + "; ".join(r["check"] for r in nfail))
sys.exit(1 if nfail else 0)
