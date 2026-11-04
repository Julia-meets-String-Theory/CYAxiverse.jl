"""Compose the configuration template of one run from the fixed methodology (v1 or v2), a user run file and optional expectations.

  relic: configs/methodology_v1.json + RUN (scenarios, approval) [+ EXPECTATIONS] -> relic template (the input of
         01_build_mode_table.py and 02_write_run_config.py, same layout as the frozen production configuration)
  theta: configs/theta_methodology_v1.json + RUN (optional "theta_analysis" overrides) [+ EXPECTATIONS] -> theta config

Slots: the methodology files hold "@run" and "@expectation" placeholders at fixed positions (so the composed file keeps
the section order of the frozen configuration). "@run" slots are filled from RUN; "@expectation" slots only from
EXPECTATIONS (expected population counts, frozen checksums of a reference case) and are removed when none is given.
Expectations can fill slots only: they never change a methodology value.

Reheating scenarios: for every (label, T_reh_GeV) the derived quantities H_R, 3H_R, 1+z_R, g_rho(T_R), g_s(T_R), the
pre-reheating constant Omega/(f^2 theta^2) and H_R/H_I,max are evaluated by the 50-digit reference implementation B
(reference_mp.py) and rounded to float64 (the relic driver re-derives 3H_R and 1+z_R and requires agreement to 1e-12).
Hard constraints of configs/reheating_policy_v1.json stop; BBN warnings are written to CHECKS_JSON and printed.
Methodology versions: v1 (configs/methodology_v1.json, run file schema cyaxiverse_relic_run/1) fixes the pre-reheating equation of
state at w_R = 0 and refuses a w_R key. v2 (configs/methodology_v2.json, run file schema cyaxiverse_relic_run/2) accepts a constant
w_R per scenario (default 0.0) in the domain -1 < w_R <= 1 (configs/reheating_policy_v2.json); at w_R = 0 it reproduces v1. The run
file schema must match the methodology file: a v1 run file with a v2 methodology (or the reverse) is refused.
Usage: python 00_compose_run_config.py relic|theta METHODOLOGY RUN OUT [--expectations EXP] [--checks CHECKS_JSON]
"""
import argparse, copy, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, dump_json  # noqa: E402

EXP, RUN = "@expectation", "@run"
ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=("relic", "theta")); ap.add_argument("methodology"); ap.add_argument("run"); ap.add_argument("out")
ap.add_argument("--expectations", default=None); ap.add_argument("--checks", default=None)
a = ap.parse_args()
def load(path, what):
    try:
        return json.load(open(path))
    except (OSError, ValueError) as e:
        sys.exit(f"{what} {path}: not readable JSON ({e})")


M = load(a.methodology, "methodology")
R = load(a.run, "run file")
X = load(a.expectations, "expectations")[a.kind] if a.expectations else {}
METHOD_VERSION = M.get("methodology_version", 1)          # v1 files carry no methodology_version
METHOD_VERSION in (1, 2) or sys.exit(f"{a.methodology}: unsupported methodology_version {METHOD_VERSION!r} (1 or 2)")
RUN_SCHEMA = {1: "cyaxiverse_relic_run/1", 2: "cyaxiverse_relic_run/2"}
if a.kind == "relic" or "reheating" in M:
    isinstance(R, dict) and R.get("schema") == RUN_SCHEMA[METHOD_VERSION] or \
        sys.exit(f"{a.run}: schema must be {RUN_SCHEMA[METHOD_VERSION]} for methodology v{METHOD_VERSION} (got {R.get('schema') if isinstance(R, dict) else R!r}); "
                 "v1 run files fix w_R = 0, v2 run files may set a constant w_R per scenario")
else:                                                    # theta methodology v1 is shared: the theta analysis does not depend on w_R
    isinstance(R, dict) and R.get("schema") in RUN_SCHEMA.values() or sys.exit(f"{a.run}: schema must be one of {sorted(RUN_SCHEMA.values())}")
# every key must be known: a misspelt option would otherwise be ignored silently (w_R is refused in a v1 run file)
RUN_KEYS = {"schema", "description", "scenarios", "production_run_approved", "theta_analysis"}
SCEN_KEYS = {"label", "T_reh_GeV", "description"} | ({"w_R"} if METHOD_VERSION == 2 and a.kind == "relic" else set())
THETA_SECTIONS = {"theta", "dark_matter", "optimisation", "enabled"}
unknown = sorted(set(R) - RUN_KEYS)
unknown and sys.exit(f"run file: unknown key(s) {unknown}; allowed: {sorted(RUN_KEYS)}")
isnum = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def slots(d, pre=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from slots(v, pre + (k,))
    elif d in (EXP, RUN):
        yield pre, d


def setp(d, p, v):
    for k in p[:-1]:
        d = d[k]
    d[p[-1]] = v


def delp(d, p):
    for k in p[:-1]:
        d = d[k]
    del d[p[-1]]


S = dict(slots(M))
cfg = copy.deepcopy(M)


def overlay(tree, pre=()):
    for k, v in tree.items():
        p = pre + (k,)
        if S.get(p) == EXP:
            setp(cfg, p, v)
        elif isinstance(v, dict) and any(s[:len(p)] == p for s in S):
            overlay(v, p)
        else:
            sys.exit(f"expectations may fill @expectation slots only; {'/'.join(p)} is not one")


overlay(X)
checks = {"kind": a.kind, "expectations": os.path.abspath(a.expectations) if a.expectations else None, "warnings": [], "notes": []}

if a.kind == "relic":
    from reference_mp import Ref  # noqa: E402  (50-digit implementation B)
    import mpmath as mp
    pol = json.load(open(os.path.join(REPO, "configs", f"reheating_policy_v{METHOD_VERSION}.json")))
    sc = R.get("scenarios")
    isinstance(sc, list) and sc and all(isinstance(s, dict) for s in sc) or sys.exit("run file: 'scenarios' must be a non-empty list of {label, T_reh_GeV}")
    for s in sc:
        bad = sorted(set(s) - SCEN_KEYS)
        bad and sys.exit(f"scenario {s.get('label')!r}: unknown key(s) {bad}; allowed: {sorted(SCEN_KEYS)} (the cosmology is fixed by the methodology" + ("; w_R is fixed at 0 by methodology v1: use a v2 methodology and run file)" if METHOD_VERSION == 1 else ")"))
    labels = [s.get("label") for s in sc]
    for lab in labels:
        isinstance(lab, str) and re.fullmatch(r"[A-Za-z0-9.+-]{1,32}", lab) or sys.exit(f"scenario label {lab!r}: {pol['label_rule']}")
    len({lab.lower() for lab in labels}) == len(labels) or sys.exit("scenario labels must be unique, also ignoring case (they are file names; "
                                                                        "macOS and Windows file systems do not distinguish case)")
    WD = M["reheating"].get("w_R_domain", {}) if METHOD_VERSION == 2 else {}
    seen_pairs = {}
    B = Ref()
    teq = float(B.T_eq())
    tmin, tmax = float(B.tab.T[0]), float(B.tab.T[-1])
    hi3 = mp.mpf(repr(float(cfg["validity"]["inflation_cut"]["three_H_I_max_GeV"])))
    tsearch = float(cfg["oscillation"]["T_search_max_GeV"])
    out = []
    for s in sc:
        t = s.get("T_reh_GeV")
        isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) and t > 0 or sys.exit(f"{s['label']}: T_reh_GeV must be a positive number")
        t = float(t)
        w = 0.0
        if METHOD_VERSION == 2:
            w = s.get("w_R", M["reheating"]["w_R_default"])
            isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(w) or \
                sys.exit(f"scenario {s['label']}: w_R must be a finite number, got {w!r}")
            -1 < w <= 1 or sys.exit(f"scenario {s['label']}: w_R = {w} is outside the supported domain -1 < w_R <= 1. "
                                    f"{WD['lower_reason'] if w <= -1 else WD['upper_reason']}")
            w = float(w)
            (t, w) in seen_pairs and sys.exit(f"scenarios {seen_pairs[(t, w)]!r} and {s['label']!r} have the same (T_reh_GeV, w_R) = ({t}, {w}): "
                                              "identical physics under two labels is ambiguous")
            seen_pairs[(t, w)] = s["label"]
            if w < -1 / 3:
                checks["warnings"].append({"scenario": s["label"], "w_R": w, "warning": WD["warn_accelerated_text"]})
            if w < WD["warn_near_de_sitter_below"]:
                checks["warnings"].append({"scenario": s["label"], "w_R": w, "warning": WD["warn_near_de_sitter_text"]})
            if w == 1 / 3:
                checks["notes"].append({"scenario": s["label"], "w_R": w, "note": WD["warn_radiation_like_text"]})
        TR = mp.mpf(repr(t)); HR = B.HR(TR)
        hard = {"T_reh_above_T_eq": t > teq, "T_reh_inside_gstar_table": tmin <= t <= tmax,
                "H_R_below_H_I_max": 3 * HR < hi3, "T_reh_below_search_ceiling": t < tsearch}
        bad = [k for k, ok in hard.items() if not ok]
        if bad:
            sys.exit(f"scenario {s['label']} (T_reh = {t} GeV) violates {bad}: " + "; ".join(pol["hard_constraints"][k] for k in bad)
                     + f" [T_eq = {teq:.6g} GeV, g* table {tmin:.6g}-{tmax:.6g} GeV]")
        if t < pol["warnings"]["bbn_warning_below_GeV"]:
            checks["warnings"].append({"scenario": s["label"], "T_reh_GeV": t, "warning": pol["warnings"]["bbn_warning"]})
        elif t < pol["warnings"]["bbn_edge_note_below_GeV"]:
            checks["notes"].append({"scenario": s["label"], "T_reh_GeV": t, "note": pol["warnings"]["bbn_edge_note"]})
        out.append({"label": s["label"], "T_reh_GeV": t, **({"w_R": w} if METHOD_VERSION == 2 else {}), "H_R_GeV": float(HR), "three_H_R_eV": float(3 * HR * mp.mpf(10) ** 9),
                    "log10_three_H_R_eV": float(mp.log10(3 * HR * mp.mpf(10) ** 9)), "one_plus_z_R": float(B.onepz_R(TR)),
                    "g_rho_T_R": float(B.tab.grho(TR)), "g_s_T_R": float(B.tab.gs(TR)),
                    # v1: the w_R = 0 constant k(T_R) = Omega/(f^2 theta^2) of every PRE mode. v2: the same quantity at m = 3 H_R (continuity
                    # point, where every w_R gives k(T_R)); a PRE mode has Omega/(f^2 theta^2) = k (m/3H_R)^(2 w_R/(1+w_R)).
                    ("Omega_pre_h2_per_fGeV2_theta2" if METHOD_VERSION == 1 else "Omega_pre_h2_per_fGeV2_theta2_at_m_eq_3H_R"): float(B.omega_pre_closed_form(1, 1, TR)),
                    **({"PRE_mass_exponent_2w_over_1pw": 2 * w / (1 + w)} if METHOD_VERSION == 2 else {}),
                    **({} if METHOD_VERSION == 2 else {"Omega_pre_h2_at_f_1e16GeV_theta_pi_over_2": float(B.omega_pre_closed_form(mp.mpf("1e16"), mp.pi / 2, TR))}),
                    "H_R_over_H_I_max": float(HR / (hi3 / 3))})
    setp(cfg, ("reheating", "scenarios"), out)
    appr = R.get("production_run_approved", False)
    isinstance(appr, bool) or sys.exit("production_run_approved must be true or false")
    setp(cfg, ("run", "production_run_approved"), appr)
    checks.update(methodology_version=METHOD_VERSION, T_eq_GeV=teq, gstar_table_range_GeV=[tmin, tmax],
                  scenarios=[{"label": s["label"], "T_reh_GeV": s["T_reh_GeV"], **({"w_R": s["w_R"]} if METHOD_VERSION == 2 else {})} for s in out])
else:
    # theta overrides: only these keys, each with its admissible range (methodology defaults otherwise)
    th = R.get("theta_analysis", {}) or {}
    isinstance(th, dict) or sys.exit("theta_analysis must be an object")
    bad = sorted(set(th) - THETA_SECTIONS)
    bad and sys.exit(f"theta_analysis: unknown section(s) {bad}; allowed: {sorted(THETA_SECTIONS)}")
    allowed = {("theta", "theta_max"), ("dark_matter", "Omega_DM_h2"), ("dark_matter", "TIGHT_DM"),
               ("optimisation", "smoothing_half_widths_dex"), ("optimisation", "primary_smoothing_half_width_dex"),
               ("optimisation", "plateau_fraction"), ("optimisation", "bootstrap_resamples"), ("optimisation", "bootstrap_seed"),
               ("optimisation", "robust_criteria")}
    for sec, d in th.items():
        if sec == "enabled":
            continue
        isinstance(d, dict) or sys.exit(f"theta_analysis.{sec} must be an object")
        for k, v in d.items():
            (sec, k) in allowed or sys.exit(f"theta_analysis.{sec}.{k} is not configurable (allowed: {sorted('.'.join(x) for x in allowed)})")
            if k in cfg[sec] and isinstance(cfg[sec][k], dict) and isinstance(v, dict):
                cfg[sec][k] = {**cfg[sec][k], **v}
            else:
                cfg[sec][k] = v
    T, D, O = cfg["theta"], cfg["dark_matter"], cfg["optimisation"]
    tm = T["theta_max"]
    isnum(tm) and 0 < tm <= 1.0 or sys.exit("theta_max must satisfy 0 < theta_max <= 1 (harmonic treatment; see theta_max_meaning)")
    if tm > 0.5:
        checks["warnings"].append({"theta_max": tm, "warning": "theta_max > 0.5: the harmonic scaling Omega ~ theta^2 deviates from the "
                                   "anharmonic result by more than ~2-3%; results above 0.5 are outside the approved Shoroq domain"})
    if tm != 0.5:
        T["domain"] = f"0 < theta <= {tm}"
    T["theta_min"] is None or sys.exit("theta_min must be null: a lower angle cutoff is not part of the v1 methodology")
    isinstance(D["TIGHT_DM"], list) and len(D["TIGHT_DM"]) == 2 and all(isnum(x) for x in D["TIGHT_DM"]) and isnum(D["Omega_DM_h2"]) or \
        sys.exit("dark_matter: Omega_DM_h2 must be a number and TIGHT_DM a list of two numbers [lo, hi]")
    lo, hi = D["TIGHT_DM"]
    D["Omega_DM_h2"] > 0 and 0 < lo < hi or sys.exit("dark_matter: need Omega_DM_h2 > 0 and 0 < TIGHT_DM[0] < TIGHT_DM[1]")
    hs = O["smoothing_half_widths_dex"]; prim = O.get("primary_smoothing_half_width_dex", 0.1)
    isinstance(hs, list) and all(isnum(h) for h in hs) and isnum(prim) and isnum(O["plateau_fraction"]) or \
        sys.exit("optimisation: smoothing half-widths, primary half-width and plateau_fraction must be numbers")
    hs and all(h > 0 for h in hs) and prim in hs or sys.exit("smoothing half-widths must be positive and contain the primary half-width")
    0 < O["plateau_fraction"] < 1 or sys.exit("plateau_fraction must be in (0, 1)")
    isinstance(O["bootstrap_resamples"], int) and not isinstance(O["bootstrap_resamples"], bool) and O["bootstrap_resamples"] >= 1 and \
        isinstance(O["bootstrap_seed"], int) and not isinstance(O["bootstrap_seed"], bool) or sys.exit("bootstrap_resamples >= 1 and bootstrap_seed must be integers")
    all(0 < e < hi for e in O["diagnostic_tight_lower_edges"]) or sys.exit("TIGHT_DM[1] must exceed the fixed diagnostic lower edges 0.8 and 0.95")
    rc = O.get("robust_criteria", {})
    isinstance(rc, dict) and set(rc) <= {"max_smoothed_max_shift_dex", "max_bootstrap_p16_p84_dex", "max_plateau80_width_dex"} and all(isnum(v) and v > 0 for v in rc.values()) \
        or sys.exit("robust_criteria: positive max_smoothed_max_shift_dex, max_bootstrap_p16_p84_dex, max_plateau80_width_dex only")
    checks["theta_analysis_overrides"] = th

for p, kind in list(slots(cfg)):
    if kind == RUN:
        sys.exit(f"run slot {'/'.join(p)} not filled")
    delp(cfg, p)                               # unfilled expectation slot: no expectation for this run
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
dump_json(cfg, a.out)
if a.checks:
    dump_json(checks, a.checks)
for w in checks["warnings"]:
    print("WARNING:", json.dumps(w))
for n in checks["notes"]:
    print("NOTE:", json.dumps(n))
print("wrote", a.out)
