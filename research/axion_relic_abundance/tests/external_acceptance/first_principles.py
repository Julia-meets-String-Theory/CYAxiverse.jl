"""First-principles recomputation of a relic run (independent of engine/: imports nothing from it; mpmath, 30 digits).

Inputs as stated in METHODOLOGY.md: H0 = 1.491e-33 eV, omega_m = 0.315*0.674^2, T_CMB = 2.7255 K, M_pl = 2.435e18 GeV,
g*(T) of data/standardmodel2018.dat (log-log interpolation, clamped outside the table), flat closure.
  POST:  rho_osc = m^2 f^2 theta^2 / 2 at 3 H(T_osc) = m, n/s conserved afterwards.
  PRE (w_R = 0):  a_osc/a_R = (3 H_R / m)^(2/3), H_R from rho_r(T_R) = 3 H_R^2 M_pl^2, rho ~ a^-3 afterwards.
  Geometry:  Omega_geometry = sum over included modes; with no included mode: 0 if every mode was evaluated (EVALUATED_ZERO),
             NA if a mode lies above 3 H_I,max (not evaluated).
Modes within 1e-12 dex of a class threshold are reported as ties (float rounding decides the side) and not counted as
mismatches. Usage: python first_principles.py RELIC_DIR -> JSON on stdout, exit 1 on any disagreement beyond tolerance
(POST/LATE 1e-10, PRE 1e-12, geometry sums 1e-10: the package's own tolerances).
"""
import bisect, csv, json, sys
import mpmath as mp

mp.mp.dps = 30
RELIC = sys.argv[1]
cfg = json.load(open(f"{RELIC}/run_config.json"))
rows = [list(map(mp.mpf, l.split()[:5])) for l in open(cfg["cosmology"]["gstar_table"]) if l.strip() and not l.startswith("#")]
rows.sort(key=lambda r: r[0]); TT = [r[0] for r in rows]; GR = [r[1] for r in rows]; GS = [r[3] for r in rows]


def interp(T, y):
    if T <= TT[0]:
        return y[0]
    if T >= TT[-1]:
        return y[-1]
    k = bisect.bisect_right(TT, T) - 1; w = mp.log(T / TT[k]) / mp.log(TT[k + 1] / TT[k])
    return mp.e ** ((1 - w) * mp.log(y[k]) + w * mp.log(y[k + 1]))


hbar = mp.mpf("6.582119569e-25"); Mpl = mp.mpf("2.435e18"); H100 = 100 / mp.mpf("3.0856775814913673e19") * hbar
H0 = mp.mpf("2.13") * mp.mpf("0.7") * mp.mpf("1e-42"); T0 = mp.mpf("2.7255") * mp.mpf("8.617333262e-14")
h = H0 / H100; Om = mp.mpf("0.315") * mp.mpf("0.674") ** 2 / h ** 2
gr0, gs0 = interp(T0, GR), interp(T0, GS)
Or = mp.pi ** 2 / 30 * gr0 * T0 ** 4 / (3 * Mpl ** 2 * H0 ** 2); OL = 1 - Om - Or
H = lambda T: H0 * mp.sqrt(Or * interp(T, GR) / gr0 * (T / T0) ** 4 + Om * interp(T, GS) / gs0 * (T / T0) ** 3 + OL)
s_ratio = lambda T: interp(T, GS) / gs0 * (T / T0) ** 3
rhoc_h2 = 3 * Mpl ** 2 * H100 ** 2


def solveT(fun, lo=T0, hi=mp.mpf("1e18")):
    flo = fun(lo)
    for _ in range(200):
        mid = mp.sqrt(lo * hi); fm = fun(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return mp.sqrt(lo * hi)


Teq = solveT(lambda T: Or * interp(T, GR) / gr0 * (T / T0) - Om * interp(T, GS) / gs0); HEQ = H(Teq)
HR = lambda TR: mp.sqrt(mp.pi ** 2 / 30 * interp(TR, GR) * TR ** 4 / (3 * Mpl ** 2))
TH = mp.pi / 2; HI3 = mp.mpf("1.4e14")
num = lambda v: None if v in ("NA", "") else mp.mpf(v)
res = {"T_eq_eV": float(Teq * 1e9), "scenarios": {}}; ok = True
for s in cfg["reheating"]["scenarios"]:
    lab = s["label"]; TR = mp.mpf(repr(float(s["T_reh_GeV"]))); hr = HR(TR)
    st = {"rows": 0, "ties": [], "cat_mismatch": 0, "state_mismatch": 0, "max_rel_POST": 0.0, "max_rel_LATE": 0.0, "max_rel_PRE": 0.0, "max_rel_Tosc": 0.0}
    gsum = {}; nexc = {}; ninc = {}
    for r in csv.DictReader(open(f"{RELIC}/tables/master_modes_{lab}.csv")):
        st["rows"] += 1
        lm = mp.mpf(r["log10_mass_eV"]); m = mp.mpf(10) ** (lm - 9); f = mp.mpf(10) ** mp.mpf(r["log10_f_pert_GeV"])
        thr = {"3H0": 3 * H0, "3H(T_eq)": 3 * HEQ, "3H_R": 3 * hr, "3H_I,max": HI3}
        tie = [k for k, v in thr.items() if abs(mp.log10(m / v)) < mp.mpf("1e-12")]
        cat = "FROZEN" if m / 3 <= H0 else "PRE_REHEATING" if m > 3 * hr else "LATE" if m / 3 <= HEQ else "POST_REHEATING"
        valid = not (m > HI3)
        if tie:
            st["ties"].append([r["geometry_id"], r["mode_index"], tie, r["category"], r["abundance_state"]])
            cat, valid = r["category"], r["inflation_valid"] == "true"
        if cat != r["category"]:
            st["cat_mismatch"] += 1; continue
        state = ("NOT_INCLUDED_FROZEN" if cat == "FROZEN" else "INFLATIONARY_VALIDITY_EXCLUDED" if not valid else
                 "NOT_INCLUDED_LATE" if cat == "LATE" else "INCLUDED_POST_REHEATING" if cat == "POST_REHEATING" else "INCLUDED_PRE_REHEATING")
        if state != r["abundance_state"]:
            st["state_mismatch"] += 1; continue
        gsum.setdefault(r["geometry_id"], mp.mpf(0))
        if state == "INFLATIONARY_VALIDITY_EXCLUDED":
            nexc[r["geometry_id"]] = nexc.get(r["geometry_id"], 0) + 1
        if state in ("INCLUDED_POST_REHEATING", "NOT_INCLUDED_LATE"):
            To = solveT(lambda T: 3 * H(T) - m); om = m ** 2 * f ** 2 * TH ** 2 / 2 / s_ratio(To) / rhoc_h2
            key, k2 = ("Omega_h2_at_theta_ref", "max_rel_POST") if cat == "POST_REHEATING" else ("Omega_h2_late_census_only_at_theta_ref", "max_rel_LATE")
            st[k2] = max(st[k2], float(abs(num(r[key]) / om - 1))); st["max_rel_Tosc"] = max(st["max_rel_Tosc"], float(abs(num(r["T_osc_GeV"]) / To - 1)))
            if cat == "POST_REHEATING":
                gsum[r["geometry_id"]] += om; ninc[r["geometry_id"]] = ninc.get(r["geometry_id"], 0) + 1
        elif state == "INCLUDED_PRE_REHEATING":
            aosc = (1 / s_ratio(TR) ** (mp.mpf(1) / 3)) * (3 * hr / m) ** (mp.mpf(2) / 3)
            om = m ** 2 * f ** 2 * TH ** 2 / 2 * aosc ** 3 / rhoc_h2
            st["max_rel_PRE"] = max(st["max_rel_PRE"], float(abs(num(r["Omega_h2_at_theta_ref"]) / om - 1))); gsum[r["geometry_id"]] += om
            ninc[r["geometry_id"]] = ninc.get(r["geometry_id"], 0) + 1
        elif r["Omega_h2_at_theta_ref"] != "NA":
            st["state_mismatch"] += 1
    gmax = 0.0; gz = 0; gna = 0; gbad = []
    for r in csv.DictReader(open(f"{RELIC}/tables/geometry_summary_{lab}.csv")):
        gid = r["geometry_id"]; exp = gsum.get(gid, mp.mpf(0)); got = num(r["Omega_geometry_h2_at_theta_ref"])
        if got is None:                                  # NA only for: nothing included and a mode not evaluated
            gna += 1
            if ninc.get(gid, 0) or not nexc.get(gid, 0):
                gbad.append(gid)
            continue
        if exp == 0:
            gz += 1
            if got != 0 or nexc.get(gid, 0) and not ninc.get(gid, 0):
                gbad.append(gid)
        else:
            gmax = max(gmax, float(abs(got / exp - 1)))
    st.update(geometries_zero_Omega=gz, geometries_Omega_NA=gna, geometry_max_rel=gmax, geometry_bad=gbad)
    good = (st["cat_mismatch"] == 0 and st["state_mismatch"] == 0 and st["max_rel_POST"] <= 1e-10 and st["max_rel_LATE"] <= 1e-10 and
            st["max_rel_Tosc"] <= 1e-10 and st["max_rel_PRE"] <= 1e-12 and gmax <= 1e-10 and not gbad)
    st["pass"] = good; ok &= good; res["scenarios"][lab] = st
res["all_pass"] = ok
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
