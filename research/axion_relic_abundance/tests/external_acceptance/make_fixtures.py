"""Build the external-acceptance fixtures in OUT from the two shipped genuine spectra (data/, see data/provenance.json).

SYNTHETIC fixtures are copies of the genuine h11 = 15 file with the spectrum/physical arrays rewritten; they are labelled
SYNTHETIC in their evidence file and test the engine, never physics claims about a geometry. One fixture per case:
  zero_mode            no physical mode above H0 (validated empty spectrum)
  int_a, int_b         every log10 m and log10 f_pert integral (pandas would infer integer columns)
  tiny_a, tiny_b       Omega far below Omega_DM for 0 < theta <= theta_max (N_TIGHT maximum = 0)
  all_late             only a LATE mode (no CDM-eligible mode, Omega = 0 by classification)
  all_inflation        only a mode above 3 H_I,max (INFLATION_INVALID_ALL: abundance not evaluated, NA)
  partial              a valid POST mode and a mode above 3 H_I,max (PARTIAL_VALIDITY, partial sum > 0)
  partial_late         a LATE mode and a mode above 3 H_I,max (PARTIAL_VALIDITY, partial sum 0)
  boundary             class thresholds: H0, 3H0 (+1 ulp), 3H(T_eq) (+-1 ulp), 3H_R(250 GeV) (+-1 ulp), 3H_I,max, f = 1e30 GeV
  bad_*                one contract violation each (must be refused with a named reason)
Usage: python make_fixtures.py OUT_DIR
"""
import hashlib, json, math, os, shutil, sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1])
os.path.exists(OUT) and sys.exit(f"{OUT} exists")
BASE = f"{HERE}/data/h11_015/np_0000001/cy_0000001/cyax.h5"
GENUINE = [f"{HERE}/data/h11_009/np_0000001/cy_0000001/cyax.h5", BASE]
THR = math.log10(2.13 * 0.7e-33)
L3H0, L3HEQ, L3HR250 = -32.34940110182734, -27.164514669232677, -3.584540753710086
LINF = math.log10(1.4e14) + 9
up = lambda x: float(np.nextafter(x, np.inf)); dn = lambda x: float(np.nextafter(x, -np.inf))
cases = {}


def write(name, np_idx, m, fp, sign=None, ls=None, edit=None):
    d = f"{OUT}/spectra/{name}/h11_015/np_{np_idx:07d}/cy_0000001"; os.makedirs(d); p = f"{d}/cyax.h5"
    shutil.copy(BASE, p)
    m = np.asarray(m, float); fp = np.asarray(fp, float); n = len(m)
    with h5py.File(p, "r+") as f:
        g = f["spectrum/physical"]
        for k in ("m", "fpert_log10", "lambda_self_log10", "lambda_self_sign", "mode_indices"):
            del g[k]
        g["m"] = m; g["fpert_log10"] = fp
        g["lambda_self_log10"] = 2 * (m - 9 - fp) if ls is None else np.asarray(ls, float)
        g["lambda_self_sign"] = np.ones(n, np.int64) if sign is None else np.asarray(sign, np.int64)
        g["mode_indices"] = np.arange(15 - n, 15, dtype=np.int64)
        g["metadata/polytope"][()] = np_idx
        if edit:
            edit(f)
    cases[name] = p
    return p


write("zero_mode", 501, [], [])
write("int_a", 502, [-20.0, -10.0, -5.0], [17.0, 16.0, 15.0])
write("int_b", 503, [-12.0, -3.0], [16.0, 14.0])
write("tiny_a", 504, [-10.1], [8.3])
write("tiny_b", 505, [-11.9], [9.8])
write("all_late", 506, [-30.3], [17.2])
write("all_inflation", 507, [24.2], [12.1])
write("partial", 509, [-12.1, 24.3], [16.2, 12.1])
write("partial_late", 510, [-30.3, 24.4], [17.2, 12.1])
write("boundary", 508, [THR, L3H0, up(L3H0), -30.0, dn(L3HEQ), L3HEQ, up(L3HEQ), -22.0, dn(L3HR250), L3HR250, up(L3HR250), 0.5, 23.0, LINF, 24.0],
      [17.0, 17.0, 17.0, 16.0, 17.0, 17.0, 17.0, 17.0, 12.0, 12.0, 12.0, 11.0, 30.0, 10.0, 10.0])
ok2 = ([-20.3, -10.7], [17.1, 16.2])
write("bad_missing_fpert", 520, *ok2, edit=lambda f: f["spectrum/physical"].__delitem__("fpert_log10"))
write("bad_sign_zero", 521, *ok2, sign=[0, 1])
write("bad_m_inf", 522, [float("inf"), -10.7], [17.1, 16.2])
write("bad_m_string", 523, *ok2, edit=lambda f: (f["spectrum/physical"].__delitem__("m"), f["spectrum/physical"].__setitem__("m", np.array([b"-20.3", b"-10.7"]))))
write("bad_cy_volume_zero", 524, *ok2, edit=lambda f: f["cytools/geometric/CY_volume"].write_direct(np.array(0.0)))
p = write("bad_truncated_h5", 525, *ok2); b = open(p, "rb").read(); open(p, "wb").write(b[: len(b) // 2])
p = write("bad_not_hdf5", 526, *ok2); open(p, "w").write("not an HDF5 file\n")
ev = f"{OUT}/SYNTHETIC_NOTICE.txt"; open(ev, "w").write("SYNTHETIC acceptance-test fixture: arrays rewritten, not a physical vacuum\n")
evsha = hashlib.sha256(open(ev, "rb").read()).hexdigest()


def spectra_list(name, paths, policy="refuse", verify=()):
    sp = []
    for q in paths:
        if q in verify:
            sp.append({"path": q, "vacuum_validation": {"state": "VERIFY"}})
        else:
            sp.append({"path": q, "vacuum_validation": {"state": "DOCUMENTED_EXTERNAL", "method": "SYNTHETIC fixture or genuine producer run",
                                                        "reference": "tests/external_acceptance", "evidence": [{"path": ev, "sha256": evsha}]}})
    json.dump({"schema": "cyaxiverse_relic_spectra/1", "vacuum_policy": {"on_unusable": policy}, "spectra": sp}, open(f"{OUT}/list_{name}.json", "w"), indent=1)


G9 = GENUINE[0]
spectra_list("genuine", GENUINE)
spectra_list("genuine_verify", GENUINE, verify=set(GENUINE))
spectra_list("zero_mixed", [G9, cases["zero_mode"]])
spectra_list("zero_only", [cases["zero_mode"]])
spectra_list("integer", [cases["int_a"], cases["int_b"]])
spectra_list("nomatch", [cases["tiny_a"], cases["tiny_b"]])
spectra_list("noeligible", [G9, cases["all_late"], cases["zero_mode"]])
spectra_list("populations", [G9, cases["zero_mode"], cases["all_late"], cases["all_inflation"], cases["partial"], cases["partial_late"]])
spectra_list("all_invalid", [cases["all_inflation"]])
spectra_list("boundary", [G9, cases["boundary"]])
for k in cases:
    if k.startswith("bad_"):
        spectra_list(k, [cases[k]])
json.dump({"schema": "cyaxiverse_relic_run/1", "description": "external acceptance run (non-Shoroq temperatures)",
           "scenarios": [{"label": "T20MeV", "T_reh_GeV": 0.02}, {"label": "T250GeV", "T_reh_GeV": 250}, {"label": "T1e9GeV", "T_reh_GeV": 1e9}],
           "theta_analysis": {"theta": {"theta_max": 0.4}, "optimisation": {"bootstrap_resamples": 50, "bootstrap_seed": 7}}},
          open(f"{OUT}/run.json", "w"), indent=1)
json.dump(cases, open(f"{OUT}/cases.json", "w"), indent=1)
print(f"{len(cases)} fixtures in {OUT}")
