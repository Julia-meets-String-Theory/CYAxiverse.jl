"""Sudden-onset systematic of the constant-w_R pre-reheating prescription (methodology v2): exact field evolution vs the m = 3H estimate.
Idealised setup (not a correction factor): constant mass, H = p/t with p = 2/(3(1+w_R)) (a ~ t^p), harmonic potential, small theta_i.
x = m t; theta'' + (3p/x) theta' + theta = 0; late-time comoving number N = x^(3p) <theta'^2/2 + theta^2/2>; sudden estimate N_s = (3p)^(3p) theta_i^2/2.
Prints N_exact / N_sudden for the w_R values of the validation matrix. Needs scipy only.   python tests/ode_onset_check_wR.py
"""
import json
import numpy as np
from scipy.integrate import solve_ivp


def ratio(p, th0=0.01, x1=6000.0):
    rhs = lambda x, y: [y[1], -3 * p / x * y[1] - y[0]]
    sol = solve_ivp(rhs, (1e-4, x1), [th0, 0.0], method="DOP853", rtol=1e-11, atol=1e-15, dense_output=True)
    xs = np.linspace(0.75 * x1, x1, 40001); th, v = sol.sol(xs)
    N = np.mean(xs ** (3 * p) * (0.5 * v ** 2 + 0.5 * th ** 2))
    return float(N / ((3 * p) ** (3 * p) * 0.5 * th0 ** 2))


out = {}
for w in (1 / 3, 0.17, 0.0, -0.25, -1 / 3, -0.42, -0.5):
    p = 2 / (3 * (1 + w)); out[f"{w:+.4f}"] = {"p": p, "exact_over_sudden": ratio(p), "sudden_over_exact": 1 / ratio(p)}
print(json.dumps(out, indent=1))
