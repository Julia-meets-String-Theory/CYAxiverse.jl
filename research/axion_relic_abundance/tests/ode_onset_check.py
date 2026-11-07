"""B2: exact field evolution vs the sudden-onset (m = 3H) estimate, and anharmonic (cosine) corrections.
Dimensionless time x = m t; background a ~ t^p (p = 1/2 radiation, 2/3 matter/modulus), H = p/t.
Comoving number N = a^3 rho/m ~ x^{3p} (theta'^2 + V'') averaged; sudden estimate N_s = x_osc^{3p} * (1/2) theta_i^2 (harmonic)
with 3 H(x_osc) = 1 -> x_osc = 3p.  Prints N_exact/N_sudden for harmonic and cosine potentials."""
import json, numpy as np
from scipy.integrate import solve_ivp

def run(p, th0, cosine):
    def rhs(x, y):
        th, v = y
        force = np.sin(th) if cosine else th
        return [v, -3 * p / x * v - force]
    x0, x1 = 1e-4, 4000.0
    sol = solve_ivp(rhs, (x0, x1), [th0, 0.0], method="DOP853", rtol=1e-11, atol=1e-14, dense_output=True)
    xs = np.linspace(3000, 4000, 20001); th, v = sol.sol(xs)
    pot = (1 - np.cos(th)) if cosine else 0.5 * th ** 2
    N = np.mean(xs ** (3 * p) * (0.5 * v ** 2 + pot))          # late: harmonic, rho/m with m = 1
    Ns = (3 * p) ** (3 * p) * 0.5 * th0 ** 2
    return N / Ns

out = {}
for name, p in (("radiation_POST", 0.5), ("modulus_PRE", 2.0 / 3.0)):
    r = {"sudden_correction_harmonic_theta0.01": run(p, 0.01, False)}
    for th in (0.01, 0.1, 0.5, 1.0, 2.0, 3.0):
        r[f"cosine_over_harmonic_theta{th}"] = run(p, th, True) / run(p, th, False)
    # theta^2 scaling check of the exact harmonic solution
    r["harmonic_N(0.5)/N(0.05)/100"] = run(p, 0.5, False) / run(p, 0.05, False)
    out[name] = r
# standard fitting function of the anharmonic factor (Lyth 1992; Visinelli & Gondolo 2009)
out["fit_ln(e/(1-th^2/pi^2))^(7/6)"] = {str(t): float(np.log(np.e / (1 - t ** 2 / np.pi ** 2)) ** (7 / 6)) for t in (0.5, 1.0, 2.0, 3.0)}
print(json.dumps(out, indent=1))
