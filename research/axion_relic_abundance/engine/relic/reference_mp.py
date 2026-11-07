"""Implementation B: direct evaluation of the approved analytic equations in 50-digit arithmetic (mpmath).

Independent of the Julia driver (A) and of the audit's float64 Python implementation (C): it has its own g* reader
(decimal strings -> mpf), its own log-log interpolation and its own bracketing root finder.

Equations (approved methodology, production_run_config.json):
  Friedmann (physical-density form)  3 M^2 H(T)^2 = rho_r(T) + rho_m(T) + rho_L
     rho_r(T) = (pi^2/30) g_rho(T) T^4,   rho_m(T) = omega_m rho_c/h^2 [g_s(T)/g_s0] (T/T0)^3,
     rho_L    = 3 M^2 H0^2 - rho_r(T0) - rho_m(T0)                                   (flat; H(T0) = H0)
  onset         m_a = 3 H(T_osc)                                FROZEN iff m_a <= 3 H0
  post-reheating (T_eq < T_osc <= T_R):  Omega h^2 = (1/2) m^2 f^2 theta^2 [s0/s(T_osc)] / (3 M^2 H100^2)
  pre-reheating  (m_a > 3 H_R), Fuzzy arXiv:2412.12012:
     (2.2)  rho_r(T_R) = 3 H_R^2 M^2                (radiation only)
     (2.1)  1+z_R = (T_R/T0) [g_s(T_R)/g_s0]^(1/3)
     (2.15) 1+z_a = (1+z_R) (m_a/3H_R)^(2/(3(1+w_R)))
     (2.10) Omega h^2 = m^2 f^2 theta^2 / [2 (1+z_a)^3 3 H100^2 M^2]
  closed form for w_R = 0 (algebraically identical):  Omega h^2 = (9/2) f^2 theta^2 H_R^2 [s0/s(T_R)] / (3 M^2 H100^2)
  constant w_R (methodology v2; -1 < w_R <= 1): (2.15) and (2.10) as printed, and the algebraically equivalent relative form
     Omega(w_R) = Omega(0) (m_a/3H_R)^(2 w_R/(1+w_R))      (omega_pre_closed_form_w)
"""
import bisect
import os
import mpmath as mp

mp.mp.dps = 50
GTABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "standardmodel2018.dat")  # Saikawa-Shirai 2018

MPL = mp.mpf("2.435e18")
HBAR = mp.mpf("6.582119569e-25")
MPC_KM = mp.mpf("3.0856775814913673e19")
H100 = 100 / MPC_KM * HBAR
T0 = mp.mpf("2.3486541805581e-13")
H0 = mp.mpf("2.13") * mp.mpf("0.7") * mp.mpf("1e-33") * mp.mpf("1e-9")      # GeV
OMEGA_M = mp.mpf("0.315") * mp.mpf("0.674") ** 2
RHO_C_H2 = 3 * MPL**2 * H100**2
TSEARCH = mp.mpf("1e18")


class MPTab:
    def __init__(self, path=GTABLE):
        rows = []
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            t = line.split()
            rows.append((mp.mpf(t[0]), mp.mpf(t[1]), mp.mpf(t[3])))
        rows.sort(key=lambda r: r[0])
        self.T = [r[0] for r in rows]
        self.lT = [mp.log(r[0]) for r in rows]
        self.lgr = [mp.log(r[1]) for r in rows]
        self.lgs = [mp.log(r[2]) for r in rows]
        self.gr_ends = (rows[0][1], rows[-1][1])
        self.gs_ends = (rows[0][2], rows[-1][2])
        self.Tf = [float(x) for x in self.T]

    def _interp(self, T, ly, ends):
        T = mp.mpf(T)
        if T <= self.T[0]:
            return ends[0]
        if T >= self.T[-1]:
            return ends[1]
        k = bisect.bisect_right(self.Tf, float(T)) - 1
        while self.T[k] > T:          # guard the float pre-search
            k -= 1
        while self.T[k + 1] <= T:
            k += 1
        w = (mp.log(T) - self.lT[k]) / (self.lT[k + 1] - self.lT[k])
        return mp.exp((1 - w) * ly[k] + w * ly[k + 1])

    def grho(self, T):
        return self._interp(T, self.lgr, self.gr_ends)

    def gs(self, T):
        return self._interp(T, self.lgs, self.gs_ends)


class Ref:
    def __init__(self, tab=None):
        self.tab = tab or MPTab()
        self.gr0, self.gs0 = self.tab.grho(T0), self.tab.gs(T0)
        self.rho_m0 = OMEGA_M * RHO_C_H2
        self.rho_r0 = mp.pi**2 / 30 * self.gr0 * T0**4
        self.rho_L = 3 * MPL**2 * H0**2 - self.rho_m0 - self.rho_r0
        self.h = H0 / H100
        self.Omega_m0 = self.rho_m0 / (3 * MPL**2 * H0**2)
        self.Omega_r0 = self.rho_r0 / (3 * MPL**2 * H0**2)
        self.Omega_L0 = self.rho_L / (3 * MPL**2 * H0**2)
        self._teq = None

    def rho_r(self, T):
        return mp.pi**2 / 30 * self.tab.grho(T) * T**4

    def rho_m(self, T):
        return self.rho_m0 * (self.tab.gs(T) / self.gs0) * (T / T0) ** 3

    def H(self, T):
        T = mp.mpf(T)
        return mp.sqrt((self.rho_r(T) + self.rho_m(T) + self.rho_L) / (3 * MPL**2))

    def s(self, T):
        return 2 * mp.pi**2 / 45 * self.tab.gs(T) * T**3

    def onepz(self, T):
        return (T / T0) * (self.tab.gs(T) / self.gs0) ** (mp.mpf(1) / 3)

    @staticmethod
    def _bisect(g, a, b, tol=mp.mpf("1e-40")):
        ga, gb = g(a), g(b)
        if ga == 0:
            return a
        if gb == 0:
            return b
        assert mp.sign(ga) != mp.sign(gb), "no bracket"
        while b - a > tol:
            c = (a + b) / 2
            gc = g(c)
            if mp.sign(gc) == mp.sign(ga):
                a, ga = c, gc
            else:
                b = c
        return (a + b) / 2

    def T_eq(self):
        if self._teq is None:
            x = self._bisect(lambda x: mp.log(self.rho_r(mp.exp(x)) / self.rho_m(mp.exp(x))), mp.log(T0), mp.log(TSEARCH))
            self._teq = mp.exp(x)
        return self._teq

    def T_osc(self, m):
        target = m / 3
        if target <= self.H(T0):
            return None                                    # FROZEN
        assert self.H(TSEARCH) >= target, "onset above the search range"
        x = self._bisect(lambda x: mp.log(self.H(mp.exp(x)) / target), mp.log(T0), mp.log(TSEARCH))
        return mp.exp(x)

    # --- Fuzzy pre-reheating
    def HR(self, TR):
        TR = mp.mpf(TR)
        return mp.sqrt(mp.pi**2 / 30 * self.tab.grho(TR) * TR**4 / (3 * MPL**2))

    def onepz_R(self, TR):
        return self.onepz(mp.mpf(TR))

    def classify(self, m, TR):
        if m / 3 <= self.H(T0):
            return "FROZEN"
        if m > 3 * self.HR(TR):
            return "PRE_REHEATING"
        return "LATE" if m / 3 <= self.H(self.T_eq()) else "POST_REHEATING"

    def omega_post(self, m, f, theta, Tosc):
        return mp.mpf(1) / 2 * m**2 * f**2 * theta**2 * (self.s(T0) / self.s(Tosc)) / RHO_C_H2

    def omega_pre_eq210_215(self, m, f, theta, TR, wR=0):
        HR = self.HR(TR)
        opz_a = self.onepz_R(TR) * (m / (3 * HR)) ** (mp.mpf(2) / (3 * (1 + mp.mpf(wR))))
        return m**2 * f**2 * theta**2 / (2 * opz_a**3 * 3 * H100**2 * MPL**2), opz_a

    def omega_pre_closed_form(self, f, theta, TR):
        TR = mp.mpf(TR)
        return mp.mpf(9) / 2 * f**2 * theta**2 * self.HR(TR) ** 2 * (self.s(T0) / self.s(TR)) / RHO_C_H2

    def omega_pre_closed_form_w(self, m, f, theta, TR, wR):
        """Constant-w_R closed form: the w_R = 0 closed form times (m/3H_R)^(2 w_R/(1+w_R)) (algebraically identical to (2.10)+(2.15))."""
        wR = mp.mpf(wR)
        return self.omega_pre_closed_form(f, theta, TR) * (m / (3 * self.HR(TR))) ** (2 * wR / (1 + wR))

    def evaluate(self, log10_m_eV, log10_f_GeV, theta, TR, three_HImax_GeV, wR=0):
        """Full per-mode evaluation (category, validity, T_osc, 1+z_osc, Omega h^2) for one scenario."""
        m = mp.power(10, mp.mpf(log10_m_eV) - 9)
        f = mp.power(10, mp.mpf(log10_f_GeV))
        theta = mp.mpf(theta)
        cat = self.classify(m, TR)
        valid = not (m > mp.mpf(three_HImax_GeV))
        out = {"category": cat, "inflation_valid": valid, "T_osc_GeV": None, "one_plus_z_osc": None,
               "Omega_h2": None, "Omega_h2_closed_form": None}
        if cat == "FROZEN" or not valid:
            return out
        if cat == "PRE_REHEATING":
            om, opz = self.omega_pre_eq210_215(m, f, theta, TR, wR)
            cf = self.omega_pre_closed_form(f, theta, TR) if wR == 0 else self.omega_pre_closed_form_w(m, f, theta, TR, wR)
            out.update(one_plus_z_osc=opz, Omega_h2=om, Omega_h2_closed_form=cf)
            return out
        T = self.T_osc(m)
        out.update(T_osc_GeV=T, one_plus_z_osc=self.onepz(T), Omega_h2=self.omega_post(m, f, theta, T))
        return out
