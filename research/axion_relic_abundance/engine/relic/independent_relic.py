"""Independent analytic re-implementation of the misalignment relic calculation (no code shared with the Julia scripts).

Physics (written from first principles, physical-density form rather than the Julia code's Omega-fraction form):
  H(T)^2 = [rho_r(T) + rho_m(T) + rho_L] / (3 M_pl^2),           M_pl = reduced Planck mass 2.435e18 GeV
  rho_r(T) = (pi^2/30) g_rho(T) T^4
  rho_m(T) = rho_m0 * [g_s(T)/g_s0] (T/T0)^3                     (a0/a = (g_s/g_s0)^{1/3} T/T0)
  oscillation:  m = 3 H(T_osc)  (brentq in log T);  frozen if m <= 3 H(T0)
  rho_a(T_osc) = (1/2) m^2 f^2 theta^2;  n_a/s conserved after T_osc  ->  rho_a,0 = rho_a(T_osc) s0/s(T_osc),  s = (2 pi^2/45) g_s T^3
  Omega_a h^2 = rho_a,0 / (3 M_pl^2 H_100^2)
Two cosmologies:
  "legacy"   : the parameters of compute_tosc_abundance_hfull.jl (h = 0.674, Omega_r0 = 9.2e-5, Omega_m0 = 0.315, flat)
  "vmm"      : H0 = 1.491e-33 eV (CYAxiverse), rho_r from T_CMB and g_rho, omega_m = 0.315*0.674^2 held fixed, flat closure
  "naive"    : h = 0.698976 changed ALONE (Omega_r0 = 9.2e-5, Omega_m0 = 0.315 kept) -- the inconsistent single-number patch
Optional early-matter-dominated (EMD) reheating estimate for T_osc > T_reh (instantaneous decay at H = H(T_reh)):
  rho_a(a_reh) = (1/2) m^2 f^2 theta^2 (H_reh/H_osc)^2 = (9/2) f^2 theta^2 H_reh^2  ->  rho_a,0 = (9/2) f^2 theta^2 H_reh^2 s0/s(T_reh)
"""
import math
import os
import numpy as np
from scipy.optimize import brentq

HBAR_GEV_S = 6.582119569e-25
MPL = 2.435e18                                   # reduced Planck mass, GeV
H100 = (100.0 / 3.0856775814913673e19) * HBAR_GEV_S
T0 = 2.3486541805581e-13                         # T_CMB, GeV (2.7255 K)
RHO_C_OVER_H2 = 3 * MPL**2 * H100**2
GTABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "standardmodel2018.dat")  # Saikawa-Shirai 2018


class GTab:
    def __init__(self, path=GTABLE):
        raw = np.loadtxt(path, comments="#")
        o = np.argsort(raw[:, 0])
        self.lT, self.lgr, self.lgs = np.log(raw[o, 0]), np.log(raw[o, 1]), np.log(raw[o, 3])

    def grho(self, T):
        return float(np.exp(np.interp(math.log(T), self.lT, self.lgr)))   # clamps outside the table, like the legacy loginterp

    def gs(self, T):
        return float(np.exp(np.interp(math.log(T), self.lT, self.lgs)))


class Cosmo:
    def __init__(self, convention, tab=None):
        self.tab = tab or GTab(); t = self.tab; self.convention = convention
        self.gr0, self.gs0 = t.grho(T0), t.gs(T0)
        if convention == "legacy":
            self.H0 = 0.674 * H100; om, orr = 0.315, 9.2e-5
            self.rho_m0 = om * 3 * MPL**2 * self.H0**2
            self.rho_r0 = orr * 3 * MPL**2 * self.H0**2              # legacy normalizes radiation by Omega_r0
        elif convention == "naive":
            self.H0 = 2.13 * 0.7e-33 * 1e-9; om, orr = 0.315, 9.2e-5
            self.rho_m0 = om * 3 * MPL**2 * self.H0**2
            self.rho_r0 = orr * 3 * MPL**2 * self.H0**2
        elif convention == "vmm":
            self.H0 = 2.13 * 0.7e-33 * 1e-9                           # CYAxiverse constants()["Hubble"]
            self.rho_m0 = 0.315 * 0.674**2 * RHO_C_OVER_H2            # omega_m held fixed
            self.rho_r0 = (math.pi**2 / 30) * self.gr0 * T0**4
        else:
            raise ValueError(convention)
        self.rho_L = 3 * MPL**2 * self.H0**2 - self.rho_m0 - self.rho_r0   # flat closure: H(T0) = H0 exactly

    def rho_r(self, T):
        return self.rho_r0 * (self.tab.grho(T) / self.gr0) * (T / T0) ** 4

    def rho_m(self, T):
        return self.rho_m0 * (self.tab.gs(T) / self.gs0) * (T / T0) ** 3

    def H(self, T):
        return math.sqrt((self.rho_r(T) + self.rho_m(T) + self.rho_L) / (3 * MPL**2))

    def s(self, T):
        return (2 * math.pi**2 / 45) * self.tab.gs(T) * T**3

    def T_eq(self):
        return math.exp(brentq(lambda x: math.log(self.rho_r(math.exp(x)) / self.rho_m(math.exp(x))), math.log(T0), math.log(1e18), xtol=1e-14, rtol=1e-15))

    def T_osc(self, m_gev, Tmax=1e18):
        if m_gev / 3 <= self.H(T0):
            return float("nan")                                       # frozen
        if self.H(Tmax) < m_gev / 3:
            return float("inf")                                       # beyond the search range
        g = lambda x: math.log(self.H(math.exp(x)) / (m_gev / 3))
        return math.exp(brentq(g, math.log(T0), math.log(Tmax), xtol=1e-14, rtol=1e-15))

    def omega_h2(self, m_gev, f_gev, theta, Tosc):
        return 0.5 * m_gev**2 * f_gev**2 * theta**2 * (self.s(T0) / self.s(Tosc)) / RHO_C_OVER_H2

    def omega_h2_emd(self, f_gev, theta, Treh):
        Hreh = self.H(Treh)
        return 4.5 * f_gev**2 * theta**2 * Hreh**2 * (self.s(T0) / self.s(Treh)) / RHO_C_OVER_H2

    def classify(self, Tosc, Treh, Teq):
        if math.isnan(Tosc):
            return "FROZEN"
        if Tosc > Treh:
            return "PRE_REHEATING"
        if Tosc <= Teq:
            return "LATE_OSCILLATING"
        return "CDM_ELIGIBLE"


def rd_closed_form(m_gev, f_gev, theta, g):
    """Pure radiation domination with constant g = g_rho = g_s: closed-form T_osc and Omega h^2 (sanity scaling only)."""
    Tosc = math.sqrt(m_gev * MPL / (3 * math.sqrt(math.pi**2 * g / 90)))
    return Tosc, 0.5 * m_gev**2 * f_gev**2 * theta**2 * (3.9309363 * T0**3 / (g * Tosc**3)) / RHO_C_OVER_H2


# ---------------------------------------------------------------------------------------------------------------------
# Fuzzy Axions and Associated Relics (Sheridan et al., arXiv:2412.12012), Section 2, transcribed equation by equation.
#   (2.1)  g_s(z) T^3(z) = g_s0 T0^3 (1+z)^3                       (z <= z_R)
#   (2.2)  3H^2 M^2 = (rho_r+rho_m+rho_L) Theta(z<z_R) + 3 H_R^2 M^2 ((1+z)/(1+z_R))^{3(1+w_R)} Theta(z>=z_R)
#          with rho_r(T_R) = 3 H_R^2 M^2   (energy conservation at reheating)
#   (2.10) Omega_a h^2 = m^2 f^2 theta^2 / [2 (1+z_a)^3 3 M_H^2 M^2],   3H(z_a) = m_a
#   (2.15) 1+z_a = (1+z_R) (m_a/3H_R)^{2/(3(1+w_R))}               (m_a > 3H_R)
#   (2.16) closed form;  (2.17) eta = g_rho(z_R)^{3/4}/g_s(z_R)
def fuzzy_HR(tab, Treh):
    """(2.2): rho_r(T_R) = 3 H_R^2 M_pl^2, rho_r = (pi^2/30) g_rho(T_R) T_R^4 (radiation ONLY)."""
    return math.sqrt((math.pi**2 / 30) * tab.grho(Treh) * Treh**4 / (3 * MPL**2))


def fuzzy_onepz_R(tab, Treh):
    """(2.1) at z_R."""
    return (Treh / T0) * (tab.gs(Treh) / tab.gs(T0)) ** (1 / 3)


def fuzzy_pre_omega_h2(tab, m_gev, f_gev, theta, Treh, wR=0.0):
    """(2.10) with (2.15): direct evaluation of the paper's pre-reheating prescription."""
    HR = fuzzy_HR(tab, Treh)
    onepz_a = fuzzy_onepz_R(tab, Treh) * (m_gev / (3 * HR)) ** (2 / (3 * (1 + wR)))
    if wR == 0:                                                   # v1 form, unchanged
        return 0.5 * m_gev**2 * f_gev**2 * theta**2 / onepz_a**3 / RHO_C_OVER_H2, onepz_a
    # constant w_R != 0 (methodology v2): (1+z_a)^3 can leave the float64 range while Omega does not, so combine in logarithms (math.exp
    # raises OverflowError, handled by the caller, instead of returning inf/0 silently)
    lnom = math.log(0.5 * m_gev**2 * f_gev**2 * theta**2 / RHO_C_OVER_H2) - 3 * math.log(onepz_a)
    if not (math.isfinite(lnom) and math.isfinite(onepz_a) and -745 < lnom < 709):
        raise OverflowError(f"Omega outside the float64 range (ln Omega = {lnom})")
    return math.exp(lnom), onepz_a


def fuzzy_216(tab, m_gev, f_gev, theta, Treh, wR=0.0):
    """(2.16)-(2.17): the paper's closed-form approximation with its quoted numerical normalization."""
    HR = fuzzy_HR(tab, Treh); eta = tab.grho(Treh) ** 0.75 / tab.gs(Treh)
    return (0.12 * (eta / 0.4) * (theta / 4.4e-3) ** 2 * (3 * HR * 1e9 / 5.7e-10) ** 0.5 * (f_gev / 1e16) ** 2
            * (m_gev / (3 * HR)) ** (2 * wR / (1 + wR)))


def audit_emd_omega_h2(tab, f_gev, theta, Treh, HR=None):
    """The audit's independently derived w_R = 0 result: (9/2) f^2 theta^2 H_R^2 s0/s(T_R) / (rho_c/h^2)."""
    HR = fuzzy_HR(tab, Treh) if HR is None else HR
    s0, sR = tab.gs(T0) * T0**3, tab.gs(Treh) * Treh**3
    return 4.5 * f_gev**2 * theta**2 * HR**2 * (s0 / sR) / RHO_C_OVER_H2
