"""Exact small-angle theta machinery (pure functions; no I/O).

R_g(theta) = Om_g (theta/theta_ref)^2 / Omega_DM. Geometry g is in the tight band [x_lo, x_hi] (closed) iff
theta_xlo,g <= theta <= theta_xhi,g with theta_x,g = theta_ref sqrt(x Omega_DM / Om_g). N(theta) is therefore a
piecewise-constant count of closed intervals; its maximum over [lo, hi] is found exactly by a sweep over the interval
endpoints (starts counted before ends at equal theta, so equal endpoints are handled as closed).
"""
import numpy as np


def theta_at(om, x, odm, th_ref):
    """theta at which R = x (analytic; +inf if om == 0)."""
    om = np.asarray(om, float)
    with np.errstate(divide="ignore"):
        return th_ref * np.sqrt(x * odm / om)


def exact_profile(a, b, lo, hi):
    """Closed intervals [a_g, b_g] clipped to [lo, hi]. Returns event points P, the count at each point, and the count on
    each open segment (P_i, P_{i+1})."""
    a2, b2 = np.maximum(a, lo), np.minimum(b, hi)
    keep = a2 <= b2
    S, E = np.sort(a2[keep]), np.sort(b2[keep])
    P = np.unique(np.concatenate([S, E, [lo, hi]]))
    vpt = np.searchsorted(S, P, "right") - np.searchsorted(E, P, "left")            # starts <= p, minus ends < p
    vseg = np.searchsorted(S, P[:-1], "right") - np.searchsorted(E, P[:-1], "right")  # starts <= p_i, minus ends <= p_i
    return P, vpt, vseg


def maximal_set(P, vpt, vseg):
    """Maximum of the step function and the (closed) set where it is attained, as merged intervals."""
    M = int(max(vpt.max(), vseg.max() if len(vseg) else vpt.max()))
    ivs, cur = [], None
    # walk the atoms in order: point P_0, segment (P_0,P_1), point P_1, ...; consecutive maximal atoms form one closed interval
    for i in range(len(P)):
        atoms = [(vpt[i], P[i], P[i])] + ([(vseg[i], P[i], P[i + 1])] if i < len(vseg) else [])
        for val, l, r in atoms:
            if val == M:
                cur = [float(l), float(r)] if cur is None else [cur[0], float(r)]
            elif cur is not None:
                ivs.append(cur); cur = None
    if cur is not None:
        ivs.append(cur)
    return M, ivs


def count_at(theta, a, b):
    """N(theta) for an array of theta by direct closed-interval membership (vectorised, chunked)."""
    theta = np.atleast_1d(theta); out = np.empty(len(theta), int)
    for i in range(0, len(theta), 2000):
        t = theta[i:i + 2000, None]
        out[i:i + 2000] = ((a[None, :] <= t) & (t <= b[None, :])).sum(1)
    return out


def classify(theta, om, odm, th_ref, lo=0.9, hi=1.0):
    """Exact classification at one theta via the analytic thresholds: 'TIGHT_DM', 'ALLOWED_UNDERABUNDANT', 'OVERABUNDANT'."""
    t_lo, t_hi = theta_at(om, lo, odm, th_ref), theta_at(om, hi, odm, th_ref)
    return np.where(theta > t_hi, "OVERABUNDANT", np.where(theta >= t_lo, "TIGHT_DM", "ALLOWED_UNDERABUNDANT"))


def evaluate(P, vpt, vseg, th):
    """Value of the exact step function at arbitrary theta inside [P_0, P_-1]."""
    th = np.atleast_1d(th); i = np.clip(np.searchsorted(P, th, "right") - 1, 0, len(P) - 1)
    return np.where(P[i] == th, vpt[i], vseg[np.clip(i, 0, len(vseg) - 1)])


def profile(om, x_lo, odm, th_ref, lo, hi, x_hi=1.0):
    a, b = theta_at(om, x_lo, odm, th_ref), theta_at(om, x_hi, odm, th_ref)
    P, vpt, vseg = exact_profile(a, b, lo, hi); M, ivs = maximal_set(P, vpt, vseg)
    return {"N_max": M, "intervals": ivs, "P": P, "vpt": vpt, "vseg": vseg, "a": a, "b": b}


def status_for(om, x_lo, odm, th_ref, lo, hi, A=None, hi_unconstrained=np.pi):
    """The single boundary rule: BOUNDARY_LIMITED if the maximising set touches hi, or if N is still rising into hi
    (positive slope on [hi/2, hi] and larger mean there than on [hi/4, hi/2]) while the unconstrained diagnostic maximum
    (harmonic formula continued to hi_unconstrained) lies above hi."""
    A = A if A is not None else profile(om, x_lo, odm, th_ref, lo, hi)
    U = profile(om, x_lo, odm, th_ref, lo, hi_unconstrained)
    tt = np.geomspace(hi / 2, hi, 401); Nt = evaluate(A["P"], A["vpt"], A["vseg"], tt)
    slope = float(np.polyfit(np.log10(tt), Nt, 1)[0])
    m_hi = float(Nt.mean()); m_lo = float(evaluate(A["P"], A["vpt"], A["vseg"], np.geomspace(hi / 4, hi / 2, 401)).mean())
    touches_upper = any(iv[1] == hi for iv in A["intervals"]); touches_lower = any(iv[0] == lo for iv in A["intervals"])
    unc_beyond = U["N_max"] > A["N_max"] and min(iv[0] for iv in U["intervals"]) > hi
    rising = slope > 0 and m_hi > m_lo
    bl = touches_upper or (rising and unc_beyond)
    status = "BOUNDARY_LIMITED" if bl else ("LOWER_BOUNDARY_LIMITED" if touches_lower else "INTERIOR")
    return status, {"touches_upper_boundary": touches_upper, "touches_lower_boundary": touches_lower, "N_tight_slope_per_dex_top_octave": slope,
                    "mean_N_tight_second_octave": m_lo, "mean_N_tight_top_octave": m_hi, "rising_into_upper_boundary": rising,
                    "unconstrained_diagnostic": {"upper": hi_unconstrained, "N_max": U["N_max"], "intervals": U["intervals"], "maximum_lies_above_domain": bool(unc_beyond)}}
