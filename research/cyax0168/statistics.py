#!/usr/bin/env python3
"""CYAX-0168 statistics module.

Implements, per ``specs/0168-structured-graph-materialization/spec.md``
("Proposed thresholds and deterministic classifier" and "Frozen calibration
simulation"):

* the nearest-rank percentile estimator (p50/p95/q99/etc.);
* a deterministic PRF and index/uniform-draw reduction reused for both paired
  BCa resampling and the calibration simulation;
* the paired BCa bootstrap interval (bias-correction ``z0``, jackknife
  acceleration ``a``, adjusted probabilities, quantile-type-7 interpolation);
* 50-significant-digit ``Decimal`` arithmetic helpers (ln/exp/sqrt via the
  ``decimal`` module's built-ins, plus hand-rolled Taylor-series sin/cos/erf,
  since ``decimal`` has no native trigonometric or error function); and
* the empirical/lognormal/two-component-tail truth-transform machinery used
  to construct classifier-truth simulation surfaces.

Honesty note: ``decimal`` documents ln/exp/sqrt as correctly rounded; the
hand-rolled Taylor-series sin/cos/erf here are high-precision best-effort
approximations, not formally proven correctly-rounded to 50 significant
digits. This is disclosed rather than silently claimed, per this worker's
evidence-honesty obligation. Per the CYAX-0168 G1 hard guard, this module is
validated only against synthetic, hand-computable known-truth fixtures in
``test_statistics.py``; it performs no T0-T4/decision-fixture access.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from decimal import Context, Decimal, localcontext
from typing import Any, Callable, Mapping, Sequence


class StatisticsError(ValueError):
    """Raised for malformed inputs or an undefined statistic (Inconclusive)."""


# ---------------------------------------------------------------------------
# Canonical framing (self-contained copy; see evaluator.py's module docstring
# for why this worker does not introduce a shared canonical module).
# ---------------------------------------------------------------------------


def _frame_null() -> bytes:
    return b"N" + (0).to_bytes(8, "big")


def _frame_bytes(value: bytes) -> bytes:
    return b"X" + len(value).to_bytes(8, "big") + value


def _frame_str(value: str) -> bytes:
    data = unicodedata.normalize("NFC", value).encode("utf-8")
    return b"S" + len(data).to_bytes(8, "big") + data


def _frame_int(value: int) -> bytes:
    data = str(value).encode("ascii")
    return b"I" + len(data).to_bytes(8, "big") + data


def _frame_bool(value: bool) -> bytes:
    return b"B" + (1).to_bytes(8, "big") + (b"\x01" if value else b"\x00")


def frame(value: Any) -> bytes:
    if value is None:
        return _frame_null()
    if isinstance(value, bool):
        return _frame_bool(value)
    if isinstance(value, int):
        return _frame_int(value)
    if isinstance(value, str):
        return _frame_str(value)
    if isinstance(value, bytes):
        return _frame_bytes(value)
    if isinstance(value, (list, tuple)):
        out = bytearray(b"A" + len(value).to_bytes(8, "big"))
        for item in value:
            out += frame(item)
        return bytes(out)
    if isinstance(value, dict):
        pairs = sorted(value.items(), key=lambda kv: unicodedata.normalize("NFC", kv[0]).encode("utf-8"))
        out = bytearray(b"O" + len(pairs).to_bytes(8, "big"))
        for key, val in pairs:
            out += _frame_str(key) + frame(val)
        return bytes(out)
    raise StatisticsError(f"unsupported type for canonical frame: {type(value)!r}")


# ---------------------------------------------------------------------------
# Nearest-rank percentile (spec: "The exact p95 estimator is nearest rank")
# ---------------------------------------------------------------------------


def nearest_rank_percentile(values: Sequence[float], p: float) -> float:
    """``x_(ceil(p*n))`` over sorted ``values`` (one-based order statistic)."""

    if not values:
        raise StatisticsError("empty sample: nearest-rank percentile is undefined")
    if not math.isfinite(float(p)) or not 0 < float(p) <= 1:
        raise StatisticsError("nearest-rank percentile p must satisfy 0 < p <= 1")
    if any(not math.isfinite(float(value)) for value in values):
        raise StatisticsError("nearest-rank percentile requires finite values")
    ordered = sorted(values)
    n = len(ordered)
    rank = math.ceil(p * n)
    rank = max(1, min(n, rank))
    return ordered[rank - 1]


def p50(values: Sequence[float]) -> float:
    return nearest_rank_percentile(values, 0.50)


def p95(values: Sequence[float]) -> float:
    return nearest_rank_percentile(values, 0.95)


def q99(values: Sequence[float]) -> float:
    return nearest_rank_percentile(values, 0.99)


# ---------------------------------------------------------------------------
# Deterministic PRF and reduction-to-index (spec: generator 2.2's algorithm,
# reused here per "Frozen calibration simulation": "Reduction to an index
# uses the generator 2.2 rejection algorithm.")
# ---------------------------------------------------------------------------

RATIFICATION_DOMAIN = "cyax-0168-ratification-v1"
SIMULATION_MODES = ("empirical", "lognormal", "two_component_tail")


def _prf_digest(seed_parts: Sequence[Any]) -> bytes:
    return hashlib.sha256(frame(list(seed_parts))).digest()


def draw_index(n: int, seed_parts_prefix: Sequence[Any], counter_start: int = 0) -> tuple[int, int]:
    """Reduction to an index in ``range(n)``.

    ``seed_parts_prefix`` is hashed together with an appended trailing
    ``counter`` (starting at ``counter_start``) until digest acceptance;
    returns ``(index, final_counter)``. Mirrors generator 2.2: interpret the
    32-byte digest as an unsigned 256-bit big-endian integer ``x``,
    ``L=floor(2**256/n)*n``, reject when ``x>=L`` (increment counter,
    rehash), otherwise return ``x % n``. ``n=1`` short-circuits without a PRF
    call, matching "If n=1, the sole candidate is tested ... without a PRF
    call".
    """

    if n <= 0:
        raise StatisticsError("draw_index requires a positive candidate count")
    if n == 1:
        return 0, counter_start
    counter = counter_start
    modulus = 1 << 256
    limit = (modulus // n) * n
    while True:
        digest = _prf_digest(list(seed_parts_prefix) + [counter])
        x = int.from_bytes(digest, "big")
        if x >= limit:
            counter += 1
            continue
        return x % n, counter


def draw_uniform(seed_parts: Sequence[Any], prec: int = 50) -> Decimal:
    """Continuous uniform draw ``u=(x+0.5)/2**256`` from one PRF digest."""

    digest = _prf_digest(seed_parts)
    x = int.from_bytes(digest, "big")
    with localcontext() as ctx:
        ctx.prec = prec + 20
        u = (Decimal(x) + Decimal("0.5")) / Decimal(1 << 256)
    return _round_to(u, prec)


# ---------------------------------------------------------------------------
# Decimal 50-significant-digit arithmetic (spec: "Frozen calibration
# simulation" arithmetic paragraph)
# ---------------------------------------------------------------------------

PI_50 = Decimal("3.1415926535897932384626433832795028841971693993751")
DEFAULT_PREC = 50


def _round_to(value: Decimal, prec: int) -> Decimal:
    ctx = Context(prec=prec)
    return ctx.plus(value)


def dec_ln(x: Decimal, prec: int = DEFAULT_PREC) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = prec + 15
        result = x.ln()
    return _round_to(result, prec)


def dec_exp(x: Decimal, prec: int = DEFAULT_PREC) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = prec + 15
        result = x.exp()
    return _round_to(result, prec)


def dec_sqrt(x: Decimal, prec: int = DEFAULT_PREC) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = prec + 15
        result = x.sqrt()
    return _round_to(result, prec)


def _series_sin_cos(r: Decimal, prec: int) -> tuple[Decimal, Decimal]:
    """Taylor series for (sin(r), cos(r)) with ``r`` already range-reduced
    to roughly ``[-pi, pi]``; internal precision must already be widened by
    the caller."""

    r2 = r * r
    sin_term = r
    sin_total = r
    cos_term = Decimal(1)
    cos_total = Decimal(1)
    tolerance = Decimal(1).scaleb(-(prec + 10))
    n = 1
    for _ in range(1000):
        sin_term = -sin_term * r2 / ((2 * n) * (2 * n + 1))
        cos_term = -cos_term * r2 / ((2 * n - 1) * (2 * n))
        sin_total += sin_term
        cos_total += cos_term
        if abs(sin_term) < tolerance and abs(cos_term) < tolerance:
            break
        n += 1
    return sin_total, cos_total


def _range_reduce(x: Decimal) -> Decimal:
    two_pi = PI_50 * 2
    k = (x / two_pi).to_integral_value()
    return x - k * two_pi


def dec_sin(x: Decimal, prec: int = DEFAULT_PREC) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = prec + 25
        r = _range_reduce(x)
        sin_r, _cos_r = _series_sin_cos(r, prec)
    return _round_to(sin_r, prec)


def dec_cos(x: Decimal, prec: int = DEFAULT_PREC) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = prec + 25
        r = _range_reduce(x)
        _sin_r, cos_r = _series_sin_cos(r, prec)
    return _round_to(cos_r, prec)


def dec_erf(x: Decimal, prec: int = DEFAULT_PREC) -> Decimal:
    """Best-effort Taylor-series ``erf`` (no closed form / stdlib support in
    ``decimal``). Accurate for ``|x|`` up to a few units; large ``|x|``
    saturates to +/-1 within the requested precision."""

    with localcontext() as ctx:
        ctx.prec = prec + 30
        if x >= Decimal(6):
            return _round_to(Decimal(1), prec)
        if x <= Decimal(-6):
            return _round_to(Decimal(-1), prec)
        two_over_sqrt_pi = Decimal(2) / PI_50.sqrt()
        term = x
        total = x
        x2 = x * x
        tolerance = Decimal(1).scaleb(-(prec + 15))
        n = 0
        for _ in range(2000):
            n += 1
            term = -term * x2 * (2 * n - 1) / (n * (2 * n + 1))
            total += term
            if abs(term) < tolerance:
                break
        result = two_over_sqrt_pi * total
    return _round_to(result, prec)


def dec_phi(x: Decimal, prec: int = DEFAULT_PREC) -> Decimal:
    """Standard normal CDF via ``dec_erf`` (Decimal, best-effort precision)."""

    with localcontext() as ctx:
        ctx.prec = prec + 15
        sqrt2 = Decimal(2).sqrt()
        result = (Decimal(1) + dec_erf(x / sqrt2, prec + 15)) / 2
    return _round_to(result, prec)


def bisect_population_p95(
    cdf: Callable[[Decimal], Decimal],
    *,
    prec: int = DEFAULT_PREC,
    tolerance: Decimal = Decimal("1e-30"),
    max_doublings: int = 256,
) -> Decimal:
    """``inf{x: F(x)>=0.95}`` by doubling-then-bisection on nonnegative
    ``x`` (spec: "Bisection starts at zero and doubles the upper endpoint
    until F(upper)>=0.95, then stops at interval width <=1e-30 ns")."""

    with localcontext() as ctx:
        ctx.prec = prec + 15
        lower = Decimal(0)
        upper = Decimal(1)
        target = Decimal("0.95")
        doublings = 0
        while cdf(upper) < target:
            upper *= 2
            doublings += 1
            if doublings > max_doublings:
                raise StatisticsError("failed to bracket p95 within max_doublings")
        while (upper - lower) > tolerance:
            mid = (lower + upper) / 2
            if cdf(mid) >= target:
                upper = mid
            else:
                lower = mid
        result = upper
    return _round_to(result, prec)


def empirical_population_p95(values: Sequence[Any]) -> Any:
    """Exact finite empirical distribution p95: nearest-rank order statistic."""

    return nearest_rank_percentile(values, 0.95)


def lognormal_population_p95(mu: Decimal, sigma: Decimal, prec: int = DEFAULT_PREC) -> Decimal:
    """``inf{x: Phi((ln x - mu)/sigma) >= 0.95}`` via bisection on the
    Decimal CDF (spec permits bisection when an exact analytic inverse
    requires an unavailable primitive; this worker has no Decimal
    ``Phi^-1``, only ``dec_phi``, so it bisects directly on ``x``)."""

    def cdf(x: Decimal) -> Decimal:
        if x <= 0:
            return Decimal(0)
        return dec_phi((dec_ln(x, prec) - mu) / sigma, prec)

    return bisect_population_p95(cdf, prec=prec)


def standard_normal_pair(u1: Decimal, u2: Decimal, prec: int = DEFAULT_PREC) -> tuple[Decimal, Decimal]:
    """Box-Muller pair: ``z0=sqrt(-2 ln u1) cos(2 pi u2)``,
    ``z1=sqrt(-2 ln u1) sin(2 pi u2)``."""

    if not (Decimal(0) < u1 < Decimal(1)) or not (Decimal(0) <= u2 < Decimal(1)):
        raise StatisticsError("Box-Muller uniforms must satisfy 0<u1<1 and 0<=u2<1")
    with localcontext() as ctx:
        ctx.prec = prec + 25
        r = (Decimal(-2) * u1.ln()).sqrt()
        theta = 2 * PI_50 * u2
        reduced = _range_reduce(theta)
        sin_t, cos_t = _series_sin_cos(reduced, prec)
        z0 = r * cos_t
        z1 = r * sin_t
    return _round_to(z0, prec), _round_to(z1, prec)


# ---------------------------------------------------------------------------
# Truth transforms (spec: "The transformation multiplies every call/unit of
# A by one positive constant a and every call/unit of B by one positive
# constant b")
# ---------------------------------------------------------------------------


def geometric_mean(values: Sequence[Decimal], prec: int = DEFAULT_PREC) -> Decimal:
    if not values:
        raise StatisticsError("geometric mean requires a nonempty sequence")
    with localcontext() as ctx:
        ctx.prec = prec + 15
        total = sum((dec_ln(v, prec) for v in values), Decimal(0))
        result = (total / len(values)).exp()
    return _round_to(result, prec)


def arithmetic_mean(values: Sequence[Decimal], prec: int = DEFAULT_PREC) -> Decimal:
    if not values:
        raise StatisticsError("arithmetic mean requires a nonempty sequence")
    with localcontext() as ctx:
        ctx.prec = prec + 15
        result = sum(values, Decimal(0)) / len(values)
    return _round_to(result, prec)


def transform_speedup_only(i_a: Decimal, i_b: Decimal, rho: Decimal, prec: int = DEFAULT_PREC) -> tuple[Decimal, Decimal]:
    """``a=1``, ``b=rho*I_A/I_B``."""

    if i_a <= 0 or i_b <= 0 or rho <= 0:
        raise StatisticsError("speedup transform requires positive indices and target")
    a = Decimal(1)
    b = _round_to(rho * i_a / i_b, prec)
    if b <= 0:
        raise StatisticsError("speedup-only transform requires b>0")
    return a, b


def transform_saving_only(
    mean_a: Decimal, mean_b: Decimal, d: Decimal, prec: int = DEFAULT_PREC
) -> tuple[Decimal, Decimal]:
    """``a=1``, ``b=(d+mean_A)/mean_B``."""

    if mean_a <= 0 or mean_b <= 0 or d < 0:
        raise StatisticsError("saving transform requires positive means and nonnegative target")
    a = Decimal(1)
    b = _round_to((d + mean_a) / mean_b, prec)
    if b <= 0:
        raise StatisticsError("additive-saving-only transform requires b>0")
    return a, b


def transform_joint(
    i_a: Decimal,
    i_b: Decimal,
    mean_a: Decimal,
    mean_b: Decimal,
    rho: Decimal,
    d: Decimal,
    prec: int = DEFAULT_PREC,
) -> tuple[Decimal, Decimal]:
    """Joint ``(rho, d)`` transform with ``d>0``; fails unless ``D>0``,
    ``a>0``, ``b>0``."""

    if min(i_a, i_b, mean_a, mean_b) <= 0 or rho <= 0 or d <= 0:
        raise StatisticsError("joint transform requires positive indices/means, rho, and d")
    with localcontext() as ctx:
        ctx.prec = prec + 15
        d_value = rho * i_a * mean_b / i_b - mean_a
        if d_value <= 0:
            raise StatisticsError("joint transform requires D>0")
        a = d / d_value
        b = rho * a * i_a / i_b
        if a <= 0 or b <= 0:
            raise StatisticsError("joint transform requires a>0 and b>0")
    return _round_to(a, prec), _round_to(b, prec)


# ---------------------------------------------------------------------------
# Paired BCa bootstrap (spec: "Proposed thresholds and deterministic
# classifier")
# ---------------------------------------------------------------------------


def _phi_float(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv_float(p: float) -> float:
    """Peter Acklam's rational approximation to the standard normal
    quantile function (~1.15e-9 relative accuracy); float64, used only for
    the general (non-calibration-truth) BCa endpoint construction."""

    if p <= 0.0 or p >= 1.0:
        raise StatisticsError("phi_inv requires p in (0,1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def _quantile_type7(sorted_values: Sequence[float], p: float) -> float:
    """Quantile type 7: ``h=1+(B-1)*p``, linear interpolation between the
    surrounding one-based order statistics."""

    b = len(sorted_values)
    if b == 0:
        raise StatisticsError("quantile type 7 requires a nonempty sample")
    if not 0.0 <= p <= 1.0 or not math.isfinite(p):
        raise StatisticsError("quantile type 7 p must lie in [0,1]")
    if b == 1:
        return sorted_values[0]
    h = 1 + (b - 1) * p
    lo = max(1, min(b, math.floor(h)))
    hi = max(1, min(b, math.ceil(h)))
    frac = h - lo
    return sorted_values[lo - 1] + frac * (sorted_values[hi - 1] - sorted_values[lo - 1])


def paired_bca_interval(
    units: Sequence[Any],
    statistic_fn: Callable[[Sequence[Any]], float],
    *,
    resamples: int = 10000,
    seed_parts_prefix: Sequence[Any] = (),
) -> dict:
    """Paired BCa 95% interval over independent ``units`` (whole paired
    processes/blocks; never individual calls). ``statistic_fn`` recomputes
    the target statistic (e.g. p95, family/profile index, speedup, saving)
    from a resampled sequence of units. Deterministic resampling reuses the
    canonical-frame PRF with domain-separated ``seed_parts_prefix``. Returns
    ``{"inconclusive": False, "point_estimate", "lower", "upper", "z0",
    "acceleration"}`` or ``{"inconclusive": True, "reason": ...}`` (spec:
    "Undefined denominator/acceleration or an adjusted probability outside
    [0,1] makes the cell Inconclusive.").
    """

    n = len(units)
    if n == 0:
        raise StatisticsError("paired_bca_interval requires a nonempty independent-unit population")
    try:
        theta_hat = statistic_fn(units)
        if not math.isfinite(float(theta_hat)):
            return {"inconclusive": True, "reason": "non-finite point statistic"}
    except (ValueError, StatisticsError, ZeroDivisionError, OverflowError) as exc:
        return {"inconclusive": True, "reason": f"point statistic undefined: {exc}"}

    replicates = []
    for b in range(resamples):
        sample = []
        for i in range(n):
            idx, _counter = draw_index(n, list(seed_parts_prefix) + ["resample", b, i])
            sample.append(units[idx])
        try:
            value = statistic_fn(sample)
        except (ValueError, StatisticsError, ZeroDivisionError, OverflowError) as exc:
            return {"inconclusive": True, "reason": f"bootstrap statistic undefined: {exc}"}
        if not math.isfinite(float(value)):
            return {"inconclusive": True, "reason": "non-finite bootstrap statistic"}
        replicates.append(value)

    if n == 1:
        return {"inconclusive": True, "reason": "acceleration undefined for n=1"}

    try:
        jack_values = [statistic_fn(units[:i] + units[i + 1 :]) for i in range(n)]
    except (ValueError, StatisticsError, ZeroDivisionError, OverflowError) as exc:
        return {"inconclusive": True, "reason": f"jackknife statistic undefined: {exc}"}
    mean_jack = sum(jack_values) / len(jack_values)
    numerator = sum((mean_jack - v) ** 3 for v in jack_values)
    denominator = 6 * (sum((mean_jack - v) ** 2 for v in jack_values)) ** 1.5
    if denominator == 0 or not math.isfinite(float(denominator)):
        return {"inconclusive": True, "reason": "undefined jackknife acceleration denominator"}
    acceleration = numerator / denominator

    less = sum(1 for r in replicates if r < theta_hat)
    equal = sum(1 for r in replicates if r == theta_hat)
    proportion = (less + 0.5 * equal) / resamples
    if proportion <= 0.0 or proportion >= 1.0:
        return {"inconclusive": True, "reason": "z0 undefined at replicate-proportion boundary"}
    z0 = _phi_inv_float(proportion)

    sorted_replicates = sorted(replicates)
    endpoints: dict[float, float] = {}
    for alpha in (0.025, 0.975):
        z_alpha = _phi_inv_float(alpha)
        denom = 1 - acceleration * (z0 + z_alpha)
        if denom == 0 or not math.isfinite(float(denom)):
            return {"inconclusive": True, "reason": "undefined adjusted probability denominator"}
        adjusted = _phi_float(z0 + (z0 + z_alpha) / denom)
        if not math.isfinite(adjusted) or not (0.0 <= adjusted <= 1.0):
            return {"inconclusive": True, "reason": "adjusted probability outside [0,1]"}
        endpoints[alpha] = _quantile_type7(sorted_replicates, adjusted)

    return {
        "inconclusive": False,
        "point_estimate": theta_hat,
        "lower": endpoints[0.025],
        "upper": endpoints[0.975],
        "z0": z0,
        "acceleration": acceleration,
    }


# ---------------------------------------------------------------------------
# Backend-neutral paired estimands and calibration truth machinery
# ---------------------------------------------------------------------------


def _pair_value(unit: Any, side: int | str) -> Any:
    """Extract one backend value from a fresh unit or warm block.

    The public calibration helpers accept either ``(A, B)``/``[A, B]``
    records, mappings with ``a``/``b`` or ``S``/``G`` keys, or objects with
    corresponding attributes.  A warm block is represented by vectors on
    each side and is handled by :func:`unit_values` below.
    """
    key = "a" if side in (0, "a", "A", "S", "s") else "b"
    if isinstance(unit, Mapping):
        for candidate in ((key, key.upper()), ("S", "G") if key == "a" else ("G", "S")):
            for name in candidate:
                if name in unit:
                    return unit[name]
        raise StatisticsError(f"paired unit has no backend {key!r}")
    if hasattr(unit, key):
        return getattr(unit, key)
    if hasattr(unit, key.upper()):
        return getattr(unit, key.upper())
    try:
        return unit[0 if key == "a" else 1]
    except (IndexError, KeyError, TypeError) as exc:
        raise StatisticsError("paired unit must contain two backend values") from exc


def unit_values(unit: Any, side: int | str) -> tuple[float, ...]:
    """Return one independent unit's observations for ``side``.

    A scalar is a fresh-process unit.  A finite sequence is a warm block and
    remains intact during resampling; no helper ever treats its calls as
    independent resampling units.
    """
    value = _pair_value(unit, side)
    if isinstance(value, (str, bytes, bytearray)):
        raise StatisticsError("latency observations must be numeric")
    if isinstance(value, Sequence):
        result = tuple(float(v) for v in value)
        if not result:
            raise StatisticsError("warm block cannot be empty")
    else:
        result = (float(value),)
    if any(not math.isfinite(v) or v <= 0 for v in result):
        raise StatisticsError("latency observations must be finite and positive")
    return result


def instance_p95(units: Sequence[Any], side: int | str) -> float:
    """Query-instance p95, preserving warm-block call vectors."""
    observations = [value for unit in units for value in unit_values(unit, side)]
    return float(nearest_rank_percentile(observations, 0.95))


def family_profile_latency_index(instance_units: Sequence[Sequence[Any]], side: int | str) -> float:
    """Equal-weight geometric mean of instance p95s."""
    return math.exp(sum(math.log(instance_p95(units, side)) for units in instance_units) / len(instance_units))


def family_profile_absolute_saving(instance_units: Sequence[Sequence[Any]], side_a: int | str = "a", side_b: int | str = "b") -> float:
    """Equal-weight arithmetic mean of paired instance p95 differences ``B-A``."""
    if not instance_units:
        raise StatisticsError("family/profile cell has no instances")
    return sum(
        instance_p95(units, side_b) - instance_p95(units, side_a)
        for units in instance_units
    ) / len(instance_units)


def family_profile_speedup(instance_units: Sequence[Sequence[Any]], side_a: int | str = "a", side_b: int | str = "b") -> float:
    """Speedup of ``side_a`` over ``side_b`` as ``I_B / I_A``."""
    a = family_profile_latency_index(instance_units, side_a)
    b = family_profile_latency_index(instance_units, side_b)
    if a <= 0:
        raise StatisticsError("latency index must be positive")
    return b / a


def warm_population_p95(blocks: Sequence[Any], side: int | str, *, prec: int = DEFAULT_PREC) -> Decimal:
    """Exact p95 of an equally weighted mixture over ordered warm positions."""
    if not blocks:
        raise StatisticsError("warm population requires at least one block")
    values = [Decimal(str(v)) for block in blocks for v in unit_values(block, side)]
    return Decimal(str(empirical_population_p95(values)))


def empirical_truth(values: Sequence[Any], *, prec: int = DEFAULT_PREC) -> Decimal:
    """Exact finite empirical population p95 without bootstrap approximation."""
    if not values:
        raise StatisticsError("empirical truth requires a nonempty population")
    return Decimal(str(empirical_population_p95([Decimal(str(v)) for v in values])))


def lognormal_truth(mu: Decimal, sigma: Decimal, *, prec: int = DEFAULT_PREC) -> Decimal:
    if sigma <= 0:
        raise StatisticsError("lognormal sigma must be positive")
    return lognormal_population_p95(Decimal(mu), Decimal(sigma), prec=prec)


def two_component_tail_truth(
    base: Decimal | Sequence[Decimal] | Callable[[Decimal], Decimal],
    weight: Decimal,
    multiplier: Decimal,
    *,
    prec: int = DEFAULT_PREC,
) -> Decimal:
    """Exact p95 for the frozen common-tail mixture.

    ``base`` may be a CDF, a finite empirical population, or (for a
    degenerate synthetic fixture) a positive scalar point mass.  The common
    tail multiplies both backends and every warm call, so the component CDF is
    ``F(x / multiplier)``.
    """
    if not (Decimal(0) <= weight <= Decimal(1)) or multiplier <= 0:
        raise StatisticsError("tail weight/multiplier outside the frozen domain")
    if weight == 0:
        if callable(base):
            return bisect_population_p95(base, prec=prec)
        if isinstance(base, Sequence) and not isinstance(base, (str, bytes, bytearray)):
            return Decimal(str(empirical_population_p95([Decimal(str(v)) for v in base])))
        return Decimal(str(base))
    if callable(base):
        cdf = base
    elif isinstance(base, Sequence) and not isinstance(base, (str, bytes, bytearray)):
        values = tuple(Decimal(str(v)) for v in base)
        if not values or any(v <= 0 for v in values):
            raise StatisticsError("empirical tail base must contain positive values")
        ordered = sorted(values)
        def cdf(x: Decimal) -> Decimal:
            return Decimal(sum(value <= x for value in ordered)) / len(ordered)
    else:
        value = Decimal(str(base))
        if value <= 0:
            raise StatisticsError("point-mass tail base must be positive")
        def cdf(x: Decimal) -> Decimal:
            return Decimal(1) if x >= value else Decimal(0)
    return mixture_population_p95(cdf, cdf, weight, multiplier, prec=prec)


def mixture_population_p95(
    cdf: Callable[[Decimal], Decimal], tail_cdf: Callable[[Decimal], Decimal],
    weight: Decimal, multiplier: Decimal, *, prec: int = DEFAULT_PREC,
) -> Decimal:
    """Exact p95 of ``(1-w)X + w*kX`` by Decimal bisection."""
    if not (Decimal(0) <= weight <= Decimal(1)) or multiplier <= 0:
        raise StatisticsError("tail weight/multiplier outside the frozen domain")
    def mixed(x: Decimal) -> Decimal:
        return (Decimal(1) - weight) * cdf(x) + weight * tail_cdf(x / multiplier)
    return bisect_population_p95(mixed, prec=prec)


def classifier_truth_transform(
    ia: Decimal, ib: Decimal, mean_a: Decimal, mean_b: Decimal,
    *, speedup: Decimal | None = None, saving: Decimal | None = None,
    prec: int = DEFAULT_PREC,
) -> tuple[Decimal, Decimal]:
    """Return the exact positive ``(a,b)`` constants for a frozen surface."""
    if speedup is None and saving is None:
        raise StatisticsError("at least one classifier target is required")
    if speedup is not None and saving is not None:
        return transform_joint(ia, ib, mean_a, mean_b, speedup, saving, prec)
    if speedup is not None:
        return transform_speedup_only(ia, ib, speedup, prec)
    return transform_saving_only(mean_a, mean_b, saving, prec)  # type: ignore[arg-type]


def paired_truth_statistics(
    instance_populations: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    *, prec: int = DEFAULT_PREC,
) -> dict[str, Decimal]:
    """Recompute classifier truth from query-instance population p95s."""
    if not instance_populations:
        raise StatisticsError("truth requires at least one query instance")
    qa = [empirical_truth(a, prec=prec) for a, _b in instance_populations]
    qb = [empirical_truth(b, prec=prec) for _a, b in instance_populations]
    ia = geometric_mean(qa, prec)
    ib = geometric_mean(qb, prec)
    return {"index_a": ia, "index_b": ib, "speedup_b_over_a": ib / ia, "saving_b_over_a": arithmetic_mean(qb, prec) - arithmetic_mean(qa, prec)}


def empirical_paired_simulation(
    units: Sequence[Any], statistic_fn: Callable[[Sequence[Any]], Any], *,
    campaigns: int = 10_000, seed: int = 0,
) -> tuple[Any, ...]:
    """Deterministically resample whole paired fresh units or warm blocks."""
    if not units or campaigns <= 0:
        raise StatisticsError("empirical simulation requires positive units/campaigns")
    n = len(units)
    values = []
    for campaign in range(campaigns):
        sample = [units[draw_index(n, [RATIFICATION_DOMAIN, seed, "empirical", campaign, draw])[0]] for draw in range(n)]
        values.append(statistic_fn(sample))
    return tuple(values)


def lognormal_unit_parameters(
    paired_unit_p95s: Sequence[tuple[Decimal, Decimal]], *, prec: int = DEFAULT_PREC,
) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    """Return log means and unbiased covariance entries for paired units."""
    if len(paired_unit_p95s) < 2:
        raise StatisticsError("lognormal covariance requires at least two units")
    logs = [(dec_ln(a, prec), dec_ln(b, prec)) for a, b in paired_unit_p95s]
    mean_a = arithmetic_mean([a for a, _b in logs], prec)
    mean_b = arithmetic_mean([b for _a, b in logs], prec)
    denom = Decimal(len(logs) - 1)
    var_a = sum((a - mean_a) ** 2 for a, _b in logs) / denom
    var_b = sum((b - mean_b) ** 2 for _a, b in logs) / denom
    cov = sum((a - mean_a) * (b - mean_b) for a, b in logs) / denom
    if var_a <= 0 or var_b <= 0 or var_a * var_b - cov * cov <= 0:
        raise StatisticsError("lognormal covariance is singular or non-positive")
    return (_round_to(mean_a, prec), _round_to(mean_b, prec), _round_to(var_a, prec), _round_to(var_b, prec), _round_to(cov, prec))


def lognormal_paired_simulation(
    paired_unit_p95s: Sequence[tuple[Decimal, Decimal]], *, campaigns: int = 10_000,
    simulation_seed: int = 0, covariance_multiplier: Decimal = Decimal(1),
    prec: int = DEFAULT_PREC,
) -> tuple[tuple[Decimal, Decimal], ...]:
    """Generate deterministic correlated lognormal paired unit p95s."""
    if campaigns <= 0 or covariance_multiplier <= 0:
        raise StatisticsError("lognormal simulation arguments are invalid")
    mu_a, mu_b, var_a, var_b, cov = lognormal_unit_parameters(paired_unit_p95s, prec=prec)
    with localcontext() as ctx:
        ctx.prec = prec + 20
        l11 = dec_sqrt(var_a, prec + 10)
        l21 = cov / l11
        remainder = var_b - l21 * l21
        if remainder <= 0:
            raise StatisticsError("lognormal covariance is singular")
        l22 = dec_sqrt(remainder, prec + 10)
        result = []
        for campaign in range(campaigns):
            u1 = draw_uniform([RATIFICATION_DOMAIN, simulation_seed, "lognormal", campaign, 0], prec + 10)
            u2 = draw_uniform([RATIFICATION_DOMAIN, simulation_seed, "lognormal", campaign, 1], prec + 10)
            z0, z1 = standard_normal_pair(u1, u2, prec + 10)
            z0 *= covariance_multiplier; z1 *= covariance_multiplier
            result.append((_round_to(dec_exp(mu_a + l11 * z0), prec), _round_to(dec_exp(mu_b + l21 * z0 + l22 * z1), prec)))
    return tuple(result)


def two_component_tail_simulation(
    paired_values: Sequence[tuple[Decimal, Decimal]], *, weight: Decimal, multiplier: Decimal,
    campaigns: int = 10_000, simulation_seed: int = 0, prec: int = DEFAULT_PREC,
) -> tuple[tuple[Decimal, Decimal], ...]:
    """Apply a common deterministic tail multiplier to paired values."""
    if not paired_values or campaigns <= 0 or not (Decimal(0) <= weight <= Decimal(1)) or multiplier <= 0:
        raise StatisticsError("two-component simulation arguments are invalid")
    result = []
    for campaign in range(campaigns):
        index, _counter = draw_index(len(paired_values), [RATIFICATION_DOMAIN, simulation_seed, "two_component_tail", campaign, "unit"])
        a, b = paired_values[index]
        tail = draw_uniform([RATIFICATION_DOMAIN, simulation_seed, "two_component_tail", campaign, "tail"], prec + 10) < weight
        factor = multiplier if tail else Decimal(1)
        result.append((_round_to(a * factor, prec), _round_to(b * factor, prec)))
    return tuple(result)


def two_component_tail_population(
    base_cdf: Callable[[Decimal], Decimal], weight: Decimal, multiplier: Decimal,
    *, prec: int = DEFAULT_PREC,
) -> Decimal:
    """Named API for the frozen two-component-tail p95 truth."""
    return mixture_population_p95(base_cdf, base_cdf, weight, multiplier, prec=prec)


# ---------------------------------------------------------------------------
# Frozen duration-category accounting
# ---------------------------------------------------------------------------


DURATION_OPERATION_KINDS = frozenset({
    "cache_helper", "process_start", "query_warmup", "query_measured",
    "process_teardown", "clean_build", "rebuild", "transition",
    "semantic_validation", "full_export", "clone_copy", "checksum",
    "monitor_start", "monitor_stop", "crash_interrupt", "rollback",
    "recovery", "publication_gate",
})


def duration_category(
    operation_kind: str, backend: str, target_tier: str, profile_id: str,
    cache_process_mode: str, query_id: str = "not_applicable",
    transition_id: str = "not_applicable",
) -> tuple[str, str, str, str, str, str, str]:
    """Construct and validate the exact seven-field category tuple."""
    if operation_kind not in DURATION_OPERATION_KINDS:
        raise StatisticsError(f"unknown duration operation_kind {operation_kind!r}")
    if operation_kind == "transition" and transition_id == "not_applicable":
        raise StatisticsError("transition category requires transition_id")
    if operation_kind != "transition" and transition_id != "not_applicable":
        raise StatisticsError("non-transition category must use not_applicable transition_id")
    if operation_kind not in {"query_warmup", "query_measured"} and query_id != "not_applicable":
        raise StatisticsError("non-query category must use not_applicable query_id")
    return (operation_kind, backend, target_tier, profile_id, cache_process_mode, query_id, transition_id)


def duration_quantile_and_upper(
    observations: Sequence[float], *, alpha: float = 0.99,
    seed_parts_prefix: Sequence[Any] = (), resamples: int = 10000,
) -> dict[str, Any]:
    """Return q99 and the one-sided BCa upper endpoint for a category mean."""
    if not observations:
        raise StatisticsError("duration category has no observations")
    q = nearest_rank_percentile(observations, alpha)
    bca = paired_bca_interval(tuple(float(x) for x in observations), lambda sample: sum(sample) / len(sample), resamples=resamples, seed_parts_prefix=seed_parts_prefix)
    if bca.get("inconclusive"):
        raise StatisticsError(f"duration BCa endpoint inconclusive: {bca.get('reason')}")
    # ``paired_bca_interval`` returns a two-sided interval.  Its upper
    # endpoint uses alpha=.975; for the frozen duration rule a one-sided 99%
    # endpoint is obtained by a dedicated alpha=0.99 implementation below.
    upper = bca_upper_endpoint(tuple(float(x) for x in observations), lambda sample: sum(sample) / len(sample), alpha=alpha, resamples=resamples, seed_parts_prefix=seed_parts_prefix)
    return {"q99": q, "bca_upper": upper, "u": max(q, upper), "count": len(observations)}


def bca_upper_endpoint(
    units: Sequence[Any], statistic_fn: Callable[[Sequence[Any]], float], *,
    alpha: float = 0.99, resamples: int = 10000,
    seed_parts_prefix: Sequence[Any] = (),
) -> float:
    """One-sided BCa endpoint with the same whole-unit resampling rule."""
    if not 0 < alpha < 1:
        raise StatisticsError("BCa alpha must lie in (0,1)")
    result = _paired_bca_endpoint(units, statistic_fn, alpha=alpha, resamples=resamples, seed_parts_prefix=seed_parts_prefix)
    return result


def _paired_bca_endpoint(
    units: Sequence[Any], statistic_fn: Callable[[Sequence[Any]], float], *,
    alpha: float, resamples: int, seed_parts_prefix: Sequence[Any],
) -> float:
    n = len(units)
    if n < 2:
        raise StatisticsError("BCa endpoint requires at least two independent units")
    theta_hat = float(statistic_fn(units))
    replicates = []
    for b in range(resamples):
        sample = [units[draw_index(n, list(seed_parts_prefix) + ["resample", b, i])[0]] for i in range(n)]
        replicates.append(float(statistic_fn(sample)))
    jack = [float(statistic_fn(units[:i] + units[i + 1 :])) for i in range(n)]
    mean_jack = sum(jack) / n
    diffs = [mean_jack - value for value in jack]
    denom = 6 * sum(d * d for d in diffs) ** 1.5
    if denom == 0:
        raise StatisticsError("BCa acceleration denominator is zero")
    acceleration = sum(d ** 3 for d in diffs) / denom
    proportion = (sum(v < theta_hat for v in replicates) + 0.5 * sum(v == theta_hat for v in replicates)) / resamples
    if not 0 < proportion < 1:
        raise StatisticsError("BCa bias correction is undefined")
    z0 = _phi_inv_float(proportion)
    za = _phi_inv_float(alpha)
    denominator = 1 - acceleration * (z0 + za)
    if denominator == 0:
        raise StatisticsError("BCa adjusted probability denominator is zero")
    adjusted = _phi_float(z0 + (z0 + za) / denominator)
    if not 0 <= adjusted <= 1:
        raise StatisticsError("BCa adjusted probability outside [0,1]")
    return _quantile_type7(sorted(replicates), adjusted)


def project_campaign_seconds(category_counts: Mapping[tuple, int], category_bounds: Mapping[tuple, float], *, headroom: float = 1.25) -> float:
    """Apply the frozen 1.25 multiplier to the complete duration manifest."""
    if headroom != 1.25:
        raise StatisticsError("CYAX-0168 duration headroom is fixed at 1.25")
    total = 0.0
    for category, count in category_counts.items():
        if category not in category_bounds or count < 0:
            raise StatisticsError("duration manifest is incomplete or invalid")
        bound = float(category_bounds[category])
        if not math.isfinite(bound) or bound < 0:
            raise StatisticsError("duration bound must be finite and nonnegative")
        total += int(count) * bound
    return headroom * total


# Concise names retained for manifest/calibration callers.
nearest_rank = nearest_rank_percentile
bca_interval = paired_bca_interval
p95_estimate = p95
