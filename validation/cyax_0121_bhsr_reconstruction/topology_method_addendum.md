# Corrective spin-topology method addendum

Addendum ID: `CYAX-0121-BHSR-TOPOLOGY-ADDENDUM-r1`.

This document supplements the immutable numerical-method manifest
`7b9c1cf9e4787372801125d7c195288d36f3cb9e094b0a2ec179493c8e8392d5`.
The original manifest remains the identity for the frozen scalar formula and
root method. This addendum identifies the corrective topology method needed
because the source residual need not be monotone on spin `[0,1]`.

## Fixed method

All topology evaluations use 256-bit `BigFloat` by default. The scan ladder is
257, 513, and 1025 equally spaced spin samples on the closed interval `[0,1]`.
A caller-selected base count `n` uses the nested ladder `[n, 2n-1, 4n-3]`.
Each finite sampled local maximum and minimum is bracketed by its adjacent scan
samples and refined by golden-section search. Extremum bracket width must reach
`1e-40`, with a maximum of 240 golden-section iterations; failure makes the
result unavailable. Every detected sign-changing root is refined by bisection
to a spin bracket width of `1e-40`, with a maximum of 140 iterations; failure
makes the result unavailable. Roots within `2e-40` are deduplicated.

An extremum whose absolute residual is at most `1e-40` is near-tangent and
numerically unresolved. Any such extremum marks the topology
`:unavailable_near_tangent`; it must not be classified as either efficient or
inefficient. The scan levels must agree in root count, interval count, topology
status, and each root/interval endpoint within `1e-30`. Otherwise the topology
is `:unavailable_topology_resolution`. Either unavailable state removes the
accepted onset and intervals from source contour products. Candidate roots and
intervals may be retained for diagnosis only.

Intervals are formed from all accepted crossings by evaluating the residual
between adjacent boundaries. The per-mode union sorts intervals and merges
only intervals with positive-width overlap; intervals that merely touch remain
separate. A scalar onset is the lower boundary of the first accepted interval
and is never a complete spin-region description. A contour is representable
only when the complete mode union is one interval with positive onset and upper
boundary at spin 1. A bounded interval, an interval starting at zero, multiple
intervals, a near-tangent state, or unstable resolution is not a single-onset
contour and must fail closed in direct and inverse likelihood evaluation.

## Limits

This is a finite scan with local-extremum refinement. Agreement across the
three resolutions is a stability check, not a proof that an arbitrarily narrow
feature was found. A result with no interval is reported as
`:no_positive_interval_resolved`; it does not prove global absence. A refined
extremum inside the near-tangent margin is unavailable, so the code does not
make a categorical sign decision there. The S1 and S4 formula cases were
recomputed with this addendum after the earlier monotonicity-based results were
invalidated.
