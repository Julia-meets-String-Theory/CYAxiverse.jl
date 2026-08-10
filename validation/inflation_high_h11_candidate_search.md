# High-\(h^{1,1}\) candidate-search validation

Date: 2026-08-07

This note records bounded validation of the deterministic low-index branch
path. It is evidence for the search implementation, not a completeness claim
for the geometry corpus or for h11=491.

## Exact branch-streaming checks

`foreach_leading_critical_branch` now supports `negative_mode_range=K:L` and
`max_negative_modes=K`. Complete enumeration remains the default. The exact
`BigInt` branch estimate is checked before lattice-offset or branch-workspace
allocation. The returned report records the full mask count, masks visited and
skipped, lattice copies, and whether the search was complete or deterministic
low-index enumeration.

Regression coverage includes:

- exact `k=1` filtering and legacy numeric-mask ordering on a determinant-four
  synthetic geometry;
- signed and zero-amplitude leading signs;
- a 150-dimensional identity charge matrix, where complete enumeration is
  rejected with the exact `2^150` count while `k=1` visits 150 masks;
- complete-vs-filtered branch coordinate and inertia consistency.

## Structured evaluator checks

The validated base-plus-pairwise evaluator proves every non-base charge
column against an oriented difference of two base columns and preserves the
original instanton order and `L` data. It falls back to the generic
log-shifted evaluator when the shape or charge proof fails.

On local real geometries `(h11, polytope, frst) = (50,1,1), (150,1,1),
(350,1,1)`, the structure proof passed. Generic-vs-structured value,
gradient, and Hessian differences were at floating-point noise for h11=50 and
150. The structured h11=350 derivative evaluation completed in approximately
0.015 s in the measured warm process; the generic dense reference was not
forced at h11=350 because its repeated `Q*Diagonal*Q'` work is not a safe
bounded check at that dimension.

## Lower-h11 calibration

The first geometry in each of h11=8, 11, 13, 14, and 16 was compared with
complete enumeration and `k=1` enumeration. Both paths used the same full
Float64 classifier. All complete and low-index runs returned zero screened
candidates, so candidate recall is `0/0` and remains unmeasured rather than
being reported as perfect recall.

| h11 | complete branches | k=1 branches | complete candidates | k=1 candidates |
|---:|---:|---:|---:|---:|
| 8  | 768 | 24 | 0 | 0 |
| 11 | 2,048 | 11 | 0 | 0 |
| 13 | 32,768 | 52 | 0 | 0 |
| 14 | 16,384 | 14 | 0 | 0 |
| 16 | 327,680 | 80 | 0 | 0 |

The low-index hypothesis is therefore not calibrated for candidate recall by
this sample and must not be described as complete.

An expanded hierarchy-stratified reference was then run on one strong and one
weak geometry at each of h11=8, 11, 13, 14, and 16. The selected strong rows
were `(8,544,1)`, `(11,438,1)`, `(13,405,1)`, `(14,443,1)`, and `(16,84,1)`;
the weak comparison rows were `(h,1,1)`. Complete branch counts ranged from
768 to 327,680. Each geometry was screened three ways: complete, `1:1`, and
`1:2`. Every complete reference had zero candidates, so the resulting recall
is still `0/0` in both hierarchy strata and remains unmeasured.

The hierarchy inventory itself was not sparse: among 1,000 geometries per
stratum, the strong-hierarchy counts were 171 (h11=8), 316 (11), 386 (13),
425 (14), and 433 (16). Thus the absence of candidates in this calibration is
not caused by sampling only weak hierarchies; it is still not evidence of
corpus-wide candidate absence.

## Allocation-boundary optimization

The first high-h11 pilot exposed two avoidable allocation sources. The
screening eigensolver now reuses its DSYEVD work arrays, and the Cholesky lower
factor is materialized once per geometry rather than once per branch. The
filtered mask traversal uses reusable bit vectors instead of BigInt recursion.
Finally, exact determinant accounting uses a checked Int Bareiss path when
intermediates fit; it falls back to the original dense BigInt determinant on
overflow. Fixed integer tests agree with dense BigInt determinants, and the
h11=300 charge matrix takes the checked path.

These changes reduced the h11=300 low-index stage allocation from 4.55 GB
(before caching the factor) / 4.12 GB (after factor caching but before the
checked determinant) to 629 MB. The latter is below the 750 MB policy ceiling.

## Bounded high-h11 pilot

The following screening-only command used one local geometry each at h11=150,
300, and 350:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_prep.jl \
  --data-dir ../../data \
  --geometry 150,1,1 --geometry 300,1,1 --geometry 350,1,1 \
  --max-branches 1000 --negative-mode-range 1:1 \
  --shard-dir /tmp/inflation-high-h11-k1/shards
```

The original command produced two successful screens and one expected branch
cap. After the allocation optimization, the resource-envelope rows were:

| h11 | filter | max branches | callbacks | candidates | stage allocated |
|---:|:---:|---:|---:|---:|---:|
| 50  | 1:1 | 1,000 | 100   | 0 | 27 MB |
| 100 | 1:1 | 1,000 | 200   | 0 | 50 MB |
| 150 | 1:1 | 1,000 | 150   | 0 | 84 MB |
| 200 | 1:1 | 1,000 | 400   | 0 | 189 MB |
| 300 | 1:1 | 1,000 | 600   | 0 | 629 MB |
| 350 | 1:1 | 2,000 | 1,050 | 0 | 971 MB |

All six structures validated. The h11=50, 100, 150, 200, 300, and 350 rows
reported zero candidates; the h11=300 and h11=350 rows reported strong
hierarchy, while the lower rows in this particular geometry sample did not.
The h11=350 row remains above the nominal 750 MB policy ceiling and is a stop
condition for larger high-h11 screening. The prior h11=350 `max_branches=1000`
run remains an expected pre-callback branch cap at the exact filtered estimate
of 1,050.

No arbitrary-precision trajectory refinement or h11=491 production scan was
run.
