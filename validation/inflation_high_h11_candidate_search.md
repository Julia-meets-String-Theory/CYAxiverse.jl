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

Results were two successful screens and one expected branch cap. The
successful branch counts were 150 at h11=150 and 600 at h11=300; h11=350 was
capped before callbacks because its exact filtered estimate was 1,050. All
successful structures validated and all candidate counts were zero. The h11=300
row allocated approximately 4.61 GB cumulatively in the per-branch
classification stage, exceeding the nominal 750 MB policy ceiling. This is a
resource diagnostic and a stop condition for larger high-h11 screening until
the classification allocation boundary is improved or explicitly reviewed.

No arbitrary-precision trajectory refinement or h11=491 production scan was
run.
