# Handoff: high-h11 Float64 candidate search and precision zoom-in

Date: 2026-08-07
Repository: `CYAxiverse.jl`
Worktree: `/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl.worktrees/inflation-reproduction-instructions`
Branch: `agents/inflation-reproduction-instructions`
Baseline commit: `150ca8a fix high-dimensional inflation screening audit`

This document is a self-contained handoff for the next agent. The immediate
objective is to implement and validate a high-h11 candidate-discovery path that
supports the intended workflow:

```text
cheap Float64 geometry/branch screen
    -> retain a small set of interesting points
    -> arbitrary-precision point and trajectory refinement
```

This is not authorization to launch a full production scan. Scientific
coverage must be measured before any high-h11 result is described as complete.

## 1. Mission and success criteria

Build a bounded, candidate-aware Float64 search for the expanded geometry
corpus, especially geometries whose h11 is too large for exhaustive
enumeration of the leading half-integer branches.

The first successful milestone is not “scan h11=491.” It is:

1. implement an exact, bounded low-index branch search;
2. exploit the observed structured charge matrices without changing their
   scientific meaning;
3. find and store Float64 candidate points with explicit search coverage;
4. calibrate the search against complete branch enumeration at lower h11;
5. only then design the arbitrary-precision refinement boundary for arbitrary
   database geometries.

A candidate-discovery result must distinguish:

- complete enumeration;
- deterministic low-index enumeration;
- deterministic branch subsampling;
- stochastic critical-point search;
- failed or resource-capped search.

Do not collapse these into one `success` meaning.

## 2. Repository and current state

The previous scan work is intentionally script-led. The following boundaries
are already implemented:

- [`src/read.jl`](src/read.jl):
  `CYAxiverse.read.oriented_potential`, which normalizes and validates
  canonical `Q`, `L`, and `K` orientation.
- [`src/generate.jl`](src/generate.jl):
  `LQtilde`, the leading branch primitives, the streaming
  `foreach_leading_critical_branch`, and reusable Float64/log-shifted
  numerical operations.
- [`scripts/inflation_scan_common.jl`](scripts/inflation_scan_common.jl):
  the locked Float64 screening sequence, branch classification, resource
  policy, and refinement-eligibility helper.
- [`scripts/inflation_scan_prep.jl`](scripts/inflation_scan_prep.jl):
  one-geometry orchestration, CSV output, resume, retry, and shard handling.
- [`scripts/inflation_scan_pilot.jl`](scripts/inflation_scan_pilot.jl):
  h11-stratified pilot selection and report aggregation.
- [`scripts/inflation_refinement_common.jl`](scripts/inflation_refinement_common.jl):
  a model-specific arbitrary-precision adapter for `:n8_poly102` only.

The current package branch enumerator is exact about branch-count overflow:
it computes the branch estimate with `BigInt` and raises `ArgumentError` when
the explicit cap is exceeded. This fixes the previous h11=150/300 behavior,
where an overflowing `Int` exponent produced zero callbacks.

The current branch enumerator still attempts the complete set whenever the
cap permits it. It does not yet support an index-filtered branch subset.

The current worktree must remain separate from the main `vmm` worktree. Do not
edit or reset the main worktree. Preserve
`INFLATION_REPRODUCTION_CHECKPOINT.md` and unrelated user changes.

## 3. Locked scientific conventions

Do not change these without updating tests and the scan contract.

For every geometry:

- `Q` is `h11 × n_instantons`;
- `L` is `2 × n_instantons`;
- `K` is `h11 × h11`;
- raw angles are in radians;
- `Q`, `L`, and `K` use the canonical orientation returned by
  `CYAxiverse.read.oriented_potential`;
- canonical/mass-basis trajectory conventions remain those of the reproduction
  fixtures.

The existing bounded screening sequence is:

1. load and orient the potential;
2. call `CYAxiverse.generate.LQtilde(Q, L)` exactly once;
3. call `instanton_hierarchy_diagnostics(L)`;
4. factor `K` once with Cholesky;
5. call `leading_hessian_mass_basis_float64`;
6. stream selected leading branches;
7. evaluate full `Q/L` derivatives and classify each retained point.

Do not materialize the full branch-coordinate matrix in the high-h11 path.
Do not run arbitrary-precision trajectories during scan preparation.

The existing Float64 point classifier reports value, gradient norm, epsilon,
eta extrema, Hessian inertia, and mode counts. The current candidate policy is
`value > 0`, at least one negative mode, `epsilon < 1`, and
`abs(min_eta) < 1`. These are screening conditions, not proof that a fully
converged stationary point or inflationary trajectory exists.

## 4. Database evidence

The data root used from this worktree is `../../data`, resolving to:

```text
/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data
```

The actual geometry groups currently include h11=4 through 50, then sampled
groups at h11=60 through 150, and additional groups at h11=180, 200, 250,
300, 310, 320, 330, and 350. Most groups from h11=50 through 150 contain 100
geometry files; later groups contain smaller samples. There is no actual
`h11_491/*/cyax.h5` group in this local data root.

`h11_491_diagnostic` is auxiliary data, not a geometry potential. Its report
contains:

- requested/computed h11: 491;
- h21: 11;
- effective rays: `495 × 491`;
- basis matrix: `491 × 496`;
- prime toric divisors: 495;
- Mori basis rays: `3696 × 491`.

The 495 value is consistent with the observed `h11 + 4` base-charge
structure, but it is not an actual inflation potential file. Treat any h11=491
instanton count inferred below as an extrapolation, not measured data.

### 4.1 Observed potential dimensions

The real HDF5 geometries follow:

```text
n_instantons = (h11 + 4) + binomial(h11 + 4, 2)
```

Examples measured directly:

| h11 | instantons | selected instantons | sampled `abs(det(Qtilde))` |
|---:|---:|---:|---:|
| 50 | 1,485 | 50 | 1–2 in sampled rows |
| 150 | 11,935 | 150 | 1–2 in sampled rows |
| 300 | 46,360 | 300 | 1–4 in sampled rows |
| 350 | 62,835 | 350 | 2–6 in sampled rows |

Across the sampled h11=50–350 rows, `Qtilde` was full rank and its determinant
was small; the observed sample range was 1–40. Do not assume that range for
unseen geometries without measuring the geometry.

### 4.2 Pairwise charge structure

For representative geometries at h11=50, 150, and 350, every one of the
non-base columns matched a pairwise difference of the first `h11+4` base
columns. Specifically, the measured match counts were:

```text
h11=50:  1,431 / 1,431 pairwise columns
h11=150: 11,781 / 11,781 pairwise columns
h11=350: 62,481 / 62,481 pairwise columns
```

The difference orientation in these files is equivalent to `q_j - q_i` for
the base vectors, with the ordering discovered from the data. A structured
evaluator may exploit this only after validating the structure for each
geometry. It must fall back to the generic evaluator if validation fails.

If the same formula holds at h11=491, the inferred instanton count would be
122,760. That number is an extrapolation from the observed structure.

### 4.3 Hierarchy structure

Many high-h11 samples have very strong hierarchy. Examples include leading
log-scale gaps of approximately 272, 340, or more and scale spans of many
millions to billions. Other rows are weak or non-hierarchical, with leading
gaps near zero. For example, sampled h11=150 and h11=300 rows included both
strong and weak cases.

Route these classes differently. Strong hierarchy supports a leading-skeleton
continuation strategy; weak hierarchy requires a more exploratory search and
must not inherit strong-hierarchy completeness claims.

## 5. Why exhaustive branches fail

For a selected lattice determinant `d = abs(det(Qtilde))`, the leading branch
count is:

```text
d * 2^h11
```

At h11=150 with `d=2`, the audited geometry has
`2 * 2^150` branches. At h11=300 with `d=4`, it has `4 * 2^300` branches.
At h11=491, even `d=1` gives approximately `6.4 × 10^147` branches.

Increasing `max_branches` is not a solution. The high-h11 path must search a
scientifically motivated subset or use a different critical-point search.

## 6. Recommended workflow

### Phase A: exact low-index branch streaming

Implement a branch streamer that accepts an index filter or negative-mode
budget without changing the existing complete-enumeration behavior. The
current leading branch code already computes the leading negative-mode count
from the selected signs and half-integer mask. Reuse that convention.

The first search should enumerate only saddles with one leading negative mode;
then add two-negative-mode branches under an explicit budget. Minima are not
inflation candidates and need not be searched in the first discovery pass,
though minima remain useful for calibration.

For a determinant `d`, the number of masks by leading index `k` is:

| leading index | count |
|---:|---:|
| 0 | `d` |
| 1 | `d * h11` |
| 2 | `d * binomial(h11, 2)` |
| 3 | `d * binomial(h11, 3)` |

At h11=491, before lattice copies:

```text
k=1: 491
k=2: 120,295
k=3: 19,608,085
```

This makes k=1 a practical first pass and k=2 a budgeted second pass. Do not
claim that low leading index is sufficient until it is calibrated against
complete enumeration at lower h11.

The filtered streamer must:

- compute branch counts exactly with `BigInt`;
- check the explicit cap before allocation;
- stream one reused `theta` vector;
- preserve deterministic mask/lattice ordering;
- report the index range, masks visited, lattice-copy count, and masks skipped;
- never return zero branches for an overflow or unsupported condition.

### Phase B: structured charge evaluator

Add a validated structured representation for the observed base-plus-pairwise
charge matrices. It should record:

- the `h11+4` base charge vectors;
- the mapping from each pairwise instanton to `(i,j)` and orientation;
- the original instanton order and all `L` amplitudes/log scales;
- a proof/check that every represented column equals its recorded charge;
- a generic fallback flag.

For each branch, evaluate phases from base phases and pair differences, then
assemble the full value, gradient, and Hessian in Float64. Preserve the exact
mathematical potential; this is an execution optimization, not a truncated
potential model.

Acceptance requirements for this evaluator:

- bitwise or numerically tight agreement with the generic evaluator on small
  synthetic cases;
- agreement within a documented tolerance on real h11=50/150/350 samples;
- agreement for value, gradient, Hessian, canonical gradient norm, epsilon,
  and Hessian eigenvalue signs;
- fallback to the generic path if the structure check fails;
- allocation and wall-time measurements for representative h11 values.

This optimization is important because the stored matrices are dense-shaped
but combinatorially sparse/structured. Do not silently convert a failed
structure check into a different scientific model.

### Phase C: Float64 candidate prefilter and correction

For every retained low-index branch:

1. evaluate the full potential, gradient, and Hessian in Float64;
2. reject obvious non-candidates cheaply using value, inertia, gradient norm,
   and eta bounds;
3. for promising points, run a bounded Float64 stationary-point correction
   (Newton/trust-region/eigenvector-following, as appropriate);
4. re-evaluate the corrected point with the full `Q/L` potential;
5. store the original branch metadata, corrected coordinates, residual,
   inertia, epsilon, eta extrema, and solver status.

Do not call an uncorrected leading branch a stationary point merely because it
has a negative Hessian mode. The current branch-point classifier is a screen;
the corrected-point status must be explicit.

### Phase D: hierarchy-aware continuation

For strong-hierarchy geometries, use homotopy continuation:

```text
leading selected potential
  -> add suppressed instanton families in batches
  -> solve/correct the nearby critical point at each stage
  -> retain only branches whose residual and inertia remain valid
```

The homotopy parameter and family ordering must be recorded. A branch that
disappears, merges, changes index, or fails to converge is a diagnostic result,
not silently discarded.

For weak-hierarchy geometries, do not rely on this continuation as a complete
method. Route them to the exploratory strategy in Phase E.

### Phase E: exploratory search for weak hierarchies

Use one or more of the following, with explicit search budgets:

- random or low-discrepancy starts for full-potential critical-point solving;
- eigenvector-following seeded by the best leading-skeleton points;
- gradient-norm minimization followed by Newton correction;
- deflation or torus-distance deduplication of already found points;
- random branch masks stratified by leading index when deterministic low-index
  enumeration exceeds budget.

These are discovery methods, not completeness methods. Record random seeds,
start counts, solver tolerances, iteration limits, deduplication tolerance,
and the number of converged points by Morse index.

### Phase F: arbitrary-precision zoom-in

Only after a Float64 point passes the candidate screen should it enter the
precision path. The future arbitrary-geometry boundary must accept:

- canonical `Q`, `L`, and `K` or a validated structured equivalent;
- Float64 candidate coordinates and branch provenance;
- requested precision;
- gradient/root tolerances;
- Hessian/inertia verification policy;
- trajectory solver method, time horizon, event policy, and tolerances;
- status, retcode, residuals, event diagnostics, allocation/output sizes, and
  failure text.

The current `:n8_poly102` adapter is not a generic implementation of this
boundary. Do not pass arbitrary database geometries into it.

## 7. Calibration and acceptance gates

The low-index hypothesis must be tested before high-h11 production work.

### Gate 1: complete lower-h11 reference

For geometries whose complete branch set fits the budget, compare:

- complete enumeration;
- k=1 only;
- k=1 plus k=2;
- any stochastic/continuation method.

At minimum use h11 values around 8, 11, 13, 14, and 16; include strong and
weak hierarchy examples. If feasible, add h11=17–20 with an appropriately
larger bounded cap.

Measure candidate recall, false-positive rate, index distribution, gradient
residuals, and hierarchy-stratified performance. A low-index method is
eligible for high-h11 use only after its recall is reported, not merely because
it found plausible points.

### Gate 2: structured evaluator agreement

Validate the structured evaluator against the generic evaluator before using
it to rank candidates. Include value/gradient/Hessian comparisons at random
torus points and at retained branch points.

### Gate 3: resource envelope

Measure cold and warm allocations, output sizes, and wall time at h11=50, 100,
150, 200, 300, and 350. The existing script policy is:

- normal tier: h11 ≤ 50;
- middle tier: 51 ≤ h11 ≤ 100;
- high-memory queue: h11 ≥ 101;
- nominal stage allocation ceiling: 750 MB;
- nominal stage output ceiling: 300 MB.

These are policy defaults, not evidence that h11=491 is safe. A high-memory
geometry must be sampled and measured before being admitted to a larger run.

### Gate 4: precision agreement

For every Float64 candidate selected for zoom-in, verify that arbitrary
precision reproduces the point and its Hessian inertia within documented
tolerances. A candidate that changes index or fails the residual check must be
marked as rejected/unstable, not promoted.

## 8. Data and persistence requirements

HDF5 remains the source of large geometry arrays. CSV is for append-only flat
screening/candidate summaries. Keep bulk shards and reports outside Git.

Every candidate or discrepancy must preserve:

- geometry identity and data-root identity;
- contract and diagnostic schema versions;
- orientation and structured-representation validation status;
- selected determinant and exact branch-count estimate;
- branch search class, index budget, masks/lattice copies visited, and random
  seed if applicable;
- Float64 precision, tolerances, solver status, residuals, inertia, epsilon,
  eta extrema, and candidate decision;
- arbitrary-precision precision/tolerances and trajectory event diagnostics
  once refinement exists;
- failure text and resource measurements.

Do not write per-geometry JSON files in the hot path. Use JSON only for
run-level provenance/reference artifacts.

## 9. Required validation commands

Run from the dedicated worktree:

```sh
git branch --show-current
git status --short --branch
git diff --check

julia --project=. --startup-file=no \
  scripts/inflation_scan_contract.jl \
  --data-dir paper_benchmarks/appendix_c \
  --geometry 8,1,1 --max-branches 100000

julia --project=. --startup-file=no test/runtests.jl
```

The full suite requires a normal execution environment where
`Distributed.addprocs` can bind local sockets. The current committed state
passed 324/324 tests. Do not weaken the Distributed test to accommodate a
sandbox restriction.

For bounded real-data checks, write output under `/tmp` or
`/private/tmp`:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_pilot.jl \
  --data-dir ../../data --h11-min 4 --h11-max 100 \
  --sample-per-h11 3 --max-geometries 30 --max-branches 100000 \
  --shard-dir /tmp/inflation-candidate-pilot/shards \
  --report /tmp/inflation-candidate-pilot/report.csv
```

The prior bounded pilot selected 30 geometries, produced 7 successful screens,
23 branch caps, zero empty enumerations, zero failures, and zero candidates.
That result is a bounded sample and does not establish corpus-level absence of
candidates.

## 10. Stop conditions and handoff deliverables

Stop and report before continuing if:

- the structured charge check fails for any production-target geometry;
- the filtered branch count or memory estimate is not exact;
- Float64 and generic evaluators disagree beyond tolerance;
- low-index recall is not measured against complete references;
- a candidate cannot be corrected to a documented Float64 residual;
- arbitrary-precision inertia or trajectory status is unstable;
- any run would exceed the bounded resource policy without explicit review.

The next agent should deliver, in order:

1. focused package/script code for index-filtered branch streaming;
2. structured-evaluator tests and measurements;
3. lower-h11 calibration results with recall by hierarchy/index stratum;
4. a bounded high-h11 candidate pilot with shard-backed provenance;
5. only then, a design or implementation of the arbitrary-geometry precision
   boundary.

Do not claim that a full h11=491 scan is complete unless the report explicitly
defines “complete,” gives the branch/search coverage, and includes the
calibration and resource evidence above.
