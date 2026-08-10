# Handoff: scale-continuation and catastrophe pilot

Date: 2026-08-07

## 1. Mission

Determine whether the geometries that returned zero fixed-point inflation
candidates contain nearby, scale-tuned hilltop or catastrophe structures that
the current fixed-potential screen cannot see.

The immediate deliverable is a bounded diagnostic pilot, not a production
high-`h11` scan and not a claim that a near-catastrophe point is an inflationary
trajectory. Keep the following statuses separate:

1. `near_catastrophe`: a corrected critical branch approaches a zero generalized
   Hessian eigenvalue under a documented scale continuation;
2. `screen_candidate`: the nearby detuned branch passes the existing Float64
   screen;
3. `refined_candidate`: the branch remains valid after bounded stationary-point
   correction and residual/inertia checks;
4. `trajectory_candidate`: a model-supported trajectory refinement enters the
   specified slow-roll window.

## 2. Repository and authoritative context

Work only in:

```text
/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl.worktrees/inflation-reproduction-instructions
```

Expected branch:

```text
agents/inflation-reproduction-instructions
```

The geometry data root is:

```text
/Users/vmehta/Documents/CYAxiverse/cyaxiverse/data
```

Read these before changing the scan contract:

- [`HANDOFF_HIGH_H11_CANDIDATE_SEARCH.md`](HANDOFF_HIGH_H11_CANDIDATE_SEARCH.md)
- [`HANDOFF_INFLATION_SCAN_POST_PILOT.md`](HANDOFF_INFLATION_SCAN_POST_PILOT.md)
- [`validation/inflation_scan_call_contract.md`](validation/inflation_scan_call_contract.md)
- [`validation/inflation_high_h11_candidate_search.md`](validation/inflation_high_h11_candidate_search.md)
- [`src/paper_benchmarks/reduced_models.jl`](src/paper_benchmarks/reduced_models.jl)
- [`src/paper_benchmarks/poly102_inflation.jl`](src/paper_benchmarks/poly102_inflation.jl)

The existing scan-prep path is script-local and lives in
`scripts/inflation_scan_common.jl`, driven by
`scripts/inflation_scan_prep.jl`. Reuse its orientation, derivative, branch,
classification, measurement, and persistence conventions. Do not silently
change the locked fixed-point scan while adding this pilot.

## 3. Current evidence and why this pilot is needed

The current scan evaluates one supplied potential per geometry. It does not
continue critical points in a scale parameter, solve an augmented
`gradient=0`/`zero-mode=0` system, or search for Hessian zero-crossings.

The checked-in 2023-minima candidate summary reports zero screened candidates.
The lower-`h11` calibration also returned zero candidates for both complete and
low-index searches. Its recall is therefore `0/0`, not perfect recall. See
[`validation/inflation_high_h11_candidate_search.md`](validation/inflation_high_h11_candidate_search.md)
for the exact strong/weak geometry rows and branch counts.

The positive fixed examples are deliberately tuned benchmarks, not discovered
generic scan hits:

- The N=5 reduced model has a catastrophe at
  `k_c = 0.674506370003365`, where its reduced ratio is `1/4` and one curvature
  vanishes.
- The N=8 poly-102 benchmark has a supplied degenerate point at approximately
  the same `k_c`, with vanishing gradient, one null Hessian mode, and positive
  heavy modes. Detuning the scale produces a shallow one-negative-mode hilltop.
- Both examples use a reduced light direction and an extremely small initial
  displacement. The N=5 e-fold values are one-dimensional reduced-model
  anchors; they are not an independent eight-field trajectory reproduction.

These examples suggest a search hypothesis: the useful signal is a branch
approaching a generalized-Hessian zero crossing, not merely a strong instanton
hierarchy. Strong and weak hierarchy strata must both be retained.

## 4. Non-negotiable scientific conventions

Use the canonical oriented `Q`, `L`, and `K` returned by
`CYAxiverse.read.oriented_potential`. Preserve the package coordinate and
periodicity convention; do not introduce an independent angle normalization.

At each corrected critical point, form the generalized Hessian problem

```text
H_theta * v_i = m_i^2 * K * v_i
```

and report the canonical/generalized eigenvalues consistently. The existing
screen uses:

```text
value > 0
negative_modes > 0
epsilon < 1
abs(min_eta) < 1
```

where `eta_i = m_i^2 / value`. These are screening conditions only. A branch
must not be called stationary or physically inflationary until its full
potential gradient residual, correction status, and trajectory status are
recorded.

For a catastrophe diagnostic, require all of the following to be explicit:

- scale parameter and whether it is physical or only a homotopy parameter;
- scale bracket or grid used;
- branch seed and branch-matching rule;
- correction residual;
- normalized null-vector residual if the augmented solve is used;
- Hessian inertia on both sides of the crossing;
- value, gradient norm, epsilon, eta extrema, and zeroish-mode count.

## 5. First scope: bounded calibration set

Start with the ten already audited geometries, not with `h11 >= 50`:

```text
strong hierarchy:
  (8,544,1), (11,438,1), (13,405,1), (14,443,1), (16,84,1)

weak comparison:
  (8,1,1), (11,1,1), (13,1,1), (14,1,1), (16,1,1)
```

These rows were selected specifically because complete enumeration and the
`1:1`/`1:2` low-index searches are already available for them. Use complete
enumeration wherever its measured branch count fits the bounded pilot budget;
otherwise record the exact branch-cap status and do not infer completeness.

Before generic geometries, add regression checks for the fixed N=5 and N=8
benchmarks. The continuation code should recover the known critical-scale
behavior and mode crossing to the documented tolerances.

## 6. Resolve scale validity before implementing continuation

Do not assume that multiplying all log-amplitudes, divisor volumes, or the
kinetic matrix is a physically valid operation for an arbitrary stored
geometry.

First identify the scale represented by the input data:

### Case A: physical scale is available

If the geometry metadata and potential construction expose a volume/scale
parameter with a documented transformation of `L` and `K`, implement that
transformation in one testable helper. Record the metadata and use this as a
physical continuation.

### Case B: only a mathematical homotopy is available

It is acceptable to use a common scale homotopy to diagnose whether a
catastrophe-like structure exists, but label every result
`scale_status=homotopy_only`. Such a result is not a physical inflation
candidate and must not enter Stage 3 trajectory refinement.

### Case C: scale model is unknown

Stop the continuation implementation, document the missing model boundary, and
ask for the correct physical convention. Do not invent a rescaling rule.

The fixed N=5/N=8 benchmark transformations may be used for benchmark tests,
but must not be generalized to arbitrary geometries without a documented
derivation.

## 7. Proposed pilot algorithm

Implement a new script-level pilot, preferably
`scripts/inflation_scale_continuation.jl`, with reusable helpers in a new
script-local common file if needed. Keep package-level changes out of this
pilot unless an existing numerical boundary is demonstrably insufficient.

### Step 1: prepare scale samples

For each selected geometry, evaluate a small configurable scale grid around
the supplied point. An initial diagnostic grid may be

```text
0.90, 0.95, 0.99, 1.00, 1.01, 1.05, 1.10
```

relative to the physical or homotopy reference scale. Do not present this
range as complete; make it a command-line/configuration input and report it.

### Step 2: obtain branch seeds

Use the existing leading-branch streamer for the first pass. Include the
complete branch set for the selected lower-`h11` rows when it fits the budget,
then compare with `negative_mode_range=1:1` and `1:2`.

Leading branch coordinates are seeds, not corrected stationary points. Preserve
their original branch metadata and index estimate.

### Step 3: correct and match branches

At each scale, run a bounded Float64 stationary-point correction from each
retained seed. Use a periodic-distance-aware branch matching rule between
adjacent scales. The rule and tolerances must be serialized.

For every correction, record one of:

```text
converged, residual_failed, inertia_failed, duplicate, unsupported, failed
```

Never silently discard a branch that merges, disappears, changes inertia, or
fails correction.

### Step 4: detect catastrophe-like behavior

Flag a bracket as `near_catastrophe` if a matched corrected branch has either:

- a smallest generalized Hessian eigenvalue that changes sign across adjacent
  scales; or
- a documented near-zero minimum eigenvalue and a branch/minima-count change.

For a promising bracket, optionally solve the augmented system

```text
gradient(theta, scale) = 0
H(theta, scale) * v = 0
dot(v, v) = 1
```

using the N=8 benchmark implementation as the numerical pattern. The generic
solver must report residuals and convergence; do not call an unverified root a
catastrophe.

### Step 5: test detuned branches

Evaluate one or more points on both sides of a confirmed or suspected
crossing. Apply the existing screen and record whether the branch has exactly
one negative mode, positive value, and small normalized tachyonic curvature.

The result should distinguish:

```text
near-catastrophe only
screen candidate
corrected candidate
unsupported physical interpretation
```

Run trajectory refinement only for a very small number of corrected candidates
and only when the scale is physical and the model is supported by the current
refinement boundary.

## 8. Output contract

Keep bulk output outside Git. Write an append-only summary and, if needed,
separate branch rows. Every row must preserve:

- geometry identity and data-root identity;
- reference scale, sampled scale, scale source, and scale status;
- hierarchy diagnostics;
- search mode and exact branch coverage/cap status;
- leading branch seed/index metadata;
- correction status, residual, and iteration count;
- branch matching/provenance identifier;
- value, gradient norm, epsilon, eta extrema, Hessian eigenvalue extrema;
- negative/zeroish/positive mode counts;
- crossing bracket and catastrophe status;
- candidate status and reason;
- wall time, allocations, output size, and failure message.

Do not overwrite the existing fixed-point scan CSVs. Use a new report such as
`/private/tmp/inflation-scale-continuation/report.csv` and a separate shard
directory. If a checked-in validation note is added, use
`validation/inflation_scale_continuation_pilot.md` and state clearly whether
the result is physical or homotopy-only.

## 9. Validation and acceptance gates

### Gate 1: fixed benchmark recovery

The pilot must reproduce the known N=5/N=8 benchmark signatures:

- N=5 critical scale `0.674506370003365` and reduced ratio `0.25`;
- a vanishing reduced curvature at the N=5 catastrophe;
- N=8 critical scale near `0.674506370003365`;
- N=8 gradient/null residuals within the existing benchmark tolerances;
- one near-zero N=8 mode with positive heavy modes at the degenerate point;
- one-negative-mode detuned branches on the appropriate side.

### Gate 2: fixed-point regression

At scale `1.0`, the continuation pilot must agree with the existing scan-prep
classifier for the ten selected geometries, up to documented correction and
orientation tolerances. It must not turn a fixed-point zero into a candidate
without recording the changed scale or corrected point.

### Gate 3: coverage and resource accounting

Report, stratified by hierarchy and `h11`:

- geometries attempted and completed;
- scale samples per geometry;
- branches seeded, corrected, matched, and lost;
- near-catastrophe brackets;
- screen candidates and corrected candidates;
- branch caps and failures;
- wall time, allocations, output bytes, and peak memory if available.

Do not run the pilot on `h11=491`; no actual `h11_491/*/cyax.h5` geometry group
exists in the local data root. Keep the nominal 750 MB stage-allocation policy
and bounded worker behavior from the existing handoff.

### Gate 4: interpretation

If no crossings are found, report “no crossings in the tested scale window,”
not “the corpus has no inflation.” If crossings are found only in a homotopy,
report them as diagnostic opportunities and do not promote them to physical
candidates.

There is no valid generic candidate-recall estimate yet: the existing
calibration is `0/0`. Preserve that limitation in the final report.

## 10. Suggested commands and tests

Use the repository environment and disable implicit startup state:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scale_continuation.jl \
  --data-dir ../../data \
  --geometry 8,544,1 --geometry 8,1,1 \
  --scale-grid 0.90,0.95,0.99,1.00,1.01,1.05,1.10 \
  --max-branches 400000 \
  --shard-dir /private/tmp/inflation-scale-continuation/shards \
  --report /private/tmp/inflation-scale-continuation/report.csv
```

The exact CLI may change, but it must expose the scale grid, branch budget,
geometry selection, output paths, and physical-vs-homotopy status.

Add focused tests for:

- scale transformation validation and rejection of unsupported metadata;
- periodic branch matching;
- inertia and zero-crossing detection;
- augmented null-vector residuals;
- benchmark N=5/N=8 recovery;
- fixed-point regression at scale `1.0`;
- append-only report schema and resume behavior if implemented.

Run at minimum:

```sh
git diff --check
julia --project=. --startup-file=no test/runtests.jl
```

## 11. Stop conditions

Stop and hand off for clarification if:

- the generic geometry scale transformation cannot be justified;
- the correction changes the scientific potential or coordinate convention;
- branch matching is ambiguous and no deterministic rule resolves it;
- a purported candidate is physical only after an undocumented rescaling;
- the pilot exceeds the resource envelope or starts materializing an
  unbounded branch set;
- a result would require arbitrary-precision trajectory integration during
  scan preparation.

## 12. Final handoff deliverables

The next agent should return:

1. the continuation script and focused tests;
2. a validation report with benchmark recovery and the ten-geometry pilot;
3. append-only CSV/shard schema documentation;
4. a table of near-catastrophe brackets and candidate statuses;
5. explicit physical-vs-homotopy interpretation;
6. resource measurements and any blocked scientific boundary;
7. a recommendation whether to proceed to a larger geometry sample.

The default recommendation after this pilot is: expand only if the method
recovers the fixed benchmarks, finds reproducible corrected crossings in the
generic sample, and stays within the measured resource envelope.
