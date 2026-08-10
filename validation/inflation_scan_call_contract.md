# Inflation scan call contract

Status: script-level contract only; no generic package scan API is introduced.

The driver is
`scripts/inflation_scan_contract.jl`. It is intentionally the only generic
layer at this stage. The package functions remain responsible for their own
scientific operations.

The bounded scan-prep driver is
`scripts/inflation_scan_prep.jl`. It reuses the same sequence, discovers or
selects geometries in deterministic order, and appends one fixed-schema CSV
row after each geometry. `success` means all preparation stages completed;
`branch_cap` means the explicit enumeration bound prevented branch
enumeration; `failed` means an unexpected load or numerical error. The driver
does not run trajectory refinement or write geometry files.

## Locked sequence

For each `GeometryIndex(h11, polytope, frst)`:

1. `CYAxiverse.read.potential(geom_idx)` returns an `AxionPotential` with
   `L`, `Q`, and `K`. The driver normalizes the orientation to
   `Q :: h11 × n_instantons`, `L :: 2 × n_instantons`, and
   `K :: h11 × h11`.
2. `CYAxiverse.generate.LQtilde(Q, L)` is called exactly once. Its
   `Qtilde`, `Qbar`, `Ltilde`, and `Lbar` are reused downstream.
3. `CYAxiverse.generate.instanton_hierarchy_diagnostics(L)` supplies the
   cheap hierarchy fields.
4. One Cholesky factorization of `K` is reused for canonical-Hessian and
   gradient-norm calculations across all candidate branches.
5. `CYAxiverse.generate.leading_hessian_mass_basis_float64(
   K, selected.Ltilde, selected.Qtilde)` supplies a Float64 mass-basis
   diagnostic. It is a screening diagnostic, not the arbitrary-precision
   trajectory calculation.
6. `CYAxiverse.generate.foreach_leading_critical_branch(
   selected; max_branches=...)` streams the leading branches only when the
   explicit branch cap permits it. The callback reuses one coordinate vector;
   the scan-prep path does not materialize the full branch matrix. An explicit
   `negative_mode_range=1:1` or `1:2` performs deterministic low-index
   enumeration; its return report records the exact full mask count, masks
   visited/skipped, lattice copies, and search classification.
7. The script evaluates the full `Q/L` potential derivatives on each streamed
   branch using a validated structured base-plus-pairwise evaluator when the
   geometry proves that representation, otherwise using the generic reusable
   log-shifted workspace. Classification policy is local to the script; it is
   not currently a package API.

An arbitrary-geometry trajectory/refinement call is deliberately not part of
this contract yet. It must accept a validated geometry-specific representation
and explicitly record solver precision, tolerances, event policy, status, and
failure diagnostics before it is added here. The current model-specific Stage
3 adapter is documented below.

## Sample contract probe

Command:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_contract.jl \
  --data-dir paper_benchmarks/appendix_c \
  --geometry 8,1,1 --max-branches 100000
```

The checked-in Appendix-C sample returned:

```text
status=:success
Q=(8,78), L=(2,78), K=(8,8)
selected_instantons=8
branch_count=256
leading_minima_count=1
candidate_slowroll_saddles=0
```

The first run includes Julia method compilation and is marked
`measurement_scope=cold`; subsequent geometries in the same process are marked
`measurement_scope=warm`. The recorded stage fields remain diagnostics rather
than a production throughput guarantee.

## Scan-prep output and resume contract

Example:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_prep.jl \
  --data-dir paper_benchmarks/appendix_c \
  --geometry 8,1,1 --max-branches 100000 \
  --summary /tmp/inflation_scan_prep.csv
```

The CSV is streamed and flushed after every geometry. It contains the geometry
identity, the explicit branch cap, all locked stage timings and allocation
measurements, and the screening summary fields. `--resume` requires an
existing matching header and skips only rows with the same contract version,
data root, branch cap, and status `success` or `branch_cap`; failed rows are
retried. This is a bounded scan-prep mechanism, not yet the production worker
pool/checkpoint/shard system required for an O(10^5) scan.

`stage_branches_s` remains in the fixed CSV schema for compatibility and is
zero in the streaming path. Branch enumeration and per-branch classification
are measured together in `stage_classify_s`. Contract version 4 also records
the diagnostic schema version, measurement scope, per-stage output sizes,
leading-index search coverage, and structured charge validation/fallback
status.

## Stage 4: coherent diagnostics

The shared script layer is
`scripts/inflation_diagnostics_common.jl`. `inflation_stage_measure` records a
stage status, error text, cold/warm/unspecified scope, wall time, allocated
bytes, and output size. It can capture exceptions without turning a failed
stage into a successful row.

`inflation_refinement_diagnostic_row` flattens the screening record and the
Stage 3 refinement summary into one candidate row. It includes screening
quantities, refinement status and retcode, event outcome, solver counters,
allocation/output diagnostics, and a separately measurable serialization
section. `inflation_append_diagnostic_row` measures formatting plus the
append/flush operation for that row. CSV formatting is kept script-level; no
package diagnostics API is introduced.

## Stage 5: append-only result and checkpoint shards

The Stage 5 persistence layer is
`scripts/inflation_scan_shards_common.jl`. A scan invocation can own one shard
file and append one row per geometry attempt:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_prep.jl \
  --data-dir DATA_ROOT --h11 8 --shard-dir /tmp/inflation-shards \
  --shard-index 1 --shard-count 4 --retries 1 --resume
```

`--shard-index` is one-based. The selected geometries are partitioned in the
same deterministic order used by the single-process driver, so four separate
invocations with indices 1 through 4 produce four worker-local shards. Each
writer validates its fixed header and flushes every row. The row retains the
shard schema version, run label, shard assignment, attempt number, start/end
timestamps, scan contract version, data root, branch cap, geometry identity,
status, error text, and all Stage 4 diagnostics.

`--resume` scans the shard directory and skips only matching `success` or
`branch_cap` terminal rows. Failed rows remain in place and are retried when
selected again; `--retries N` records each failed attempt before the next one.
This makes interruption recoverable without materializing a global result
array. The shard writer is intentionally script-level and has one writer per
file; the package APIs and numerical call sequence are unchanged.

After the workers finish, merge shards deterministically while preserving all
attempt rows:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_merge_shards.jl \
  --shard-dir /tmp/inflation-shards \
  --output /tmp/inflation-scan-merged.csv
```

The current Stage 5 driver covers bounded screening, where the explicit branch
cap is the per-geometry bound. It does not impose a hard wall-clock kill on an
arbitrary-precision trajectory; that remains coupled to the future
geometry-specific refinement worker rather than being hidden in scan-prep.

## Stage 6: stratified real-geometry pilot

The bounded real-data pilot is
`scripts/inflation_scan_pilot.jl`. It samples evenly spaced geometries within
each available `h11` group, applies an optional global cap, and delegates every
geometry call to the Stage 5 shard-backed scan-prep driver. Since instanton
count, hierarchy, and candidate count are screening outputs, those dimensions
are used for post-screen grouping rather than pre-screen selection.

Example:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_pilot.jl \
  --data-dir ../../data --sample-per-h11 1 --max-geometries 20 \
  --max-branches 100000 --shard-dir /tmp/inflation-pilot-shards \
  --report /tmp/inflation-pilot-report.csv
```

The pilot report groups exact `h11` rows by instanton-count bin, strong/not-
strong hierarchy, and candidate-count bin. The measured 20-geometry pilot is
recorded in `validation/inflation_scan_stage6_pilot.md`. It found 7 successful
screens, 13 explicit branch-cap outcomes, and no failures after fixing the
high-dimensional `2^h11` integer-overflow bookkeeping bug. It also showed that
the h11=300 sample produced approximately 589 MB of stage allocations and 226
MB of stage output, reinforcing that high-dimensional screening needs an
explicit resource policy before a large scan.

## High-dimensional enumeration audit and policy

The h11=150 and h11=300 pilot geometries were audited directly after the
integer-safety fix. Their selected charge matrices are full rank with exact
determinants 2 and 4, respectively. The branch counts are therefore
`2 * 2^150` and `4 * 2^300`; with `max_branches=100000`, both now raise the
explicit branch-cap `ArgumentError` before the callback. The earlier zero
streamed branches were an Int overflow in the mask count, not a mathematically
empty leading set. The h11=15 comparison geometry streams 32,768 branches.

The package computes branch estimates with `BigInt` and converts to an Int
mask count only after the cap check. Scan policy remains script-local:

- normal screening tier: `h11 <= 50`;
- middle screening tier: `51 <= h11 <= 100`;
- high-memory queue: `h11 >= 101`;
- Stage 3 refinement requires a successful candidate in the normal or middle
  tier, with at most 750 MB cumulative stage allocation and 300 MB stage
  output.

`inflation_screening_tier`, `inflation_refinement_eligible`, and the exact
lower-bound helper are policy/diagnostic helpers only; they do not alter the
locked numerical call sequence.

## Real-geometry bounded scalability slice

The first real-data slice used the local `../../data` corpus with
`--max-branches 100000`. The first geometry in each process was excluded from
the warm averages because it includes Julia compilation and method-cache work.
`stage_allocated_bytes` is the total of all preparation stages, not retained
memory; the streaming callback prevents the branch-coordinate matrix from
being retained.

| h11 | warm geometries | mean branches | max branches | mean classify s | mean stage bytes | max stage bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 9 | 739.6 | 1,024 | 0.004968 | 2,900,404 | 3,993,824 |
| 10 | 9 | 1,934.2 | 4,096 | 0.017375 | 9,204,213 | 19,367,744 |
| 11 | 4 | 7,168.0 | 10,240 | 0.080929 | 34,513,420 | 49,466,784 |

These are bounded slices, not a full-corpus throughput claim. The remaining
per-branch allocation is the public `LAPACK.syevd!` eigensolver workspace:
on the warm synthetic h11=14 probe it allocated 4,432 bytes per call, while
the derivative and BLAS triangular-solve stages were allocation-free. Reusing
that LAPACK workspace would require a separate, version-sensitive numerical
boundary; it is the next targeted optimization rather than an aggregate scan
API.

## Stage 3: optional high-precision refinement

The Stage 3 boundary is implemented in
`scripts/inflation_refinement_common.jl`. It is intentionally script-level and
model-specific at this point because the validated arbitrary-precision solver
is `n8_poly102`; it does not yet accept an arbitrary scan-prep `Q/L/K` geometry.

An eligible candidate must provide:

- `candidate_id`, for provenance;
- `model=:n8_poly102`, selecting the registered refiner;
- positive `delta_k`, the model-specific detuning; and
- `accepted=true`, supplied by the Float64 screening policy.

`inflation_refinement_config` makes precision, tolerances, displacement,
solver horizon, step limits, sample count, basis, and `maxiters` explicit. The
refiner returns a scalar summary plus the compact trajectory result. The
summary records `:not_selected`, `:unsupported_model`, `:completed`, or
`:failed`, along with the final-finite-exit event policy, solver counters,
wall time, allocated bytes, output size, and error text when applicable.

A bounded 64-bit pilot at `max_time=10` completed with 40 accepted steps,
`entered_slow_roll=true`, `end_event=:tmax`, and 6.6845119358 e-folds. Its
cold allocation was approximately 975 MB; a warmed repeat allocated
180,987,872 bytes in 0.157 seconds, while the compact returned result was
15,936 bytes. A shorter `max_time=0.01` smoke call completed with 18 accepted
steps and no slow-roll window, but still allocated approximately 814 MB cold.
This confirms that the refinement stage must remain opt-in and
candidate-limited. It also means this adapter is a contract/provenance
boundary, not yet a production refiner for the real-geometry corpus.

## Workflow API boundaries and remaining candidates

This is a deliberately short inventory of functions used by the current
inflation reproduction and screening work. It identifies reusable numerical
boundaries while keeping orchestration and policy in the scripts.

| Current script function | Eventual package boundary | Decision for this phase |
| --- | --- | --- |
| Former duplicate orientation helpers in `inflation_scan_common.jl` and `analyze_inflation_candidates.jl` | `CYAxiverse.read.oriented_potential`, returning the package's canonical `Q`, `L`, and `K` orientation | Promoted; both callers now use the package helper, while thresholds and policy remain script-local |
| `inflation_scan_common.jl::_normalized_derivatives` and `analyze_inflation_candidates.jl::derivatives` | `generate.logshifted_derivative_workspace` plus the in-place `generate.logshifted_derivatives!` evaluator | Promoted narrowly; the reusable buffers and numerical stabilization are package-owned, while scan thresholds are not |
| `inflation_scan_common.jl::_classify_point` and `analyze_inflation_candidates.jl::classify_point` | A reusable canonical-gradient and Hessian diagnostic, accepting a caller-owned factorization or prepared context | Strong API candidate; keep `epsilon < 1`, `|eta| < 1`, and saddle-selection policy in the script |
| `inflation_scan_common.jl::_classify_branches` | `generate.foreach_leading_critical_branch` for streaming, plus per-branch diagnostics later; not an aggregate `scan` API | Streaming branch enumeration promoted; aggregate classification policy remains script-local |
| `inflation_scan_common.jl::run_geometry` | No package equivalent: this is orchestration across loading, screening, timing, and failure policy | Keep script-only |
| `analyze_inflation_candidates.jl::reduced_solve` | No new API: it is a thin wrapper around `leading_critical_branches` | Keep script-only |
| `inflation_reproduction.jl::n5_fixture` and `n8_fixture` | No generic package API; these assemble benchmark-specific serialized fixtures | Keep benchmark script-only |
| `inflation_reproduction.jl::print_basis_audit` | Possibly a benchmark-specific `basis_audit` helper under `paper_benchmarks` if the audit is reused | Defer; do not generalize during scan-prep |
| `inflation_refinement_common.jl::refine_inflation_candidate` | A future model-dispatched refinement boundary with explicit candidate and solver metadata | Implemented script-level for `:n8_poly102`; do not promote until arbitrary-geometry trajectory inputs exist |
| `benchmark_inflation_scalability.jl::trajectory` | A future model-specific trajectory entry point with explicit solver/event metadata | Keep under `paper_benchmarks`; it is not a generic geometry scan API |

The current implementation shares the locked call sequence between the
contract probe and `scripts/inflation_scan_prep.jl`, and promotes only the
canonical orientation boundary in addition to the existing derivative
workspace and streaming branch primitives. Unrelated batch, persistence, and
plotting helpers remain outside this inventory.

The derivative workspace and streaming branch enumerator are deliberate narrow
exceptions: they are reusable numerical primitives now used by the script,
with focused value/coordinate regression tests. The per-branch eigensolver
allocation remains a measured follow-up target; it is not hidden by these APIs.
