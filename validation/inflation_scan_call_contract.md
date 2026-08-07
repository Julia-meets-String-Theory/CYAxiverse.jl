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
`branch_cap` means the explicit materialization bound prevented branch
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
   the scan-prep path does not materialize the full branch matrix.
7. The script evaluates the full `Q/L` potential derivatives on each streamed
   branch using the package's reusable log-shifted workspace. Classification
   policy is local to the script; it is not currently a package API.

The generic trajectory/refinement call is deliberately not part of this
contract yet. It must accept a validated geometry-specific representation and
explicitly record solver precision, tolerances, event policy, status, and
failure diagnostics before it is added here.

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

The first run includes Julia method compilation. The recorded stage fields are
diagnostic only until a warmup policy is added to the eventual scan driver.

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
are measured together in `stage_classify_s`.

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

## Workflow API candidates for a later overhaul

This is a deliberately short inventory of functions used by the current
inflation reproduction and screening work. It identifies reusable numerical
boundaries without changing the package in this scan-prep phase.

| Current script function | Eventual package boundary | Decision for this phase |
| --- | --- | --- |
| `inflation_scan_common.jl::_oriented_potential` and `analyze_inflation_candidates.jl::oriented_potential` | A central input-normalization and validation function returning the package's canonical `Q`, `L`, and `K` orientation | Keep script-local; avoid two independent implementations when the API work begins |
| `inflation_scan_common.jl::_normalized_derivatives` and `analyze_inflation_candidates.jl::derivatives` | `generate.logshifted_derivative_workspace` plus the in-place `generate.logshifted_derivatives!` evaluator | Promoted narrowly; the reusable buffers and numerical stabilization are package-owned, while scan thresholds are not |
| `inflation_scan_common.jl::_classify_point` and `analyze_inflation_candidates.jl::classify_point` | A reusable canonical-gradient and Hessian diagnostic, accepting a caller-owned factorization or prepared context | Strong API candidate; keep `epsilon < 1`, `|eta| < 1`, and saddle-selection policy in the script |
| `inflation_scan_common.jl::_classify_branches` | `generate.foreach_leading_critical_branch` for streaming, plus per-branch diagnostics later; not an aggregate `scan` API | Streaming branch enumeration promoted; aggregate classification policy remains script-local |
| `inflation_scan_common.jl::run_geometry` | No package equivalent: this is orchestration across loading, screening, timing, and failure policy | Keep script-only |
| `analyze_inflation_candidates.jl::reduced_solve` | No new API: it is a thin wrapper around `leading_critical_branches` | Keep script-only |
| `inflation_reproduction.jl::n5_fixture` and `n8_fixture` | No generic package API; these assemble benchmark-specific serialized fixtures | Keep benchmark script-only |
| `inflation_reproduction.jl::print_basis_audit` | Possibly a benchmark-specific `basis_audit` helper under `paper_benchmarks` if the audit is reused | Defer; do not generalize during scan-prep |
| `benchmark_inflation_scalability.jl::trajectory` | A future model-specific trajectory entry point with explicit solver/event metadata | Keep under `paper_benchmarks`; it is not a generic geometry scan API |

The immediate implementation therefore only shares the locked call sequence
between the contract probe and `scripts/inflation_scan_prep.jl`. No candidate
other than the derivative workspace and streaming branch enumerator is promoted
into the package yet, and
unrelated batch, persistence, and plotting helpers are outside this inventory.

The derivative workspace and streaming branch enumerator are deliberate narrow
exceptions: they are reusable numerical primitives now used by the script,
with focused value/coordinate regression tests. The per-branch eigensolver
allocation remains a measured follow-up target; it is not hidden by these APIs.
