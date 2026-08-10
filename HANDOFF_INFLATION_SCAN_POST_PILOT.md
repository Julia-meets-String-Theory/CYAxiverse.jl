# Handoff: inflation scan after Stage 6 pilot

Date: 2026-08-07

This handoff continues the bounded inflation-scan work after completion of the
six-stage plan. It is the authoritative continuation document for scan
development. Preserve the scientific and engineering conventions below before
making further changes.

## Mission and scope

The immediate goal is to turn the measured bounded screening path into a safe,
candidate-aware scan workflow. The next work is not an O(10^5) production run
and is not a generic arbitrary-geometry trajectory solver.

The next priorities are:

1. Audit high-dimensional geometries that currently return `success` with zero
   streamed branches.
2. Establish explicit branch, memory, and refinement eligibility policies.
3. Run a second, candidate-focused real-geometry pilot.
4. Begin the planned promotion of reusable numerical boundaries from scripts
   into package APIs, without promoting scan orchestration or policy.

Do not broaden this into a full repository overhaul. Changes should remain
within the inflation screening/refinement path and directly related API work.

## Repository and worktree

- Branch: `agents/inflation-reproduction-instructions`
- Worktree:
  `/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl.worktrees/inflation-reproduction-instructions`
- Data corpus used for the real pilot: `../../data`
- Main worktree on branch `vmm` is separate; do not edit it for this work.
- Implementation base commit: `6441e00 add stratified inflation scan pilot`
- Previous Stage 5 commit: `57ac75e add inflation scan result shards`

The historical
`INFLATION_REPRODUCTION_CHECKPOINT.md` is preserved as a tracked file in
commit `38d3a3d`. Preserve that file. It records the earlier
trajectory-reproduction continuation; do not delete, reset, or overwrite it.
This document supersedes its scan-specific status.

## Completed implementation

The six-stage scan plan is implemented at script level:

| Stage | Current implementation | Status |
|---|---|---|
| 1. Geometry selection and HDF5 loading | `scripts/inflation_scan_prep.jl` and `CYAxiverse.read.potential` | bounded path complete; legacy discovery still materializes its selected index list |
| 2. Float64 screening | `scripts/inflation_scan_common.jl` | complete and contract-locked |
| 3. Optional refinement | `scripts/inflation_refinement_common.jl` | model-specific `:n8_poly102` only; not arbitrary geometry |
| 4. Diagnostics | `scripts/inflation_diagnostics_common.jl` | complete; coherent timing/allocation/output measurements |
| 5. Shards/checkpointing | `scripts/inflation_scan_shards_common.jl` | complete for one-writer-per-shard invocations |
| 6. Real pilot | `scripts/inflation_scan_pilot.jl` | complete as a bounded screening pilot |

Important boundaries:

- There is no package-level `inflation_scan` API.
- `run_geometry` remains script orchestration and must remain script-local for
  now.
- Stage 5 does not create an automatic worker pool. Separate invocations use
  deterministic shard partitions; each process owns one shard file.
- Stage 3 does not accept arbitrary scan-prep `Q/L/K` geometries.
- No arbitrary-precision trajectory is run during scan-prep or the real pilot.

## Locked scientific and data conventions

Do not change these conventions without an explicit scientific review and a
contract/test update.

### Geometry orientation

For every geometry:

- `Q` is `h11 × n_instantons` (axions by instantons).
- `L` is `2 × n_instantons`.
- `K` is `h11 × h11`.
- Raw angle coordinates are in radians.
- Canonical/mass-basis trajectory conventions remain those recorded by the
  reproduction fixtures; the scan-prep classifier is only a Float64 screen.

`_oriented_potential` validates and normalizes the loaded orientation before
any screening call. Do not create a second competing orientation convention.

### Locked screening call sequence

For one `GeometryIndex(h11, polytope, frst)`, preserve this order:

1. `CYAxiverse.read.potential(geom_idx)` loads `L`, `Q`, and `K`.
2. Normalize to the canonical orientations above.
3. Call `CYAxiverse.generate.LQtilde(Q, L)` exactly once.
4. Call `CYAxiverse.generate.instanton_hierarchy_diagnostics(L)`.
5. Factor `K` once with Cholesky and reuse that factorization.
6. Call `CYAxiverse.generate.leading_hessian_mass_basis_float64` for the
   Float64 screening diagnostic.
7. Stream
   `CYAxiverse.generate.foreach_leading_critical_branch(selected;
   max_branches=...)`.
8. Evaluate full `Q/L` derivatives through the reusable log-shifted
   derivative workspace and classify streamed branches.

Do not materialize the full leading-branch coordinate matrix in the scan path.
Do not insert arbitrary-precision trajectory integration into this sequence.

### Status meanings

For scan-prep rows:

- `success`: all bounded screening stages completed. This does not mean an
  inflation trajectory exists or that a candidate was found.
- `branch_cap`: the explicit branch bound prevented enumeration. This is an
  expected bounded outcome, not an unexpected numerical failure.
- `failed`: an unexpected load or numerical error occurred.

The high-dimensional audit below may introduce an explicit empty-enumeration
status. Do not silently reinterpret `success` until that policy is decided and
tested.

For Stage 3 refinement rows, preserve the existing statuses:
`:not_selected`, `:unsupported_model`, `:completed`, and `:failed`.

### Diagnostics

- Scan contract version is currently `"3"`.
- Diagnostic schema version is currently `"1"`.
- Measurement scopes are `:cold`, `:warm`, and `:unspecified`.
- The first geometry in a process is cold; subsequent geometries are warm.
- `stage_allocated_bytes` is cumulative allocation measured by each stage, not
  retained memory.
- `stage_output_bytes` is the shallow/recursive summary size of stage results;
  it is not the same quantity as allocation.
- `GC.gc(false)` is performed before shared stage measurements.
- Preserve solver status, event policy, precision, tolerances, and failure text
  whenever trajectory refinement is measured.

### Persistence

- HDF5 remains the source of large numerical geometry data.
- CSV is for flat, append-only per-geometry or per-candidate summaries.
- Each shard has one writer and is flushed after every row.
- Shard rows retain attempts; failed attempts must not be overwritten.
- `--resume` skips only matching terminal `success` or `branch_cap` rows with
  the same contract/data-root/branch-cap configuration.
- The deterministic merge preserves all attempt rows.
- JSON is reserved for structured run-level provenance/reference artifacts;
  do not add per-geometry JSON files to the hot path.
- Julia `.jls` fixtures are binary Julia reproduction fixtures, not scan
  interchange output.

## Stage 6 pilot findings

The pilot command was:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_pilot.jl \
  --data-dir ../../data \
  --sample-per-h11 1 --max-geometries 20 --max-branches 100000 \
  --shard-dir /private/tmp/inflation-pilot-real.AJpGcG/shards \
  --report /private/tmp/inflation-pilot-real.AJpGcG/report.csv \
  --run-id stage6-real-20
```

Selected h11 values:

```text
4, 7, 9, 12, 15, 17, 20, 23, 25, 28, 31, 34, 36, 39, 42, 44, 47, 50, 150, 300
```

Terminal pilot outcome after resuming the two initially failed rows:

| Measurement | Value |
|---|---:|
| Successful screens | 7 |
| Explicit branch-cap rows | 13 |
| Unexpected failures | 0 |
| Total screening wall time | 6.106 s |
| Mean screening wall time | 0.305 s |
| Total stage allocations | 962,636,720 bytes |
| Mean stage allocations | 48,131,836 bytes |
| Maximum streamed branch count | 32,768 |

All 20 sampled geometries had zero candidate slow-roll saddles. The only
strong-hierarchy sample was h11=15. h11=150 and h11=300 completed with zero
streamed leading branches, while using approximately 100 MB / 29.6 MB and
589 MB / 226 MB allocated/output bytes respectively.

The pilot exposed an integer-overflow bookkeeping bug: `2^h11` was evaluated
as an `Int` and became zero for h11=150/300. The normalization now uses
`BigInt` in `_leading_branch_det_qtilde`. Keep the regression tests for this
case.

Full package validation after the Stage 6 commit passed **298/298 tests**.
The vacua worker test prints an intentional `KeyError` for an incomplete
synthetic HDF5 file and then records that failure; its testset still passes.

Detailed pilot results are in `validation/inflation_scan_stage6_pilot.md`.

## Immediate next work, in order

### 1. Audit empty high-dimensional enumeration

Before any larger pilot, inspect h11=150 and h11=300 directly:

- Record `size(selected.Qtilde)`, `size(selected.Qbar)`, rank, determinant,
  branch-estimate inputs, and the exact branch enumerator return path.
- Determine whether zero streamed branches means a mathematically empty
  leading set, a reduced representation with no eligible leading branches, or
  an unreported cap/unsupported condition.
- Compare against a small geometry at h11=15 with known nonzero branch count.
- Add an explicit result/status field if an empty enumeration is a meaningful
  bounded outcome. Do not label it `failed` unless it is an error.
- Preserve the BigInt normalization and add a high-h11 regression test.

This is the immediate gate. Do not connect physical refinement to a row until
this classification is unambiguous.

### 2. Define resource and eligibility policy

Use the measured pilot to formalize, in the script contract:

- which h11/branch-count tiers are eligible for full screening;
- when `branch_cap` is terminal and when a higher cap is allowed;
- maximum stage allocation/output thresholds;
- whether h11=150/300 are rejected, sampled separately, or sent to a
  high-memory queue;
- which statuses are eligible for Stage 3 refinement.

Policy belongs in the scan script/configuration, not in low-level package
numerical functions.

### 3. Run a candidate-focused pilot

The first pilot found no slow-roll candidates. Increase sampling in low and
middle h11 groups and deliberately cover hierarchy and instanton-count bins.
Keep the run bounded, shard-backed, and screening-only. Report:

- candidate count and candidate rate;
- branch-cap and empty-enumeration rates;
- warm wall time;
- allocations and output sizes;
- failures by h11 and geometry identity.

Do not infer that zero candidates means the corpus has no candidates from the
20-geometry pilot.

## Planned package API expansion

This is an explicit follow-up task. It must be done as a numerical-boundary
promotion, not as a wholesale script rewrite.

### Promote or design these package boundaries

1. **Canonical potential normalization**

   Consolidate the duplicate logic in
   `scripts/inflation_scan_common.jl::_oriented_potential` and
   `scripts/analyze_inflation_candidates.jl::oriented_potential` into one
   package-owned validation/orientation helper. It should return canonical
   `Q`, `L`, and `K` orientation and validate dimensions/finiteness. It must
   not encode scan thresholds or candidate policy.

2. **Prepared screening context**

   Design a package-owned context or preparation boundary holding immutable
   `Q/L/K`, `LQtilde` results, the Cholesky factor, and caller-owned reusable
   derivative/classification workspaces. The context must make mutable state
   isolation explicit and must not retain all branches.

3. **Canonical point diagnostics**

   Promote the numerical core of `_classify_point!` as a prepared-context or
   caller-workspace API. It should return value, canonical gradient norm,
   epsilon, eta extrema, mass signs, and mode counts. It must not decide
   `epsilon < 1`, `|eta| < 1`, saddle eligibility, or refinement selection;
   those remain script policy.

4. **Streaming branch diagnostic boundary**

   Keep the existing package
   `generate.foreach_leading_critical_branch` streaming primitive. If a new
   package callback is needed, make it a per-branch numerical callback boundary,
   not an aggregate `inflation_scan` API and not a result writer.

5. **Future arbitrary-geometry refinement boundary**

   Only after a validated geometry-specific trajectory representation exists,
   design a model-dispatched refinement API with explicit precision,
   tolerances, solver method, event policy, status, and failure metadata. Do
   not generalize the current `:n8_poly102` adapter prematurely.

### Keep these script-local

- `run_geometry` orchestration and stage ordering;
- screening thresholds and candidate policy;
- branch/memory/resource caps;
- status aggregation and retry policy;
- CSV formatting, shard writers, resume, and deterministic merge;
- h11-stratified pilot selection and report aggregation;
- model-specific benchmark fixtures and basis audits;
- arbitrary-precision solver configuration until the geometry boundary is real.

### API promotion acceptance criteria

Before promoting a function:

- identify all current script callers and avoid duplicate implementations;
- add value/orientation/coordinate regression tests;
- test caller-owned workspace reuse and aliasing/isolation;
- test high-h11 integer safety, especially exponentiation and determinant-like
  bookkeeping;
- measure allocations before and after promotion;
- preserve the locked scan call sequence and CSV schema;
- document the new API in package docs and update the scan contract;
- keep policy and orchestration tests at script level.

Do not add a generic package `inflation_scan` function merely to hide the
workflow. The package should own reusable numerical primitives; the script
should own the experiment definition and execution policy.

## Required validation commands

Run from the worktree above:

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

The full suite requires a normal terminal or execution environment that
allows Julia `Distributed.addprocs` to bind local sockets. In the validated
run, all testsets passed for a total of 298/298 tests. Do not weaken or remove
the Distributed worker test to accommodate a sandbox restriction.

For a bounded real pilot:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_pilot.jl \
  --data-dir ../../data --sample-per-h11 1 --max-geometries 20 \
  --max-branches 100000 --shard-dir /tmp/inflation-pilot-shards \
  --report /tmp/inflation-pilot-report.csv
```

Keep bulk shards and reports outside Git. Commit only compact validation
notes, schema changes, code, and tests.

## Safety and handoff rules

- Never use `git reset --hard`, `git checkout --`, or broad cleanup commands.
- Preserve unrelated worktree modifications and
  `INFLATION_REPRODUCTION_CHECKPOINT.md`.
- Do not overwrite published fixtures or comparison CSVs with unvalidated
  trajectory results.
- Do not launch a large scan from this handoff.
- Do not claim production throughput from the bounded pilot.
- Record geometry identity, `Q`, `L`, `K`, contract versions, branch cap,
  precision/tolerances, solver status, event diagnostics, and failure text for
  every discrepancy.
- Commit coherent, reviewable increments and update this handoff or the
  validation contract when the scan boundary changes.
