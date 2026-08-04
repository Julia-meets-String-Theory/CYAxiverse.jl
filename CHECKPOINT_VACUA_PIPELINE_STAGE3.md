# Vacua Pipeline Stage 1-3 Checkpoint

Date: 2026-08-02
Branch: `vacua-pipeline`
Julia: 1.12.6
BLAS threads: 8

## Implementation

- `scripts/vacua_pipeline.jl` validates geometry inputs, records search provenance, protects existing results, writes through a same-directory temporary copy, and preserves `spectrum` and legacy HDF5 groups.
- Search dispatch supports `legacy`, `leading_branches`, and `reduced_jlm`.
- Verification metadata distinguishes `verified`, `verified_selected_branch_set`, and `not_applicable` finite-search results.
- `scripts/batch_vacua_pipeline.jl` provides deterministic selection, bounded budgets, matching-configuration skips, mismatch protection, dry runs, force target listing, incremental CSV summaries, and per-geometry failure capture.
- `src/generate.jl`, `src/read.jl`, and tests retain legacy result fields and readers.

## Validation

Commands:

```sh
julia --project=. test/runtests.jl
julia --project=. scripts/reproduce_2023_n8.jl
git diff --check
```

Results:

- Full Julia suite passed. Persistence passed 22/22; paper/repository benchmarks passed 69/69; all spectrum and orientation testsets passed.
- N=8 checkpoint passed: `k=0.66` gives 1127 critical points and 5 minima; `k=0.68` gives 1065 critical points and 1 minimum.
- Synthetic method dispatch passed: leading branches reports 16 selected branches and `certified_selected_branch_set`; square reduced JLM reports 4 minima, multiplicity 4, and `exact_determinant_branch`; finite legacy search reports `not_applicable` verification.
- Branch enumeration correctly throws when `max_branches` is exceeded.

## Isolated real-data pilot

All pilot files were copied to `/tmp/cyaxiverse-vacua-pilot.ynXqxT`; source HDF5 files were not modified.

- Single geometry: 1/1 completed with reduced JLM, estimate 4, verified 4, 9.26 seconds.
- Ten-geometry h11=33 pilot: 10/10 completed, 0 failed, 10.05 seconds.
- Complete copied h11=33 slice: 100/100 completed, 0 failed, 0 blocked, 21.19 seconds.
- Matching rerun: 0 done, 100 skipped, 0 blocked, 0 failed.
- Method-mismatch dry run: 0 done, 0 skipped, 100 blocked, 0 failed.
- HDF5 audit: copied result contains both `spectrum` and `vacua_pipeline`; status is `completed`, method is `reduced_jlm`, and verification is `verified`.

## Remaining scope

This checkpoint establishes completion of Stages 1-3. The handoff's broader publication checklist still requires the full mandatory Stage 4 comparison matrix, large-geometry/thread-scaling Stage 5 benchmarks, and a fresh-environment test report before publication claims. No full h11=4:50 scan or source-data overwrite was launched.
