# Plan — CYAX-0215

## Approach

This is a brownfield record. The implementation exists at candidate `ab48726`
(PR #215). The plan therefore maps each requirement to its implementation and
evidence, and defines the path to approval and merge. It does not plan new
implementation.

## Requirement map

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| R-001 | `src/minimizer.jl`: `minimize` objective, `hess!`, `hess` | `test/runtests.jl` "Phase-hoisted Hessian parity" (includes the end-to-end minimum) |
| R-002 | `src/minimizer.jl`: `minimize` (`x_abstol`, `g_tol`) | `test/runtests.jl` "minimize resolves stationarity relative to the instanton scales" |
| R-003 | `src/minimizer.jl`: `minimize` (`"stationary"`) | same testset (hierarchical, constant, mixed weak direction) |
| R-004 | `src/minimizer.jl`: `_is_converged_minimum`, `subspace_minimize` | `test/runtests.jl` "subspace_minimize keeps only converged minima with Hsign == 1" |
| R-005 | `src/minimizer.jl`: `subspace_minimize` origin rule | same testset (degenerate, constant and rank-deficient cases) |
| R-006 | `src/generate.jl`: `vacua_id_basis` | `test/runtests.jl` "vacua_id_basis handles charges with no dependent columns (2,2,1 geometry)" |
| R-007 | `scripts/vacua_pipeline.jl`: `_legacy_vacua_search` | Integration evidence only: `test/full.jl` "Vacua pipeline persistence and validation" (legacy batch case). The `Int` type contract is not directly tested |
| R-008 | `src/generate.jl`: `spectrum_mode_counts` | `test/runtests.jl` "spectrum_mode_counts partitions a symmetric spectrum" |
| R-009 | `scripts/batch_vacua_pipeline.jl` | `test/full.jl` "Vacua pipeline persistence and validation" (default and help text) |
| R-010 | docstrings, help text, PR #215 description | docs build; text review |

## Path to merge

1. **G0.** An independent Scientific/Numerical Reviewer reviews the draft
   `spec.md` at a named commit SHA. The owner then approves that exact
   revision in a PR #215 comment that names the SHA.
2. **If G0 amends a decision:** change the implementation, repeat the break
   checks and suites, and update the implementation candidate.
3. **Approval recording.** A commit that touches only this directory sets
   `status: approved`, `last_reviewed` and `approval_ref` (the G0 comment). It
   changes no requirement and no implementation. Its result is the final merge
   candidate.
4. **G1.** Confirm the evidence in the PR #215 description is on the
   implementation candidate. Confirm with
   `git diff --stat <implementation candidate>..<final merge candidate>` that
   only this directory changed, and that `spec.md` differs from the reviewed
   revision only in its approval metadata. Confirm CI passes on the final
   merge candidate.
5. **G2.** Independent review of the entire final merge candidate (the whole
   PR #215 diff against `vmm`). Then the owner decides the merge into `vmm`.

## Escalation

Stop for owner direction if review finds a decision not listed in R-001 to
R-010, or a requirement whose evidence does not support it.
