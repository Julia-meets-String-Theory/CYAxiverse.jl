# Plan — CYAX-0215

## Approach

This is a brownfield record. The implementation exists at candidate `a6a07da`
(PR #215). It replaced `ab48726`, `dc42633`, `b309a55`, `dbd7b62` and
`4abbd60` after the G0 reviews of drafts `444fcc3`, `6b033b5`, `0442ead`,
`92f4c71`, `898029d` and `033f673`, which did not pass, and after the owner
amendments recorded in `spec.md`, one of which removed condition (2) of
R-003. The review of `92f4c71` required text corrections only, so the
candidate did not change. The review of `898029d` required a help-text
correction, so the candidate changed to `4abbd60`; that change is text only
(help text, docstrings, comments and their test). After the review of
`033f673` the owner kept `--method legacy` as the batch default, so the
candidate changed to `a6a07da`. Against `a6a07da`, the review of `94a61f7`
did not pass, and the reviews of `043cc3f` and `f88be82` passed; the
corrections after them are text only. The plan therefore maps each
requirement to its implementation and evidence, and defines the path to
approval and merge. It does not plan new implementation.

## Requirement map

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| R-001 | `src/minimizer.jl`: `minimize` objective, `hess!`, `hess`; `_phase_hessian!` (shared with `id_minimize(::Vector, ::Matrix)`) | `test/runtests.jl` "Phase-hoisted Hessian parity" (includes the end-to-end minimum); objective value `-16/45` in "subspace_minimize keeps only converged minima with Hsign == 1" |
| R-002 | `src/minimizer.jl`: `minimize` (`x_abstol`, `g_tol`) | `test/runtests.jl` "minimize resolves stationarity relative to the instanton scales" |
| R-003 | `src/minimizer.jl`: `minimize` (`"stationary"`) | same testset (hierarchical and mixed weak direction, which guard condition (3); constant potential, guarded jointly by (1) and (3)); the `L = (1, 1e-6)` regression for the removal of condition (2). Condition (1) is not isolated; condition (3) implies it |
| R-004 | `src/minimizer.jl`: `_is_converged_minimum`, `subspace_minimize` | `test/runtests.jl` "subspace_minimize keeps only converged minima with Hsign == 1". The `converged` conjunct is not isolated. The rank-deficient flat-family behaviour is a recorded probe (`spec.md` R-004), not a test |
| R-005 | `src/minimizer.jl`: `subspace_minimize` origin rule | same testset (degenerate, constant and rank-deficient cases) |
| R-006 | `src/generate.jl`: `vacua_id_basis` | `test/runtests.jl` "vacua_id_basis handles charges with no dependent columns (2,2,1 geometry)" |
| R-007 | `scripts/vacua_pipeline.jl`: `_legacy_vacua_search` | Integration evidence only: `test/full.jl` "Vacua pipeline persistence and validation" (legacy batch case). The `Int` type contract is not directly tested |
| R-008 | `src/generate.jl`: `spectrum_mode_counts` | `test/runtests.jl` "spectrum_mode_counts partitions a symmetric spectrum" |
| R-009 | `scripts/batch_vacua_pipeline.jl`: default kept at legacy, help text | `test/full.jl` "Vacua pipeline persistence and validation" (default and help text) |
| R-010 | docstrings (`minimize`, `subspace_minimize`, `id_minimize(::Vector, ::Matrix)`), help text, PR #215 description | docs build; text review; reachability of the minimizer from the legacy pipeline (call graph and the `vacua_id` cases in `spec.md`) |

## Path to merge

1. **G0.** An independent Scientific/Numerical Reviewer reviews the draft
   `spec.md` at a named commit SHA. The owner then approves that exact
   revision in a PR #215 comment that names the SHA. By owner direction, the
   provenance and editorial corrections after the review of `f88be82` are not
   re-reviewed; the approval names the corrected revision and `f88be82`
   (`spec.md` G0).
2. **If G0 amends a decision, or a blocking finding needs an implementation
   change:** change the implementation, repeat the break checks and suites,
   update the implementation candidate, and repeat T001, T002, T003 and T005
   for it.
3. **Approval recording.** A commit that touches only this directory sets
   `status: approved`, `last_reviewed` and `approval_ref` (the G0 comment). It
   changes no requirement and no implementation. Its result is the final merge
   candidate.
4. **G1.** Confirm the evidence in the PR #215 description is on the
   implementation candidate. Confirm with
   `git diff --stat <implementation candidate>..<final merge candidate>` that
   only this directory changed, and that `spec.md` differs from the approved
   revision only in its approval metadata. Confirm CI passes on the final
   merge candidate.
5. **G2.** Independent review of the entire final merge candidate (the whole
   PR #215 diff against `vmm`). Then the owner decides the merge into `vmm`.

## Escalation

Stop for owner direction if review finds a decision not listed in R-001 to
R-010, or a requirement whose evidence does not support it.
