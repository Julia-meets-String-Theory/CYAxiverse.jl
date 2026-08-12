# CODE-03 Spec Review

Review date: 2026-08-12  
Fixed point: `vmm` merge commit `81e43c4`  
SCI-06 source: `validation/sci06_hierarchy_window_validation.md`  
Priority source: `handoffs_checkpoints/PRIORITIES_2_3_COPILOT_HANDOFF.md`

## Required review frame

CODE-03 asks whether the approved SCI-06 hierarchy-aware and mass-window
feature was rebased onto `vmm` without changing validated numerical semantics.
The review checks the public API, existing spectrum readers/constructors,
indices, boundaries, signs, quartic behavior, fallback state, validation, and
performance evidence.

## Specification checklist

| Requirement | Review result | Evidence |
| --- | --- | --- |
| Full-window equality with the arbitrary-precision reference | Pass | SCI-06 validation matrix and focused tests |
| Lower-threshold compatibility with the hybrid API | Pass | SCI-06 validation matrix and focused tests |
| Interior, empty, and exact-boundary windows | Pass | Focused window-edge testset |
| Permutation invariance and extreme scale spans | Pass | Focused validation testset |
| Nearly degenerate and strongly mixed-charge fallback behavior | Pass | Hierarchy diagnostics and fallback tests |
| Zero-based mode indices and inclusive boundaries | Pass | API tests and user-guide contract |
| Mass offsets, signs, quartic/self-coupling semantics | Pass | PQ/HP comparison and window coupling tests |
| Concrete input validation and result types | Pass | Dimension/domain tests and typed result structs |
| Existing constructors and HDF5 readers | Preserved | No persistence-schema changes in CODE-03 integration |
| Targeted-window performance claim | Pass with scope | Synthetic benchmark: 1.58x wall-time reduction; higher allocations documented |
| Visible provisional/fallback status | Pass | Diagnostics retain fallback and provisional fields |

## Numerical semantics

The integration preserves the SCI-06 convention

\[
W = K_L^{-1} H K_L^{-T}, \qquad
H = \sum_a \Lambda_a^4 q_a q_a^T,
\]

with arbitrary-precision inertia for the two window boundaries and the full
high-precision eigensystem as the reference fallback. No threshold, boundary
margin, mass normalization, sign treatment, or physical-window definition was
changed during CODE-03 integration.

## Follow-up CI correction

The merged PR's Linux Julia 1.12 job failed before completing the suite in the
`Geometry-level LQtilde orientation` test. The failure was a `BoundsError` in
`read._canonicalize_generated_potential` while matching leading charge
columns. The follow-up changes only the implementation of that internal
column-matching loop; the canonical representatives, coefficient checks, raw
opt-out, and returned orientation are unchanged.

## Scientific disposition

CODE-03's spectrum specification is satisfied after the mechanical CI fix.
The SCI-06 scientific limitations remain in force: the certificate is a
conservative numerical screen rather than a theorem, the benchmark is not a
production high-`h11` scaling claim, and no population-level, Bayesian,
inflationary, or black-hole-superradiance conclusion is implied.

## Verification record

The follow-up gates were run on Julia 1.12.6:

| Check | Result |
| --- | --- |
| Focused `read.oriented_potential` reproduction | Pass; returned the expected canonical `(Q, L, K)` tuple |
| `julia --startup-file=no --project=. -e 'using Pkg; Pkg.test()'` | Pass; all package testsets passed, including `Geometry-level LQtilde orientation` (14/14) |
| `julia --startup-file=no --project=. bin/audit.jl` | Pass; JET, Aqua, and physical-potential sanity checks all passed |
| `env -u PYTHON julia --compiled-modules=no --startup-file=no --project=. -e 'using CYAxiverse; println(CYAxiverse.greet_CYAxiverse())'` | Pass; printed `Hello CYAxiverse!` |
| `julia --startup-file=no --project=docs docs/make.jl` | Pass; HTML documentation rendered successfully |
| `git diff --check` | Pass |

The documentation build emitted only non-fatal offline Makie texture-atlas
and Documenter remote-HEAD warnings. The source fix is limited to the
internal canonicalization loop that triggered the post-merge Julia 1.12 CI
failure; the validated SCI-06 semantics remain unchanged.
