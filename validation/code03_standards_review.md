# CODE-03 Standards Review

Review date: 2026-08-12  
Fixed point: `vmm` merge commit `81e43c4`  
Review branch: `codex/code03-ci-review`

## Review frame

This report covers the CODE-03 integration of the approved SCI-06
hierarchy/window spectrum feature onto `vmm`. The standards sources are the
repository's `.github/copilot-instructions.md` and the Julia-quality and
integration handoffs. No `.copilot/AGENTS.md` file is present in the merged
`vmm` tip.

The review is limited to integration, API consistency, numerical-code
reliability, tests, documentation, and the CI regression found after merge.
It does not change mass normalization, window definitions, certification
thresholds, eigenvalue sign conventions, or persisted scientific schemas.

## Findings

### Package and API structure

- The hierarchy/window APIs remain under `CYAxiverse.generate` and the
  orientation boundary remains under `CYAxiverse.read`.
- Existing `PhysicalAxionSpectrum` constructors and HDF5 readers are
  preserved.
- Geometry overloads and matrix-level entry points remain available.
- Result objects use concrete named/struct fields rather than introducing a
  new broad `Any` boundary.

### Julia and numerical conventions

- The implementation targets Julia 1.12, as declared by `Project.toml` and CI.
- High-precision spectrum paths retain arbitrary-precision eigensystem and
  inertia calculations; no new Float64 narrowing was introduced.
- Charge and instanton matrices retain the established orientation and
  `[sign, log10 scale]` representation.
- Window mode indices remain zero-based and mass bounds remain inclusive.
- Python remains optional for the core Julia package.

### Documentation and whitespace

- Public hierarchy/window APIs and diagnostics are documented in the user
  guide and API pages.
- `git diff --check` is clean for the follow-up branch.

### CI regression

The merged PR's Linux Julia 1.12 job exposed a failure in the unrelated but
shared `read.oriented_potential` canonicalization path. The duplicate-leading
charge lookup used a view-based `findfirst` predicate that passed locally but
raised a `BoundsError` in the CI fixture. The follow-up replaces that lookup
with an explicit, dimension-stable column comparison. This preserves the
existing representative selection and coefficient-consistency behavior while
removing the fragile iterator/view interaction.

## Standards disposition

The CODE-03 integration satisfies the repository coding and API conventions
after the CI regression fix. The only follow-up change is mechanical and does
not alter the validated SCI-06 numerical semantics.

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
and Documenter remote-HEAD warnings. The CI follow-up branch contains the
source fix and both required review reports.
