# Issue 148 G3 repair evidence

This record supersedes the failed candidate record
`docs/src/issue_148_g3_evidence.md` and addresses
`docs/src/issue_148_g3_independent_review_1c57ed2.md`.  The repaired code and
replay were tested at commit
`4abd9c31aba1d387bac27769730c4e0dc7d0c4a0` with Julia 1.12.6 on Darwin arm64.
The source, P96/CYTools, source-twelve, zero-phase, fixed-saxion, and positive
alpha contracts remain those of `docs/src/issue_148_g3_control_audit.md`.

## Corrected diagnostics

The projected derivatives now follow the established potential convention

```text
V(theta) = sum_I a_I [1 - cos(2 pi q_I.theta)]
D_v^3 V = -(2 pi)^3 sum_I a_I sin(2 pi q_I.theta) (q_I.v)^3
D_v^4 V = -(2 pi)^4 sum_I a_I cos(2 pi q_I.theta) (q_I.v)^4.
```

The full-factor Hessian uses `(2 pi)^2`, and its null residual is retained
separately from the normalized Hessian residual.  Every continuation step and
every BigFloat result now records:

- normalized gradient, null, and combined augmented residuals;
- full Eq. (19) gradient and Hessian-null residuals;
- the P96-metric canonical Hessian spectrum, one-null count, and minimum
  transverse eigenvalue;
- full-factor D3/D4, cone/divisor/action, metric, and conditioning data.

The canonical Hessian is independently formed as
`L^{-1} H U^{-1}` with `K=L L'`, then symmetrized before eigendecomposition.  The
precision-aware reporting floor separates the near-null mode from the
observed positive transverse spectrum.

## Status and continuation corrections

The initial augmented state is validated before continuation.  A perturbed
state with zero requested steps now returns `:invalid_initial_state`, stores
`converged=false`, and retains its large residual.  A valid state with
`n_steps=0` returns `:max_steps`; resource exhaustion is never mapped to
`:completed`.  The disabled-corrector boundary remains `:step_failed`.

The fixed-alpha method is named accurately as predictor/corrector.  The
incorrect pseudo-arclength alias and `bordered_condition` label were removed;
the stored condition is the fixed-alpha corrector Jacobian condition.  The
result records attempted, accepted, and rejected counts plus source-table,
metric-contract, and control provenance.

The positive signed seed gives one local branch.  The replay stores 46 states,
which are one seed plus 45 accepted transitions.  There were 49 attempted
loop iterations and 39 rejected backtracking trials.  The final stored state
is `alpha=1.434567377e-4`, `k=0.5006853945`, with status
`k_bounds_reached`.  The supported fate is persistence across the traced
segment followed by an imposed lower-`k` bound; this is a numerical boundary,
not a physical termination claim.  The opposing signed seed remains a
conservative `seed_failed` diagnostic and is not evidence that a second branch
is absent.

Across all 46 stored states, the canonical spectrum has exactly one reported
near-null mode and positive transverse minimum.  The minimum transverse
eigenvalue is `0.0105283777`; the minimum P96 metric eigenvalue is
`1.2875523e-4` (independently verified; `2.328173e-4` is the endpoint
value); the minimum source action is `7.009883`.  The final state keeps
positive volume, curve controls, and full-factor diagnostics.

## Precision evidence at exact controls

Independent and chained BigFloat solves use exact rational controls
`1//100000000`, `1//1000000`, `1//100000`, and `1//10000`; Float64 traced
states are only initial guesses.  Each solve reconstructs source geometry and
the P96 metric at its target precision.

| alpha | 128-bit iterations | independent 256-bit | chained 256-bit | independent/chained `k` difference |
|---:|---:|---:|---:|---:|
| `1e-8` | 3 | 5 | 3 | `4.56e-72` |
| `1e-6` | 218 | 219 | 3 | `3.93e-73` |
| `1e-5` | 9 | 10 | 3 | `2.31e-74` |
| `1e-4` | 7 | 9 | 3 | `1.77e-74` |

All twelve solves converged, retained one canonical null mode and positive
transverse minimum, and passed the declared residual thresholds.  At the
`1e-4` point the independent/chained 256-bit scale and periodic-state
differences are below `1e-60`.

At the representative `alpha=1e-8` chained 256-bit point:

```text
normalized gradient residual = 9.1645e-77
normalized null residual     = 2.439e-77
full projected D3            =  7.845685e-24
full projected D4            =  4.682939e-24
full Hessian-null residual   = 2.0506e-104
half-factor residual         = 1.0253e-104
```

The replay directly recomputes D4 and the full Hessian with the independent
formulas and asserts agreement.  The D3 control diagnostic retains the audited
normalized partial sensitivity `-1.3384279422e6` and its one-sided
finite-difference agreement; the separately reported full-factor derivative
remains available for scale transparency.

## Focused replay

Commands were run with
`JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia`:

```text
julia --startup-file=no --project=. scripts/issue_148_g3_control_evidence.jl
```

Result: exit `0`, `Issue 148 G3 local-control evidence: PASS`, including the
corrected D4/full-Hessian checks, invalid-initial and max-step regressions,
45-transition accounting, along-segment spectrum controls, and exact-control
128/256 precision checks.

```text
julia --startup-file=no --project=. scripts/audit_issue_148_g3_controls.jl
```

Result: `79/79` prerequisite checks passed.

```text
julia --startup-file=no --project=. scripts/issue_148_g1_replay_checks.jl
```

Result: `validation_checks_complete=true`.

`git diff --check` passed.  G2 remains radial and `:unresolved`; no classifier
cutoff, source term set, persisted schema, G1 behavior, G4 path, or physical
termination claim was changed.
