# Issue 148 G3 local-control evidence

This record covers the bounded G3 implementation after the prerequisite audit.
It is revision-specific evidence for fresh review; it does not promote the
accepted G2 radial event to `:fold` or `:cusp`, and it does not make a global
off-ray population claim.

The implementation and replay were tested at code commit
`1c57ed2be15741ffcdf0227f0210cbf664d989f6` (Julia 1.12.6, Darwin arm64).
The audit contract is `docs/src/issue_148_g3_control_audit.md`, and the
approved radial metric remains `docs/src/issue_148_n8_approved_metric_contract.md`.

## Contract implemented

The replay reconstructs the source-twelve, zero-phase, fixed-saxion model on
the audited one-sided slice

```text
t(k, alpha) = sqrt(k) (T_REF + alpha U)
T_REF       = (1,4,4,-2,4,3,3,3)
U           = (0,1,2,-1,1,1,1,1)
0 <= alpha <= 1/20.
```

The intersection tensor and 39 Mori generators are the exact audited tables.
At every geometry evaluation the implementation recomputes

```text
tau_i = 1/2 kappa_ijk t_j t_k
V     = 1/6 kappa_ijk t_i t_j t_k
Kinv  = 4 (tau tau' - V A),   K = inv(Kinv)
S_I   = q_I . tau
Lambda_I^4 = (8 pi / V^2) S_I exp(-2 pi S_I).
```

The solver divides all amplitudes by their largest positive term only to
condition the stationary and null equations.  This common positive factor
does not change those equations.  Each accepted continuation step and each
high-precision diagnostic retains the full Eq. (19) factor, actions, source
coefficients, volume, curve volumes, divisor actions, and P96 metric checks.
The continuation result records source-table provenance, the P96/CYTools
metric contract, the source-twelve control, per-step corrector conditioning,
backtracking counts, and aggregate attempted, accepted, and rejected steps.

## Bounded continuation result

The replay starts from the accepted G2 event, maps to source coordinates, and
constructs both signed null-mode seeds at `alpha=1e-8`.

| Check | Observed result |
|---|---|
| side `-1` seed | `seed_converged`, `k=0.6733226692111264`, residual `9.55e-15` |
| side `+1` seed | `seed_failed`, residual `4.7365e-5` |
| accepted branch | 46 accepted steps, all augmented residuals below `1e-10` |
| bounded fate | `k_bounds_reached` at `alpha=1.434567377e-4`, `k=0.5006853945` |
| step accounting | 49 attempted, 46 accepted, 39 rejected backtracking trials |
| endpoint geometry | positive volume, positive P96 minimum eigenvalue `2.328173e-4`, minimum divisor action `7.009883` |
| disabled-corrector boundary | `step_failed` with `max_corrector_iterations=0` |

The branch therefore terminates within the declared `k` bounds.  The replay
does not infer behavior beyond that boundary.  The failed opposing signed
seed is retained as a conservative local branch outcome; no negative-alpha
claim is made.  The G2 radial classification remains `:unresolved`.

The accepted path uses an analytic alpha predictor and a damped Newton
corrector for the 17 augmented equations in `(theta, null_vector, k)`.
Every candidate recomputes exact source geometry before residual acceptance.
The `ds` value is reduced after failed corrections and is increased only after
fast successful corrections; failed candidates are counted rather than
silently dropped.

## Precision and canonical diagnostics

The same converged positive-alpha seed is solved independently at 128 and 256
bits, then the 256-bit solve is repeated from the 128-bit result.  The seed
conversion is only an initial guess; each solve reconstructs the exact integer
charge table, rational intersection data, P96 metric, actions, and full
Eq. (19) amplitudes at its own `BigFloat` precision.

| Solve | Iterations | normalized gradient residual |
|---|---:|---:|
| 128-bit | 3 | `1.5903e-32` |
| independent 256-bit | 5 | `8.0235e-76` |
| chained 256-bit | 3 | `9.1645e-77` |

The chained 256-bit scale differs from the 128-bit result by
`3.1784e-33`; independent and chained 256-bit `k` values agree to
`3.7720e-72`.  These are genuine Newton iterations at the stated precision,
not precision-bit relabeling.

At the chained 256-bit point (`alpha=1e-8`), the independent diagnostics are

```text
normalized |gradient| = 9.1645e-77
|H v|                 = 2.439e-77
metric null norm      = 2.2073379408e-2
full-factor scale     = 5.185959e-3
minimum action        = 9.426517
maximum full amplitude= 9.258603e-28
full projected D3     = 7.845685e-24
full projected D4     = -4.682939e-24
```

The local control witness uses the P96-metric-normalized radial null vector.
The normalized partial derivative of the projected cubic at the radial event
is `-1.3384279422e6`.  A one-sided exact-geometry finite difference at
`delta alpha=1e-10` gives `-1.3384279406e6`, satisfying the replay's relative
error bound of `1e-5`.  The corresponding full-factor derivative is also
reported (`-1.1127e-21`), because the full amplitudes are exponentially small
at these actions; the normalized value is the conditioning diagnostic used
for the stationarity equations.

## Exact replay

All commands below use the host Julia installation and the isolated issue
depot used for the focused checks:

```text
JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. scripts/issue_148_g3_control_evidence.jl
```

Result: exit `0`, `Issue 148 G3 local-control evidence: PASS`.

The prerequisite and preserved-scope checks also passed at this tree:

```text
JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. scripts/audit_issue_148_g3_controls.jl
```

Result: `79/79` audit checks passed, including exact geometry, positive-alpha
control boundaries, action/amplitude rank, and the audit finite-difference
control witness.

```text
JULIA_DEPOT_PATH=/tmp/julia_depot_issue148:/Users/vmehta/.julia \
  julia --startup-file=no --project=. scripts/issue_148_g1_replay_checks.jl
```

Result: `validation_checks_complete=true`.

`git diff --check` also passed before the implementation commit.  No persisted
schema, classifier cutoff, source term set, or G1 behavior was changed.
