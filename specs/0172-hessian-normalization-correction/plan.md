# Implementation Plan — CYAX-0172

## Governing specification

Canonical candidate specification:
`specs/0172-hessian-normalization-correction/spec.md`

The specification is currently **draft**. No source implementation is allowed
until CYAX-0172 G0 is satisfied and durable owner approval is recorded.

## Coverage

| Requirement / Gate | Planned implementation | Planned verification |
| --- | --- | --- |
| R-001 | Change only the Hessian prefactor in `scripts/phase_volume_detuning_scan.jl` from `2π²` to `4π²` / `(2π)^2` | Closed-form analytic matrix comparison |
| R-002 | Preserve/update identity fixture | `4π² I₂` package test |
| R-003 | Add one nontrivial direct-formula fixture | Independent matrix assembly, no `_hessian` reuse |
| R-004 | Freeze analytic one-dimensional root | Compare refined result to `0.5*log10(phi)` |
| R-005 | Reconstruct historical formula only inside test/evidence code | Verify factor-two scaling and eigenvalue-sign invariance |
| R-006 | Replay a representative bounded catastrophe case | Compare bracket, refined `k_c`, type, side signatures |
| R-007 | No change to potential/homotopy/claim fields | Exact diff + focused assertions |
| R-008 | Record exact candidate and evidence | Fresh independent Scientific/Numerical Review |
| G0 | Spec/Scientific review + owner approval | Durable review identities + approval ref |
| G1 | Focused correction | Tests + diff inspection |
| G2 | Replay invariance | Before/after evidence |
| G3 | Exact candidate acceptance | Independent review bound to candidate |

## Existing architecture

The only intended production-code mutation is
`scripts/phase_volume_detuning_scan.jl::_hessian`.

The existing testset in `test/runtests.jl` already exercises:

- the identity Hessian;
- invalid `k_grid` ordering;
- one-dimensional detuning and arbitrary-precision refinement.

No package module ownership, public API, persistence schema, or search pipeline
is moved by this work.

## Proposed approach

1. Freeze and approve the S2 spec before code edits.
2. In one implementation worker, replace the prefactor with a direct
   representation of `(2π)^2` and keep all other helper bytes unchanged unless
   focused tests require an explanatory comment.
3. Strengthen focused tests with:
   - direct identity oracle;
   - nontrivial direct analytic matrix oracle;
   - analytic golden-ratio `k_c` oracle;
   - historical-vs-corrected factor-two relation.
4. Run the existing bounded scan fixture and one representative catastrophe
   replay under the exact candidate.
5. Record exact before/after evidence, commands, environment, commit and tree.
6. Freeze the candidate and obtain independent Scientific/Numerical Review.
7. Repair only attributable findings; changed candidate bytes require fresh
   exact review.
8. Return to Control Desk. Do not merge or close the Issue automatically.

## Numerical/oracle design

The primary oracle is analytic, not finite-difference-based:

```math
H = (2π)^2 Q^T diag(A cos(2π(Qθ+δ))) Q.
```

The one-dimensional zero-mode root is also analytic:

```math
k_c = 0.5 log10((1+sqrt(5))/2).
```

This avoids a tolerance-sensitive finite-difference oracle and directly repairs
the self-consistency weakness identified in the prior r5 handoff review.

## Data/API/schema impact

None intended.

No persisted data schema changes.
No public API additions/removals.
No new dependency.

## Verification strategy

Focused first:

- targeted phase/volume-detuning testset;
- analytic matrix fixture;
- analytic `k_c` fixture;
- historical/corrected scaling relation;
- representative replay.

Then:

- `python3 scripts/agent_verify.py diff-check`;
- broader package verification as practical;
- exact diff review;
- independent Scientific/Numerical Review of the frozen candidate.

The known current Hessian expectation failure is the target defect and must not
be mislabeled as unrelated baseline noise.

## Migration / compatibility

No migration.

Historical P0 evidence retains the old behavior as history. Only future helper
execution uses the corrected normalization.

## Risk and stop conditions

Stop and return to the owner if:

- a correct implementation would require changing the potential or phase/charge
  conventions;
- the analytic `k_c` fixture moves beyond tolerance;
- branch/catastrophe classification changes;
- another observable changes beyond the global Hessian scaling;
- the correction requires production paths outside the approved scope;
- a review finding exposes a new scientific choice rather than an
  implementation defect.

## Task decomposition

One implementation worker should own edit -> focused tests -> diagnosis ->
correction -> retest because this is a tightly coupled numerical correction.

Independent scientific review must use a fresh reviewer after the candidate is
frozen.
