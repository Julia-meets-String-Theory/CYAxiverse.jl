---
spec_id: CYAX-0172
title: Correct phase/volume-detuning Hessian normalization
issue: 172
class: S2
status: draft
workstream: catastrophic-inflation numerical correctness
parent: null
depends_on: [170]
created: 2026-09-22
last_reviewed: null
review_required: Spec Reviewer + Scientific/Numerical Reviewer
approval_ref: N/A while draft
---

# Correct phase/volume-detuning Hessian normalization

## Objective

Correct the factor-of-two Hessian normalization defect in
`scripts/phase_volume_detuning_scan.jl` from `2π²` to the mathematically
required `(2π)^2 = 4π²`, while preserving the existing phase convention,
charge convention, amplitude-homotopy semantics, catastrophe locations,
branch classifications, and `homotopy_only` scientific claim boundary.

The correction is intentionally narrow. It changes the numerical value of the
helper Hessian by one global positive factor of two and must not change the
potential or reinterpret the scan as a physical divisor-volume calculation.

## Motivation

CYAX-0170 P0 froze the current `2π²` behavior as historical defect evidence,
not intended behavior. Issue #172 is the separately governed corrective work
authorized by the repository owner.

At current `vmm` commit
`995163f0058488ea183ac645045ed8b1636bef4a`, the helper source blob
`cd861aa8a257ae36c9dadef6486449a0497c35a4` computes

```julia
T(2) * T(π)^2 * (transpose(Q) * (weighted .* Q))
```

while the implemented potential is

```math
V(\theta)
=
\sum_a A_a(k)
\left[
1 - \cos\left(2\pi (Q\theta + \delta)_a\right)
\right],
```

whose Hessian is

```math
H(\theta)
=
(2\pi)^2
Q^T
\operatorname{diag}\left(
A_a(k)\cos\left(2\pi(Q\theta+\delta)_a\right)
\right)
Q.
```

The existing package test already expects `4π² I` for the identity fixture,
so the current Fast suite exposes this mismatch.

## Current baseline

### Source facts

- CYAX-0170 P0 scientific reference:
  `7a40285bb5c313f7e8746b90644d5f45bb67be44`.
- CYAX-0170 owner disposition, Issue #170 comment `5685945127`:
  preserve `2π² I` as historical defect evidence and govern the `4π² I`
  correction separately under Issue #172.
- Current intended potential convention uses phases measured in cycles and
  arguments `2π(Qθ+δ)`.
- Current amplitude homotopy is
  `A_a(k) = L[a,1] * 10^(k*L[a,2])`; it is explicitly nonphysical and
  labelled `homotopy_only`.

### Current implementation facts

At `vmm@995163f0058488ea183ac645045ed8b1636bef4a`:

- `scripts/phase_volume_detuning_scan.jl::_hessian` uses `2π²`;
- `test/runtests.jl`, in the `phase and volume detuning scan` testset,
  expects `4π² I` for
  `Q = I`, `L = [[1,0],[1,0]]`, `θ = 0`, `δ = 0`;
- the same testset contains a one-dimensional two-instanton detuning fixture
  with a sign-changing zero mode.

### Owner-approved convention

The intended correction is exactly the second derivative of the existing
potential convention. No alternative normalization is open for selection in
this work.

## Scope

This specification includes:

- correcting the Hessian prefactor in
  `scripts/phase_volume_detuning_scan.jl`;
- adding or refining focused regression checks needed to establish analytic
  potential/Hessian consistency;
- replaying the existing bounded one-dimensional golden-ratio detuning fixture
  as the exact representative catastrophe witness for CYAX-0172 G2;
- recording exact before/after evidence and candidate identity;
- obtaining fresh independent scientific/numerical review of the exact
  corrected candidate.

## Non-scope

This specification does not authorize:

- changing the potential formula;
- changing phase coordinates or the meaning of phase values;
- changing charge orientation or basis conventions;
- reinterpreting `k` as a physical divisor-volume map;
- changing fixed-saxion or `homotopy_only` claim boundaries;
- adding new catastrophe-search coverage;
- observational optimization;
- modifying CYAX-0170 P0 evidence;
- resolving the separate CYAX-0173 Float64/high-precision spectrum dispute;
- unrelated modularisation or package-version work.

## Scientific claim boundary

### This specification may establish

- that the helper Hessian matches the exact second derivative of its implemented
  potential convention;
- that the historical helper Hessian was low by one global factor of two;
- that a positive global Hessian rescaling leaves zero-mode locations and
  eigenvalue signs invariant for otherwise identical inputs;
- that representative bounded catastrophe results retain the same location and
  branch classification within predeclared numerical tolerances.

### This specification must not establish

- that the `k` homotopy is a physical Kähler-volume deformation;
- observational viability;
- Kähler-moduli stabilization;
- correctness of unrelated spectrum routes;
- any revision of the frozen historical P0 record.

## Fixed conventions and invariants

The owner-approved evidence model is **B+A**: B proves mathematical correctness
with an exact/high-precision phase; A preserves the existing Float64 regression
witness. Neither substitutes for the other.

1. Phases are measured in cycles.
2. Potential arguments remain `2π(Qθ+δ)`.
3. `Q` has one row per instanton in this helper.
4. Amplitudes remain
   `L[:,1] .* 10 .^ (k .* L[:,2])`.
5. The corrected Hessian is
   `4π² Q' Diagonal(A .* cos.(2π(Qθ+δ))) Q`.
6. The correction is one strictly positive global factor of two relative to
   the historical helper for every identical finite input.
7. `homotopy_only` remains nonphysical diagnostic status.
8. Historical `2π²` evidence remains historical and is not rewritten.

## Requirements

### R-001 — Correct exact Hessian prefactor

For every valid helper input, `hessian(...)` SHALL implement the exact analytic
matrix

```math
(2\pi)^2 Q^T \operatorname{diag}(A\cos(2\pi(Q\theta+\delta)))Q.
```

No other factor, coordinate convention, or amplitude expression may change.

### R-002 — Identity normalization fixture

For

```text
Q = I₂
L = [[1,0],[1,0]]
theta = [0,0]
phase = [0,0]
k = 1
```

the helper SHALL return `4π² I₂` to the existing package-test tolerance.

The test must evaluate `4π² I₂` independently of the helper implementation.

### R-003 — General analytic oracle

At least one nontrivial finite fixture with non-identity `Q`, nonzero
`theta`, nonzero phase, and unequal amplitudes SHALL compare the corrected
helper against an independently assembled closed-form matrix using the formula
in R-001.

The oracle must not call the helper's private `_hessian` implementation.

### R-004 — Analytic catastrophe-location oracles

The existing one-dimensional replay fixture remains typed exactly as

```text
Q = reshape([1.0,1.0], 2, 1)
L = [[2.0,-1.0],[1.0,1.0]]
theta = [0.0]
phase = [0.4,0.0]   # Float64 input
```

For an **exact decimal** phase `p = 0.4 = 2/5`, the zero-mode equation is

```math
2\,10^{-k}\cos(0.8\pi) + 10^k = 0,
```

with exact reference root

```math
k_{\phi}
=
\frac{1}{2}\log_{10}\!\left(\frac{1+\sqrt{5}}{2}\right)
\approx
0.1044938201249893668846360446.
```

However, the frozen replay witness begins as `Float64(0.4)`. Its exact binary
value is

```text
p64 = 3602879701896397 / 9007199254740992
    = 0.40000000000000002220446049250313080847263336181640625.
```

The analytic root of the **actual frozen typed input** is therefore

```math
k_{64}
=
\frac{1}{2}\log_{10}\!\left[-2\cos(2\pi p_{64})\right]
\approx
0.1044938201249893888954169152.
```

The two references have different purposes and MUST NOT be conflated:

1. the arbitrary-precision bisection/refinement, with stopping tolerance
   `1e-20`, SHALL converge to the root of the actual Float64-origin function
   and agree with `k64` at a tolerance justified by the final bisection
   interval;
2. comparison of the frozen Float64 witness to the exact-decimal golden-ratio
   reference `k_phi` is a scientific cross-check only and SHALL use the
   predeclared absolute acceptance tolerance `1e-12`.

The `1e-20` value is a numerical refinement tolerance, not a claim that the
Float64-origin input equals exact decimal `2/5` to that accuracy.

### R-004B — Exact/high-precision mathematical-correctness fixture

In addition to the preserved Float64 replay witness in R-004/R-006, the
implementation SHALL add a **separate** exact/high-precision fixture that
constructs the phase without any Float64 round-trip.

Freeze the B fixture as:

```text
Q = reshape([1,1], 2, 1)
L = [[2,-1],[1,1]]
theta = [BigFloat("0")]
phase = [BigFloat("0.4"), BigFloat("0")]
k_low = BigFloat("0.10")
k_high = BigFloat("0.15")
precision_bits = 256
bisection/refinement stopping tolerance = 1e-20
analytic acceptance tolerance = 1e-20
```

The exact/high-precision phase represents `p = 2/5` directly. Its analytic
zero-mode root is

```math
k_{\phi}
=
\frac{1}{2}\log_{10}\!\left(\frac{1+\sqrt{5}}{2}\right).
```

The B fixture SHALL evaluate the corrected high-precision Hessian/root path and
satisfy

```text
abs(k_B - k_phi) <= 1e-20
```

without constructing `phase`, `theta`, or the bracket endpoints through
Float64 values.

This fixture is **mathematical-correctness evidence only**. It does not replace
or redefine the A/Float64 fixture, which remains the sole G2
historical/regression replay witness.

### R-005 — Historical/corrected scaling relation

For identical valid inputs, a test-only reconstruction of the historical helper
formula SHALL satisfy

```text
H_corrected = 2 * H_historical
```

within arithmetic tolerance.

Eigenvalue signs SHALL agree away from exact zero, and any zero-mode bracket
defined only by sign change SHALL remain unchanged.

### R-006 — Exact representative bounded replay

The R-004 golden-ratio fixture is also the **sole predeclared representative
G2 replay witness**. No second or post-hoc-selected catastrophe case is required
for this correction.

Freeze the replay inputs before implementation as:

```text
source vmm commit = 995163f0058488ea183ac645045ed8b1636bef4a
helper blob = cd861aa8a257ae36c9dadef6486449a0497c35a4
test/runtests.jl blob = 67a2c0d6d4ead46fb6acf58a0b6e3acfc41e22d0
Q = reshape([1.0, 1.0], 2, 1)
L = [[2.0, -1.0], [1.0, 1.0]]
theta = [0.0]
phase = [0.4, 0.0]
k_grid = range(0.05, 0.20; length=4)
precision_bits = 256
bisection/refinement stopping tolerance = 1e-20
Float64-facing exact-decimal oracle acceptance = 1e-12
expected coarse sign-change bracket = [0.10, 0.15]
typed phase = p64 = 3602879701896397 / 9007199254740992
typed-input analytic root = 0.5*log10(-2*cos(2*pi*p64))
exact-decimal cross-check root = 0.5*log10((1+sqrt(5))/2)
```

The exact pre-correction historical helper and the corrected helper SHALL be
evaluated on those same inputs. The evidence SHALL compare:

- the coarse sign-change bracket `[0.10, 0.15]`;
- refined `k_c`, checked against the typed-input oracle `k64` at the
  refinement-justified tolerance and against `k_phi` only at `1e-12`;
- catastrophe type / branch classification;
- eigenvalue signs on both sides of the bracket;
- the factor-two Hessian/eigenvalue magnitude relation where finite.

The witness is selected by this specification, before implementation output is
observed. Any change not explained solely by the positive factor-of-two Hessian
scaling is a stop condition.

### R-007 — Preserve scientific boundary

The change SHALL NOT alter:

- potential values;
- phase vectors or phase convention;
- amplitude-homotopy construction;
- `scale_status == :homotopy_only`;
- missing `N_e`, `n_s`, scalar-amplitude, and `r` fields in
  `ScanCandidate`.

### R-008 — Exact evidence and review

The implementation candidate SHALL record exact source/base identity, changed
paths, focused commands and outcomes, representative replay evidence, and the
candidate commit/tree.

The exact corrected candidate SHALL receive independent Scientific/Numerical
Review before scientific acceptance.

## Acceptance gates

### CYAX-0172 G0 — Governing-spec approval

**Objective:** establish the durable S2 contract before implementation.

**Acceptance:**

- exact spec/plan/tasks receive independent Spec Review;
- the normalization/oracle/replay contract receives independent
  Scientific/Numerical Review;
- blocking findings are repaired and the changed bytes re-reviewed;
- the owner explicitly approves the reviewed normative revision, and
  `approval_ref` records that durable approval.

**Stop condition:** unresolved normalization, coordinate, physical-meaning, or
oracle ambiguity.

### CYAX-0172 G1 — Focused implementation correctness

**Objective:** implement only the factor-of-two correction and required focused
tests/evidence.

**Acceptance:**

- R-001 through R-005 pass, including both the B exact/high-precision oracle
  fixture and the A typed-input regression/oracle checks;
- diff is confined to approved helper/test/spec/evidence scope;
- no P0 evidence is modified.

**Stop condition:** implementation requires changing any fixed convention or
scientific boundary.

### CYAX-0172 G2 — Bounded replay invariance

**Objective:** establish that the correction does not move or reclassify the
governed catastrophe event.

**Acceptance:** R-006 and R-007 pass on exact recorded fixtures.

**Stop condition:** catastrophe location, branch classification, or another
scientific observable changes beyond the predeclared tolerance for reasons not
explained by the global positive Hessian scaling.

### CYAX-0172 G3 — Independent exact-candidate acceptance

**Objective:** independently assess the exact implementation and evidence.

**Acceptance:** independent Scientific/Numerical Review has no blocking finding
and binds its verdict to the exact candidate commit/tree.

A passing review is evidence for Control Desk/owner acceptance; it is not merge
or Issue-closure authority.

## Verification requirements

Required evidence includes:

1. the identity `4π² I₂` fixture;
2. one general nontrivial analytic closed-form fixture;
3. the distinct exact/high-precision B fixture with `p=2/5` constructed without
   Float64 round-trip and `abs(k_B-k_phi) <= 1e-20`;
4. the typed-input analytic root
   `k64 = 0.5*log10(-2*cos(2*pi*p64))` for the frozen Float64 phase;
5. the exact-decimal golden-ratio cross-check
   `k_phi = 0.5*log10(phi)` with absolute acceptance `1e-12`;
6. historical-vs-corrected factor-two and sign invariance;
7. the exact R-006 replay witness, using the frozen inputs and source identities
   above;
8. focused package tests containing the phase/volume-detuning testset;
9. `scripts/agent_verify.py diff-check`;
10. broader package verification as practical, with the known pre-correction
   Hessian failure treated as the target defect rather than hidden baseline;
11. exact candidate independent Scientific/Numerical Review.

Unobserved checks are not PASS.

## Interfaces and compatibility

No public Julia API or persisted scientific schema is intended to change.

The helper's numerical output changes only by the scientifically intended
Hessian normalization. Package-version impact is deferred to the reviewed
release boundary.

## Dependencies and blockers

- Depends on the owner-approved CYAX-0170 P0 disposition that assigns this
  correction to Issue #172.
- Does not depend on resolving CYAX-0173.
- Does not depend on the broader CYAX-0131 search.

## Open owner decisions

None are intentionally left open in the normalization contract.

If implementation or review identifies a need to alter a fixed convention,
claim boundary, catastrophe classification, or acceptance tolerance beyond
this specification, stop for owner decision and re-review the specification.

## Completion criterion

CYAX-0172 is scientifically complete when the approved specification is
implemented on an exact candidate, G1 and G2 evidence are satisfied, independent
G3 Scientific/Numerical Review has no blocker, and Control Desk/owner
reconciliation accepts the correction without changing the historical P0
record.
