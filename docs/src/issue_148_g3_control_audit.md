# Issue 148 G3 off-ray geometry and local-control audit

Status: **prerequisite PASS; bounded G3 implementation may start without a new
scientific-owner decision**.  This audit identifies one source-faithful,
one-sided non-radial deformation and verifies that it supplies an independent
amplitude control while preserving the available geometric and EFT evidence.
It does not continue a catastrophe, classify the G2 event as a cusp, or make an
inflation, population, stabilization, or global-cone claim.

The main new fact is that the primitive two-cycle direction

```text
u = (0, 1, 2, -1, 1, 1, 1, 1)
```

is in the closure of the source toric Kähler cone, is not radial, makes every
effective-curve volume nondecreasing, and gives rank two when paired with the
paper's radial control in both action space and relative-amplitude space.  It
also breaks the cancellation of the projected cubic coefficient at first
order.  Zero phases therefore do not justify assuming that the accepted radial
degeneracy remains quartic off ray.  The accepted G2 classification remains
`:unresolved`.

## Claim and evidence identity

The claim audited here is only the prerequisite in Issue 148 comment
[5609698767](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148#issuecomment-5609698767): construct one valid non-radial Kähler control and
measure its local action/amplitude rank before expensive continuation.  The
owner clarification explicitly allows persistence, termination, splitting,
unfolding, a class change, or extra null directions as later G3 outcomes.

| Evidence | Identity |
|---|---|
| Issue and current G3 framing | Issue 148; body and owner comment `5609698767`, retrieved 2026-09-09 |
| Paper | *Catastrophic Inflation in the Axiverse*, `arXiv:2608.14780v1`; PDF SHA-256 `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6` |
| Accepted G0 | `62c5a61`; baseline audit at `docs/src/issue_148_g0_baseline_audit.md` |
| Approved metric contract | P96, `d1a0b70`; `docs/src/issue_148_n8_approved_metric_contract.md` |
| Accepted G2 | implementation `cc8ac73668ac488a492dd16008c0b790a4e4ef3b`; decision `f9b04ed74bc30cc8e0071fcbe18179a4f85016e2` |
| Tracked N=8 geometry | `paper_benchmarks/appendix_c/h11_008/np_0000001/cy_0000001/cyax.h5`; SHA-256 `8fc483bc71a7e3f356b512adcfd44555f86a86bb765566c3f650984bf0ef1f4a` |
| Source reconstruction | CYTools `1.4.12`, Python `3.14.6`, NumPy `2.4.6`, SymPy `1.14.0`; source Eq. (93) height vector; exact arithmetic enabled |
| Audit base | clean `f9b04ed74bc30cc8e0071fcbe18179a4f85016e2`, Julia `1.12.6`, Darwin arm64 |

The tracked directory name `appendix_c` is stale.  In the v1 paper the N=8
geometry is Appendix D.

The evidence classes used below are explicit:

| Class | Content in this audit |
|---|---|
| Source fact | Paper equations, Appendix-D geometry/table, and the paper's radial potent-curve statement |
| Owner-approved extension | P96 period-one coordinates and Eq. (96)/CYTools matrix authority |
| Implementation fact | Existing G2 source-twelve amplitudes omit only a common `8pi/V^2` factor; the root P96 metric is stored at Float64 precision |
| Empirical verification | Exact CYTools reconstruction, HDF5/root comparisons, interval probes, ranks, and cubic finite difference |
| Inference | Monotone curve volumes transfer the paper's radial potent-curve control to positive `alpha`; nonzero partial cubic sensitivity means off-ray quartic behavior cannot be assumed |

## Source facts

These statements are fixed by the paper rather than inferred from repository
behavior.

1. Equations (14)--(16) define the real Kähler form through two-cycle volumes
   and the divisor volumes
   `tau_i = (1/2) kappa_ijk t_j t_k`.  A physical Kähler form has positive
   volume on every effective curve.
2. Equation (19) gives the leading diagonal coefficient, up to the benchmark's
   fixed common assumptions, as
   `Lambda_I^4 = (8pi/V^2) S_I exp(-2pi S_I)`, with
   `S_I=q_I.tau`.  Equations (19)--(20) use period-one axions and
   `cos(2pi Q theta + delta)`.
3. Equation (25) defines the radial control by `tau(k)=k tau(1)`.  Therefore a
   scaling `s` of two-cycle coordinates satisfies `k=s^2`, not `k=s`.
4. Appendix D fixes the vertices, point ordering, height vector, GLSM-basis
   divisor volumes, source-twelve charges/actions, total volume, and Eq. (96)
   metric spectrum at the stretched-cone tip.
5. The paper states that at the N=8 radial event all sub-unit curves are
   nilpotent, all potent curves are larger than one, and the smallest divisor
   volume is of order ten.  It does not publish the individual potent-ray
   labels or their margins.
6. The benchmark is fixed-saxion.  It does not establish saxion stabilization.

## Exact geometric reconstruction

The source height vector produces a fine, regular, star triangulation with 38
simplices.  In the paper's eight-dimensional divisor basis, CYTools gives 64
nonzero sorted triple intersections and 39 toric Mori generators.  Their
canonical fingerprints are:

```text
simplices SHA-256  6c078a79c51a61b93267639a262adfb251486c82c3aa0f29a7d3426a3026a7ee
kappa SHA-256      c430e6b4da2006eb2c32b198bceb1fc63c8179e5fc09fd956ad23139fa679444
Mori SHA-256       f247c53a8ea44cc5db5c614abba5b86e1a1113c5a210591891b2b3b94031caf6
divisor basis      (1,3,4,5,6,9,10,11) in zero-based CYTools labels
```

The exact two-cycle tip is

```text
t_ref = (1,4,4,-2,4,3,3,3).
```

Although one coordinate is negative, this is a basis coordinate, not a curve
volume.  The 39 physical toric curve volumes are `MORI*t_ref` and range from
one to three.  Exact contraction of the intersection tensor gives

```text
V(t_ref)   = 126
tau(t_ref) = (45,17,17,29/2,29/2,31/2,31/2,25).
```

Contracting the twelve Table-1 charges with this `tau` gives exactly

```text
(14,29/2,29/2,31/2,31/2,31/2,31/2,16,17,17,25,45),
```

so the action list is derived from the two-cycle geometry rather than treated
as independent divisor coordinates.

For generic `t`, define

```text
A_ij(t)       = kappa_ijk t_k
Kinv_CY(t)    = 4 [tau(t) tau(t)' - V(t) A(t)]
K_P96(t)      = inverse(Kinv_CY(t)).
```

At `t_ref`, the exact rational reconstruction of `K_P96` agrees with the root
P96 matrix to `8.50e-19` maximum absolute error after the root matrix's
Float64 rounding.  This is the CYTools matrix convention selected by the P96
owner decision.  It is deliberately not described as a literal derivation of
paper Eq. (23), whose documented factor-two conflict remains.

## The selected non-radial slice

The bounded slice is

```text
t(k,alpha)   = sqrt(k) [t_ref + alpha u]
u            = (0,1,2,-1,1,1,1,1)
alpha        in [0,1/20].
```

The vectors `t_ref` and `u` have rank two.  `u` is a primitive extremal ray of
the reconstructed toric Kähler cone, and all 39 entries of `MORI*u` are
nonnegative.  Thus this deformation is neither an arbitrary change of divisor
volumes nor radial scaling in another parameterization.

All dependent quantities are recomputed:

```text
tau(k,alpha) = k tau(t_ref + alpha u)
V(k,alpha)   = k^(3/2) V(t_ref + alpha u)
S_I          = q_I . tau(k,alpha)
K_P96(k,a)   = K_P96(t_ref + alpha u)/k^2
Lambda_I^4   = [8pi/V(k,alpha)^2] S_I exp(-2pi S_I).
```

The source-twelve identities and exact zero phases stay fixed throughout the
slice.  Terms are not reselected when their ordering changes.

At the accepted 256-bit radial event
`k_c=0.6745063700033668455...`, the endpoint checks are:

| Quantity | `alpha=0` | `alpha=1/20` | Gate |
|---|---:|---:|---|
| minimum toric curve volume | `0.8212834` | `0.8212834` | positive; no curve decreases |
| minimum prime-toric-divisor volume/action | `9.4430892` | `9.5779905` | greater than one |
| Calabi--Yau volume | `69.7990687` | `72.7988938` | positive and increasing |
| minimum P96 metric eigenvalue | `1.283037e-4` | `1.217104e-4` | positive |

Eleven equally spaced rational `alpha` samples all had positive volume,
prime-divisor volumes above one, and a positive-definite P96 metric; the
smallest sampled metric eigenvalue was `1.217104e-4`.  Exact endpoint
derivatives show that every prime divisor is nondecreasing.  The exact
coefficients of the quadratic `dV/dalpha` are `(107,52,11/2)`, so the total
volume is also increasing over the full interval.

The smallest toric curve is below one because the source radial event itself
is below the stretched-cone tip.  The EFT claim does not require every curve
to exceed one: it requires every potent curve to exceed one.  The paper
asserts this at `alpha=0`.  Since `MORI*u >= 0`, every effective curve,
including every potent curve, is nondecreasing for `alpha >= 0`; the paper's
potent-curve control therefore transfers across the audited interval without
needing unpublished potent labels.  The negative-alpha side is not certified
by this argument and is outside the proposed implementation packet.

## Local action and amplitude rank

At fixed `k_c`, the exact local action Jacobian for the eight independent
shape-coordinate directions is

```text
dS/dt = k_c Q A(t_ref),
```

and has rank eight.  Its singular values are

```text
(22.7332,16.5675,14.7990,14.7180,8.01707,5.96954,5.42185,4.72327).
```

The reduced source-twelve log amplitude is

```text
ell_I^red = log(S_I) - 2pi S_I.
```

Its `12 x 8` Jacobian also has rank eight, with smallest singular value
`29.2192`.  Adding the Eq. (19) factor gives

```text
ell_I^full = log(8pi) - 2log(V) + ell_I^red.
```

The volume term is the same for every row.  Row-centering removes global
amplitude and makes the centered full and reduced Jacobians identical to
working precision.  The centered full Jacobian still has rank eight; its
smallest singular value is `23.8033`.

For the two controls used in the pilot, `rho=log(k)` and `alpha`,

```text
dS/dalpha = k_c (4,12,12,3,11,3,11,6,4,12,18,30)
```

and the `12 x 2` action-control matrix has rank two, with singular values
`(56.0275,8.14452)`.  After removing global amplitude, the log-amplitude
control matrix also has rank two, with singular values
`(158.439,42.2188)`.  The best radial fit to the off-ray action derivative
leaves a relative residual of `0.316508`, so `alpha` is a genuinely independent
control.

The global `V^-2` contribution is explicit rather than silently discarded:

```text
d ell_global/d log(k) = -3
d ell_global/d alpha  = -1.698413  at alpha=0.
```

It cannot affect stationarity, nullity, or relative-amplitude rank at a fixed
control point, because it multiplies every potential term by the same positive
scalar.  It must be restored for any absolute potential or observable claim.

## Cubic coefficient test

The replay refines the accepted radial event from the G2 GLSM seed with the
exact integer charges, rational actions, exact zero phases, and 256-bit
arithmetic.  It normalizes the radial null direction with the exact
P96/CYTools metric, removes only the common amplitude scale, and evaluates

```text
D_v^3 V = -(2pi)^3 sum_I a_I sin(2pi q_I.theta) (q_I.v)^3.
```

At the radial event the normalized cubic is `-5.40e-21`, consistent with the
zero-phase cancellation at the refined point.  Holding the event point and
its normalized null direction fixed, the exact partial control derivative is

```text
partial_alpha D_v^3 V = -1.338427942e6.
```

A one-sided `delta alpha=1e-7` finite difference gives
`-1.338426289e6`, a relative difference of `1.24e-6`.  This is an empirical
local partial derivative, not the total derivative along a yet-unknown
catastrophe locus.  It establishes that the legitimate Kähler control breaks
the radial cubic cancellation at first order; a later solver must determine
whether motion of `theta`, `v`, and `k` restores a fold condition, splits the
locus, terminates it, or produces another allowed outcome.

This test does not alter the G2 classifier or convert its accepted
`:unresolved` result into a cusp claim.

## Contract decision

No new owner choice is required for the bounded local-discriminant
implementation below.

- The geometry map is fixed by the source vertices, height vector, basis, and
  exact intersections.
- The source diagonal coefficient and common `8pi/V^2` factor are fixed by
  Eq. (19).  The existing G2 reduced coefficient differs only by that common
  scalar, so it is sufficient for catastrophe equations; the full factor is
  separately recomputed and reported.
- P96 selected the Eq. (96)/CYTools numerical-matrix authority.  Applying the
  same exact `Kinv_CY=4(tau tau'-VA)` reconstruction off ray is the continuous
  CYTools extension that reproduces approved `M96/k^2` at `alpha=0`.  It does
  not choose P23 or the author raw-radian convention.
- The benchmark's equal, fixed one-loop-magnitude assumption, zero phases,
  source-twelve truncation, and fixed saxions are preserved.

Owner direction would be required before adding moduli-dependent one-loop
determinants, changing the source-twelve term set, using the 78-term generated
potential, changing P96/P23/A96 normalization, accepting negative `alpha`
without potent-ray evidence, changing the catastrophe criterion, or making an
absolute observable claim.  None is needed to determine the local fate of the
accepted radial event on the audited positive-alpha slice.

## Proposed bounded G3 implementation packet

- **Class:** `IMPLEMENTATION`, followed by fresh independent scientific
  review.
- **Objective:** determine the local fate of the accepted source-twelve radial
  one-null degeneracy for `alpha in [0,1/20]` using the exact slice and
  coefficient/metric maps above.
- **Acceptance:** recompute `t,tau,V,K,S,Lambda` at every trial point; solve the
  augmented stationarity/null/normalization equations from the accepted event;
  search both local fold seeds if the nonzero cubic-control derivative unfolds
  the radial solution; independently verify representative solutions at
  128/256 bits; report gradient, null, normalization, inertia, transverse
  spectrum, projected third/fourth derivatives, cone/divisor/potent-transfer
  controls, and truthful termination/splitting/failure statuses.  Any of the
  owner-approved fate outcomes is a valid result; a persisted curve is not
  required.
- **Inputs:** base `f9b04ed`; this audit and replay; G0 `62c5a61`; P96
  `d1a0b70`; accepted G2 implementation `cc8ac73` and decision `f9b04ed`;
  paper/source/HDF5 identities above.
- **Constraints:** period-one GLSM coordinates, argument `2pi Q theta`, exact
  source-twelve charges, exact zero phases, fixed saxions, P96/CYTools metric,
  positive-alpha interval only, no term reselection, no G2 reclassification,
  no normalization/coordinate/criterion/schema/API change, no inflation,
  population, global-cone, or G4 work.
- **Worker ownership:** ordinary local seed construction, scaling, solver
  conditioning, precision refinement, failure diagnosis, correction, focused
  tests, and replay evidence.
- **Escalation:** return only a demonstrated source/contract contradiction or
  a need to cross one of the owner boundaries above.  Solver failure or a
  terminated/split locus is a scientific result to diagnose and record, not by
  itself a request to change the contract.
- **Lease:** 20 minutes for initial local solves and one checkpoint at actual
  expiry if still making concrete progress.

No expensive locus tracing was run in this prerequisite audit.

## Exact commands and observed results

1. `gh api repos/Julia-meets-String-Theory/CYAxiverse.jl/issues/comments/5609698767`
   exited zero and returned the exact owner G3 framing.  `gh issue view 148
   --json number,title,body,url` exited zero and returned the durable gate body.
2. `shasum -a 256
   /Users/vmehta/Documents/CYAxiverse/cyaxiverse/catastrophicKS.pdf` exited zero
   and returned the paper hash recorded above.
3. The source-height CYTools reconstruction was run with the following exact
   command (line wrapping is only for display):

   ```console
   XDG_CACHE_HOME=/private/tmp/cyax-g3-cytools-cache conda run --no-capture-output -n cytools python -c 'import cytools,numpy as np,json,hashlib,importlib.metadata as m; from cytools import Polytope; cytools.config.enable_experimental_features(); V=np.array([[0,0,0,1],[1,0,0,0],[-1,-1,1,0],[-1,1,-1,0],[1,-1,-1,-1],[1,1,1,-1],[0,-1,0,0],[0,0,-1,0],[0,0,1,0],[0,1,0,0]],int); h=[0,11,11,13,13,14,14,11,11,11,11,11,12]; tr=Polytope(V).triangulate(heights=h,backend="cgal"); cy=tr.get_cy(); t=np.array([1,4,4,-2,4,3,3,3.]); ints=cy.intersection_numbers(in_basis=True,exact_arithmetic=True,format="dok"); canon=lambda x:json.dumps(x,separators=(",",":"),sort_keys=True).encode(); out={"cytools":m.version("cytools"),"hodge":[cy.h11(),cy.h21(),cy.chi()],"frst":[tr.is_fine(),tr.is_regular(),tr.is_star()],"simplices_sha256":hashlib.sha256(canon(tr.simplices().tolist())).hexdigest(),"basis":cy.divisor_basis().tolist(),"tip":cy.toric_kahler_cone().tip_of_stretched_cone(1).tolist(),"volume":cy.compute_cy_volume(t),"tau":cy.compute_divisor_volumes(t,in_basis=True).tolist(),"curve_range":[float(np.min(cy.compute_curve_volumes(t))),float(np.max(cy.compute_curve_volumes(t)))],"mori_shape":list(cy.toric_mori_cone(in_basis=True).rays().shape),"mori_sha256":hashlib.sha256(canon(cy.toric_mori_cone(in_basis=True).rays().tolist())).hexdigest(),"kappa_entries":len(ints),"kappa_sha256":hashlib.sha256(canon([[list(k),str(v)] for k,v in sorted(ints.items())])).hexdigest()}; print(json.dumps(out,sort_keys=True))'
   ```

   It exited zero with CYTools `1.4.12`, Hodge data `(8,28,-40)`,
   `FRST=(true,true,true)`, `V=126`, the Eq. (95) divisor volumes, curve range
   `[1,3]`, 64 exact intersection entries, 39 by 8 Mori data, and the three
   fingerprints above.  Experimental mode was used only for CYTools
   exact-rational intersections and emitted its documented warning.
4. `python3 scripts/agent_verify.py run -- julia --startup-file=no --project=.
   scripts/audit_issue_148_g3_controls.jl` exited zero in `13.76 s`.  Testsets
   passed `20/20`, `49/49`, and `10/10`; the exact event refinement converged,
   all ranks and interval checks passed, `offray_continuation_started=false`,
   and `g2_classification_preserved=unresolved`.

The focused replay is `scripts/audit_issue_148_g3_controls.jl`.
