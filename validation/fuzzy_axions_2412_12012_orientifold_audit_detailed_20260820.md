# Final inherited-orientifold audit: h11 = 2, 3, 4

Date: 2026-08-20
Branch: codex/orientifold-overcount-20260819
Scope: final post-review full-population audits for favorable Kreuzer–Skarke populations at $h^{1,1}=2,3,4$
Implementation: [scripts/inherited_orientifold_candidates.py](../scripts/inherited_orientifold_candidates.py)
Audit driver: [scripts/reproduce_fuzzy_axions_h11_4.py](../scripts/reproduce_fuzzy_axions_h11_4.py)

This document is both a results report and a handoff for further student
projects. Read §1 first for the main conclusion, §5 for the mathematical
background, §6 for the code changes that are already in place, and §8 for the
remaining boundaries and possible continuation projects. The word
“certified” always means “accepted by an explicit test implemented here”; it
does not mean that every mathematical question about the orientifold has been
settled. The phrase “source-matched” means that the implementation follows the
equations and logical conditions cited from the source papers. It does not
mean that the implementation reproduces every part of the source pipeline.

> **Rerun status (2026-08-20, corrected general-`L` code):** The general-`L`
> overcount repair is now implemented and validated by a fresh h11=2 and
> h11=3 rerun. Fixed components are labelled by pointwise-`L`-invariant cones
> of the original fan `Sigma` (source eqs. (4.33)–(4.35)) with their original
> ray scale, and the eq. (4.45) dual-vertex parity condition is now imposed on
> every dual vertex whose facet meets a non-simplicial/non-smooth cone along a
> proper face (the corrected `any`-pairing facet test, source line ~629).
>
> Effect of the correction: at h11=2 no accepted orientifold changed — the
> inherited and $h^{1,1}_-=0$ lower bounds stay (10, 10) and the accepted set
> is identical polytope-by-polytope. At h11=3 exactly one previously accepted
> inherited orientifold is now correctly rejected (polytope index 229; see
> §3.3), so the h11=3 inherited and $h^{1,1}_-=0$ lower bounds move from 81 to
> **80**. The general-`L` diagnostic evidence-row counts decrease because the
> corrected enumeration uses the smaller, source-faithful original-`Sigma`
> pointwise-invariant cone universe.
>
> Scope: this rerun covers **h11=2 and h11=3 only**. Every h11=4 figure in this
> report (the general-`L` diagnostics, the inherited/$h^{1,1}_-=0$ counts, and
> the 20260819 artifact) is the prior auxiliary-`Sigma_L` value and is
> **superseded pending an h11=4 rerun with the corrected code**. The base
> populations, identity-sector diagnostics, and the independent trilayer
> benchmark are unchanged. See the [checkpoint review note](./fuzzy_axions_2412_12012_checkpoint_review_20260820.md).
>
> **Gap-analysis correction (2026-08-20):** A class-level review of the
> retained artifacts shows that only 6 unaccepted h11=2 classes and 28
> unaccepted h11=3 classes have candidate-linked unresolved-component
> evidence. Even granting all of them would give conditional ceilings 16/32
> and 108/253, leaving deficits 16 and 145. The generic 402 and 1,892 skipped
> surface rows are evidence attempts, not classes or exhaustive candidate
> verdicts. Therefore `non_smooth_ambient_cone` must not be presented as the
> explanation of the full Table 1 gap. See the
> [comprehensive gap review](./fuzzy_axions_2412_12012_gap_classification_20260820.md).

## 1. Executive conclusion

The h11=2 and h11=3 populations were rerun with the corrected general-`L` code
and the h11=4 population is shown from the prior run (see the rerun status note
above). Each worker exited with status 0, produced a valid JSON artifact, and
reported no CYTools cache-save warning. The base population and the independent
$h^{2,1}_+=0$ trilayer benchmark agree with the corresponding rows of Table 1
in Sheridan et al. The inherited-orientifold audit itself remains a
conservative lower count:

| $h^{1,1}$ | favorable polytopes | final FRST classes | audited inherited O3/O7 classes | Table 1 inherited target | audited $h^{1,1}_-=0$ classes | Table 1 $h^{1,1}_-=0$ target | audited trilayer $h^{2,1}_+=0$ classes | Table 1 trilayer target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 36 | 10 | 32 | 10 | 32 | 11 | 11 |
| 3 | 243 | 274 | 80 | 253 | 80 | 253 | 66 | 66 |
| 4 | 1185 | 1760 | 435† | 1559 | 429† | 1554 | 267 | 267 |

† The h11=4 inherited and $h^{1,1}_-=0$ counts are the prior
auxiliary-`Sigma_L` values, superseded pending an h11=4 rerun with the
corrected code. The h11=2 and h11=3 rows are the corrected 2026-08-20 rerun.

The audited orientifold numbers are code-certified class-level counts, not a
claim that the paper’s Table 1 is incorrect. The gap means that the present
implementation has not yet certified every source-paper orientifold using the
evidence available to it. In particular, every skipped general-$L$
fixed-surface attempt in these runs has the machine-readable reason
non_smooth_ambient_cone: the provenance cone is not a smooth unimodular
ambient four-cone, so the current source-matched Cartier/$n_S$ certificate
refuses to promote that candidate.

The reviewed implementation materially increased the certified inherited
population relative to the pre-review baseline (0, 4, 5 at
$h^{1,1}=2,3,4$) to (10, 80, 435†). It still does not close the Table 1 gap,
and the remaining records must not be relabelled as accepted merely because
the code could not finish a sufficient smoothness proof.

The phrase “conservative lower count” has a precise meaning here. A candidate
is counted only when every required test returns a passing result. If a test
cannot run because an input is missing or a geometric certificate is outside
the current implementation, the candidate remains unresolved. Thus an audit
count can increase when new evidence is implemented, but an unresolved record
cannot be promoted simply because no counterexample was found.

## 2. What was audited

The audit reads the favorable $N$-lattice mirror of the relevant
Kreuzer–Skarke polytopes, enumerates all FRSTs, quotients them into FRST
classes using the source-matched two-face relation, and evaluates two related
but distinct populations:

Here $h^{1,1}$ is the number of independent Kähler-modulus directions of the
Calabi–Yau, while $h^{2,1}$ counts complex-structure directions. A subscript
$+$ or $-$ denotes the part even or odd under the orientifold involution. The
word “population” refers to a collection of geometry classes satisfying a
specified set of filters; it does not refer to a population of dynamical
universes. Here “mirror” names the lattice presentation used by the
Kreuzer–Skarke database. Students do not need to construct a mirror map or
compute periods to follow this audit; the relevant integer polytope is already
provided as input.

1. the special trilayer involution and its independent
   $h^{2,1}_+(X,\mathcal I)=0$ diagnostic; and
2. the full inherited-orientifold search over lattice involutions $L$, torus
   shifts, and the two coefficient-parity choices $\lambda_f$, with the
   fixed-locus smoothness machinery from Moritz et al. enabled.

The audit counts a CY class as inherited only if at least one class-level
candidate reaches accepted_verified_orientifold with $\lambda_f=1$, the
O3/O7 convention used for Table 1. It counts the $h^{1,1}_-=0$ subset only
when that accepted candidate also has $h^{1,1}_-=0$. An O5/O9
$\lambda_f=0$ candidate is not allowed to satisfy the Table 1 O3/O7 row.

The full machine-readable output contains per-polytope details and the
opt-in general-$L$ diagnostics. The output schema is
cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1. Standalone candidate
manifests use cyaxiverse-inherited-orientifold-candidate-2.3; the nested
general-$L$ diagnostics use cyaxiverse-general-L-fixed-surface-diagnostics-1.1.
The general-`L` diagnostic schema advanced to 1.1 because its enumeration basis
changed to the original-`Sigma` pointwise-invariant cones; the field structure
is unchanged.

For continuation work, treat each JSON artifact as an evidence ledger rather
than as a table of final answers. The top-level summaries give counts, while
the `details` entries and candidate records explain why an individual class
was accepted, rejected, or left unavailable. A useful first exercise is to
trace one class from its polytope record, through its candidate actions, to
the final class-level count before changing any code.

## 3. Final audit artifacts and integrity evidence

The JSON files are in /private/tmp because they are large generated artifacts
rather than package source files. Their sizes and SHA-256 hashes are:

| $h^{1,1}$ | artifact | bytes | SHA-256 |
|---:|---|---:|---|
| 2 | [cyax-orientifold-rerun-h11-2-20260820.json](/private/tmp/cyax-orientifold-rerun-h11-2-20260820.json) | 48,576,572 | c4cc9bc93a8d590d42a2ef98fdb58156da442493f07d5854b8ca8e4873f9ebb0 |
| 3 | [cyax-orientifold-rerun-h11-3-20260820.json](/private/tmp/cyax-orientifold-rerun-h11-3-20260820.json) | 264,658,672 | 05ce23cc5fa0819a83ecbde0d8784628e5e540a874e8e6ba291ce2ed30f63e42 |
| 4† | [cyax-orientifold-final-reviewed-h11-4-20260819.json](/private/tmp/cyax-orientifold-final-reviewed-h11-4-20260819.json) | 1,365,317,574 | a5849fa7c02040dc4986653168262eac4d299a492d58fe7728e948afcd875769 |

The h11=2 and h11=3 artifacts are the corrected 2026-08-20 rerun (reproduction
schema 1.1). † The h11=4 artifact is the prior 2026-08-19 auxiliary-`Sigma_L`
run (reproduction schema 1.0), retained as the superseded reference pending an
h11=4 rerun. Its earlier hash was
`a5849fa7c02040dc4986653168262eac4d299a492d58fe7728e948afcd875769`.

Independent post-run checks confirmed, for every artifact:

- input.population_complete is true, now certified against the Table 1
  favorable-polytope target (36 loaded for h11=2, 243 for h11=3) rather than a
  bare record-limit heuristic;
- the requested $h^{1,1}$ is 2, 3, or 4 as appropriate;
- details has one entry per favorable polytope;
- orientifold_reason_diagnostics.h11 matches the requested Hodge number;
- the diagnostic attempt count equals certified plus skipped; and
- JSON parsing and SHA-256 computation succeed.

The three audit workers reported no known CYTools cache-save warning. The
focused local tests can still emit the known non-fatal cache warning when
CYTools attempts to write /Users/vmehta/Library/Caches/CYTools/; that warning
does not affect the generated audit artifacts and was not present in the
three final audit runs.

### 3.1 Population and orientifold counts

| quantity | h11 = 2 | h11 = 3 | h11 = 4 |
|---|---:|---:|---:|
| favorable polytopes | 36 | 243 | 1185 |
| raw FRSTs | 48 | 525 | 5330 |
| FRST classes | 36 | 274 | 1760 |
| raw trilayer polytopes | 13 | 80 | 292 |
| raw trilayer FRST classes | 13 | 90 | 415 |
| non-frozen trilayer FRST classes | 12 | 70 | 285 |
| trilayer $h^{2,1}_+=0$ FRST classes | 11 | 66 | 267 |
| identity torus actions | 251 | 2296 | 13218 |
| identity-action CY classes | 36 | 274 | 1760 |
| identity-valid O3/O7 CY classes | 25 | 198 | 1153 |
| source-evidence inherited O3/O7 classes | 10 | 80 | 435† |
| source-evidence $h^{1,1}_-=0$ classes | 10 | 80 | 429† |

The identity-valid O3/O7 row is a useful diagnostic, but it is not the Table
1 inherited count. Table 1 asks whether a CY admits an inherited orientifold;
the full audit includes nonidentity $L$ and torus-shift sectors as well.

### 3.2 General-$L$ fixed-surface diagnostics

| $h^{1,1}$ | diagnostic surface attempts | diagnostic evidence rows | skipped surfaces | skipped reason | unresolved candidate components |
|---:|---:|---:|---:|---|---:|
| 2 | 1164 | 762 | 402 | non_smooth_ambient_cone (402) | 33 |
| 3 | 5510 | 3618 | 1892 | non_smooth_ambient_cone (1892) | 117 |
| 4† | 25398 | 17024 | 8374 | non_smooth_ambient_cone (8374) | 327 |

The diagnostic evidence-row count is not a count of fixed components of every
specific torus shift, because the table does not attach the source eq. (4.35)
integrality condition to every $t$. It does not by itself mean that the
whole orientifold candidate is accepted: all fixed components, coefficient
parity, the $h^{1,1}_-$ filter, and the O3/O7 condition must also pass.
Conversely, an unresolved component is not a proof of a singular orientifold;
it is a proof that this implementation does not possess the sufficient source
evidence needed to accept it. The h11=2 and h11=3 rows above are the corrected
2026-08-20 rerun; the h11=4† row is the superseded pre-fix run, retained pending
its rerun. The attempt and evidence-row counts fell at h11=2 and h11=3 because
the corrected enumeration ranges over the smaller, source-faithful
original-`Sigma` pointwise-invariant cone universe rather than the auxiliary
fan `Sigma_L`. See the [checkpoint review note](./fuzzy_axions_2412_12012_checkpoint_review_20260820.md).

### 3.3 Overcount correction traced at h11 = 3

The corrected general-`L` code removes exactly one previously accepted
inherited orientifold relative to the pre-fix run, at polytope index 229. The
responsible lattice involution is

$$
L=\begin{pmatrix}1&0&-2&-2\\0&1&-1&1\\0&0&-1&0\\0&0&0&-1\end{pmatrix}.
$$

In the pre-fix run this candidate reached accepted_verified_orientifold with
$\lambda_f=1$: the eq. (4.45) dual-vertex parity condition was checked against
too few dual vertices, because the earlier facet test required every ray of a
cone to pair $-1$ with the dual vertex. With the corrected `any`-pairing test
(source line ~629), the dual vertices whose facet meets a non-simplicial or
non-smooth cone along a proper face are also flagged, eq. (4.45) then fails, and
the candidate is correctly recorded as fixed_point_set_non_smooth. This was
confirmed by re-running the same FRST class through both the base-commit and the
corrected enumerator: the base code yields one $\lambda_f=1$ acceptance for
this involution, the corrected code yields none, and every other polytope is
unchanged at h11=2 and h11=3 (the h11=2 accepted set is identical
polytope-by-polytope). This is the overcount the corrected code is designed to
remove; it is a downward correction to a conservative lower bound (81 → 80), not
a change to the paper's Table 1 target of 253.

## 4. Comparison with Sheridan et al. Table 1

The relevant source is Sheridan et al., [Fuzzy Axions and Associated Relics
(arXiv:2412.12012)](https://arxiv.org/abs/2412.12012),
especially its [HTML version and Table 1](https://arxiv.org/html/2412.12012).
The Table 1 population numbers are:

| $h^{1,1}$ | favorable | FRST | inherited | inherited with $h^{1,1}_-=0$ | inherited with $h^{1,1}_-=0,\ h^{2,1}_+=0$ | models |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 36 | 32 | 32 | 11 | 2 |
| 3 | 243 | 274 | 253 | 253 | 66 | 263 |
| 4 | 1185 | 1760 | 1559 | 1554 | 267 | 3348 |

The present audit reproduces the source base populations exactly and the
$h^{2,1}_+=0$ trilayer counts exactly, but its source-evidence orientifold
counts are lower:

| $h^{1,1}$ | inherited gap (Table 1 − audit) | $h^{1,1}_-=0$ gap | trilayer gap |
|---:|---:|---:|---:|
| 2 | 22 | 22 | 0 |
| 3 | 173 | 173 | 0 |
| 4† | 1124 | 1125 | 0 |

This comparison should be interpreted as a certification-boundary result.
The audit requires an explicit, source-matched chain of evidence for every
fixed component. It does not infer smoothness from the absence of a detected
problem, and it does not turn smoothness_verification_unavailable into an
acceptance. The residual gap is therefore evidence about the current
implementation’s proof coverage, not evidence against the published Table 1
counts.

The final counts are nevertheless a substantial improvement over the
pre-review baseline produced before the nef/orbifold implementation and the
review fixes: 0, 4, and 5 inherited classes at $h^{1,1}=2,3,4$, respectively.
The increases to 10, 80, and 435† show that the added evidence is active, while
the remaining gap shows that it is not yet complete (the h11=4 figure remains
the pre-fix value pending its rerun).

The comparison is therefore one-sided. The audit can demonstrate that a class
has passed the implemented source-matched checks, but it cannot demonstrate
that every class outside the audited set is singular. The gap is a useful map
of missing proof coverage and a starting point for new work.

## 5. Mathematical and physical framework

This section introduces only the geometry needed to understand the audit. The
main idea is simple: the calculation describes a Calabi–Yau threefold as a
hypersurface inside a toric fourfold, then checks whether a proposed order-two
symmetry gives a physically acceptable orientifold. The toric language is a
compact way to do exact calculations with integer data; it is not necessary to
learn all of algebraic geometry to follow the logic below.

### 5.1 From an integer polytope to a Calabi–Yau threefold

Start with the integer lattice

$$
N\simeq\mathbb Z^4
$$

and its dual lattice $M=N^\vee$. The natural pairing between them is written
$\langle m,n\rangle$. A four-dimensional reflexive polytope
$\Delta\subset N_\mathbb R$ is a convex polytope whose vertices lie in the
lattice and whose origin is its only interior lattice point. Its polar dual

$$
\Delta^\circ=\{m\in M_\mathbb R:\langle m,n\rangle\geq -1
\text{ for every }n\in\Delta\}
$$

is also a lattice polytope. This duality is useful because the lattice points
of $\Delta^\circ$ label the monomials that can appear in the Calabi–Yau
hypersurface equation.

To build the ambient space, subdivide the boundary data into cones meeting at
the origin. The resulting collection of cones is a fan $\Sigma$, and it
defines a toric fourfold $V_\Sigma$. For this audit, the fan comes from a
fine, regular, star triangulation (FRST); the next subsection explains those
words. Each one-dimensional cone, or ray, gives a toric divisor $D_\rho$.
The anticanonical divisor is

$$
-K_{V_\Sigma}=\sum_{\rho\in\Sigma(1)}D_\rho.
$$

An anticanonical section is a polynomial $f$ built from the allowed monomials
and defines a hypersurface

$$
X=\{f=0\}\subset V_\Sigma.
$$

Because $X$ represents the anticanonical class, the adjunction formula gives

$$
K_X=(K_{V_\Sigma}+X)|_X=0.
$$

Thus $X$ is a Calabi–Yau threefold: a complex three-dimensional space with
vanishing first Chern class, in the usual smooth case. In the string
compactification, $X$ is the six-dimensional internal space. Its Kähler and
complex-structure data enter the four-dimensional effective theory, including
axion decay constants, instanton actions, and the orientifold projection.

The audit uses the favorable populations from Sheridan et al. Here,
“favorable” means that the divisor classes needed by the database and its
intersection calculations are inherited from the toric ambient space in a
controlled way. It is a statement about this presentation and population; it
does not mean that every divisor on every Calabi–Yau phase is ambient.

### 5.2 Triangulations, smoothness, and what is being counted

An FRST is a particular way of subdividing the polytope boundary into cones:

- **fine** means that the required lattice points are used;
- **regular** means that the subdivision comes from a suitable convex height
  function; and
- **star** means that the cones are formed with the origin as their common
  apex.

The cones are also simplicial, so each cone is generated by linearly
independent rays. A maximal cone gives a local coordinate chart on the ambient
space. If its generators form a lattice basis, the cone is unimodular and the
chart is smooth. This is the smoothness test used in the audit.

Different triangulations can represent the same Calabi–Yau class under the
equivalence relation used by the source database. The report therefore keeps
two kinds of count separate:

1. the raw number of FRSTs; and
2. the number of equivalence classes after deduplication.

Table 1 uses the class-level count. The orientifold audit also deduplicates at
this level: if one symmetry action is accepted for a class, that class enters
the inherited count once.

This distinction is important for diagnostic data. For example, 17024
fixed-surface evidence rows at $h^{1,1}=4$ means 17024 pieces of evidence
recorded while examining candidates. It does not mean 17024 Calabi–Yau
threefolds, accepted orientifolds, or necessarily 17024 distinct fixed
components for one particular torus shift.

### 5.3 The order-two symmetry: lattice involution plus torus shift

An inherited orientifold begins with an integer linear map

$$
L:N\longrightarrow N,qquad L^2=I.
$$

The condition $L^2=I$ says that applying $L$ twice gives the identity, so $L$
is an involution. It must also preserve the polytope and the fan/divisor data;
otherwise it does not define a symmetry of the chosen toric model.

The linear map is combined with a translation on the torus part of the toric
space:

$$
\widetilde I=\phi_{[t]}\circ L.
$$

The vector $t$ records possible half-period phases of the torus coordinates.
The square of this affine action contains a translation by $2t$. Therefore a
candidate is a genuine $mathbb Z_2$ symmetry only when the required lattice
condition

$$
2t\in N
$$

is satisfied, after allowing for the relevant lattice identifications. The
audit keeps $t$ as an exact rational vector. Rounding a half- or quarter-
integral entry could change whether this condition holds.

The projectors onto the invariant and anti-invariant directions are

$$
P_\pm^L=\frac{1}{2}(I\pm L).
$$

They separate directions with $L v=v$ from those with $L v=-v$. The possible
shifts are organized by the corresponding projected lattices, schematically

$$
H_+^L=\frac{P_+^L(N)}{2P_+^L(N)},\qquad
H_-^L=\frac{P_-^L(N)}{2P_-^L(N)}.
$$

Here a quotient means that two shifts differing by twice a projected lattice
vector are treated as equivalent. The calculation uses Smith and Hermite
normal forms: these are exact integer row/column-reduction procedures that
determine lattice kernels and quotient membership. This is safer than trying
only a bounded list of integer coefficients, especially after changing to a
non-permutation $GL(4,\mathbb Z)$ basis.

### 5.4 The holomorphic three-form and the two orientifold branches

A Calabi–Yau threefold has a distinguished holomorphic three-form $\Omega$.
The parameter $\lambda_f\in\{0,1\}$ records the sign convention for how the
orientifold acts on this form and, equivalently in the source construction,
which coefficient-parity branch is being used. The audit uses:

- $\lambda_f=1$ for the O3/O7 branch used in the Sheridan et al. Table 1
  inherited population; and
- $\lambda_f=0$ for the O5/O9 branch, which is searched for completeness but
  is not part of the Table 1 O3/O7 count.

The identity check is intentionally asymmetric. The action $L=I,t=0$ is kept
as a trivial smooth fixture for $\lambda_f=0$. For $\lambda_f=1$, the same
zero-shift action would force every coefficient in the hypersurface equation
to vanish in the source phase convention. It therefore does not describe a
generic physical O3/O7 hypersurface and is rejected.

The involution also splits the $(1,1)$-forms into even and odd parts:

$$
H^{1,1}(X)=H^{1,1}_+(X)\oplus H^{1,1}_-(X),
\qquad
h^{1,1}=h^{1,1}_++h^{1,1}_-.
$$

The condition $h^{1,1}_-=0$ is an extra population filter. It is not a test
that the orientifold exists or is smooth, so it is applied only after the
O3/O7 and smoothness checks.

### 5.5 Fixed sets and parity tests for the hypersurface

The fixed locus of an involution is the set of points that are mapped to
themselves. In the toric calculation, the relevant pieces are found by
intersecting the ambient cones with the real subspace fixed by $L$. These
intersections form a finite lower-dimensional fan, denoted $\Sigma_L$; this is
the construction associated with eq. (4.26) of Moritz et al.

There is an important lattice detail. The source fixed-component formula,
eq. (4.35), labels a component using a cone $\sigma$ in the original fan
$\Sigma$ that is fixed point by point. It uses the original ray vectors,
including their lattice scale, in the half-ray sum. The implementation now
uses these original cones rather than replacing them by primitive rays of
$\Sigma_L$. This matters because changing a ray to a shorter primitive vector
can change an exact parity calculation. The correction is covered by a
nonprimitive-invariant-pair regression test and is validated by the $h^{1,1}=2$
and $h^{1,1}=3$ reruns; the $h^{1,1}=4$ general-$L$ rows still await the same
rerun. See the [checkpoint review note](./fuzzy_axions_2412_12012_checkpoint_review_20260820.md).

For a fixed component, restrict the hypersurface polynomial $f$ to that
component. A simple parity rule predicts when this restriction is identically
zero:

$$
(\dim\sigma+\lambda_f)\bmod 2=1.
$$

Here $\dim\sigma$ is the number of independent cone directions. In the audit
this rule sets the `f_vanishes_identically` flag. It is only a first
classification. It does not show that a nonzero restriction avoids a
positive-dimensional component, and it does not show that an identically
vanishing component is smooth.

The coefficient test from source eq. (4.45) is evaluated on dual-polytope
vertices $q$. In the code’s phase convention, the required parity is

$$
2\langle t,q\rangle+\lambda_f\equiv 0\pmod 2.
$$

The pairing $\langle t,q\rangle$ is the ordinary lattice/dual-lattice
pairing. The audit includes dual vertices whose facets meet non-simplicial or
non-smooth fan cones, as required by the source extension of this condition.
If those dual vertices are not available, the result is recorded as
“source eq. (4.45) dual-vertex parity evidence is unavailable.” Missing input
is never treated as a successful parity check.

### 5.6 Why positive-dimensional fixed components need extra checks

A fixed component can behave in two qualitatively different ways. The
hypersurface equation can vanish identically on it, in which case the whole
component lies inside $X$. Or the restricted equation can be nonzero, in which
case a generic hypersurface usually intersects it. For a positive-dimensional
component, that intersection can still pass through a singular or orbifold
stratum of the toric space. The audit therefore needs a sufficient certificate
that the restricted hypersurface is well behaved, not only a parity check.

The source argument uses an invariant point in the Kähler cone. Physically,
this is a choice of Kähler parameters preserved by the involution and lying in
the region where the relevant curve and divisor volumes are positive. It also
distinguishes this Kähler condition from smoothness in a possibly singular or
non-simplicial auxiliary fan. The present implementation certifies the
FRST-preserving cases; more general non-simplicial or non-unimodular cases
need an explicit extension of the check.

The implementation performs the following checks:

1. Construct the lower-dimensional quotient or star fan in the lattice fixed
   by $L$.
2. Check that the fixed-cone lattice is exactly integral and saturated, so no
   lattice points have been lost.
3. Check that the quotient cones are full-dimensional, simplicial, and cover
   the required space to form a complete fan.
4. Restrict the ambient anticanonical divisor data to this fan. The relevant
   divisor data must be compatible on overlaps; this is the toric version of
   having a well-defined line bundle.
5. Test nefness. In this setting, nefness means that the restricted line
   bundle has nonnegative degree on the relevant curves. Combinatorially, the
   test is equivalent to checking convexity of the local support functions.
6. Enumerate the lattice points of the restricted section polytope and check
   the positive-dimensional singular strata.

For a divisor $D=\sum_\rho a_\rho D_\rho$, the support-function test gives
exact rational inequalities. A negative margin fails the test. If the fan,
the divisor data, or the finite section-polytope search cannot be certified,
the result is marked unavailable rather than accepted by default.

The last check handles orbifold strata directly. On a positive-dimensional
toric orbit, a restricted section with more than one lattice point is a
non-monomial Laurent polynomial. A generic such polynomial has zeros on the
orbit, so the candidate is rejected by this certificate. A single lattice
point gives a monomial, which is nowhere zero on the torus orbit and is the
avoidance case certified by the test.

During review, the quotient-fan test was strengthened. It now checks primitive
rays, exact pairwise common-face intersections, and a strict positive-hull
condition placing the origin in the interior of the ray hull. A numerically
ambiguous linear-programming margin is rejected conservatively.

### 5.7 Fixed surfaces and the $n_S$ smoothness test

Now consider a two-dimensional fixed surface $S$ on which the hypersurface
equation vanishes identically. The source uses the following characteristic-
class quantity to test whether the resulting surface is free of the isolated
nodal singularities relevant here:

$$
n_S=\int_S c_2\!\left(\mathcal O(K_V^{-1}|_S)\otimes N^*_{S/V}\right).
$$

Here $c_2$ is the second Chern class, $T_V$ and $T_S$ denote the tangent
bundles of the ambient space and surface, and $N^*_{S/V}$ is the conormal
bundle describing how $S$ sits inside the ambient space. Using the toric Euler
sequence and adjunction, the implementation evaluates the equivalent formula

$$
n_S=\int_S\left(c_2(T_V)|_S-c_2(T_S)+c_1(T_S)^2\right).
$$

The integral means: take the degree-four part of the characteristic-class
expression on the complex surface $S$ and evaluate it on $S$. In the source
convention,

- $n_S=0$ passes the smoothness test; and
- $n_S\ne0$ is a source-matched obstruction.

The direction of this test was explicitly checked by regression tests.

For a general, non-diagonal $L$, the code constructs the invariant lattice,
the two-dimensional quotient fan, the toric divisor intersections, and the
restricted ambient divisor data. It checks every ambient cone that supplies
data to the quotient. If two ambient rays map to one quotient ray with
incompatible primitive direction or divisor scale, the code does not silently
choose one. It returns an incomplete-fan reason instead.

An independent real $h^{1,1}=3$ surface calculation is documented in
[fuzzy_axions_2412_12012_general_L_n_s_independent_validation_20260819.md](./fuzzy_axions_2412_12012_general_L_n_s_independent_validation_20260819.md).
For that selected surface, the exported quotient fan and restricted divisor
data independently reproduce

$$
c_2(T_V)|_S=-4,\qquad c_2(T_S)=4,\qquad c_1(T_S)^2=8,\qquad n_S=0.
$$

This is a surface-level validation of the calculation, not a claim about the
full population count.

## 6. Review findings and implemented repairs

This section is a code-and-evidence ledger. It records what the implementation
now guarantees and what remains outside its acceptance boundary. Keep the
following status distinction when reading the JSON output:

- `accepted_verified_orientifold` means that every required test returned a
  passing result for the requested O3/O7 branch;
- a rejection status means that at least one test found a condition that the
  candidate does not satisfy; and
- an unavailable status means that the code could not obtain or certify the
  required evidence. Unavailable is not the same as rejected, and it is not a
  passing result.

Parfit’s implementation added the source §4.6 machinery and the general-$L$
$n_S$ path. A separate read-only review identified four correctness risks
before the final audits:

1. missing dual-vertex parity evidence could flow to a smooth verdict;
2. the complete-fan predicate was too weak and could accept a closed-looking
   but non-covering cone collection;
3. general-$L$ provenance did not validate every ambient record and could
   overwrite duplicate quotient-ray data; and
4. fixed-ray extraction used floating SVD plus bounded rational reconstruction.

The current branch repairs these as follows:

- missing parity evidence is an explicit unavailable status;
- fixed-ray intersections use exact rational nullspaces and primitive integer
  reconstruction;
- the fan certificate checks exact intersections, incidence, primitive rays,
  and vector-space coverage;
- every ambient provenance cone must provide local Cartier data, and all
  induced quotient-ray directions/scales must be compatible; and
- fixtures explicitly distinguish “empty but available dual-vertex evidence”
  from “missing dual-vertex evidence.”

The focused suite in
[scripts/test_inherited_orientifold_candidates.py](../scripts/test_inherited_orientifold_candidates.py)
passes 30/30 tests. The fixed-locus suite
[scripts/test_h21_plus_zero_fixed_locus.py](../scripts/test_h21_plus_zero_fixed_locus.py)
passes 1/1 test. The tests include the following source-sensitive cases:

- $n_S=0$ is smooth and $n_S\ne0$ is rejected;
- missing eq. (4.45) data remains unavailable;
- a smooth nef positive-dimensional fixed component is certified;
- a non-nef restriction is rejected;
- an orbifold stratum with more than one section-polytope lattice point is
  rejected; and
- a first-quadrant cone collection with two incidences per ray is not accepted
  as a complete fan.

The positive-component fixtures are deliberately small synthetic toric
fixtures. The full h11=2,3,4 audits are the real CYTools population checks;
the synthetic fixtures do not replace them.

## 7. Reproducibility and verification commands

All commands below use the local cytools environment and the target worktree.
The three full audits were run in parallel by separate Luna Max workers, with
distinct output files:

    cd /Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse-orientifold-overcount
    source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
    conda activate cytools

    PYTHONDONTWRITEBYTECODE=1 python -B scripts/test_inherited_orientifold_candidates.py
    PYTHONDONTWRITEBYTECODE=1 python -B scripts/test_h21_plus_zero_fixed_locus.py
    PYTHONDONTWRITEBYTECODE=1 python -B scripts/test_reproduce_fuzzy_axions_h11_4.py

    # KS mirror partitions were downloaded from the calabi-yau-data/polytopes-4d
    # Hugging Face dataset. h11=2 is contained in the 05- and 06-vertex
    # partitions (36 favorable); h11=3 in the 05-, 06-, and 07-vertex partitions
    # (243 favorable). Higher-vertex partitions contain no h11=2 or h11=3 rows.

    PYTHONDONTWRITEBYTECODE=1 python -B scripts/reproduce_fuzzy_axions_h11_4.py \
      --h11 2 \
      --parquet-dir /private/tmp/cyax-ks-mirror-h11-2 \
      --orientifold-audit --orientifold-reason-diagnostics --keep-details \
      --output /private/tmp/cyax-orientifold-rerun-h11-2-20260820.json

    PYTHONDONTWRITEBYTECODE=1 python -B scripts/reproduce_fuzzy_axions_h11_4.py \
      --h11 3 \
      --parquet-dir /private/tmp/cyax-ks-mirror-h11-3 \
      --orientifold-audit --orientifold-reason-diagnostics --keep-details \
      --output /private/tmp/cyax-orientifold-rerun-h11-3-20260820.json

    # h11=4 was not rerun in this pass; its rows above remain the 2026-08-19
    # auxiliary-Sigma_L run and are superseded pending a corrected-code rerun.

Integrity checks used after the worker runs included:

    sha256sum /private/tmp/cyax-orientifold-rerun-h11-2-20260820.json
    sha256sum /private/tmp/cyax-orientifold-rerun-h11-3-20260820.json
    # h11=4 (superseded): sha256sum /private/tmp/cyax-orientifold-final-reviewed-h11-4-20260819.json

The observed focused-test result was:

    Ran 38 tests ... OK   # scripts/test_inherited_orientifold_candidates.py
    Ran 1 test ... OK     # scripts/test_h21_plus_zero_fixed_locus.py
    Ran 8 tests ... OK    # scripts/test_reproduce_fuzzy_axions_h11_4.py

The h11=2 and h11=3 audits above were rerun with the corrected general-`L`
code. The full Julia 1.12 package suite (`Pkg.test()`) also passed with exit 0.

The known CYTools cache warning may occur in focused local tests when the
environment denies a cache write. It is non-fatal when the test process and
audit artifact complete. No such warning appeared in the final h11=2,3,4
worker reports.

## 8. Residual limitations and next scientific boundary

The remaining gap is scientifically interpretable and should remain visible:

1. Ambient smoothness boundary. All skipped general-$L$ surface attempts in
   these runs are non_smooth_ambient_cone. The current certificate relies on
   smooth unimodular ambient four-cones for integral local Cartier data. The
   next step is source-compatible Cartier/intersection and orbifold Chow data
   on non-unimodular provenance cones; this implementation does not yet
   perform that work. The separate non-simplicial $L$-symmetric fan work
   requires the §4.3 subdivision/removal/gluing construction.
2. Conservative unavailability. Missing parity, missing provenance,
   incomplete quotient fans, non-nef restrictions, bounded section searches,
   and ambiguous fan-coverage margins are not accepted by default. This is a
   proof-coverage limitation, not an automatic singularity theorem in every
   case.
3. General-$L$ fan scope. The implemented exact quotient certificate handles
   bounded positive-component dimensions 1 through 3. The source discussion of
   the non-simplicial symmetrized fan requires further work if the population
   is to be certified beyond the current auxiliary-fan boundary.
4. Population versus model counts. These runs did not execute the Julia model
   stage, so the Table 1 model-count column is not being claimed. The
   $h^{2,1}_+=0$ benchmark is an independent population gate and is reported
   because it is exactly reproduced; it is not evidence that downstream model
   enumeration has reached 2, 263, or 3348 models.
5. No h11=5 run. The present scope ends at $h^{1,1}=4$, as requested.

The immediate outstanding item is to rerun the h11=4 population with the
corrected general-`L` code so its superseded rows above can be refreshed. The
deeper next scientifically meaningful step is not to relax the acceptance
status: it is to derive and independently validate the source-compatible
Cartier/intersection treatment for the non-smooth ambient provenance sector,
separately assess the non-simplicial symmetric fan, and then rerun the
populations with the new evidence boundary documented.

### 8.1 Suggested student continuation projects

The projects below are ordered from the smallest reproducibility task to the
most geometry-heavy extension. Each project should begin with a small fixture
or one selected Calabi–Yau class, and only then move to a population run.

#### Project 1 — Refresh the h11=4 audit

**Question.** What changes when the corrected original-$\Sigma$ fixed-component
enumeration is applied to h11=4?

**Start with.** Run the focused tests in §7, inspect the h11=2 and h11=3
corrected artifacts, and then run the h11=4 command with a new output path.
Compare the new counts, status totals, schema version, and SHA-256 hash with
the superseded artifact.

**Minimum result.** Produce a new machine-readable artifact and a short note
that updates every h11=4 number in this report. If the run fails, record the
first reproducible failure and its input class; do not replace the old result
with a partial run.

This is a good first project for a student who wants to learn the data flow,
reproducibility practice, and Python tests before changing the geometry.

#### Project 2 — Build a class-level gap ledger

**Question.** Which specific Calabi–Yau classes remain unresolved, and which
geometric test prevents each class from being certified?

**Start with.** Use the candidate records and the existing gap-analysis
scripts to join component-level statuses back to FRST-class identifiers.
Keep separate columns for diagnostic attempts, evidence rows, candidate-linked
components, and distinct CY classes. These quantities are deliberately not
interchangeable.

**Minimum result.** Produce a small JSON or Markdown ledger with one row per
class, a deterministic status priority, and tests for the class-level totals.
The ledger should show whether a class has a genuine unresolved candidate or
only generic diagnostic rows that are not attached to an accepted action.

This project is mainly a data-analysis and software-engineering project, but
it teaches an important scientific lesson: a large intermediate count does
not automatically explain a population-level gap.

#### Project 3 — Extend the certificate beyond smooth ambient cones

**Question.** How should the fixed-surface calculation work when the ambient
four-cone is non-unimodular or singular?

**Start with.** Read the source discussion in Moritz et al. §§4.3–4.6 and
work through one hand-checkable cone. Identify which steps currently rely on a
smooth lattice basis: local divisor data, intersection numbers, Chern-class
computations, and the treatment of orbifold strata.

**Minimum result.** Implement one source-compatible local certificate, validate
it against a hand calculation and a synthetic fixture, and expand the audit
only for the reason `non_smooth_ambient_cone`. A successful project must also
state which singularities remain outside the new certificate.

This is an advanced project in toric geometry. It should never begin by
changing the status rule from “unavailable” to “accepted.” The new status must
follow from a derived and independently checked calculation.

#### Project 4 — Construct the non-simplicial symmetric fan

**Question.** Can the $L$-symmetric fan be built directly using the
subdivision, removal, and gluing construction described in the source?

**Start with.** Implement the construction for one involution and inspect the
result geometrically: list its rays and cones, verify their intersections,
and check whether they cover the intended fixed subspace. Compare the result
with the auxiliary-fan construction used by the current code.

**Minimum result.** Supply a fan validator, a few synthetic examples, and a
comparison report for selected real candidates. Only after these agree should
the new fan be connected to fixed-locus or smoothness decisions.

This project is appropriate for a student who wants a more mathematical
extension. The key deliverable is an independently testable fan construction,
not only a larger orientifold count.

#### Project 5 — Pass validated orientifolds into CYAxiverse.jl

**Question.** What data must the Julia package receive before its axion and
visible-sector calculations can use an orientifold safely?

**Start with.** Design a small result object containing the geometry identity,
exact $L$ and $t$, the $\lambda_f$ branch, the $h^{1,1}_\pm$ split, fixed
components, and the evidence status for every required check. Preserve exact
rational data and the source/schema versions. The current package has
downstream axion and visible-sector routines, but this audit should be treated
as the gate that supplies validated O3/O7 input; it is not a complete brane or
Standard Model construction.

**Minimum result.** Add a documented reader or conversion boundary, round-trip
tests for a small fixture, and one end-to-end example that refuses unresolved
or O5/O9 candidates. Keep the Python audit output and Julia representation
traceable to the same geometry and commit.

This project connects the geometry audit to the scientific package. It is a
good choice for a student interested in software design, exact arithmetic, and
how orientifold data changes the axion effective theory.

For every project, retain the same scientific rules:

- distinguish a rejection from an unavailable proof;
- use exact integer or rational arithmetic where lattice membership or parity
  is involved;
- add a small regression fixture before running a full population; and
- record the source equations, input artifact, code commit, dependency
  versions, and verification command for every new claim.

## 9. References and source links

- Moritz et al., [Orientifolds of Calabi–Yau hypersurfaces (arXiv:2305.06363)](https://arxiv.org/abs/2305.06363), with the [HTML text](https://arxiv.org/html/2305.06363). The fixed-locus construction, coefficient parity, fixed-surface condition, and $n_S$ smoothness criterion used here are from §§4.3–4.6, especially eqs. (4.26), (4.35), (4.45), (4.48), and (4.50).
- Sheridan et al., [Fuzzy Axions and Associated Relics (arXiv:2412.12012)](https://arxiv.org/abs/2412.12012), with the [HTML text and Table 1](https://arxiv.org/html/2412.12012). Table 1 supplies the favorable, FRST, inherited-orientifold, $h^{1,1}_-=0$, $h^{2,1}_+=0$, and model-count comparison targets.
- [validation/fuzzy_axions_2412_12012_general_L_n_s_machinery_20260819.md](./fuzzy_axions_2412_12012_general_L_n_s_machinery_20260819.md), the implementation handoff for the general-$L$ $n_S$ calculation.
- [validation/HANDOFF_general_L_followups_20260819.md](./HANDOFF_general_L_followups_20260819.md), the diagnostic and follow-up handoff.
- [validation/fuzzy_axions_2412_12012_general_L_n_s_independent_validation_20260819.md](./fuzzy_axions_2412_12012_general_L_n_s_independent_validation_20260819.md), an independent real h11=3 surface calculation.
