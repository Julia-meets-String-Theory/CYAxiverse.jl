# Final inherited-orientifold audit: h11 = 2, 3, 4

Date: 2026-08-20
Branch: codex/orientifold-overcount-20260819
Scope: final post-review full-population audits for favorable Kreuzer–Skarke populations at \(h^{1,1}=2,3,4\)
Implementation: [scripts/inherited_orientifold_candidates.py](../scripts/inherited_orientifold_candidates.py)
Audit driver: [scripts/reproduce_fuzzy_axions_h11_4.py](../scripts/reproduce_fuzzy_axions_h11_4.py)

## 1. Executive conclusion

The final audits completed successfully for all three Hodge numbers in the
reviewed orientifold worktree. Each worker exited with status 0, produced a
valid JSON artifact, and reported no CYTools cache-save warning. The base
population and the independent \(h^{2,1}_+=0\) trilayer benchmark agree with
the corresponding rows of Table 1 in Sheridan et al. The inherited-orientifold
audit itself remains a conservative lower count:

| \(h^{1,1}\) | favorable polytopes | final FRST classes | audited inherited O3/O7 classes | Table 1 inherited target | audited \(h^{1,1}_-=0\) classes | Table 1 \(h^{1,1}_-=0\) target | audited trilayer \(h^{2,1}_+=0\) classes | Table 1 trilayer target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 36 | 10 | 32 | 10 | 32 | 11 | 11 |
| 3 | 243 | 274 | 81 | 253 | 81 | 253 | 66 | 66 |
| 4 | 1185 | 1760 | 435 | 1559 | 429 | 1554 | 267 | 267 |

The audited orientifold numbers are code-certified class-level counts, not a
claim that the paper’s Table 1 is incorrect. The gap means that the present
implementation has not yet certified every source-paper orientifold using the
evidence available to it. In particular, every skipped general-\(L\)
fixed-surface attempt in these runs has the machine-readable reason
non_smooth_ambient_cone: the provenance cone is not a smooth unimodular
ambient four-cone, so the current source-matched Cartier/\(n_S\) certificate
refuses to promote that candidate.

The reviewed implementation materially increased the certified inherited
population relative to the pre-review baseline (0, 4, 5 at
\(h^{1,1}=2,3,4\)) to (10, 81, 435). It still does not close the Table 1 gap,
and the remaining records must not be relabelled as accepted merely because
the code could not finish a sufficient smoothness proof.

## 2. What was audited

The audit reads the favorable \(N\)-lattice mirror of the relevant
Kreuzer–Skarke polytopes, enumerates all FRSTs, quotients them into FRST
classes using the source-matched two-face relation, and evaluates two related
but distinct populations:

1. the special trilayer involution and its independent
   \(h^{2,1}_+(X,\mathcal I)=0\) diagnostic; and
2. the full inherited-orientifold search over lattice involutions \(L\), torus
   shifts, and the two coefficient-parity choices \(\lambda_f\), with the
   fixed-locus smoothness machinery from Moritz et al. enabled.

The audit counts a CY class as inherited only if at least one class-level
candidate reaches accepted_verified_orientifold with \(\lambda_f=1\), the
O3/O7 convention used for Table 1. It counts the \(h^{1,1}_-=0\) subset only
when that accepted candidate also has \(h^{1,1}_-=0\). An O5/O9
\(\lambda_f=0\) candidate is not allowed to satisfy the Table 1 O3/O7 row.

The full machine-readable output contains per-polytope details and the
opt-in general-\(L\) diagnostics. The output schema is
cyaxiverse-fuzzy-axions-h11-4-reproduction-1.0; candidate records use
cyaxiverse-inherited-orientifold-candidate-2.2, and the nested general-\(L\)
diagnostics use cyaxiverse-general-L-fixed-surface-diagnostics-1.0.

## 3. Final audit artifacts and integrity evidence

The JSON files are in /private/tmp because they are large generated artifacts
rather than package source files. Their sizes and SHA-256 hashes are:

| \(h^{1,1}\) | artifact | bytes | SHA-256 |
|---:|---|---:|---|
| 2 | [cyax-orientifold-final-reviewed-h11-2-20260819.json](/private/tmp/cyax-orientifold-final-reviewed-h11-2-20260819.json) | 54,557,352 | 27e7b18148bc6730089c86fabb99bd2dae0ff8c91ed721ce302f60106feb647a |
| 3 | [cyax-orientifold-final-reviewed-h11-3-20260819.json](/private/tmp/cyax-orientifold-final-reviewed-h11-3-20260819.json) | 286,183,888 | 9de41a8967895c3f0de36317b83ce44bb978ae8971b84c573dc30795e2c6dc9f |
| 4 | [cyax-orientifold-final-reviewed-h11-4-20260819.json](/private/tmp/cyax-orientifold-final-reviewed-h11-4-20260819.json) | 1,365,317,574 | a5849fa7c02040dc4986653168262eac4d299a492d58fe7728e948afcd875769 |

Independent post-run checks confirmed, for every artifact:

- input.population_complete is true;
- the requested \(h^{1,1}\) is 2, 3, or 4 as appropriate;
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
| trilayer \(h^{2,1}_+=0\) FRST classes | 11 | 66 | 267 |
| identity torus actions | 251 | 2296 | 13218 |
| identity-action CY classes | 36 | 274 | 1760 |
| identity-valid O3/O7 CY classes | 25 | 198 | 1153 |
| source-evidence inherited O3/O7 classes | 10 | 81 | 435 |
| source-evidence \(h^{1,1}_-=0\) classes | 10 | 81 | 429 |

The identity-valid O3/O7 row is a useful diagnostic, but it is not the Table
1 inherited count. Table 1 asks whether a CY admits an inherited orientifold;
the full audit includes nonidentity \(L\) and torus-shift sectors as well.

### 3.2 General-\(L\) fixed-surface diagnostics

| \(h^{1,1}\) | surface attempts | certified surfaces | skipped surfaces | skipped reason | unresolved candidate components |
|---:|---:|---:|---:|---|---:|
| 2 | 1362 | 862 | 500 | non_smooth_ambient_cone (500) | 33 |
| 3 | 6314 | 3934 | 2380 | non_smooth_ambient_cone (2380) | 117 |
| 4 | 25398 | 17024 | 8374 | non_smooth_ambient_cone (8374) | 327 |

The word “certified” in this table refers to a fixed-surface \(n_S\) evidence
record. It does not by itself mean that the whole orientifold candidate is
accepted: all fixed components, coefficient parity, the \(h^{1,1}_-\) filter,
and the O3/O7 condition must also pass. Conversely, an unresolved component
is not a proof of a singular orientifold; it is a proof that this
implementation does not possess the sufficient source evidence needed to
accept it.

## 4. Comparison with Sheridan et al. Table 1

The relevant source is Sheridan et al., [Fuzzy Axions from String
Compactifications (arXiv:2412.12012)](https://arxiv.org/abs/2412.12012),
especially its [HTML version and Table 1](https://arxiv.org/html/2412.12012).
The Table 1 population numbers are:

| \(h^{1,1}\) | favorable | FRST | inherited | inherited with \(h^{1,1}_-=0\) | inherited with \(h^{1,1}_-=0,\ h^{2,1}_+=0\) | models |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 36 | 32 | 32 | 11 | 2 |
| 3 | 243 | 274 | 253 | 253 | 66 | 263 |
| 4 | 1185 | 1760 | 1559 | 1554 | 267 | 3348 |

The present audit reproduces the source base populations exactly and the
\(h^{2,1}_+=0\) trilayer counts exactly, but its source-evidence orientifold
counts are lower:

| \(h^{1,1}\) | inherited gap (Table 1 − audit) | \(h^{1,1}_-=0\) gap | trilayer gap |
|---:|---:|---:|---:|
| 2 | 22 | 22 | 0 |
| 3 | 172 | 172 | 0 |
| 4 | 1124 | 1125 | 0 |

This comparison should be interpreted as a certification-boundary result.
The audit requires an explicit, source-matched chain of evidence for every
fixed component. It does not infer smoothness from the absence of a detected
problem, and it does not turn smoothness_verification_unavailable into an
acceptance. The residual gap is therefore evidence about the current
implementation’s proof coverage, not evidence against the published Table 1
counts.

The final counts are nevertheless a substantial improvement over the
pre-review baseline produced before the nef/orbifold implementation and the
review fixes: 0, 4, and 5 inherited classes at \(h^{1,1}=2,3,4\), respectively.
The increases to 10, 81, and 435 show that the added evidence is active, while
the remaining gap shows that it is not yet complete.

## 5. Mathematical and physical framework

This section gives the concepts needed to interpret the audit at the level of
a senior PhD student or early postdoc working in string compactifications and
cosmology.

### 5.1 Reflexive polytopes, toric fourfolds, and the CY hypersurface

Let \(N\simeq\mathbb Z^4\) be the toric one-parameter-subgroup lattice and
\(M=N^\vee\) its dual. A four-dimensional reflexive polytope
\(\Delta\subset N_\mathbb R\) has a polar dual
\(\Delta^\circ\subset M_\mathbb R\) with the origin as its unique interior
lattice point. A suitable fan \(\Sigma\), obtained from a fine, regular,
star triangulation (FRST) of the boundary data, defines a toric fourfold
\(V_\Sigma\). Its anticanonical divisor is

\[
-K_{V_\Sigma}=\sum_{\rho\in\Sigma(1)}D_\rho.
\]

A generic anticanonical section \(f\in H^0(V_\Sigma,-K_{V_\Sigma})\)
defines a Calabi–Yau threefold

\[
X=\{f=0\}\subset V_\Sigma,
\]

by adjunction, because \(K_X=(K_{V_\Sigma}+X)|_X=0\). In physics, \(X\)
provides the internal space for the four-dimensional compactification; its
Kähler and complex-structure data determine, among other things, axion decay
constants, instanton actions, and the Hodge-theoretic orientifold projection.

The audit works with the favorable \(N\)-lattice populations used in
Sheridan et al. “Favorable” means, in this context, that the relevant divisor
classes are inherited from the toric ambient space sufficiently directly for
the database construction and intersection calculations used by the source
pipeline. It is a property of the chosen presentation/population, not a
universal statement that every divisor on \(X\) is ambient in every phase.

### 5.2 FRSTs, equivalence classes, and what is being counted

An FRST is a simplicial, crepant subdivision of the relevant toric boundary
data. Its maximal four-cones give local affine toric charts. A smooth FRST
has unimodular maximal cones, so the corresponding affine charts are smooth.

Several triangulations can define the same database-level CY class under the
source’s equivalence relation. The audit therefore reports both raw FRST
counts and final FRST-class counts. Table 1 compares with the class-level
numbers. The orientifold audit also deduplicates accepted triples at the
class level: one accepted action is enough for a class to enter the inherited
count.

This distinction matters for interpreting the diagnostic surface counts. A
number such as 17024 certified surfaces at \(h^{1,1}=4\) is not 17024 CYs and
not 17024 accepted orientifolds. It is a count of fixed-surface evidence rows
encountered while examining candidate data.

### 5.3 Inherited lattice involutions and torus shifts

An inherited orientifold starts with an integral lattice involution

\[
L:N\longrightarrow N,\qquad L^2=I,
\]

that preserves the polytope and the relevant fan/divisor data. It is combined
with a torus translation:

\[
\widetilde I=\phi_{[t]}\circ L.
\]

The \(+1\) and \(-1\) projectors are

\[
P_\pm^L=\frac{1}{2}(I\pm L).
\]

The source shift search is organized by the projected lattices, schematically

\[
H_+^L=\frac{P_+^L(N)}{2P_+^L(N)},\qquad
H_-^L=\frac{P_-^L(N)}{2P_-^L(N)},
\]

with exact integral-lattice arithmetic. The square of the affine action carries
the translation represented by \(2t\); the audit therefore rejects a shift
coset unless the required \(2t\in N\) condition for a genuine \(\mathbb Z_2\)
involution is satisfied. This is why the code stores the representative
carefully as an exact rational rather than rounding quarter- or half-integral
components.

The implementation uses Smith/Hermite normal-form calculations for lattice
kernels and quotient membership. This avoids a bounded coefficient search
that can miss valid lattice identifications after conjugating a diagonal
involution by a non-permutation \(GL(4,\mathbb Z)\) basis change.

### 5.4 \(\lambda_f\), the holomorphic three-form, and O-plane types

The coefficient-parity parameter \(\lambda_f\in\{0,1\}\) records the sign of
the holomorphic three-form in the source convention. In this audit,

- \(\lambda_f=1\) is the O3/O7 branch used for the Sheridan et al. Table 1
  inherited population;
- \(\lambda_f=0\) is the O5/O9 branch and is retained in the candidate search
  for completeness but does not satisfy the Table 1 O3/O7 count.

The identity sanity contract is correspondingly asymmetric. \(L=I,t=0\) is
kept as a trivial smooth fixture for \(\lambda_f=0\). For \(\lambda_f=1\), the
same zero-shift action would force all hypersurface coefficients to vanish in
the source phase convention, so it is not accepted as a physical generic
O3/O7 hypersurface.

The involution acts on \(H^{1,1}(X)\), giving

\[
H^{1,1}(X)=H^{1,1}_+(X)\oplus H^{1,1}_-(X),
\qquad
h^{1,1}=h^{1,1}_++h^{1,1}_-.
\]

The \(h^{1,1}_-=0\) filter is an additional population condition. It is not
equivalent to accepting the orientifold, and it should be applied only after
the O3/O7 and smoothness checks.

### 5.5 Fixed loci, the auxiliary fan, and coefficient parity

The toric fixed-locus construction uses the intersections of the ambient
cones with the \(L\)-fixed real subspace. These intersections form the finite
auxiliary fan \(\Sigma_L\), the construction associated with eq. (4.26) of
Moritz et al. A fixed component is labelled by a cone \(\sigma\in\Sigma_L\)
and a projected-lattice representative \(\nu\). The integrality condition is
the source eq. (4.35) condition implemented by the exact rational arithmetic in
the candidate enumerator.

The restriction of the hypersurface polynomial to a fixed component can
vanish identically for a parity reason. The current implementation records
that combinatorial condition as

\[
(\dim\sigma+\lambda_f)\bmod 2=1,
\]

which is the source-matched f_vanishes_identically flag. This flag is only a
first fixed-locus classification. It does not prove that a non-vanishing
positive-dimensional component is avoided by a generic hypersurface, nor does
it prove that an identically vanishing component is smooth.

For the coefficient test, the source eq. (4.45) condition is evaluated on
dual vertices \(q\), with the exact parity combination

\[
2\langle t,q\rangle+\lambda_f\equiv 0\pmod 2
\]

in the code’s phase convention. The implementation also supplies dual
vertices whose dual facets meet non-simplicial or non-smooth fan cones, as
required by the source extension of the condition. If the dual-vertex input
is absent, the result is now explicitly
source eq. (4.45) dual-vertex parity evidence is unavailable; it is never
silently treated as a successful parity check.

### 5.6 Positive-dimensional non-vanishing components: nefness and orbifolds

This is the point behind the earlier statement that “the patch
conservatively leaves positive-dimensional non-vanishing fixed components
unavailable without the source’s nef/orbifold evidence.” Suppose a fixed
component \(F\) has positive toric dimension and the restricted hypersurface
section does not vanish identically on \(F\). To accept the candidate, one
needs a sufficient reason that a generic section of the restricted
anticanonical line bundle is well behaved on \(F\), including at orbifold
strata.

The implementation therefore:

1. constructs the quotient/star fan in the invariant fixed lattice;
2. checks exact integrality and saturation of the fixed-cone lattice;
3. checks that the quotient cones are full-dimensional, simplicial, and form
   a complete fan;
4. restricts the ambient anticanonical Cartier data to the quotient fan;
5. tests nefness by exact support-function inequalities; and
6. enumerates the restricted section-polytope lattice points and checks the
   positive-dimensional singular strata.

For a toric divisor \(D=\sum_\rho a_\rho D_\rho\), nefness can be tested by
whether the local support functions are convex, equivalently whether the
local Cartier representatives lie in the associated divisor polytope. The
code records the exact rational inequality margins and rejects a negative
margin. If the fan, Cartier provenance, or bounded section-polytope search
cannot be certified, the result is unavailable.

For an orbifold cone whose orbit has positive dimension, the restricted
section must not have a generic zero on that orbit. The exact section-polytope
test records the lattice points on the corresponding face. A face with more
than one lattice point supports a non-monomial Laurent restriction, so a
generic section intersects the stratum and the candidate is rejected. A
single lattice point is the monomial/avoidance case certified by this test.

The quotient-fan predicate was strengthened during review. It no longer
accepts a collection merely because every codimension-one face appears twice:
it also checks primitive rays, exact pairwise common-face intersections, and a
strict positive-hull condition placing the origin in the interior of the ray
hull. Numerically ambiguous LP margins are rejected conservatively.

### 5.7 Identically vanishing fixed surfaces and \(n_S\)

For a two-dimensional fixed surface \(S\) on which the hypersurface
restriction vanishes identically, the source’s eq. (4.50) condition is encoded
through

\[
n_S=\int_S c_2\!\left(\mathcal O(K_V^{-1}|_S)\otimes N_{S/V}\right).
\]

Using the toric Euler sequence and adjunction, the implementation evaluates
the equivalent expression

\[
n_S=\int_S\left(c_2(T_V)|_S-c_2(T_S)+c_1(T_S)^2\right).
\]

In the source convention, \(n_S=0\) is the required smoothness condition for
avoiding isolated nodal points on the fixed surface; \(n_S\ne0\) is a
source-matched obstruction. This polarity was explicitly regression-tested.

For general \(L\), the code constructs the invariant lattice, the
two-dimensional quotient fan, toric divisor intersections, and restricted
ambient Cartier coefficients. Every ambient provenance cone is checked. If
two ambient rays map to one quotient ray with incompatible primitive direction
or divisor scale, the code refuses to overwrite one record with another and
returns an incomplete-fan reason.

An independent real h11=3 surface calculation is documented in
[fuzzy_axions_2412_12012_general_L_n_s_independent_validation_20260819.md](./fuzzy_axions_2412_12012_general_L_n_s_independent_validation_20260819.md).
For that selected surface, the exported quotient fan and restricted Cartier
coefficients independently reproduce

\[
c_2(T_V)|_S=-4,\qquad c_2(T_S)=4,\qquad c_1(T_S)^2=8,\qquad n_S=0.
\]

That validation is a surface-level check; it is not itself a population-count
claim.

## 6. Review findings and implemented repairs

Parfit’s implementation added the source §4.6 machinery and the general-\(L\)
\(n_S\) path. A separate read-only review identified four correctness risks
before the final audits:

1. missing dual-vertex parity evidence could flow to a smooth verdict;
2. the complete-fan predicate was too weak and could accept a closed-looking
   but non-covering cone collection;
3. general-\(L\) provenance did not validate every ambient record and could
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

- \(n_S=0\) is smooth and \(n_S\ne0\) is rejected;
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

    PYTHONDONTWRITEBYTECODE=1 python -B scripts/reproduce_fuzzy_axions_h11_4.py \
      --h11 2 \
      --parquet-dir /private/tmp/cyax-ks-mirror-h11-4 \
      --orientifold-audit --orientifold-reason-diagnostics --keep-details \
      --julia-binary /Users/vmehta/.juliaup/bin/julia \
      --output /private/tmp/cyax-orientifold-final-reviewed-h11-2-20260819.json

    PYTHONDONTWRITEBYTECODE=1 python -B scripts/reproduce_fuzzy_axions_h11_4.py \
      --h11 3 \
      --parquet-dir /private/tmp/cyax-ks-mirror-h11-4 \
      --orientifold-audit --orientifold-reason-diagnostics --keep-details \
      --julia-binary /Users/vmehta/.juliaup/bin/julia \
      --output /private/tmp/cyax-orientifold-final-reviewed-h11-3-20260819.json

    PYTHONDONTWRITEBYTECODE=1 python -B scripts/reproduce_fuzzy_axions_h11_4.py \
      --h11 4 \
      --parquet-dir /private/tmp/cyax-ks-mirror-h11-4 \
      --orientifold-audit --orientifold-reason-diagnostics --keep-details \
      --julia-binary /Users/vmehta/.juliaup/bin/julia \
      --output /private/tmp/cyax-orientifold-final-reviewed-h11-4-20260819.json

Integrity checks used after the worker runs included:

    sha256sum /private/tmp/cyax-orientifold-final-reviewed-h11-2-20260819.json
    sha256sum /private/tmp/cyax-orientifold-final-reviewed-h11-3-20260819.json
    sha256sum /private/tmp/cyax-orientifold-final-reviewed-h11-4-20260819.json

The observed focused-test result was:

    Ran 30 tests ... OK
    Ran 1 test ... OK

The known CYTools cache warning may occur in focused local tests when the
environment denies a cache write. It is non-fatal when the test process and
audit artifact complete. No such warning appeared in the final h11=2,3,4
worker reports.

## 8. Residual limitations and next scientific boundary

The remaining gap is scientifically interpretable and should remain visible:

1. Ambient smoothness boundary. All skipped general-\(L\) surface attempts in
   these runs are non_smooth_ambient_cone. The current certificate relies on
   smooth unimodular ambient four-cones for integral local Cartier data. A
   non-unimodular provenance cone may still be resolvable after a more general
   orbifold/crepant analysis, but this implementation does not perform that
   analysis.
2. Conservative unavailability. Missing parity, missing provenance,
   incomplete quotient fans, non-nef restrictions, bounded section searches,
   and ambiguous fan-coverage margins are not accepted by default. This is a
   proof-coverage limitation, not an automatic singularity theorem in every
   case.
3. General-\(L\) fan scope. The implemented exact quotient certificate handles
   bounded positive-component dimensions 1 through 3. The source discussion of
   the non-simplicial symmetrized fan requires further work if the population
   is to be certified beyond the current auxiliary-fan boundary.
4. Population versus model counts. These runs did not execute the Julia model
   stage, so the Table 1 model-count column is not being claimed. The
   \(h^{2,1}_+=0\) benchmark is an independent population gate and is reported
   because it is exactly reproduced; it is not evidence that downstream model
   enumeration has reached 2, 263, or 3348 models.
5. No h11=5 run. The present scope ends at \(h^{1,1}=4\), as requested.

The next scientifically meaningful step is therefore not to relax the
acceptance status. It is to implement and independently validate the source’s
orbifold/crepant treatment for the non-smooth ambient provenance sector, then
rerun the three populations with that new evidence boundary documented.

## 9. References and source links

- Moritz et al., [Orientifolds of Calabi–Yau hypersurfaces (arXiv:2305.06363)](https://arxiv.org/abs/2305.06363), with the [HTML text](https://arxiv.org/html/2305.06363). The fixed-locus construction, coefficient parity, fixed-surface condition, and \(n_S\) smoothness criterion used here are from §§4.3–4.6, especially eqs. (4.26), (4.35), (4.45), (4.48), and (4.50).
- Sheridan et al., [Fuzzy Axions from String Compactifications (arXiv:2412.12012)](https://arxiv.org/abs/2412.12012), with the [HTML text and Table 1](https://arxiv.org/html/2412.12012). Table 1 supplies the favorable, FRST, inherited-orientifold, \(h^{1,1}_-=0\), \(h^{2,1}_+=0\), and model-count comparison targets.
- [validation/fuzzy_axions_2412_12012_general_L_n_s_machinery_20260819.md](./fuzzy_axions_2412_12012_general_L_n_s_machinery_20260819.md), the implementation handoff for the general-\(L\) \(n_S\) calculation.
- [validation/HANDOFF_general_L_followups_20260819.md](./HANDOFF_general_L_followups_20260819.md), the diagnostic and follow-up handoff.
- [validation/fuzzy_axions_2412_12012_general_L_n_s_independent_validation_20260819.md](./fuzzy_axions_2412_12012_general_L_n_s_independent_validation_20260819.md), an independent real h11=3 surface calculation.
