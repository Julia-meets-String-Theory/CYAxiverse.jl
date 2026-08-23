# Exact-action orientifold formula ledger

Date: 2026-08-22  
Status: source validation and approved implementation complete  
Scope: inherited holomorphic involutions of anticanonical Calabi--Yau
hypersurfaces in toric fourfolds, and the population convention used by the
fuzzy-axion study.

The bounded MPCP/refinement extension is documented separately in
[`validation/mpcp_refinement_ledger_20260823.md`](mpcp_refinement_ledger_20260823.md).
That ledger adds the Batyrev height-one/MPCP conventions, CYTools 1.4.12
decision table, refined `Q'` contract, source-keyed global/local point mapping,
facet-interior certificates, and the three-index replay boundary;
the formulas below remain the authoritative exact-action and fixed-locus
ledger.

## Decision

The formulas needed to recompute the fixed locus and the favorable-hypersurface
Hodge split from an exact witness `(L,t,lambda_f)` are source-verified. The
sign in the Lefschetz formula and the identification `lambda_f = 1` with the
O3/O7 branch are fixed by the source equations; neither is a free convention.

The historical special-shift diagnostic is not an acceptance computation.
Identity actions use the exact canonical `p0=2t` kernel. General actions use
the exact component chain and the support-aware Euler derivation below; any
missing, singular, or orbifold component evidence remains terminal
`unavailable`.

## Primary sources and immutable local copies used for this audit

1. Jakob Moritz, *Orientifolding Kreuzer-Skarke*, arXiv:2305.06363v1
   (2023), especially §§4.4--4.6. The source archive was downloaded from
   `https://export.arxiv.org/e-print/2305.06363v1`; archive SHA-256
   `9e36c8a0a9fa9c5e876329fc6f1d56b46896125842da5f5a44652ea16e48a992`.
   TeX anchors below refer to `KS_orientifolds.tex` in that archive.
2. Elijah Sheridan et al., *Fuzzy Axions and Associated Relics*,
   arXiv:2412.12012v1 (2024), §4.1, Table 1, and §4.2.1. The locally supplied
   archive `/Users/vmehta/Downloads/fuzzy-2412.12012v1.tar.gz` has SHA-256
   `905db55f2ab72e2b94ba9175148cd5a4976756e95ce37e64622bffdbc4d7bcea`.
   TeX anchors refer to `main.tex` in that archive.
3. V. I. Danilov and A. G. Khovanskii, *Newton Polyhedra and an Algorithm
   for Computing Hodge--Deligne Numbers*, Math. USSR-Izvestiya **29** (1987),
   279--298, especially §§3.2 and 5.1. The primary English PDF is
   [available from the University of Toronto](https://www.math.toronto.edu/askold/1986-Izvestiya-5-english-s-Danilovym.pdf);
   the bibliographic record and DOI are
   [MathNet, DOI 10.1070/IM1987v029n02ABEH000970](https://www.mathnet.ru/eng/im1541).
   The exact PDF text anchors used here are lines 512--517 and 706--715.
4. David A. Cox, John B. Little, and Henry K. Schenck, *Toric Varieties*,
   Graduate Studies in Mathematics 124, AMS (2011), first edition. Exact
   anchors below are Theorem 8.1.6, Lemma 12.5.2, Theorem 12.5.3,
   Example 13.1.1, equation (13.1.2), and Proposition 13.1.2.

The arXiv identifiers link to the public primary records:
[2305.06363](https://arxiv.org/abs/2305.06363) and
[2412.12012](https://arxiv.org/abs/2412.12012). No secondary source supplies a
formula in this ledger.

## Source-anchor correction for the MPCP replay

Moritz's FRST convention is the direct statement at
`KS_orientifolds.tex` line 245: the triangulation ignores lattice points
interior to facets. The anticanonical monomial construction and its dual-face
interpretation are in lines 285--292. For the bounded source rows, the parent
polytope therefore retains eight global points while CYTools returns seven
boundary FRST points. Point 7 is certified to be interior to a facet by the
facet point/vertex lists. The implication that the omitted ray does not add a
generic anticanonical divisor is **derived** from the monomial exponent
`<p,q>+1`, the dual-face dimension, and Batyrev Proposition 4.4.1/Theorem
4.4.2; it is not attributed as a direct CYTools theorem.

Sheridan et al.'s relevant setup is `main.tex` lines 1260--1282 (favorable
FRST-class population), 1295--1303 (class versus orientifold counting), and
1337--1349 (the h11=2 worked geometry and split). Those lines define source
scope and comparison conventions; they do not select a refinement by a Hodge
target.

The replay source gate now requires exact `polytope_id`, source row, global
coordinates, expected point/boundary counts, and `(h11,h21,chi)` before any
CYTools geometry is constructed. Selected simplices must declare
`simplices_index_space` (`triangulation_local` or `polytope_global`) and local
indices are mapped to the frozen global parent by exact coordinate lookup.
Source ID, coordinate, point-count, Hodge, dual-parity, and index-space
mismatches are terminal. This is an implementation acceptance contract, not a
new scientific result.

## Notation and package mapping

| Source symbol | Meaning | Package/sibling symbol |
| --- | --- | --- |
| `N`, `M` | rank-four fan lattice and its dual | integer ray vectors; integer dual-polytope vertices |
| `Sigma` | original ambient toric fan | `triangulation_cones` |
| `L` | integral involutive lattice automorphism | `lattice_matrix` / `matrix` |
| `t` | algebraic-torus shift, with `2t in N` | ledger `torus_shift`; exact `Fraction` tuple in the sibling implementation |
| `lambda_f in Z_2` | hypersurface-polynomial covariance choice | ledger `lambda_f` |
| `P_+^L`, `P_-^L` | `(id+L)/2`, `(id-L)/2` | projected-lattice enumeration with `sign=+1,-1` |
| `nu in H_-^L` | disconnected fixed-component phase label | fixed-component `nu` |
| `sigma` | pointwise-`L`-invariant cone of the original `Sigma` | `sigma_rays` from `_pointwise_invariant_cone_keys` |
| `F_tildeI(sigma,nu)` | ambient fixed component | `fixed_components` record |
| `F_I` | union of induced fixed components in `X` | identity-action diagnostic; general-`L` unavailable |
| `chi(F_I)`, `chi(X)` | topological Euler characteristics | identity-action diagnostic and eq. (4.51) inputs |

All fixed-component statements below assume that `L` preserves the relevant
fan and that `(L composed with phi_[t])^2=id`. In the source gauge this requires
`2t in N` (Moritz lines 420--438). The favorable Hodge formula additionally
assumes that `H_2(X,Z) -> H_2(V,Z)` is an isomorphism. Smoothness claims require
both Kähler smoothness and complex-structure smoothness (lines 605--619).

## Formula ledger

### Auxiliary fan: Moritz eq. (4.26)

Source: §4.4, TeX lines 491--506. The symbolic formula is

```tex
\Sigma_L:=\{\sigma \cap N^L_{\mathbb R}\}_{\sigma\in\Sigma}.
```

Mapping: `build_auxiliary_fan(triangulation_cones, matrix)` intersects each
ambient cone with the `+1` eigenspace of `L`. `Sigma_L` supplies component
geometry and normal-direction evidence. It does **not** replace the original
fan as the universe of `sigma` labels in eq. (4.35).

Assumptions: `L Sigma=Sigma`; exact rational intersections; rays are primitive
in the appropriate lattice. The auxiliary fan may be simplicial but need not
be smooth (source lines 498--508).

### Fixed-component labels and conditions: Moritz eqs. (4.34)--(4.35)

Source: §4.4, derivation in lines 511--549 and summary in lines 551--555.
The phase-label group and smooth-normal-bundle condition are

```tex
H_-^L:=P_-^L(N)/(2P_-^L(N)),
```

```tex
(\nu,\sigma)\in H_-^L\times\Sigma:\quad
t+\nu+\frac12\sum_{p\in\sigma(1)}p\in N,
```

with `sigma` pointwise invariant under `L`.

Mapping: `enumerate_projected_lattice_representatives(matrix,-1)` supplies
`nu`; `_pointwise_invariant_cone_keys` supplies original-`Sigma` cones without
primitive rescaling; `_fixed_component_records` applies the half-ray formula
only when `_half_ray_shortcut_proof` certifies the source's smooth-normal
precondition. Otherwise the sibling code falls back to the general quotient
condition derived from the phase equation at TeX lines 522--539.

Algebraic check: multiplying the eq. (4.35) condition by two gives
`2t+2nu+sum sigma(1) in 2N`. This is the integer parity form used by an exact
implementation; it does not permit replacing `t` by the stored representative
of `2t`. For `L=id`, `H_-^L` is trivial and the formula reduces to Moritz's
eq. (4.36), TeX lines 571--579.

Assumptions: the displayed shortcut is for components with smooth normal
bundle. Component containment must be removed as described at lines 541--549.

### Hypersurface covariance and parity: Moritz eqs. (4.41)--(4.46)

Source: §4.5 and §4.6, TeX lines 584--646.

The complete six-equation chain, with source anchors, is:

1. Eq. (4.41), lines 587--590: the hypersurface zero set transforms
   covariantly,

```tex
f\mapsto\gamma_f f,
```

2. Eq. (4.42), lines 591--595: direct action on the anticanonical polynomial,

```tex
f=\sum_q\psi_qs_q\mapsto
e^{2\pi i\sum_p\eta_p}\sum_q
\psi_{L(q)}e^{2\pi i\langle t,q\rangle}s_q.
```

3. Eq. (4.43), lines 597--601: the resulting coefficient relation,

```tex
\psi_{L(q)}=e^{2\pi i(\langle t,q\rangle+\lambda_f/2)}\psi_q.
```

4. Eq. (4.44), lines 621--624: when a required vertex monomial is projected
   out, the source identifies the dangerous factorization

```tex
f=x_p f'.
```

5. Eq. (4.45), lines 625--629: for an `L`-fixed dual vertex needed by the
   source smoothness test,

```tex
2\langle t,q\rangle+\lambda_f=0\pmod 2.
```

6. Eq. (4.46), lines 631--636: on an ambient fixed component,

```tex
f|_{\mathcal F_{\tilde{\mathcal I}}(\sigma,\nu)}\equiv0
\quad\hbox{if}\quad \dim(\sigma)+\lambda_f=1\pmod2.
```

Mapping: `_dual_vertex_parity_evidence` evaluates the pairing with exact
rationals; `lambda_f` is part of each exact live action record and must remain
unchanged. `_fixed_component_records` applies eq. (4.46) only after the
smooth-half-ray certificate described below, and otherwise uses the exact
invariant restricted support prescribed in the next section. Thus the
component is classified as contained only when its actual covariant support
is empty.

Algebraic O-plane derivation: `X` has complex dimension three. If
`lambda_f=1`, eq. (4.46) gives an even-dimensional fixed locus in `X` for
both parities of `dim(sigma)`: an identically contained component has dimension
`4-dim(sigma)` when `dim(sigma)` is even, and a transverse section has
dimension `3-dim(sigma)` when it is odd. These are surfaces or points, hence
O7/O3 loci by Moritz §2, TeX lines 215--220. Thus the package mapping
`lambda_f=1 -> O3/O7` is derived, not assumed. `lambda_f=0` similarly gives
odd-dimensional fixed loci and the O5/O9 branch.

Assumptions: the coefficient relation is applied to the exact action on dual
lattice points (`L` acts contragrediently on `M`); the polynomial is a generic
covariant anticanonical section after the required coefficient tuning.

### Invariant restricted support boundary (owner-approved derived implementation)

Moritz's anticanonical monomial is defined in TeX lines 285--296,
`s_q=\prod_p x_p^{\langle p,q\rangle+1}`. Equation (4.30), TeX lines
511--547, supplies the exact Cox phase/lattice witness for every fixed
component. The smooth shortcut in equations (4.34)--(4.35), TeX lines
549--555, is used only after smooth-half-ray and smooth normal-direction
evidence is certified. Equation (4.42), TeX lines 587--601, then gives the
coefficient covariance
`psi[L(q)] = exp(2*pi*i*(<t,q>+lambda_f/2))*psi[q]`.

For a component that does not have the smooth shortcut certificate, the
implementation enumerates the exact dual lattice points returned by the
CYTools dual polytope (using exact vertices only as the documented fallback),
computes the displayed Cox exponents, removes monomials vanishing on the Cox
chart, and applies the eq. (4.42) covariance orbit by orbit. The restriction
is identically zero **if and only if** this invariant restricted support is
empty. The record retains the dual-point action, phase, monomial exponents,
quotient-character/Newton-face coordinates, chart nonvanishing witness, Cox
alpha/lattice witness, and finite-index stabilizer evidence. This is an
owner-approved derived implementation inference from the cited monomial and
covariance equations; it is not claimed as a separately stated Moritz
prescription.

The actual support/Newton face, never a reconstructed complete line system,
is passed to the transverse nondegeneracy and ordinary-Euler kernels. A
nonzero monomial has an empty zero locus on its torus chart and contributes
zero ordinary Euler. A genuinely contained component remains subject to the
source smoothness and Eq. (4.50) gates (TeX lines 638--657); Eq. (4.50) is
evaluated only after exact containment certifies a two-dimensional component.
Eq. (4.51), TeX lines 661--668, is applied only after the complete fixed-locus
Euler evidence and refined `H^2` action are certified.

The bounded generic fixtures include a singleton `q_0=(1,-1,-1,-1)` with
monomial `x_1^2` and empty intersection on the fixed Cox chart, plus a
non-smooth index-two cone whose exact invariant support is empty. The
immutable class-33 A determinant-four/determinant-two components and class-33
B surface all have the singleton `q_0` support and are transverse; both FRSTs
therefore produce only the transverse `chi(F_I)=272` contribution and the
split `(h^{1,1}_+,h^{1,1}_-,h^{2,1}_+,h^{2,1}_-)=(2,0,0,132)`. These are bounded
fixtures, not a population claim.

### Induced fixed locus and smoothness: Moritz eqs. (4.47)--(4.50)

Source: §4.6, TeX lines 638--657. The induced component is

```tex
\mathcal F_{\mathcal I}(\sigma,\nu)
:=\mathcal F_{\tilde{\mathcal I}}(\sigma,\nu)\cap X.
```

For a fixed surface `S` contained in `X`, the nodal count and required
condition are

```tex
n^{\mathcal S}_{df=0}=\int_{\mathcal S}
c_2(\mathcal O(K_V^{-1})|_{\mathcal S}\otimes
\mathcal N^*_{\mathcal S/V}),
\qquad n^{\mathcal S}_{df=0}=0.
```

Mapping: `_general_fixed_surface_n_s_table` in the sibling branch expands the
conormal-bundle expression as
`integral_S(c2(T_V)|_S-c2(T_S)+c1(T_S)^2)`. The asterisk on `N^*` is material:
the normal-bundle sign convention is superseded. The zero test applies only
to complex-two-dimensional ambient fixed components for which
`dim(sigma)+lambda_f=1 mod 2`.

Assumptions: `S` and its normal data are smooth enough for the Chern-class
calculation. Moritz separately requires the fixed component itself to be
smooth when `f` vanishes identically. For a non-identically-vanishing section,
the source instead requires nefness and avoidance of orbifold singularities
(lines 655--659). Missing Cartier, fan, or smoothness evidence must therefore
produce `unavailable`, never acceptance.

### Lefschetz/Hodge split: Moritz eq. (4.51)

Source: §4.7, TeX lines 661--668. For a favorable hypersurface,

```tex
h^{2,1}_-=h^{1,1}_-+\frac{\chi(\mathcal F_{\mathcal I})-\chi(X)}4-1.
```

The `h11` split is obtained from the `+1` and `-1` eigenspaces of the induced
action on `H^2(V)` (lines 661--666). The remaining Hodge number is
`h21_plus=h21-h21_minus`.

Clearly marked algebraic rearrangement:

```text
h21_plus
 = h21 - h21_minus
 = h21 - h11_minus - (chi(F_I)-chi(X))/4 + 1.
```

Therefore

```text
h21_plus = 0
 iff chi(F_I) = chi(X) + 4*(h21-h11_minus+1).
```

Using `chi(X)=2*(h11-h21)` for a Calabi--Yau threefold gives the equivalent
integer check

```text
chi(F_I) = 2*h11 + 2*h21 - 4*h11_minus + 4.
```

For `h11_minus=0`, the sibling diagnostic's historical expression
`h21_minus=(chi(F_I)-chi(X))/4-1` has the source sign. It is valid only after
`chi(F_I)` has been computed from the **same exact `(L,t,lambda_f)` action**.

Assumptions: favorable embedding, smooth `X`, a validated holomorphic
involution, complete irreducible fixed-locus enumeration, correct component
Euler characteristics, and no double counting of contained components.

## Published h11=2 cross-check

Sheridan et al. §4.2.1 gives the polytope at `main.tex` lines 1337--1345 and
states at line 1348

```tex
(h^{1,1}_+,h^{1,1}_-,h^{2,1}_+,h^{2,1}_-)=(2,0,0,132).
```

Thus `h11=2`, `h21=132`, and `h11_minus=0`. The rearranged Moritz formula
requires

```text
chi(X) = 2*(2-132) = -260,
chi(F_I) = -260 + 4*(132+1) = 272,
h21_minus = (272-(-260))/4 - 1 = 132,
h21_plus = 132-132 = 0.
```

The arithmetic exactly matches the published split. It is important that this
does not identify the published worked polytope with the bounded class-33
replay. Sheridan's six ray columns at `main.tex` lines 1337--1345 include
`(-1,3,-2,-1)` as the second ray. The immutable class-33 global coordinates in
`scripts/mpcp_immutable_source.py` instead contain `(-2,1,-1,-1)` and carry
the source key `source_row=29`,
`polytope_id=lattice-points-sha256:e777ace9bcfd967af24cfa8f56c098e455c0e665a9dc786d05280a6cb5ea12cf`.
These are different coordinate-keyed source objects. Class-33 FRST B is a
derived h11=2/Hodge diagnostic with the same expected `(2,132,-260)` data; it
is not a reproduction of the Sheridan worked geometry. A source reproduction
must freeze the displayed matrix under its own immutable coordinate identity.
The value `272` and split `(2,0,0,132)` therefore remain an algebraic
cross-check and a derived fixture, not a claim that class 33 is Sheridan's
worked polytope or that a published split was used as an acceptance input.

## Sheridan population and benchmark convention

Source: *Fuzzy Axions and Associated Relics*, §4.1.

- The scientific set is favorable CY **FRST classes** with
  `2 <= h11 <= 7` that admit an inherited O3/O7 involution with `h11_minus=0`
  (`main.tex` lines 1260--1282).
- Table 1 counts CYs/FRST classes with **at least one** qualifying involution,
  not every involution. The figure discussed nearby counts every orientifold;
  the distinction is explicit at lines 1291--1303.
- Table 1's `h11_minus=h21_plus=0` row is
  `11, 66, 267, 1033, 3623, 12253` for `h11=2,...,7`
  (`main.tex` lines 1221--1253).
- The paper states that, over this range, the `h21_plus=0` inherited
  orientifolds are the trilayer involutions (`main.tex` lines 1295--1299).
- The benchmark above fixes the `h11=2` Hodge split but does not publish
  `chi(F_I)` directly; `272` is the transparent consequence of Moritz eq.
  (4.51), not an additional quoted datum.

These are population conventions, not permission to substitute a trilayer or
special-shift diagnostic for verification of an arbitrary ledger witness.

### Full phase test versus the conditional smooth shortcut

Moritz's full phase condition is the quotient-lattice construction in
`KS_orientifolds.tex` lines 511--539 (eq. (4.30)). For every original-fan
pointwise invariant cone `sigma` and every `nu` representative, test

```text
t + nu in N_R / span_Q(sigma),
```

by pairing with an exact integer annihilator of `span_Q(sigma)`, and retain
the annihilator, rational quotient coordinates, and the exact `nu` witness.
This is the general path. Moritz's half-ray expression

```text
t + nu + 1/2 sum(sigma(1)) in N
```

is eq. (4.35), summarized at lines 551--555, and is conditional on the smooth
normal-bundle assumptions established at lines 498--508. The implementation
may use it only after `_half_ray_shortcut_proof` has certified a smooth
original cone, a simplicial full-dimensional auxiliary normal fan, ambient
unimodular provenance, and all required incidence data. A missing certificate
or a determinant-two/four quotient falls back to eq. (4.30); it never enables
the shortcut. This distinction is an implementation of the source boundary,
not a change to the source formula.

The following class-33 FRST A/B values are exact derived witnesses, retained
as regression fixtures rather than population claims. Labels `1`--`6` are the
non-origin Cox/GLSM divisor labels in the immutable seven-point boundary
configuration. The full refined GLSM matrix in both phases is

```text
Q = [[ 4,  1,  1,  1,  1,  0],
     [-2, -1, -1, -1,  0,  1]].
```

| witness | original sigma labels | determinant or saturation index | nonzero-coordinate GLSM columns | Cox phase generator | chart witness |
| --- | --- | --- | --- | --- | --- |
| FRST A determinant-four point | `(2,3,4,5)` | `|det(sigma)|=4` | `Q[:,(1,6)] = [[4,0],[-2,1]]` | `(1/4,0)`; pairings `(1,0)` | maximal cone chart, complement `(1,6)` |
| FRST A determinant-two point | `(2,3,4,6)` | `|det(sigma)|=2` | `Q[:,(1,5)] = [[4,1],[-2,0]]` | `(0,1/2)`; pairings `(-1,0)` | maximal cone chart, complement `(1,5)` |
| FRST B determinant-two surface | `(5,6)`; rays `(-2,-1,0,0),(0,1,0,0)` | fixed-cone index `2` | `Q[:,(1,2,3,4)] = [[4,1,1,1],[-2,-1,-1,-1]]` | `(1/2,1/2)`; pairings `(1,0,0,0)` | face of maximal cone `(2,3,5,6)`, complement `(1,4)` |

For the point rows the determinant equals the finite Cox stabilizer order.
For the surface row the gcd of the maximal two-by-two minors of the fixed
cone matrix and of the complementary GLSM matrix is `2`; this is the same
finite stabilizer order. In each chart, set Cox coordinates in the cone to
zero and coordinates in the displayed complement to one. The complement
monomial is then nonzero and no Stanley--Reisner generator is contained in
the zero set, which is the explicit irrelevant-ideal chart-existence test.
The Cox quotient/chart terminology follows Cox--Little--Schenck, *Toric
Varieties* (1st ed.), §5.1; the full GLSM relation check remains the exact
`M Q = Q P` contract in this ledger's H2 section.

The same fixtures retain Moritz eq. (4.46), `KS_orientifolds.tex` lines
631--636, as a conditional shortcut only. Their determinant-two/four and
index-two cones do not certify the smooth-half-ray hypotheses. The exact
Eq. (4.42) support audit instead finds the singleton `q_0` monomial for both
A components and the B surface, so all three are transverse. The A point
intersections and B surface Newton face contribute zero, while the common
transverse component contributes `272`; both FRSTs give `(2,0,0,132)`. No
Eq. (4.50) contraction is performed because no dim-2 component is exactly
contained. The detailed support/chart evidence is recorded in the bounded
MPCP ledger.

## Audit of `../CYAxiverse-orientifold-overcount`

Audited branch: `codex/orientifold-overcount-20260819`, HEAD
`a2018d98bd558ff9903c2296d7e8e122c84c1434`.

### Source-validated elements retained at HEAD

- `build_auxiliary_fan` follows eq. (4.26) using exact intersection rays.
- `_pointwise_invariant_cone_keys` correctly restores original-`Sigma` cones
  and original ray scale for eq. (4.35); auxiliary rays are geometry evidence,
  not component labels.
- `_fixed_component_records` uses exact rational/SNF arithmetic, applies the
  half-ray shortcut only with a smooth-normal certificate, and removes
  contained components.
- The shift path distinguishes the enumerated representative of `2t` from
  `t`, requires `2t in N`, then halves it exactly.
- `_dual_vertex_parity_evidence` contracts exact rational `2t` with dual
  vertices before the modulo-two test.
- `_general_fixed_surface_n_s_table` uses the conormal-bundle sign and fails
  unavailable on missing fan, Cartier, lattice, smoothness, nefness, or
  orbifold-avoidance evidence. Its toric surface formula has a separate exact
  Chow-ring cross-check recorded in the sibling validation note.

These statements validate the listed formula mappings. They do not establish
that every end-to-end candidate or population record is correct.

### Superseded conventions in branch history

- Before `f5f397c`, the stored projected-lattice vector was consumed as `t`
  although it represented `2t`; the correction divided once more by two.
- Before `1620b5a`, parity coerced components of exact `2t` with `int()` and
  there was no fail-closed `2t in N` involution filter. The two errors affected
  nonidentity `L` in opposite directions.
- Before `4c3f7c5`, an eq. (4.50) number could be trusted when the two-cone was
  simplicial even if the surface met non-smooth ambient structure; the
  smooth-star gate now makes that evidence unavailable.
- Before the corrections summarized by `b191af1` and `31cf465`, fixed
  components could be labelled by primitive auxiliary-fan rays rather than
  original-`Sigma` rays. Primitive normalization can change the half-ray sum;
  that convention is superseded. Corrected h11=2,3 artifacts were produced in
  the sibling branch; its old h11=4 artifact remains superseded and was not
  rerun here.
- Historical prose calling `N_{S/V}` (normal) the bundle in eq. (4.50) is
  superseded; the source uses `N^*_{S/V}` (conormal).

### Remaining validation boundary

No source anchor identifies the identity-only `_h21_plus_zero_diagnostic`
with a general-`L` computation. The identity wrapper derives `p0=2t` from the
canonical witness without shift substitution. General `L` uses the separate
componentwise kernel below. Contained singular quotient components remain
unavailable; the owner-approved ordinary-Euler extension for transverse
components is source-delimited below.

### Bounded class-33 B support audit (2026-08-23)

The immutable class-33 FRST B replay (hash prefix `57e8dcae`, full hash held in
`scripts/mpcp_immutable_source.py`) supplies the original-`Sigma` surface
`sigma(1)=\{(-2,-1,0,0),(0,1,0,0)\}` and its exact index-two Cox lattice.
The exact dual-point audit applies the Eq. (4.42) covariance to every dual
lattice point. The surviving invariant restricted support is the singleton
`q_0=(1,-1,-1,-1)`, with `s_{q_0}=x_1^2`; the Cox chart is certified nonzero.
The component is therefore transverse, not contained, and its actual Newton
face gives ordinary Euler zero. Eq. (4.50) is not evaluated because its
premise—a certified contained dim-2 component—is false. The transverse
component contributes `272`, so the bounded FRST B replay has
`chi(F_I)=272` and Eq. (4.51) split `(2,0,0,132)`.

This replaces the prior parity-only/formal-`n_S` diagnostic. The support rule
is an owner-approved derived implementation inference from Moritz TeX lines
285--296 and 587--601, not a source claim that Eq. (4.50) applies to this
transverse component. CYTools dual lattice points are used first, with exact
vertices only as the documented fallback; no complete line system is rebuilt.

## Implementation acceptance contract for the later kernel

Implementation may proceed only if it:

1. enumerates every live accepted `(L,t,lambda_f)` action for each immutable
   ledger-member FRST class, treating each exact live record as its own
   canonical action rather than privileging the summary's arbitrary witness;
2. verifies `L`, fan invariance, `(L phi_[t])^2=id`, O3/O7 parity, Kähler
   smoothness, coefficient parity, and complex-structure/fixed-locus
   smoothness;
3. enumerates original-`Sigma` `(sigma,nu)` components, applies eq. (4.46),
   removes containment duplicates, and computes every Euler characteristic;
   use the parity shortcut only with a certified smooth half-ray; otherwise
   retain the exact Eq. (4.42) invariant restricted support and its Cox/chart
   evidence;
4. records an exact-action input digest and diagnostic status;
5. reproduces the h11=2 fixture's `chi(F_I)=272` and published split; and
6. fails closed on every missing assumption or unavailable geometric datum.

No scientific sign or convention remains for owner selection in these
formulas. Implementation of the source contract was approved; the outcome and
remaining general-`L` boundary are recorded below.

## Implementation outcome after approval

The approved follow-up implemented and validated the exact identity-`L`
reduction from the canonical action: it decodes `t` exactly, sets `p0=2t`
without selecting a replacement shift, computes every identity-action fixed
component, and applies eq. (4.51) with integer/divisibility checks. The
Sheridan fixture returns `chi(F_I)=272` and `(2,0,0,132)`.

The owner subsequently approved the derived general-`L` Euler contract below.
The analytic contained/transverse, dimension-zero, orbifold-rejection, and
nonidentity end-to-end fixtures pass. The Sheridan `h11=2` reference passes
through the independently retained identity kernel with `chi(F_I)=272` and
split `(2,0,0,132)`. The recovered immutable terminal JSONLs provide a
fully audited lattice-matrix-only join for the summary's optional action
cross-reference: h11=2 covers 25/25 accepted class entries and h11=3 covers
201/201, with no missing keys, conflicts, collisions, or stable-ID failures.

The summary `accepted_witness` is an arbitrary accepted action, not the
scientific class representative. The implementation therefore exact-
evaluates every live accepted O3/O7 `h11_minus=0` action and selects by action
digest among those with `h21_plus=0`. The owner-approved support extension is
bounded to the immutable class-33 A/B fixtures in this task; it is not a
population rerun. No h11=2/3/4/5 population scan or production artifact write
was performed, and the production status remains `not_validated`.

## Derived component Euler formula and strict smoothness boundary

### Exact H2 quotient/relation contract

For the prime-divisor permutation matrix `P`, derive the induced H2 action
from the full GLSM quotient/relation matrix `Q` by solving
`M Q = Q P` over exact rationals. Require a full-row-rank `Q`, an integral
solution `M`, an exactly zero residual on every GLSM column, and `M^2=I`.
`basis_matrix` is a divisor-slot selector and is not used for this action.
Persist the selected nonzero minor, its determinant, matrix shapes, exact
integrality result, and zero-residual result as H2 proof evidence.

This section is an algebraic derivation used by the implementation, not a
formula attributed to Moritz. Moritz supplies the fixed components and their
hypersurface parity.

For a **smooth complete** toric component `C`, Cox--Little--Schenck Theorem
8.1.6 gives the toric Euler sequence. Example 13.1.1 and equation (13.1.2)
give

```tex
c(T_C)=\prod_{\rho\in\Sigma_C(1)}(1+D_\rho).
```

Proposition 13.1.2 and Chern--Gauss--Bonnet give

```tex
\chi(C)=\int_C c_{\dim C}(T_C).
```

For a smooth divisor `i:D -> C`, the normal exact sequence and divisor
adjunction give `c(T_D)=i^*c(T_C)/(1+i^*[D])`. Pushing forward by `i`
multiplies by `[D]`, hence the explicitly derived formula

```tex
\chi(D)=\int_C\left[c(T_C)\frac{[D]}{1+[D]}\right]_{\dim C}.
```

Mapping: `toric_fixed_component_euler.component_euler_from_certificate` uses
the first integral when Moritz eq. (4.46) says `f|C` vanishes identically. It
uses the second with `[D]=(-K_V)|_C` otherwise. The exact restricted Cartier
coefficients and original-`Sigma` quotient-star fan come from
`orientifold_general_l_geometry._positive_component_section_certificate`.

Cox--Little--Schenck Lemma 12.5.2 and Theorem 12.5.3 establish rational Chow
intersections for complete simplicial toric varieties. They do not make the
tangent sheaf a vector bundle on a singular orbifold. Therefore this kernel
requires every quotient maximal cone to have exact determinant one. A merely
simplicial/orbifold component is `unavailable`, even when a generic section
avoids its singular strata. No smooth-neighborhood extension is inferred.

The exact degree map is solved from integral linear divisor relations,
Stanley--Reisner vanishing, and unit intersection for each smooth maximal
cone. The rational system must have a unique solution, and every final Euler
number must be integral. Floating-point determinants and intersections do not
control acceptance. Contained fixed surfaces separately require Moritz eq.
(4.50) evidence `n_S=0`; the Chern-number calculation does not replace that
smoothness condition. The Chern path remains smooth-only. A separate theorem,
not a singular tangent-bundle extension, handles eligible transverse
components.

Contained zero-dimensional components have an additional fail-closed local
contract. The component record must carry certified local smoothness, integral
restricted Cartier data, and a unimodular local cone determinant. A missing
certificate, or a determinant-two/four quotient point, is `unavailable`; the
kernel does not fabricate a certified point, subtract a point, or use that
point to force Euler-characteristic divisibility.

### Sheridan identity special case

The Sheridan benchmark explains why the identity kernel remains independent.
Its transverse component with `sigma_rays=[[1,0,0,0]]` has fixed-lattice
basis (stored by columns)
`[[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]`, quotient annihilator
`[[0,1,0,0],[0,0,1,0],[1,0,0,0]]`, and quotient rays
`[-2,3,-1]`, `[0,-1,0]`, `[0,0,1]`, `[0,1,0]`, `[1,0,0]`.
The maximal cones
`{[-2,3,-1],[0,-1,0],[0,0,1]}` and
`{[-2,3,-1],[0,0,1],[0,1,0]}` have exact determinant two. This is a genuine
simplicial/orbifold quotient, so it cannot test the smooth formula. The
canonical-`p0` identity computation supplies the validated `chi(F_I)=272`;
the implementation does not infer an orbifold Chern-class extension.

## Ordinary-Euler orbit extension for transverse components

This section records an owner-approved derivation. It is **not** a Moritz
formula. It computes the ordinary topological Euler characteristic of a
transverse hypersurface by its disjoint toric-orbit pieces. It never computes
a stringy Euler number, orbifold Euler number, or singular-fan Chern integral.

### Primary-source anchors

1. A. G. Khovanskii, *Newton Polyhedra and the Genus of Complete
   Intersections*, Functional Analysis and Its Applications **12** (1978),
   English translation pp. 38--46, DOI `10.1007/BF01077562`.
   Section 3, Theorem 2 (translation p. 43; PDF text lines 282--286) states
   for a nondegenerate hypersurface in `(C*)^k`

   ```tex
   E(Z_f)=(-1)^{k-1}k!V(\Delta).
   ```

   Here `V` is Euclidean volume normalized so that the orbit character
   lattice has covolume one. The implementation uses exactly the equivalent
   convention

   ```tex
   E(Z_f)=(-1)^{k-1}\operatorname{Vol}_{\mathbb Z}(\Delta),
   \qquad \operatorname{Vol}_{\mathbb Z}=k!V.
   ```

   There is no second multiplication by `k!`. The standard two- and
   three-dimensional lattice simplices therefore both have normalized volume
   one; dilation of the three-simplex by two has normalized volume eight.

2. The same paper, introduction and Section 3 (translation pp. 38 and
   42--43; PDF text lines 36--38 and 255--264), decomposes a compactified
   hypersurface by toric orbits and sums their Euler characteristics.
   Independently, V. I. Danilov and A. G. Khovanskii, *Newton Polyhedra and an
   Algorithm for Computing Hodge--Deligne Numbers*, Math. USSR Izvestiya
   **29** (1987), Proposition 1.6 (pp. 282--283; PDF text lines 240--258),
   proves additivity for a finite disjoint union of locally closed
   subvarieties. Section 3.2 (p. 288; lines 512--517) identifies the
   intersection on a toric orbit with the Laurent polynomial restricted to
   the corresponding Newton face.

3. Danilov--Khovanskii Section 3.6 (p. 289; lines 536--547) defines
   nondegeneracy by transverse intersection with every toric stratum and says
   a generic Laurent polynomial with the prescribed polytope is
   nondegenerate. Batyrev--Cox, *On the Hodge Structure of Projective
   Hypersurfaces in Toric Varieties*, arXiv `alg-geom/9306011v1`, Definition
   4.13 and Proposition 4.15 (pp. 14--15; PDF text lines 677--687), state the
   orbitwise nondegeneracy condition and generic nondegeneracy for an ample
   toric linear system, with nondegenerate implying quasi-smooth. Their
   Definition 3.1 and Propositions 3.5/3.12 (pp. 9--12) delimit
   quasi-smooth/V-submanifold terminology on a simplicial toric variety.

The implementation does not weaken `ample` to `nef` by citation to
Batyrev--Cox. Instead, the exact Cartier/nef certificate provides the complete
basepoint-free toric section support, and the scientific model explicitly
selects a generic member. Every orbit used in the Khovanskii formula must
have a full-dimensional Newton face in the saturated orbit character lattice.
This is the direct Khovanskii generic-nondegeneracy setting. A missing
genericity certificate is terminal unavailable.

### Derived algorithm and package-symbol mapping

Let `C=C_sigma` be a certified maximal transverse fixed component. For every
cone `tau` of its quotient-star fan, the orbit is `(C*)^k`, where
`k=dim(C)-dim(tau)`. The exact restricted section polytope is intersected with

```tex
\langle m,v_\rho\rangle=-a_\rho,\qquad \rho\subset\tau.
```

Differences of its lattice points are expressed in the saturated character
lattice `M_tau=ker(M -> Hom(N_tau,Z))`. The source cone rays may generate a
proper finite-index sublattice of their real span. This does not change the
orbit character lattice: the code computes the exact integer annihilator of
that span with Smith-normal-form kernel arithmetic, then expresses every
Newton-face difference in this saturated lattice and rejects nonintegral
coordinates. The same exact coordinates are used to certify the difference
lattice for the lower-dimensional product case below. For a full-dimensional face
`Delta_tau`, Khovanskii gives

```tex
\chi(D\cap O(\tau))=(-1)^{k-1}\operatorname{Vol}_{\mathbb Z}(\Delta_\tau).
```

The case `k=0` is explicit: a certified nonzero monomial restricts to a
nonzero constant, so its zero locus is empty and contributes zero. A
positive-dimensional monomial restriction also has empty zero locus. An
empty face or a nonintegral character coordinate is unavailable. If the
non-monomial Newton face has affine dimension `r<k`, the implementation now
uses the product statement in Danilov--Khovanskii §5.1 (1987, PDF text
lines 706--715): after replacing the character lattice by the lattice of the
face span, the hypersurface is `Z' x (C*)^(k-r)`. Since ordinary Euler
characteristic is multiplicative and `chi(C*)=0`, this orbit contribution is
exactly zero. The implementation accepts that zero only after it has checked
the integral lattice construction described above; it does not infer a
stringy or orbifold Euler number.

`toric_fixed_component_euler.transverse_component_euler_orbit` implements
the orbit sum. `normalized_lattice_volume` computes exact geometric lattice
volume; `normalized_lattice_volume_ehrhart` independently computes the same
number as the `k`th finite difference of the Ehrhart polynomial. Every
full-dimensional orbit face must pass both calculations. On a smooth fan the
complete orbit sum must also equal `component_euler_from_certificate`, the
independent Chow/adjunction path.

The extension applies only when
`_positive_component_section_certificate` supplies exact restricted Cartier
data, certified nefness, a generic-section certificate, and certified
avoidance of every positive-dimensional singular stratum. A component that
meets a singular stratum is unavailable. Contained components never enter
this extension: they continue to require a smooth fan, Chern--Gauss--Bonnet,
and (for surfaces) Moritz eq. (4.50) `n_S=0`.

### Independent analytic validation

The smooth `P1` and `P1 x P1` fixtures give exact equality between orbit
additivity and smooth Chow/adjunction. For the simplicial orbifold
`P(1,1,2)`, the generic anticanonical curve has orbit contributions `-8+8=0`,
so its ordinary Euler characteristic is zero. Adding the primitive crepant
ray `(0,-1)` gives a smooth toric refinement; divisor adjunction on that fan
independently returns zero. These tests validate ordinary Euler only and do
not assign an orbifold/stringy Euler characteristic to the ambient surface.

The production gate remains closed until the full permitted h11=2 diagnostic
returns the expected 11 classes and the subsequent permitted h11=3 diagnostic
passes its expected target and coverage checks.
