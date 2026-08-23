# Bounded MPCP and equivariant-refinement ledger

Date: 2026-08-23 (source-identity correction)
Status: bounded implementation contract; no population claim
Scope: three explicitly requested replay indices (`26`, `31`, and `33`).

The support-aware certificate contract is schema
`cyaxiverse-bounded-mpcp-replay-1.3`, certificate schema
`cyaxiverse-bounded-mpcp-certificate-1.1`, and formula ledger schema
`cyaxiverse-mpcp-formula-ledger-20260823-2`. The validator rejects an older
schema or formula version before digest comparison; a stale certificate is
therefore terminal and cannot silently authorize a legacy parity result.

The direct Moritz anchors used by this extension are: anticanonical monomials
`s_q` in `KS_orientifolds.tex` TeX lines 285--296; the unconditional Cox phase
condition Eq. (4.30), lines 511--547; the smooth-half-ray conditions and
Eqs. (4.34)--(4.35), lines 549--555; coefficient covariance Eq. (4.42), lines
587--601; the parity shortcut Eq. (4.46), lines 631--641; the contained
surface diagnostic Eq. (4.50), lines 638--657; and the Lefschetz/Hodge
relation Eq. (4.51), lines 661--668. The nonsmooth actual-support rule is an
owner-approved derived implementation inference from the monomial and
covariance equations. It is not presented as a new direct source equation.

This ledger records the distinction between a selected FRST and an auxiliary
refinement used to test an inherited action. It does not authorize a complete
scan, a new subdivision convention, or a production database write. A replay
record is terminal when its input does not provide the exact source key
(`polytope_id`, the immutable source row, and global coordinates), the original selected
FRST, the lattice action, or the source-compatible subdivision data.

The correction is bounded to immutable KS mirror rows for classes `26`, `31`,
and `33` (source Parquet row indices `21`, `27`, and `29`). Each parent has
eight global lattice points and seven boundary points; point 7 is interior to
a facet and is omitted from the CY hypersurface FRST. The expected Hodge/Euler
values are `(2,120,-236)`, `(2,128,-252)`, and `(2,132,-260)`, respectively.

## Primary source anchors

The source hierarchy is Batyrev first, then the toric intersection and
additivity references used by the existing exact-action ledger.

| claim | source anchor | status in this implementation |
| --- | --- | --- |
| A reflexive pair is characterized by integral facet distance one, and its polar is again reflexive | V. V. Batyrev, *Dual Polyhedra and Mirror Symmetry for Calabi--Yau Hypersurfaces in Toric Varieties*, arXiv:alg-geom/9310003v1, Definition 4.1.5, PDF p. 17, lines 791--800; Theorem 4.1.6, p. 17, lines 802--817 | direct citation |
| A generic Δ-regular anticanonical hypersurface is Calabi--Yau with canonical singularities | Batyrev, Theorem 4.1.9, PDF p. 18, lines 834--855 | direct citation |
| A maximal projective triangulation gives an MPCP toric morphism | Batyrev, Theorem 2.2.24, PDF pp. 11--12, lines 488--508; Section 4.2, pp. 19--20, lines 887--901 | direct citation |
| The hypersurface MPCP refinement exists for every Δ-regular hypersurface; in dimension at most four it is smooth | Batyrev, Theorem 4.2.2 and Corollary 4.2.3, PDF p. 20, lines 907--916 | direct citation |
| Exceptional divisor classes contribute to the resolved Picard group according to boundary lattice points and dual-face interior points | Batyrev, Proposition 4.4.1, PDF p. 25, lines 1272--1282; Theorem 4.4.2, p. 25, lines 1283--1303, equation (2) | direct citation |
| The resolved threefold Euler number is a property of the MPCP hypersurface, not of an arbitrary selected triangulation | Batyrev, Section 4.5, PDF pp. 27--29, lines 1471--1482 and 1706--1714 | direct citation |
| Toric orbit intersections with a nondegenerate hypersurface are evaluated face by face, and locally closed Euler characteristics add | Danilov--Khovanskii, *Newton Polyhedra and an Algorithm for Computing Hodge--Deligne Numbers*, Math. USSR-Izv. 29 (1987), Proposition 1.6, pp. 282--283 (PDF text lines 240--258); Section 3.2, p. 288 (lines 512--517); Section 3.6, p. 289 (lines 536--547) | direct citation; ordinary Euler only |
| Chow relations and toric Chern classes require the stated smooth/simplicial hypotheses | Cox--Little--Schenck, *Toric Varieties*, AMS GSM 124 (2011), Theorem 8.1.6, Lemma 12.5.2, Theorem 12.5.3, Example 13.1.1, equation (13.1.2), and Proposition 13.1.2 | direct citation; see the exact-action ledger for the expanded derivation |
| Fixed components, parity, and the Hodge eigenspace relation | Moritz, *Orientifolding Kreuzer--Skarke*, arXiv:2305.06363v1, equations (4.26), (4.35), (4.45), (4.46), (4.50), (4.51), `KS_orientifolds.tex` lines 491--506, 511--555, 625--636, 638--668 | direct citation; formulas and assumptions are expanded in `validation/orientifold_exact_action_formula_ledger_20260822.md` |
| A CYTools FRST ignores points interior to facets | Moritz, `KS_orientifolds.tex` line 245 (rendered in the public HTML discussion of the 4D reflexive polytope and FRST) | direct source convention |
| Sheridan's finite FRST-class setup | Sheridan et al., *Fuzzy Axions and Associated Relics*, arXiv:2412.12012v1, `main.tex` lines 1260--1282 and 1295--1303; h11=2 setup lines 1337--1349 | direct source scope; no population inference here |
| Facet-interior omission through the dual face and anticanonical restriction | Moritz, `KS_orientifolds.tex` lines 245 and 285--292, together with Batyrev Proposition 4.4.1 and Theorem 4.4.2 | derived implementation certificate, not a direct CYTools theorem |

The Sheridan h11=2 worked example must remain a separate source identity. Its
second ray is `(-1,3,-2,-1)` (`main.tex` lines 1337--1345), whereas immutable
class 33 uses `(-2,1,-1,-1)` at global coordinate slot 2, source row 29, and
`polytope_id=lattice-points-sha256:e777ace9bcfd967af24cfa8f56c098e455c0e665a9dc786d05280a6cb5ea12cf`.
The class-33 FRST A/B replay is therefore a derived MPCP/API fixture with the
same h11=2 Hodge data, not a coordinate-level replay of Sheridan's displayed
polytope. A future source replay must add the displayed matrix under its own
immutable coordinate hash; no Hodge match may substitute for that identity.

The Batyrev PDF line numbers above are the text-extraction anchors from the
arXiv source linked in the existing ledger. Page numbers are PDF pages, not
the printed-page numbering of a later journal edition.

## Formula and convention ledger

### Crepant height-one rays (direct citation plus implementation inference)

For a reflexive four-polytope `Delta*` in `N`, every boundary lattice point
`v` used as a ray lies on a facet at lattice height one. Equivalently, if a
facet of the polar polytope has primitive support `m_F` with
`<m_F,v>=-1`, then the ray generator is the primitive lattice vector on the
height-one slice. Adding all such boundary lattice points and taking cones
over a boundary triangulation leaves the anticanonical support function
unchanged. Hence the toric morphism is crepant:

```text
K_{V'} = pi^* K_V.
```

Batyrev's Theorem 2.2.24 supplies the direct existence/MPCP statement. The
height-one ray wording and the displayed canonical-divisor equality are the
toric calculation used by the implementation and are a derived inference
from the theorem plus the toric discrepancy formula; they are not a new
claim attributed to CYTools.

### Fine, regular, star, and maximal projective triangulations

The required ambient refinement is represented by a triangulation of the
boundary point configuration of `Delta*` that is:

1. **fine**: every permitted boundary lattice point is a vertex;
2. **regular/projective**: it is induced by a convex piecewise-linear height
   function (a projective support function);
3. **star**: every maximal cell contains the origin, so cones over the cells
   refine the fan of the reflexive polytope; and
4. **maximal**: no permitted height-one point can be inserted while retaining
   a proper triangulation.

The first three labels are the CYTools API contract. “MPCP” is attached only
after the Batyrev hypotheses and the hypersurface checks are recorded. A
degenerate regular subdivision is retained as a subdivision record; it is not
silently tie-broken into a different FRST.

### Facet-interior points and the two coordinate spaces

Moritz's source convention is explicit: the FRST triangulates the boundary
point configuration while **ignoring points interior to facets**
(`KS_orientifolds.tex` line 245; direct source convention). In the three immutable
rows, the parent `Polytope` therefore has eight points while each built-in
boundary FRST has seven local points. The omitted point is recorded with its
exact global coordinate and the facet whose lattice-point list contains it but
whose vertex list does not.

The geometric explanation is a derived certificate. Moritz's anticanonical
monomial formula (lines 285--292) assigns exponent `<p,q>+1` to the coordinate
for a primal point `p`. If `p` is interior to a facet of `Delta*`, the dual
face is a vertex of `Delta`; the corresponding anticanonical restriction has
a surviving dual-vertex monomial and the generic hypersurface does not acquire
a divisor from that facet-interior ray. This explains why the boundary FRST
may omit the point while the frozen global polytope must retain it for exact
coordinate identity and index mapping. The restriction statement is derived
from the displayed monomial/dual-face formula and Batyrev's exceptional-face
accounting; it is not presented as a direct CYTools theorem.

The implementation has two explicit index spaces:

1. `polytope_global`: indices into the frozen eight-point parent;
2. `triangulation_local`: indices into the seven-point CYTools FRST.

Every local simplex is mapped to parent indices by exact coordinate lookup.
Missing coordinates, duplicate coordinates, a failed omitted-point
certificate, or a changed index-space label is terminal. No
`Polytope(triangulation.points())` reconstruction is permitted.

### Equivariant symmetric heights

Let `T'` be the preserved selected FRST, let `h'` be an interior height vector
in its secondary cone, and let `L` permute the point configuration. The
source construction uses

```text
h_p = (h'_p + h'_{L(p)})/2.
```

This is the direct Moritz source construction (`KS_orientifolds.tex`
lines 471--489, with the fan/action conditions in lines 330--339). The
implementation additionally verifies `h_{L(p)}=h_p` exactly as rational
numbers, keeps the origin height convention explicit, and asks CYTools for
the lower subdivision without tie-breaking where the backend supports it.
Those exact equality and backend-dispatch checks are derived implementation
inferences, not source equations.

### Refined fixed components and auxiliary quotient fan

The selected FRST remains the source geometry identity. An auxiliary
subdivision/refinement may be used to construct a quotient or fixed-locus fan,
but it must never replace the selected FRST in the class key. For each exact
action `(L,t,lambda_f)`, the implementation:

- labels components by original-`Sigma` cones, as required by Moritz
  equations (4.33)--(4.35);
- stores the auxiliary quotient fan separately, including non-simplicial cells;
- applies equation (4.46) before computing a section Euler number; and
- emits `unavailable` for a missing Cartier, smoothness, genericity, or
  source-compatible removal/gluing certificate.

This is a direct application of Moritz, with the “separate auxiliary fan” and
the fail-closed status being implementation policy.

#### Exact Cox/GLSM phase witnesses (derived class-33 fixtures)

Moritz eq. (4.30), `KS_orientifolds.tex` lines 511--539, is the unconditional
phase test. For every original-`Sigma` pointwise invariant `sigma`, test the
exact quotient-lattice integrality of `t+nu` modulo
`span_Q(sigma)`, retaining the integer annihilator and rational quotient
coordinates. Moritz eq. (4.35), lines 551--555, is only a smooth-normal
shortcut; the smoothness and incidence preconditions are in lines 498--508.
The replay uses eq. (4.30) whenever that certificate is absent, including
determinant-two/four quotient cones. It does not infer a smooth half-ray
answer from a merely simplicial cone.

The immutable class-33 FRST A/B witness uses the full refined GLSM matrix

```text
Q = [[ 4,  1,  1,  1,  1,  0],
     [-2, -1, -1, -1,  0,  1]].
```

| FRST phase | fixed cone/face | exact index | complementary GLSM columns | phase witness | Cox chart |
| --- | --- | --- | --- | --- | --- |
| A | point `(2,3,4,5)` | `|det|=4`, stabilizer order 4 | `Q[:,(1,6)] = [[4,0],[-2,1]]` | `(1/4,0)` with pairings `(1,0)` | complement `(1,6)` |
| A | point `(2,3,4,6)` | `|det|=2`, stabilizer order 2 | `Q[:,(1,5)] = [[4,1],[-2,0]]` | `(0,1/2)` with pairings `(-1,0)` | complement `(1,5)` |
| B | surface `(5,6)` with rays `(-2,-1,0,0),(0,1,0,0)` | fixed-cone saturation index 2, stabilizer order 2 | `Q[:,(1,2,3,4)] = [[4,1,1,1],[-2,-1,-1,-1]]` | `(1/2,1/2)` with pairings `(1,0,0,0)` | face of `(2,3,5,6)`, chart complement `(1,4)` |

These are derived finite-index witnesses, not source-published invariants.
The point determinant and the surface gcd of maximal minors agree with the
finite Cox stabilizer order. Set Cox coordinates on the chart-cone rays to
zero and the displayed complement to one. The complement monomial is nonzero
and no Stanley--Reisner generator is contained in the zero set, proving chart
existence for this replay. This is the Cox quotient/irrelevant-ideal
construction of Cox--Little--Schenck, *Toric Varieties* (1st ed.), §5.1; the
full GLSM action remains the exact `M Q'=Q' P` proof above.

With `lambda_f=1`, all three witnesses satisfy Moritz eq. (4.46), lines
631--636, when its smooth-half-ray preconditions are certified. For the
determinant-two/four and non-smooth branches, the implementation instead
uses the actual Eq. (4.42) support: both A point components and the B surface
retain the singleton `q_0=(1,-1,-1,-1)` (`x_1^2`) and are therefore
transverse, not contained. Their zero-dimensional intersections have Euler
zero; the B surface's actual Newton face also gives Euler zero. Both FRST A
and B have only the transverse contribution `chi(F_I)=272`, with exact
Eq. (4.51) split `(2,0,0,132)`. Eq. (4.50) is not called because no
two-dimensional component is exactly contained.

### Ordinary Euler characteristic

For a smooth toric component `C`, the existing ledger derives

```text
chi(C) = integral_C c_dim(C)(T_C),
chi(C intersect D) = integral_C [c(T_C) D/(1+D)]_dim(C),
```

from the toric Euler sequence, adjunction, and Chern--Gauss--Bonnet. For an
eligible transverse component in a possibly simplicial ambient presentation,
ordinary Euler is instead the additive orbit sum

```text
chi(C intersect X) = sum_tau (-1)^(k_tau-1) Vol_Z(Delta_tau),
```

over nondegenerate hypersurface pieces in torus orbits of dimension `k_tau`,
with the zero-dimensional orbit contribution treated explicitly. This is
the Danilov--Khovanskii/Khovanskii path recorded in the exact-action ledger.
It is **ordinary topological Euler**, never stringy or orbifold Euler. A
singular quotient fan without the required genericity and lattice certificate
is terminal `unavailable`.

### Full refined GLSM relation

Let `Q'` be the full refined GLSM quotient/relation matrix, with one column for
each prime toric divisor of the refined ambient fan, including exceptional
height-one rays when their divisor intersects the hypersurface. Let `P` be the
permutation induced by `L` on those columns. The induced integral action on
the full refined `H^2` quotient is defined by

```text
M Q' = Q' P.
```

The implementation selects a nonzero `h11 x h11` minor only to solve the
equation, then verifies every column, exact integrality, zero residual, and
`M^2=I`. Selecting a minor is an algebraic device, not a divisor-basis
substitution. `Q'` must be built after refinement; using the original `Q` when
exceptional rays intersect `X` is terminal `incomplete_refined_glsm`.

### Resolved Hodge split (Eq. 4.51)

For the resolved smooth favorable hypersurface and the same exact action,
Moritz equation (4.51) is

```text
h21_minus = h11_minus + (chi(F_I) - chi(X))/4 - 1,
h21_plus  = h21 - h21_minus.
```

The resolved `h11` is read from the refined `CalabiYau` object or the
source-matched Batyrev face formula. It is not assumed to remain two. The
derived acceptance identity is

```text
chi(F_I) = chi(X) + 4*(h21 - h11_minus + 1)  iff  h21_plus = 0.
```

This calculation is valid only after the refined `Q'`, `H^2` action, complete
fixed-component list, ordinary Euler evidence, and favorable/smoothness gates
are all present.

## CYTools 1.4.12 decision table and provenance

The audited environment is CYTools `1.4.12`, Python `3.14.6`, NumPy `2.4.6`
or later in the active environment, SciPy `1.18.0`, and the public classes
`Polytope`, `Triangulation`, `ToricVariety`, `CalabiYau`, and `Fan`.
Version strings are recorded in every replay manifest; an unavailable version
is not guessed.

| operation | CYTools 1.4.12 public API | built-in-first decision | exact/project fallback |
| --- | --- | --- | --- |
| lattice points, vertices, facets, dual | `Polytope.points`, `vertices`, `facets`/`faces`, `dual` | use and serialize the returned integer arrays | exact rank/determinant only for validation; missing data is terminal |
| all MPCP candidates | `Polytope.all_triangulations(only_fine=True, only_regular=True, only_star=True, include_points_interior_to_facets=False)` | enumerate every yielded candidate up to the declared cap; record omitted-point/facet certificates; never select by result | no triangulation fallback; a missing API is terminal `all_triangulations_unavailable` |
| explicit source FRST | `Polytope.triangulate(simplices=..., check_input_simplices=True, include_points_interior_to_facets=False)` | reconstruct on the frozen global parent after exact local-to-global coordinate mapping | standalone triangulation-point polytopes are forbidden |
| regular height witness | `Triangulation.heights`, `secondary_cone` | use documented height/secondary-cone data when available | exact rational symmetrization and equality checks; no numerical height invention |
| lower subdivision | `poly.vc().subdivide(..., make_fine=False, check_heights=False, cure_heights=...)` when the backend exposes it | retain non-simplicial cells | exact cell-complex invariance only; missing lower-subdivision backend is terminal |
| fan and toric variety | `Triangulation.fan`, `get_toric_variety`, `ToricVariety.fan_cones`, `is_smooth` | use for fan, cone, smoothness, and ray data | exact cone determinants/SNF for certificates only |
| Calabi--Yau invariants | `Triangulation.get_cy`, `CalabiYau.h11`, `h21`, `chi`, `is_smooth` | recompute on each refined candidate | no inferred Hodge values |
| GLSM/SR/intersections | `glsm_charge_matrix`, `glsm_linear_relations`, `sr_ideal`, `intersection_numbers`, `divisor_basis` | use full refined matrices and invariants | exact `M Q'=Q'P`, SR/intersection checks, and SNF only at the boundary |
| automorphisms | `Polytope.automorphisms`, `inequivalent_Z2_actions` where available | use for candidate action enumeration | exact generic equivariance/action-lift fallback with a reason code |
| fixed components and Euler | no public CYTools orientifold kernel | project exact Moritz/Chow/orbit-Euler kernels | every call records `fallback_reason` and evidence status |

The table is a dispatch contract, not a claim that every method is present on
every object. The driver feature-detects the method and records whether it was
used, absent, raised, or returned data with an unsupported shape.

### Class-33 B invariant-support API boundary

The bounded class-33 B witness (selected FRST hash
`57e8dcae74298839b9a208e9411125bed1c73459f8ce4ae14b23152ac6f7ebb0`) now
uses the exact dual lattice points and Eq. (4.42) covariance before any
Eq. (4.50) diagnostic. The original-`Sigma` surface
`sigma(1)=\{(-2,-1,0,0),(0,1,0,0)\}` has fixed-cone index `2`, and its exact
restricted support contains the singleton `q_0=(1,-1,-1,-1)` with monomial
`x_1^2`. The Cox chart is certified nonzero, so the surface is transverse and
its actual Newton face gives ordinary Euler zero. Exact containment is false;
therefore no `n_S` table is evaluated. This is an owner-approved derived
support inference from Moritz TeX lines 285--296 and 587--601, not a new
source formula.

## Bounded replay contract

Only indices `26`, `31`, and `33` are in scope. The driver accepts an explicit
replay manifest containing the point set, selected FRST simplices, and action
records. It preserves the selected FRST hash and does not silently load a
different class. The immutable fixture records the source Parquet SHA-256 and
the authoritative source rows `21`, `27`, and `29`; a row relabelling is
terminal even when the coordinate hash is otherwise valid. For each index it
records:

- selected-FRST identity and all original CYTools flags;
- every compatible local/global refinement yielded before the resource cap;
- duplicate, action-incompatible, non-fine, non-regular, non-star,
  non-simplicial, unsupported, and cap terminal records;
- CYTools versions and API dispatch outcomes;
- full refined divisor/ray data and the `Q'` action proof;
- refined Hodge data and Eq. (4.51) status; and
- all fallback reasons, including the exact kernel used for fixed loci/Euler.

No result is selected by matching a target count, Hodge number, or Table 1
entry. Source identity, immutable source row, global point count, boundary point count,
Hodge/Euler values, dual parity, local simplex index space, and FRST identities
are acceptance gates; each mismatch is terminal. Table 1 status is a
comparison field only: `match`, `mismatch`, or `not_comparable` after the
bounded run. This is a finite replay diagnostic, not a representative or
complete sample. The analyzer scope is exactly `26`, `31`, and `33`; it does
not run h11=2/3/4/5 population scans or write production data.

### Exact-trilayer certificate bridge

`scripts/mpcp_bounded_analysis.py` emits a replay certificate only for an
action whose complete bounded evaluation is `refined_action_evaluated`, with
computed ordinary Euler evidence and an exact refined GLSM proof. Its key
binds the immutable source Parquet SHA-256 and row, the canonical global
coordinates and `polytope_id`, the evaluated FRST hash, the complete source
action witness and digest, the verified CYTools `1.4.12` API contract, the
formula/certificate schema versions, and a digest over the full component,
Euler, and refined-H2 evidence. Runtime versions, caps, and every recorded
fallback reason remain in the certificate.

`trilayer_involutions.evaluate_exact_trilayer_action` accepts such a
certificate only as an optional replay witness. It verifies the immutable key
against the live global coordinates, FRST, and exact action, then recomputes
the exact trilayer topology path. A missing certificate is labelled as an
unavailable source-certified record until its bounded replay is rerun; a
stale, tampered, or mismatched certificate is terminal.
`build_orientifold_axion_database.py` passes certificates by
`(polytope_id, frst_hash)` and persists the complete certificate and
verification status beside the exact action provenance when a database build
is explicitly authorized. It never uses a class index or target Euler/Hodge
value to select a FRST or action.

The bounded fixtures produce certificates for both selected FRSTs of classes
26 and 31 (`chi(F_I)=248` and `264`, with splits `(2,0,0,120)` and
`(2,0,0,128)`). Both class-33 FRSTs now also pass the transverse support path:
the determinant-four/determinant-two A components and the B surface retain
the singleton `q_0` support, each intersection is transverse, and the only
nonzero contribution is `chi(F_I)=272`, giving `(2,0,0,132)`. The resulting
certificates bind the support/Newton-face evidence in the component digest.
Eq. (4.50) is not evaluated for class 33 because no dim-2 component is
contained. These are bounded fixture outcomes, not hard-coded selection rules
or population claims.

### Bounded structural rank-3/rank-4 support re-audit

The generic structural fixtures were re-audited at component ranks three and
four using the same exact dual-point/support kernel, with no expected-count or
class-result selector. Each rank retained one nonzero invariant support
monomial, a certified Cox chart, and `restriction_identically_zero=false`.
This is a bounded structural audit only; it is not an h11=3 or h11=4
population scan and no production artifact was written.

Version impact: implementation-only scientific infrastructure; no
`Project.toml` bump is included. A release-boundary review is required before
any persisted schema or scientific population claim changes.

## Verification record

Files added or updated in this bounded task:

- `validation/mpcp_refinement_ledger_20260823.md`;
- `validation/orientifold_exact_action_formula_ledger_20260822.md` (link to
  this ledger);
- `scripts/mpcp_bounded_analysis.py`;
- `scripts/toric_fixed_component_euler.py`;
- `scripts/orientifold_general_l_geometry.py`;
- `scripts/inherited_orientifold_candidates.py`;
- `scripts/trilayer_involutions.py`;
- `scripts/test_mpcp_bounded_analysis.py`;
- `scripts/test_build_orientifold_axion_database.py`;
- `scripts/mpcp_immutable_source.py`;
- `scripts/test_mpcp_cytools_source.py`.

Focused commands and observed results:

```text
python -m py_compile scripts/orientifold_general_l_geometry.py scripts/toric_fixed_component_euler.py scripts/inherited_orientifold_candidates.py scripts/mpcp_bounded_analysis.py scripts/trilayer_involutions.py scripts/test_mpcp_bounded_analysis.py scripts/test_mpcp_cytools_source.py scripts/test_trilayer_involutions.py scripts/test_build_orientifold_axion_database.py
PYTHONPATH=scripts conda run --no-capture-output -n cytools python scripts/test_mpcp_bounded_analysis.py
  20 tests, OK

PYTHONPATH=scripts CYTOOLS_CACHE_DIR=/private/tmp/cyax-mpcp-cache-correction \
  conda run --no-capture-output -n cytools python scripts/test_mpcp_cytools_source.py
  4 tests, OK (including class-33 A/B Cox/GLSM phase/support witnesses)

CYTOOLS_CACHE_DIR=/private/tmp/cyax-mpcp-cache3 PYTHONPATH=scripts \
  conda run --no-capture-output -n cytools python scripts/test_trilayer_involutions.py
  12 tests, OK

CYTOOLS_CACHE_DIR=/private/tmp/cyax-mpcp-cache3 PYTHONPATH=scripts \
  conda run --no-capture-output -n cytools python scripts/test_inherited_orientifold_candidates.py
  21 tests, OK

PYTHONPATH=scripts CYTOOLS_CACHE_DIR=/private/tmp/cyax-build-cache \
  conda run --no-capture-output -n cytools python scripts/test_build_orientifold_axion_database.py
  37 tests, OK

git diff --check
  OK
```

The CYTools runs emitted the environment's existing Python 3.14 SWIG
deprecation warnings and an exit-time CYTools cache warning because this
managed checkout cannot write the default
`/Users/vmehta/Library/Caches/CYTools/twoface_ineqs.pkl.gz.tmp.*` path. The
tests themselves passed; no cache or repository data was written. A live
CYTools smoke replay of an explicit five-point reflexive simplex yielded one
fine/regular/star triangulation, `h11=1`, `h21=101`, `chi=-200`, and a verified
full refined GLSM relation; its fixed-locus Euler status was correctly
`unavailable` because the synthetic object has no project fixed-component
evidence. This is an API smoke check, not a KS result.

The bounded analyzer was rerun only on immutable classes `26`, `31`, and `33`.
Each row had 8 global/7 boundary points, two FRST refinements, exact source
Hodge/Euler agreement, a verified dual parity check, and two retained
refinement records. All two class-26 and class-31 FRST actions reached
`computed` ordinary Euler evidence (`248` and `264`). Both class-33 FRSTs
reached `computed` with `chi(F_I)=272`; A's determinant-four/determinant-two
components and B's index-two surface each retain singleton support
`q_0=(1,-1,-1,-1)`, and both have split `(2,0,0,132)`. No Eq. (4.50) table is
constructed for class 33 because exact containment is false. The generic
bounded structural rank-three/rank-four support re-audit also returned one
certified nonzero support monomial and a certified Cox chart in both cases;
these are fixture diagnostics, not population records. No production h11
scan, HDF5/database write, or result-selected refinement was run. Version
impact is implementation-only; the package version bump remains deferred to
the release boundary.
