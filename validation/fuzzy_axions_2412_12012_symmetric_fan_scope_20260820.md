# Symmetric-fan diagnostic scope

Date: 2026-08-20
Source: Moritz et al., arXiv:2305.06363, section 4.3
Implementation: `scripts/symmetric_fan_diagnostic.py`

This diagnostic measures a missing search route. It does not change the
supplied-FRST orientifold audit and does not promote a singular ambient fan to
an accepted orientifold.

## Population and identifiers

The diagnostic uses these separate units:

| Identifier | Meaning | Equivalence rule | Artifact field |
|---|---|---|---|
| supplied FRST | Selected representative used as the source height input | `Triangulation.is_equivalent(..., on_faces_dim=2)` defines one existing FRST class representative per polytope | `supplied_frst_id`, `frst_class_index` |
| induced symmetric subdivision | Lower-hull cell complex obtained after (h_p=(h'_p+h'_{L(p)})/2) | Exact canonical sorted ray-cell complex; no tie-broken triangulation is substituted | `symmetric_subdivision_id`, `cells` |
| final CY/FRST class | Only a simplicial induced subdivision that can be converted to a `Triangulation` | `Triangulation.is_equivalent(..., on_faces_dim=2)` against existing class representatives | `final_frst_class_id`, `checks.class_equivalence` |

The lattice matrix is a candidate input, not a class identifier. The scan
tests only non-identity involutions that preserve the polytope point set and
do not preserve the supplied FRST. A matrix that preserves the supplied FRST
is outside this diagnostic's opportunity population because the production
audit already handles that branch.

## Construction and checks

The implementation records the full point configuration, original heights,
origin-relative heights, symmetrized heights, matrix, cell complex, and all
check methods. It calls:

```text
poly.vc().subdivide(
    heights=h_sym,
    backend="ppl",
    make_fine=False,
    check_heights=False,
    cure_heights=True,
)
```

PPL retains degenerate lower cells. When CYTools cannot evaluate
`Fan.is_regular()` for a non-triangulation, the retained symmetrized heights
and direct PPL lower-hull construction are recorded as the source-certified
regularity witness. The diagnostic never inspects only a CGAL tie-broken
triangulation.

The checks are:

1. exact cell-complex invariance under (L);
2. CYTools fineness, with an explicit used-label fallback;
3. regularity from `Fan.is_regular()` or the PPL lower-hull witness;
4. the CYTools Gorenstein-Fano star certificate;
5. the source two-face condition: each maximal cell meets a two-face in at
   most three rays and the two-face point configuration is covered;
6. source two-face FRST equivalence for simplicial outputs only.

Every attempt ends in exactly one of:

```text
constructed
already_represented
two_face_failed
regularity_failed
star_failed
fineness_failed
resource_limited
explicitly_unavailable
```

`constructed` means that the bounded diagnostic checks passed. It is not an
accepted orientifold result. A non-simplicial construction still requires the
source removal/gluing step, subdivision-aware fixed-locus fan, parity,
H2-action, and smooth Calabi–Yau/fixed-locus evidence before it can affect a
population count.

## Reproduction boundary

Example bounded command:

```bash
XDG_CACHE_HOME=/private/tmp/cyax-symmetric-subdivision-cache \
conda run --no-capture-output -n cytools python scripts/symmetric_fan_diagnostic.py \
  --h11 3 \
  --parquet-dir /private/tmp/cyax-ks-mirror-h11-4 \
  --pair-limit 10 \
  --output /private/tmp/cyax-symmetric-fan-h11-3-diagnostic.json
```

The output refuses to overwrite an existing path and records the source
commit, input mirror, scope limits, heights, cell complex, check methods, and
terminal status for every tested pair. A limited run is not a
population-complete nonexistence claim.

The first bounded h11=3 run in this continuation used the available mirror
`/private/tmp/cyax-ks-mirror-h11-3` and `--pair-limit 10`. It selected 110
polytopes and 119 supplied FRST class representatives before finding 10
non-preserving pairs. The output had:

```text
already_represented       2
constructed                1
two_face_failed            7
all other terminal states  0
```

The two `already_represented` records matched an existing class through
`Triangulation.is_equivalent(on_faces_dim=2)`. The one `constructed` record
was non-simplicial, so it has no `final_frst_class_id`; it is structural fan
evidence, not an accepted orientifold. The seven failures each retained a
specific two-face intersection. The replay artifact was written outside the
checkout at
`/private/tmp/cyax-symmetric-fan-h11-3-diagnostic-20260820.json` with SHA-256
`6559f7653ee6e724754d9ed554d437030bf37691cf1f6ebec2d6a8e6d0989d87`.

Version impact: no package behavior or persisted package schema changes on
this feature branch. If this diagnostic is later connected to accepted
orientifold enumeration, it becomes a scientific search-population change and
requires at least a pre-1.0 minor bump at the reviewed release boundary.
