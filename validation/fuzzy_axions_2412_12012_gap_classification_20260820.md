# Table 1 orientifold-gap review: h11 = 2, 3, with h11 = 4 gated

Date: 2026-08-20

Paper: Sheridan et al., [*Fuzzy Axions and Associated Relics*](https://arxiv.org/html/2412.12012), especially Table 1

Orientifold source: Moritz et al., [*Orientifolds of Calabi–Yau hypersurfaces*](https://arxiv.org/html/2305.06363)

Scope: corrected complete-population artifacts for \(h^{1,1}=2,3\); preparation, not a result, for \(h^{1,1}=4\)

## 1. Outcome and confidence

The discrepancy is real, but the current artifacts cannot identify it as a
paper-versus-code set difference. They establish an exact aggregate count gap:

| \(h^{1,1}\) | favorable target/code | FRST target/code | inherited target/code | inherited gap | \(h^{1,1}_-=0\) target/code | trilayer target/code |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 / 36 | 36 / 36 | 32 / 10 | **22** | 32 / 10 | 11 / 11 |
| 3 | 243 / 243 | 274 / 274 | 253 / 80 | **173** | 253 / 80 | 66 / 66 |

Confidence is high in these arithmetic comparisons and in the base-population
and trilayer matches. Confidence is not yet sufficient to allocate the 22 and
173 missing counts to individual paper classes: Table 1 publishes aggregate
counts, the paper's supplemental repository does not publish a complete
orientifold-class manifest, and the audit artifacts do not retain every
candidate terminal record.

The identity-sector values 25 and 198 are action-level parity diagnostics.
They are not Table 1 inherited-orientifold counts. The model stage is null in
both artifacts, so the Table 1 model values 2 and 263 were not reproduced.

## 2. Exact artifact-only classification

The deterministic analysis is implemented in
[`scripts/analyze_fuzzy_axions_orientifold_gap.py`](../scripts/analyze_fuzzy_axions_orientifold_gap.py)
and tested by
[`scripts/test_analyze_fuzzy_axions_orientifold_gap.py`](../scripts/test_analyze_fuzzy_axions_orientifold_gap.py).
It uses `h11_minus_zero_classes` as the retained accepted class identifiers.
This is valid for these two artifacts because their inherited and
\(h^{1,1}_-=0\) counts are equal; the analyzer now rejects an artifact where
those counts differ.

The mutually exclusive partition of every code-unaccepted FRST class is:

| \(h^{1,1}\) | certified inherited | unaccepted with candidate-linked unavailable evidence | unaccepted not classified by retained terminal ledger |
|---:|---:|---:|---:|
| 2 | 10 | 6 | 20 |
| 3 | 80 | 28 | 166 |

A class enters the candidate-linked category only when an
`unresolved_components` row contains both a candidate ID and a reason code.
Generic `surface_attempts` rows and partial candidate contexts are annotations,
not exhaustive class verdicts. The code-unaccepted classes are not known to be
the paper's missing classes, because the two populations have no shared
published identifier map.

If every retained candidate-linked unavailable class were granted while all
other verdicts, search scope, and evidence boundaries remained fixed, the
conditional ceilings would be:

| \(h^{1,1}\) | certified | candidate-linked unavailable | conditional ceiling | target | deficit still unexplained |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 6 | 16 | 32 | **16** |
| 3 | 80 | 28 | 108 | 253 | **145** |

This is accounting, not a prediction of acceptance. It proves a narrower and
important point: the retained candidate-linked `non_smooth_ambient_cone`
evidence cannot, by itself, close the Table 1 gap. The earlier detailed audit
was therefore too broad when it presented all skipped surface rows as the
principal explanation of the population deficit.

## 3. What the diagnostic rows do and do not show

| \(h^{1,1}\) | surface attempts | certified rows | skipped rows | skip reason | unresolved candidate-component rows |
|---:|---:|---:|---:|---|---:|
| 2 | 1,164 | 762 | 402 | `non_smooth_ambient_cone` | 33 |
| 3 | 5,510 | 3,618 | 1,892 | `non_smooth_ambient_cone` | 117 |

These are evidence attempts, not CY counts. Multiple rows can belong to the
same candidate and class, and generic rows need not be linked to any retained
candidate. `non_smooth_ambient_cone` means that the current local
Cartier/\(n_S\) certificate requires a smooth unimodular ambient four-cone and
did not obtain one. It does not mean the fixed locus or hypersurface is
singular. Moritz et al. explicitly allow singular or non-simplicial symmetric
ambient fans when the Calabi–Yau hypersurface and fixed locus avoid their
singularities.

## 4. Source-to-code gap matrix

| Priority | Source requirement | Current code or evidence | Likely effect | Presently measurable? |
|---:|---|---|---|---|
| 1 | Count all admissible inherited O3/O7 actions at the CY-class level. | The artifacts retain accepted class IDs, surface diagnostics, and only partial candidate contexts. They do not retain a complete per-candidate terminal ledger. | Prevents attribution of 20/166 unaccepted classes and blocks a true candidate-by-candidate funnel. | Yes as missing coverage; no as acceptance magnitude. |
| 2 | Construct an \(L\)-symmetric fan using symmetrized heights and, when required, subdivision/removal/gluing. | `validate_orientifold` requires \(L\) to preserve the one supplied FRST, and the manifest declares `supplied_frst_only`. | Can omit orientifolds realized only on another compatible symmetric subdivision. | Not exhaustively measured. A five-case h11=3 probe was inconclusive: one fan was already represented by another class and four failed the two-face condition. |
| 3 | For each pointwise invariant cone \(\sigma\), reduce the phase label modulo \(\operatorname{span}(\sigma)\), then remove contained fixed components. | `_fixed_component_records` removes a proper-face record only when its canonical `nu` is exactly equal. | Can retain redundant components and allow one unavailable/rejected record to block a candidate. | A diagnostic-row probe found no such pair in 1,164/5,510 surface rows, but full fixed-component records were not retained, so the population risk remains unmeasured. |
| 4 | Use the half-ray form of eq. (4.35) only in its stated smooth-\(\sigma\), smooth-normal-direction case; otherwise use the general quotient condition. | The code applies the half-ray integrality formula to every pointwise invariant original-fan face. | For non-smooth \(\sigma\), it can mis-enumerate or mislabel fixed components. The direction and size are unknown. | No. |
| 5 | Certify smooth fixed loci even in admissible singular/non-simplicial symmetric ambient fans. | General-\(L\) surface evidence stops at a non-unimodular containing four-cone. | Conservative undercount for source-admissible cases. | Only 6/28 unaccepted classes are linked to this boundary in retained candidate rows; generic row counts are much larger and cannot be promoted to class counts. |
| 6 | Compute the integral \(H^2\) action exactly. | `validate_orientifold` obtains coefficients with floating `numpy.linalg.lstsq`, rounds, and checks with a tolerance. | Numerical false rejection is possible in principle. | No observed h11=2/3 failure; low priority until the terminal ledger shows this status. |

The two dominant uncertainties are therefore search completeness and missing
terminal provenance, not a demonstrated geometric disagreement with the
paper. The non-smooth-ambient certificate is a real implementation boundary,
but the artifacts show that it is not a complete explanation.

## 5. Reproducibility boundary

| \(h^{1,1}\) | artifact | bytes | SHA-256 |
|---:|---|---:|---|
| 2 | `/private/tmp/cyax-orientifold-rerun-h11-2-20260820.json` | 48,576,572 | `c4cc9bc93a8d590d42a2ef98fdb58156da442493f07d5854b8ca8e4873f9ebb0` |
| 3 | `/private/tmp/cyax-orientifold-rerun-h11-3-20260820.json` | 264,658,672 | `05ce23cc5fa0819a83ecbde0d8784628e5e540a874e8e6ba291ce2ed30f63e42` |

Both artifacts use schema
`cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1`, declare complete input
populations, and identify source commit
`e7a8f51d775410ba847eea886471af8c5accc3bd`. Both also declare
`git_dirty=true`. The artifact hashes fix the results, but the exact source
tree that generated them is not reconstructible from the recorded commit
alone. A future publication-grade rerun must record a clean commit or a
cryptographic diff/tree hash.

## 6. Gate for h11 = 4

The earlier h11=4 values 435 inherited and 429 with \(h^{1,1}_-=0\) came from
the superseded auxiliary-\(\Sigma_L\) implementation. They are not a current
comparison with Table 1 values 1,559 and 1,554, and they must not be reused as
the corrected h11=4 result.

Before the expensive h11=4 rerun:

1. Add a lossless class/candidate terminal ledger to the reproduction
   artifact. Record every matrix validation failure, every \((L,t,\lambda_f)\)
   terminal status, every fixed-component reason, and the accepted witness.
2. Implement and regression-test phase-label reduction modulo
   \(\operatorname{span}(\sigma)\), including contained-component removal.
3. Separate the smooth-\(\sigma\) shortcut from the general fixed-component
   condition, with a source-derived treatment for non-smooth \(\sigma\).
4. Define the symmetric-fan search population and record fan-construction
   terminal failures. A supplied-FRST-only run must retain that label and must
   not be called a complete reproduction of the source population.
5. Rerun h11=2 and h11=3 first. Require stable base/trilayer counts, a complete
   terminal funnel, exact artifact provenance, and independent review of any
   newly accepted class.
6. Only then run h11=4 with the same schema and checks.

This sequence avoids spending the h11=4 runtime on an artifact that cannot
explain its own deficit.

## 7. Verification and version impact

The analyzer is artifact-only and deliberately rejects h11=4. It refuses to
overwrite an existing output path. The focused verification is recorded in
the final handoff for this change.

Adding this analyzer and documentation does not change package or scientific
behavior, so its immediate version impact is none. The proposed fixes to
fixed-component enumeration or symmetric-fan coverage are scientific-behavior
changes. Under the repository's pre-1.0 policy, they require at least a minor
version increment at the reviewed release boundary; no feature-branch version
bump is included here.

## 8. Primary sources

- Sheridan et al., [Table 1 and orientifold population definitions](https://arxiv.org/html/2412.12012).
- Moritz et al., [invariant-fan construction, fixed components, and smoothness conditions](https://arxiv.org/html/2305.06363).
- Sheridan et al., [supplemental fuzzy-axion data repository](https://github.com/sheride/fuzzy_axions). Its published data do not provide the complete orientifold-class manifest needed for a set-level comparison.
