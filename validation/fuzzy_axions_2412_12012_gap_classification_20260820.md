# Table 1 orientifold-gap review: h11 = 2, 3, 4, 5

Date: 2026-08-20 (updated 2026-08-21: h11 = 4 and h11 = 5 completed)

Paper: Sheridan et al., [*Fuzzy Axions and Associated Relics*](https://arxiv.org/html/2412.12012), especially Table 1

Orientifold source: Moritz et al., [*Orientifolds of Calabi–Yau hypersurfaces*](https://arxiv.org/html/2305.06363)

Scope: corrected complete-population artifacts for $h^{1,1}=2,3$ (single
reproduction artifact) and $h^{1,1}=4,5$ (sharded, details-absent
terminal-ledger populations, analyzed with the sharded audit mode)

## 1. Outcome and confidence

The discrepancy is real, but the artifacts cannot identify it as a
paper-versus-code set difference. They establish an exact aggregate count gap
at every $h^{1,1}$. All Table 1 target values below are source-verified
against arXiv:2412.12012v1 Table 1 (`tab:ScanData`).

| $h^{1,1}$ | favorable t/c | FRST t/c | inherited t/c | inherited gap | $h^{1,1}_-=0$ t/c | $h^{1,1}_-=0$ gap | trilayer t/c |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 / 36 | 36 / 36 | 32 / 10 | **22** | 32 / 10 | **22** | 11 / 11 |
| 3 | 243 / 243 | 274 / 274 | 253 / 80 | **173** | 253 / 80 | **173** | 66 / 66 |
| 4 | 1,185 / 1,185 | 1,760 / 1,760 | 1,559 / 1,146 | **413** | 1,554 / 1,144 | **410** | 267 / 267 |
| 5 | 4,897 / 4,897 | 11,713 / 11,713 | 9,530 / 6,219 | **3,311** | 9,459 / 6,186 | **3,273** | 1,033 / 1,033 |

The base populations (favorable polytopes, FRST classes) and the
$h^{2,1}_+=0$ trilayer benchmark reproduce Table 1 **exactly** at all four
$h^{1,1}$. The inherited and $h^{1,1}_-=0$ counts are conservative
code-certified lower bounds, not paper-error findings.

Confidence is high in these arithmetic comparisons and in the base-population
and trilayer matches. Confidence is not sufficient to allocate the missing
counts to individual paper classes: Table 1 publishes aggregate counts, the
paper's supplemental repository does not publish a complete orientifold-class
manifest, and (for $h^{1,1}=2,3$) the single artifact does not retain every
candidate terminal record. For $h^{1,1}=4,5$ the complete terminal ledger
does retain every record (see §2), but Table 1 still supplies no class-identity
map, so a set-level attribution remains impossible.

The identity-sector values 25 and 198 are action-level parity diagnostics.
They are not Table 1 inherited-orientifold counts. The model stage is null in
both artifacts, so the Table 1 model values 2 and 263 were not reproduced.

## 2. Exact artifact-only classification

The deterministic analysis is implemented in
[`scripts/analyze_fuzzy_axions_orientifold_gap.py`](../scripts/analyze_fuzzy_axions_orientifold_gap.py)
and tested by
[`scripts/test_analyze_fuzzy_axions_orientifold_gap.py`](../scripts/test_analyze_fuzzy_axions_orientifold_gap.py).
For $h^{1,1}=2,3$ it uses `h11_minus_zero_classes` as the retained accepted
class identifiers; this is valid because those artifacts' inherited and
$h^{1,1}_-=0$ counts are equal (the analyzer rejects an artifact where they
differ). For $h^{1,1}=4,5$ it reads the **complete sharded terminal ledger**:
every matrix validation and every enumerated $(L,t,\lambda_f)$ candidate emits
one terminal row, so each class is classified exhaustively. Shards are unioned
by geometry identity (`polytope_normal_form_id`), each sidecar is SHA-256
verified, and the class set is derived from the ledger rows themselves (no
`--keep-details` needed).

The mutually exclusive partition of every code-unaccepted FRST class is:

| $h^{1,1}$ | certified inherited | unaccepted: unavailable evidence (ceiling driver) | unaccepted: terminal rejection / unclassified | classification |
|---:|---:|---:|---:|---|
| 2 | 10 | 6 | 20 | partial (artifact-only) |
| 3 | 80 | 28 | 166 | partial (artifact-only) |
| 4 | 1,146 | 85 | 529 | **exhaustive** (complete terminal ledger) |
| 5 | 6,219 | 823 | 4,671 | **exhaustive** (complete terminal ledger) |

For $h^{1,1}=2,3$, a class enters the ceiling-driver category only when an
`unresolved_components` row contains both a candidate ID and a reason code;
generic `surface_attempts` rows and partial candidate contexts are annotations,
and the last column (20/166) is *unclassified* residue the single artifact
cannot resolve. For $h^{1,1}=4,5$, the complete ledger classifies **every**
unaccepted class as either candidate-linked unavailable evidence or a
**definitive terminal rejection** — there is no unclassified residue. In no
case are the code-unaccepted classes known to be the paper's missing classes,
because the populations share no published identifier map.

If every candidate-linked unavailable class were granted while all other
verdicts, search scope, and evidence boundaries remained fixed, the conditional
ceilings would be:

| $h^{1,1}$ | certified | unavailable (ceiling driver) | inherited ceiling | inherited target | inherited deficit | $h^{1,1}_-=0$ ceiling | $h^{1,1}_-=0$ target | $h^{1,1}_-=0$ deficit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 6 | 16 | 32 | **16** | 16 | 32 | **16** |
| 3 | 80 | 28 | 108 | 253 | **145** | 108 | 253 | **145** |
| 4 | 1,146 | 85 | 1,231 | 1,559 | **328** | 1,229 | 1,554 | **325** |
| 5 | 6,219 | 823 | 7,042 | 9,530 | **2,488** | 7,009 | 9,459 | **2,450** |

This is accounting, not a prediction of acceptance. It proves a narrower and
important point at every $h^{1,1}$: the retained candidate-linked
unavailable-evidence classes cannot, by themselves, close the Table 1 gap — a
large residual deficit survives even the ceiling. For $h^{1,1}=4,5$ the
statement is stronger still, because the exhaustive ledger shows that residual
is composed of definitive terminal rejections, not of unclassified classes. The
earlier detailed audit was therefore too broad when it presented all skipped
surface rows as the principal explanation of the population deficit.

## 3. What the diagnostic rows do and do not show

These per-surface reason diagnostics are retained only for the $h^{1,1}=2,3$
artifacts. The $h^{1,1}=4,5$ populations were run **ledger-only**
(`details=null`, no reason diagnostics); their evidence boundary is instead read
from the exhaustive terminal ledger (the "unavailable evidence" column of §2).

| $h^{1,1}$ | surface attempts | certified rows | skipped rows | skip reason | unresolved candidate-component rows |
|---:|---:|---:|---:|---|---:|
| 2 | 1,164 | 762 | 402 | `non_smooth_ambient_cone` | 33 |
| 3 | 5,510 | 3,618 | 1,892 | `non_smooth_ambient_cone` | 117 |

These are evidence attempts, not CY counts. Multiple rows can belong to the
same candidate and class, and generic rows need not be linked to any retained
candidate. `non_smooth_ambient_cone` means that the current local
Cartier/$n_S$ certificate requires a smooth unimodular ambient four-cone and
did not obtain one. It does not mean the fixed locus or hypersurface is
singular. Moritz et al. explicitly allow singular or non-simplicial symmetric
ambient fans when the Calabi–Yau hypersurface and fixed locus avoid their
singularities.

## 4. Source-to-code gap matrix

| Priority | Source requirement | Current code or evidence | Likely effect | Presently measurable? |
|---:|---|---|---|---|
| 1 | Count all admissible inherited O3/O7 actions at the CY-class level. | The artifacts retain accepted class IDs, surface diagnostics, and only partial candidate contexts. They do not retain a complete per-candidate terminal ledger. | Prevents attribution of 20/166 unaccepted classes and blocks a true candidate-by-candidate funnel. | Yes as missing coverage; no as acceptance magnitude. |
| 2 | Construct an $L$-symmetric fan using symmetrized heights and, when required, subdivision/removal/gluing. | `validate_orientifold` requires $L$ to preserve the one supplied FRST, and the manifest declares `supplied_frst_only`. | Can omit orientifolds realized only on another compatible symmetric subdivision. | Not exhaustively measured. A five-case h11=3 probe was inconclusive: one fan was already represented by another class and four failed the two-face condition. |
| 3 | For each pointwise invariant cone $\sigma$, reduce the phase label modulo $\operatorname{span}(\sigma)$, then remove contained fixed components. | `_fixed_component_records` removes a proper-face record only when its canonical `nu` is exactly equal. | Can retain redundant components and allow one unavailable/rejected record to block a candidate. | A diagnostic-row probe found no such pair in 1,164/5,510 surface rows, but full fixed-component records were not retained, so the population risk remains unmeasured. |
| 4 | Use the half-ray form of eq. (4.35) only in its stated smooth-$\sigma$, smooth-normal-direction case; otherwise use the general quotient condition. | The code applies the half-ray integrality formula to every pointwise invariant original-fan face. | For non-smooth $\sigma$, it can mis-enumerate or mislabel fixed components. The direction and size are unknown. | No. |
| 5 | Certify smooth fixed loci even in admissible singular/non-simplicial symmetric ambient fans. | General-$L$ surface evidence stops at a non-unimodular containing four-cone. | Conservative undercount for source-admissible cases. | Only 6/28 unaccepted classes are linked to this boundary in retained candidate rows; generic row counts are much larger and cannot be promoted to class counts. |
| 6 | Compute the integral $H^2$ action exactly. | `validate_orientifold` obtains coefficients with floating `numpy.linalg.lstsq`, rounds, and checks with a tolerance. | Numerical false rejection is possible in principle. | No observed h11=2/3 failure; low priority until the terminal ledger shows this status. |

The two dominant uncertainties are therefore search completeness and missing
terminal provenance, not a demonstrated geometric disagreement with the
paper. The non-smooth-ambient certificate is a real implementation boundary,
but the artifacts show that it is not a complete explanation.

## 5. Reproducibility boundary

`h11 = 2, 3` — single corrected reproduction artifacts:

| $h^{1,1}$ | artifact | bytes | SHA-256 |
|---:|---|---:|---|
| 2 | `/private/tmp/cyax-orientifold-rerun-h11-2-20260820.json` | 48,576,572 | `c4cc9bc93a8d590d42a2ef98fdb58156da442493f07d5854b8ca8e4873f9ebb0` |
| 3 | `/private/tmp/cyax-orientifold-rerun-h11-3-20260820.json` | 264,658,672 | `05ce23cc5fa0819a83ecbde0d8784628e5e540a874e8e6ba291ce2ed30f63e42` |

Both use schema `cyaxiverse-fuzzy-axions-h11-4-reproduction-1.1`, declare
complete input populations, identify source commit
`e7a8f51d775410ba847eea886471af8c5accc3bd`, and declare `git_dirty=true`. The
artifact hashes fix the results, but the exact source tree is not
reconstructible from the recorded commit alone.

`h11 = 4, 5` — sharded, ledger-only populations, preserved zstd-compressed
(lossless; each sidecar's recorded `sidecar_sha256` still validates after
decompression) with per-directory `SHA256SUMS.txt`:

| $h^{1,1}$ | durable population directory | shards | records / classes |
|---:|---|---:|---|
| 4 | `data/orientifold_h11_4_population_20260821/` (`h4.shard00{0..3}-of-004.*`) | 4 | 86,156 / 1,760 |
| 5 | `data/orientifold_h11_5_population_20260821/` (`h5f.shard00{0..2}-of-003.*`) | 3 | 483,546 / 11,713 |

Both were produced at source commit `b1aa7e4…` (schema
`cyaxiverse-orientifold-terminal-ledger-1.1`). The gap-analyzer output for each
is preserved beside its population as `h{4,5}_gap_analysis.json.zst` (with a
`.sha256`), regenerable from the population in ≈1 min (h11=4) / ≈7 min (h11=5).
A future publication-grade rerun should still record a clean commit or a
cryptographic diff/tree hash.

## 6. h11 = 4 and h11 = 5 completion (formerly gated)

The earlier h11=4 values 435 inherited and 429 with $h^{1,1}_-=0$ came from
the superseded auxiliary-$\Sigma_L$ implementation. They are **not** the
corrected result and must not be reused: the corrected h11=4 figures are
1,146 / 1,144 (§1).

The prerequisites this section previously listed are now satisfied. Both
populations were regenerated under the current normal-form geometry keying with
a lossless terminal ledger (every matrix validation and $(L,t,\lambda_f)$
terminal status, fixed-component reason, and accepted witness), h11=2/3 were
rerun first and remain stable, and the gap-analyzer was extended to consume the
sharded ledger-only populations. The six Table 1 target values per $h^{1,1}$
were source-verified against arXiv:2412.12012v1 `tab:ScanData` before entering
`TABLE_1_TARGETS`. The remaining source-method items (phase-label reduction
modulo $\operatorname{span}(\sigma)$; a source-derived non-smooth-$\sigma$
treatment; a defined symmetric-fan search population with recorded
fan-construction failures) are open smoothness-certificate boundaries that
would raise the certified counts if resolved; they are not required for the
present conservative lower-bound + conditional-ceiling accounting.

## 7. Verification and version impact

The analyzer accepts h11=2,3 (single artifact) and h11=4,5 (sharded, details-
absent terminal ledger); it refuses to overwrite an existing output path. The
suite (`python -m unittest test_analyze_fuzzy_axions_orientifold_gap`) is green,
including the h11=2/3 regression, the source-verified-target guard, and the
sharded union / disjointness / completeness / SHA-mismatch cases. The sharded
mode was validated end-to-end on the real h11=4 and h11=5 populations,
reproducing the code counts exactly (certified 1,146 / 6,219 over 1,760 / 11,713
classes).

Adding the h11=4,5 targets to `TABLE_1_TARGETS`/`SUPPORTED_H11` is a
scientific-accounting change (a new certified/ceiling claim for two $h^{1,1}$),
and the sharded-input mode is a reader/tooling change. Under the repository's
pre-1.0 policy both warrant at least a minor version increment at the reviewed
`vmm`→`main` boundary; no feature-branch `Project.toml` bump is included here.
The open smoothness-certificate fixes in §6 remain separate future
scientific-behavior changes.

## 8. Primary sources

- Sheridan et al., [Table 1 and orientifold population definitions](https://arxiv.org/html/2412.12012).
- Moritz et al., [invariant-fan construction, fixed components, and smoothness conditions](https://arxiv.org/html/2305.06363).
- Sheridan et al., [supplemental fuzzy-axion data repository](https://github.com/sheride/fuzzy_axions). Its published data do not provide the complete orientifold-class manifest needed for a set-level comparison.
