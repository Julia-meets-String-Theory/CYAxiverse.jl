# Final audit summary: h11 = 2, 3, 4, 5

Status: complete, post-review, 2026-08-20 (updated 2026-08-21: h11 = 4 and
h11 = 5 completed)
Detailed report: [full self-contained report](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md)

> **Rerun status (updated 2026-08-21):** The general-`L` overcount repair is
> implemented and validated. h11=2 and h11=3 were rerun with the corrected code
> (fixed components labelled by original-`Sigma` pointwise-invariant cones,
> source eqs. (4.33)–(4.35), with their original ray scale; eq. (4.45) parity
> uses the corrected `any`-pairing facet test): at h11=2 no accepted orientifold
> changed; at h11=3 one previously accepted inherited orientifold is now
> correctly rejected (polytope 229), moving the h11=3 lower bounds from 81 to 80.
> **h11=4 and h11=5 have since been regenerated** as fresh, sharded, ledger-only
> populations under the current normal-form geometry keying; the prior
> auxiliary-`Sigma_L` h11=4 values (435 / 429) are superseded and replaced below.
> See the [checkpoint review note](./fuzzy_axions_2412_12012_checkpoint_review_20260820.md)
> and the [gap-classification review](./fuzzy_axions_2412_12012_gap_classification_20260820.md).

## Result at a glance

All four \(h^{1,1}\) are now current: h11=2,3 from single corrected artifacts and
h11=4,5 from their sharded terminal-ledger populations. Table 1 targets are
source-verified against arXiv:2412.12012v1 `tab:ScanData`.

| h11 | favorable | FRST classes | inherited O3/O7 evidence | Table 1 target | h11-minus-zero evidence | Table 1 target | h21-plus-zero trilayer | Table 1 target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 36 | 10 | 32 | 10 | 32 | 11 | 11 |
| 3 | 243 | 274 | 80 | 253 | 80 | 253 | 66 | 66 |
| 4 | 1,185 | 1,760 | 1,146 | 1,559 | 1,144 | 1,554 | 267 | 267 |
| 5 | 4,897 | 11,713 | 6,219 | 9,530 | 6,186 | 9,459 | 1,033 | 1,033 |

Interpretation: the base populations and \(h^{2,1}_+=0\) benchmark reproduce
Table 1 exactly at all four \(h^{1,1}\), while the inherited-orientifold counts
remain conservative code-certified lower bounds. This is not evidence that
Table 1 is wrong; it identifies an evidence-coverage gap in the current
smoothness certificate. The gap-classification review quantifies the
conditional ceiling: even promoting every candidate-linked unavailable class
leaves a large residual deficit at every \(h^{1,1}\).

## Diagnostic status

| h11 | general-L attempts | diagnostic evidence rows | skipped | skipped reason | unresolved candidate components |
|---:|---:|---:|---:|---|---:|
| 2 | 1164 | 762 | 402 | non_smooth_ambient_cone | 33 |
| 3 | 5510 | 3618 | 1892 | non_smooth_ambient_cone | 117 |

Per-surface reason diagnostics exist only for the h11=2,3 artifacts. The fresh
h11=4 and h11=5 populations were run ledger-only, so their evidence boundary is
read from the complete terminal ledger instead (85 and 823 candidate-linked
unavailable-evidence classes respectively; see the
[gap-classification review](./fuzzy_axions_2412_12012_gap_classification_20260820.md)).
These are diagnostic evidence rows, not certified fixed components for every
specific torus shift and not accepted orientifold counts. The complete
interpretation is in the detailed report’s
[general-\(L\) diagnostics section](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#32-general-l-fixed-surface-diagnostics)
and [residual limitations](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#8-residual-limitations-and-next-scientific-boundary).

## Verification artifacts

- [h11=2 JSON](/private/tmp/cyax-orientifold-rerun-h11-2-20260820.json) —
  48,576,572 bytes,
  SHA-256 c4cc9bc93a8d590d42a2ef98fdb58156da442493f07d5854b8ca8e4873f9ebb0.
- [h11=3 JSON](/private/tmp/cyax-orientifold-rerun-h11-3-20260820.json) —
  264,658,672 bytes,
  SHA-256 05ce23cc5fa0819a83ecbde0d8784628e5e540a874e8e6ba291ce2ed30f63e42.
- h11=4 (current, sharded ledger-only): durable population
  `data/orientifold_h11_4_population_20260821/` — 4 shards, 86,156 records /
  1,760 classes, zstd-compressed with `SHA256SUMS.txt`; gap-analyzer output
  `h4_gap_analysis.json.zst`.
- h11=5 (current, sharded ledger-only): durable population
  `data/orientifold_h11_5_population_20260821/` — 3 shards, 483,546 records /
  11,713 classes, zstd-compressed with `SHA256SUMS.txt`; gap-analyzer output
  `h5_gap_analysis.json.zst`.
- h11=4† (superseded, prior run, retained for history only):
  [cyax-orientifold-final-reviewed-h11-4-20260819.json](/private/tmp/cyax-orientifold-final-reviewed-h11-4-20260819.json) —
  1,365,317,574 bytes,
  SHA-256 a5849fa7c02040dc4986653168262eac4d299a492d58fe7728e948afcd875769.

Focused verification: 38/38 candidate tests, 1/1 fixed-locus test, and 8/8
driver tests passed; the full Julia 1.12 `Pkg.test()` suite also passed. The
gap-analyzer suite (27 tests) is green and its sharded mode reproduces the
h11=4/5 code counts exactly on the real populations.
The implementation and review repairs are summarized in the detailed report’s
[review section](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#6-review-findings-and-implemented-repairs).

## What to do next

Keep the conservative statuses. The h11=4 (and h11=5) reruns with the corrected
general-`L` code are now complete, so the superseded rows are refreshed. The
deeper next scientific step is unchanged: derive and validate source-compatible
Cartier/intersection and orbifold data for non-unimodular ambient provenance
cones, separately assess the non-simplicial symmetric fan, then rerun the
populations — this is what would raise the certified counts toward the Table 1
targets. See the detailed report’s
[mathematical framework](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#5-mathematical-and-physical-framework)
and [Table 1 comparison](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#4-comparison-with-sheridan-et-al-table-1).

Primary papers: [Moritz et al.](https://arxiv.org/abs/2305.06363) and
[Sheridan et al.](https://arxiv.org/abs/2412.12012).
