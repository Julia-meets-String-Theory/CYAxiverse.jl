# Final audit summary: h11 = 2, 3, 4

Status: complete, post-review, 2026-08-20
Detailed report: [full self-contained report](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md)

> **Rerun status (2026-08-20, corrected general-`L` code):** The general-`L`
> overcount repair is implemented and validated by an h11=2 and h11=3 rerun:
> fixed components are labelled by original-`Sigma` pointwise-invariant cones
> (source eqs. (4.33)–(4.35)) with their original ray scale, and the eq. (4.45)
> parity condition uses the corrected `any`-pairing facet test. At h11=2 no
> accepted orientifold changed; at h11=3 one previously accepted inherited
> orientifold is now correctly rejected (polytope 229), moving the h11=3
> inherited and \(h^{1,1}_-=0\) lower bounds from 81 to 80. This rerun covers
> **h11=2 and h11=3 only**; every h11=4 figure below is the prior
> auxiliary-`Sigma_L` value, superseded pending an h11=4 rerun. See the
> [checkpoint review note](./fuzzy_axions_2412_12012_checkpoint_review_20260820.md).

## Result at a glance

The h11=2 and h11=3 audits were rerun with the corrected general-`L` code and
the h11=4 row is shown from the prior run. Each JSON output is valid, hashed,
and free of the known CYTools cache-save warning.

| h11 | favorable | FRST classes | inherited O3/O7 evidence | Table 1 target | h11-minus-zero evidence | Table 1 target | h21-plus-zero trilayer | Table 1 target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 36 | 10 | 32 | 10 | 32 | 11 | 11 |
| 3 | 243 | 274 | 80 | 253 | 80 | 253 | 66 | 66 |
| 4 | 1185 | 1760 | 435† | 1559 | 429† | 1554 | 267 | 267 |

† The h11=4 inherited and \(h^{1,1}_-=0\) counts are the prior
auxiliary-`Sigma_L` values, superseded pending an h11=4 rerun.

Interpretation: the base populations and \(h^{2,1}_+=0\) benchmark reproduce
Table 1, while the inherited-orientifold counts remain conservative
code-certified lower bounds. This is not evidence that Table 1 is wrong; it
identifies an evidence-coverage gap in the current smoothness certificate.

## Diagnostic status

| h11 | general-L attempts | diagnostic evidence rows | skipped | skipped reason | unresolved candidate components |
|---:|---:|---:|---:|---|---:|
| 2 | 1164 | 762 | 402 | non_smooth_ambient_cone | 33 |
| 3 | 5510 | 3618 | 1892 | non_smooth_ambient_cone | 117 |
| 4† | 25398 | 17024 | 8374 | non_smooth_ambient_cone | 327 |

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
- h11=4† (superseded, prior run):
  [cyax-orientifold-final-reviewed-h11-4-20260819.json](/private/tmp/cyax-orientifold-final-reviewed-h11-4-20260819.json) —
  1,365,317,574 bytes,
  SHA-256 a5849fa7c02040dc4986653168262eac4d299a492d58fe7728e948afcd875769.

Focused verification: 38/38 candidate tests, 1/1 fixed-locus test, and 8/8
driver tests passed; the full Julia 1.12 `Pkg.test()` suite also passed.
The implementation and review repairs are summarized in the detailed report’s
[review section](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#6-review-findings-and-implemented-repairs).

## What to do next

Keep the conservative statuses. The immediate outstanding item is to rerun the
h11=4 population with the corrected general-`L` code so its superseded rows can
be refreshed. The deeper next scientific step is to derive and validate
source-compatible Cartier/intersection and orbifold data for non-unimodular
ambient provenance cones, separately assess the non-simplicial symmetric fan,
then rerun the populations. See the detailed report’s
[mathematical framework](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#5-mathematical-and-physical-framework)
and [Table 1 comparison](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#4-comparison-with-sheridan-et-al-table-1).

Primary papers: [Moritz et al.](https://arxiv.org/abs/2305.06363) and
[Sheridan et al.](https://arxiv.org/abs/2412.12012).
