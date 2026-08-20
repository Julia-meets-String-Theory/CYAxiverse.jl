# Final audit summary: h11 = 2, 3, 4

Status: complete, post-review, 2026-08-20
Detailed report: [full self-contained report](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md)

## Result at a glance

The final Luna Max audits completed successfully in the reviewed target
worktree. All three JSON outputs are valid, hashed, and free of the known
CYTools cache-save warning.

| h11 | favorable | FRST classes | inherited O3/O7 evidence | Table 1 target | h11-minus-zero evidence | Table 1 target | h21-plus-zero trilayer | Table 1 target |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 36 | 36 | 10 | 32 | 10 | 32 | 11 | 11 |
| 3 | 243 | 274 | 81 | 253 | 81 | 253 | 66 | 66 |
| 4 | 1185 | 1760 | 435 | 1559 | 429 | 1554 | 267 | 267 |

Interpretation: the base populations and \(h^{2,1}_+=0\) benchmark reproduce
Table 1, while the inherited-orientifold counts remain conservative
code-certified lower bounds. This is not evidence that Table 1 is wrong; it
identifies an evidence-coverage gap in the current smoothness certificate.

## Diagnostic status

| h11 | general-L attempts | certified | skipped | skipped reason | unresolved candidate components |
|---:|---:|---:|---:|---|---:|
| 2 | 1362 | 862 | 500 | non_smooth_ambient_cone | 33 |
| 3 | 6314 | 3934 | 2380 | non_smooth_ambient_cone | 117 |
| 4 | 25398 | 17024 | 8374 | non_smooth_ambient_cone | 327 |

The complete interpretation is in the detailed report’s
[general-\(L\) diagnostics section](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#32-general-l-fixed-surface-diagnostics)
and [residual limitations](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#8-residual-limitations-and-next-scientific-boundary).

## Verification artifacts

- [h11=2 JSON](/private/tmp/cyax-orientifold-final-reviewed-h11-2-20260819.json) —
  54,557,352 bytes,
  SHA-256 27e7b18148bc6730089c86fabb99bd2dae0ff8c91ed721ce302f60106feb647a.
- [h11=3 JSON](/private/tmp/cyax-orientifold-final-reviewed-h11-3-20260819.json) —
  286,183,888 bytes,
  SHA-256 9de41a8967895c3f0de36317b83ce44bb978ae8971b84c573dc30795e2c6dc9f.
- [h11=4 JSON](/private/tmp/cyax-orientifold-final-reviewed-h11-4-20260819.json) —
  1,365,317,574 bytes,
  SHA-256 a5849fa7c02040dc4986653168262eac4d299a492d58fe7728e948afcd875769.

Focused verification: 30/30 candidate tests and 1/1 fixed-locus test passed.
The implementation and review repairs are summarized in the detailed report’s
[review section](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#6-review-findings-and-implemented-repairs).

## What to do next

Keep the conservative statuses. The next scientific step is to implement and
validate the source’s orbifold/crepant treatment for non-smooth ambient
provenance cones, then rerun the populations. See the detailed report’s
[mathematical framework](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#5-mathematical-and-physical-framework)
and [Table 1 comparison](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md#4-comparison-with-sheridan-et-al-table-1).

Primary papers: [Moritz et al.](https://arxiv.org/abs/2305.06363) and
[Sheridan et al.](https://arxiv.org/abs/2412.12012).
