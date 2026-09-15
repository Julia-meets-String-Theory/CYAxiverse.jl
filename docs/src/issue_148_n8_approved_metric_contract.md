# Issue 148 approved N8 metric contract

The scientific owner explicitly approved the recommended contract in this
manager session on 2026-09-10. This resolves the boundary identified in the
[metric audit](issue_148_n8_metric_boundary_audit.md), commit
`dd12209bd1bdcbc37d8be2f6801786fee854ab8d`. G0/G1 remain PASS. This approval
authorizes G2 implementation; it does not accept G2 or authorize G3 work.

## Approved scientific contract: P96

- Scientific N8 outputs use period-one GLSM coordinates `theta` with Fourier
  argument `2pi Q theta` and `K_theta(k)=M96/k^2`.
- `M96` is the precise reconstructed reference metric reproducing the Eq.96
  benchmark spectrum, in the same GLSM basis. The reconstruction and its
  source identity must be verified at working precision.
- For the equivalent raw-radian coordinate `x=2pi theta`, the metric is
  `G_x(k)=M96/[k^2(2pi)^2]`. Other basis changes require the corresponding
  explicit metric congruence transformation.
- The author executable's raw-radian metric `M96/k^2` is retained only as
  explicitly labeled author reproduction (A96), not as the scientific G2
  canonical normalization. Preserve legacy behavior rather than silently
  changing unrelated author-reproduction APIs.
- The owner deliberately selects the reported Eq.96/CYTools matrix authority
  over the `2M96` obtained from literal differentiation of displayed Eqs.17/23.
  The factor-two source inconsistency remains documented. Do not describe
  `M96` as a derivation of the displayed Eq.23 normalization.

This approval fixes both the coordinate convention and numerical-matrix
authority. It does not change the potential, phase, radial scale, physical
claim boundary, or persisted schema. Matching the catastrophe scale does not
establish equivalence between P96 and A96 canonical values.

## G2 execution constraints

Reproduce the source N8 radial multifield catastrophe using genuine branch
continuation with intrinsic identity and independent augmented-solve checks.
Use pseudo-arclength as the preferred baseline, or demonstrate the adequacy
of an equivalent well-posed method near singularity. Audit the N8 source and
calculation path for hidden Float64 construction; use exact/rational source
data and a genuine Float64-to-BigFloat refinement/stability ladder. Include
gradient, null-vector, normalization, canonical Hessian, projected higher-
derivative classification, and old post-hoc matching comparisons. No G3 work.

The published twelve-term potential and ten-term author truncation remain
distinct. Use the source twelve-term benchmark as the primary G2 reference and
the corresponding existing augmented solve as an independent target; label any
ten-term comparison explicitly. Do not tune term selection or normalization to
force benchmark agreement. Escalate evidence contradicting that source-fixed
interpretation.

Implementation worker: standard-context Opus4.6 through Claude Code
`--safe-mode`, then Luna/xhigh if Opus reaches its usage limit. Fresh Sol/xhigh
independent review follows implementation; the manager alone adjudicates G2.
