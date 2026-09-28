# S4 bosenova formula check and topology correction

Model ID: `bosenova-10-solar-mass`. Route identity:
`REFERENCE_2021_BOSENOVA`. This uses the frozen source-mode manifest
`a891735d45dd80e57689917416577203e43a68ec0c7e96b23897cb7c306725ed` and
numerical method manifest
`7b9c1cf9e4787372801125d7c195288d36f3cb9e094b0a2ec179493c8e8392d5`.
The corrective topology-method addendum is
`909d283bd19af956ee2881537ee90dce07c0e1c7976717ac27cc7ca6402f7d39`.
It was finalized and hash-bound into the implementation before these S1 and S4
outputs were recomputed. Source routes are Eq. (9) and Eq. (13) of
arXiv:2103.06812v2, and the strict Eq. (26) criterion of
arXiv:2009.07206v1. Inputs use 256-bit `BigFloat`, `c_Bose=5`, reduced
`M_Pl=2.435e18 GeV`, and the three frozen timescales.

The named regression input is `M_BH=10 M_sun`, `mu=4.3e-12 eV`, mode `|211>`,
and signed `lambda_iiii=-1e-73`. The sign remains in provenance while
`abs(lambda_iiii)` enters `f_pert=sqrt(mu^2/abs(lambda_iiii))` and the
bosenova occupation. It gives `f_pert=1.3597793938724e16 GeV`,
`N_Bose=7.48743262348655e76`, and `N_max=8.34731776078895e76`.

## Corrective topology audit

The frozen numerical-method manifest specifies bisection. The source-formula
counterexample recorded in S1 showed that a negative residual at spin 1 does
not imply that the residual is negative throughout `[0,1]`. The original S4
results based on that endpoint assumption are invalidated. The immutable
manifest is retained; the separately hashed addendum specifies 257/513/1025
uniform scan levels, golden-section local-extremum refinement to spin width
`1e-40` (maximum 240 iterations), root bisection to width `1e-40` (maximum
140 iterations), and a `1e-40` residual near-tangent margin. An extremum
inside that margin is unavailable. Root/interval counts, topology status, and
endpoints must agree across levels within `1e-30`. Scan agreement is a
stability check, not proof that arbitrarily narrow features were found.

At `tau_BH=1e10 yr`, the per-mode transitions are:

| Mode | Detected efficient interval | Topology |
| --- | ---: | --- |
| `|211>` | [0.9052031016575758205, 1] | single onset to extremal spin |
| `|322>` | [0.5804534245375564644, 1] | single onset to extremal spin |
| `|433>` | [0.4089594027217139398, 1] | single onset to extremal spin |
| `|544>` | [0.3190532611839051358, 0.9717335435535871624] | bounded interval; two crossings |
| `|655>` | none resolved | no positive interval resolved |

The union of the five modes is `[0.3190532611839051358, 1]`: the `|544>`
interval overlaps the `|433>` interval. The union onset is set by `|544>` and
is `0.3190532611839051358`, lower than the `|211>` onset by
`0.5861498404736706848`. The `|544>` per-mode result is not itself a
single-boundary exclusion contour.

For `|211>`, the Eq. (26) onsets at the other frozen timescales remain:

| `tau_BH` (yr) | Onset spin |
| ---: | ---: |
| `4.5e7` | 0.9052031016679705996302132257 |
| `4.5e6` | 0.9052031017619465031581628351 |

## Low-alpha and near-tangent regressions

At `M_BH=1 M_sun`, `mu=4.3e-12 eV`, `lambda_iiii=-1e-73`,
`tau_BH=1e10 yr`, and mode `|211>`, the Eq. (26) residual is
`+2.5238411056e10` at spin 0.5 and `-173.56198036` at spin 1. The resolved
efficient interval is `[0.1281665047765220284, 0.9999999989528558772]`.
The lower crossing is an onset; the residual at extremal spin does not
classify spins above its upper crossing.

The near-tangent stress case at `M_BH=1 M_sun`, the same `mu`, age, and
`|211>` mode, uses
`lambda_iiii=-1.7180957110518061656012412304276707960189280606045e-65`.
It resolves a positive Eq. (26) maximum `1.5935038084870699091e-27` at spin
`0.6257692360535021388397666715`, with roots
`0.6257692360535008233366463567` and
`0.6257692360535034543428869862`. The bounded interval is retained.

The governing 2020 Eq. (14) prints `(m omega_nlm-mu_0)^2`, while the 2018
Eq. (15) prints `(mu_ax-m Omega_H)^2`. The frozen contract names the 2020
formula, which is retained. The difference and the 2020 prose description of
a single spin bound remain unresolved. No equivalence to the published Fig. 3
single-curve interpretation or Appendix-B contour is claimed where the
computed mode union has multiple spin boundaries.

Regression tests verify positive residual below and above a detected onset,
the second high-spin crossing in the low-alpha case, the near-tangent peaks,
three-resolution stability, and fail-closed behavior at a residual extremum
inside the near-tangent margin or when topology changes across scan levels.
Equality in the bosenova efficiency decision remains not-efficient.
Off-diagonal quartics and cubic interactions remain omitted as required by
this reference model.

This is a named formula check, not a population probability or a source-wide
constraint. Source-wide probability remains unavailable because the frozen
Appendix-B data do not define a complete Gaussian-sigma set.
