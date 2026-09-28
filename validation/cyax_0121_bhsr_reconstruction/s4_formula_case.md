# S4 bosenova formula check

Route identity: `REFERENCE_2021_BOSENOVA`. This uses the frozen mode and method
manifests, source Eq. (9), Eq. (13) of arXiv:2103.06812v2, and the strict Eq.
(26) criterion of arXiv:2009.07206v1. It uses 256-bit `BigFloat`,
`c_Bose=5`, reduced `M_Pl=2.435e18 GeV`, and the three frozen timescales.

The named regression input is `M_BH=10 M_sun`, `mu=4.3e-12 eV`, mode `|211>`,
and signed `lambda_iiii=-1e-73`. The sign remains in provenance while
`abs(lambda_iiii)` enters `f_pert=sqrt(mu^2/abs(lambda_iiii))` and the
bosenova occupation. It gives `f_pert=1.3597793938724e16 GeV`,
`N_Bose=7.48743262348655e76`, and `N_max=8.34731776078895e76`.

At `tau_BH=1e10 yr`, the mode envelope gives these per-mode transitions:

| Mode | Eq. (26) transition spin |
| --- | ---: |
| `|211>` | 0.9052031016575758205322773675 |
| `|322>` | 0.5804534245375564644237927854 |
| `|433>` | 0.4089594027217139398078603996 |
| `|544>` | no root on `[0,1]` |
| `|655>` | no root on `[0,1]` |

The envelope is set by `|433>` and shifts down by `0.4962436989` from
`|211>` alone. The |211> transition roots for the other frozen timescales are:

| `tau_BH` (yr) | Eq. (26) transition spin |
| ---: | ---: |
| `4.5e7` | 0.9052031016679705996302132257 |
| `4.5e6` | 0.9052031017619465031581628351 |

Regression tests check negative residual below each 1e10-year transition and
positive residual above it. The decision is strict: residual equality returns
not-efficient. A low iteration limit fails closed. Off-diagonal quartics and
cubic interactions remain omitted as required by this reference model.

This is a named formula check, not a population probability or a source-wide
constraint. The source-wide probability remains unavailable while authoritative
mass/spin sigma conventions are missing for the Appendix-B data.
