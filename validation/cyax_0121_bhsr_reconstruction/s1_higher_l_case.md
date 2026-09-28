# S1 higher-l material effect and topology correction

This run uses model ID `analytic-10-solar-mass` and method identity
`REFERENCE_2021_ANALYTIC`, with the frozen source-mode manifest
`a891735d45dd80e57689917416577203e43a68ec0c7e96b23897cb7c306725ed` and
numerical method manifest
`7b9c1cf9e4787372801125d7c195288d36f3cb9e094b0a2ec179493c8e8392d5`.
The handoff SHA-256 is
`39f5a823a67edca0ffc0a2b4221ba68901e6f2438efb5e6bbbe3ae2d8ceef170`.
Inputs are `M_BH=10 M_sun`, `mu=4.3e-12 eV`, `tau=1e10 yr`,
`Delta_a_star=0.1`, 256-bit `BigFloat`, and the frozen nodeless modes
`n_r=0`, `m=l`, `l=1..5`.

## Corrective topology audit

The frozen numerical-method manifest specifies bisection. A source-formula
counterexample showed that endpoint sign alone does not determine whether the
residual is positive somewhere inside `[0,1]`. The original S1/S4 results based
on that monotonicity assumption are invalidated. This addendum leaves all three
frozen manifests byte-identical and adds a topology audit: uniform scans at
257, 513, and 1025 points; refinement of sampled local extrema; bisection of
all detected sign crossings; and comparison of crossing counts and endpoints
between resolutions. A topology is available only when all three resolutions
agree within `1e-30`; otherwise the result is
`:unavailable_topology_resolution`. The scan is a corrective audit beyond the
frozen bisection recipe. It is a finite-resolution method and does not prove
that a narrower feature missed by all three scans is absent.

The scalar `critical_spin` result is only the first efficiency onset. Each row
also records all detected crossings, efficient intervals, their mode union,
and whether that union can be represented by one onset extending to spin 1.
Bounded or disjoint intervals are not treated as an above-threshold exclusion
contour.

## Five-mode material effect at 10 solar masses

| Mode | Detected efficient interval | Topology |
| --- | ---: | --- |
| `|211>` | [0.9052031016575710060, 1] | single onset to extremal spin |
| `|322>` | [0.5804534248873714874, 1] | single onset to extremal spin |
| `|433>` | [0.4089697568655615232, 1] | single onset to extremal spin |
| `|544>` | none detected | no positive interval at all three resolutions |
| `|655>` | none detected | no positive interval at all three resolutions |

The five-mode onset is `0.4089697568655615232`, set by `|433>`. Adding
higher-l modes lowers the onset by `0.4962333447920094828` relative to
`|211>` alone. Here the union remains one interval extending to spin 1.

## Low-alpha regression and source interpretation

At `M_BH=1 M_sun` with the same `mu`, age, and `|211>` mode, the printed
2020 Eq. (14) residual is `+2.8136713513e7` at spin 0.5 and
`-172.50682646` at spin 1. The resolved interval is
`[0.1281682643098299287, 0.9999990663471144157]`, with two crossings.
The lower crossing is an onset; spins above the upper crossing are not in the
efficient interval.

The governing 2020 Eq. (14) prints the rate factor
`[k^2(1-a_*^2)+4 r_+^2(m omega_nlm-mu_0)^2]`. The 2018 Eq. (15) prints a
different factor involving `(mu_ax-m Omega_H)^2`. The reconstruction retains
the exact 2020 Eq. (14) expression named in the frozen manifest. The
cross-source discrepancy and the 2020 prose description of a single spin
bound remain unresolved. This bounded interval does not establish equivalence
to the published Fig. 3 single-curve interpretation or an Appendix-B contour.

No digitized numerical target points from Fig. 3 are present in the frozen
source manifest. This is formula-level evidence only; no published-plot
agreement is claimed.
