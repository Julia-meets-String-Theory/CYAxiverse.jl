# S1 higher-l material effect

This run uses method identity `REFERENCE_2021_ANALYTIC`, the frozen source and
numerical method manifests, Eq. (9) and Eqs. (12)-(14) of arXiv:2009.07206v1,
256-bit `BigFloat`, direct bisection on spin `[0,1]`, tolerance `1e-40`, and a
maximum of 140 iterations. The handoff SHA-256 is
`39f5a823a67edca0ffc0a2b4221ba68901e6f2438efb5e6bbbe3ae2d8ceef170`.

Inputs: `M_BH=10 M_sun`, `mu=4.3e-12 eV`, `tau=1e10 yr`,
`Delta_a_star=0.1`; frozen nodeless modes `n_r=0`, `m=l`, `l=1..5`.

| Mode | Critical spin |
| --- | ---: |
| `|211>` | 0.9052031016575710060319535299 |
| `|322>` | 0.5804534248873714874151312491 |
| `|433>` | 0.4089697568655615231970547385 |
| `|544>` | no root on `[0,1]` |
| `|655>` | no root on `[0,1]` |

The five-mode union boundary is the smallest present threshold, `0.4089697569`,
set by `|433>`. Including higher-l modes moves the threshold down by
`0.4962333448` from the `|211>`-only value. Roots that are absent mean the
growth condition remains below threshold through spin 1 on this specified
input; they are not silently omitted from the per-mode result.

This is an equation-level consistency case, not a numerical comparison against
Fig. 3. A source-grounded digitized target point set is still unavailable, so
no published-plot agreement is claimed.
