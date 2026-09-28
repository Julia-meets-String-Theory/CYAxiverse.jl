# S2 source audit and continued-fraction validation

## Governing equation and authorized correction

Status: the CF implementation is source-grounded after the owner-authorized
correction to a printed source typo. The frozen target convergence gate remains
`UNAVAILABLE_REFINE_CONVERGENCE`.

Route identity: `CF_2018_VALIDATION`. Model ID for the frozen target:
`CF-2018-FROZEN-TARGET`. Frozen source-mode manifest SHA-256:
`a891735d45dd80e57689917416577203e43a68ec0c7e96b23897cb7c306725ed`.
Frozen numerical-method manifest SHA-256:
`7b9c1cf9e4787372801125d7c195288d36f3cb9e094b0a2ec179493c8e8392d5`.
The governing handoff SHA-256 is
`39f5a823a67edca0ffc0a2b4221ba68901e6f2438efb5e6bbbe3ae2d8ceef170`.

The reconstruction preserves the printed expression in arXiv:1805.02016v2,
Appendix A.4, Eq. (80):

`q = ±sqrt(mu^2 - omega^2)`,
`chi = (mu - 2 omega^2)/q`.

The second expression is dimensionally inconsistent: its numerator adds a
mass to a mass squared. The owner authorized the documented correction from
Dolan, arXiv:0705.2880v2, Sec. III, Eq. (34):

`chi = (mu^2 - 2 omega^2)/q`.

This agrees with Yoshino and Kodama, PTEP 2014, 043E02, Appendix A.3,
Eq. (A9), which writes `chi = M(2 omega^2 - mu^2)/k`; setting `q=-k` gives
the same expression in `M=1` units. The bound-state branch used here is
`q=-sqrt(mu^2-omega^2)` with `Re(q)<0`, as required by the decaying radial
solution. The source's printed form and the adopted correction are both
recorded here; the source manifest itself is unchanged.

The radial recurrence follows Dolan Eqs. (33)-(48), with its angular
separation eigenvalue computed from Dolan's angular equation, Eq. (29), using
`c^2=a^2(omega^2-mu^2)`. The tridiagonal angular basis is the same-parity
ladder `l, l+2, ...`, because the `cos^2(theta)` operator couples degrees
separated by two. Dolan's final CF condition is Eq. (48), consistent with
arXiv:1805.02016v2 Appendix A.4, Eq. (94). This route does not derive the
recurrence from the malformed 2018 intermediate radial equation.

## Bounded Appendix-A typo audit

| 2018 source locator | Finding | Classification and treatment |
| --- | --- | --- |
| Appendix A.2, Eq. (57) | The displayed Klein-Gordon mass term is `-mu_ax`; the action Eq. (1) and Dolan Eq. (25) require a squared mass. | Confirmed typographical omission of the square; this printed equation is not used to generate the CF coefficients. |
| Appendix A.2, Eq. (59) | The rendered angular equation has malformed operator/bracket placement. | Unusable as printed. The angular ODE is taken from Dolan Eq. (29), whose convention is cross-checked against Yoshino-Kodama Eq. (A2). |
| Appendix A.2, Eq. (60) | The displayed radial equation has missing or misplaced factors relative to the separated Kerr radial equation. | Unresolved transcription discrepancy in that intermediate equation. The implementation uses Dolan Eq. (28) and the explicit recurrence Eqs. (33)-(48), not a repair inferred from Eq. (60). |
| Appendix A.2, Eq. (61) | The expansion is written with an argument involving `a^2(mu^2-omega^2)`, while Dolan uses `c^2=a^2(omega^2-mu^2)` and Yoshino-Kodama use `-k^2 a^2`. | Sign convention is not explicit enough to determine whether it is absorbed into the expansion coefficients. The implementation bypasses this expansion and solves the angular ODE directly. |
| Appendix A.4, Eqs. (79)-(80) | Eq. (79) defines `q`; Eq. (80) prints `chi=(mu-2omega^2)/q`. | Eq. (80) is dimensionally inconsistent and is preserved as printed above. The owner-authorized correction is Dolan Eq. (34), independently consistent with Yoshino-Kodama Eq. (A9). |
| Appendix A.4, Eqs. (84), (92), (94) | Eq. (84) prints `c1/c3` without subscripts; Eq. (92)'s ratio is malformed; Eq. (94) gives the final CF condition. | Dolan Eqs. (38) and (43) confirm the `c_1` and `c_3` subscripts. Eq. (92) is not used; the final condition is evaluated from the recurrence and checked against Dolan Eq. (48). |

The source locators are arXiv:1805.02016v2 HTML Appendix A.2, lines 489-504,
and Appendix A.4, lines 553-595 (PDF p. 22). Dolan's recurrence and angular
equation are in arXiv:0705.2880v2, Sec. III, Eqs. (28)-(48). The independent
convention check is Yoshino and Kodama, PTEP 2014, 043E02, Appendix A.2-A.3,
Eqs. (A1)-(A13).

## Numerical checks

The frozen target is `alpha=M*mu=0.1`, `a*=0.9`, mode `|211>`, with the
manifest-prescribed precision/order/angular ladders and root residual bound
`1e-24`. All roots on that ladder meet the residual bound and have `Re(q)<0`.
At fixed precision 256 bits and angular truncation 13, the order-128 to
order-256 change is `1.8484119119e-10` in `M*omega` and `6.9768447992e-2`
in `M*Im(omega)`. The method-manifest acceptance threshold is `1e-8`, so the
frozen target fails its growth-rate refinement gate. The precision and angular
refinement comparisons pass. The result is unavailable under the frozen gate;
it is not promoted by an extended-order diagnostic.

An additional diagnostic at orders 8192 and 16384, 256 bits, angular
truncation 17, and residual tolerance `1e-50` gives respectively
`M*Im(omega)=6.87515569769905e-12` and
`6.87515569769904e-12`, a relative change of `7.5654949e-16`. The final CF
residual is `1.7794e-74`, and `Re(q)<0`. The small-coupling Eq. (14) estimate
from arXiv:2009.07206v1 is `Gamma=4.85435282499119e-12` in `M=1` units; the
CF `Im(omega)` is 1.41628677 times this estimate. The source identifies
`Gamma` with `M*omega_I`, so this comparison uses `Im(omega)`, not
`2*Im(omega)`. The 41.6% difference remains an approximation or convention
discrepancy; it is not fitted away, and it does not change the frozen gate.

Independent rounded-alpha spot checks use Dolan, Phys. Rev. D 76, 084001
(2007), Table III, PDF p. 11. The same result appears as Table 1, lines
278-282, in arXiv:0705.2880v2 HTML. The table reports maximum growth rates;
these fixed-alpha calculations are comparisons with those rounded maxima,
not claims that the maximum search was reproduced.

| `a*` | fixed `M*mu` | CF `M*Im(omega)` | rounded table maximum | relative difference | order 1024 to 2048 change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.99 | 0.421 | 1.50432665350714e-7 | 1.50e-7 | 0.28844% | 6.36836e-14 |
| 0.98 | 0.393 | 1.11221129790675e-7 | 1.11e-7 | 0.19922% | 7.61016e-15 |

Both comparisons use orders 512, 1024, and 2048, angular truncations 9 and 17,
256-bit arithmetic, and residual tolerance `1e-24`. The angular 9 and 17
results agree at displayed precision, every root has `Re(q)<0`, and all
residuals pass the bound. These checks validate the recurrence against
independent published rates; they do not pass the frozen target ladder.
