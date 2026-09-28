# S2 continued-fraction source ambiguity

Status: `SCIENTIFIC_CONTRACT_AMBIGUITY` for the S2 route.

Route identity: `CF_2018_VALIDATION`. The targeted numerical validation is
blocked before implementation by the exact versioned source equations in
arXiv:1805.02016v2 Appendix A.4, Eqs. (A29)-(A30) (PDF p. 22; HTML Eqs. (79)-
(80)). Eq. (A29) states
`q = ±sqrt(mu^2 - omega^2)`. Eq. (A30) prints
`chi = (mu - 2 omega^2)/q`.

In the radial ansatz, `chi` is an exponent and must be dimensionless. The
printed numerator combines `mu` (mass dimension one) and `omega^2` (mass
dimension two), while `q` has mass dimension one. The expression is therefore
dimensionally inconsistent. Replacing `mu` with `mu^2` would repair dimensions,
but that correction is not in the frozen source contract and would be an
unsupported inference. No CF frequency, rate, or convergence claim is emitted.

The frozen target remains `alpha=0.1`, `a*=0.9`, `|211>`, with the specified
precision/order/truncation ladder. The exact source equation must be clarified
in a reviewed successor before a numerical result can be assigned to
`CF_2018_VALIDATION`. This block does not prevent the bounded S1, S3 formula,
or S4 formula work.
