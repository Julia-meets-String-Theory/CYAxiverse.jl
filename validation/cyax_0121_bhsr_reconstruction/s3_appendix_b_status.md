# S3 Appendix-B formula status

The formula-level implementation is available under route identity
`REFERENCE_2018_APPENDIX_B`, source arXiv:1805.02016v2, Appendix B
Eqs. (95)-(98). The bounded artifact model ID is
`S3-FORMULA-AND-SOURCE-WIDE-STATUS`. It is bound to source-mode manifest
`a891735d45dd80e57689917416577203e43a68ec0c7e96b23897cb7c306725ed`, method
manifest `7b9c1cf9e4787372801125d7c195288d36f3cb9e094b0a2ec179493c8e8392d5`,
and observational-data manifest
`9da9f70efc6e654e3c38d963bf6a9e5ad9bec791a56ff2b43025bf019a7a87db`.

The implementation covers the zero-covariance one-dimensional projection for
both `y=f(x)` and inverse `x=g(y)` branches, nearest inverse-branch derivative,
Gaussian interval between two inverse branches, source finite-difference
rules, conservative total-contour cusp projection, Eq. (96) products, and
Eq. (95) `P_ex=1-P_allowed`. The probability tree multiplies probabilities
over the complete declared black-hole ensemble for each axion, then over the
explicit complete set of source-relevant axions. It fails closed on missing,
duplicate, unresolved, or unsupported rows, ambiguous inverse topology, and
contour or derivative evaluation outside the supplied support. It does not
insert neutral probabilities or extrapolate. The exclusion threshold is the
strict `P_ex > 0.9545` criterion.

## Source ensemble and contour support

The 24 black-hole identities in the frozen observational manifest are the
2018 Table-I ensemble used by the Appendix-B likelihood audit. The stellar
contour grid is a separate source object: Fig. 3 of arXiv:2009.07206v1, with
fixed scalar mass `4.3e-12 eV`, covers black-hole masses from `0.1` to `100`
solar masses. The stellar grid does not cover the entire 2018 Table-I
ensemble. These nine Table-I rows lie outside its mass support:

`Mrk 335`, `Fairall 9`, `Mrk 79`, `NGC 3783`, `MCG-6-30-15`, `NGC 7469`,
`Ark 120`, `Mrk 110`, and `NGC 4051`.

The implementation reports these as unsupported and does not evaluate the
2021 stellar contour at their masses. It does not present the 2021 grid as the
2018 source's SMBH contour. The exact 2021 Fig. 3 numeric point set is not
available as a digitized source table in this reconstruction, so no
published-contour comparison is claimed.

## Why the source-wide probability is unavailable

The frozen observational manifest identifies all 24 2018 rows and records
one-sided spin bounds, asymmetric intervals, and mixed 90%/95% confidence
levels. Appendix B Eqs. (97)-(98) require Gaussian standard deviations; the
source does not define conversions for these uncertainty forms or identify a
complete subset with all required sigmas. Six spin rows are censored:
Cygnus X-1, GRS 1915+105, NGC 3783, MCG-6-30-15, Mrk 110, and NGC 4051. In
addition, the separate 2021 stellar contour support omits the nine Table-I
rows named above. No source-wide likelihood or subset probability is reported.

These limits are independent: resolving a sigma convention would not supply
the missing contour support, and the stellar contour must not be extrapolated
to fill those rows. The source locators are arXiv:1805.02016v2 HTML lines
597-619 for Appendix B and arXiv:2009.07206v1 Fig. 3 and its source data
description. Manifest hashes are recorded in `SHA256SUMS`.
