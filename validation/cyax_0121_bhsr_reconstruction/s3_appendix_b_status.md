# S3 Appendix-B formula status

The formula-level implementation is available under route identity
`REFERENCE_2018_APPENDIX_B`, source arXiv:1805.02016v2, Appendix B
Eqs. (95)-(98). The bounded artifact model ID is
`S3-FORMULA-AND-SOURCE-WIDE-STATUS`. It is bound to source-mode manifest
`a891735d45dd80e57689917416577203e43a68ec0c7e96b23897cb7c306725ed`, method
manifest `7b9c1cf9e4787372801125d7c195288d36f3cb9e094b0a2ec179493c8e8392d5`,
and observational-data manifest
`9da9f70efc6e654e3c38d963bf6a9e5ad9bec791a56ff2b43025bf019a7a87db`.

The formula routines cover the zero-covariance one-dimensional projection for
both `y=f(x)` and inverse `x=g(y)` branches, nearest inverse-branch derivative,
Gaussian interval between two inverse branches, source finite-difference
rules, conservative total-contour cusp projection, Eq. (96) products, and
Eq. (95) `P_ex=1-P_allowed`. Numerical outputs from caller-supplied contours
and sigmas are labeled diagnostics: they carry a diagnostic probability field,
while the authoritative probability field remains missing. Caller-supplied
mode labels or a `source_backed` flag do not establish provenance. The sigma
gate distinguishes a structurally complete diagnostic input from the frozen
manifest's source authority; it reports `status=unavailable` even when a
synthetic input fixture has complete-looking rows.

The probability tree still checks the complete declared black-hole ensemble
for each axion and the explicit complete set of source-relevant axions. It
fails closed on missing, duplicate, unresolved, or unsupported rows, ambiguous
inverse topology, contour or derivative evaluation outside supplied support,
and the absent source-defined Gaussian sigma set. Under the current frozen
manifest, even fully populated caller-asserted rows return missing geometry
probabilities and no threshold result. The strict `P_ex > 0.9545` criterion is
reported only when an authoritative source sigma set exists; that condition is
not met here.

The source-bound 2021 contour path also checks the full spin-region topology.
It returns unavailable if its mode union has a bounded interval, disjoint
intervals, or scan-resolution instability. A first crossing alone does not
establish that all larger spins are excluded. The exact 2020 Eq. (14) rate
factor differs from the 2018 Eq. (15) factor, and the 2020 prose describes a
single spin bound. That source-interpretation discrepancy remains unresolved;
the 2021 contour is not treated as an Appendix-B contour when its full topology
cannot be represented by one onset boundary.

Every Regge row/grid and dependent likelihood result also carries the
content-addressed corrective topology-method addendum SHA-256
`909d283bd19af956ee2881537ee90dce07c0e1c7976717ac27cc7ca6402f7d39`, in
addition to the unchanged frozen numerical-method manifest hash. The addendum
records the 257/513/1025 scan, refined extrema and roots, the `1e-40`
near-tangent margin, endpoint stability rule, interval union, and fail-closed
contour conditions. The original manifest remains byte-identical.

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

Result identities distinguish formula calculations from the Fig. 3 target.
Scalar topology and per-mass rows carry generic Eq. (12)-(14) target IDs for
the two listed stellar ages. A contour grid receives a Fig. 3 target ID only
when it uses the full 1201-point `0.1`-to-`100 M_sun` mass range, fixed
`mu=4.3e-12 eV`, the complete frozen mode family, and the frozen numerical
settings. Other grids carry the generic equation target ID and remain
diagnostic. Grid requests outside the Fig. 3 mass range fail closed. Likelihood
metadata carries the contour target ID and its `tau_years`; caller-supplied
closures have missing target and age metadata.

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
