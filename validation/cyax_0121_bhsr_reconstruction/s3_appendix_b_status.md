# S3 Appendix-B formula status

Route identity: `REFERENCE_2018_APPENDIX_B`, source arXiv:1805.02016v2,
Appendix B Eqs. (95)-(98). The implementation provides the zero-covariance
one-dimensional projection for both `y=f(x)` and inverse `x=g(y)` branches,
the nearest inverse-branch derivative, Gaussian interval between inverse
branches, source finite-difference rules, conservative total-contour cusp
projection, Eq. (96) products, and Eq. (95) `P_ex=1-P_allowed`.

The probability-tree helper preserves a product over black holes for each
axion, followed by a product over source-relevant axions. Any missing input
propagates `missing`; no neutral probabilities are inserted. The threshold
check is the strict `P_ex > 0.9545` criterion.

The source-wide Table-I result remains unavailable. The frozen data manifest
identifies 24 rows and reports one-sided spin bounds, asymmetric intervals,
and mixed 90%/95% confidence levels. The source does not define conversions
from those cases to the Gaussian standard deviations required by Eqs. (97) and
(98), and it does not identify a complete subset with all required sigmas.
The six censored spin rows are Cygnus X-1, GRS 1915+105, NGC 3783,
MCG-6-30-15, Mrk 110, and NGC 4051. No source-wide or subset probability is
reported here.

This status cites source lines 597-619 of the versioned HTML source and the
unchanged `observational_data_manifest.json` hash recorded in `SHA256SUMS`.
