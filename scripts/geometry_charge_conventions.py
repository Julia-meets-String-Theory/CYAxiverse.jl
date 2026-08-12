"""Canonical charge-row conventions shared by the CYTools generators.

The author workflow treats repeated charge rows as redundant generators before
constructing the instanton potential.  Apply that convention at the raw
effective-cone boundary, before pairwise differences and coefficients are
formed.  This helper deliberately removes exact duplicate rows only; it does
not primitive-normalize nonprimitive lattice rays.
"""

import numpy as np


def canonicalize_unique_charge_rows(charges):
    """Return unique integral charge rows and removal provenance.

    CYTools returns lattice rays as numeric arrays.  A nonintegral ray is a
    malformed input for this potential convention and must fail before any
    Kähler-control or coefficient calculation can consume it.
    """

    rows = np.asarray(charges)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError("charge rows must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(rows)):
        raise ValueError("charge rows must be finite")
    rounded = np.rint(rows)
    if not np.allclose(rows, rounded, rtol=0.0, atol=1e-10):
        raise ValueError("charge rows must be integral lattice vectors")

    integral_rows = rounded.astype(np.int64)
    unique_rows = np.unique(integral_rows, axis=0)
    return unique_rows.astype(float), {
        "raw_count": int(integral_rows.shape[0]),
        "canonical_count": int(unique_rows.shape[0]),
        "duplicates_removed": int(integral_rows.shape[0] - unique_rows.shape[0]),
        "convention": "unique_effective_cone_rays",
    }
