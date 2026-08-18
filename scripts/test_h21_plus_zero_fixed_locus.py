"""Regression test for the h^{2,1}_+(X,I)=0 fixed-locus Euler-characteristic
check in ``reproduce_fuzzy_axions_h11_4.py``.

Context: the h11=4 population reproduction originally gated its
``h11_minus_zero_h21_plus_zero`` population count (paper target: 267) on
``_frozen_conifold_diagnostic`` (a separate orientifold-background
smoothness condition, Moritz eqs. 4.48-4.50), giving 285 -- 18 too many,
with no way to identify which 18 were spurious. The actual criterion named
by that population -- h^{2,1}_+(X,I)=0, Moritz eq. (4.51) -- was never
independently checked; the code assumed it held automatically for any
favorable trilayer polytope. It does not: fewer than half of the 285
frozen-conifold-accepted classes in the real h11=4 population satisfy the
Hodge identity exactly.

``_h21_plus_zero_diagnostic`` (and the ``_fixed_locus_components`` /
``_fixed_locus_euler_characteristic`` helpers it calls) implement eq.
(4.51) directly by enumerating the irreducible fixed-locus components
F_I(sigma) (Moritz eqs. 4.34-4.50) and computing chi(F_I). This test
reconstructs the paper's own explicit, independent worked example -- the
h11=2 polytope of Sec. 4.2.1, eq. (4.2) -- and checks that the computed
Hodge splitting matches the paper's own stated
(h^{1,1}_+,h^{1,1}_-,h^{2,1}_+,h^{2,1}_-) = (2, 0, 0, 132) exactly, with no
tuning: this is the only paper-supplied ground truth available for this
identity, independent of anything computed for the h11=4 population.

Applying the same diagnostic across the real h11=4 population (415
trilayer FRST classes) reproduces the paper's 267-class population target
exactly; that full-population run is expensive (CYTools over the full KS
mirror) and is not repeated here, see
validation/fuzzy_axions_2412_12012_h21_plus_fixed_locus_20260818.md for the
recorded evidence.

Requires the local CYTools conda env, same as
``test_inherited_orientifold_candidates.py``.
"""

import unittest

from cytools import Polytope

from reproduce_fuzzy_axions_h11_4 import (
    _frst_classes,
    _h21_plus_zero_diagnostic,
    _trilayer_candidate,
)

# Moritz eq. (4.2): the 4D reflexive polytope points, given as columns of
# the stated matrix.
H11_2_POINTS = [
    [1, 0, 0, 0],
    [-1, 3, -2, -1],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [-2, -1, 0, 0],
    [0, 1, 0, 0],
]


class H21PlusZeroFixedLocusTest(unittest.TestCase):
    def test_h11_2_worked_example_matches_paper_hodge_splitting(self):
        poly = Polytope(H11_2_POINTS)

        # Sanity: confirm this is the paper's own polytope before trusting
        # anything downstream (h11=2, h21=132, favorable, 6 vertices, all
        # stated explicitly in Sec. 4.2.1).
        self.assertTrue(poly.is_reflexive())
        self.assertEqual(poly.h11(lattice="N"), 2)
        self.assertEqual(poly.h21(lattice="N"), 132)
        self.assertTrue(poly.is_favorable(lattice="N"))
        self.assertEqual(len(poly.vertices()), 6)

        trilayer = _trilayer_candidate(poly)
        self.assertIsNotNone(trilayer, "expected the paper's own example to be trilayer")

        _, classes = _frst_classes(poly)
        self.assertEqual(len(classes), 1, "expected a single FRST class for this example")

        result = _h21_plus_zero_diagnostic(poly, classes[0], trilayer["p0"])

        self.assertEqual(result["status"], "h21_plus_zero")
        # h^{1,1}_-=0 is built into the L=I reduction of eq. (4.51); the
        # paper states (h^{1,1}_+,h^{1,1}_-,h^{2,1}_+,h^{2,1}_-)=(2,0,0,132).
        self.assertAlmostEqual(result["h21_plus"], 0.0, places=6)
        self.assertAlmostEqual(result["h21_minus"], 132.0, places=6)


if __name__ == "__main__":
    unittest.main()
