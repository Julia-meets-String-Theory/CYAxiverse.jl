"""CYAX-0232 R-001 / R-005 F5 row (T021): the F5 mass evaluator at c = 1.

Portable: needs NumPy only, so it runs in CI where CYTools is absent.
"""
import unittest

import numpy as np

import fuzzy_axion_mass_evaluator as evaluator


class FuzzyAxionMassEvaluatorNormalizationTests(unittest.TestCase):
    """CYAX-0232 R-005 F5 row and T021: the optimizer mass evaluator at c = 1."""

    # The R-004 non-diagonal fixture (specs/0232-kinetic-normalization/plan.md).
    TAU = np.asarray([1.0, 20.0, 30.0])
    KINV = np.asarray([[3.0, 1.0], [1.0, 2.0]])
    Q = np.asarray([[1, 1, 0], [0, 1, 1]])
    GS = 0.5029733
    # Julia model-stage pins (test/kinetic_normalization.jl), with W = W0 = 1.
    JULIA_C1 = (24.258982913, -1.141289065)
    JULIA_C_HALF = (24.409497911, -0.990774067)

    def _mass(self, axion_index, inverse_metric):
        values = {
            "prime_divisor_volumes": self.TAU,
            "inverse_metric": inverse_metric,
            "cy_volume": 50.0,
        }
        return evaluator._fuzzy_axion_mass_log10(
            values, self.Q, axion_index=axion_index, gs=self.GS, w0=1.0
        )

    def test_matches_julia_model_stage_at_c1(self):
        # The evaluator's superpotential adds sum(exp(-2 pi tau)) to W0;
        # m^2 is linear in |W|, so its masses sit 0.5 log10|W| above the
        # Julia pins, which use W = W0.
        shift = 0.5 * np.log10(1.0 + np.exp(-2.0 * np.pi * self.TAU).sum())
        for index, pin in enumerate(self.JULIA_C1, start=1):
            self.assertAlmostEqual(self._mass(index, self.KINV), pin + shift, delta=1e-8)

    def test_c_half_comparison_scales_masses_by_inverse_sqrt2(self):
        # R-005: 2 Kinv in place of Kinv is the c = 1/2 evaluation.
        shift = 0.5 * np.log10(1.0 + np.exp(-2.0 * np.pi * self.TAU).sum())
        for index, pin in enumerate(self.JULIA_C_HALF, start=1):
            c1 = self._mass(index, self.KINV)
            c_half = self._mass(index, 2.0 * self.KINV)
            self.assertAlmostEqual(c1 - c_half, -0.5 * np.log10(2.0), delta=1e-10)
            self.assertAlmostEqual(c_half, pin + shift, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
