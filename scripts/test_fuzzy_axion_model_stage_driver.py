"""Integration test for the priority-4 Python-to-Julia model-stage bridge.

``scripts/reproduce_fuzzy_axions_h11_4.py``'s ``--model-stage`` flag hands
per-(X, FRST) Kähler-point export records to
``scripts/fuzzy_axion_model_stage_driver.jl`` through an HDF5 file (Priority
1's ``Q``, ``tau``, ``cy_volume``, ``inverse_metric``), because HDF5.jl is
already a hard dependency and ``CYAxiverse``'s ``Project.toml`` has no
JSON-parsing package.

The physics itself (``enumerate_fuzzy_axion_models`` and everything it calls)
is already unit-tested in ``test/runtests.jl``'s "Fuzzy-axion model-stage
evaluator" testset. What is new and untested at that layer is the *bridge*:
HDF5.jl reads an array written by ``h5py`` with its dimensions reversed
relative to the NumPy shape it was written with (confirmed empirically before
``fuzzy_axion_model_stage_driver.jl`` was written -- a Python ``(3, 4)`` array
round-trips as Julia ``size (4, 3)``, the literal transpose, not merely a
relabelling). This test isolates that specific risk: it runs the exact same
non-square, non-symmetric synthetic ``(Q, tau, cy_volume, inverse_metric)``
record through two independent paths --

1. the production path: ``_write_model_stage_input`` (HDF5) then
   ``fuzzy_axion_model_stage_driver.jl`` (the same driver
   ``reproduce_fuzzy_axions_h11_4.py`` invokes), and
2. a direct Julia call with the identical arrays passed as literals, with no
   HDF5 round-trip at all --

and asserts they produce the exact same model count and the same per-model
``(axion_index, qcd_divisor_index, lambda)`` records. A transpose bug in the
bridge would make these two paths disagree (or crash with a
``DimensionMismatch``, since ``Q`` here is deliberately non-square (h11=2,
N=6) so a missing/extra transpose cannot silently pass a shape check).

Requires the local CYTools conda env (``julia`` on PATH, ``h5py`` importable)
per this initiative's established environment gotchas -- same as
``test_inherited_orientifold_candidates.py``.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from reproduce_fuzzy_axions_h11_4 import _write_model_stage_input

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPO_ROOT / "scripts" / "fuzzy_axion_model_stage_driver.jl"

# Deliberately non-square (h11=2, N=6) and non-symmetric so a missing or
# backwards transpose in the bridge cannot silently pass a shape check or
# happen to produce the same values by accident.
SYNTHETIC_Q = [
    [1, 0, 1, 0, 2, 1],
    [0, 1, 0, 1, 1, 2],
]
SYNTHETIC_TAU = [1.5, 2.0, 1.2, 1.8, 3.0, 2.5]
SYNTHETIC_CY_VOLUME = 5.0
SYNTHETIC_INVERSE_METRIC = [
    [2.0, 0.3],
    [0.3, 1.5],
]
GS = 0.5
W0_REAL = 1.0
W0_IMAG = 0.0


class ModelStageDriverBridgeTest(unittest.TestCase):
    def test_hdf5_bridge_matches_direct_julia_call(self):
        record = {
            "glsm_charge_matrix": SYNTHETIC_Q,
            "prime_divisor_volumes": SYNTHETIC_TAU,
            "cy_volume": SYNTHETIC_CY_VOLUME,
            "inverse_metric": SYNTHETIC_INVERSE_METRIC,
        }

        with tempfile.TemporaryDirectory(prefix="fuzzy_axion_bridge_test_") as workdir_name:
            workdir = Path(workdir_name)
            input_path = workdir / "input.h5"
            bridged_output_path = workdir / "bridged_output.h5"

            _write_model_stage_input([record], input_path)
            subprocess.run(
                [
                    "julia",
                    f"--project={REPO_ROOT}",
                    str(DRIVER),
                    str(input_path),
                    str(bridged_output_path),
                    str(GS),
                    str(W0_REAL),
                    str(W0_IMAG),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            # No JSON package is available in this project (see the bridge's
            # own docstring for why HDF5 was chosen instead) -- print a
            # simple, unambiguous delimited line instead of round-tripping
            # through a serialization format this test is specifically
            # trying to avoid depending on.
            direct_julia_program = f"""
            using CYAxiverse
            Q = {_julia_int_matrix_literal(SYNTHETIC_Q)}
            tau = {_julia_float_vector_literal(SYNTHETIC_TAU)}
            cy_volume = {SYNTHETIC_CY_VOLUME}
            inverse_metric = {_julia_float_matrix_literal(SYNTHETIC_INVERSE_METRIC)}
            gs = {GS}
            w0 = Complex{{Float64}}({W0_REAL}, {W0_IMAG})
            prefactor_P = CYAxiverse.paper_benchmarks.fuzzy_axion_prefactor_P(gs)
            instanton_terms = exp.(-2π .* tau)
            superpotential = CYAxiverse.paper_benchmarks.fuzzy_axion_flux_superpotential(
                w0, instanton_terms)
            kahler_pot = CYAxiverse.paper_benchmarks.fuzzy_axion_kahler_potential(cy_volume)
            gravitino_mass = CYAxiverse.paper_benchmarks.fuzzy_axion_gravitino_mass(
                prefactor_P, kahler_pot, superpotential; mplanck_ev=1.0)
            models = CYAxiverse.paper_benchmarks.enumerate_fuzzy_axion_models(
                Q, tau, cy_volume, prefactor_P, gravitino_mass, inverse_metric)
            for model in models
                println("MODEL ", model.axion_index, " ", model.qcd_divisor_index, " ", model.lambda)
            end
            println("TOTAL ", length(models))
            """
            direct_result = subprocess.run(
                ["julia", f"--project={REPO_ROOT}", "-e", direct_julia_program],
                check=True,
                capture_output=True,
                text=True,
            )

            # Read the bridged output while `workdir` still exists -- it is
            # deleted as soon as this `with` block exits.
            with h5py.File(bridged_output_path, "r") as file:
                bridged_total = int(file["total_model_count"][()])
                bridged_axion_index = np.asarray(file["model_axion_index"])
                bridged_qcd_divisor_index = np.asarray(file["model_qcd_divisor_index"])
                bridged_lambda = np.asarray(file["model_lambda"])

        direct_lines = [line for line in direct_result.stdout.splitlines() if line.strip()]
        direct_models = sorted(
            tuple(float(part) for part in line.split()[1:])
            for line in direct_lines
            if line.startswith("MODEL")
        )
        direct_total = int(next(line.split()[1] for line in direct_lines if line.startswith("TOTAL")))

        bridged_models = sorted(
            (
                float(bridged_axion_index[i]),
                float(bridged_qcd_divisor_index[i]),
                float(bridged_lambda[i]),
            )
            for i in range(len(bridged_axion_index))
        )

        self.assertEqual(bridged_total, direct_total)
        self.assertGreater(bridged_total, 0, "synthetic fixture should produce at least one model")
        self.assertEqual(len(bridged_models), len(direct_models))
        for bridged_model, direct_model in zip(bridged_models, direct_models):
            self.assertEqual(bridged_model[0], direct_model[0])  # axion_index
            self.assertEqual(bridged_model[1], direct_model[1])  # qcd_divisor_index
            self.assertAlmostEqual(bridged_model[2], direct_model[2], places=8)  # lambda

    def test_write_model_stage_input_hdf5_shapes(self):
        record = {
            "glsm_charge_matrix": SYNTHETIC_Q,
            "prime_divisor_volumes": SYNTHETIC_TAU,
            "cy_volume": SYNTHETIC_CY_VOLUME,
            "inverse_metric": SYNTHETIC_INVERSE_METRIC,
        }

        with tempfile.TemporaryDirectory(prefix="fuzzy_axion_bridge_test_") as workdir_name:
            input_path = Path(workdir_name) / "input.h5"
            _write_model_stage_input([record], input_path)
            with h5py.File(input_path, "r") as file:
                self.assertEqual(int(file["record_count"][()]), 1)
                group = file["records/0"]
                self.assertEqual(np.asarray(group["Q"]).shape, (2, 6))
                self.assertEqual(np.asarray(group["tau"]).shape, (6,))
                self.assertEqual(np.asarray(group["inverse_metric"]).shape, (2, 2))
                self.assertEqual(float(group["cy_volume"][()]), SYNTHETIC_CY_VOLUME)


def _julia_int_matrix_literal(rows):
    row_strings = [" ".join(str(int(v)) for v in row) for row in rows]
    return "[" + "; ".join(row_strings) + "]"


def _julia_float_matrix_literal(rows):
    row_strings = [" ".join(repr(float(v)) for v in row) for row in rows]
    return "[" + "; ".join(row_strings) + "]"


def _julia_float_vector_literal(values):
    return "[" + ", ".join(repr(float(v)) for v in values) + "]"


if __name__ == "__main__":
    unittest.main()
