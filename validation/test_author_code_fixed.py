#!/usr/bin/env python3
"""Regression test for the corrected actual-author comparison module."""

from pathlib import Path
import os
import sys
import types

import numpy as np


def install_import_stubs():
    """Provide only import-time dependencies used by the author module."""
    scipy = types.ModuleType("scipy")
    scipy.linalg = types.ModuleType("scipy.linalg")
    scipy.integrate = types.ModuleType("scipy.integrate")
    scipy.optimize = types.ModuleType("scipy.optimize")
    scipy.linalg.qr = lambda *args, **kwargs: None
    scipy.integrate.solve_ivp = lambda *args, **kwargs: None
    scipy.optimize.minimize = lambda *args, **kwargs: None
    sys.modules.update({
        "scipy": scipy,
        "scipy.linalg": scipy.linalg,
        "scipy.integrate": scipy.integrate,
        "scipy.optimize": scipy.optimize,
    })

    cytools = types.ModuleType("cytools")
    cytools.Polytope = object
    cytools.Cone = object
    cytools.read_polytopes = lambda *args, **kwargs: None
    cytools.fetch_polytopes = lambda *args, **kwargs: None
    sys.modules["cytools"] = cytools

    hsnf = types.ModuleType("hsnf")
    hsnf.smith_normal_form = lambda matrix: None
    sys.modules["hsnf"] = hsnf


def load_fixed_module():
    fixed_dir = Path(
        os.environ.get(
            "CYAXIVERSE_AUTHOR_FIXED_DIR",
            "/Users/vmehta/Documents/CYAxiverse/cyaxiverse/"
            "CN_Axiverse_code/ks_axiverse_python_collaborator/"
            "validation/cyaxiverse_fixed",
        )
    ).expanduser().resolve()
    install_import_stubs()
    sys.path.insert(0, str(fixed_dir))
    import Camcode_full_2 as author_code
    return author_code


def main():
    author_code = load_fixed_module()
    charges = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=float)
    kinv = np.diag([2.0, 3.0])
    tau = np.array([4.0, 5.0])
    volume = 7.0
    _, logs, signs = author_code.generate_charges_2(
        charges, kinv, tau, volume
    )

    kinetic_dot = float(charges[1] @ kinv @ charges[0])
    tau_dot = float((charges[1] + charges[0]) @ tau)
    expected_log = (
        np.log(8 * np.pi / volume**2)
        + np.log(abs(np.pi * kinetic_dot + tau_dot))
        - 2 * np.pi * tau_dot
    )
    assert np.isclose(logs[0], expected_log, rtol=0, atol=1e-12)
    assert signs[0] == np.sign(np.pi * kinetic_dot + tau_dot)

    # This deliberately separates the corrected and archived signs.
    negative_kinv = np.diag([-5.0, 0.0])
    negative_tau = np.array([4.0, 4.0])
    _, _, negative_signs = author_code.generate_charges_2(
        charges, negative_kinv, negative_tau, volume
    )
    old_sign = np.sign(
        charges[1] @ negative_kinv @ charges[0]
        + (charges[1] + charges[0]) @ negative_tau
    )
    corrected_sign = np.sign(
        np.pi * (charges[1] @ negative_kinv @ charges[0])
        + (charges[1] + charges[0]) @ negative_tau
    )
    assert old_sign != corrected_sign
    assert negative_signs[0] == corrected_sign
    print("author-code correction: PASS")


if __name__ == "__main__":
    main()
