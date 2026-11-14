"""CYTools-free F5 fuzzy-axion mass evaluator (CYAX-0232 R-001, T021).

Moved verbatim from ``reproduce_fuzzy_axions_h11_4.py``, which re-exports it,
so that its c = 1 equivalence test runs where CYTools is not installed.
"""
from __future__ import annotations

import numpy as np


def _fuzzy_axion_mass_log10(
    values,
    glsm_charge_matrix,
    *,
    axion_index,
    gs=0.5,
    w0=1.0 + 0.0j,
):
    """Compute one leading-instanton mass in the model-stage convention.

    Keep this evaluator local to the diagnostic optimizer.  The canonical
    model-stage path remains the Julia implementation and all-prime-divisor
    selection.
    """
    tau = np.asarray(values["prime_divisor_volumes"], dtype=float).reshape(-1)
    inverse_metric = np.asarray(values["inverse_metric"], dtype=float)
    charge_matrix = np.asarray(glsm_charge_matrix, dtype=int)
    cy_volume = float(values["cy_volume"])
    if tau.size == 0 or charge_matrix.shape != (inverse_metric.shape[0], tau.size):
        raise ValueError("optimizer mass inputs have inconsistent Q, tau, and metric shapes")
    if not np.all(np.isfinite(tau)) or np.any(tau <= 0.0):
        raise ValueError("optimizer mass inputs require positive finite divisor volumes")
    if not np.isfinite(cy_volume) or cy_volume <= 0.0:
        raise ValueError("optimizer mass inputs require a positive finite CY volume")
    if not np.isfinite(gs) or gs <= 0.0:
        raise ValueError("optimizer mass inputs require a positive finite gs")

    metric = 0.5 * (inverse_metric + inverse_metric.T)
    try:
        # c = 1 supergravity kinetic normalization (CYAX-0232): the canonical
        # factor is cholesky(Kinv / 2), as CYAxiverse.kinetic_factor.
        cholesky = np.linalg.cholesky(0.5 * metric)
    except np.linalg.LinAlgError as exc:
        raise ValueError("optimizer mass inputs require a positive-definite inverse metric") from exc

    prefactor = float(gs) ** 4 / 128.0
    superpotential = complex(w0) + np.exp(-2.0 * np.pi * tau).sum()
    gravitino_mass_planck = np.sqrt(prefactor) * abs(superpotential) / cy_volume
    if not np.isfinite(gravitino_mass_planck) or gravitino_mass_planck <= 0.0:
        raise ValueError("optimizer mass prefactor is non-finite or non-positive")
    log10_prefactor = (
        np.log10(8.0 * np.pi)
        + 0.5 * np.log10(prefactor)
        + np.log10(gravitino_mass_planck)
        - np.log10(cy_volume)
    )
    scale_logs = np.log10(tau) - 2.0 * np.pi * tau * np.log10(np.e)
    scale_logs = scale_logs + log10_prefactor
    order = np.argsort(-scale_logs, kind="stable")

    h11 = charge_matrix.shape[0]
    selected = []
    orthogonal = []
    for column_index in order:
        direction = charge_matrix[:, int(column_index)].astype(float)
        original_norm_squared = float(direction @ direction)
        if original_norm_squared == 0.0:
            continue
        residual = direction.copy()
        for _ in range(2):
            for basis_vector in orthogonal:
                residual -= float(residual @ basis_vector) * basis_vector
        residual_norm_squared = float(residual @ residual)
        if residual_norm_squared > np.finfo(float).eps * original_norm_squared:
            selected.append(int(column_index))
            orthogonal.append(residual / np.sqrt(residual_norm_squared))
            if len(selected) == h11:
                break
    if len(selected) != h11:
        raise ValueError("optimizer mass charge matrix has fewer than h11 independent columns")
    if not 1 <= int(axion_index) <= h11:
        raise ValueError(f"axion_index must be in [1, {h11}]")

    qleading = charge_matrix[:, selected].T @ cholesky
    pq_basis = []
    mass_logs = []
    for row_index, row in enumerate(qleading):
        direction = row.copy()
        for basis_vector in pq_basis:
            direction -= float(direction @ basis_vector) * basis_vector
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm == 0.0:
            raise ValueError("optimizer mass charge rows are linearly dependent")
        pq_basis.append(direction / direction_norm)
        f_log10 = np.log10(1.0 / (2.0 * np.pi * direction_norm**2))
        mass_logs.append(
            0.5
            * (
                scale_logs[selected[row_index]]
                - f_log10
                - np.log10(2.0 * np.pi)
            )
            + 9.0
            + np.log10(2.435e18)
            + np.log10(2.0 * np.pi)
        )
    return float(mass_logs[int(axion_index) - 1])
