#!/usr/bin/env python3
"""Independent Python transcription of the poly-102 Mathematica trajectory.

This intentionally does not import CYAxiverse.  It mirrors the author code's
raw potential, canonical map, gradient-flow parameter, slow-roll diagnostics,
and scan-based entry/end-event logic.  The author calculation uses high
precision; this independent SciPy transcription therefore reports the final
finite slow-roll exit when machine-precision transient crossings are present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


KC = 0.674506370003365
BEST_X = np.array([
    0.0, 1.539958173041265, 4.743227134138319, 0.03083815375363047,
    6.252347153425956, 4.774065287891951, 4.71238898069045, 0.0,
])
TAU = np.array([14.0, 14.5, 14.5, 15.5, 15.5, 15.5, 15.5, 16.0, 17.0, 17.0])
Q = np.array([
    [-1, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, -1, 1, -1, 1, 0, 1, 0],
    [0, 1, -1, -1, 1, 1, 0, 0],
    [1, 0, 0, -1, -1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
], dtype=float)
K_RAW = np.array([
    [2.694e-4, -1.272e-4, -1.272e-4, -6.889e-5, -8.405e-5, -7.585e-6, -7.585e-6, -1.345e-4],
    [-1.272e-4, 4.674e-4, -1.093e-6, -3.058e-5, 3.998e-5, 1.455e-4, -7.495e-5, 1.801e-4],
    [-1.272e-4, -1.093e-6, 4.674e-4, -3.058e-5, 3.998e-5, -7.495e-5, 1.455e-4, 1.801e-4],
    [-6.889e-5, -3.058e-5, -3.058e-5, 3.301e-4, -1.298e-4, -8.825e-5, -8.825e-5, 1.019e-7],
    [-8.405e-5, 3.998e-5, 3.998e-5, -1.298e-4, 4.838e-4, 1.651e-4, 1.651e-4, 4.343e-6],
    [-7.585e-6, 1.455e-4, -7.495e-5, -8.825e-5, 1.651e-4, 2.369e-4, 1.645e-5, 2.121e-6],
    [-7.585e-6, -7.495e-5, 1.455e-4, -8.825e-5, 1.651e-4, 1.645e-5, 2.369e-4, 2.121e-6],
    [-1.345e-4, 1.801e-4, 1.801e-4, 1.019e-7, 4.343e-6, 2.121e-6, 2.121e-6, 2.300e-4],
])


def symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * values**power) @ vectors.T


def coefficient_weights(k: float) -> np.ndarray:
    """Stable form of Mathematica's `-L[-14] * L[tau]` coefficients."""
    return (14.0 * k) ** 2 * (TAU / 14.0) * np.exp(
        -2.0 * np.pi * k * (TAU - 14.0)
    )


def raw_derivatives(theta: np.ndarray, k: float):
    weights = coefficient_weights(k)
    arguments = Q @ theta
    value = float(np.sum(weights * (1.0 - np.cos(arguments))))
    gradient = Q.T @ (weights * np.sin(arguments))
    hessian = Q.T @ ((weights * np.cos(arguments))[:, None] * Q)
    return value, gradient, hessian


def canonical_data(chi: np.ndarray, k: float):
    metric = K_RAW / k**2
    inverse_root = symmetric_power(metric, -0.5)
    theta = inverse_root @ chi
    value, raw_gradient, raw_hessian = raw_derivatives(theta, k)
    gradient = inverse_root.T @ raw_gradient
    hessian = inverse_root.T @ raw_hessian @ inverse_root
    speed_squared = float(gradient @ gradient)
    epsilon = 0.5 * speed_squared / max(value * value, 1e-300)
    if speed_squared == 0.0:
        eta_parallel = math.inf
    else:
        tangent = gradient / math.sqrt(speed_squared)
        eta_parallel = float(tangent @ hessian @ tangent / max(value, 1e-300))
    return theta, value, gradient, hessian, epsilon, eta_parallel


def basis_data(k: float):
    metric = K_RAW / k**2
    inverse_root = symmetric_power(metric, -0.5)
    _, _, raw_hessian = raw_derivatives(BEST_X, k)
    canonical_hessian = inverse_root.T @ raw_hessian @ inverse_root
    values, vectors = np.linalg.eigh(canonical_hessian)
    raw_vectors = inverse_root @ vectors
    kinetic_values, kinetic_vectors = np.linalg.eigh(metric)
    kinetic_raw = [
        vector / math.sqrt(vector @ metric @ vector)
        for vector in kinetic_vectors.T
    ]
    unstable = raw_vectors[:, int(np.argmin(values))]
    draft = max(kinetic_raw, key=lambda vector: abs(vector @ metric @ unstable))
    raw_coordinate = np.zeros(8)
    raw_coordinate[0] = 1.0
    raw_coordinate /= math.sqrt(raw_coordinate @ metric @ raw_coordinate)
    return {
        "A_draft_kinetic": draft,
        "B_author_canonical_hessian": unstable,
        "C_canonical_hessian": unstable,
        "D_raw_coordinate": raw_coordinate,
        "E_mass_eigenbasis": unstable,
    }, values, raw_vectors, canonical_hessian


def root_in_interval(function, left: float, right: float) -> float:
    return float(brentq(function, left, right, xtol=1e-10, rtol=1e-12))


def trajectory(delta_k: float, direction: np.ndarray, *,
               max_time: float = 1e6, sample_count: int = 20):
    k = KC + float(delta_k)
    metric = K_RAW / k**2
    root = symmetric_power(metric, 0.5)
    initial_theta = BEST_X - 1e-8 * direction
    initial_chi = root @ initial_theta

    def rhs(_time, state):
        _, value, gradient, _, _, _ = canonical_data(state[:8], k)
        return np.concatenate((-gradient, [value]))

    solution = solve_ivp(
        rhs,
        (0.0, max_time),
        np.concatenate((initial_chi, [0.0])),
        method="BDF",
        rtol=1e-9,
        atol=1e-12,
        max_step=100.0,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    def end_function(time: float) -> float:
        state = solution.sol(time)
        diagnostics = canonical_data(state[:8], k)
        return max(diagnostics[4], abs(diagnostics[5])) - 1.0

    times = solution.t
    values = np.array([end_function(float(time)) for time in times])
    entry_indices = np.where((values[:-1] > 0.0) & (values[1:] <= 0.0))[0]
    if len(entry_indices) == 0:
        return {
            "delta_k": delta_k,
            "k": k,
            "entered_slow_roll": False,
            "end_event": "no_slow_roll_window",
            "efolds": 0.0,
            "initial_theta": initial_theta.tolist(),
            "initial_epsilon": float(values[0]),
            "samples": [],
        }
    entry_index = int(entry_indices[0])
    entry_time = root_in_interval(end_function, times[entry_index], times[entry_index + 1])
    exit_indices = np.where(
        (times[entry_index + 1:-1] > entry_time)
        & (values[entry_index + 1:-1] < 0.0)
        & (values[entry_index + 2:] >= 0.0)
    )[0]
    if len(exit_indices):
        # Stiff machine-precision integrations can briefly cross the
        # slow-roll boundary during the heavy-mode transient.  The final
        # finite crossing is the one that brackets the physical exit.
        exit_index = entry_index + 1 + int(exit_indices[-1])
        exit_time = root_in_interval(end_function, times[exit_index], times[exit_index + 1])
        end_event = "eta_parallel" if abs(
            canonical_data(solution.sol(exit_time)[:8], k)[5]
        ) >= canonical_data(solution.sol(exit_time)[:8], k)[4] else "epsilon"
    else:
        exit_time = float(times[-1])
        end_event = "tmax"

    slow_roll_entry_n = float(solution.sol(entry_time)[-1])
    initial_n = float(solution.sol(0.0)[-1])
    exit_n = float(solution.sol(exit_time)[-1])
    sample_n = np.linspace(initial_n, exit_n, sample_count)
    samples = []
    for target_n in sample_n:
        sample_time = root_in_interval(
            lambda time: float(solution.sol(time)[-1] - target_n),
            0.0, exit_time,
        )
        state = canonical_data(solution.sol(sample_time)[:8], k)
        samples.append({
            "n": float(target_n),
            "theta": state[0].tolist(),
            "epsilon": float(state[4]),
            "eta_parallel": float(state[5]),
            "potential": float(state[1]),
        })
    initial = canonical_data(initial_chi, k)
    return {
        "delta_k": delta_k,
        "k": k,
        "entered_slow_roll": True,
        "end_event": end_event,
        "entry_n": initial_n,
        "slow_roll_entry_n": slow_roll_entry_n,
        "end_n": exit_n,
        "efolds": exit_n - initial_n,
        "initial_theta": initial_theta.tolist(),
        "initial_epsilon": float(initial[4]),
        "initial_eta_parallel": float(initial[5]),
        "samples": samples,
        "solver": {
            "method": "scipy.BDF",
            "rtol": 1e-9,
            "atol": 1e-12,
            "max_step": 100.0,
            "event_policy": "final_finite_exit",
        },
    }


def main(output: Path, detunings: tuple[float, ...]):
    directions, eigenvalues, raw_vectors, canonical_hessian = basis_data(KC)
    results = []
    for delta_k in detunings:
        directions_at_k, _, _, _ = basis_data(KC + delta_k)
        for label in ("B_author_canonical_hessian", "E_mass_eigenbasis"):
            result = trajectory(delta_k, directions_at_k[label])
            result["basis"] = label
            results.append(result)
            print(
                f"{label} delta={delta_k:g}: N_e={result['efolds']:.10g} "
                f"event={result['end_event']} samples={len(result['samples'])}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "source": "independent transcription of poly102_core.wl",
        "kc": KC,
        "basis_eigenvalues": eigenvalues.tolist(),
        "canonical_hessian": canonical_hessian.tolist(),
        "raw_eigenvectors": raw_vectors.tolist(),
        "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("validation/author_poly102_reference_miniforge.json"))
    parser.add_argument("--detunings", default="1e-7,0.0015320548620798324")
    args = parser.parse_args()
    main(args.output, tuple(float(value) for value in args.detunings.split(",")))
