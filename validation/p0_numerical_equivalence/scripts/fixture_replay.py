#!/usr/bin/env python3
"""Replay and fingerprint the P0 numerical-semantics fixture corpus.

This helper intentionally has no CYTools, NumPy, or package dependency.  It
replays the small source-level policies that define the F1--F13 contract and
prints a machine-readable report.  It is an evidence helper, not a replacement
for the historical Julia implementation or a production numerical kernel.
"""

from __future__ import annotations

import hashlib
import math
import pathlib
import sys
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
MIN_LOGSCALE = math.log10(sys.float_info.min)
TWO_PI = 2.0 * math.pi


def _as_float(value):
    if isinstance(value, str):
        if value == "-Inf":
            return float("-inf")
        if value == "+Inf":
            return float("inf")
        return float(value)
    return float(value)


def _matrix(rows):
    return [[_as_float(value) for value in row] for row in rows]


def _solve_2x2(a, b):
    """Solve a 2x2 system column-by-column using Float64 arithmetic."""
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    result = [[0.0 for _ in b[0]] for _ in range(2)]
    for col in range(len(b[0])):
        result[0][col] = (b[0][col] * a[1][1] - a[0][1] * b[1][col]) / det
        result[1][col] = (a[0][0] * b[1][col] - b[0][col] * a[1][0]) / det
    return result


def _transformed_q(data):
    q = _matrix(data["q"])
    basis = data.get("coordinate_basis")
    if basis is None:
        return q
    basis = _matrix(basis)
    if len(basis) == 2 and len(basis[0]) == 2:
        return _solve_2x2(basis, q)
    raise ValueError("fixture helper only supports 2x2 coordinate_basis")


def _logscale(data):
    return [_as_float(value) for value in data["l"][1]]


def _coefficients(data):
    return [_as_float(value) for value in data["l"][0]]


def _row_logscale(q, logs):
    result = []
    for row in q:
        support = [index for index, value in enumerate(row) if value != 0.0]
        result.append(max((logs[index] for index in support), default=max(logs)))
    return result


def _scaled_amplitudes(q, coeffs, logs, row_logs):
    rows = []
    for row, row_log in zip(q, row_logs):
        values = []
        for col, value in enumerate(row):
            if value == 0.0:
                values.append(0.0)
                continue
            delta = logs[col] - row_log
            # Julia's NaN >= minimum_logscale is false, as is Python's.
            values.append(coeffs[col] * 10.0**delta if delta >= MIN_LOGSCALE else 0.0)
        rows.append(values)
    return rows


def _global_amplitudes(coeffs, logs):
    """Evaluate the all-term global/log-shifted policy."""
    shift = max(logs)
    return [coefficient * 10.0 ** (log - shift) for coefficient, log in zip(coeffs, logs)]


def _dot(column_index, q, theta):
    return sum(q[row][column_index] * theta[row] for row in range(len(q)))


def _physical_hessian(q, coeffs, logs, row_logs, theta, phases):
    n = len(q)
    p = len(logs)
    result = [[0.0 for _ in range(n)] for _ in range(n)]
    for row in range(n):
        for col in range(n):
            for term in range(p):
                if q[row][term] == 0.0 or q[col][term] == 0.0:
                    continue
                delta = logs[term] - (row_logs[row] + row_logs[col]) / 2.0
                if delta < MIN_LOGSCALE:
                    continue
                argument = TWO_PI * _dot(term, q, theta) + phases[term]
                result[row][col] += (
                    q[row][term]
                    * q[col][term]
                    * coeffs[term]
                    * 10.0**delta
                    * math.cos(argument)
                )
    return [[TWO_PI**2 * value for value in row] for row in result]


def _eigs_2x2(matrix):
    a, b = matrix[0]
    _, d = matrix[1]
    center = (a + d) / 2.0
    radius = math.sqrt(((a - d) / 2.0) ** 2 + b * b)
    return [center - radius, center + radius]


def _fixture_result(data):
    # F11 is a named source fixture whose L values are deliberately generated
    # by the governed instanton_scales(qdot_tau, k) route, not duplicated in
    # this small hostile-input replay corpus.
    if "l" not in data:
        return {"source_fixture": True}
    q = _transformed_q(data)
    logs = _logscale(data)
    coeffs = _coefficients(data)
    row_logs = _row_logscale(q, logs)
    scaled = _scaled_amplitudes(q, coeffs, logs, row_logs)
    result = {
        "row_logscale": row_logs,
        "transformed_q": q,
        "support": [[value != 0.0 for value in row] for row in q],
        "scaled_amplitudes": scaled,
        "global_amplitudes": _global_amplitudes(coeffs, logs),
    }
    if "theta" in data and len(q) == 2 and len(q[0]) <= 4:
        theta = [_as_float(value) for value in data["theta"]]
        phases = [_as_float(value) for value in data.get("phases", [0.0] * len(logs))]
        hessian = _physical_hessian(q, coeffs, logs, row_logs, theta, phases)
        result["physical_hessian"] = hessian
        result["hessian_eigenvalues"] = _eigs_2x2(hessian)
        if data["id"] == "F10-near-degenerate-classification":
            scale = max(max(abs(value) for value in result["hessian_eigenvalues"]), 1.0)
            tolerance = 100.0 * 1e-10 * scale
            values = result["hessian_eigenvalues"]
            result["final_zero_tolerance"] = tolerance
            result["final_inertia"] = [
                sum(value < -tolerance for value in values),
                sum(abs(value) <= tolerance for value in values),
                sum(value > tolerance for value in values),
            ]
    return result


def _close(actual, expected, atol=2e-14):
    if isinstance(expected, str):
        return expected == "-Inf" and math.isinf(actual) and actual < 0
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _close(x, y, atol) for x, y in zip(actual, expected)
        )
    return math.isclose(actual, float(expected), rel_tol=2e-13, abs_tol=atol)


def main():
    paths = sorted(FIXTURES.glob("F*.toml"))
    if len(paths) != 13:
        raise SystemExit(f"expected 13 fixture files, found {len(paths)}")
    manifest = tomllib.loads((FIXTURES / "manifest.toml").read_text(encoding="utf-8"))
    manifest_by_id = {item["id"]: item for item in manifest["fixture"]}
    report = []
    for path in paths:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
        result = _fixture_result(data)
        checks = []
        digest = hashlib.sha256(raw).hexdigest()
        manifest_item = manifest_by_id.get(data["id"])
        checks.append({
            "field": "manifest_sha256",
            "ok": manifest_item is not None and manifest_item["sha256"] == digest,
        })
        expected = data.get("expected", {})
        for key in ("row_logscale", "transformed_q", "scaled_amplitudes",
                    "global_amplitudes", "support", "final_zero_tolerance",
                    "final_inertia"):
            if key in expected and key in result:
                checks.append({"field": key, "ok": _close(result[key], expected[key])})
        if data["id"] == "F6-coordinate-displacement":
            hilltop = [_as_float(v) for v in data["hilltop"]]
            direction = [_as_float(v) for v in data["mass_direction"]]
            displacement = _as_float(data["displacement"])
            sign = _as_float(data["displacement_sign"])
            theta_initial = [
                (h + sign * displacement * direction[index]) % 1.0
                for index, h in enumerate(hilltop)
            ]
            checks.append({
                "field": "theta_initial",
                "ok": _close(theta_initial, expected["theta_initial"]),
            })
        report.append({
            "id": data["id"],
            "file": str(path.relative_to(ROOT.parent)),
            "sha256": digest,
            "checks": checks,
            "derived": result,
        })
    failed = [item for item in report for check in item["checks"] if not check["ok"]]
    for item in report:
        print(item)
    if failed:
        raise SystemExit(f"{len(failed)} fixture checks failed")


if __name__ == "__main__":
    main()
