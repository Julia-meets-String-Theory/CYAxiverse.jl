"""Compact, serializable EFT-reference rows for the Glimmers schema.

The geometry generator owns dense-potential reconstruction.  This module only
consumes a bounded ``Q``/``L`` view long enough to replay the existing
CYAxiverse potential-match and exact rank conventions, then emits a compact
row with scalar values, source indices, and factorized-charge references.
It never stores a dense potential or a charge vector in a model row.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math

import numpy as np

try:
    from qed_divisor_assignment import (
        QEDAssignmentFailure,
        classify_qed_leading_status,
        record_potential_match,
    )
except ModuleNotFoundError as exc:
    if exc.name != "qed_divisor_assignment":
        raise
    from scripts.qed_divisor_assignment import (
        QEDAssignmentFailure,
        classify_qed_leading_status,
        record_potential_match,
    )


ROW_SCHEMA_VERSION = "glimmers-eft-row-1.1"
POTENTIAL_DERIVATION_VERSION = "cyaxiverse-oriented-potential-L-v1"
DEFAULT_DERIVATION_STATUS = "adapted_compact_eft_reference"
ACCEPTED_TERMINAL_STATUS = "accepted_model_row"
CHARGE_FACTORIZED_SCHEMA_VERSION = "glimmers-charge-factorized-1.1"
QCD_VOLUME_TARGET = 40.0
QED_VOLUME_MAX = 127.5

POTENTIAL_ORIENTATION = "h11 x N_instantons; charge vectors are columns"
POTENTIAL_L_CONVENTION = "2 x N; rows are sign/mantissa and log10 Lambda^4"
PAIRWISE_DIFFERENCE_CONVENTION = (
    "q_pair[:, k] = q_direct[:, pair_j[k]] - q_direct[:, pair_i[k]]"
)

REQUIRED_FIELDS = (
    "model_id",
    "geometry_id",
    "assignment_hash",
    "assignment_pool_rank",
    "assignment_pool_size",
    "qcd_divisor_index",
    "qed_divisor_index",
    "qcd_divisor_label",
    "qed_divisor_label",
    "qcd_volume_scale",
    "qcd_volume",
    "qed_volume",
    "qed_potential_source",
    "qed_unsorted_potential_index",
    "qed_post_sort_source_position",
    "qed_log10_lambda4",
    "qed_leading_status",
    "leading_rank_certificate",
    "charge_factorized_schema_version",
    "normalization_map_version",
    "derivation_status",
    "potential_derivation_version",
    "potential_l_convention",
    "qed_charge_reference",
)

TERMINAL_STATUSES = (
    "accepted_model_row",
    "potential_term_mismatch",
    "rank_span_classification_failure",
    "missing_assignment_derived_data",
    "invalid_row_schema",
    "invalid_geometry_reference",
    "model_duplicate_assignment",
    "output_collision",
    "user_decision_required",
)

_DENSE_ROW_KEYS = frozenset(
    {
        # Potential and charge arrays are transient reconstruction products,
        # never persisted row fields.
        "q",
        "Q",
        "l",
        "L",
        "k",
        "K",
        "Kinv",
        "kinv",
        "k_inverse",
        "inverse_kahler_metric",
        "qcd_charge",
        "qed_charge",
        "pair_i",
        "pair_j",
        "potential",
        "potential_arrays",
        # Dense geometric quantities are likewise reconstructed on demand.
        "divisor_volumes",
        "prime_divisor_volumes",
        "effective_divisor_volumes",
        "curve_volumes",
        "volume",
        "volumes",
        "CY_volume",
        "cy_volume",
    }
)


class EFTRowFailure(RuntimeError):
    """Report a row-construction outcome under the fixed terminal taxonomy."""

    def __init__(self, terminal_status, reason, record=None):
        if terminal_status not in TERMINAL_STATUSES[1:]:
            raise ValueError(f"unknown EFT row terminal status {terminal_status!r}")
        super().__init__(str(reason))
        self.terminal_status = terminal_status
        # ``category`` keeps failure handling parallel with QEDAssignmentFailure.
        self.category = terminal_status
        self.reason = str(reason)
        self.record = {} if record is None else dict(record)


def _failure(status, reason, record=None):
    raise EFTRowFailure(status, reason, record)


EFTRowSchemaFailure = EFTRowFailure


def _jsonable(value):
    """Convert NumPy scalar containers to JSON and Parquet scalar values."""
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _json_scalar(value, field):
    """Encode one nested label/reference as a deterministic string scalar."""
    try:
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        _failure("invalid_row_schema", f"{field} is not JSON serializable: {exc}")


def _mapping(value, name, status):
    if not isinstance(value, Mapping):
        _failure(status, f"{name} must be a mapping")
    return value


def _required(mapping, field, status):
    value = mapping.get(field)
    if value is None:
        _failure(status, f"missing required {field}")
    return value


def _first_present(mapping, names):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _integer(value, field, status, *, minimum=None):
    if isinstance(value, (bool, np.bool_)):
        _failure(status, f"{field} must be an integer, not a boolean")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        _failure(status, f"{field} must be an integer: {exc}")
    try:
        if value != integer:
            _failure(status, f"{field} must be integral")
    except (TypeError, ValueError):
        _failure(status, f"{field} must be an integer")
    if minimum is not None and integer < minimum:
        _failure(status, f"{field} must be at least {minimum}")
    return integer


def _finite_float(value, field, status, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        _failure(status, f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        _failure(status, f"{field} must be a finite number: {exc}")
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive and " if positive else ""
        _failure(status, f"{field} must be {qualifier}finite")
    return number


def _nonempty_string(value, field, status):
    if not isinstance(value, str) or not value:
        _failure(status, f"{field} must be a non-empty string")
    return value


def _potential_inputs(potential, q, l, qed_charge, direct_count, source_index):
    """Resolve the bounded potential view and its source metadata."""
    if potential is None:
        potential = {}
    potential = _mapping(potential, "potential", "potential_term_mismatch")
    q = q if q is not None else _first_present(potential, ("q", "Q"))
    l = l if l is not None else _first_present(potential, ("l", "L"))
    qed_charge = (
        qed_charge
        if qed_charge is not None
        else _first_present(potential, ("qed_charge", "em_charge", "charge"))
    )
    direct_count = (
        direct_count
        if direct_count is not None
        else _first_present(potential, ("direct_count", "direct_source_count"))
    )
    source_index = (
        source_index
        if source_index is not None
        else _first_present(
            potential,
            (
                "source_index",
                "qed_potential_source_index",
                "qed_unsorted_potential_index",
                "qed_instanton_index",
            ),
        )
    )
    missing = [
        name
        for name, value in (
            ("Q", q),
            ("L", l),
            ("qed_charge", qed_charge),
            ("direct_count", direct_count),
            ("source_index", source_index),
        )
        if value is None
    ]
    if missing:
        _failure(
            "potential_term_mismatch",
            "unresolved potential fields: " + ", ".join(missing),
        )
    return q, l, qed_charge, direct_count, source_index, potential


def _replay_potential_conventions(q, l, qed_charge, direct_count, source_index):
    """Replay the existing exact source-match and rank/span conventions."""
    try:
        q_array = np.asarray(q)
        l_array = np.asarray(l, dtype=float)
        charge_array = np.asarray(qed_charge)
    except (TypeError, ValueError) as exc:
        _failure("potential_term_mismatch", f"potential arrays are invalid: {exc}")

    if q_array.ndim != 2 or l_array.ndim != 2 or charge_array.ndim == 0:
        _failure("potential_term_mismatch", "potential arrays have incompatible ranks")
    if not np.all(np.isfinite(l_array)):
        _failure("potential_term_mismatch", "potential scales contain non-finite values")
    if l_array.shape[0] != 2 or l_array.shape[1] != q_array.shape[1]:
        _failure("potential_term_mismatch", "potential Q/L shapes are incompatible")
    if np.any(l_array[0, :] == 0.0):
        _failure("potential_term_mismatch", "potential sign/mantissa row contains zero")

    direct_count = _integer(
        direct_count,
        "direct_count",
        "potential_term_mismatch",
        minimum=0,
    )
    source_index = _integer(
        source_index,
        "source_index",
        "potential_term_mismatch",
        minimum=0,
    )
    if direct_count > q_array.shape[1]:
        _failure("potential_term_mismatch", "direct_count exceeds potential term count")
    if source_index >= q_array.shape[1]:
        _failure("potential_term_mismatch", "source_index exceeds potential term count")
    if source_index >= direct_count and source_index != q_array.shape[1] - 1:
        _failure(
            "potential_term_mismatch",
            "a non-direct QED source must be the appended final potential column",
        )

    try:
        match = record_potential_match(
            q_array, l_array, qed_charge, direct_count, source_index
        )
    except QEDAssignmentFailure as exc:
        _failure("potential_term_mismatch", str(exc), getattr(exc, "record", None))
    except (TypeError, ValueError, OverflowError) as exc:
        _failure("potential_term_mismatch", f"potential matching failed: {exc}")

    try:
        certificate = classify_qed_leading_status(
            q_array, l_array, source_index
        )
    except QEDAssignmentFailure as exc:
        status = (
            "potential_term_mismatch"
            if getattr(exc, "category", None) == "potential_term_mismatch"
            else "rank_span_classification_failure"
        )
        _failure(status, str(exc), getattr(exc, "record", None))
    except (TypeError, ValueError, OverflowError, ArithmeticError) as exc:
        _failure("rank_span_classification_failure", f"exact rank replay failed: {exc}")

    if not isinstance(certificate, Mapping):
        _failure("rank_span_classification_failure", "rank helper returned a non-record")
    status = certificate.get("status")
    if status not in {"leading", "dependent"}:
        _failure(
            "rank_span_classification_failure",
            f"rank helper returned unsupported status {status!r}",
        )
    if certificate.get("method") != "exact_rational_incremental_rank":
        _failure(
            "rank_span_classification_failure",
            "rank certificate does not record the exact rational method",
        )
    for field in ("selected_source_indices", "ordered_source_indices", "selected_rank"):
        if certificate.get(field) is None:
            _failure("rank_span_classification_failure", f"rank certificate lacks {field}")
    return match, dict(certificate), direct_count, source_index


def _labels(assignment):
    qcd_label = _required(assignment, "qcd_divisor_label", "missing_assignment_derived_data")
    qed_label = _required(assignment, "qed_divisor_label", "missing_assignment_derived_data")
    # Validate labels now so malformed labels cannot be hidden in a JSON string.
    normalized = []
    for name, label in (("qcd_divisor_label", qcd_label), ("qed_divisor_label", qed_label)):
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        if isinstance(label, str):
            try:
                label = json.loads(label)
            except (TypeError, ValueError) as exc:
                _failure("missing_assignment_derived_data", f"{name} is not JSON label metadata: {exc}")
        try:
            label_array = np.asarray(label)
        except (TypeError, ValueError) as exc:
            _failure("missing_assignment_derived_data", f"{name} is not a coordinate label: {exc}")
        if label_array.ndim != 1 or label_array.size == 0:
            _failure("missing_assignment_derived_data", f"{name} must be a coordinate label")
        normalized.append(
            [
                _integer(value, f"{name} coordinate", "missing_assignment_derived_data")
                for value in label_array.tolist()
            ]
        )
    return _json_scalar(normalized[0], "qcd_divisor_label"), _json_scalar(
        normalized[1], "qed_divisor_label"
    )


def serialize_eft_row(
    geometry,
    assignment,
    potential=None,
    *,
    model_id=None,
    q=None,
    l=None,
    qed_charge=None,
    direct_count=None,
    source_index=None,
    derivation_status=DEFAULT_DERIVATION_STATUS,
    derivation_version=POTENTIAL_DERIVATION_VERSION,
):
    """Build one compact EFT row from one geometry and one assignment.

    ``q`` and ``l`` are bounded, ephemeral arrays in the existing CYAxiverse
    orientation: ``Q`` is ``h11 × N`` and ``L`` is ``2 × N``.  They are used
    only to call :func:`record_potential_match` and
    :func:`classify_qed_leading_status`; neither array nor any charge vector is
    placed in the returned row.  Missing potential data is terminal rather
    than represented by a null field.
    """
    geometry = _mapping(geometry, "geometry", "invalid_geometry_reference")
    assignment = _mapping(assignment, "assignment", "missing_assignment_derived_data")
    geometry_id = _nonempty_string(
        _required(geometry, "geometry_id", "invalid_geometry_reference"),
        "geometry_id",
        "invalid_geometry_reference",
    )
    assignment_geometry_id = assignment.get("geometry_id")
    if assignment_geometry_id is not None and str(assignment_geometry_id) != geometry_id:
        _failure(
            "invalid_geometry_reference",
            "assignment geometry_id does not match the geometry reference",
        )

    if model_id is None:
        model_id = _first_present(assignment, ("model_id",))
    model_id = _nonempty_string(model_id, "model_id", "missing_assignment_derived_data")

    assignment_hash = _nonempty_string(
        _required(assignment, "assignment_hash", "missing_assignment_derived_data"),
        "assignment_hash",
        "missing_assignment_derived_data",
    )
    pool_rank = _integer(
        _required(assignment, "assignment_pool_rank", "missing_assignment_derived_data"),
        "assignment_pool_rank",
        "missing_assignment_derived_data",
        minimum=0,
    )
    pool_size = _integer(
        _required(assignment, "assignment_pool_size", "missing_assignment_derived_data"),
        "assignment_pool_size",
        "missing_assignment_derived_data",
        minimum=1,
    )
    if pool_rank >= pool_size:
        _failure("missing_assignment_derived_data", "assignment pool rank is outside its pool")

    qcd_index = _integer(
        _required(assignment, "qcd_divisor_index", "missing_assignment_derived_data"),
        "qcd_divisor_index",
        "missing_assignment_derived_data",
        minimum=0,
    )
    qed_index = _integer(
        _required(assignment, "qed_divisor_index", "missing_assignment_derived_data"),
        "qed_divisor_index",
        "missing_assignment_derived_data",
        minimum=0,
    )
    if qcd_index == qed_index:
        _failure("missing_assignment_derived_data", "QCD and QED divisor indices must differ")
    qcd_label, qed_label = _labels(assignment)
    qcd_volume_scale = _finite_float(
        _required(assignment, "qcd_volume_scale", "missing_assignment_derived_data"),
        "qcd_volume_scale",
        "missing_assignment_derived_data",
        positive=True,
    )
    qcd_volume = _finite_float(
        _required(assignment, "qcd_volume", "missing_assignment_derived_data"),
        "qcd_volume",
        "missing_assignment_derived_data",
        positive=True,
    )
    qed_volume = _finite_float(
        _required(assignment, "qed_volume", "missing_assignment_derived_data"),
        "qed_volume",
        "missing_assignment_derived_data",
        positive=True,
    )
    if not math.isclose(qcd_volume, QCD_VOLUME_TARGET, rel_tol=0.0, abs_tol=1e-9):
        _failure(
            "missing_assignment_derived_data",
            "qcd_volume is not the approved normalized value 40.0",
        )
    if not qed_volume <= QED_VOLUME_MAX:
        _failure(
            "missing_assignment_derived_data",
            "qed_volume exceeds the inclusive 127.5 bound",
        )

    factorized_version = geometry.get("charge_factorized_schema_version")
    assignment_factorized_version = assignment.get("charge_factorized_schema_version")
    if (
        factorized_version is None
        or assignment_factorized_version is not None
        and str(assignment_factorized_version) != str(factorized_version)
    ):
        _failure(
            "invalid_geometry_reference",
            "geometry and assignment charge-factorization versions are unresolved or differ",
        )
    factorized_version = _nonempty_string(
        str(factorized_version),
        "charge_factorized_schema_version",
        "invalid_geometry_reference",
    )
    if factorized_version != CHARGE_FACTORIZED_SCHEMA_VERSION:
        _failure(
            "invalid_geometry_reference",
            "geometry uses an unsupported charge-factorized schema version",
        )
    normalization_version = _nonempty_string(
        _required(assignment, "normalization_map_version", "missing_assignment_derived_data"),
        "normalization_map_version",
        "missing_assignment_derived_data",
    )
    geometry_normalization_version = geometry.get("normalization_map_version")
    if (
        geometry_normalization_version is not None
        and str(geometry_normalization_version) != normalization_version
    ):
        _failure(
            "invalid_geometry_reference",
            "geometry and assignment normalization-map versions differ",
        )
    derivation_status = _nonempty_string(
        derivation_status,
        "derivation_status",
        "invalid_row_schema",
    )
    derivation_version = _nonempty_string(
        derivation_version,
        "potential_derivation_version",
        "invalid_row_schema",
    )

    q, l, qed_charge, direct_count, source_index, potential = _potential_inputs(
        potential, q, l, qed_charge, direct_count, source_index
    )
    match, certificate, direct_count, source_index = _replay_potential_conventions(
        q, l, qed_charge, direct_count, source_index
    )
    logical_term_count = int(np.asarray(q).shape[1])
    pair_source_count = logical_term_count - direct_count
    if source_index >= direct_count:
        pair_source_count -= 1
    if pair_source_count < 0:
        _failure("potential_term_mismatch", "potential has no valid factorized pair-source count")

    row = {
        "model_id": model_id,
        "geometry_id": geometry_id,
        "assignment_hash": assignment_hash,
        "assignment_pool_rank": pool_rank,
        "assignment_pool_size": pool_size,
        "qcd_divisor_index": qcd_index,
        "qed_divisor_index": qed_index,
        "qcd_divisor_label": qcd_label,
        "qed_divisor_label": qed_label,
        "qcd_volume_scale": qcd_volume_scale,
        "qcd_volume": qcd_volume,
        "qed_volume": qed_volume,
        "qed_potential_source": _nonempty_string(
            match["qed_potential_source"],
            "qed_potential_source",
            "potential_term_mismatch",
        ),
        "qed_unsorted_potential_index": int(match["qed_unsorted_potential_index"]),
        "qed_post_sort_source_position": int(match["qed_post_sort_source_position"]),
        "qed_log10_lambda4": _finite_float(
            match["qed_potential_scale"],
            "qed_log10_lambda4",
            "potential_term_mismatch",
        ),
        "qed_leading_status": certificate["status"],
        "leading_rank_certificate": _json_scalar(
            certificate, "leading_rank_certificate"
        ),
        "charge_factorized_schema_version": factorized_version,
        "normalization_map_version": normalization_version,
        "derivation_status": derivation_status,
        "eft_row_schema_version": ROW_SCHEMA_VERSION,
        "potential_derivation_version": derivation_version,
        "terminal_status": ACCEPTED_TERMINAL_STATUS,
        "qed_charge_exact_match": True,
        "qed_charge_reference": _json_scalar(
            {
                "storage": "geometry_references_only",
                "orientation": POTENTIAL_ORIENTATION,
                "source_index": source_index,
                "source_kind": match["qed_potential_source"],
                "direct_source_count": direct_count,
                "pair_source_count": pair_source_count,
                "pair_ordering": "lexicographic_i_then_j_with_i_less_than_j",
                "direct_charge_coefficients": "+1",
                "pair_charge_coefficients": "[-1, +1]",
                "pair_source_indices": "geometry-level pair_i/pair_j",
                "difference_convention": PAIRWISE_DIFFERENCE_CONVENTION,
            },
            "qed_charge_reference",
        ),
        "potential_l_convention": POTENTIAL_L_CONVENTION,
    }

    # Preserve useful identity/normalization scalars when upstream records
    # provide them, while keeping them assignment- or geometry-scoped.
    for name in (
        "geometry_hash",
        "geometry_schema_version",
        "geometry_file",
        "h11",
        "h21",
        "model_seed",
        "row_order",
        "qcd_radial_scale",
        "minimum_prime_volume",
        "minimum_effective_volume",
    ):
        source = assignment if name in assignment else geometry
        if name in source and source[name] is not None:
            value = source[name]
            if name in {"h11", "h21", "model_seed", "row_order"}:
                value = _integer(value, name, "invalid_row_schema")
            elif name in {
                "qcd_radial_scale",
                "minimum_prime_volume",
                "minimum_effective_volume",
            }:
                value = _finite_float(value, name, "invalid_row_schema")
            elif not isinstance(value, (str, int, float, bool)):
                value = _json_scalar(value, name)
            row[name] = _jsonable(value)

    validate_eft_row(row, geometry=geometry, assignment=assignment)
    return row


def validate_eft_row(row, *, geometry=None, assignment=None):
    """Validate a serialized row and its geometry/assignment references."""
    if not isinstance(row, Mapping):
        _failure("invalid_row_schema", "row must be a mapping")
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        _failure("invalid_row_schema", "row lacks required fields: " + ", ".join(missing))
    null_fields = [field for field, value in row.items() if value is None]
    if null_fields:
        _failure("invalid_row_schema", "accepted rows cannot contain null fields")
    dense_fields = sorted(set(row).intersection(_DENSE_ROW_KEYS))
    if dense_fields:
        _failure(
            "invalid_row_schema",
            "compact rows cannot contain dense potential/charge fields: "
            + ", ".join(dense_fields),
        )
    if row.get("terminal_status", ACCEPTED_TERMINAL_STATUS) != ACCEPTED_TERMINAL_STATUS:
        _failure("invalid_row_schema", "serializer rows must be accepted model rows")
    if row["qed_leading_status"] not in {"leading", "dependent"}:
        _failure("invalid_row_schema", "qed_leading_status is not leading or dependent")
    if not isinstance(row["leading_rank_certificate"], str):
        _failure("invalid_row_schema", "leading_rank_certificate must be a JSON string scalar")
    try:
        certificate = json.loads(row["leading_rank_certificate"])
    except (TypeError, ValueError) as exc:
        _failure("invalid_row_schema", f"leading rank certificate is not valid JSON: {exc}")
    if not isinstance(certificate, Mapping):
        _failure("invalid_row_schema", "leading rank certificate must decode to an object")
    if certificate.get("status") != row["qed_leading_status"]:
        _failure("invalid_row_schema", "rank certificate status disagrees with row status")
    if certificate.get("method") != "exact_rational_incremental_rank":
        _failure("invalid_row_schema", "row does not record exact rational rank derivation")
    for field in (
        "model_id",
        "geometry_id",
        "assignment_hash",
        "qed_potential_source",
        "qcd_divisor_label",
        "qed_divisor_label",
        "charge_factorized_schema_version",
        "normalization_map_version",
        "derivation_status",
        "potential_derivation_version",
        "qed_charge_reference",
    ):
        _nonempty_string(row[field], field, "invalid_row_schema")
    decoded_json = {}
    for field in ("qcd_divisor_label", "qed_divisor_label", "qed_charge_reference"):
        try:
            decoded_json[field] = json.loads(row[field])
        except (TypeError, ValueError) as exc:
            _failure("invalid_row_schema", f"{field} is not valid JSON: {exc}")
    certificate_selected = certificate.get("selected_source_indices")
    certificate_ordered = certificate.get("ordered_source_indices")
    if not isinstance(certificate_selected, list) or not isinstance(certificate_ordered, list):
        _failure("invalid_row_schema", "rank certificate source indices must be lists")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in certificate_selected + certificate_ordered):
        _failure("invalid_row_schema", "rank certificate source indices must be integers")
    if len(set(certificate_ordered)) != len(certificate_ordered):
        _failure("invalid_row_schema", "rank certificate ordered source indices are not unique")
    if not isinstance(certificate.get("selected_rank"), int) or isinstance(certificate.get("selected_rank"), bool):
        _failure("invalid_row_schema", "rank certificate selected_rank must be an integer")
    if certificate["selected_rank"] != len(certificate_selected):
        _failure("invalid_row_schema", "rank certificate selected_rank disagrees with selected indices")
    source_index = row["qed_unsorted_potential_index"]
    if source_index not in certificate_ordered:
        _failure("invalid_row_schema", "rank certificate does not contain the QED source index")
    if row["qed_post_sort_source_position"] != certificate_ordered.index(source_index):
        _failure("invalid_row_schema", "post-sort source position disagrees with rank certificate")
    source_is_selected = source_index in certificate_selected
    if (row["qed_leading_status"] == "leading") != source_is_selected:
        _failure("invalid_row_schema", "leading/dependent status disagrees with selected source indices")
    charge_reference = decoded_json["qed_charge_reference"]
    if not isinstance(charge_reference, Mapping):
        _failure("invalid_row_schema", "qed_charge_reference must decode to an object")
    if charge_reference.get("source_index") != source_index:
        _failure("invalid_row_schema", "charge reference source index disagrees with row")
    if charge_reference.get("source_kind") != row["qed_potential_source"]:
        _failure("invalid_row_schema", "charge reference source kind disagrees with row")
    if charge_reference.get("difference_convention") != PAIRWISE_DIFFERENCE_CONVENTION:
        _failure("invalid_row_schema", "charge reference difference convention is not canonical")
    if charge_reference.get("storage") != "geometry_references_only":
        _failure(
            "invalid_row_schema",
            "charge reference does not point to reference-only geometry storage",
        )
    if charge_reference.get("pair_ordering") != "lexicographic_i_then_j_with_i_less_than_j":
        _failure("invalid_row_schema", "charge reference pair ordering is not canonical")
    if charge_reference.get("pair_charge_coefficients") != "[-1, +1]":
        _failure("invalid_row_schema", "charge reference pair coefficients are not canonical")
    if row["potential_l_convention"] != POTENTIAL_L_CONVENTION:
        _failure("invalid_row_schema", "row L convention is not the existing CYAxiverse convention")
    if row["charge_factorized_schema_version"] != CHARGE_FACTORIZED_SCHEMA_VERSION:
        _failure("invalid_row_schema", "row charge-factorization version is unsupported")
    for field in (
        "assignment_pool_rank",
        "assignment_pool_size",
        "qcd_divisor_index",
        "qed_divisor_index",
        "qed_unsorted_potential_index",
        "qed_post_sort_source_position",
    ):
        _integer(row[field], field, "invalid_row_schema", minimum=0)
    if row["assignment_pool_rank"] >= row["assignment_pool_size"]:
        _failure("invalid_row_schema", "assignment pool rank is outside its pool")
    for field in ("qcd_volume_scale", "qcd_volume", "qed_volume", "qed_log10_lambda4"):
        _finite_float(row[field], field, "invalid_row_schema")
    if not math.isclose(row["qcd_volume"], QCD_VOLUME_TARGET, rel_tol=0.0, abs_tol=1e-9):
        _failure("invalid_row_schema", "qcd_volume is not 40.0")
    if not row["qed_volume"] <= QED_VOLUME_MAX:
        _failure("invalid_row_schema", "qed_volume exceeds the inclusive 127.5 bound")

    if geometry is not None:
        geometry = _mapping(geometry, "geometry", "invalid_geometry_reference")
        if str(geometry.get("geometry_id")) != str(row["geometry_id"]):
            _failure("invalid_geometry_reference", "row geometry_id differs from geometry reference")
        geometry_factorized = geometry.get("charge_factorized_schema_version")
        if (
            geometry_factorized is not None
            and str(geometry_factorized) != row["charge_factorized_schema_version"]
        ):
            _failure(
                "invalid_geometry_reference",
                "row charge-factorization version differs from geometry reference",
            )
    if assignment is not None:
        assignment = _mapping(assignment, "assignment", "missing_assignment_derived_data")
        if assignment.get("geometry_id") is not None and str(assignment["geometry_id"]) != row["geometry_id"]:
            _failure("invalid_geometry_reference", "row assignment points to another geometry")
        for field in (
            "assignment_hash",
            "assignment_pool_rank",
            "assignment_pool_size",
            "qcd_divisor_index",
            "qed_divisor_index",
            "qcd_volume_scale",
            "qcd_volume",
            "qed_volume",
            "normalization_map_version",
        ):
            if field in assignment and assignment[field] is not None:
                expected = _jsonable(assignment[field])
                actual = row[field]
                if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                    if float(actual) != float(expected):
                        _failure("invalid_row_schema", f"row field {field} differs from assignment")
                elif str(actual) != str(expected):
                    _failure("invalid_row_schema", f"row field {field} differs from assignment")
    return True


def row_to_json(row):
    """Return a deterministic JSON representation after schema validation."""
    validate_eft_row(row)
    try:
        return json.dumps(_jsonable(dict(row)), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        _failure("invalid_row_schema", f"row is not JSON serializable: {exc}")


def row_from_json(payload):
    """Decode and validate one JSON row without restoring dense arrays."""
    try:
        row = json.loads(payload)
    except (TypeError, ValueError) as exc:
        _failure("invalid_row_schema", f"row JSON is invalid: {exc}")
    validate_eft_row(row)
    return row


# Short aliases make the integration point easy to discover without creating
# a second implementation or a generator dependency.
build_eft_row = serialize_eft_row
deserialize_eft_row = row_from_json
