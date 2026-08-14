"""Fresh-CYTools ensemble manifest contract for the Glimmers proof of principle.

This helper records selection and reproducibility metadata without deciding
that a fresh finite ensemble is historical, uniform, representative, or an
exact reproduction of the paper.  It is intentionally independent of the
geometry generator and may be used by the later integration assignment.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

try:
    from glimmers_provenance import (
        PROVENANCE_SCHEMA_VERSION,
        ProvenanceError,
        TERMINAL_STATUSES,
        sha256_file,
        stable_digest,
        stable_seed,
        validate_provenance_digest,
    )
except ImportError:  # pragma: no cover - supports package-style imports.
    from .glimmers_provenance import (
        PROVENANCE_SCHEMA_VERSION,
        ProvenanceError,
        TERMINAL_STATUSES,
        sha256_file,
        stable_digest,
        stable_seed,
        validate_provenance_digest,
    )


MANIFEST_SCHEMA_VERSION = "glimmers-fresh-ensemble-manifest-1.1"
H11_VALUES = (50, 100, 200, 491)
DEFAULT_POLYTOPE_COUNTS = {"50": 50, "100": 50, "200": 30, "491": 1}
ACCEPTED_ROW_RANGE = {"minimum": 100_000, "maximum": 200_000}
HISTORICAL_MAPPING_STATUSES = frozenset({"not_attempted_by_policy", "no_match_claim"})
CLAIM_LABELS = (
    "fresh_favorable_cytools_proof_of_principle",
    "adapted_model_reuse",
    "no_historical_polytope_match_claim",
    "no_exact_200000_reproduction_claim",
)
NATIVE_H491_ARGUMENTS = {
    "N": 100,
    "make_star": True,
    "triang_method": "fast",
    "max_npts": 17,
    "N_face_triangs": 1000,
    "as_generator": True,
    "heights_only": False,
    "backend": "cgal",
}
FORBIDDEN_SAMPLER_TERMS = frozenset(
    {"historical", "representative", "uniform", "complete", "exact_reproduction"}
)


def _fail(status: str, message: str, *, details=None):
    raise ProvenanceError(status, message, details=details)


def _copy(value):
    return copy.deepcopy(value)


def _source_manifest_digest(source_manifest):
    """Return a stable hash for a source-manifest path or JSON value."""
    if source_manifest is None:
        return None
    if isinstance(source_manifest, (str, os.PathLike, Path)):
        return sha256_file(source_manifest)
    return stable_digest(source_manifest)


def default_sampler_by_h11():
    """Return the approved per-h11 sampler names and immutable controls."""
    return {
        "50": {
            "name": "Polytope.random_triangulations_fast",
            "purpose_label": "biased_fast_reference",
            "arguments": {
                "N": 10,
                "c": 0.2,
                "max_retries": 500,
                "make_star": True,
                "only_fine": True,
                "backend": "cgal",
                "as_list": False,
                "progress_bar": False,
            },
            "completeness_status": "not_claimed",
            "historical_equivalence_status": "not_claimed",
        },
        "100": {
            "name": "Polytope.random_triangulations_fast",
            "purpose_label": "biased_fast_reference",
            "arguments": {
                "N": 10,
                "c": 0.2,
                "max_retries": 500,
                "make_star": True,
                "only_fine": True,
                "backend": "cgal",
                "as_list": False,
                "progress_bar": False,
            },
            "completeness_status": "not_claimed",
            "historical_equivalence_status": "not_claimed",
        },
        "200": {
            "name": "Polytope.random_triangulations_fast",
            "purpose_label": "biased_fast_reference",
            "arguments": {
                "N": 10,
                "c": 0.2,
                "max_retries": 500,
                "make_star": True,
                "only_fine": True,
                "backend": "cgal",
                "as_list": False,
                "progress_bar": False,
            },
            "completeness_status": "not_claimed",
            "historical_equivalence_status": "not_claimed",
        },
        "491": {
            "name": "Polytope.ntfe_frts",
            "purpose_label": "package_native_ntfe_fast_reference",
            "arguments": {
                "N": 100,
                "make_star": True,
                "triang_method": "fast",
                "max_npts": 17,
                "N_face_triangs": 1000,
                "as_generator": True,
                "heights_only": False,
                "backend": "cgal",
            },
            "completeness_status": "not_claimed",
            "historical_equivalence_status": "not_claimed",
        },
    }


def approved_parity_convention():
    """Return the user-approved identity O3/O7 and full CYTools basis rule."""
    identity = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    return {
        "involution_type": "O3/O7",
        "identity_involution": True,
        "lattice_matrix": identity,
        "h11_plus": "h11",
        "h11_minus": 0,
        "equations": {"h11_plus": "h11_plus=h11", "h11_minus": "h11_minus=0"},
        "axion_basis_convention": "user-approved-full-cytools-h11-c4-even",
        "basis_statement": (
            "Use the full CYTools h11-dimensional basis as the C4-axion sector; "
            "apply no additional parity projection."
        ),
        "provenance": (
            "User-approved convention; the identity encodes the run convention "
            "and is not an independently inferred physical orientifold."
        ),
    }


def _normalise_query(source_query):
    if not isinstance(source_query, Mapping) or not source_query:
        _fail("source_query_unrecorded", "fresh-ensemble source query is missing")
    query = _copy(source_query)
    criteria = query.get("query_criteria", query.get("criteria"))
    if not isinstance(criteria, Mapping):
        _fail(
            "source_query_unrecorded",
            "source query must record lattice, favorable, reflexive, and full-dimensional criteria",
        )
    required = ("lattice", "favorable", "reflexive", "full_dimensional")
    missing = [key for key in required if key not in criteria]
    if missing:
        _fail(
            "source_query_unrecorded",
            f"source query is missing criteria: {', '.join(missing)}",
            details={"missing": missing},
        )
    if criteria["lattice"] != "N" or any(criteria[key] is not True for key in required[1:]):
        _fail(
            "source_query_unrecorded",
            "source query criteria must use N-lattice and boolean favorable/reflexive/full-dimensional flags",
        )
    if query.get("source") is None and query.get("source_identity") is None:
        _fail("source_query_unrecorded", "source query must record its CYTools source identity")
    if query.get("fresh") is not True or criteria.get("fresh") is False:
        _fail("source_query_unrecorded", "source query must use a fresh selection")
    if not isinstance(query.get("result_count"), int) or query["result_count"] < 0:
        _fail("source_query_unrecorded", "source query must record a non-negative result count")
    returned_order = query.get("returned_order")
    if not isinstance(returned_order, Sequence) or isinstance(returned_order, (str, bytes)):
        _fail("source_query_unrecorded", "source query must record the complete returned order")
    if len(returned_order) != query["result_count"]:
        _fail("source_query_unrecorded", "source-query result count and returned order differ")
    revision = query.get("source_revision") or query.get("revision")
    mirror_hash = (
        query.get("local_mirror_hash")
        or query.get("source_manifest_sha256")
        or query.get("local_mirror_manifest_sha256")
    )
    if not revision and not mirror_hash:
        _fail(
            "source_query_unrecorded",
            "source query must record a source revision or local mirror hash",
        )
    query["query_criteria"] = dict(criteria)
    query["source_revision"] = None if not revision else str(revision)
    query["local_mirror_hash"] = None if not mirror_hash else str(mirror_hash)
    if query.get("source_manifest_sha256"):
        digest = str(query["source_manifest_sha256"])
        if len(digest) != 64:
            _fail("missing_input_hash", "source manifest hash must be a SHA-256 digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            _fail("missing_input_hash", "source manifest hash is not hexadecimal")
            raise AssertionError from exc
    query["recorded"] = True
    return query


def _normalise_polytopes(retained_polytopes):
    if not isinstance(retained_polytopes, Sequence) or isinstance(retained_polytopes, (str, bytes)):
        _fail("source_query_unrecorded", "retained polytopes must be an ordered sequence")
    result = []
    seen = set()
    for returned_index, item in enumerate(retained_polytopes):
        if not isinstance(item, Mapping):
            _fail("source_query_unrecorded", "each retained polytope must be a mapping")
        for forbidden_key in ("historical_id", "historical_polytope_id", "historical_match"):
            if item.get(forbidden_key) not in (None, False, ""):
                _fail(
                    "historical_match_claim_forbidden",
                    f"historical mapping field is forbidden: {forbidden_key}",
                )
        fingerprint = (
            item.get("polytope_fingerprint")
            or item.get("fingerprint")
            or item.get("polytope_id")
        )
        if not fingerprint:
            _fail("source_query_unrecorded", "every retained polytope needs a stable fingerprint")
        fingerprint = str(fingerprint)
        if "historical" in fingerprint.lower():
            _fail(
                "historical_match_claim_forbidden",
                "a historical-looking polytope identifier cannot be recorded as a fresh fingerprint",
            )
        try:
            h11 = int(item["h11"])
        except (KeyError, TypeError, ValueError) as exc:
            _fail("source_query_unrecorded", "every retained polytope needs an integer h11")
            raise AssertionError from exc
        if h11 not in H11_VALUES:
            _fail("source_query_unrecorded", f"unsupported fresh-ensemble h11={h11}")
        order = item.get("retained_order", item.get("polytope_ordinal", returned_index))
        try:
            order = int(order)
        except (TypeError, ValueError) as exc:
            _fail("source_query_unrecorded", "retained polytope order must be an integer")
            raise AssertionError from exc
        if fingerprint in seen:
            _fail("source_query_unrecorded", f"duplicate retained polytope fingerprint: {fingerprint}")
        seen.add(fingerprint)
        record = _copy(item)
        historical_status = item.get("historical_mapping_status", "not_attempted_by_policy")
        if historical_status not in HISTORICAL_MAPPING_STATUSES:
            _fail(
                "historical_match_claim_forbidden",
                "historical mapping status must be explicit and non-identifying",
            )
        record.update(
            {
                "h11": h11,
                "returned_index": returned_index,
                "retained_order": order,
                "polytope_fingerprint": fingerprint,
                "historical_mapping_status": historical_status,
            }
        )
        record.pop("historical_id", None)
        record.pop("historical_polytope_id", None)
        record.pop("historical_match", None)
        result.append(record)
    return result


def _normalise_samplers(sampler_by_h11):
    supplied = default_sampler_by_h11() if sampler_by_h11 is None else sampler_by_h11
    if not isinstance(supplied, Mapping):
        _fail("source_query_unrecorded", "sampler_by_h11 must be a mapping")
    result = {}
    for h11 in H11_VALUES:
        key = str(h11)
        value = supplied.get(key, supplied.get(h11))
        if not isinstance(value, Mapping):
            _fail("source_query_unrecorded", f"sampler metadata is missing for h11={h11}")
        record = _copy(value)
        name = record.get("name") or record.get("sampler") or record.get("sampler_name")
        arguments = record.get("arguments", record.get("args"))
        if not name or not isinstance(arguments, Mapping):
            _fail("source_query_unrecorded", f"sampler metadata is incomplete for h11={h11}")
        name_aliases = {
            "fast": "Polytope.random_triangulations_fast",
            "ntfe_frts": "Polytope.ntfe_frts",
        }
        record["name"] = name_aliases.get(str(name), str(name))
        record["arguments"] = dict(arguments)
        record.setdefault("purpose_label", "fresh_favorable_proof_of_principle")
        record.setdefault("completeness_status", "not_claimed")
        record.setdefault("historical_equivalence_status", "not_claimed")
        if h11 == 491:
            if record["name"] != "Polytope.ntfe_frts" or record["arguments"].get("triang_method") != "fast":
                _fail(
                    "user_decision_required",
                    "h11=491 must record the native Polytope.ntfe_frts fast sampler",
                )
            mismatches = {
                key_name: {"expected": expected, "actual": record["arguments"].get(key_name)}
                for key_name, expected in NATIVE_H491_ARGUMENTS.items()
                if record["arguments"].get(key_name) != expected
            }
            if mismatches:
                _fail(
                    "user_decision_required",
                    "h11=491 native ntfe_frts fast settings changed",
                    details={"mismatches": mismatches},
                )
        elif record["name"] != "Polytope.random_triangulations_fast":
            _fail(
                "user_decision_required",
                f"h11={h11} must record Polytope.random_triangulations_fast",
            )
        if record["completeness_status"] != "not_claimed":
            _fail(
                "historical_match_claim_forbidden",
                f"sampler completeness status must remain not_claimed for h11={h11}",
            )
        if record["historical_equivalence_status"] != "not_claimed":
            _fail(
                "historical_match_claim_forbidden",
                f"sampler historical-equivalence status must remain not_claimed for h11={h11}",
            )
        claim_text = " ".join(
            str(record.get(key, ""))
            for key in ("name", "purpose_label")
        ).lower()
        if any(term in claim_text for term in FORBIDDEN_SAMPLER_TERMS):
            _fail(
                "historical_match_claim_forbidden",
                f"sampler labels contain a forbidden population claim for h11={h11}",
            )
        result[key] = record
    return result


def seed_record(base_seed, *, scope, h11=None, polytope_ordinal=None, sampler_name=None, index=None):
    """Derive and record one seed with all inputs and its derivation rule."""
    parts = ["glimmers-fresh-ensemble", int(base_seed), str(scope)]
    if h11 is not None:
        parts.append(int(h11))
    if polytope_ordinal is not None:
        parts.append(int(polytope_ordinal))
    if sampler_name is not None:
        parts.append(str(sampler_name))
    if index is not None:
        parts.append(int(index))
    return {
        "scope": str(scope),
        "h11": None if h11 is None else int(h11),
        "polytope_ordinal": None if polytope_ordinal is None else int(polytope_ordinal),
        "sampler_name": None if sampler_name is None else str(sampler_name),
        "index": None if index is None else int(index),
        "seed": stable_seed(*parts),
        "derivation": "stable_sha256(glimmers-fresh-ensemble, base_seed, scope, h11, polytope_ordinal, sampler_name, index)",
        "derivation_inputs": parts,
    }


def derive_ensemble_seeds(base_seed, retained_polytopes, sampler_by_h11=None):
    """Derive the deterministic per-polytope/per-sampler seeds for the manifest."""
    polytopes = _normalise_polytopes(retained_polytopes)
    samplers = _normalise_samplers(sampler_by_h11)
    return [
        seed_record(
            base_seed,
            scope="polytope_sampler",
            h11=item["h11"],
            polytope_ordinal=item["retained_order"],
            sampler_name=samplers[str(item["h11"])]["name"],
        )
        for item in polytopes
    ]


def _normalise_seeds(derived_seeds):
    if derived_seeds is None:
        _fail("missing_input_hash", "derived seeds must be explicitly recorded")
    if isinstance(derived_seeds, Mapping):
        records = []
        for name, value in derived_seeds.items():
            if isinstance(value, Mapping):
                record = _copy(value)
                record.setdefault("scope", str(name))
            else:
                record = {"scope": str(name), "seed": value, "derivation": "caller-supplied"}
            records.append(record)
    elif isinstance(derived_seeds, Sequence) and not isinstance(derived_seeds, (str, bytes)):
        records = [_copy(item) for item in derived_seeds]
    else:
        _fail("missing_input_hash", "derived seeds must be a sequence or mapping")
    if not records:
        _fail("missing_input_hash", "derived seed record is empty")
    seen = set()
    for record in records:
        if not isinstance(record, Mapping) or record.get("seed") is None or not record.get("derivation"):
            _fail("missing_input_hash", "each derived seed needs a value and derivation string")
        if not isinstance(record.get("seed"), int) or isinstance(record.get("seed"), bool):
            _fail("missing_input_hash", "each derived seed must be an integer")
        key = (
            record.get("scope"),
            record.get("h11"),
            record.get("polytope_ordinal"),
            record.get("sampler_name"),
            record.get("index"),
        )
        if key in seen:
            _fail("missing_input_hash", f"duplicate derived seed identity: {key}")
        seen.add(key)
    return records


def _normalise_parity(parity_convention):
    parity = approved_parity_convention() if parity_convention is None else _copy(parity_convention)
    if parity.get("involution_type") != "O3/O7" or parity.get("identity_involution") is not True:
        _fail("user_decision_required", "only the approved identity O3/O7 convention is allowed")
    if parity.get("h11_plus") != "h11" or parity.get("h11_minus") != 0:
        _fail("user_decision_required", "parity metadata must preserve h11_plus=h11 and h11_minus=0")
    if parity.get("axion_basis_convention") != "user-approved-full-cytools-h11-c4-even":
        _fail("user_decision_required", "the full CYTools axion-basis convention must remain explicit")
    return parity


def _claim_record():
    return {
        "claim_labels": list(CLAIM_LABELS),
        "ensemble": "fresh_favorable_not_historical_not_uniform_not_representative",
        "algorithm": "adapted_model_reuse_proof_of_principle",
        "historical_mapping_status": "not_attempted_by_policy",
        "historical_polytope_claim": "no_historical_polytope_match_claim",
        "paper_reproduction_status": "no_exact_200000_reproduction_claim",
        "completeness_status": "not_claimed",
        "uniformity_status": "not_claimed",
        "representativeness_status": "not_claimed",
    }


def _validate_claims(claims):
    if not isinstance(claims, Mapping) or list(claims.get("claim_labels", ())) != list(CLAIM_LABELS):
        _fail("historical_match_claim_forbidden", "manifest claim labels do not match the approved fixed set")
    if claims.get("historical_mapping_status") not in HISTORICAL_MAPPING_STATUSES:
        _fail("historical_match_claim_forbidden", "historical mapping status must be explicit and non-identifying")
    if claims.get("historical_polytope_claim") != "no_historical_polytope_match_claim":
        _fail("historical_match_claim_forbidden", "historical polytope matching is forbidden by policy")
    if claims.get("paper_reproduction_status") != "no_exact_200000_reproduction_claim":
        _fail("historical_match_claim_forbidden", "exact 200000 reproduction wording is forbidden by policy")
    for key in ("completeness_status", "uniformity_status", "representativeness_status"):
        if claims.get(key) != "not_claimed":
            _fail("historical_match_claim_forbidden", f"{key} must remain not_claimed")
    return True


def _normalise_row_accounting(accepted_row_count):
    try:
        count = int(accepted_row_count)
    except (TypeError, ValueError) as exc:
        _fail("user_decision_required", "accepted row count must be an integer")
        raise AssertionError from exc
    if count < 0 or count > ACCEPTED_ROW_RANGE["maximum"]:
        _fail(
            "user_decision_required",
            "accepted row count must be between zero and the declared maximum",
        )
    minimum = ACCEPTED_ROW_RANGE["minimum"]
    return {
        "declared_accepted_row_range": dict(ACCEPTED_ROW_RANGE),
        "accepted_row_count": count,
        "range_status": "within_declared_range" if count >= minimum else "below_declared_range",
        "paper_count_mapping_status": "not_exact_reproduction",
    }


def _validate_provenance(provenance):
    if not isinstance(provenance, Mapping):
        _fail("missing_input_hash", "a clean provenance record is required")
    validate_provenance_digest(provenance)
    repository = provenance.get("repository", {})
    if repository.get("status") != "clean" or repository.get("status_porcelain"):
        _fail("provenance_dirty_tree", "fresh ensemble cannot be built from a dirty worktree")
    task_file = provenance.get("task_file", {})
    if not task_file.get("sha256"):
        _fail("missing_input_hash", "task-file hash is missing from provenance")
    if not provenance.get("source_hashes"):
        _fail("missing_input_hash", "source hashes are missing from provenance")
    source_query = provenance.get("source_query")
    if not source_query:
        _fail("source_query_unrecorded", "source query is missing from provenance")
    roots = provenance.get("roots", {})
    output_root = roots.get("output_root")
    if not output_root:
        _fail("output_collision", "provenance must record an output root")
    if Path(output_root).exists():
        _fail("output_collision", f"production output root already exists: {output_root}")
    return True


def _manifest_without_digest(manifest):
    return {key: value for key, value in manifest.items() if key != "manifest_digest"}


def build_fresh_ensemble_manifest(
    provenance,
    *,
    source_query=None,
    retained_polytopes,
    base_seed,
    derived_seeds,
    source_manifest=None,
    cytools_version=None,
    sampler_by_h11=None,
    parity_convention=None,
    accepted_row_count=0,
    stop_reason="provenance gate passed; production output not generated by this helper",
    status=None,
    historical_mapping_status="not_attempted_by_policy",
    expected_polytope_counts=None,
    run_metadata=None,
):
    """Build a validated fresh-ensemble manifest without creating bulk output."""
    _validate_provenance(provenance)
    if source_query is None:
        source_query = provenance.get("source_query")
    query = _normalise_query(source_query)
    source_manifest_hash = _source_manifest_digest(source_manifest)
    recorded_manifest_hash = query.get("source_manifest_sha256")
    if recorded_manifest_hash and source_manifest_hash and recorded_manifest_hash != source_manifest_hash:
        _fail("missing_input_hash", "source manifest hash does not match the supplied manifest")
    if source_manifest_hash:
        query["source_manifest_sha256"] = source_manifest_hash
    elif recorded_manifest_hash:
        query["source_manifest_sha256"] = str(recorded_manifest_hash)
    if cytools_version is None:
        cytools_version = provenance.get("environment_versions", {}).get("cytools", "unavailable")
    if not isinstance(cytools_version, str) or not cytools_version.strip():
        _fail("missing_input_hash", "CYTools version must be recorded")
    polytopes = _normalise_polytopes(retained_polytopes)
    samplers = _normalise_samplers(sampler_by_h11)
    seeds = _normalise_seeds(derived_seeds)
    parity = _normalise_parity(parity_convention)
    if historical_mapping_status not in HISTORICAL_MAPPING_STATUSES:
        _fail("historical_match_claim_forbidden", "historical mapping status is not an allowed explicit status")
    rows = _normalise_row_accounting(accepted_row_count)
    expected = DEFAULT_POLYTOPE_COUNTS if expected_polytope_counts is None else {
        str(key): int(value) for key, value in expected_polytope_counts.items()
    }
    retained_counts = {str(h11): 0 for h11 in H11_VALUES}
    for item in polytopes:
        retained_counts[str(item["h11"])] += 1
    source_shortfall = any(retained_counts[key] < expected.get(key, 0) for key in retained_counts)
    if status is None:
        status = "fresh_source_shortfall" if source_shortfall else "provenance_validated"
    if status not in TERMINAL_STATUSES:
        _fail("user_decision_required", f"unknown manifest terminal status: {status}")
    if not isinstance(stop_reason, str) or not stop_reason.strip():
        _fail("user_decision_required", "the manifest must record the actual stop reason")
    claims = _claim_record()
    claims["historical_mapping_status"] = historical_mapping_status
    _validate_claims(claims)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": status,
        "stop_reason": stop_reason,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "provenance_digest": provenance["provenance_digest"],
        "provenance": _copy(provenance),
        "cytools_version": str(cytools_version),
        "source_query": query,
        "selection": {
            "ensemble_kind": "fresh_favorable_cytools",
            "fresh": True,
            "lattice": "N",
            "favorable": True,
            "reflexive": True,
            "full_dimensional": True,
            "selection_route": "retain the requested prefix of the deterministic source-query order",
            "uniformity_status": "not_claimed",
            "representativeness_status": "not_claimed",
            "expected_polytope_counts": expected,
            "retained_polytope_counts": retained_counts,
            "retained_polytopes_in_source_order": polytopes,
        },
        "sampler_by_h11": samplers,
        "seed_policy": {
            "base_seed": int(base_seed),
            "derivation": "Every derived seed is recorded with its scope, inputs, and stable SHA-256 derivation.",
            "derived_seeds": seeds,
        },
        "parity_convention": parity,
        "row_accounting": rows,
        "claims": claims,
        "terminal_statuses": sorted(TERMINAL_STATUSES),
        "output_policy": {
            "output_root": provenance["roots"]["output_root"],
            "must_be_fresh": True,
            "no_bulk_output_created_by_helper": True,
            "no_overwrite": True,
        },
        "run_metadata": {} if run_metadata is None else _copy(run_metadata),
    }
    manifest["manifest_digest"] = stable_digest(manifest)
    validate_fresh_ensemble_manifest(manifest)
    return manifest


def manifest_digest(manifest) -> str:
    """Return the digest of a manifest excluding its self-digest field."""
    return stable_digest(_manifest_without_digest(manifest))


def validate_fresh_ensemble_manifest(manifest):
    """Validate claim boundaries and reproducibility fields of one manifest."""
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        _fail("user_decision_required", "manifest schema version is unsupported")
    if not isinstance(manifest.get("cytools_version"), str) or not manifest["cytools_version"].strip():
        _fail("missing_input_hash", "manifest CYTools version is missing")
    recorded = manifest.get("manifest_digest")
    if not recorded or recorded != manifest_digest(manifest):
        _fail("missing_input_hash", "manifest digest is absent or does not match its contents")
    _validate_provenance(manifest.get("provenance"))
    if manifest.get("provenance_digest") != manifest["provenance"].get("provenance_digest"):
        _fail("missing_input_hash", "manifest and nested provenance digests disagree")
    _normalise_query(manifest.get("source_query"))
    _normalise_polytopes(manifest.get("selection", {}).get("retained_polytopes_in_source_order", []))
    _normalise_samplers(manifest.get("sampler_by_h11"))
    _normalise_parity(manifest.get("parity_convention"))
    _validate_claims(manifest.get("claims"))
    rows = manifest.get("row_accounting", {})
    _normalise_row_accounting(rows.get("accepted_row_count"))
    if not isinstance(manifest.get("stop_reason"), str) or not manifest["stop_reason"].strip():
        _fail("user_decision_required", "manifest stop reason is missing")
    if manifest.get("output_policy", {}).get("must_be_fresh") is not True:
        _fail("output_collision", "manifest must require a fresh output root")
    return True


def write_fresh_ensemble_manifest(path, manifest):
    """Atomically write one manifest and refuse to overwrite an existing file."""
    validate_fresh_ensemble_manifest(manifest)
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        _fail("output_collision", f"manifest output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
        if destination.exists():
            _fail("output_collision", f"manifest output appeared during atomic write: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(destination)


__all__ = [
    "ACCEPTED_ROW_RANGE",
    "CLAIM_LABELS",
    "H11_VALUES",
    "HISTORICAL_MAPPING_STATUSES",
    "MANIFEST_SCHEMA_VERSION",
    "NATIVE_H491_ARGUMENTS",
    "approved_parity_convention",
    "build_fresh_ensemble_manifest",
    "default_sampler_by_h11",
    "derive_ensemble_seeds",
    "manifest_digest",
    "seed_record",
    "validate_fresh_ensemble_manifest",
    "write_fresh_ensemble_manifest",
]
