"""Trace the retained h11=3 Track-2 losses without rerunning geometry.

This is an artifact-only replay.  It reads the pre-Track-2 diagnostic JSON and
the post-Track-2 JSONL ledger, reconstructs the historical fixed-component
rule from :mod:`compare_fixed_component_sets`, and compares it with the
component records persisted by the generalized implementation.  It never
loads a polytope or enumerates a population.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_fixed_component_sets import (  # noqa: E402
    _component_delta,
    _decode_vector,
    _old_fixed_component_records,
)
import inherited_orientifold_candidates as candidates  # noqa: E402


LOST_WITNESSES = {
    "82:1": "7d78053b848394861324c8752d68f98fef6380b5c4a1894d203a1df923cb436a",
    "86:0": "35b64aa6352f06f3e8a2a2fe2804235d4981eead2e376abfe60c5bc22e92c04d",
    "231:1": "cc5a1a533f4c76cab482b50f385b78a71b45950be01e04cdc3a4a41eec09702b",
}

# This is the only retained 220:1 candidate context in the pre-Track-2
# certified-surface diagnostics.  It is deliberately not treated as a lost
# accepted witness: the old diagnostic status is already unavailable.
UNRESOLVED_220_CANDIDATE = (
    "5f38dd2b55a76a7ada11c3bbabd9f971ddcd8dcc05cbbed4c55717f12868b3f7"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_candidate_contexts(value):
    if isinstance(value, dict):
        if isinstance(value.get("candidate_id"), str):
            yield value
        for child in value.values():
            yield from _walk_candidate_contexts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_candidate_contexts(child)


def _recovered_triangulation_cones(row):
    """Recover original maximal fan cones from auxiliary provenance.

    The ledger's ``pointwise_invariant_cones`` field is derived from
    ``auxiliary_fan``.  It is not the original fan.  Every auxiliary face
    carries its source ``ambient_cones`` provenance, whose union recovers the
    maximal triangulation cones needed by the b191/4f26 baseline.
    """

    triangulation_cones = set()
    for auxiliary_cone in row.get("auxiliary_fan", []):
        for ambient_cone in auxiliary_cone.get("ambient_cones", []):
            triangulation_cones.add(
                tuple(
                    sorted(
                        tuple(int(value) for value in ray)
                        for ray in ambient_cone
                    )
                )
            )
    if not triangulation_cones:
        raise ValueError(
            f"candidate {row.get('candidate_id')} lacks ambient-cone provenance; "
            "the b191/4f26 old component set is unreconstructible"
        )
    return triangulation_cones


def _recovered_fixed_cone_keys(row):
    triangulation_cones = _recovered_triangulation_cones(row)
    matrix = np.asarray(row["lattice_matrix"], dtype=int)
    return tuple(
        tuple(
            tuple(int(value) for value in ray)
            for ray in cone
        )
        for cone in candidates._pointwise_invariant_cone_keys(
            sorted(triangulation_cones), matrix
        )
    )


def _auxiliary_face_keys(row):
    return {
        tuple(
            tuple(int(value) for value in ray)
            for ray in (cone.get("rays", []) if isinstance(cone, dict) else cone)
        )
        for cone in row.get("auxiliary_fan", [])
    }


def _trace_candidate(row, old_context, expected_class):
    class_key = f"{row['polytope_index']}:{row['frst_class_index']}"
    if class_key != expected_class:
        raise ValueError(
            f"candidate {row['candidate_id']} is in {class_key}, expected {expected_class}"
        )
    matrix = np.asarray(row["lattice_matrix"], dtype=int)
    fixed_cone_keys = _recovered_fixed_cone_keys(row)
    auxiliary_face_keys = _auxiliary_face_keys(row)
    old_components = _old_fixed_component_records(
        matrix,
        _decode_vector(row["torus_shift"]),
        int(row["lambda_f"]),
        fixed_cone_keys,
    )
    new_components = row["fixed_component_evidence"]["fixed_point_components"]
    delta = _component_delta(old_components, new_components)
    return {
        "class": expected_class,
        "candidate_id": row["candidate_id"],
        "matrix_id": row["matrix_id"],
        "polytope_index": int(row["polytope_index"]),
        "frst_class_index": int(row["frst_class_index"]),
        "frst_hash": row.get("frst_hash"),
        "lattice_matrix": row["lattice_matrix"],
        "lambda_f": int(row["lambda_f"]),
        "torus_shift": row["torus_shift"],
        "old_terminal_status": old_context.get(
            "candidate_terminal_status", "missing_from_pre_diagnostic"
        ),
        "old_smoothness_reason": (
            "pre-Track-2 diagnostic retained candidate_terminal_status only; "
            "no standalone old smoothness-reason field was persisted"
        ),
        "new_terminal_status": row["terminal_status"],
        "new_smoothness_reason": row.get("terminal_reason"),
        "old_fixed_component_rule": (
            "half-ray integrality for every original-fan pointwise-invariant face, "
            "with exact-nu proper-face containment"
        ),
        "generalized_fixed_component_rule": (
            "smooth-half-ray shortcut only with a retained proof; otherwise "
            "quotient-lattice integrality modulo span_Q(sigma), canonical phase "
            "reduction modulo span_Q(sigma), and proper-face containment"
        ),
        "pointwise_invariant_cone_rays": [
            [list(ray) for ray in cone]
            for cone in fixed_cone_keys
        ],
        "fixed_cone_reconstruction": {
            "source": "union(auxiliary_fan[*].ambient_cones)",
            "recovered_triangulation_cone_count": len(
                _recovered_triangulation_cones(row)
            ),
            "recovered_pointwise_face_count": len(fixed_cone_keys),
            "auxiliary_face_count": len(auxiliary_face_keys),
            "recovered_triangulation_cones_differ_from_auxiliary_faces": (
                _recovered_triangulation_cones(row) != auxiliary_face_keys
            ),
            "recovered_pointwise_faces_differ_from_auxiliary_faces": (
                set(fixed_cone_keys) != auxiliary_face_keys
            ),
        },
        "old_fixed_components": old_components,
        "new_fixed_components": new_components,
        "component_delta": delta,
        "acceptance_delta": {
            "old_status": old_context.get("candidate_terminal_status"),
            "new_status": row["terminal_status"],
            "changed": old_context.get("candidate_terminal_status")
            != row["terminal_status"],
            "cause": (
                "the generalized enumeration adds or retains quotient-lattice "
                "fixed components that are not present in the old half-ray set; "
                "their smoothness evidence is unavailable"
            ),
        },
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-json", type=Path, required=True)
    parser.add_argument("--post-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def main(argv=None):
    args = _parse_args(argv)
    pre = json.loads(args.pre_json.read_text())
    pre_contexts = {
        context["candidate_id"]: context
        for context in _walk_candidate_contexts(pre.get("details", []))
    }

    needed_ids = set(LOST_WITNESSES.values()) | {UNRESOLVED_220_CANDIDATE}
    post_rows = {}
    with args.post_jsonl.open() as handle:
        for line in handle:
            row = json.loads(line)
            candidate_id = row.get("candidate_id")
            if candidate_id in needed_ids:
                post_rows[candidate_id] = row

    missing_post = sorted(needed_ids - set(post_rows))
    if missing_post:
        raise ValueError(f"candidate IDs missing from post ledger: {missing_post}")

    traces = []
    for class_key, candidate_id in LOST_WITNESSES.items():
        if candidate_id not in pre_contexts:
            raise ValueError(f"accepted witness missing from pre diagnostics: {candidate_id}")
        trace = _trace_candidate(post_rows[candidate_id], pre_contexts[candidate_id], class_key)
        if trace["old_terminal_status"] != "accepted_verified_orientifold":
            raise ValueError(
                f"retained old witness {candidate_id} is not marked accepted: "
                f"{trace['old_terminal_status']}"
            )
        traces.append(trace)

    unresolved_context = pre_contexts.get(UNRESOLVED_220_CANDIDATE)
    unresolved_trace = _trace_candidate(
        post_rows[UNRESOLVED_220_CANDIDATE],
        unresolved_context or {},
        "220:1",
    )
    if not any(
        trace["fixed_cone_reconstruction"][
            "recovered_triangulation_cones_differ_from_auxiliary_faces"
        ]
        for trace in traces + [unresolved_trace]
    ):
        raise ValueError(
            "bounded target trace did not demonstrate that maximal triangulation "
            "cones differ from auxiliary faces"
        )
    pre_220_contexts = [
        context
        for detail in pre.get("details", [])
        if int(detail.get("polytope_index", -1)) == 220
        for context in _walk_candidate_contexts(detail)
    ]
    accepted_220 = [
        context
        for context in pre_220_contexts
        if context.get("candidate_terminal_status") == "accepted_verified_orientifold"
    ]
    pre_220_details = [
        detail
        for detail in pre.get("details", [])
        if int(detail.get("polytope_index", -1)) == 220
    ]
    pre_220_accepted_classes = sorted(
        {
            int(class_index)
            for detail in pre_220_details
            for class_index in detail.get("orientifold_action_audit", {}).get(
                "h11_minus_zero_classes", []
            )
        }
    )

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    report = {
        "schema_version": "cyaxiverse-track2-lost-h11-3-trace-1.0",
        "claim_status": (
            "evidence_trace_only; this report does not decide whether the generalized "
            "scientific rule should be accepted"
        ),
        "scope": {
            "target_classes": ["82:1", "86:0", "220:1", "231:1"],
            "h11": 3,
            "geometry_rerun": False,
            "population_rerun": False,
            "h11_4_rerun": False,
            "inputs": "retained pre-Track-2 JSON and post-Track-2 JSONL rows only",
        },
        "provenance": {
            "source_commit": source_commit,
            "package_version": "0.2.0 (Project.toml)",
            "version_impact": (
                "scientific fixed-locus behavior change; pre-1.0 minor bump required "
                "at the reviewed release boundary, deferred on this feature branch"
            ),
            "git_dirty": bool(dirty),
            "dirty_paths_at_generation": dirty,
            "pre_json": {
                "path": str(args.pre_json),
                "sha256": _sha256(args.pre_json),
                "source_commit_in_artifact": pre.get("run_provenance", {}).get(
                    "source_commit"
                ),
                "accepted_count": pre.get("counts", {}).get(
                    "source_evidence_inherited_orientifold_cys"
                ),
            },
            "post_jsonl": {
                "path": str(args.post_jsonl),
                "sha256": _sha256(args.post_jsonl),
            },
            "post_summary": (
                {
                    "path": str(args.post_jsonl) + ".summary.json",
                    "sha256": _sha256(Path(str(args.post_jsonl) + ".summary.json")),
                    "accepted_class_count": sum(
                        bool(item.get("accepted_for_table_1"))
                        for item in json.loads(
                            Path(str(args.post_jsonl) + ".summary.json").read_text()
                        ).get("class_funnel", [])
                    ),
                }
                if Path(str(args.post_jsonl) + ".summary.json").exists()
                else None
            ),
        },
        "source_rule_trace": {
            "historical_rule": (
                "Track-2 baseline from b191/4f26: eq. (4.35) half-ray integrality "
                "on original-fan pointwise-invariant faces, followed by exact-nu "
                "proper-face containment"
            ),
            "generalized_rule": (
                "eq. (4.30) quotient-lattice integrality for non-smooth sigma; "
                "phase labels reduced modulo span_Q(sigma); contained components "
                "removed in the proper-face direction"
            ),
            "implementation_files": [
                "scripts/compare_fixed_component_sets.py::_old_fixed_component_records",
                "scripts/inherited_orientifold_candidates.py::_fixed_component_records",
            ],
            "reconstructibility_boundary": (
                "The post ledger does not directly persist original-fan pointwise-"
                "invariant keys. For these rows, the baseline is recoverable by taking "
                "the union of auxiliary_fan[*].ambient_cones, recovering maximal "
                "triangulation cones, and recomputing those keys exactly; the report "
                "records that recovered set and the maximal-cone/auxiliary-face "
                "difference. The pre diagnostic records source_commit=e7a8f51 "
                "with a dirty tree and does not preserve the complete producing tree; "
                "its old candidate status fields are evidence, not a byte-for-byte "
                "source reconstruction."
            ),
        },
        "lost_accepted_witnesses": traces,
        "unresolved_220_1": {
            "candidate_id_checked": UNRESOLVED_220_CANDIDATE,
            "candidate_class": "220:1",
            "old_status_in_retained_pre_diagnostic": unresolved_trace[
                "old_terminal_status"
            ],
            "new_status_in_post_ledger": unresolved_trace["new_terminal_status"],
            "old_smoothness_reason": unresolved_trace["old_smoothness_reason"],
            "new_smoothness_reason": unresolved_trace["new_smoothness_reason"],
            "component_trace": unresolved_trace,
            "retained_evidence_checked": [
                "pre JSON details[*].orientifold_action_audit.reason_diagnostics, "
                "including certified_surfaces[*].candidate_context",
                "pre JSON details entries with polytope_index=220",
                "post JSONL candidate row for the exact candidate ID",
                "post Track-2 summary class funnel, which marks 220:1 not accepted",
            ],
            "pre_220_candidate_context_count": len(pre_220_contexts),
            "pre_220_accepted_contexts_found": len(accepted_220),
            "pre_class_level_accepted": 1 in pre_220_accepted_classes,
            "pre_class_level_accepted_classes": pre_220_accepted_classes,
            "discrepancy": (
                "The retained pre artifact proves class-level acceptance for 220:1, "
                "but its diagnostic payload has no accepted candidate context for "
                "that class. Candidate 5f38... is present, but its old status is "
                "already smoothness_verification_unavailable, so it cannot identify "
                "the lost accepted witness."
            ),
            "additional_artifact_needed": (
                "The original pre-Track-2 candidate-level ledger (or a source-faithful "
                "pre-rule replay) containing every 220:1 candidate ID, lambda_f, torus "
                "shift, old fixed-component records, and old terminal/smoothness status. "
                "Do not infer this witness from the retained summary diagnostics."
            ),
        },
        "reproducibility": {
            "trace_command": (
                "cd /Users/vmehta/Documents/CYAxiverse/cyaxiverse/"
                "CYAxiverse-orientifold-overcount && PYTHONPATH=scripts conda run "
                "--no-capture-output -n cytools python scripts/"
                "trace_lost_h11_3_fixed_components.py "
                "--pre-json /private/tmp/cyax-orientifold-rerun-h11-3-20260820.json "
                "--post-jsonl /private/tmp/cyax-orientifold-ledger-h11-3-20260820.jsonl "
                "--output /Users/vmehta/Documents/CYAxiverse/cyaxiverse/"
                "handoffs_checkpoints/fuzzy_axions_orientifold_track_2_lost_h11_3_"
                "trace_20260820.json"
            ),
            "focused_test_command": (
                "KMP_DUPLICATE_LIB_OK=TRUE NUMBA_CACHE_DIR=/private/tmp/"
                "cyax-track2-tests-numba OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                "MKL_NUM_THREADS=1 NUMBA_DISABLE_CACHING=1 PYTHONPATH=scripts "
                "conda run --no-capture-output -n cytools python -m unittest "
                "scripts.test_inherited_orientifold_candidates."
                "FixedComponentContainmentReductionTests"
            ),
            "verification_boundary": (
                "The trace command reads retained artifacts only; it does not invoke "
                "CYTools geometry, h11=4, or a population rerun."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
