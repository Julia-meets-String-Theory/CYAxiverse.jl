"""Test the artifact-only Table 1 gap classifier."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analyze_fuzzy_axions_orientifold_gap as gap
from orientifold_terminal_ledger import TerminalLedgerWriter


H11_2 = Path("/private/tmp/cyax-orientifold-rerun-h11-2-20260820.json")
H11_3 = Path("/private/tmp/cyax-orientifold-rerun-h11-3-20260820.json")


def _detail(
    polytope_index=0,
    class_count=1,
    accepted=None,
    attempts=None,
    unresolved=None,
):
    return {
        "polytope_index": polytope_index,
        "frst_class_count": class_count,
        "orientifold_action_audit": {
            "h11_minus_zero_classes": list(accepted or []),
            "reason_diagnostics": {
                "surface_attempts": list(attempts or []),
                "unresolved_components": list(unresolved or []),
            },
        },
    }


def _provenance():
    return {
        "source_commit": "fixture-commit",
        "git_dirty": False,
        "runtime_versions": {"python": "fixture"},
        "input_partition_manifest": {"status": "complete", "version": "fixture"},
    }


def _matrix_row(class_index, *, polytope_id="polytope-0", frst_hash=None):
    return {
        "polytope_index": 0,
        "frst_class_index": class_index,
        "polytope_id": polytope_id,
        "frst_hash": frst_hash or f"frst-{class_index}",
        "matrix_id": f"matrix-{class_index}",
        "candidate_id": f"matrix-{class_index}",
        "lambda_f": None,
        "torus_shift": None,
        "h11_parity": {"h11_plus": 2, "h11_minus": 0},
        "fixed_component_evidence": {
            "status": "not_evaluated",
            "reason": "matrix fixture",
        },
        "terminal_status": "matrix_validation_passed",
        "terminal_reason_code": "matrix_validation_passed",
        "record_kind": "matrix_validation",
    }


def _candidate_row(
    class_index,
    candidate_id,
    status,
    *,
    lambda_f,
    polytope_id="polytope-0",
    frst_hash=None,
):
    return {
        "polytope_index": 0,
        "frst_class_index": class_index,
        "polytope_id": polytope_id,
        "frst_hash": frst_hash or f"frst-{class_index}",
        "matrix_id": f"matrix-{class_index}",
        "candidate_id": candidate_id,
        "lambda_f": lambda_f,
        "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1},
        "h11_parity": {"h11_plus": 2, "h11_minus": 0},
        "fixed_component_evidence": {
            "fixed_point_components": [],
            "fixed_point_set": {"description": "fixture"},
            "smoothness": {"status": "fixture"},
        },
        "terminal_status": status,
        "terminal_reason_code": status,
        "record_kind": "candidate",
    }


def _terminal_fixture(rows, class_count):
    directory = tempfile.TemporaryDirectory(prefix="cyax-gap-ledger-")
    path = Path(directory.name) / "ledger.jsonl"
    writer = TerminalLedgerWriter(path, provenance=_provenance())
    for row in rows:
        writer.write(row)
    summary = writer.close()
    data = {
        "details": [{"polytope_index": 0, "frst_class_count": class_count}],
        "terminal_ledger": summary,
    }
    return directory, data


class GapClassifierUnitTests(unittest.TestCase):
    def test_terminal_funnel_uses_only_lambda_f_one_candidates(self):
        rows = [
            _matrix_row(0),
            _candidate_row(0, "o5-accepted", "accepted_verified_orientifold", lambda_f=0),
            _candidate_row(0, "o3-rejected", "fixed_point_set_non_smooth", lambda_f=1),
            _matrix_row(1),
            _candidate_row(
                1,
                "o5-unavailable",
                "smoothness_verification_unavailable",
                lambda_f=0,
            ),
            _candidate_row(1, "o3-rejected-2", "torus_shift_not_involution", lambda_f=1),
        ]
        directory, data = _terminal_fixture(rows, 2)
        self.addCleanup(directory.cleanup)

        result = gap._terminal_ledger_audit(data, Path("fixture.json"))

        self.assertEqual(
            result["category_counts"],
            {"unaccepted_exhaustive_terminal_rejection": 2},
        )
        self.assertEqual(
            result["terminal_status_counts"],
            {
                "fixed_point_set_non_smooth": 1,
                "torus_shift_not_involution": 1,
            },
        )
        self.assertEqual(
            result["all_terminal_status_counts"]["accepted_verified_orientifold"],
            1,
        )
        self.assertEqual(
            result["unaccepted_class_records"][0]["candidate_attempt_count"],
            1,
        )

    def test_numerical_geometry_failure_is_unavailable_evidence(self):
        rows = [
            _matrix_row(0),
            _candidate_row(
                0,
                "o3-numerical",
                "numerical_geometry_failure",
                lambda_f=1,
            ),
        ]
        directory, data = _terminal_fixture(rows, 1)
        self.addCleanup(directory.cleanup)

        result = gap._terminal_ledger_audit(data, Path("fixture.json"))

        self.assertEqual(
            result["category_counts"],
            {"unaccepted_exhaustive_with_unavailable_evidence": 1},
        )
        self.assertEqual(
            result["unaccepted_class_records"][0][
                "unavailable_evidence_candidate_count"
            ],
            1,
        )

    def test_per_class_polytope_and_frst_identity_must_be_consistent(self):
        rows = [
            _matrix_row(0),
            _candidate_row(0, "o3-a", "fixed_point_set_non_smooth", lambda_f=1),
            _candidate_row(
                0,
                "o3-b",
                "fixed_point_set_non_smooth",
                lambda_f=1,
                frst_hash="different-frst",
            ),
        ]
        directory, data = _terminal_fixture(rows, 1)
        self.addCleanup(directory.cleanup)

        with self.assertRaisesRegex(gap.ArtifactError, "class identity mismatch"):
            gap._terminal_ledger_audit(data, Path("fixture.json"))

    def test_surface_unavailable_is_annotation_without_candidate_linkage(self):
        detail = _detail(
            attempts=[
                {
                    "frst_class_index": 0,
                    "status": "unavailable",
                    "reason_code": "non_smooth_ambient_cone",
                    "reason": "ambient certificate unavailable",
                }
            ]
        )

        record = gap._class_record(detail, 0, accepted=False)

        self.assertEqual(
            record["category"],
            "unaccepted_not_classified_by_retained_terminal_ledger",
        )
        self.assertFalse(record["candidate_linked_unavailable"])

    def test_candidate_linked_unavailable_is_class_partition_evidence(self):
        detail = _detail(
            unresolved=[
                {
                    "candidate_id": "candidate-1",
                    "reason_code": "non_smooth_ambient_cone",
                    "candidate_terminal_status": "fixed_point_set_non_smooth",
                    "frst_class_index": 0,
                }
            ]
        )

        record = gap._class_record(detail, 0, accepted=False)

        self.assertEqual(
            record["category"], "unaccepted_with_candidate_linked_unavailable"
        )
        self.assertTrue(record["candidate_linked_unavailable"])
        self.assertEqual(record["candidate_linked_unavailable_component_count"], 1)
        self.assertEqual(record["candidate_linked_candidate_ids"], ["candidate-1"])
        self.assertIn("not a singularity determination", record["reason"])
        self.assertIn("paper-error finding", record["reason"])

    def test_partial_candidate_terminal_context_is_annotation_only(self):
        detail = _detail(
            attempts=[
                {
                    "frst_class_index": 0,
                    "status": "certified",
                    "reason_code": None,
                    "candidate_context": {
                        "candidate_terminal_status": "fixed_point_set_non_smooth"
                    },
                }
            ]
        )

        record = gap._class_record(detail, 0, accepted=False)

        self.assertEqual(
            record["category"],
            "unaccepted_not_classified_by_retained_terminal_ledger",
        )
        self.assertFalse(record["candidate_linked_unavailable"])

    def test_no_diagnostic_is_not_a_rejection(self):
        record = gap._class_record(_detail(), 0, accepted=False)

        self.assertEqual(
            record["category"],
            "unaccepted_not_classified_by_retained_terminal_ledger",
        )
        self.assertIn("not an exhaustive candidate verdict", record["reason"])

    def test_surface_only_evidence_is_not_promoted_to_a_class_verdict(self):
        detail = _detail(
            attempts=[
                {
                    "frst_class_index": 0,
                    "status": "certified",
                    "reason_code": None,
                }
            ]
        )

        record = gap._class_record(detail, 0, accepted=False)

        self.assertEqual(
            record["category"],
            "unaccepted_not_classified_by_retained_terminal_ledger",
        )

    def test_category_partition_is_mutually_exclusive_for_fixture_records(self):
        details = [
            _detail(0, accepted=[0]),
            _detail(
                1,
                unresolved=[
                    {
                        "frst_class_index": 0,
                        "candidate_id": "candidate-1",
                        "reason_code": "non_smooth_ambient_cone",
                    }
                ],
            ),
            _detail(
                2,
                attempts=[
                    {
                        "frst_class_index": 0,
                        "status": "certified",
                        "reason_code": None,
                        "candidate_context": {
                            "candidate_terminal_status": "fixed_point_set_non_smooth"
                        },
                    }
                ],
            ),
            _detail(3),
        ]
        records = [
            gap._class_record(detail, 0, accepted=(index == 0))
            for index, detail in enumerate(details)
        ]
        categories = [record["category"] for record in records]

        self.assertEqual(
            categories,
            [
                "certified_inherited",
                "unaccepted_with_candidate_linked_unavailable",
                "unaccepted_not_classified_by_retained_terminal_ledger",
                "unaccepted_not_classified_by_retained_terminal_ledger",
            ],
        )
        self.assertEqual(len(categories), 4)

    def test_unsupported_h11_is_rejected_by_scope(self):
        # The scope guard fires before any file access.
        with self.assertRaisesRegex(gap.ArtifactError, "supports"):
            gap.load_artifact(H11_2, 6)

    def test_h11_four_and_five_are_now_supported(self):
        self.assertIn(4, gap.SUPPORTED_H11)
        self.assertIn(5, gap.SUPPORTED_H11)

    def test_source_verified_table_1_targets(self):
        # arXiv:2412.12012v1 Table 1 / tab:ScanData, source-verified.
        self.assertEqual(
            gap.TABLE_1_TARGETS[4],
            {
                "favorable_polytopes": 1185,
                "frst_classes": 1760,
                "inherited_orientifold_cys": 1559,
                "h11_minus_zero_orientifold_cys": 1554,
                "h11_minus_zero_h21_plus_zero_orientifold_cys": 267,
                "models": 3348,
            },
        )
        self.assertEqual(
            gap.TABLE_1_TARGETS[5],
            {
                "favorable_polytopes": 4897,
                "frst_classes": 11713,
                "inherited_orientifold_cys": 9530,
                "h11_minus_zero_orientifold_cys": 9459,
                "h11_minus_zero_h21_plus_zero_orientifold_cys": 1033,
                "models": 29898,
            },
        )

    def test_aggregate_gap_is_not_treated_as_class_id_mapping(self):
        result = gap._comparison(253, 80)

        self.assertEqual(result["target_gap_count"], 173)
        self.assertEqual(result["code_output"], 80)
        self.assertNotIn("target_gap_class_ids", result)

    def test_inherited_partition_requires_equal_retained_h11_zero_ids(self):
        data = {
            "counts": {
                "source_evidence_inherited_orientifold_cys": 1,
                "source_evidence_h11_minus_zero_orientifold_cys": 0,
            },
            "details": [_detail()],
        }

        with self.assertRaisesRegex(
            gap.ArtifactError,
            "only h11_minus_zero_classes identifiers are retained",
        ):
            gap._class_level_audit(data, 2)


def _geo_matrix_row(polytope_index, class_index, normal_form_id):
    row = _matrix_row(
        class_index,
        polytope_id=f"polytope-{polytope_index}",
        frst_hash=f"frst-{polytope_index}-{class_index}",
    )
    row["polytope_index"] = polytope_index
    row["polytope_normal_form_id"] = normal_form_id
    row["matrix_id"] = f"matrix-{polytope_index}-{class_index}"
    row["candidate_id"] = f"matrix-{polytope_index}-{class_index}"
    return row


def _geo_candidate_row(polytope_index, class_index, normal_form_id, candidate_id, status, *, lambda_f):
    row = _candidate_row(
        class_index,
        candidate_id,
        status,
        lambda_f=lambda_f,
        polytope_id=f"polytope-{polytope_index}",
        frst_hash=f"frst-{polytope_index}-{class_index}",
    )
    row["polytope_index"] = polytope_index
    row["polytope_normal_form_id"] = normal_form_id
    row["matrix_id"] = f"matrix-{polytope_index}-{class_index}"
    return row


def _shard_ledger(rows, *, prefix="cyax-gap-shard-"):
    """Write one shard's rows to a real JSONL sidecar; return (dir, summary)."""
    directory = tempfile.TemporaryDirectory(prefix=prefix)
    path = Path(directory.name) / "ledger.jsonl"
    writer = TerminalLedgerWriter(path, provenance=_provenance())
    for row in rows:
        writer.write(row)
    summary = writer.close()
    return directory, summary


# A synthetic Table 1 entry so the sharded path can be exercised without
# un-gating the real (source-verification-pending) h11=4/5 targets.
_SYNTH_H11 = 97
_SYNTH_TARGETS = {
    "favorable_polytopes": 2,
    "frst_classes": 4,
    "inherited_orientifold_cys": 3,
    "h11_minus_zero_orientifold_cys": 3,
    "h11_minus_zero_h21_plus_zero_orientifold_cys": 1,
    "models": 5,
}


def _shard_artifact(summary, *, index, count, shard_favorable, total_favorable, counts):
    return {
        "schema_version": gap.REPRODUCTION_SCHEMA_VERSION,
        "run_provenance": {"source_commit": "fixture-commit"},
        "input": {
            "requested_h11": _SYNTH_H11,
            "population_complete": False,
            "shard": {
                "index": index,
                "count": count,
                "is_sharded": True,
                "shard_favorable_polytopes": shard_favorable,
                "total_favorable_polytopes": total_favorable,
            },
        },
        "counts": counts,
        "paper_targets": dict(_SYNTH_TARGETS),
        "terminal_ledger": summary,
    }


def _two_shard_population():
    """Build a disjoint two-shard population: 4 classes, 1 certified, 1 unavailable."""
    dir0, summary0 = _shard_ledger(
        [
            _geo_matrix_row(0, 0, "nf-0"),
            _geo_candidate_row(0, 0, "nf-0", "o3-accept", "accepted_verified_orientifold", lambda_f=1),
            _geo_matrix_row(0, 1, "nf-0"),
            _geo_candidate_row(0, 1, "nf-0", "o3-unavail", "smoothness_verification_unavailable", lambda_f=1),
        ]
    )
    dir1, summary1 = _shard_ledger(
        [
            _geo_matrix_row(1, 0, "nf-1"),
            _geo_candidate_row(1, 0, "nf-1", "o3-rej-a", "fixed_point_set_non_smooth", lambda_f=1),
            _geo_matrix_row(1, 1, "nf-1"),
            _geo_candidate_row(1, 1, "nf-1", "o3-rej-b", "torus_shift_not_involution", lambda_f=1),
        ]
    )
    shard0 = _shard_artifact(
        summary0,
        index=0,
        count=2,
        shard_favorable=1,
        total_favorable=2,
        counts={
            "favorable_polytopes": 1,
            "frst_classes": 2,
            "source_evidence_inherited_orientifold_cys": 1,
            "source_evidence_h11_minus_zero_orientifold_cys": 1,
            "h21_plus_zero_trilayer_frst_classes": 1,
        },
    )
    shard1 = _shard_artifact(
        summary1,
        index=1,
        count=2,
        shard_favorable=1,
        total_favorable=2,
        counts={
            "favorable_polytopes": 1,
            "frst_classes": 2,
            "source_evidence_inherited_orientifold_cys": 0,
            "source_evidence_h11_minus_zero_orientifold_cys": 0,
            "h21_plus_zero_trilayer_frst_classes": 0,
        },
    )
    return (dir0, dir1), [shard0, shard1]


class ShardedTerminalLedgerUnitTests(unittest.TestCase):
    def test_union_of_disjoint_shards_builds_one_exhaustive_funnel(self):
        (dir0, dir1), shards = _two_shard_population()
        self.addCleanup(dir0.cleanup)
        self.addCleanup(dir1.cleanup)

        result = gap._sharded_terminal_ledger_audit(
            [(shards[0], Path("shard0.json"), None), (shards[1], Path("shard1.json"), None)],
            expected_frst_classes=4,
            source_label="fixture",
        )

        self.assertEqual(result["total_frst_class_count"], 4)
        self.assertEqual(result["certified_class_count"], 1)
        self.assertEqual(
            result["category_counts"],
            {
                "certified_inherited": 1,
                "unaccepted_exhaustive_terminal_rejection": 2,
                "unaccepted_exhaustive_with_unavailable_evidence": 1,
            },
        )
        self.assertEqual(result["shard_count"], 2)
        self.assertEqual(result["shard_class_counts"], [2, 2])
        self.assertEqual(result["class_identity_basis"], "polytope_normal_form_id")
        self.assertIsNone(result["sidecar_sha256"])
        self.assertEqual(len(result["shard_sidecar_sha256"]), 2)

    def test_class_repeated_across_shards_is_a_disjointness_error(self):
        # Both shards claim the same geometry identity (nf-dup, class 0).
        dir0, summary0 = _shard_ledger(
            [
                _geo_matrix_row(0, 0, "nf-dup"),
                _geo_candidate_row(0, 0, "nf-dup", "o3-a", "fixed_point_set_non_smooth", lambda_f=1),
            ]
        )
        dir1, summary1 = _shard_ledger(
            [
                _geo_matrix_row(9, 0, "nf-dup"),
                _geo_candidate_row(9, 0, "nf-dup", "o3-b", "fixed_point_set_non_smooth", lambda_f=1),
            ]
        )
        self.addCleanup(dir0.cleanup)
        self.addCleanup(dir1.cleanup)

        with self.assertRaisesRegex(gap.ArtifactError, "not disjoint by geometry"):
            gap._sharded_terminal_ledger_audit(
                [
                    ({"terminal_ledger": summary0}, Path("s0.json"), None),
                    ({"terminal_ledger": summary1}, Path("s1.json"), None),
                ],
                expected_frst_classes=1,
                source_label="fixture",
            )

    def test_wrong_expected_class_count_is_rejected(self):
        (dir0, dir1), shards = _two_shard_population()
        self.addCleanup(dir0.cleanup)
        self.addCleanup(dir1.cleanup)

        with self.assertRaisesRegex(gap.ArtifactError, "expected 5"):
            gap._sharded_terminal_ledger_audit(
                [(shards[0], Path("s0.json"), None), (shards[1], Path("s1.json"), None)],
                expected_frst_classes=5,
                source_label="fixture",
            )

    def test_shard_record_count_metadata_mismatch_is_rejected(self):
        (dir0, dir1), shards = _two_shard_population()
        self.addCleanup(dir0.cleanup)
        self.addCleanup(dir1.cleanup)
        shards[0]["terminal_ledger"] = dict(shards[0]["terminal_ledger"])
        shards[0]["terminal_ledger"]["record_count"] += 1

        with self.assertRaisesRegex(gap.ArtifactError, "record count does not match"):
            gap._sharded_terminal_ledger_audit(
                [(shards[0], Path("s0.json"), None), (shards[1], Path("s1.json"), None)],
                expected_frst_classes=4,
                source_label="fixture",
            )

    def test_sidecar_sha_mismatch_is_rejected(self):
        (dir0, dir1), shards = _two_shard_population()
        self.addCleanup(dir0.cleanup)
        self.addCleanup(dir1.cleanup)
        shards[0]["terminal_ledger"] = dict(shards[0]["terminal_ledger"])
        shards[0]["terminal_ledger"]["sidecar_sha256"] = "0" * 64

        with self.assertRaisesRegex(gap.ArtifactError, "SHA-256 does not match"):
            gap._sharded_terminal_ledger_audit(
                [(shards[0], Path("s0.json"), None), (shards[1], Path("s1.json"), None)],
                expected_frst_classes=4,
                source_label="fixture",
            )


class ShardedAnalysisTests(unittest.TestCase):
    def _patched(self):
        targets = dict(gap.TABLE_1_TARGETS)
        targets[_SYNTH_H11] = dict(_SYNTH_TARGETS)
        return (
            mock.patch.object(gap, "SUPPORTED_H11", (2, 3, _SYNTH_H11)),
            mock.patch.object(gap, "TABLE_1_TARGETS", targets),
        )

    def test_end_to_end_sharded_accounting(self):
        (dir0, dir1), shards = _two_shard_population()
        self.addCleanup(dir0.cleanup)
        self.addCleanup(dir1.cleanup)
        supported, table = self._patched()
        with supported, table:
            analysis = gap.analyze_sharded_artifact(
                shards,
                _SYNTH_H11,
                sidecars=[None, None],
                source_paths=[Path("s0.json"), Path("s1.json")],
            )

        self.assertTrue(analysis["population"]["population_complete"])
        inherited = analysis["orientifold_comparison"]["inherited_orientifold_cys"]
        self.assertEqual(inherited["code_output"], 1)
        self.assertEqual(inherited["target_gap_count"], 2)
        ceiling = inherited["conditional_ceiling"]
        self.assertEqual(ceiling["certified_code_count"], 1)
        self.assertEqual(ceiling["candidate_linked_unavailable_class_count"], 1)
        self.assertEqual(ceiling["conditional_ceiling_count"], 2)
        self.assertEqual(ceiling["conditional_ceiling_deficit"], 1)
        self.assertEqual(analysis["class_level_audit"]["certified_class_count"], 1)
        self.assertEqual(analysis["terminal_ledger_audit"]["shard_count"], 2)

    def test_incomplete_shard_set_is_rejected(self):
        (dir0, dir1), shards = _two_shard_population()
        self.addCleanup(dir0.cleanup)
        self.addCleanup(dir1.cleanup)
        supported, table = self._patched()
        with supported, table, self.assertRaisesRegex(gap.ArtifactError, "shard indices"):
            gap._validate_sharded_artifacts(
                [shards[0], shards[0]], _SYNTH_H11, [Path("s0.json"), Path("s0b.json")]
            )

    def test_mismatched_paper_targets_rejected(self):
        (dir0, dir1), shards = _two_shard_population()
        self.addCleanup(dir0.cleanup)
        self.addCleanup(dir1.cleanup)
        shards[1]["paper_targets"] = dict(shards[1]["paper_targets"])
        shards[1]["paper_targets"]["inherited_orientifold_cys"] = 999
        supported, table = self._patched()
        with supported, table, self.assertRaisesRegex(gap.ArtifactError, "paper_targets"):
            gap._validate_sharded_artifacts(
                shards, _SYNTH_H11, [Path("s0.json"), Path("s1.json")]
            )

    def test_sharded_h11_gate_still_blocks_unsupported_h11(self):
        (dir0, dir1), shards = _two_shard_population()
        self.addCleanup(dir0.cleanup)
        self.addCleanup(dir1.cleanup)
        # Without patching SUPPORTED_H11, the synthetic h11 is refused.
        with self.assertRaisesRegex(gap.ArtifactError, "sharded analysis supports"):
            gap._validate_sharded_artifacts(
                shards, _SYNTH_H11, [Path("s0.json"), Path("s1.json")]
            )


@unittest.skipUnless(H11_2.exists() and H11_3.exists(), "corrected audit artifacts are absent")
class CorrectedArtifactIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = gap.analyze_paths({2: H11_2, 3: H11_3})
        cls.by_h11 = {entry["h11"]: entry for entry in cls.result["analyses"]}

    def test_single_artifact_scope_defers_sharded_h11(self):
        self.assertEqual(self.result["scope"]["h11"], [2, 3])
        self.assertEqual(
            self.result["scope"]["sharded_h11_analyzed_separately"], [4, 5]
        )

    def test_h11_two_class_level_accounting(self):
        entry = self.by_h11[2]
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "target_gap_count"
            ],
            22,
        )
        self.assertEqual(entry["class_level_audit"]["total_frst_class_count"], 36)
        self.assertEqual(entry["class_level_audit"]["certified_class_count"], 10)
        self.assertEqual(
            entry["fixed_surface_diagnostics"],
            {
                "surface_attempt_count": 1164,
                "certified_surface_count": 762,
                "skipped_surface_count": 402,
                "skip_reason_counts": {"non_smooth_ambient_cone": 402},
                "unresolved_candidate_component_count": 33,
                "interpretation": (
                    "diagnostic rows are evidence attempts, not accepted orientifold "
                    "classes; surface statuses are annotations, and only unresolved "
                    "components linked to a candidate enter the class partition"
                ),
            },
        )
        self.assertEqual(
            entry["class_level_audit"]["category_counts"],
            {
                "unaccepted_not_classified_by_retained_terminal_ledger": 20,
                "unaccepted_with_candidate_linked_unavailable": 6,
            },
        )
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "conditional_ceiling"
            ]["conditional_ceiling_count"],
            16,
        )
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "conditional_ceiling"
            ]["conditional_ceiling_deficit"],
            16,
        )
        self.assertEqual(
            entry["orientifold_comparison"][
                "h11_minus_zero_h21_plus_zero_orientifold_cys"
            ]["target_gap_count"],
            0,
        )

    def test_h11_three_class_level_accounting(self):
        entry = self.by_h11[3]
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "target_gap_count"
            ],
            173,
        )
        self.assertEqual(entry["class_level_audit"]["total_frst_class_count"], 274)
        self.assertEqual(entry["class_level_audit"]["certified_class_count"], 80)
        self.assertEqual(
            entry["fixed_surface_diagnostics"]["surface_attempt_count"], 5510
        )
        self.assertEqual(
            entry["fixed_surface_diagnostics"]["skipped_surface_count"], 1892
        )
        self.assertEqual(
            entry["fixed_surface_diagnostics"]["skip_reason_counts"],
            {"non_smooth_ambient_cone": 1892},
        )
        self.assertEqual(
            entry["fixed_surface_diagnostics"][
                "unresolved_candidate_component_count"
            ],
            117,
        )
        self.assertEqual(
            entry["class_level_audit"]["category_counts"],
            {
                "unaccepted_not_classified_by_retained_terminal_ledger": 166,
                "unaccepted_with_candidate_linked_unavailable": 28,
            },
        )
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "conditional_ceiling"
            ]["conditional_ceiling_count"],
            108,
        )
        self.assertEqual(
            entry["orientifold_comparison"]["inherited_orientifold_cys"][
                "conditional_ceiling"
            ]["conditional_ceiling_deficit"],
            145,
        )
        self.assertEqual(
            entry["orientifold_comparison"][
                "h11_minus_zero_h21_plus_zero_orientifold_cys"
            ]["target_gap_count"],
            0,
        )

    def test_every_candidate_linked_class_has_explicit_boundary_evidence(self):
        for entry in self.by_h11.values():
            for record in entry["class_level_audit"]["unaccepted_class_records"]:
                if record["category"] == "unaccepted_with_candidate_linked_unavailable":
                    self.assertTrue(record["candidate_linked_unavailable"])
                    self.assertTrue(record["candidate_linked_candidate_ids"])
                    self.assertTrue(record["unresolved_component_count"])


if __name__ == "__main__":
    unittest.main()
