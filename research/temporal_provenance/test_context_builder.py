#!/usr/bin/env python3
"""Focused tests for the temporal-provenance context prototype."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from context_builder import LedgerError, derive_dispositions, load_ledger, render_context


LEDGER = HERE / "pilots" / "cyax-0159" / "ledger.jsonl"
GRAPH_MANIFEST = HERE / "pilots" / "cyax-0159" / "graph_manifest.json"
RUN_01_SCORECARD = HERE / "pilots" / "cyax-0159" / "runs" / "run-01" / "scorecard.md"
RUN_02_SCORECARD = HERE / "pilots" / "cyax-0159" / "runs" / "run-02" / "scorecard.md"
RUN_02_RESPONSE = HERE / "pilots" / "cyax-0159" / "runs" / "run-02" / "derived-response.md"


class PilotLedgerTests(unittest.TestCase):
    def setUp(self):
        self.records = load_ledger(LEDGER)

    def _add_relation(
        self,
        records,
        *,
        identifier,
        predicate,
        target,
        epistemic_status="accepted",
        qualifiers=None,
    ):
        template = next(
            record
            for record in records
            if record["record_type"] == "assertion"
        )
        relation = copy.deepcopy(template)
        relation.update(
            id=identifier,
            subject="impl:pr160@bd3e866",
            predicate=predicate,
            object={"ref": target},
            epistemic_status=epistemic_status,
            qualifiers=qualifiers or {},
        )
        records.append(relation)

    def test_pilot_validates_and_has_unique_ids(self):
        identifiers = [record["id"] for record in self.records]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_every_assertion_has_canonical_provenance(self):
        resources = {
            record["id"]: record
            for record in self.records
            if record["record_type"] == "resource"
        }
        for assertion in (
            record for record in self.records if record["record_type"] == "assertion"
        ):
            for citation in assertion["provenance"]:
                self.assertTrue(resources[citation["source"]]["canonical"])

    def test_superseded_implementation_is_not_current(self):
        dispositions = derive_dispositions(self.records)
        self.assertEqual(dispositions["impl:pr160@2b9b056"], {"superseded"})
        self.assertEqual(dispositions["impl:pr160@bd3e866"], {"current"})

    def test_rejected_relationship_does_not_change_target(self):
        records = copy.deepcopy(self.records)
        self._add_relation(
            records,
            identifier="assert:rejected-supersession",
            predicate="supersedes",
            target="impl:pr160@bd3e866",
            epistemic_status="rejected",
        )
        dispositions = derive_dispositions(records)
        self.assertEqual(dispositions["impl:pr160@bd3e866"], {"current"})
        self.assertEqual(dispositions["assert:rejected-supersession"], {"rejected"})

    def test_superseded_relationship_does_not_change_target(self):
        records = copy.deepcopy(self.records)
        self._add_relation(
            records,
            identifier="assert:retire-supersession",
            predicate="supersedes",
            target="assert:f965-supersedes-2b9",
        )
        dispositions = derive_dispositions(records)
        self.assertEqual(dispositions["impl:pr160@2b9b056"], {"current"})
        self.assertEqual(
            dispositions["assert:f965-supersedes-2b9"],
            {"superseded"},
        )

    def test_disputed_relationship_does_not_change_target(self):
        records = copy.deepcopy(self.records)
        self._add_relation(
            records,
            identifier="assert:dispute-supersession",
            predicate="contradicts",
            target="assert:f965-supersedes-2b9",
            qualifiers={"effect": "disputes"},
        )
        dispositions = derive_dispositions(records)
        self.assertEqual(dispositions["impl:pr160@2b9b056"], {"current"})
        self.assertEqual(
            dispositions["assert:f965-supersedes-2b9"],
            {"disputed"},
        )

    def test_resolved_relationship_does_not_change_target(self):
        records = copy.deepcopy(self.records)
        self._add_relation(
            records,
            identifier="assert:resolve-supersession",
            predicate="resolves",
            target="assert:f965-supersedes-2b9",
        )
        dispositions = derive_dispositions(records)
        self.assertEqual(dispositions["impl:pr160@2b9b056"], {"current"})
        self.assertEqual(
            dispositions["assert:f965-supersedes-2b9"],
            {"resolved"},
        )

    def test_required_dependency_propagates_staleness(self):
        records = copy.deepcopy(self.records)
        template = next(record for record in records if record["record_type"] == "assertion")
        dependent = copy.deepcopy(template)
        dependent.update(
            id="assert:test-dependent",
            subject="claim:cyax0159-complete",
            predicate="depends_on",
            object={"ref": "impl:pr160@2b9b056"},
            qualifiers={"strength": "required"},
        )
        records.append(dependent)
        dispositions = derive_dispositions(records)
        self.assertIn("dependency_stale", dispositions["claim:cyax0159-complete"])

    def test_rendered_context_contains_all_benchmark_sections(self):
        rendered = render_context(self.records)
        for heading in (
            "Governing authority",
            "Authoritative current state",
            "Requirement and implementation chain",
            "Supersession and dispute history",
            "Evidential basis",
            "Dependencies and unresolved questions",
            "Next valid action",
            "Canonical source index",
        ):
            self.assertIn(f"## {heading}", rendered)

    def test_rendered_context_states_six_lesson_lifecycle_explicitly(self):
        rendered = render_context(self.records)
        self.assertIn(
            "L-0001–L-0006 are validated, unpromoted, and none is semantically superseded",
            rendered,
        )

    def test_pr_revision_history_is_distinct_from_lesson_lifecycle(self):
        rendered = render_context(self.records)
        self.assertIn(
            "PR #160 revision history is distinct from the lesson lifecycle",
            rendered,
        )
        self.assertIn("## Authoritative current state", rendered)

    def test_review_caveat_uses_scoped_observed_surface(self):
        resources = {
            record["id"]: record
            for record in self.records
            if record["record_type"] == "resource"
        }
        surface = resources["github:pr:160:observed-surface"]
        self.assertIn("observed public PR page only", surface["authority"]["scope"])
        caveat = next(
            record
            for record in self.records
            if record.get("id") == "assert:review-boundary-concerns-final-review"
        )
        self.assertIn(
            "github:pr:160:observed-surface",
            {citation["source"] for citation in caveat["provenance"]},
        )

    def test_run02_source_opening_budget_is_strict(self):
        manifest = json.loads(GRAPH_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["run"], "run-02")
        self.assertEqual(manifest["source_opening_budget"]["preferred"], 0)
        self.assertEqual(manifest["source_opening_budget"]["maximum"], 2)
        self.assertIn("material ambiguity", manifest["instructions"])

    def test_run01_scorecard_records_failure_classification(self):
        scorecard = RUN_01_SCORECARD.read_text(encoding="utf-8")
        self.assertIn("10/12", scorecard)
        self.assertIn("12/12", scorecard)
        self.assertIn("Automatic-failure classification: none", scorecard)
        self.assertIn("Efficiency classification: **fail", scorecard)

    def test_run02_records_provisional_pass_and_zero_source_openings(self):
        scorecard = RUN_02_SCORECARD.read_text(encoding="utf-8")
        response = RUN_02_RESPONSE.read_text(encoding="utf-8")
        self.assertIn("Efficiency classification: **pass", scorecard)
        self.assertIn("provisional pass", scorecard)
        self.assertIn("2,359 words", scorecard)
        self.assertIn("6,215 measured words", scorecard)
        self.assertIn("Canonical artifacts opened: none", response)

    def test_salience_is_rejected_as_stored_authority_adjacent_state(self):
        raw = [json.loads(line) for line in LEDGER.read_text().splitlines() if line]
        raw[1]["salience"] = 1.0
        with self.assertRaisesRegex(LedgerError, "salience is derived"):
            from context_builder import validate_ledger

            validate_ledger(raw)


if __name__ == "__main__":
    unittest.main()
