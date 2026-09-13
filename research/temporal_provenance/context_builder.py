#!/usr/bin/env python3
"""Validate a temporal-provenance JSONL ledger and render cold-start context.

The prototype intentionally uses only the Python standard library.  Canonical
artifacts remain authoritative; ledger records are curated, derived links that
can be deleted and rebuilt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RECORD_TYPES = {"meta", "resource", "assertion"}
RESOURCE_KINDS = {
    "Action",
    "Artifact",
    "Claim",
    "Computation",
    "Decision",
    "Episode",
    "ExternalSource",
    "Implementation",
    "Requirement",
    "Verification",
    "WorkItem",
}
PREDICATES = {
    "blocks",
    "concerns",
    "contradicts",
    "depends_on",
    "documented_in",
    "governs",
    "has_outcome",
    "has_status",
    "implements",
    "next_valid_action",
    "produced_by",
    "resolves",
    "supports",
    "supersedes",
    "uses",
    "verifies",
}
EPISTEMIC_STATUSES = {
    "accepted",
    "extracted",
    "observed",
    "rejected",
    "reported",
    "unresolved",
    "verified",
}
AUTHORITY_CLASSES = {
    "agent_extraction",
    "approved_spec",
    "external_source",
    "implementation_evidence",
    "owner_decision",
    "verification_evidence",
    "workflow_state",
}
REVIEW_STATUSES = {"unreviewed", "curator_checked", "independently_reviewed"}
ACCEPTED_REVIEW_STATUSES = {"curator_checked", "independently_reviewed"}
CANDIDATE_EPISTEMIC_STATUSES = {"extracted", "unresolved"}
RELATION_EFFECTS = {"disputes", "refutes"}
DEPENDENCY_STRENGTHS = {"context", "required", "supporting"}
INACTIVE_DISPOSITIONS = {
    "candidate",
    "contradicted",
    "dependency_stale",
    "disputed",
    "rejected",
    "resolved",
    "superseded",
}


class LedgerError(ValueError):
    """Raised when a ledger violates a prototype invariant."""


def _parse_time(value: str | None, field: str) -> dt.datetime | None:
    if value is None:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"{field} must be an RFC3339 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise LedgerError(f"{field} must include a timezone: {value!r}")
    return parsed


def load_ledger(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise LedgerError(f"{path}:{line_number}: record must be an object")
            record["_line"] = line_number
            records.append(record)
    validate_ledger(records)
    return records


def validate_ledger(records: list[dict[str, Any]]) -> None:
    if not records or records[0].get("record_type") != "meta":
        raise LedgerError("the first record must be meta")
    if sum(record.get("record_type") == "meta" for record in records) != 1:
        raise LedgerError("the ledger must contain exactly one meta record")

    identifiers: dict[str, dict[str, Any]] = {}
    resources: dict[str, dict[str, Any]] = {}
    assertions: dict[str, dict[str, Any]] = {}
    for record in records:
        record_type = record.get("record_type")
        line = record.get("_line")
        if record_type not in RECORD_TYPES:
            raise LedgerError(f"line {line}: unsupported record_type {record_type!r}")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise LedgerError(f"line {line}: non-empty id is required")
        if identifier in identifiers:
            raise LedgerError(f"line {line}: duplicate id {identifier!r}")
        identifiers[identifier] = record
        if "salience" in record:
            raise LedgerError(
                f"line {line}: salience is derived at retrieval time and must not be stored"
            )
        if record_type == "resource":
            _validate_resource(record)
            resources[identifier] = record
        elif record_type == "assertion":
            _validate_assertion_shape(record)
            assertions[identifier] = record

    meta = records[0]
    if meta.get("schema_version") != "cyax-temporal-provenance-0.1":
        raise LedgerError("unsupported schema_version")
    if meta.get("root") not in resources:
        raise LedgerError("meta.root must reference a resource")

    for assertion in assertions.values():
        line = assertion["_line"]
        subject = assertion["subject"]
        object_ref = assertion["object"]["ref"]
        if subject not in identifiers:
            raise LedgerError(f"line {line}: unknown subject {subject!r}")
        if object_ref not in identifiers:
            raise LedgerError(f"line {line}: unknown object ref {object_ref!r}")
        for citation in assertion["provenance"]:
            source = citation.get("source")
            if source not in resources:
                raise LedgerError(f"line {line}: unknown provenance source {source!r}")
            if not resources[source].get("canonical", False):
                raise LedgerError(
                    f"line {line}: provenance source {source!r} is not canonical"
                )
            if not isinstance(citation.get("anchor"), str) or not citation["anchor"]:
                raise LedgerError(f"line {line}: every provenance citation needs an anchor")

    _validate_relation_qualifiers(assertions)
    _validate_supersession_cycles(assertions)


def _validate_resource(record: dict[str, Any]) -> None:
    line = record["_line"]
    if record.get("kind") not in RESOURCE_KINDS:
        raise LedgerError(f"line {line}: unsupported resource kind {record.get('kind')!r}")
    if not isinstance(record.get("label"), str) or not record["label"]:
        raise LedgerError(f"line {line}: resource label is required")
    if not isinstance(record.get("canonical"), bool):
        raise LedgerError(f"line {line}: resource canonical flag is required")
    authority = record.get("authority")
    if not isinstance(authority, dict):
        raise LedgerError(f"line {line}: resource authority is required")
    if authority.get("class") not in AUTHORITY_CLASSES:
        raise LedgerError(f"line {line}: unsupported authority class")
    if not isinstance(authority.get("scope"), str) or not authority["scope"]:
        raise LedgerError(f"line {line}: authority scope is required")
    _parse_time(record.get("observed_at"), f"line {line} observed_at")
    if record["canonical"]:
        locator = record.get("locator")
        if not isinstance(locator, str) or not locator:
            raise LedgerError(f"line {line}: canonical resources require a locator")


def _validate_assertion_shape(record: dict[str, Any]) -> None:
    line = record["_line"]
    if record.get("predicate") not in PREDICATES:
        raise LedgerError(f"line {line}: unsupported predicate {record.get('predicate')!r}")
    if not isinstance(record.get("subject"), str) or not record["subject"]:
        raise LedgerError(f"line {line}: assertion subject is required")
    obj = record.get("object")
    if not isinstance(obj, dict) or set(obj) != {"ref"} or not isinstance(obj["ref"], str):
        raise LedgerError(f"line {line}: assertion object must contain exactly one ref")
    if record.get("epistemic_status") not in EPISTEMIC_STATUSES:
        raise LedgerError(f"line {line}: unsupported epistemic_status")
    _parse_time(record.get("recorded_at"), f"line {line} recorded_at")
    valid_time = record.get("valid_time")
    if not isinstance(valid_time, dict) or set(valid_time) != {"from", "to"}:
        raise LedgerError(f"line {line}: valid_time must contain from and to")
    valid_from = _parse_time(valid_time["from"], f"line {line} valid_time.from")
    valid_to = _parse_time(valid_time["to"], f"line {line} valid_time.to")
    if valid_from and valid_to and valid_to < valid_from:
        raise LedgerError(f"line {line}: valid_time.to precedes valid_time.from")
    provenance = record.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise LedgerError(f"line {line}: assertion provenance is required")
    curation = record.get("curation")
    if not isinstance(curation, dict):
        raise LedgerError(f"line {line}: curation metadata is required")
    if curation.get("review_status") not in REVIEW_STATUSES:
        raise LedgerError(f"line {line}: unsupported curation review_status")
    if not isinstance(curation.get("method"), str) or not curation["method"]:
        raise LedgerError(f"line {line}: curation method is required")


def _validate_relation_qualifiers(assertions: dict[str, dict[str, Any]]) -> None:
    for assertion in assertions.values():
        line = assertion["_line"]
        qualifiers = assertion.get("qualifiers", {})
        if not isinstance(qualifiers, dict):
            raise LedgerError(f"line {line}: qualifiers must be an object")
        if assertion["predicate"] == "contradicts":
            if qualifiers.get("effect") not in RELATION_EFFECTS:
                raise LedgerError(
                    f"line {line}: contradicts requires effect disputes or refutes"
                )
        elif "effect" in qualifiers:
            raise LedgerError(f"line {line}: effect is only valid on contradicts")
        if assertion["predicate"] == "depends_on":
            if qualifiers.get("strength") not in DEPENDENCY_STRENGTHS:
                raise LedgerError(
                    f"line {line}: depends_on requires a dependency strength"
                )
        elif "strength" in qualifiers:
            raise LedgerError(f"line {line}: strength is only valid on depends_on")


def _validate_supersession_cycles(assertions: dict[str, dict[str, Any]]) -> None:
    graph: dict[str, set[str]] = defaultdict(set)
    for assertion in assertions.values():
        if assertion["predicate"] == "supersedes":
            graph[assertion["subject"]].add(assertion["object"]["ref"])

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise LedgerError(f"supersession cycle includes {node!r}")
        if node in visited:
            return
        visiting.add(node)
        for target in graph.get(node, set()):
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def _candidate_relation_ids(assertions: list[dict[str, Any]]) -> set[str]:
    """Return preserved relations that lack state-changing authority."""

    return {
        assertion["id"]
        for assertion in assertions
        if assertion.get("epistemic_status") in CANDIDATE_EPISTEMIC_STATUSES
        or (
            assertion.get("epistemic_status") != "rejected"
            and assertion.get("curation", {}).get("review_status")
            not in ACCEPTED_REVIEW_STATUSES
        )
    }


def _active_relation_ids(assertions: list[dict[str, Any]]) -> set[str]:
    """Return relation assertions that are initially eligible for evaluation.

    A relation may derive state only after curation review has accepted it for
    use and its epistemic state is no longer extracted, unresolved, or
    rejected.  Candidate relations remain in the ledger and rendered context,
    but cannot gain state-changing authority merely by being materialized.

    Final activity is resolved in :func:`derive_dispositions`, because a
    relation can itself be the target of another relation or become stale
    through a required dependency.  Later activity is resolved fail-closed by
    the fixed point.
    """

    candidates = _candidate_relation_ids(assertions)
    return {
        assertion["id"]
        for assertion in assertions
        if assertion.get("epistemic_status") != "rejected"
        and assertion["id"] not in candidates
    }


def derive_dispositions(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    assertions = [record for record in records if record["record_type"] == "assertion"]
    candidate_relations = _candidate_relation_ids(assertions)
    active_relations = _active_relation_ids(assertions)
    while True:
        dispositions: dict[str, set[str]] = defaultdict(lambda: {"current"})
        for record in records:
            dispositions[record["id"]]

        for assertion in assertions:
            if assertion.get("epistemic_status") == "rejected":
                dispositions[assertion["id"]].discard("current")
                dispositions[assertion["id"]].add("rejected")
            elif assertion["id"] in candidate_relations:
                dispositions[assertion["id"]].discard("current")
                dispositions[assertion["id"]].add("candidate")

        for assertion in assertions:
            if assertion["id"] not in active_relations:
                continue
            target = assertion["object"]["ref"]
            if assertion["predicate"] == "supersedes":
                dispositions[target].discard("current")
                dispositions[target].add("superseded")
            elif assertion["predicate"] == "contradicts":
                dispositions[target].discard("current")
                effect = assertion.get("qualifiers", {}).get("effect")
                dispositions[target].add("contradicted" if effect == "refutes" else "disputed")
            elif assertion["predicate"] == "resolves":
                dispositions[target].discard("current")
                dispositions[target].add("resolved")

        changed = True
        while changed:
            changed = False
            for assertion in assertions:
                if assertion["id"] not in active_relations:
                    continue
                if assertion["predicate"] != "depends_on":
                    continue
                if assertion.get("qualifiers", {}).get("strength") != "required":
                    continue
                subject = assertion["subject"]
                dependency = assertion["object"]["ref"]
                if "candidate" in dispositions[dependency]:
                    continue
                if "current" not in dispositions[dependency] and "dependency_stale" not in dispositions[subject]:
                    dispositions[subject].discard("current")
                    dispositions[subject].add("dependency_stale")
                    changed = True

        inactive_relations = {
            assertion["id"]
            for assertion in assertions
            if assertion["id"] in active_relations
            and dispositions[assertion["id"]] & INACTIVE_DISPOSITIONS
        }
        if not inactive_relations:
            return dispositions
        active_relations -= inactive_relations


def _citation_text(assertion: dict[str, Any], resources: dict[str, dict[str, Any]]) -> str:
    citations = []
    for citation in assertion["provenance"]:
        source = resources[citation["source"]]
        citations.append(
            f"{source['label']} — {citation['anchor']} [{source['authority']['class']}]"
        )
    return "; ".join(citations)


def _render_relation(
    assertion: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    dispositions: dict[str, set[str]],
) -> str:
    subject = records_by_id[assertion["subject"]]
    target_id = assertion["object"]["ref"]
    target = records_by_id[target_id]
    subject_label = subject.get("label", subject.get("id", assertion["subject"]))
    target_label = target.get("label", target.get("id", target_id))
    status = ", ".join(sorted(dispositions[assertion["id"]]))
    return (
        f"- **{subject_label}** `{assertion['predicate']}` **{target_label}** "
        f"({assertion['epistemic_status']}; {status}). "
        f"Source: {_citation_text(assertion, records_by_id)}"
    )


def render_context(records: list[dict[str, Any]]) -> str:
    meta = records[0]
    records_by_id = {record["id"]: record for record in records}
    resources = {
        record["id"]: record for record in records if record["record_type"] == "resource"
    }
    assertions = [record for record in records if record["record_type"] == "assertion"]
    dispositions = derive_dispositions(records)
    by_predicate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assertion in assertions:
        by_predicate[assertion["predicate"]].append(assertion)

    sections = [
        ("Governing authority", ("governs", "documented_in")),
        ("Authoritative current state", ("has_status", "has_outcome")),
        ("Requirement and implementation chain", ("implements", "verifies")),
        ("Supersession and dispute history", ("supersedes", "contradicts", "resolves")),
        ("Evidential basis", ("supports", "produced_by", "uses")),
        ("Dependencies and unresolved questions", ("depends_on", "blocks", "concerns")),
        ("Next valid action", ("next_valid_action",)),
    ]

    lines = [
        f"# Cold-start context: {meta['title']}",
        "",
        f"Generated from `{meta['schema_version']}` for `{meta['pilot']}`.",
        "",
        "> This is a disposable, curated context view. It is not authoritative. "
        "Every material line cites a canonical source and its authority scope. "
        "Retrieval order does not change authority.",
        "",
        "## Reconstruction task",
        "",
        meta["reconstruction_task"],
    ]
    for heading, predicates in sections:
        selected = [
            assertion
            for predicate in predicates
            for assertion in by_predicate.get(predicate, [])
        ]
        lines.extend(["", f"## {heading}", ""])
        if selected:
            for assertion in selected:
                lines.append(_render_relation(assertion, records_by_id, dispositions))
        else:
            lines.append("- No curated assertion in this pilot. Abstain rather than infer.")

    lines.extend(["", "## Canonical source index", ""])
    for resource in resources.values():
        if not resource["canonical"]:
            continue
        revision = f" @ `{resource['revision']}`" if resource.get("revision") else ""
        lines.append(
            f"- `{resource['id']}` — {resource['label']}: {resource['locator']}{revision} "
            f"[{resource['authority']['class']}; {resource['authority']['scope']}]"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="validate without rendering")
    args = parser.parse_args()

    try:
        records = load_ledger(args.ledger)
        if not args.check:
            rendered = render_context(records)
            if args.output:
                args.output.write_text(rendered, encoding="utf-8")
            else:
                print(rendered, end="")
    except (LedgerError, OSError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
