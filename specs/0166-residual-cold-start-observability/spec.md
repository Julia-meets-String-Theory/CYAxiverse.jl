---
spec_id: CYAX-0166
title: Residual cold-start reconstruction observability benchmark
issue: 166
class: S2
status: draft
workstream: Infrastructure
parent: 162
depends_on: [163, 117]
created: 2026-09-13
last_reviewed: 2026-09-13
review_required: independent methodology/privacy review and repository-owner approval before subject launch
approval_ref: null
---

# CYAX-0166 — Residual cold-start reconstruction observability benchmark

## Objective

Determine whether the approved lightweight structured provenance/context
approach leaves a repeatable orientation or reconstruction failure that
justifies another memory mechanism.

This specification designs one experiment. It does not authorize subject
launch, implementation of durable memory infrastructure, or production use.
The design branch starts from integrated `vmm` revision
`7a40285bb5c313f7e8746b90644d5f45bb67be44`.

The approved CYAX-0163 specification and merged PR #164 establish the control
architecture. The source-pinned comparator checkpoint in Issue #162 comment
`5655535824` is supporting evidence for measurement before architecture
expansion; it is not a CYAX-0163 amendment and does not authorize a mechanism.

## Authority and approval boundary

The control architecture is the approved CYAX-0163 result:

```text
authoritative durable sources
  → deterministic provenance-aware materialization
  → concise structured context
  → fresh agent
```

Canonical GitHub and repository artifacts keep their existing authority.
Event frequency is diagnostic only. It cannot establish scientific or project
authority. This specification remains draft with `approval_ref: null`; no
fresh-agent run may start until the complete preregistration, source snapshot,
answer key, context, event schema, and implementation have passed independent
methodology/privacy review and received repository-owner approval.

## Held-out subject

Use Issue #117, **Local Cartier, zero-dimensional, and orbifold certificates:
owner approval and point-certificate follow-up**.

Rationale:

- #117 was not used to design the CYAX-0163 context mechanism or its #159,
  #157, and #155 pilots;
- it is bounded and has a durable known current-state answer;
- it contains owner-scoped authority, merged and superseded PR history,
  explicit dependencies, evidence limits, and a valid next action;
- it is not primarily the fuzzy/Table-1 historical-forensics problem.

The source freeze must capture, at minimum:

- Issue #117 body, live state, Project state, and owner comment `5556641428`;
- PR #101 at merged head `563a367a8a2336522c3f284747d23744ac82edc9`
  and merge commit `2162e81fb77700753549c839c6208daecafa325f`;
- PR #103 as closed/unmerged and its supersession comment `5515545835`;
- PR #104 at merged head `2ea2e10c772475d89979d2e0f7a1b02a3b63133d`
  and merge commit `07552d0c1615b3e8ed047d3480f64fd96912c74e`;
- Issue #112 as the separate broader non-simplicial/orbifold research line;
- repository files at `vmm@7a40285bb5c313f7e8746b90644d5f45bb67be44`,
  including the projection ledger and the local-evidence producer.

Any source-state change before launch invalidates the freeze. Re-freeze and
repeat review; do not silently transfer approval.

## Required reconstruction outputs

A fresh subject must reconstruct:

1. authoritative current state;
2. governing decisions and specification status;
3. relevant supersession and history;
4. supporting evidence and its limits;
5. unresolved questions or risks;
6. next valid action.

The answer key uses twelve binary items, two per output:

| Item | Required fact |
| --- | --- |
| K1 | #117 is an open bounded follow-up; integration of earlier ledger/driver work does not complete its remaining replay and direct-test work. |
| K2 | The approved boundary is narrow ordinary-Euler evidence from smooth, unimodular local cones; it does not establish a population claim. |
| K3 | The governing authority is Issue #117 plus owner comment `5556641428`; no standalone approved CYAX-0117 feature specification exists in the frozen sources. |
| K4 | Owner approval resumes bounded h11=4/h11=5 replay gates but does not approve broader orbifold/stringy-Euler or non-simplicial mathematics. |
| K5 | PR #101 merged the projection ledger and independently reproduced its digest; it made no scientific-code change. |
| K6 | PR #103 is closed/unmerged and superseded by PR #104; its reused branch/content is not the #117 point-certificate task. |
| K7 | The ledger records 1,146 rows and projection SHA-256 `ebe14b02d312993fd85ce98b2aa882701b1c14f6aff25065e5d566a2f2ad504b`. |
| K8 | `_derive_zero_dimensional_local_evidence` exists and fails closed, but the frozen source set has no focused direct unit test for that producer and no completed bounded repaired replay. |
| K9 | Broader non-simplicial/orbifold certificate mathematics remains separate and unresolved in #112. |
| K10 | CYTools availability, direct producer coverage, and bounded replay evidence remain execution risks; no unavailable cases may be promoted from the current evidence. |
| K11 | The next valid action is focused direct testing of the local-evidence producer, followed by bounded h11=4/h11=5 replay under the approved narrow contract and normal review gates. |
| K12 | The subject must abstain from population, orbifold/stringy-Euler, or broader scientific conclusions not established by the frozen evidence. |

The final frozen answer key must cite exact source anchors for every item and
must be independently reviewed before launch.

## Requirements

| Requirement | Required behavior | Verification gate |
| --- | --- | --- |
| R-001 Frozen identity | Freeze the selected work item, canonical source bundle, answer key, prompt, structured context, event schema, model/configuration, run count, scoring, automatic failures, privacy rules, and stop conditions before launch. | Content manifest records byte sizes and SHA-256 values; independent review and owner approval cite the exact preregistration revision. |
| R-002 Comparable fresh runs | Run four genuinely fresh subjects with identical frozen prompt, context, source availability, model, and reasoning configuration. | Launch manifest proves four admissible isolated runs and exact input hashes. |
| R-003 Reconstruction scoring | Score K1–K12 blindly and retain the six output dimensions. | Machine-checked binary scorecards plus independent scoring review. |
| R-004 Minimal observability | Record only the enumerated experiment events needed to identify residual lookup, orientation, freshness, provenance, correctness, and action failures. | Event validator rejects unknown event types/fields and nonconforming values. |
| R-005 Privacy and disposal | Raw events remain private, contain no semantic payload or durable user/session identity, and are destroyed after aggregate verification. | Synthetic privacy tests; pre-publication scan; independent aggregate verification; durable deletion attestation without raw payload. |
| R-006 Decision discipline | Apply the A/B/C gate below without context tuning from held-out results. | Frozen analysis script/rules; final report maps every conclusion to a preregistered gate. |
| R-007 Architecture boundary | Do not introduce or select a graph, map, external memory product, transcript miner, durable memory database, or Julia structural index. | Changed-file/dependency review and explicit non-scope statement. |

## Private event schema

Every raw record has only:

- `schema_version = "cyax-0166-events-1.0"`;
- `run_token`, a random experiment-local token destroyed with the raw log;
- `seq`, a monotonically increasing integer with no wall-clock time;
- `event_type`;
- the event-specific fields below.

| Event | Allowed fields | Meaning |
| --- | --- | --- |
| `canonical_open` | `artifact_token` | Open of one frozen canonical artifact. The aggregator derives total, distinct, and reopen counts. |
| `search` | `query_family_token` | Search for project state/evidence. The token is a per-experiment salted HMAC of a preregistered query-family label, never query prose. |
| `governing_locate` | `artifact_role`, `result` = `found` / `not_found` / `ambiguous` | Whether the governing Issue, owner decision, or spec-status fact was located. |
| `provenance_lookup` | `claim_slot` = `K1`…`K12` | A material answer item required reopening its source. |
| `derived_insufficient` | `dimension` = `authority` / `state` / `history` / `evidence` / `uncertainty` / `action` | Supplied structured context was insufficient for a required output. |
| `derived_freshness_check` | `result` = `fresh` / `stale` / `unknown` | Mechanical result for the frozen context/source bundle. |
| `reconstruction_error` | `category` = `authority` / `current_state` / `supersession` / `evidence` / `uncertainty` / `dependency` / `action` | Evaluator-coded error after response freeze. |
| `next_action_verdict` | `result` = `valid` / `invalid` / `abstain` | Evaluator verdict for the proposed next action. |
| `owner_correction_due_to_reconstruction` | `result = true` | Explicit evaluator/owner flag only; never inferred from conversational wording. |

Artifact and query-family tokens use HMAC-SHA-256 with a fresh random
per-experiment salt. The salt, raw events, and run tokens are private and are
destroyed after aggregate verification. Public artifacts may contain only
bounded aggregate counts, denominators, preregistered categories, scorecards,
and conclusions. They must contain no transcript text, prompts, quotations,
conversation/task/session IDs, paths, URLs, usernames, hostnames, machine IDs,
scientific prose from tool use, raw tokens, or salt.

## Run protocol

- Four fresh subjects.
- Model/configuration: `gpt-5.6-sol`, reasoning `high`, matching the successful
  #163 adversarial subjects. If unavailable, amend and re-review this packet;
  do not substitute silently.
- Identical frozen prompt, structured context, and canonical source bundle.
- No useful inherited conversation history.
- Raw tool events collected by a narrow experiment wrapper, not transcript
  mining.
- Response and event-log hashes frozen before scoring.
- Blind K1–K12 scoring and event aggregation occur only after all responses are
  frozen.
- An infrastructure-invalid launch may be replaced once before scoring, with
  the failed launch retained only in a private operational log. No subject is
  coached or rerun after observing answer quality.

## Automatic failures

Any admissible run fails automatically if it:

- invents an owner decision;
- treats a superseded/rejected claim as current;
- asserts a material conclusion without durable provenance;
- silently resolves a documented scientific disagreement or evidence gap;
- relies on inaccessible private context while claiming durable reconstruction;
- treats retrieval salience as authority;
- proposes an action that skips an owner, scientific, or evidence gate; or
- emits public/raw instrumentation that violates R-005.

## Decision gate

### Outcome A — no meaningful residual failure

Conclude that the current lightweight approach is sufficient for this tested
class only if all four admissible runs score 12/12, have no automatic failure,
locate the governing authority, return `fresh`, propose a valid next action,
and show no repeated residual signature in at least two runs.

A repeated residual signature means the same one of these occurs in at least
two of four runs: any canonical artifact reopen; any repeated query-family
token; a `not_found`/`ambiguous` governing result; the same
`derived_insufficient` dimension; or the same `provenance_lookup` claim slot.
Distinct and total canonical opens remain reported diagnostics, not authority
or an independent pass criterion.

### Outcome B — reproducible orientation/navigation failure

If a repeated residual signature occurs without an automatic correctness
failure, classify it first. A later proposal may address only that observed
failure with the smallest directly testable mechanism. This result does not
itself authorize a map, graph, backend, or new Issue.

### Outcome C — authority/provenance failure

Any authority, current-state, supersession, provenance, uncertainty, dependency,
or invalid-next-action automatic failure stops the experiment. Return to the
approved CYAX-0163 semantics and repair the structured context before proposing
retrieval machinery.

## Stop conditions

Stop before launch if any frozen source changes, a hash differs, the selected
model/configuration is unavailable, privacy validation fails, independent
review is not PASS, or owner approval does not cite the exact packet revision.
Stop during execution on semantic payload leakage, source asymmetry, input hash
mismatch, subject contamination, or an automatic authority/provenance failure.

Do not tune the context from held-out responses. Do not run fuzzy/Table-1 work.

## Compatibility and version impact

Design and experiment-only impact is **none**. No package behavior, public API,
dependency, persisted scientific schema, or package version changes are
authorized.
