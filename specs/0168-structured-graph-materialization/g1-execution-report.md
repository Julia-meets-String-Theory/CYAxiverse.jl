# CYAX-0168 G1 execution report

## Gate result

**CYAX-0168 G1: NOT SATISFIED — preregistration/common-contract defect**

Execution started from exact PR #169 head
`a0ba1efcb71fe5d4c9e4dad365203f6a5ca94fb2` after live verification that the
PR was open, draft, and unmerged and that Issue #168 comment `5668898317`
remained the valid G0 approval event.

G1 stopped during the mandatory independent generator 2.2 reproduction. Two
independently implemented phase-2 constructions agree on the complete 130
base-motif assertions for the shared tiny conformance input but legally diverge
at phase 2 because the approved specification does not freeze:

1. the ordering in which the remaining blocks become “consecutive
   components”; or
2. the direction of each adjacent WorkItem `depends_on` assertion in the
   required chain.

One implementation partitions by block ordinal and emits each adjacent chain
edge from the earlier item to the later item. The other partitions by
primary-ID order and emits each later item as depending on the previous item.
Both phase-2 choices satisfy the current prose. The first differing chain bytes
at construction ordinal 130 create different assertion IDs before either
separate implementation nonconformance described below. That first
identity-bearing difference alone fails byte-level generator reproduction and
shows that the approved phase-2 prose does not determine one canonical
snapshot. Later differences are not used as evidence for, or causally
attributed to, the contract defect.

The primary implementation also has a separate phase-3 nonconformance: its
ordinary candidate vector contains the subject WorkItem and relies on the
post-PRF self-edge rejection. The specification instead defines the frozen
candidate vector as every *other* same-component WorkItem. Changing the vector
length can change PRF mapping and retry traces. This implementation defect can
create additional later differences and is not evidence for, or attributed to,
the phase-2 contract defect. It does not remove the first divergence at phase 2.

The manager comparison is frozen in
`research/cyax0168/evidence/generator-independence-failure.json`.

## Frozen comparison

Input: `conformance-tiny`, `P-low`, seed `900001`, 100 entities. This tuple is
not a T0–T4 or C0–C3 fixture.

| Item | Primary implementation | Independent implementation |
| --- | --- | --- |
| entities / literals / physical source revisions / assertions | 100 / 10 / 510 / 500 | 100 / 10 / 510 / 500 |
| logical snapshot ID | `cyax-snapshot-sha256:f58102d64790359a88e3cfc4393e169b4cf6ab775400d999e4d8efca802d2a29` | `cyax-snapshot-sha256:b63a35f36b933f27d3d1e119ea1b4ec98038756c8c0a7f445f34eb203f70b988` |
| byte/checksum identity | fail | fail |

The implementations share 138 final assertion IDs and each has 362 distinct
assertion IDs. The extra common IDs beyond the 130 base motif records arise
from later deterministic records that do not resolve the phase-2 ambiguity.

This meets the approved stop criterion: two competent implementations following
the two phase-2 constructions permitted by the approved prose can materially
diverge in identity-bearing fixture bytes before phase 3. The defect must return
to CYAX-0168 G0 amendment and independent rereview. It must not be repaired
normatively inside this G1 execution. The separate primary phase-3
nonconformance must also be corrected before a later generator reproduction can
pass, but that implementation correction cannot select the missing phase-2
semantics.

The independent implementation has another separate nonconformance in phase 5:
it includes isolated WorkItems in supersession even though phase 1 excludes
isolated blocks from cross-block phases. Both implementations produce the same
four supersession assertions at ordinals 146–149; the independent implementation
then adds an isolated-block supersession at ordinal 150. This shifts its parallel
and filler provenance ordinals independently of the phase-2 defect. This later
shift is disclosed but is not part of the G0-escalation evidence.

## Work completed before the stop

- Common registered semantic model, canonical framing and IDs, authority and
  temporal evaluators, RetrievalBundle validation, immutable source/snapshot
  checksums, deltas, publication, crash, rollback, and recovery machinery.
- Primary and independent generator 2.2 implementations and conformance tests.
- Q01–Q12 evaluator, measurement/control helpers, statistical primitives,
  classifier and duration projection implementation with synthetic tests.
- SQLite S implementation and tiny synthetic build/rebuild/reopen/tamper/crash
  smoke.
- Exact-host Ladybug artifact, license, privacy-safe host manifest, offline
  installation, `THREADS=1`, network-disabled tiny synthetic smoke, logical
  export, deterministic rebuild, tamper, partial-build and recovery evidence.

These implementation artifacts are provisional G1 evidence. They do not
ratify the defective generator contract or establish CYAX-0168 G1.

The approved spec/plan/tasks statements that G1 was “not yet executed” describe
the pre-dispatch state at their frozen approval revision. This report is the
newer execution record: a G1 attempt occurred and stopped at generator
independence. That lifecycle clarification does not amend normative intent.

## Work not executed after the stop

- No C0–C3 calibration fixture was accepted or frozen. A primary-worker C0
  in-memory pre-independence probe was discarded as inadmissible evidence.
- No equal-budget calibration tuning freeze was performed.
- No statistical-precision ratification was performed.
- No 48-hour campaign-duration ratification was performed.
- No T0–T4 decision fixture was materialized, executed, profiled, inspected
  through S/G, or used to derive backend-specific identities or results.
- CYAX-0168 G2–G4 were not entered.

## Required next governance action

Return to CYAX-0168 G0. A reviewed amendment must freeze the exact phase-2
component order and `depends_on` chain direction, update generator versioning
if required by the identity change, and then obtain explicit owner approval
before a new G1 execution.
