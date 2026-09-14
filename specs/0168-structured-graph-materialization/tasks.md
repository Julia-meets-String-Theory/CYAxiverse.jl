# CYAX-0168 tasks

These tasks decompose evidence readiness. GitHub remains authoritative for live
Issue, Project, review, PR, and merge state.

The prior exact-head rereview returned **PASS WITH REQUIRED REVISIONS**. The
independent #117 K1–K12 source audit passed and is not reopened. Prior design
tasks T2–T4 are retained as superseded history; they do not establish that the
repaired contracts are complete. CYAX-0168 G0 remains unsatisfied until a new
independent architecture/methodology rereview passes the new exact head and the
repository owner approves it.

- [x] T1 [R-004] — Audit #117 K1–K12 against captured canonical sources.
  - Outcome: the independent audit passed; corrected K7/K8/K10/K11 preserves
    historical replay evidence without promoting it to a later gate.
  - Boundary: no chronology rewrite is authorized in the rereview repair.
- [x] T2-superseded [R-001, R-002, R-007] — Prior semantic-contract repair.
  - Historical outcome: supplied the contract reviewed at the starting head.
  - Superseded because R1–R4 found Claim/literal closure, owner-event authority,
    assertion identity/time, and snapshot projection incomplete.
- [x] T3-superseded [R-005, R-006, R-008] — Prior scale/query/update repair.
  - Historical outcome: supplied generator 2.0 and update text at the starting
    head.
  - Superseded because R5–R7 found removal, byte generation, selection, and
    aggregation incomplete.
- [x] T4-superseded [R-003, R-009, R-010] — Prior methodology repair.
  - Historical outcome: supplied the controls reviewed at the starting head.
  - Superseded because R8–R9 found calibration/cache/precision and valid
    resource-breach disposition incomplete.
- [x] T5-rereview-repair [R-001–R-010] — Repair R1–R9 without redesigning the
  accepted CYAX-0163 architecture.
  - Outcome: registered Claim types; literal-closed bundles; exact decision
    events; semantic assertion IDs/time; semantic snapshot projection; canonical
    assertion-ID removal deltas; byte-complete generator 2.1; Q01–Q12 selection
    and aggregation; pre-access calibration/cache/precision rules; and the
    resource/control classifier table are specified.
  - Verify: requirement/plan/task convergence, adversarial terminology/scope
    scan, frontmatter parse, diff checks, and independent exact-head rereview.
- [ ] T6 [R-001–R-010, CYAX-0168 G0] — Obtain a new independent
  architecture/methodology rereview, then repository-owner approval tied to the
  exact repaired head.
  - Outcome: approval or a bounded revision request.
  - Stop: `status: draft`, `approval_ref: null`, or any unresolved decision. Do
    not mark the rereview complete here.
- [ ] T7 [R-001, R-002, R-004–R-009, CYAX-0168 G1] — Under later dispatch,
  freeze source/registry/gold/generator/query/calibration manifests and v1
  validators only after G0.
  - Verify: identity, registry, generator checksum, query selection,
    reference-closure, tamper, freshness, cache-helper harness, and crash
    tests.
- [ ] T8 [R-003, R-007, R-009, CYAX-0168 G1] — Implement S, repeat approved-host
  Ladybug smoke, implement G only if it passes, and run calibration-only
  precision validation before any decision-fixture access.
  - Verify: offline pins, DDL/index/config/plan review, precision thresholds,
    ratified counts, and complete logical exports.
  - Stop: Ladybug failure; FalkorDBLite requires amended approval.
- [ ] T9 [R-004, R-006–R-008, CYAX-0168 G2] — Pass frozen gold, S/G parity,
  literal-closed bundle, full-export, delta, isolation, and crash-recovery gates.
  - Stop: preserve the minimal failure; no associated performance is admissible.
- [ ] T10 [R-009, R-010, CYAX-0168 G3] — Under later execution dispatch, run
  paired T0–T3 with cache and process-tree resource evidence and frozen
  aggregation.
  - Verify: controls, sample completeness, valid-breach disposition, block-level
    BCa output, and hard envelope.
- [ ] T11 [R-010, CYAX-0168 G4] — Conditionally run T4, classify, independently
  review evidence, and report to #168/#162.
  - Verify: five-part eligibility and deterministic classifier.
  - Stop: no production adoption or CYAX-0166/CYAX-0167 revision.

No benchmark implementation, dependency, generated database/fixture,
timing/resource run, CYAX-0166/CYAX-0167 edit, new rereview, or owner approval
is part of T5-rereview-repair.
