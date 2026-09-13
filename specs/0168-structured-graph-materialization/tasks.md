# CYAX-0168 tasks

These tasks decompose evidence readiness. GitHub remains authoritative for live
Issue, Project, review, PR, and merge state.

- [x] T1 [R-004] — Audit #117 K1–K12 against captured canonical sources.
  - Outcome: corrected K7/K8/K10/K11 preserves the historical repaired replay
    and identifies the missing post-approval gates.
  - Evidence: Issue #117/comment `5556641428`, PRs #101/#103/#104, Issue #112,
    and `validation/orientifold_unaffected_projection_ledger_20260824.md` at the
    frozen repository revision.
  - Remaining gate: independent source review before CYAX-0168 or CYAX-0166 use.
- [x] T2 [R-001, R-002, R-007] — Repair the executable semantic contracts.
  - Outcome: closed enums, predicate signatures, Claim/literal representation,
    authority/time rules, source bundle, stable IDs, two identities, freshness,
    atomic publication, and `RetrievalBundle` v1 are specified.
  - Verify: requirement/plan/task convergence and independent rereview.
- [x] T3 [R-005, R-006, R-008] — Preregister scale, query, and update semantics.
  - Outcome: generator 2.0, profile/seed matrix, query-instance/gold rules,
    deterministic answer objects, and immutable N→N+1 transitions are frozen.
  - Verify: design inspection now; implementation tests only after approval.
- [x] T4 [R-003, R-009, R-010] — Repair fairness and decision methodology.
  - Outcome: stop-only fallback, calibration budget, host/cache/process controls,
    repetitions/BCa design, resource envelope, T4 gate, exact thresholds, and
    deterministic ambiguous-case classifier are frozen proposals.
  - Verify: design inspection now; independent methodology rereview and owner
    decisions remain required.
- [ ] T5 [R-001–R-010, CYAX-0168 G0] — Obtain independent source and design
  rereviews, then repository-owner approval tied to the exact repaired head.
  - Outcome: approval or a bounded revision request.
  - Stop: `status: draft`, `approval_ref: null`, or any unresolved decision.
- [ ] T6 [R-002, R-004–R-006, CYAX-0168 G1] — Under later dispatch, build and
  freeze source/gold/generator/query/calibration manifests and v1 validators.
  - Verify: identity/invariant/gold/tamper/freshness/crash tests.
- [ ] T7 [R-003, R-007, CYAX-0168 G1] — Implement S, repeat approved-host
  Ladybug smoke, and implement G only if it passes.
  - Verify: offline pins, DDL/index/config/plan review, complete logical exports.
  - Stop: Ladybug failure; FalkorDBLite requires amended approval.
- [ ] T8 [R-004, R-006–R-008, CYAX-0168 G2] — Pass frozen gold, S/G parity,
  full-export, transition, isolation, and crash-recovery gates.
  - Stop: preserve the minimal failure; no associated performance is admissible.
- [ ] T9 [R-009, R-010, CYAX-0168 G3] — Under later execution dispatch, run
  paired T0–T3 with process-tree resource evidence and frozen analysis.
  - Verify: controls, sample completeness, block-level BCa output, hard envelope.
- [ ] T10 [R-010, CYAX-0168 G4] — Conditionally run T4, classify, independently
  review evidence, and report to #168/#162.
  - Verify: five-part eligibility and deterministic classifier.
  - Stop: no production adoption or CYAX-0166 revision in this task.

No benchmark implementation, dependency, generated database, timing, resource
run, CYAX-0166/CYAX-0167 edit, rereview, or approval is part of T1–T4.
