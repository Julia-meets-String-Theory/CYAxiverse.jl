# CYAX-0168 tasks

These tasks decompose evidence readiness. GitHub remains authoritative for live
Issue, Project, review, PR, and merge state.

Exact head `185d56dfced79fe9adc51c5573bcd3bd3d198d1f` received **PASS WITH
REQUIRED REVISIONS**. The common provenance architecture, B2, main B4, B5
condition 3, and the independent #117 K1–K12 source audit passed and are not
reopened. Prior design
tasks T2–T4 are retained as superseded history; they do not establish that the
repaired contracts are complete. The convergence-gated independent
architecture/methodology rereview returned **PASS** at exact head
`438aaaa69d4b965de29ea967cc05f02274f56e57`, and the repository owner approved
that exact design in Issue #168 comment `5658274383`. **CYAX-0168 G0 is
satisfied.**

The earlier exact head `4005f60da7fa330cc964bbf3688b65fabb8dfd8b`
received its prior **PASS WITH REQUIRED REVISIONS** and remains the historical
input to T6-rereview-repair-2.

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
- [x] T6-rereview-repair-2 [R-005, R-006, R-009, R-010] — Repair residual
  B1–B5 preregistration defects without reopening accepted architecture.
  - Outcome: generator 2.2 freezes PRF purposes, rejection/collision behavior,
    and assertion-stating provenance; Q11 and Q02 are closed; precision and
    48-hour campaign ratification are reproducible; both cache modes and the
    family/profile quantifiers are Boolean; and T4 condition 3 uses observed
    statistics only.
  - Verify: independent byte-generation cases; query category/order cases;
    empirical/lognormal/two-component/null/alternative simulation cases;
    projected-over-48-hours failure; classifier truth table; deterministic T4
    margin cases; convergence, terminology, privacy, frontmatter, and diff checks.
- [x] T7-rereview-repair-3 [R-005, R-009, R-010] — Perform the final bounded
  C1–C4 repair without reopening accepted architecture or provenance sections.
  - Outcome: every generator PRF purpose has an exhaustive retry table;
    C0–C3 directly match T0–T3 and simulation truth uses actual p95-derived
    classifier estimands; CI crossings are evaluated as one joint uncertainty
    set; and T4 condition 4 uses a fixed monotone upper-envelope projection plus
    exact `R<=0.75L` headroom.
  - Verify: duplicate/self/valid/cycle/filler generator cases; scale-map,
    unequal-dispersion, p95-saving, and 95/100/105/125% truth cases; precision-
    pass/duration-fail and inverse G1 cases; five reachable-outcome CI cases;
    below/equal/above-75%, insufficient/censored/breach, and nonmonotonic T4
    projection cases; convergence, privacy, scope, frontmatter, and diff checks.
- [x] T8-convergence-rereview-owner-gate [R-001–R-010, CYAX-0168 G0] — Obtain a new independent
  architecture/methodology rereview, then repository-owner approval tied to the
  exact repaired head.
  - Outcome: independent rereview **PASS** at exact head
    `438aaaa69d4b965de29ea967cc05f02274f56e57`, followed by repository-owner
    approval in Issue #168 comment `5658274383`.
  - Evidence: the spec is `status: approved`, its `approval_ref` resolves to the
    approval comment, and CYAX-0168 G0 is satisfied.
- [ ] T9 [R-001, R-002, R-004–R-009, CYAX-0168 G1] — Under later dispatch,
  freeze source/registry/gold/generator/query/calibration manifests and v1
  validators only after G0.
  - Verify: identity, registry, generator checksum, query selection,
    reference-closure, tamper, freshness, cache-helper harness, and crash
    tests.
- [ ] T10 [R-003, R-007, R-009, CYAX-0168 G1] — Implement S, repeat approved-host
  Ladybug smoke, implement G only if it passes, and run calibration-only
  statistical-precision and campaign-duration validation before any
  decision-fixture access.
  - Verify: offline pins, DDL/index/config/plan review, precision thresholds,
    conservative 48-hour projection, ratified counts, and complete logical
    exports.
  - Stop: Ladybug failure; FalkorDBLite requires amended approval.
- [ ] T11 [R-004, R-006–R-008, CYAX-0168 G2] — Pass frozen gold, S/G parity,
  literal-closed bundle, full-export, delta, isolation, and crash-recovery gates.
  - Stop: preserve the minimal failure; no associated performance is admissible.
- [ ] T12 [R-009, R-010, CYAX-0168 G3] — Under later execution dispatch, run
  paired T0–T3 with cache and process-tree resource evidence and frozen
  aggregation.
  - Verify: controls, sample completeness, valid-breach disposition, block-level
    BCa output, and hard envelope.
- [ ] T13 [R-010, CYAX-0168 G4] — Conditionally run T4, classify, independently
  review evidence, and report to #168/#162.
  - Verify: five-part eligibility and deterministic classifier.
  - Stop: no production adoption or CYAX-0166/CYAX-0167 revision.

No benchmark implementation, dependency, generated database/fixture,
timing/resource run, CYAX-0166/CYAX-0167 edit, or production-backend adoption
is part of this approval-state synchronization. A later G1 failure is a valid
experimental outcome and does not reopen design unless it exposes a
preregistration defect or an owner proposes a normative amendment. CYAX-0168 G0
is satisfied; CYAX-0168 G1 is the next gate, and its
implementation/calibration tasks remain incomplete.
