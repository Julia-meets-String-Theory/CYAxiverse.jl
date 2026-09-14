# CYAX-0168 implementation plan

## Status and sequencing

Draft final bounded rereview repair only. Exact head
`185d56dfced79fe9adc51c5573bcd3bd3d198d1f` received **PASS WITH REQUIRED
REVISIONS**; the common provenance architecture, B2, main B4, and B5 condition
3 passed, and the residual C1–C4 findings are repaired here. This plan exposes
the later execution sequence but
does not authorize benchmark code, fixtures, dependencies, timing, or resource
runs. The independent K1–K12 source audit has passed and is not reopened.
Architecture/methodology rereview and owner approval of the new exact revision
remain CYAX-0168 G0 prerequisites.
The earlier exact head `4005f60da7fa330cc964bbf3688b65fabb8dfd8b`
and its B1–B5 repair remain recorded as prior rereview history.

1. Freeze the immutable F-real source bundle, Claim-key/type registry, owner
   actor/role evidence, and exact owner-decision-event registry while preserving
   the passed independent K1–K12 chronology audit.
2. Implement and test the registered Claim/Literal model, complete
   literal-closed `RetrievalBundle` v1, and context-compiler input validator.
3. Implement assertion identity over every semantic state/provenance field,
   deterministic `asserted_at`, nonsemantic `built_at`, the semantic snapshot
   projection, and separate physical/build checksums.
4. Implement exact canonical deltas with `remove_assertion_ids[]`; prove
   dependency insertion/removal, replacement, supersession, rollback, crash
   interruption, and independent full-build N+1 equality without tombstones.
5. Implement generator 2.2's closed PRF-purpose registry and its exhaustive
   per-purpose self/duplicate rejection table, with phase-4 cycles outside PRF
   retries, assertion-stating synthetic provenance, and base versus
   additional source-revision rules. Require a second independent implementation
   to reproduce every byte, assertion identity, and complete snapshot checksum.
6. Freeze Q11 to the benchmark-local block-claim slot, Q02's query-local
   authority ordering, the Q01–Q12 decision-instance table, and independent
   gold; then implement the raw→instance→family/profile→paired-CI map.
7. Build and freeze only the direct scale-matched C0→T0, C1→T1, C2→T2,
   and C3→T3 non-decision calibration fixtures with their independent seeds;
   decision fixtures remain unmaterialized and inaccessible to either backend.
8. Implement the exact filesystem-cache-warm helper/evidence protocol, frozen
   empirical/lognormal/two-component calibration simulation, population
   rescaling and truth verification against the classifier's actual p95-derived
   estimands, statistical precision ratification, and exact scale-keyed
   campaign-duration categories; decision access remains prohibited.
9. Implement S as normalized indexed SQLite with bidirectional indexes,
   recursive CTE cycle/depth controls, one connection/thread, and complete
   logical export.
10. On the approved Linux host, freeze the complete Ladybug wheelhouse and repeat
   the offline smoke gate. If it fails, stop; any FalkorDBLite work requires an
   amended, rereviewed, owner-approved design. Otherwise implement minimal G.
11. Give S and G the same calibration-only tuning budget; hash final schemas,
   queries, configs, pragmas, dependencies, and plans; run statistical precision
   and 48-hour duration ratification; only after both pass materialize decision
   fixtures and complete
   CYAX-0168 G1 and G2 against frozen gold, S-versus-G parity, and full export.
   Preserve the smallest failure and stop before decision-fixture access on a
   precision, common-contract, or implementation failure.
12. Validate successor-snapshot transitions, independent clones, crash
   interruption, rollback/recovery, process-tree accounting, context identity,
   the both-cache-mode classifier truth table, exact family/profile joint
   quantifiers, and exhaustive reachable-outcome evaluation over all crossing
   CI predicates on synthetic records.
13. Under a later explicit execution dispatch, run the paired T0–T3 campaign on
   the approved host. Distinguish invalid controls from valid resource breaches
   and classify them by the frozen precedence.
14. Evaluate T4 condition 3 from direct T3 observations and condition 4 from
    the frozen T1–T3 monotone upper-envelope projection and exact `R<=0.75L`
    headroom rule; run T4 only if all five preregistered conditions pass. Apply
    the approved classifier,
    independently review the evidence, and return a bounded result
    to #168/#162 without production adoption.
15. Before owner approval, obtain a convergence-gated independent exact-head
    rereview. A blocker must demonstrate implementation/classification
    divergence, post-result discretion, authority/provenance/correctness
    violation, or a false guarantee under the five spec criteria; lesser notes
    are non-blocking implementation, G1-validation, or future-refinement items.

Later CYAX-0168 G1 may legitimately fail because statistical precision is
inadequate, the projected campaign exceeds 48 hours, approved-host Ladybug smoke
fails, or the resource envelope is infeasible. Such an experimental result
stops progression but does not reopen the design unless it demonstrates a
preregistration defect or an owner proposes a normative amendment.

## Requirement convergence

| Requirement | Planned artifacts/actions | Verification |
| --- | --- | --- |
| R-001 | Claim-key/type registry, v1 schemas, owner-decision registry, enum/predicate validators, authority/time evaluator | unregistered key/type failure; owner acknowledgement/question/proposal/non-owner adversaries; transition, signature, temporal, dispute/staleness tests |
| R-002 | complete semantic assertion IDs, `asserted_at`, content-addressed source bundle, semantic projection, physical/build checksums, freshness and atomic publishers | state-change ID, label-only stability, rule-version change, repeat-build identity, collision, tamper, stale/unavailable source and crash tests |
| R-003 | normalized S and pinned minimal G over identical snapshot input | DDL/index/config/plan review, no backend authority flags, complete-export equality |
| R-004 | F-real bundle, corrected Claim records, exact source anchors | preserve the passed independent K1–K12 audit; no chronology rewrite |
| R-005 | byte-complete generator 2.2, closed PRF registry and exhaustive purpose-specific rejection rules, assertion-stating provenance, plus T0–T4 profile/seed manifests | two independent implementations reproduce every source byte, record, assertion ID, and logical checksum; self/duplicate/valid/filler retries, phase-4 cycle-removal trace, and invariants |
| R-006 | exact Q11 slot, Q02 rank, Q01–Q12 selection table, reference evaluator, pre-backend instances and frozen gold | category exclusion, authority-order, population/selector/tie/parameter/invalid-role tests and deterministic gold checksums |
| R-007 | S/G adapters and canonical literal-closed `RetrievalBundle` v1 serializer/compiler input | empty/order/unique/no-dangling/reference-closure tests plus gold, pairwise, and complete-bundle equality |
| R-008 | canonical N→N+1 deltas with exact assertion-ID removals and independent full target | dependency insert/remove, replacement, supersession, target identity/export equality, crash/rollback/recovery and isolation tests |
| R-009 | C0–C3 direct scale matches, calibration non-access freeze, cache helper, actual-p95 truth simulation, scale-keyed duration report, host manifest, paired runner and exact aggregation | scale-map/no-cross-scale-reuse, unequal-dispersion and p95-saving truth, pre-access hash, cache evidence, precision and 48-hour thresholds, host/control audit, paired block-level BCa and resource accounting |
| R-010 | resource/control disposition, both-mode classifier, exact family/profile quantifiers, joint CI reachable outcomes, thresholds/precedence, direct-observation T4 condition 3, and deterministic projected-headroom condition 4 | fresh/warm, joint-quantifier, G/S/both timeout/breach, five joint-CI cases, boundary, relational-Hybrid, T4 margin/projection/headroom tables, followed by owner approval |

## Escalation and stop conditions

Return to the owner if F-real authority/chronology remains ambiguous; the
registered vocabulary must expand; semantics would differ by backend; an immutable
source cannot be captured; host fsync/process accounting is inadequate; the
Ladybug pin or offline smoke changes; a calibration/decision fixture leaks;
host, repetitions, statistical power, duration envelope, T4 gate, or thresholds
need amendment; or any
result is used to imply production adoption.

## Interface, lesson, and version impact

The only later CYAX-0166 interface is snapshot identity, the shared evaluator,
canonical `RetrievalBundle`, and bounded context compiler. No CYAX-0166/CYAX-0167
edit occurs here. This repair applies the validated chronology lesson by
preserving historical replay evidence and the validated evidence-state lesson
by not treating it as post-approval acceptance. It does not create a new
repository-wide lesson.

Version impact is none. All future benchmark dependencies remain optional and
outside the Julia package runtime.
