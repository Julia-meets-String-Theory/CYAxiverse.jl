# CYAX-0168 implementation plan

## Status and sequencing

Draft repaired design only. This plan exposes the later execution sequence but
does not authorize benchmark code, fixtures, dependencies, timing, or resource
runs. The independent K1–K12 source audit has passed and is not reopened.
Architecture/methodology rereview and owner approval of the new exact revision
remain CYAX-0168 G0 prerequisites.

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
5. Implement generator 2.1 exactly for the approved profiles/seeds, including
   every generated record field and exact cycle-edge replacement. Require a
   second independent implementation to reproduce the complete snapshot
   checksum.
6. Freeze the Q01–Q12 per-query decision-instance table and independent gold,
   then implement the raw→instance→family/profile→paired-CI aggregation map.
7. Build and freeze only the non-decision calibration corpus; decision fixtures
   remain unmaterialized and inaccessible to either backend.
8. Implement the exact filesystem-cache-warm helper/evidence protocol and the
   calibration-only repetition-precision harness; decision access remains
   prohibited.
9. Implement S as normalized indexed SQLite with bidirectional indexes,
   recursive CTE cycle/depth controls, one connection/thread, and complete
   logical export.
10. On the approved Linux host, freeze the complete Ladybug wheelhouse and repeat
   the offline smoke gate. If it fails, stop; any FalkorDBLite work requires an
   amended, rereviewed, owner-approved design. Otherwise implement minimal G.
11. Give S and G the same calibration-only tuning budget; hash final schemas,
   queries, configs, pragmas, dependencies, and plans; run precision validation
   and ratify counts; only then materialize decision fixtures and complete
   CYAX-0168 G1 and G2 against frozen gold, S-versus-G parity, and full export.
   Preserve the smallest failure and stop before decision-fixture access on a
   precision, common-contract, or implementation failure.
12. Validate successor-snapshot transitions, independent clones,
   crash interruption, rollback/recovery, process-tree accounting, context
   identity, and the complete resource/control classifier table on synthetic
   records.
13. Under a later explicit execution dispatch, run the paired T0–T3 campaign on
   the approved host. Distinguish invalid controls from valid resource breaches
   and classify them by the frozen precedence.
14. Run T4 only if all five preregistered conditions pass. Apply the approved
    classifier, independently review the evidence, and return a bounded result
    to #168/#162 without production adoption.

## Requirement convergence

| Requirement | Planned artifacts/actions | Verification |
| --- | --- | --- |
| R-001 | Claim-key/type registry, v1 schemas, owner-decision registry, enum/predicate validators, authority/time evaluator | unregistered key/type failure; owner acknowledgement/question/proposal/non-owner adversaries; transition, signature, temporal, dispute/staleness tests |
| R-002 | complete semantic assertion IDs, `asserted_at`, content-addressed source bundle, semantic projection, physical/build checksums, freshness and atomic publishers | state-change ID, label-only stability, rule-version change, repeat-build identity, collision, tamper, stale/unavailable source and crash tests |
| R-003 | normalized S and pinned minimal G over identical snapshot input | DDL/index/config/plan review, no backend authority flags, complete-export equality |
| R-004 | F-real bundle, corrected Claim records, exact source anchors | preserve the passed independent K1–K12 audit; no chronology rewrite |
| R-005 | byte-complete generator 2.1 plus T0–T4 profile/seed manifests | two independent implementations reproduce every record and logical checksum; exact cycle-removal trace and invariants |
| R-006 | exact Q01–Q12 selection table, reference evaluator, pre-backend instances and frozen gold | population/selector/tie/parameter/invalid-role tests and deterministic gold checksums |
| R-007 | S/G adapters and canonical literal-closed `RetrievalBundle` v1 serializer/compiler input | empty/order/unique/no-dangling/reference-closure tests plus gold, pairwise, and complete-bundle equality |
| R-008 | canonical N→N+1 deltas with exact assertion-ID removals and independent full target | dependency insert/remove, replacement, supersession, target identity/export equality, crash/rollback/recovery and isolation tests |
| R-009 | calibration non-access freeze, cache helper, precision report, host manifest, paired runner and exact aggregation | pre-access hash, cache read evidence, precision thresholds, host/control audit, paired block-level BCa and resource-accounting validation |
| R-010 | resource/control disposition table, exact thresholds and precedence classifier | G/S/both timeout and breach cases, CI overlap, boundary/table tests, followed by owner approval |

## Escalation and stop conditions

Return to the owner if F-real authority/chronology remains ambiguous; the
registered vocabulary must expand; semantics would differ by backend; an immutable
source cannot be captured; host fsync/process accounting is inadequate; the
Ladybug pin or offline smoke changes; a calibration/decision fixture leaks;
host, repetitions, envelope, T4 gate, or thresholds need amendment; or any
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
