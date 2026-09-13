# CYAX-0168 implementation plan

## Status and sequencing

Draft repaired design only. This plan exposes the later execution sequence but
does not authorize benchmark code, fixtures, dependencies, timing, or resource
runs. Independent K1–K12 source review, architecture/methodology rereview, and
owner approval of the exact revision are CYAX-0168 G0 prerequisites.

1. Freeze the immutable F-real source bundle and independently review every
   K1–K12 Claim/anchor. Preserve the historical repaired replay while keeping
   it distinct from the required post-owner-approval replay.
2. Implement and test the closed Entity/Claim/Literal, enum, predicate,
   authority, temporal, stable-ID, source-bundle, snapshot, freshness, atomic
   publication, evaluator, and `RetrievalBundle` v1 contracts.
3. Implement generator 2.0 exactly for the approved profiles/seeds. Produce
   query instances and independent frozen gold before either backend adapter.
4. Build the non-decision calibration corpus. Give S and G the same frozen
   tuning budget, then hash schemas, queries, configs, pragmas, dependencies,
   and plans before decision fixtures are revealed.
5. Implement S as normalized indexed SQLite with bidirectional indexes,
   recursive CTE cycle/depth controls, one connection/thread, and complete
   logical export.
6. On the approved Linux host, freeze the complete Ladybug wheelhouse and repeat
   the offline smoke gate. If it fails, stop; any FalkorDBLite work requires an
   amended, rereviewed, owner-approved design. Otherwise implement minimal G.
7. Run CYAX-0168 G1 and G2 against frozen gold, then S-versus-G parity and full
   export. Preserve the smallest failure and stop before performance on any
   common-contract or implementation failure.
8. Validate successor-snapshot transitions, independent-clone repetitions,
   crash interruption, rollback/recovery, process-tree accounting, context
   identity, and analysis/classifier behavior on synthetic records.
9. Under a later explicit execution dispatch, run the paired T0–T3 campaign on
   the approved host. Keep invalid/resource-breach samples and classify them by
   the frozen precedence.
10. Run T4 only if all five preregistered conditions pass. Apply the approved
    classifier, independently review the evidence, and return a bounded result
    to #168/#162 without production adoption.

## Requirement convergence

| Requirement | Planned artifacts/actions | Verification |
| --- | --- | --- |
| R-001 | v1 schemas, enum/predicate validators, authority/time evaluator | closed-vocabulary, transition, signature, literal, authority, temporal, supersession/dispute/staleness tests |
| R-002 | content-addressed source bundle, framed IDs/checksums, freshness and atomic publishers | repeat-build identity; collision, tamper, stale/unavailable source, partial-build and stage-by-stage crash tests |
| R-003 | normalized S and pinned minimal G over identical snapshot input | DDL/index/config/plan review, no backend authority flags, complete-export equality |
| R-004 | F-real bundle, corrected Claim records, exact source anchors | independent K1–K12 source review before reuse |
| R-005 | generator 2.0 plus T0–T4 profile/seed manifests | exact counts/quotas/invariants and repeated byte/checksum equality |
| R-006 | reference evaluator, pre-backend query instances and frozen gold | stratum coverage, deterministic witnesses/minimal sets, gold checksum review |
| R-007 | S/G adapters and canonical `RetrievalBundle` v1 serializer | null/empty/order/dedup/direction/multiplicity/reference tests plus gold and pairwise equality |
| R-008 | immutable N→N+1 deltas and independently cloned backend transitions | target identity/export equality, crash/rollback/recovery and isolation tests |
| R-009 | calibration freeze, approved host manifest, paired runner and analysis | hash freeze, host/control audit, synthetic BCa/block-independence and resource-accounting validation |
| R-010 | exact threshold table and precedence classifier | boundary and ambiguous-case table tests, followed by owner approval |

## Escalation and stop conditions

Return to the owner if F-real authority/chronology remains ambiguous; the
closed vocabulary must expand; semantics would differ by backend; an immutable
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
