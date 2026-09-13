# CYAX-0168 implementation plan

## Status

Draft design only. No benchmark fixture generation, materializer
implementation, timing run, or production dependency is authorized until the
exact packet receives independent architecture/methodology review and
repository-owner approval.

## Approach

1. Freeze public sources for Issue #117, review K1–K12 against the canonical
   source revisions, and create a citation-bearing expected assertion set.
2. Implement canonical JSONL snapshot serialization, schema validation,
   semantic checksum calculation, and the shared CYAX-0163 disposition
   evaluator. Test tamper and stale-materialization rejection first.
3. Implement generator `cyax-0168-scale-1.0` with exact tier counts, frozen
   seeds/topology profiles, invariant checks, and repeat-generation hashes.
4. Implement the normalized indexed SQLite S materializer and adapter. Freeze
   DDL, pragmas, recursive-CTE depth/cycle rules, and query plans.
5. Repeat the Ladybug 0.20.4 smoke gate on the approved execution host. If it
   passes, implement only the minimal G materializer and adapter. If it fails,
   stop and amend/review the FalkorDBLite fallback before continuing.
6. Implement Q01–Q12 once as semantic query specifications with backend-specific
   candidate retrieval only. Canonicalize both outputs through the shared
   evaluator and `RetrievalBundle` serializer.
7. Run G1 and G2. Preserve minimal failing fixtures and stop on any mismatch;
   do not tune expected semantics to backend output.
8. Implement measurement wrappers for disk, build/update time, CPU, RSS,
   latency, adapter bytes, context bytes/tokens, randomized order, and raw
   manifests. Validate the analysis on synthetic timing records.
9. Under a later explicit dispatch, run T0–T3. Run T4 only if both backends pass
   every T3 correctness/resource gate. Apply the approved frozen outcome rules.
10. Obtain independent evidence review, publish a bounded report to #168, and
    return the result and any separate adoption decision to #162.

## Requirement mapping

| Requirement | Planned artifact/action | Verification |
| --- | --- | --- |
| R-001 | common schemas and evaluator | schema, admissibility, supersession, dispute, staleness tests |
| R-002 | snapshot manifest/checksum and backend metadata | rebuild/export equality plus tamper/stale/partial-build rejection |
| R-003 | SQLite DDL/indexes/CTEs | query-plan review and Q01–Q12 control results |
| R-004 | Ladybug pin and graph adapter | license/hash/offline/import/build/reopen/path-query smoke evidence |
| R-005 | #117 source bundle and known-answer records | citation review and golden semantic tests |
| R-006 | deterministic scale generator | exact counts, invariant checks, repeated byte hashes |
| R-007 | query specs, adapters, `RetrievalBundle` | contract validation and diagnostics-stripped byte equality |
| R-008 | correctness gate runner | automatic stop and minimal failing fixture |
| R-009 | measurement and analysis harness | synthetic calibration, raw manifests, independent methodology review |
| R-010 | classifier and final report | deterministic classification and owner handoff |

## Escalation conditions

Return to the owner before proceeding if the source authority or K1–K12 answer
key is ambiguous; the entity/predicate vocabulary must expand; semantic logic
would need backend-specific behavior; the Ladybug pin/smoke changes; a backend
requires network access during operation; the workload, resource envelope,
repetitions, or thresholds need amendment; or a result would imply production
adoption.

## Version impact

None. Benchmark dependencies remain optional and isolated from the Julia
package. Any public/package integration requires a separate reviewed scope and
the normal `vmm` → `main` release boundary.
