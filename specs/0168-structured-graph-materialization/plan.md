# CYAX-0168 implementation plan

## Status and sequencing

Approved authority/host repair; CYAX-0168 G0 is satisfied and CYAX-0168 G1 is
the next gate under a later execution dispatch, not yet executed. The
convergence-gated independent architecture/methodology rereview returned
**PASS** at exact design head
`438aaaa69d4b965de29ea967cc05f02274f56e57`; that verdict remains technical
evidence. Issue #168 comment `5658274383` was recorded without a valid explicit
owner-decision checkpoint and must not be used as owner approval; the
correction is recorded in `authority-correction-a.md` and Issue #168
correction comment `5664014428`. Exact earlier head
`185d56dfced79fe9adc51c5573bcd3bd3d198d1f` received **PASS WITH REQUIRED
REVISIONS**; the common provenance architecture, B2, main B4, and B5 condition
3 passed, and the residual C1–C4 findings are repaired here. This plan exposes
the later execution sequence but
does not authorize benchmark code, fixtures, dependencies, timing, or resource
runs. The independent K1–K12 source audit has passed and is not reopened. The
independent bounded rereview of the exact repaired head
`f2283f6a1f58600d05ed0a50535d07013ac6fb2c` returned **PASS**; that verdict is
technical evidence only and did not by itself create owner authority. The
repository owner has since explicitly approved CYAX-0168 Decisions 1–7 in
Issue #168 comment `5668898317` (`approval_ref`), satisfying CYAX-0168 G0.
CYAX-0168 G1 is now the next gate; it requires a later execution dispatch and
has not yet been executed.
The earlier exact head `4005f60da7fa330cc964bbf3688b65fabb8dfd8b`
and its B1–B5 repair remain recorded as prior rereview history.
The independent macOS architecture/methodology rereview of exact head
`e8aa76fafb0eb015af0aa1bba45381e3021beb83` returned **PASS WITH REQUIRED
REVISIONS**. That repair was limited to its three blocking methodology findings.

A subsequent bounded repair replaces physical-machine authority with one
host-portable paired benchmark protocol: normative portable semantic, workload,
measurement, statistical, classifier, and claim contracts are separated from
campaign-specific host manifests; each campaign freezes one admissible host
manifest and executes S and G on that same host; results are host-scoped; raw
statistics are never automatically pooled across dissimilar hosts; and the
owner's Apple-silicon/macOS ARM64 machine is only the initial reference
execution host. Repetition counts, independent units, statistics, latency and
resource thresholds, the 48-hour ceiling, the `ladybug==0.20.4` candidate pin,
and every reviewed initial-macOS control are unchanged. Promotion to a normal
dependency, a project-wide default backend, or a production commitment
additionally requires independent replication on at least one further frozen
admissible host/environment. The independent bounded rereview of the new exact
repaired head `f2283f6a1f58600d05ed0a50535d07013ac6fb2c` returned **PASS**;
the repository owner has since approved those choices as Decisions 1–7 in
Issue #168 comment `5668898317`, satisfying CYAX-0168 G0. CYAX-0168 G1 is the
next gate and has not yet been executed.

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
8. Implement the portable `preconditioned-warm-cache` helper/evidence
   protocol in its initial macOS form, including ordered complete manifests
   before conditioning,
   strongest supported non-mutating query modes, and full post-pair file-set/
   logical/allocated-byte/hash/sparse/clone verification; implement the frozen
   empirical/lognormal/two-component calibration simulation, population
   rescaling and truth verification against the classifier's actual p95-derived
   estimands, statistical precision ratification, and exact scale-keyed
   campaign-duration categories; decision access remains prohibited.
9. Implement S as normalized indexed SQLite with bidirectional indexes,
   recursive CTE cycle/depth controls, one connection and one query-execution
   worker, and complete logical export. Freeze Ladybug to `THREADS=1` or its
   exact documented equivalent.
10. Once approved, on the campaign's frozen execution host — initially the
   exact frozen macOS ARM64 reference host — freeze the complete Ladybug
   wheelhouse and repeat the offline smoke gate for that host. If it fails,
   stop; any
   FalkorDBLite work requires an amended, rereviewed, owner-approved design.
   Otherwise implement minimal G.
11. Give S and G the same calibration-only tuning budget; hash final schemas,
   queries, configs, pragmas, dependencies, and plans; run statistical precision
   and 48-hour duration ratification. Complete CYAX-0168 G1 only from its
   calibration-only evidence, preserve the smallest failure, and stop. Decision
   fixtures remain inaccessible until a separate CYAX-0168 G2 dispatch.
12. Validate successor-snapshot transitions, independent clones, crash
   interruption, rollback/recovery, fresh-process macOS `ru_maxrss` accounting,
   named Mach diagnostics, logical/allocated storage accounting with one frozen
   APFS inspection API/algorithm and zero-shared-extent rebuild-or-exclude behavior, frozen
   Energy Mode, nominal thermal pressure, normal memory pressure, zero page-
   out/swap-I/O deltas, process topology, context identity,
   the both-cache-mode classifier truth table, exact family/profile joint
   quantifiers, and exhaustive reachable-outcome evaluation over all crossing
   CI predicates on synthetic records.
13. Under a later explicit execution dispatch, run the paired T0–T3 campaign
   with S and G on the campaign's single frozen admissible host, initially the
   macOS reference host, under its frozen manifest. Distinguish invalid controls
   from valid resource breaches and classify them by the frozen precedence.
14. Evaluate T4 condition 3 from direct T3 observations and condition 4 from
    the frozen T1–T3 monotone upper-envelope projection: logical materialization
    against cache/RAM feasibility, allocated materialization against the capped
    disk resource rule, and hard metrics against the exact `R<=0.75L` headroom
    rule. Run T4 only if all five preregistered conditions pass. Apply
    the then-reviewed and owner-approved classifier,
    independently review the evidence, and return a bounded host-scoped result
    to #168/#162 — reported with its campaign host manifest identity, without
    pooling raw statistics across hosts, and without production adoption.
Later CYAX-0168 G1 may legitimately fail because statistical precision is
inadequate, the projected campaign exceeds 48 hours, exact-host Ladybug smoke
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
| R-009 | C0–C3 direct scale matches, calibration non-access freeze, pre/post-manifest non-mutating cache protocol, actual-p95 truth simulation, scale-keyed duration report, initial campaign host/Energy-Mode manifest, one-query-execution-worker paired runner and exact aggregation | scale-map/no-cross-scale-reuse, unequal-dispersion and p95-saving truth, pre-access hash, cache stabilization and mutation invalidation, precision and 48-hour thresholds, power/Energy-Mode/thermal/memory-pressure/page-out/swap/descendant audit, both objective-detector and diagnostics-only competing-load branches, paired block-level BCa, fresh-process `ru_maxrss`, Mach diagnostics, and APFS zero-sharing/rebuild/exclude accounting |
| R-010 | explicit hard gates, capped relative memory/disk rules, resource/control disposition, both-mode classifier, exact family/profile quantifiers, joint CI reachable outcomes, unchanged latency thresholds/precedence, direct-observation T4 condition 3, and logical-versus-allocated deterministic T4 condition 4 | fresh/warm, capped-allowance boundaries, APFS clone ambiguity, G/S/both timeout/breach, five joint-CI cases, relational-Hybrid, T4 logical/allocated projection/headroom tables, host-scoped reporting with the pooling prohibition and replication requirement, followed by owner approval |

## Escalation and stop conditions

Return to the owner if F-real authority/chronology remains ambiguous; the
registered vocabulary must expand; semantics would differ by backend; an immutable
source cannot be captured; host fsync/process accounting is inadequate; the
Ladybug pin or offline smoke changes; a calibration/decision fixture leaks; no
admissible campaign host manifest can be frozen, or a portable measurement
control has no exact implementation on the intended host; a portable contract,
repetitions, statistical power, duration envelope, T4 gate, or thresholds
need amendment; raw statistics would be pooled across dissimilar hosts; or any
result is used to imply production adoption or replication that has not
occurred.

## Interface, lesson, and version impact

The only later CYAX-0166 interface is snapshot identity, the shared evaluator,
canonical `RetrievalBundle`, and bounded context compiler. No CYAX-0166/CYAX-0167
edit occurs here. This repair applies the validated chronology and evidence-state
lessons by preserving the independent review record without treating a drafted
or published approval statement as an actual owner decision. The reusable
authority-provenance failure is recorded as a candidate lesson pending review.

Version impact is none. All future benchmark dependencies remain optional and
outside the Julia package runtime.
