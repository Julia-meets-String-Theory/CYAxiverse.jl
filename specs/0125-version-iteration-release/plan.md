# CYAX-0125 Gate A implementation plan

Status: derived from the Approved `spec.md` at commit
`afc90e63f4703a53a51f3fd296e52edbac325862` (blob
`5b0d2f80804e87eca0cee25869cf3f4bdf490c24`). The exact reviewed
normative content revision is `cf0f9c39256b7a58e179af92a46fbf1cb651ff76`.
Both independent SPEC and STANDARDS reviews passed under rubric SHA-256
`418f2d5a276cbdb74b8ad331b532d59219ef33b4e9fabc2b3d4a21c55bc06c72`.
Owner provenance is Issue #125 comments `5746793852` and `5752268640`.

## Boundaries and baseline

The branch starts from `origin/vmm` commit
`995163f0058488ea183ac645045ed8b1636bef4a` and tree
`36f9d101ec362ff91cf0d32005321cbda80acf2b`. Keep
`Project.toml = 0.2.0`; preserve `main` at its independent released version,
legacy `v-0.1`, Issue #172 history, scientific behavior and persisted scientific
schemas. Do not perform Gate B, real historical designation, production DEV
adoption, closure, publication, tag creation or `vmm → main` reconciliation.
The target iteration is `version-lifecycle-retrofit-2026-09`, version-bearing
package infrastructure with reviewed patch impact for later aggregation.

## Implementation slices

### A1 — Promote repository policy and release-neutral content

Update `AGENTS.md`, the SDD and integration-release skills, workflow guide,
PR template, README and installation docs to describe principal and
maintenance lines, target iteration, version impact, Gate A/B separation,
the global reservation rule, exact-tree certification and evidence, and
release-neutral instructions. Make `docs/make.jl` and Documentation workflow
choose development/versioned/stable channels from verified ref/event context.
Never require a tracked edit at public release. Keep existing generated API
docs. Add focused documentation-routing and version-checker tests.

### A2 — Version and static identity library

Implement a small standard-library Python package under
`scripts/version_lifecycle/` with explicit canonical codec and version
functions. Retain `scripts/check_version_bump.py` as the compatibility entry
point, backed by the shared parser. Add `iterations.toml` with a versioned
prospective/retrospective schema and no real retrospective entries. Implement
the exact `refs/heads/vmm:iterations.toml` selector, protected anchor/tag
enumeration, four-digest canonical snapshot, stale detection, occupied-set
derivation and global allocation view. Validate prospective anchor metadata
and historical declared-versus-actual truth separately. Use fixture Git
repositories for pre-merge tests because the canonical source is not live on
`vmm` until merge.

### A3 — Canonical event authority and allocation

Implement versioned canonical JSONL event schemas, all required event fields,
contiguous 12-digit IDs, UTC timestamps, transition validation and append-only
byte-prefix checks. Provide a protected `release-events` orphan-branch
bootstrap/verification path and non-force expected-head append. Append
reconciliation must be idempotent by transaction ID and canonical payload;
uncertain remote outcomes retain freezes. Model global occupation,
principal exact next sentinel, maintenance lowest available same-line patch,
DEV prepare/open/consume/abort, and no reuse. Static mutation and event
mutation use one controlled exclusion boundary; if its protection is not
available, writers return a blocked result.

### A4 — Closure, bootstrap, candidate and release transactions

Implement guarded transaction helpers for closure/anchor/outgoing-reservation
consumption/reopen and first maintenance-line bootstrap, with durable phase
evidence and forward recovery. Implement one candidate lifecycle with a
remote-durable ref, withdrawal retention, exact-tree certification and
explicit tree/commit binding. Implement principal main freeze/ancestry
disposition and maintenance-line release checks, durable release intent,
irreversible protected public-tag boundary, canonical released event,
publication evidence and bidirectional terminal validation. The Gate A
commands support dry-run and isolated fixture repositories. Production
mutations require the live protections and owner-approved release authority
that Gate A does not provide.

### A5 — Deterministic verification and CI

Add focused positive and negative tests for every state transition and
failure boundary in the approved spec. Run the focused suite first, then
`agent_verify.py` snapshot/diff-check/package/audit/docs/Python-free import,
workflow validation and remote CI as applicable. Record exact commands,
results, identities and unavailable checks. A failed or unobserved check is
never recorded as PASS. Avoid tests that merely mirror an implementation
branch; assert externally observable event bytes, Git refs, version strings,
release identities and status/reason outcomes.

### A6 — Integration, review and live controls

Converge the spec, plan, tasks, implementation, tests and sanitized PR
evidence. Obtain independent exact-candidate SPEC and STANDARDS review of
the implementation. The review is not merge authority. After an explicit
owner merge decision, merge the reviewed candidate, create/protect the
minimal orphan event branch, and verify `vmm`, `main`, line, anchor, event and
canonical-tag controls. Capture ruleset IDs/patterns, UTC retrieval time,
canonical snapshots/digests and observed enforcement. Obtain fresh
post-settings closure review of merged source and live state before marking
Gate A complete.

## Contract-to-evidence map

| Requirement | Implementation owner / proof |
| --- | --- |
| R-001 | A1 policy promotion; owner Issue refs and approved spec metadata; merge boundary audit. |
| R-002 | A1 line roles, A4 principal guard; main/principal fixture. |
| R-003 | A1 maintenance policy, A4 lineage guard; maintenance fixture with unchanged main. |
| R-004 | A1 target metadata, A2 registry; stable ID fixture. |
| R-005 | A2 exact grammar and checker; Julia equivalence probe plus final/DEV/invalid cases. |
| R-006 | A3 exact principal sentinel; unavailable blocked reason test. |
| R-007 | A3 same-line maintenance search; gap/competition test. |
| R-008 | A4 release guard; DEV closure/main/tag/publication rejection. |
| R-009 | A4 principal promotion guard; regression test. |
| R-010 | A3 global reservation; cross-line conflict test. |
| R-011 | A3 terminal consumption; different-final and no-reuse tests. |
| R-012 | A2 static schema, selector and anchor validator; registry fixture. |
| R-013 | A3 orphan event branch and single-ledger validator; topology test. |
| R-014 | A2/A3 exact combined allocation view; digest and head fixture. |
| R-015 | A2 retrospective/closed/anchor/tag occupancy; collision tests. |
| R-016 | A3 reservation/candidate/withdrawal/release occupancy; replay tests. |
| R-017 | A3 expected-head append; concurrent writer, stale and uncertain response tests. |
| R-018 | A2/A3 exclusion boundary; static mutation race test and blocked result. |
| R-019 | A4 annotated immutable closure anchor; exact SHA/tree fixture. |
| R-020 | A2 prospective metadata validator; required/forbidden field tests. |
| R-021 | A2 retrospective schema; synthetic historical mismatch fixture only. |
| R-022 | A1 content and docs routing; vmm/principal/maintenance channel tests and docs build. |
| R-023 | A4 immutable checkout certification; changed-tree rejection. |
| R-024 | A4 pinned independent identities; unsupported binding blocked test. |
| R-025 | A4 single candidate path; duplicate opening and ordering tests. |
| R-026 | A4 main freeze, ancestry disposition and equality; principal release fixture. |
| R-027 | A4 maintenance lineage/contemporaneous main; nonrollback fixture. |
| R-028 | A4 closure/reopen phase machine; ordered transition and freeze tests. |
| R-029 | A4 failed outgoing consumption; frozen/forward recovery result test. |
| R-030 | A4 first maintenance bootstrap; create-if-absent/CAS fixture. |
| R-031 | A4 bootstrap correspondence; uncertain creation/failure tests. |
| R-032 | A4 remotely durable candidate ref; retained terminal ref fixture. |
| R-033 | A3/A4 withdrawal consumption; no-reuse and retained ref test. |
| R-034 | A4 post-tag forward-only recovery; no rollback test. |
| R-035 | A4/A6 canonical-tag protection and enforcement evidence; unavailable reason test. |
| R-036 | A1 transition policy; legacy tag/#172 preservation diff audit. |
| R-037 | A1 package impact policy; package-facing/internal examples. |
| R-038 | All slices; unchanged package version and absence of production event/tag/adoption. |
| R-039 | A6 fresh merged-source/settings/refs closure review. |
| R-040 | A3 event schema/validator; malformed, gap, duplicate, exhaustion cases. |
| R-041 | A4 closure UTC anchor binding; malformed/inferred/mismatch cases. |
| R-042 | A4 release evidence schema and publication artifact; principal/maintenance field fixtures. |
| R-043 | A4 three-direction tag/event/Release validation; pending/INVALID cases. |
| R-044 | A4 tree-bound transfer and commit-bound recertification; exact subject tests. |

## Stop and escalation points

Stop a production writer on unresolved canonical selector, unavailable
exclusion/protection, ambiguous append outcome, uncertain branch creation,
incomplete outgoing reservation consumption, absent durable candidate,
unproven main freeze, unsupported certification binding, changed certified
tree, or missing tag/event/publication correspondence. Preserve the frozen
line/version and return the exact spec status/reason where defined. New
owner-level semantics or a changed normative choice require a spec amendment
and fresh review before the affected implementation continues.

The final pre-merge review and owner merge decision are separate gates.
Passing review never authorizes an actual package release or Gate B work.
