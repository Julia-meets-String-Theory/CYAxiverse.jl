# CYAX-0125 Gate A execution tasks

This checklist tracks execution and evidence readiness only. Issue #125 and
GitHub remain the live work-state record. The Approved spec at
`afc90e63f4703a53a51f3fd296e52edbac325862` is normative.

## Prerequisite evidence

- [x] Verify fresh `origin/vmm` and `origin/main` baseline identities and
  preserve unrelated worktrees. Evidence: spec plan baseline; exact remote
  commits recorded in the handoff and rechecked before branch creation.
- [x] Verify both durable owner decisions, including global DEV reservation.
  Evidence: Issue #125 comments `5746793852` and `5752268640`.
- [x] Pin the canonical SPEC/STANDARDS rubric and obtain independent draft
  reviews. Evidence: rubric SHA-256 in spec; both reviewers PASS on the exact
  Approved artifact `afc90e63f4703a53a51f3fd296e52edbac325862`.
- [x] Approve exact S2 spec before deriving this plan and checklist. Evidence:
  `spec.md` frontmatter, reviewed content revision `cf0f9c39256b7a58e179af92a46fbf1cb651ff76`.

## A1 — Policy and release-neutral source

- [x] Promote principal/maintenance lifecycle, reservation, version-impact,
  certification, Gate A/B and transition rules in policy/skills/workflow.
  Verify every changed rule agrees with R-001–R-004, R-010–R-011,
  R-036–R-038; stop on an unapproved normative choice.
- [x] Update README/installation/Documenter/workflow for ref-derived
  development, principal-versioned/stable and maintenance-versioned docs.
  Verify all three ref routes, generated API preservation and docs build;
  stop if source requires a release-time tracked edit.
- [x] Extend package-version checker with the shared canonical final/DEV
  parser. Verify Julia-equivalent positive cases and rejected aliases,
  prereleases and build forms; stop if a current pre-retrofit check regresses.

## A2 — Static authority and allocation

- [x] Add versioned `iterations.toml` schema and validator for prospective
  metadata, synthetic retrospective truth and protected anchor equality.
  Verify R-012, R-019–R-021 fixtures; stop before real Gate B population.
- [x] Implement exact canonical selector and four-digest static snapshot.
  Verify source/ref/anchor/tag identities, sort/preimage rules, tampering,
  stale source and unresolved-selector reason; stop if live selector cannot
  be established after merge.
- [x] Implement global occupied-set derivation and principal/maintenance
  allocation against exact static and mutable heads. Verify R-005–R-011,
  R-014–R-016 collisions, consumed reservation and no-reuse cases; stop on
  an unproven AVAILABLE designation.

## A3 — Mutable authority

- [x] Implement versioned canonical JSONL event schema, UTC/ID parser,
  type-specific transitions and append-only byte-prefix validator. Verify
  R-013, R-040, malformed/gap/duplicate/exhaustion and forbidden transition
  cases; stop on corrupt or ambiguous history.
- [x] Implement single protected orphan-event branch bootstrap and
  non-force expected-head append with idempotent uncertain-push recovery.
  Verify concurrent writers, changed static snapshot, transaction ID
  collision and frozen uncertain outcome; stop if exclusion/protection is
  unavailable.
- [x] Implement reservation prepare/open/consume/abort and same-line
  maintenance allocation state. Verify R-006–R-011, R-016–R-018,
  outgoing terminality and no-reuse; stop if a reservation outcome is
  uncertain.

## A4 — Closure and release lifecycle

- [x] Implement guarded closure anchor, explicit UTC time, outgoing
  reservation consumption and deterministic reopen state machine. Verify
  R-019, R-028–R-029, R-041 phase ordering, failed consumption and freeze;
  stop if closure/tree/time correspondence cannot be proved.
- [x] Implement first maintenance-line bootstrap with approved base,
  create-if-absent ref, immediate freeze, DEV install, reservation activation
  and line-open event. Verify R-030–R-031 uncertainty and no-work-before-
  correspondence; stop if bootstrap ref/protection state is unknown.
- [x] Implement one remotely durable candidate/open/withdrawal path and
  certification harness identity/transfer rules. Verify R-023–R-025,
  R-032–R-033, R-044, retained refs, changed trees and recertification;
  stop before tag if binding is unsupported or proof is missing.
- [x] Implement principal freeze/ancestry/exact-tree and maintenance
  lineage/release checks, durable intent, protected tag gate, released event,
  publication evidence and bidirectional validator. Verify R-008–R-009,
  R-026–R-027, R-034–R-035, R-042–R-043 with positive and pending/invalid
  fixtures; stop on unproven final version, main freeze or tag protection.

## A5 — Verification and convergence

- [x] Run focused event/allocation/closure/release/evidence tests first;
  record exact commands/results and synthetic release-evidence packet with
  every R-042 identity field and canonical storage location.
- [x] Run `git diff --check`, `agent_verify.py snapshot` and `diff-check`,
  package suite, audit, docs build, Python-free core import and workflow
  validation. Record the unchanged Hessian and two-JET baseline failures
  accurately.
- [ ] Observe remote CI for the final exact candidate; do not inherit results
  from superseded commit `877dca1`.
- [x] Converge spec ↔ plan ↔ tasks ↔ implementation ↔ evidence and sanitize
  PR content. Confirm `Project.toml=0.2.0`, no production event, DEV, release,
  tag, historical designation or Gate B mutation.

## Accepted review correction pass

- [x] Bind static and mutable allocation authorities to one canonical remote
  identity and reject cross-repository combinations.
- [x] Require separately verified global protection for every canonical
  `vX.Y.Z` tag while excluding legacy `v-0.1` and rejecting partial rules.
- [x] Require a fully validated durable `release_intent_prepared` event for
  forward tag reconciliation and bound maintenance patch allocation to
  `UInt32`.
- [x] Validate principal/maintenance main SHAs and canonical final versions,
  with exact before/at-event equality before the tag boundary.
- [x] Move CLI authority fetches into an isolated bare repository and prove
  the inspected checkout is unchanged on success and failure.
- [x] Correct serialized snapshot staleness, exact/bounded docs tag routing,
  certification banner publication, and pre-tag public-value sanitation.
- [x] Resolve documentation Git refs, commits, trees and package versions
  independently; keep external attestation/live-settings proof explicit as a
  later gate.
- [x] Re-run adversarial regressions for the caller-forgeable occupancy proof,
  complete orphan-ledger history/actual-parent binding, and explicit `vmm` CI
  fixture clone.

## A6 — Review, merge decision and live closure

- [x] Run independent SPEC/STANDARDS review on `52b2569`; record both
  `REQUEST_CHANGES` verdicts and correct strict remote-advertisement parsing,
  CLI writer encapsulation and complete durable-intent identity binding.
- [x] Run independent SPEC/STANDARDS review on `6d9a4d5`; record SPEC `PASS`
  and STANDARDS `REQUEST_CHANGES`, then correct tag-only peeling, strict
  ASCII/LF framing, remote-ref race detection and duplicated writer logic.
- [x] Run independent SPEC/STANDARDS review on `49147d7`; record both
  `REQUEST_CHANGES` verdicts, then route every remaining remote-advertisement
  reader through the strict shared parser and bind `candidate_id` across the
  candidate-open, prepared-intent and released-event identities.
- [ ] Obtain fresh independent exact-candidate SPEC/STANDARDS pre-merge
  reviews; reconcile all blocking findings and bind the final candidate SHA.
- [ ] Present final reviewed PR and evidence for an explicit owner merge
  decision. Review PASS does not constitute this decision.
- [ ] After authorized merge, establish and prove actual branch, anchor,
  maintenance, event and future canonical-tag protections/freeze controls.
  Record IDs, patterns, UTC retrieval, canonical snapshots/digests and
  observed enforcement without weakening existing `main` protection.
- [ ] Obtain fresh independent post-settings closure review of merged source,
  refs, event authority, protections and final verification. Mark Gate A
  complete only after findings are reconciled. Keep Gate B separate.

## Lesson outcome

- [x] Record this review round as a work-specific fail-closed testing lesson:
  normal Git fixtures do not exercise malformed or duplicate transport
  advertisements, so authority parsers need explicit adversarial records. No
  new repository-wide normative rule is required.
