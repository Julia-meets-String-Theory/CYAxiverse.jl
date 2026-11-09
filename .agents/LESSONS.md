# Lessons & Corrections

This is a small repository-wide registry of reusable failure patterns and
preventive checks. It is knowledge memory, not a diary, transcript archive,
personal scorecard, workflow ledger, or second authority system. Entries must
preserve the generalizable correction rather than the private event that led
to it.

## Authority and use

CYAX-0151's authority order remains unchanged:

```text
AGENTS.md + relevant CYAxiverse skill
    ↓
approved feature spec.md
    ↓
plan.md
    ↓
tasks.md
    ↓
GitHub Issue / Project
    ↓
implementation
```

Validated lessons are cross-cutting advisory inputs consulted after the
governing sources are known and before planning or implementation. They are not
an additional authority tier. Lessons never override `AGENTS.md`, a relevant
normative skill, an approved governing specification, or another artifact in
the approved hierarchy. If a lesson conflicts with an authoritative artifact,
do not silently follow the lesson: reconcile the conflict against the higher
sources and correct or supersede the lesson or update the appropriate artifact
through its normal review. Candidate lessons are proposals, not authority, and
are not a source of merge, Issue, Project, review, or scientific-acceptance
state. GitHub Issues, pull requests, and Projects retain their existing roles
for live work and status.

Read only applicable `validated` lessons for the task. Do not require every
agent to read the complete historical registry.

## Lifecycle and promotion

The knowledge lifecycle is:

```text
candidate → validated → superseded
```

- **candidate** records a sanitized, potentially reusable observation. It is
  advisory and must not be used as a substitute for a governing contract.
- **validated** means a reviewer found the pattern generalizable, accurately
  represented, privacy-safe, and supported by durable evidence. Validation does
  not promote the lesson into a normative policy.
- **superseded** retains historical traceability after a reviewed correction,
  narrower lesson, changed context, or normative promotion replaces the
  lesson. Set `Superseded by` to the replacement lesson or normative source.

The promotion ladder is local observation or correction → candidate → reviewed
validated lesson → repeated, stable, or high-impact rule → separately
reviewed promotion to `AGENTS.md`, the SDD skill, or another normative source.
Promotion requires the normal review and approval for the destination artifact;
it is never automatic. When a normative source adopts the rule, mark the lesson
`superseded`, record that repository-relative source and its reviewed revision
or approval reference in `Superseded by` and `Promotion status`, and stop
applying the lesson independently. If only part is promoted, replace the
original with a narrower candidate or validated lesson for any remaining
guidance. The historical entry remains traceable but inactive.

Owner redirections, private corrections, and local observations are first
sanitized and abstracted before becoming a candidate. Do not copy raw chat
text, private conversation or share URLs, transcript excerpts, local session
identifiers, machine-specific context, or resolving private references. If a
correction does not yield a generalizable prevention rule, do not add it.

## Compact lesson schema

Every entry uses these fields:

```text
ID: L-xxxx
Status: candidate | validated | superseded
Date: YYYY-MM-DD
Type: owner-redirection | peer-review | scientific-error | semantic-drift |
      scope-drift | state/provenance | tooling | testing | privacy | workflow
Scope / tags: ...
Observed failure: ...
Correction: ...
Root cause: ... (when known)
Preventive rule / check: ...
Applicability / exceptions: ...
Evidence / durable reference: ...
Superseded by: L-xxxx | <repository-relative normative source + section> | N/A
Promotion status: not promoted | promoted to <source> at <revision/approval ref>
```

## Seed lessons

The six initial entries were validated through the independent review and
repository-owner approval recorded in PR #160 comment `5636316397`. They are
deliberately limited to patterns supported by durable repository or public
GitHub history; they do not assert private conversation details.

## L-0001 — Candidate or repaired evidence is not scientific acceptance

ID: L-0001
Status: validated
Date: 2026-09-11
Type: scientific-error
Scope / tags: scientific acceptance, candidate evidence, gate review, Issue #148

Observed failure: Candidate or repaired evidence can be treated as though it
were an accepted scientific result before the required independent review and
owner approval have occurred.

Correction: Keep candidate, repaired, rejected, independently reviewed, and
accepted states distinct. Claim a scientific gate only from its durable review
and approval record, not from the existence of a repaired artifact or a
matching intermediate result.

Root cause: Evidence production was conflated with scientific gate acceptance.

Preventive rule / check: Before writing PASS, accepted, or population-level
language, identify the governing scientific gate and its required review,
approval, and evidence record. If that record is absent, use candidate or
provisional language and stop at the applicable gate.

Applicability / exceptions: Applies to scientific claims and gate records.
Process lessons may discuss candidate evidence, but they do not establish
scientific acceptance.

Evidence / durable reference: Issue #148 / PR #149 gate history, including the
durable candidate/rejection and later fresh-independent-review acceptance
records in repository history.

Superseded by: N/A
Promotion status: not promoted

## L-0002 — Namespace nested gates explicitly

ID: L-0002
Status: validated
Date: 2026-09-11
Type: owner-redirection
Scope / tags: gate namespacing, semantic drift, SDD, scientific review, Issue #151

Observed failure: Nested workflows use overlapping labels such as G2 or G3,
so a bare gate label can be read as a CYAX-0151 process gate or an Issue #148
scientific gate.

Correction: Always qualify a gate with its workflow or Issue, for example
`CYAX-0151 G3`, `#148 scientific G2`, or `#148 scientific G3`.

Root cause: Cross-workflow references omitted the namespace from a label that
was locally meaningful in more than one workflow.

Preventive rule / check: In specs, Issues, PRs, reviews, and handoffs, write
the process/Issue namespace with every gate status. Do not use an ambiguous
bare `G2` or `G3` when more than one gate vocabulary is in scope.

Applicability / exceptions: A single-workflow local note may abbreviate after
defining its namespace, but cross-workflow or durable status references retain
the explicit namespace.

Evidence / durable reference: Sanitized owner-redirection-derived pattern
checked against the durable SDD and pilot context in Issue #151, PR #154, and
Issue #148; no private correction or resolving private reference is retained.

Superseded by: N/A
Promotion status: not promoted

## L-0003 — Brownfield migration preserves supersession chronology

ID: L-0003
Status: validated
Date: 2026-09-11
Type: workflow
Scope / tags: brownfield migration, supersession, SDD, historical traceability

Observed failure: A brownfield migration can treat the original Issue body as
the final contract and silently rewrite historical work to match a later
interpretation.

Correction: Reconstruct a concise chronological decision and supersession
record from durable Issue wording, owner decisions, accepted gate records,
PRs, and validation artifacts. Use the latest durably owner-approved,
non-superseded normative intent as the migrated contract while preserving
earlier wording as history.

Root cause: Historical evidence and current normative intent were collapsed
into one retrospective narrative.

Preventive rule / check: Before migration, identify which durable decision
superseded each earlier statement; do not project newly introduced requirement
IDs backward onto historical execution. Stop for owner clarification when
precedence cannot be established safely.

Applicability / exceptions: Applies to brownfield migration and backfill.
Greenfield work starts from its current approved specification and does not
need a fabricated historical chronology.

Evidence / durable reference: `specs/0151-sdd-v1/clarification-a.md`, section 3,
and its durable SDD migration history in Issue #151 / PR #154.

Superseded by: N/A
Promotion status: not promoted

## L-0004 — `tasks.md` is not authoritative live status

ID: L-0004
Status: validated
Date: 2026-09-11
Type: state/provenance
Scope / tags: tasks, GitHub status, Project, merge state, SDD

Observed failure: An execution checklist in `tasks.md` is treated as the live
source for PR draft/ready state, merge state, Issue closure, Kanban state, or
review ownership.

Correction: Use `tasks.md` for subordinate execution decomposition and
evidence-readiness. Read current merge, Issue, Project, and review state from
GitHub's Issue, PR, and Project surfaces.

Root cause: The purpose of an execution artifact was confused with the live
workflow control plane.

Preventive rule / check: Before reporting current status, check the governing
Issue, linked PR, Project state, and current review owner. Do not mark a task
complete merely to imply that a PR merged or an Issue closed.

Applicability / exceptions: A task file may record planned or completed work
and evidence, but it must not become a second live status ledger.

Evidence / durable reference: `specs/0151-sdd-v1/clarification-a.md`, section 5,
and the corresponding Issue #151 clarification.

Superseded by: N/A
Promotion status: not promoted

## L-0005 — Do not falsify workflow state for advisory WIP limits

ID: L-0005
Status: validated
Date: 2026-09-11
Type: workflow
Scope / tags: WIP, GitHub Project, state integrity, SDD G2

Observed failure: Advisory WIP counters can create pressure to move or relabel
work so the displayed limits look satisfied even though the underlying work
has not changed state.

Correction: WIP counts and workflow states must reflect the actual governing
Issue state. If an advisory limit is saturated, report the saturation or seek
an explicit workflow decision; never fabricate a state transition to make a
dashboard green.

Root cause: A monitoring aid was treated as a hard objective rather than an
advisory view of reality.

Preventive rule / check: Reconcile the Project state with active Issues before
reporting WIP. Keep the experimental/advisory qualification visible and make
state changes only for real workflow transitions.

Applicability / exceptions: Applies to advisory Project WIP limits and status
reporting. It does not change the approved status lifecycle or scientific gate
criteria.

Evidence / durable reference: Issue #151 G2 PASS comment `5625939343` and
`specs/0151-sdd-v1/clarification-a.md`, section 6.

Superseded by: N/A
Promotion status: not promoted

## L-0006 — Machine-local paths are not scientific provenance

ID: L-0006
Status: validated
Date: 2026-09-11
Type: privacy
Scope / tags: scientific provenance, privacy, reproducibility, public paths

Observed failure: A machine-local path or execution-context locator is used
as public scientific provenance, even though it is private, non-portable, and
does not identify the source or artifact for independent replay.

Correction: Identify public scientific material with durable source/revision,
artifact or content hash, repository-relative paths, selection or witness
identity, and sanitized wording. Omit private or machine-specific context;
if safety is uncertain, stop for owner direction.

Root cause: A local execution locator was confused with a durable scientific
identity and publication-safe provenance record.

Preventive rule / check: Before a scientific or workflow write, scan paths and
provenance for absolute filesystem locations, usernames, hostnames, device or
session identifiers, and private attachment/context locators. Replace them
with durable public references or omit them under the privacy boundary.

Applicability / exceptions: Repository-relative paths and public revision/hash
references are allowed. Private local context may remain local; it is not
made public by being useful for debugging. This lesson contains no
Fuzzy/Table-1-specific claim.

Evidence / durable reference: `AGENTS.md` privacy and scientific-claim
boundaries; `specs/0155-private-safe-chat-checkpoints/spec.md`, R-001 and R-005;
Issue #157 / PR #158 public-path remediation history.

Superseded by: N/A
Promotion status: not promoted

## L-0007 — Handoff draft is not launch authorization

ID: L-0007
Status: candidate
Date: 2026-09-11
Type: owner-redirection
Scope / tags: delegation launch, authorization semantics, state/provenance

Observed failure: Discussion or refinement of a proposed delegation packet was
treated as launch authorization, and work/state advanced before explicit owner
authorization.

Correction: Treat discussion and refinement of a proposed packet as advisory, not
authorization. Present the final packet and obtain explicit owner authorization
before launch. After approval, ordinary already-authorized worker
diagnose/edit/check/correct/recheck work remains covered by that authorization.
Only a material post-approval change to authorized scope or terms requires
renewed approval.

Root cause: Plan updates were conflated with owner-governed launch authorization.

Preventive rule / check: Require a distinct pre-launch authorization checkpoint.
Treat packet edits as planning detail unless they materially change authorized
scope or terms; require renewed approval before applying such a change.

Applicability / exceptions: Applies to owner-redirection and delegated workflows
with explicit owner-governed execution. It does not replace scientific or merge
gates already governed by higher sources.

Evidence / durable reference: `AGENTS.md` section 6 and orchestration section 2
provide adjacent delegation mechanisms only. This sanitized candidate has no
cited public incident proof.

Superseded by: N/A
Promotion status: not promoted

## L-0008 — Delegation contract must name capability and budget

ID: L-0008
Status: candidate
Date: 2026-09-11
Type: workflow
Scope / tags: delegation packet, capacity planning, escalation policy

Observed failure: A delegated task was launched with implicit role, capability,
and effort assumptions, which reduced contract clarity and made fallback behavior
ambiguous.

Correction: Before delegation, state the role, model/capability target, reasoning
target, topology, ordered product-neutral fallback policy, effort budget/lease,
escalation conditions, and worker-reuse expectations. Do not encode a
concrete/current fallback sequence or model list.

Root cause: The packet lacked explicit operational bounds and fallback triggers.

Preventive rule / check: Reject a packet that omits any required contract field.
Verify that its fallback policy is ordered and product-neutral without encoding a
concrete/current sequence. Renew owner authorization only for a material change
to any authorized scope or term. Ordinary diagnose/edit/check/correct/recheck
loops and use of an already approved fallback or worker-reuse policy remain
authorized.

Applicability / exceptions: Applies to multi-agent and handoff-based work where
worker-level reuse is expected. Single-agent work without handoff may use a lighter
form.

Evidence / durable reference: `AGENTS.md` section 6 and orchestration sections
2, 5, 7, and 8 provide only partial, adjacent mechanism support for packet
fields and worker reuse. This sanitized candidate has no cited public incident
proof.

Superseded by: N/A
Promotion status: not promoted

## L-0009 — Handoff state follows observed execution outcome

ID: L-0009
Status: candidate
Date: 2026-09-11
Type: state/provenance
Scope / tags: handoff state, status integrity, orchestration

Observed failure: Handoff bookkeeping conflated the immediate operation result
with the later worker terminal return, so launch state could be inferred from the
wrong phase.

Correction: Use the immediate handoff operation result as the phase-1 launch
decision. Only success creates or attaches new execution task/worker state.
Decline, cancellation, or failure creates no new task/worker state, leaves prior
state unchanged, and is not silently retried. After successful launch, later worker
terminal states (`DONE`, `BLOCKED`, `FAILED`) are a separate phase-2 downstream
contract.

Root cause: State bookkeeping mixed handoff operation status with downstream return
state.

Preventive rule / check: Record phase 1 from the operation result and phase 2 from
the later worker terminal return. Do not make phase-2 evidence a prerequisite for
the phase-1 success transition; apply phase-2 checks only after successful launch.

Applicability / exceptions: Applies to workflows using explicit worker return
contracts and observed execution results. Internal planning notes may be prepared
before execution and do not by themselves move execution state.

Evidence / durable reference: `AGENTS.md` section 6 and orchestration section 3
provide adjacent support only for the later phase-2 `DONE`, `BLOCKED`, `FAILED`
worker-return contract. These sources do not prove the phase-1 incident or
define handoff-operation result semantics. This sanitized candidate has no
cited public incident proof.

Superseded by: N/A
Promotion status: not promoted

## L-0010 — Do not delegate directly executable owner-authorized repository actions

ID: L-0010
Status: candidate
Date: 2026-09-24
Type: workflow
Scope / tags: Control Desk, owner authorization, GitHub actions, delegation boundary, handoff minimization

Observed failure: A bounded repository action that had already reached its owner-decision boundary and was directly executable through the available authenticated GitHub surface was redirected to a separate execution context. This introduced an unnecessary handoff and extra owner coordination even though no downstream implementation worker was required.

Correction: Distinguish downstream delegated implementation from direct owner-authorized repository administration. When the owner explicitly authorizes a bounded action, all governing gates are already satisfied, and the current surface can execute the action directly, verify the exact current state and perform the action on that surface. Use a Manager, Work, or other downstream execution context only when the task actually requires delegated implementation, local execution, unavailable capability, or another governed execution role.

Root cause: The no-automatic-downstream-launch boundary was overgeneralized into a rule that repository mutations should be redirected to another execution surface. Orchestration role boundaries were conflated with direct connector capability and explicit owner authorization.

Preventive rule / check: Before creating or proposing a handoff, ask whether the requested action requires a downstream implementation worker or is an already-authorized direct action available on the current surface. If it is direct, bounded, gated, and executable here, perform it after exact-state verification rather than inventing a new execution boundary. Never use a handoff merely because the action mutates GitHub.

Applicability / exceptions: This lesson does not bypass specifications, required review, merge gates, scientific gates, privacy constraints, or owner-controlled authorization. If the action requires code changes, local testing, unavailable permissions, a distinct evidential role, or another execution context required by the governing contract, use the applicable handoff path instead.

Evidence / durable reference: Sanitized owner-redirection-derived pattern. The repository's delegation and lesson-lifecycle rules provide adjacent mechanism support; no private transcript, private URL, or resolving conversation reference is retained here.

Superseded by: N/A
Promotion status: not promoted

## L-0011 — Keep Handoff Review separate from specialist review roles

ID: L-0011
Status: candidate
Date: 2026-09-30
Type: workflow
Scope / tags: review topology, Handoff Review, SPEC, STANDARDS, SCIENTIFIC, Manager delegation, evidence reuse

Observed failure: A specialist review axis was routed directly through the Handoff
Review surface, conflating the workflow component that reviews a governed handoff
with the downstream specialist reviewer role that should be delegated by a Manager.

Correction: Keep the review topology explicit. Review Desk performs comprehensive
Handoff Review of governed Manager packets. After a current passing Handoff Review
and owner-manual Manager dispatch, the Manager delegates the applicable independent
SPEC, STANDARDS, and SCIENTIFIC reviewers. A specialist review accidentally produced
on the wrong surface may be retained as secondary corroborating evidence or a
double-check, but it does not satisfy the specialist-review gate and its verdict
must not be transferred into the correctly routed review.

Root cause: The review surface and the reviewer role were treated as interchangeable
because both can inspect the same candidate and rubric.

Preventive rule / check: Before activating any review, classify it explicitly as
either Handoff Review or a specialist review axis. If it is Handoff Review, route it
through Review Desk. If it is SPEC, STANDARDS, or SCIENTIFIC, require a Manager to
receive the governed packet first and delegate the specialist reviewer under the
bound capability policy. Preserve useful wrong-surface review evidence for
corroboration, but require a fresh correctly routed specialist review and never use
the earlier verdict to suppress findings, skip an axis, or establish acceptance.

Applicability / exceptions: Applies to governed delegated workflows that distinguish
Handoff Review from Manager-delegated specialist review axes. Ordinary informal peer
review outside that topology may use a lighter process, but must not be represented
as satisfying a governed specialist or Handoff Review gate.

Evidence / durable reference: Sanitized owner-redirection-derived pattern checked
against the repository's delegated-review topology and CYAX-0159 lessons lifecycle.
No private exchange path, transcript, or resolving private reference is retained in
this candidate.

Superseded by: N/A
Promotion status: not promoted

## L-0012 — Reclassify when review finds a scientific acceptance change

ID: L-0012
Status: candidate
Date: 2026-11-06
Type: owner-redirection
Scope / tags: SDD classification, S0/S2 boundary, scientific acceptance, review

Observed failure: A pull request classified as PR-only S0 (a test-suite
repair) also changed scientific acceptance criteria of the legacy minimizer
and, in one commit, a batch default. The classification was not revisited when
those changes appeared, so they were implemented and reviewed without a
governing specification or owner approval.

Correction: When a review or the implementation itself shows that a change
alters a scientific acceptance criterion, counting unit, default that selects
scientific results, or similar contract, stop and reclassify the work as S2,
open or reuse a governing Issue, and draft a specification for owner approval
before continuing that part of the work.

Root cause: The class was fixed at intake from the stated goal ("repair the
tests") and not re-checked against what the diff actually changed.

Preventive rule / check: At every review round, compare the diff against the
S2 triggers in the SDD skill (acceptance criteria, counting, persisted
scientific schema, scientific defaults). Any hit reopens the classification;
record the reclassification in the PR description.

Applicability / exceptions: Applies to S0 and S1 work whose diff grows. Pure
test or tooling changes that only observe scientific behaviour stay in their
class.

Evidence / durable reference: PR #215 and governing Issue #218;
`specs/0215-minimizer-acceptance-contract/spec.md`, *Status of this record* and
*Review and amendment history*.

Superseded by: N/A
Promotion status: not promoted

## L-0013 — Confirm the code path reaches a fix before claiming result impact

ID: L-0013
Status: candidate
Date: 2026-11-06
Type: scientific-error
Scope / tags: persisted results, impact claims, reachability, provenance

Observed failure: A draft specification and help text stated that stored
legacy vacuum counts were stale because of a minimizer defect. The legacy
pipeline never reached the minimizer: every such row failed earlier, by
construction. The premise had to be withdrawn after review.

Correction: Before claiming that a fix changes, invalidates or repairs
persisted or published results, establish from the call graph and a concrete
reproduction that the code path that produced those results actually executes
the fixed code, and record that evidence.

Root cause: The impact of a defect in a shared function was assumed to extend
to every caller, without checking reachability from the pipeline that wrote
the results.

Preventive rule / check: For each impact claim, name the producing entry point,
show it reaches the changed code on the relevant inputs (or show it cannot),
and state the claim only at that strength.

Applicability / exceptions: Applies to impact statements in specifications,
help text, release notes and Issues. It does not require re-running historical
populations.

Evidence / durable reference: `specs/0215-minimizer-acceptance-contract/spec.md`,
*Current baseline* and R-010; PR #215 review history (the withdrawn
stale-results premise).

Superseded by: N/A
Promotion status: not promoted

## L-0014 — Moving a check between CI jobs: compare what each pull request runs

ID: L-0014
Status: candidate
Date: 2026-11-06
Type: testing
Scope / tags: CI, gate coverage, job restructuring, qualification

Observed failure: A test-suite restructuring moved a process-isolated
qualification run out of the job that ran on every pull request into a new
job gated on a path list. The new gate missed several inputs of that run, and
pull requests to other bases no longer ran it, so some pull requests stopped
exercising a check they had run before. The independent review of the whole
change found it; the per-commit reviews had not.

Correction: When a check moves between CI jobs or behind a new condition, list
for each kind of pull request (by base branch, changed paths, labels, event)
whether it ran the check before and after, and close or explicitly accept every
difference. Derive a path gate from what the moved job actually loads, not from
the most obvious directories.

Root cause: The change was reviewed as "the check still exists" rather than
"every pull request that ran it still runs it".

Preventive rule / check: Attach a before/after table of trigger conditions to
any CI change that moves or gates a check, and test the gate against real
commit ranges.

Applicability / exceptions: Applies to CI workflow changes; not to adding new
checks that did not run before.

Evidence / durable reference: PR #215, G2 review record and commit `ec465e0`;
`scripts/ci_requires_qualification.py` and its tests; Issue #222.

Superseded by: N/A
Promotion status: not promoted

## L-0015 — Verify which ref produces the cache that new pull requests restore

ID: L-0015
Status: candidate
Date: 2026-11-06
Type: tooling
Scope / tags: GitHub Actions cache, CI performance, cache scope, quota

Observed failure: Every new pull request to the development branch `vmm`
precompiled the whole Julia test environment from scratch ("Cache not found";
479 packages). In this repository's workflow, caches were saved only on pull
request merge refs, which other pull requests did not restore, and CI did not
run on pushes to `vmm`, so no cache existed on the base branch. The
pull-request caches also filled most of the repository's 10 GB cache quota.

Correction: Produce the shared cache on pushes to the base branch, let
pull-request jobs restore it without saving large entries, and prune stale and
closed-pull-request caches in a trusted push-only job.

Root cause: Cross-pull-request cache reuse was assumed from the cache action's
defaults instead of checked against which refs actually wrote caches and which
refs a pull-request run can restore from.

Preventive rule / check: Before relying on cross-pull-request cache reuse,
identify the ref that creates the cache and verify, in the log of a fresh pull
request's run, that it restores that entry ("Cache restored from key" with the
expected key). Keep total cache size well under the quota.

Applicability / exceptions: Applies to repositories whose development branch is
not the default branch and to any workflow that relies on caches across pull
requests. Restore rules depend on GitHub's cache scoping and the workflow's
triggers; re-verify after changing either.

Evidence / durable reference: Issue #223; PR #224;
`specs/0223-ci-cache-lifecycle/spec.md`; CYAX-0223 G2 evidence comments on
#223 (PR #225 runs restored the `vmm` entry and precompiled 7 packages instead
of 479).

Superseded by: N/A
Promotion status: not promoted

## L-0016 — Julia depot garbage collection, cache options and `Pkg.test` interact

ID: L-0016
Status: candidate
Date: 2026-11-06
Type: tooling
Scope / tags: julia-actions/cache, Pkg.gc, Pkg.test, test-only dependencies

Observed failure: A cache change set `delete-old-caches: 'false'` in
`julia-actions/cache@v2` to keep test jobs read-only, assuming the option only
controlled deletion. In the action's source (`action.yml` and
`handle_caches.jl` at commit `d10a6fd8f31b`, the `v2` tag used by CI) that
option skips the whole post step, which also runs `Pkg.gc()`, so the change
would have stopped garbage collection of saved caches. A first replacement ran
`Pkg.gc()` as the last step, after `Pkg.test`.

What was observed and what was inferred:
- Observed in source (above): the post step skipped by the option runs
  `Pkg.gc()` before attempting deletions.
- Observed in a throwaway depot (Julia 1.12.6): `Pkg.gc()` deletes a package
  orphaned more than seven days earlier and keeps its first orphan date in
  `orphaned.toml`, so a package reinstalled at the same path is deleted again
  at the next collection.
- Observed in Pkg 1.12 source: `Pkg.test` resolves test-only dependencies in a
  temporary environment that is removed afterwards.
- Inferred (not observed in CI): running `Pkg.gc()` after `Pkg.test` can mark
  dependencies used only by the test environment as unused; once the
  collection delay has passed, those packages may be removed from subsequently
  saved caches. CI used Julia 1.12.7.

Correction: When turning off the action's cache deletion, run `Pkg.gc()`
explicitly, after the environments are set up and before the tests, so that
anything it deletes and the tests need is reinstalled and saved. Make the step
non-fatal, as upstream.

Root cause: Upstream action behaviour and Pkg's orphan bookkeeping were assumed
from option names instead of read from source.

Preventive rule / check: Before changing a cache option, read the action's
`action.yml` and helper scripts at the revision in use, and record that
revision. Reason about which manifests exist when `Pkg.gc()` runs, and state
which conclusions were observed and which were inferred.

Applicability / exceptions: Applies to Julia CI using `julia-actions/cache` with
`Pkg.test`, at the action and Pkg versions above; re-verify on upgrade. Jobs
that save no cache need no garbage collection.

Evidence / durable reference: PR #224; its review history in
`specs/0223-ci-cache-lifecycle/spec.md` (*Review history and owner decisions*);
commits `a87f4bf` and `ad73600`; `julia-actions/cache` commit `d10a6fd8f31b`.
CI evidence of packages deleted by the pre-test `Pkg.gc()` and then reinstalled
by the tests remains pending under CYAX-0223 G2 on #223.

Superseded by: N/A
Promotion status: not promoted

## L-0017 — Test skip semantics depend on the loader; named-file runners must fail closed

ID: L-0017
Status: candidate
Date: 2026-11-06
Type: testing
Scope / tags: Python unittest, optional dependencies, test runners, fail-closed

Observed failure: A test module was made to raise `unittest.SkipTest` at import
when CYTools is absent. Test discovery reported that as a skip, but a runner
that loaded the module by name (`python -m unittest <file>`) crashed, so the
repository's verification wrapper failed. Two successive fixes then broke the
runner's contract in other ways: `TestLoader.discover` also imported every
package under the directory, and importing by stem could run a different file
than the one listed and report a false pass.

Correction: A named-file test runner must not report success unless every
requested test file is correctly resolved and accounted for as executed tests
or an explicit skip. Reject invalid names and missing files before import,
verify the imported module's identity, turn an import-time `SkipTest` into an
explicit skip, fail on unexpected import errors and on test modules without
tests, and avoid unintentionally discovering additional test targets.
Post-import identity checks prevent false passes but do not guarantee that a
wrongly resolved module executed no code before rejection.

Root cause: Skip semantics differ between unittest loaders, and each fix was
first tested only on the default file list rather than against the runner's
contract.

Preventive rule / check: When adding an import-time skip, run every runner that
loads that module (discovery, by-name, wrappers). When writing or changing a
named-file runner, test that contract against shadowed names, packages beside
test files and import errors, not only the default list.

Applicability / exceptions: Applies to Python test tooling in this repository.
Discovery-based runs (`unittest discover`) remain the CI entry point.

Evidence / durable reference: PR #225 (commits `605bfb5`, `74ee31a`,
`9af7ef0`) and its review rounds recorded in the PR description;
`scripts/run_python_regression_tests.py`; commit `46f23b0`.

Superseded by: N/A
Promotion status: not promoted

## L-0018 — Row and column shapes in broadcast expressions

ID: L-0018
Status: candidate
Date: 2026-11-06
Type: scientific-error
Scope / tags: Julia broadcasting, numerical kernels, objective and Hessian

Observed failure: The legacy minimizer's objective and Hessian helper
broadcast the 1×N row `cos.(x' * QV)` against an N×N matrix and an N-vector,
so the optimizer received functions that were not V and its derivatives. The
result had the right type and no error was raised.

Correction: In numerical kernels, make the shape of every broadcast operand
explicit (vector vs 1×N row vs matrix), and test the objective and its
derivatives against a hand-derived value at a point where the terms do not
coincide (for example mixed-sign instanton scales).

Root cause: Broadcasting a row against a vector silently produces a matrix;
tests with uniform parameters could not detect it.

Preventive rule / check: For each objective, gradient and Hessian, add one test
with non-uniform parameters and a closed-form expected value, and break-check
it against the previous implementation.

Applicability / exceptions: Applies to Julia numerical code; the general check
(non-degenerate fixtures with independent expected values) applies to any
numerical kernel.

Evidence / durable reference: PR #215;
`specs/0215-minimizer-acceptance-contract/spec.md`, R-001 and its
verification row.

Superseded by: N/A
Promotion status: not promoted

## L-0019 — Review every file in delegated commits before pushing

ID: L-0019
Status: candidate
Date: 2026-11-09
Type: privacy
Scope / tags: delegation, commit review, build output, publication boundary

Observed failure: A pushed commit included Python bytecode files from a
`__pycache__` folder that `.gitignore` did not cover. Bytecode can embed
absolute source paths. The files were found after the push, before merge.

Correction: The branch history was rewritten to remove the files. The root
`.gitignore` now ignores every `__pycache__/` folder, not one directory only.

Root cause: Only one bytecode folder was ignored, and the pushed range was not
checked against the files that the task named.

Preventive rule / check: Before pushing commits made by another agent, read the
unfiltered `git diff --stat` for the whole range and `git show --stat` for each
commit, and reject any file that the task did not name, especially binary or
generated files. Tell delegated agents to stage named paths only.

Applicability / exceptions: Applies to every push of commits that the pushing
agent did not write itself. L-0006 covers paths in text that agents write; this
lesson covers files committed by accident. Removing a pushed file from the
branch does not remove the old commits from the hosting service.

Evidence / durable reference: PR #227 (the history rewrite is noted in its
description); commit `ff40cda` (`.gitignore`); `.agents/LESSONS.md` L-0006.

Superseded by: N/A
Promotion status: not promoted

## L-0020 — Test failure messages must not print the environment

ID: L-0020
Status: candidate
Date: 2026-11-09
Type: privacy
Scope / tags: tests, CI logs, environment variables, secrets

Observed failure: A test checked with `assertNotIn(name, environment)` that a
fixture removed Git variables. If it failed, the message would print the whole
environment dictionary, which can include access tokens, into the CI job log.
The risk was found and fixed before merge.

Correction: The test compares variable names only, and a comment states why.

Root cause: Default assertion messages print the full container or values. The
test passed the whole environment where a list of names was enough.

Preventive rule / check: When a test reads the process environment or another
store that can hold secrets, compare only the keys or the specific values that
the test controls, and check what the assertion prints on failure.

Applicability / exceptions: Applies to tests in any language that run in CI.
Values that the test itself sets may be compared.

Evidence / durable reference: PR #227; commit `861d9dc`
(`scripts/test_ci_requires_qualification.py`).

Superseded by: N/A
Promotion status: not promoted
