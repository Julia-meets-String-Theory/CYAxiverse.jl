# CYAX-0157 Ablation — Answer Key

**Status:** Held back from experimental subjects. Committed for auditability
before either reconstruction context exists.

**Snapshot:** `cyax-0157-ablation-20260913T0122Z`

**Authority boundary:** Every fact below is scoped to the frozen source
snapshot observed at 2026-09-13 approximately 01:22 UTC. No fact is invented
beyond what the canonical sources state at that observation.

---

## K1 — Governing Issue #157 identity and scope

Issue #157 ("Remove current machine-local path disclosures") requests removal
of confirmed personal and machine-local filesystem disclosures from current
public CYAxiverse surfaces while preserving scientific identity, numerical
behavior, persisted schemas, and reproducibility. The scope includes
sanitizing Issue/PR/comment bodies, replacing tracked personal path defaults,
preserving scientific algorithms and values, adding focused coverage, and
inventorying stale public branches as a separate live-reference class.

**Source:** `github:Julia-meets-String-Theory/CYAxiverse.jl#157` body at
SHA-256 `409b589c...` (fixture `sources/issue_157.json`)

## K2 — Supporting PR #158 identity and relationship

PR #158 ("Sanitize machine-local path configuration") implements the
remediation specified by Issue #157. It moves personal data and Slurm
locations to explicit environment configuration, replaces local paths with
configurable resolution, sanitizes documentation and notebooks, and adds
focused coverage. PR #158 does not complete Issue #157: the PR body
explicitly states "Issue #157 remains open. PR #158 does not complete R-005."

**Source:** `github:Julia-meets-String-Theory/CYAxiverse.jl#158` body at
SHA-256 `4dd0e855...` (fixture `sources/pr_158.json`)

## K3 — Issue OPEN and Project Verification at observation

At the observation time, Issue #157 state is OPEN and its Project status is
Verification on the "CYAxiverse Research & Development" project.

**Source:** GitHub API `state` and `projectItems[].status` fields for Issue
#157 as captured in `sources/issue_157.json`.

## K4 — PR #158 OPEN, non-draft, unmerged, current head/base, Project state

PR #158 is OPEN, non-draft (`isDraft: false`), unmerged (`mergedAt: null`),
and mergeable. Its head commit is
`8f6a9c28ac4b778b07f244dbbbc0076c86dc1a45` with tree
`fc980d28f3faa51f66031fe2e1fb5281b2ad5a48`. Its base branch is `vmm` at
commit `856012f015a866bf7ff352bc50e8d10c250855e6`. Its Project status is
Verification.

**Source:** GitHub API fields for PR #158 as captured in
`sources/pr_158.json`.

## K5 — Approved S1 spec and current-surface remediation at exact head

The CYAX-0157 specification (`specs/0157-public-path-remediation/spec.md`,
content SHA-256 `81c8ad7b...`) is approved for S1 implementation. The
specification header states: "Implementation status: PR #158 remediates the
reviewed working-tree surfaces, but R-005 remains open until every surviving
public branch is scanned and classified." The spec defines five requirements:

- **R-001** Sanitize current GitHub prose
- **R-002** Configure executable paths at runtime
- **R-003** Preserve scientific behavior and evidence identity
- **R-004** Preserve operational compatibility deliberately
- **R-005** Close every live-reference route before history review

At the current PR head, the current working-tree remediation surface is
implemented but R-005 is explicitly incomplete. The specification and PR do
not claim that merge would complete Issue #157.

**Source:** `specs/0157-public-path-remediation/spec.md` at blob
`1c670b6f...` from commit `8f6a9c28...`

## K6 — Verification and test boundary

The exact CI checks at the PR head show:
- **Fast tests:** COMPLETED, FAILURE (completed 2026-09-11T19:27:09Z)
- **Documentation build:** COMPLETED, SUCCESS (completed 2026-09-11T18:36:05Z)
- **Full test suite:** COMPLETED, SKIPPED (completed 2026-09-11T18:34:40Z)

The Fast test FAILURE is at the unchanged pre-existing Hessian expectation
mismatch at `test/runtests.jl:1317` (observed `19.739208802178716` versus
expected `39.47841760435743`). This mismatch predates PR #158 and is not
caused by the remediation.

The PR body reports that all new #157/#158 focused tests passed before the
suite reached the pre-existing mismatch: optional Python 9/9, notebook static
checks 20/20, notebook runtime initialization 10/10, retired-entrypoint
guards 54/54, Slurm-log resolution 6/6, and data-directory resolution 10/10.

**Source:** PR #158 checks as captured in `sources/pr_158_checks.json`;
PR #158 body verification section in `sources/pr_158.json`.

## K7 — Review-evidence state and limitation

The PR body reports: "independent re-review — APPROVE, no findings, against
the exact head above" (referring to `8f6a9c28...`). However, the GitHub API
`reviews[]` array for PR #158 is empty (fixture `sources/pr_158_reviews.json`).
No formal GitHub review decision (APPROVED, CHANGES_REQUESTED, or COMMENTED)
is exposed through the API at the observation time.

This means the independent re-review approval is self-reported in the PR
body. The GitHub review surface does not independently confirm it. A correct
reconstruction must note this evidence limitation rather than treating the
PR body claim as an independently verified GitHub review.

**Source:** PR #158 body in `sources/pr_158.json` (self-reported review);
`sources/pr_158_reviews.json` (empty).

## K8 — Unresolved R-005

R-005 ("Close every live-reference route before history review") requires
that surviving public branches receive a read-only privacy scan and
preservation/reachability classification before any branch deletion or history
rewriting. At the observation time, no public branch has been deleted,
force-pushed, or rewritten. The scan and classification have not been
performed. R-005 is the explicit completion gate for Issue #157.

Both the Issue body and PR body state this: "R-005 remains open and
deferred: surviving public branches still require the separate read-only
privacy scan and preservation/reachability classification."

**Source:** Issue #157 body in `sources/issue_157.json`; PR #158 body in
`sources/pr_158.json`; `specs/0157-public-path-remediation/spec.md` R-005
section.

## K9 — PR completion vs Issue completion

PR #158 must not auto-close Issue #157 upon merge. Merge and Issue completion
are distinct actions because R-005 remains open. The specification states
"PR #158 must not auto-close Issue #157 at this stage." The PR body states
"This PR must not close Issue #157." Merging PR #158 would incorporate the
current-surface remediation into `vmm`, but Issue #157 stays open until
R-005 is completed.

**Source:** `specs/0157-public-path-remediation/spec.md` implementation
status; PR #158 body in `sources/pr_158.json`.

## K10 — Prior head stale vs current head

The first PR comment (2026-09-11T13:49:52Z) requested independent re-review
against prior head `1ed97e181e3203524d5435c43f58c1a3ff03d813`. The second
comment (2026-09-11T20:50:22Z) requested re-review against the current head
`8f6a9c28ac4b778b07f244dbbbc0076c86dc1a45`, incorporating the current `vmm`
base `856012f015a866bf7ff352bc50e8d10c250855e6`.

The prior head `1ed97e18...` is stale: it has been superseded by the current
head. The current re-review request and the PR body verification both
reference only the current head. A correct reconstruction must not treat the
prior head as current.

**Source:** PR #158 comments in `sources/pr_158_comments.json`; PR #158
`headRefOid` in `sources/pr_158.json`.

## K11 — Uncertainties and abstentions

The following evidence limitations apply at the observation time:

- **No independent CI rerun:** The scorer/curator has not independently
  rerun CI. The CI results are from the GitHub Actions surface as recorded.
- **Timestamp-scoped GitHub surface:** All GitHub observations are scoped
  to the observation time (approximately 2026-09-13T01:22Z). State may
  change after the snapshot.
- **Self-reported verification:** The PR body verification section
  (focused test counts, review approval) is authored by the PR creator.
  It is not independently confirmed by GitHub review records or an
  independent test run.
- **Empty reviews array:** GitHub exposes no formal review decision,
  leaving the review status as a self-reported claim.
- **No production execution:** No production CYTools geometry generation,
  database scan, retired workflow, or external evidence generation was
  independently verified.
- **Branch surface not scanned:** R-005 branch scan has not been
  performed; the scope of remaining live-reference exposure is not yet
  determined.

A correct reconstruction must state these limitations explicitly and abstain
from claims that require evidence beyond the frozen snapshot.

**Source:** Derived from the observation boundaries of `source_snapshot.json`
and captured fixtures, and the explicit non-execution statements in the
PR body.

## K12 — Next valid action

The next valid action for this work item is:

- **Obtain and record an exact-head review and owner merge decision for
  PR #158** against head `8f6a9c28ac4b778b07f244dbbbc0076c86dc1a45`. This
  requires an independent review decision (formal GitHub review or recorded
  owner decision) and, if approved, an owner-authorized merge into `vmm`.
- **If PR #158 is merged, keep Issue #157 open.** Merge incorporates the
  current-surface remediation but does not complete R-005.
- **Separately perform R-005:** Execute the surviving public branch
  read-only scan, produce the preservation/reachability classification,
  and act on the results before any branch deletion or history rewriting.
  R-005 completion is the gate for closing Issue #157.

These are distinct actions. Merging PR #158 does not authorize branch
deletion, history rewriting, or Issue closure.

**Source:** `specs/0157-public-path-remediation/spec.md` R-005;
PR #158 body in `sources/pr_158.json`; Issue #157 body in
`sources/issue_157.json`.

---

## Scoring instructions

Apply each item K1–K12 independently. Score 1 if the response contains the
key fact with correct scope and source attribution. Score 0 if the fact is
absent, materially incorrect, or incorrectly scoped (e.g., claims merge
occurred, claims Issue is closed, treats prior head as current, omits
evidence limitation on review surface).

Check automatic failure conditions from `preregistration.md` before
computing the point total. Any automatic failure overrides the point score
to 0.
