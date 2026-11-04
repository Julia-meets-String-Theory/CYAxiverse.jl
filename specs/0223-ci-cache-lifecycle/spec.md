---
spec_id: CYAX-0223
title: Warm vmm Julia cache and CI cache lifecycle
issue: 223
class: S1
status: draft
workstream: CI workflow triggers, Julia cache identities, permissions and cache lifecycle
parent: null
depends_on: []
created: 2026-11-03
last_reviewed: null
review_required: repository-owner approval
approval_ref: N/A while draft
version_impact: none (CI only)
---

# Warm vmm Julia cache and CI cache lifecycle

## Objective

New pull requests to `vmm` restore a Julia depot cache produced on `vmm`
instead of precompiling the whole test environment, without pull-request runs
filling the repository's 10 GB Actions cache quota, and without giving
cache-deletion rights to jobs that run pull-request code.

## Current baseline

Facts from the CI logs and the repository cache list on 2026-11-03
(Issue #223):

- The first run of a new PR to `vmm` is cold: PR #190's Fast tests job reported
  "Cache not found" and precompiled 479 packages in 1549 s. Later runs of the
  same PR restore its own cache.
- `CI.yml` runs on push only for `main` and tags, so no CI cache exists on
  `vmm`. A PR can restore caches from its own merge ref and from the base or
  default branch, not from other PRs. GitHub's documentation does not state the
  order in which the base branch is searched; CYAX-0223 G2 records the ref that
  is actually restored.
- Storage was 9.6 GB of the 10 GB quota in 10 entries; PR #215's Fast tests
  entry was 2.1 GB.
- `julia-actions/cache@v2` deletes superseded entries only with
  `actions: write`; `CI.yml` grants `contents: read`, so superseded entries
  stay. On branches other than the default branch, a post step runs `Pkg.gc()`
  (default seven-day delay) and then tries the deletion;
  `delete-old-caches: 'false'` skips that whole post step, garbage collection
  included. It does not stop saving.
- Its default `cache-name` is `julia-cache;workflow=<workflow>;job=<job id>`;
  the primary key appends `os=<OS>;` and `run_id=…;run_attempt=…`, and restore
  uses the prefix up to `run_id=`.
- GitHub's `cache-mode` (`read`, `write`, `write-only`, `none`) is settable at
  workflow or job level; a save that the mode does not permit is logged and
  skipped without failing the job. The documentation does not say that
  `cache-mode` accepts expressions, so this specification uses fixed values.

## Scope

`.github/workflows/CI.yml` and tests of its wiring and of the cache-maintenance
selection logic.

## Non-scope

Committing `Manifest.toml`; changing which tests run on pull requests, their
commands or flags; the qualification gate and its coverage (#222);
recompilation after new registry releases; a scheduled cleanup (schedules run
only from the default branch, `main`); a hybrid that lets PRs save after a
base-cache miss.

## Requirements

### R-001 — Push trigger on `vmm`

`CI.yml` SHALL run on push to `vmm` as well as `main` and tags.

### R-002 — Fast tests cache producer

WHEN a push to `vmm` runs `CI.yml`, a job `fast-cache` SHALL run the same steps
as the Fast tests job (checkout, Julia 1.12 x64, Julia cache, build, notebook
environment, package tests with the same environment variables), then the
garbage collection of R-006, and save its depot cache under the Fast tests
cache name `julia-cache;workflow=CI;job=fast`. It SHALL NOT run on pull
requests, on pushes to `main` or on tags (a tag ref's cache can be restored only
by runs on that tag).

### R-003 — Restore-only Fast tests on pull requests

The Fast tests job SHALL keep its current condition (pull requests only) and
use the cache name `julia-cache;workflow=CI;job=fast` with job-level
`cache-mode: read`, so it restores the newest matching entry available to the
PR (its own ref, or the base or default branch) and saves none.

### R-004 — One writer per cache key

No two jobs that can run in the same workflow run SHALL use the same cache
name. `fast` (pull requests only) and `fast-cache` (push to `vmm` only) share a
name and never run in the same workflow run; every other job keeps its default,
job-specific name.

### R-005 — Full test suite reach unchanged

The Full test suite SHALL run exactly where it runs now: on pushes other than
to `vmm` (that is, `main` and tags), on manual runs, on PRs to `main`, and with
the `full-tests` label. It SHALL NOT run on push to `vmm`.

### R-006 — Read-only test jobs

Every job that checks out and runs repository or PR code SHALL keep the
workflow's `contents: read` and no other write permission, and SHALL set
`delete-old-caches: 'false'` on `julia-actions/cache`. Every job that can save
a Julia cache (lifecycle, `fast-cache`, qualification, Full) SHALL end with
`Pkg.gc()` at its default delay, replacing the garbage collection that the
skipped post step performed. That step is housekeeping, as upstream: it has
`continue-on-error: true`, so its failure neither fails the job nor stops the
cache save. The restore-only PR Fast tests job saves nothing and has no such
step.

### R-007 — Cache maintenance job

A job `cache-maintenance` SHALL:

1. run only on push events, in a dedicated concurrency group with
   `cancel-in-progress: false`, without waiting for other jobs, with
   `timeout-minutes: 10`;
2. have permissions `actions: write` and `pull-requests: read` only;
3. not check out the repository or run any repository or PR file; its logic is
   inline in `CI.yml` and calls only the GitHub REST API;
4. list all caches of the repository, oldest first by creation
   (`sort=created_at&direction=asc`), following every page up to 100 pages,
   into a fixed snapshot that holds each cache ID once, before selecting
   anything; a listing that does not end within 100 pages is an error;
5. group entries by restore identity: Git ref, the key up to `run_id=`, and the
   cache `version` (cached paths and compression);
6. select for deletion (a) every entry of a ref matching exactly
   `refs/pull/<digits>/merge` whose pull request is closed when checked;
   (b) every entry of such a ref whose key starts with
   `julia-cache;workflow=CI;job=fast;`, since pull requests no longer save that
   cache and such an entry would be restored instead of the `vmm` base; and
   (c) in every other group, every entry except the newest by `created_at`,
   compared as times, ties broken by cache ID;
7. never select the newest entry of a group on `refs/heads/vmm` or
   `refs/heads/main`;
8. delete only selected entries, by cache ID; log every selected and deleted
   ID, with refs and keys JSON-quoted so that no logged value can start a
   workflow command; log and skip an entry that is already gone (HTTP 404); and
   on any other API error stop, report failure (non-zero exit) and log what was
   already deleted, deleting nothing unselected. A failure before selection
   completes deletes nothing.

### R-008 — Pull-request behaviour unchanged

For pull requests, job conditions, test commands, `--check-bounds=yes`, the
qualification gate (`scripts/ci_requires_qualification.py`) and the
`qualification` job condition SHALL behave as before. Lifecycle, qualification
and Full keep saving their own caches.

## Accepted limitations

- PR Fast runs give up reuse of their own earlier runs: each run recompiles
  the PR's own changes and any dependency released since the last `vmm`
  warm-up. Expected to be acceptable for ordinary PRs; measured (CYAX-0223 G2).
- If the `vmm` Fast cache is missing, evicted or incompatible, every PR Fast run
  starts cold until the next push to `vmm`. Frequency and cost are recorded.
- Pull requests not based on `vmm` (release PRs to `main`, stacked PRs) run
  Fast tests cold on every run: `fast-cache` saves only on `vmm`, PRs no longer
  save, and maintenance deletes older PR Fast tests entries. Before this change
  such a PR reused its own cache after its first run. This follows from the
  owner's decision to produce the Fast tests cache on `vmm` only.
- `Pkg.gc()` can delete packages used only by the tests (for example
  CairoMakie, resolved in `Pkg.test`'s temporary environment) once they have
  been unused for seven days by the depot's own record, so a `vmm` push's saved
  cache can lack them until the next push. Upstream behaved the same on
  branches other than the default branch.
- Maintenance runs only on pushes. Between pushes, lifecycle, qualification and
  labelled Full runs on PRs still save entries (about 0.5, 0.4 and 1.6 GB), and
  GitHub can evict caches, including the `vmm` base. Obsolete entries are
  pruned at the next maintenance run; absence of eviction is not promised.
- Maintenance has its own concurrency group without cancellation. If a third
  maintenance job queues while one runs, GitHub replaces the older pending job,
  which is then shown as cancelled in its push's CI run. The newer job reads a
  fresh snapshot, so no pruning is lost.
- Each push to `vmm` whose CI run is not superseded runs lifecycle,
  `fast-cache`, qualification and maintenance. The workflow's concurrency group
  (one per ref, no cancellation for pushes) replaces a pending run when a newer
  push to `vmm` arrives, so the replaced push runs none of these jobs; the newer
  run saves the cache. Each such run takes about 30 minutes wall time and about
  50 runner-minutes on PR #215's figures. Qualification after each merge is a
  scheduling effect, not a change to #222.

## Acceptance gates

### CYAX-0223 G0 — Specification approval

The owner approves this specification in a durable comment that names its
commit SHA.

### CYAX-0223 G1 — Implementation (pre-merge)

- `scripts/test_ci_cache_workflow.py` checks the wiring of R-001 to R-008 in
  `CI.yml`, including the exact permission sets, the final non-fatal garbage
  collection step and the maintenance time limit, and, by executing the inline
  maintenance logic against a fake API, the selection rules of R-007: request
  order, pagination with a list that shifts between pages, de-duplication,
  grouping by ref, key prefix and version, newest kept by parsed time, closed,
  open and reopened PRs, PR Fast tests entries, strict ref pattern, protected
  `vmm`/`main` entries, quoted logging, the page limit, and failure without
  unselected deletion. Each test fails before the change it covers.
- All existing Python tests pass, including
  `scripts/test_ci_requires_qualification.py`.
- On the PR's own CI run: the Fast tests job attempts a restore (a miss is
  expected, because no `vmm` Fast tests cache exists before the first push to
  `vmm`), logs the skipped save, and no new `job=fast` entry appears on the
  PR's merge ref; the other jobs behave as before, and each job that saves a
  cache ends with `Pkg.gc()`.

### CYAX-0223 G2 — Post-merge evidence (on #223, not a merge gate)

| Observation | Required record |
| --- | --- |
| First push to `vmm` after merge | `fast-cache` saves a `job=fast` entry on `refs/heads/vmm`; maintenance log |
| Next new PR to `vmm` | Fast tests restores the `vmm` entry; key, ref and precompilation count |
| Repeated runs of one PR | No `job=fast` entry saved; precompilation time of later runs; total storage |
| PR with dependency releases newer than the `vmm` cache | Precompilation count and time |
| PR with no usable `vmm` cache | Recorded as a base-cache miss, with its cost |
| Push after a PR closes (same-repository or fork) | Only that PR's merge-ref entries deleted |
| Failed `fast-cache` on a `vmm` push | Previous `vmm` Fast entry kept |
| Every push to `vmm` | Size of the `vmm` Fast tests entry, to show whether `Pkg.gc()` bounds its growth; whether the `Pkg.gc()` step failed |
| PR runs after a `vmm` push whose `Pkg.gc()` deleted packages | Precompilation and download counts, to show the cost of packages missing from the base |

Single-run timings are evidence, not guarantees. The restore-only policy is
revisited if measurements show that repeated compilation makes PR iteration
substantially slower.

## Interfaces and compatibility

CI only. No package, API, schema, environment or version change.

## Review history and owner decisions

- Intake (#223): the owner decided the restore-only Fast policy, push-only
  maintenance and the accepted limitations, after two reviews of the design.
- The author then chose a separate push-only `fast-cache` producer sharing the
  Fast tests cache name, because `cache-mode` has no documented expression
  support.
- Independent review of `d80aa21` (spec and implementation): not passed.
  Blocking findings: unstable pagination could delete the only `vmm` Fast tests
  entry (B1); `delete-old-caches: 'false'` also skips `Pkg.gc()`, which the
  baseline omitted (B2); the pre-merge gate expected a restore that cannot
  happen on this PR (B3). Non-blocking findings included grouping without the
  cache version, string-compared times, unquoted logged keys, permission tests
  that missed added write permissions, the producer running on `main` and tags,
  and stale PR Fast tests entries shadowing the `vmm` base.
- Owner decisions after that review: add `Pkg.gc()` at the end of each job that
  saves a Julia cache; run `fast-cache` on pushes to `vmm` only; delete Fast
  tests entries on pull-request refs. The other findings are corrected as
  above.
- Independent re-review of `7c27bf8`: passed; all earlier findings closed. New
  non-blocking findings: PRs not based on `vmm` now always start Fast tests
  cold; `Pkg.gc()` can drop test-only packages from the `vmm` base; a
  `Pkg.gc()` failure would fail the job; the listing had no page cap and the
  job no time limit; a superseded `vmm` push runs none of the jobs. By owner
  decision all five are addressed before approval: the first, second and fifth
  as accepted limitations and CYAX-0223 G2 records, the third and fourth in
  R-006 and R-007.

## Open owner decisions

None.

## Completion criterion

CYAX-0223 G0 and G1 pass and the PR is merged into `vmm`; CYAX-0223 G2
evidence is then recorded on #223.
