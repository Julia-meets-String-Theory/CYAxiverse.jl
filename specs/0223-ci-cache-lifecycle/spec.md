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
  `vmm`. A PR can restore caches from its own merge ref, its base branch and the
  default branch, not from other PRs.
- Storage was 9.6 GB of the 10 GB quota in 10 entries; PR #215's Fast tests
  entry was 2.1 GB.
- `julia-actions/cache@v2` deletes superseded entries only with
  `actions: write`; `CI.yml` grants `contents: read`, so superseded entries
  stay. Its `delete-old-caches: false` controls deletion only, not saving.
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

WHEN a push event runs `CI.yml`, a job `fast-cache` SHALL run the same steps as
the Fast tests job (checkout, Julia 1.12 x64, Julia cache, build, notebook
environment, package tests with the same environment variables) and save its
depot cache under the Fast tests cache name
`julia-cache;workflow=CI;job=fast`. It SHALL NOT run on pull requests.

### R-003 — Restore-only Fast tests on pull requests

The Fast tests job SHALL keep its current condition (pull requests only) and
use the cache name `julia-cache;workflow=CI;job=fast` with job-level
`cache-mode: read`, so it restores the newest matching entry (its PR ref, then
`vmm` or `main` as base) and saves none.

### R-004 — One writer per cache key

No two jobs that can run in the same workflow run SHALL use the same cache
name. `fast` (pull requests only) and `fast-cache` (push only) share a name and
never run in the same workflow run; every other job keeps its default,
job-specific name.

### R-005 — Full test suite reach unchanged

The Full test suite SHALL run exactly where it runs now: on pushes other than
to `vmm` (that is, `main` and tags), on manual runs, on PRs to `main`, and with
the `full-tests` label. It SHALL NOT run on push to `vmm`.

### R-006 — Read-only test jobs

Every job that checks out and runs repository or PR code SHALL keep the
workflow's `contents: read` and no other write permission, and SHALL set
`delete-old-caches: 'false'` on `julia-actions/cache`.

### R-007 — Cache maintenance job

A job `cache-maintenance` SHALL:

1. run only on push events, in a dedicated concurrency group with
   `cancel-in-progress: false`, without waiting for other jobs;
2. have permissions `actions: write` and `pull-requests: read` only;
3. not check out the repository or run any repository or PR file; its logic is
   inline in `CI.yml` and calls only the GitHub REST API;
4. list all caches of the repository, following every page, into a fixed
   snapshot before selecting anything;
5. group entries by restore identity: Git ref plus the key up to `run_id=`;
6. select for deletion (a) every entry of a ref matching exactly
   `refs/pull/<digits>/merge` whose pull request is closed when checked, and
   (b) in every other group, every entry except the newest by `created_at`;
7. never select the newest entry of a group on `refs/heads/vmm` or
   `refs/heads/main`;
8. delete only selected entries, by cache ID; log every selected and deleted
   ID; log and skip an entry that is already gone (HTTP 404); and on any other
   API error stop, report failure (non-zero exit) and log what was already
   deleted, deleting nothing unselected. A failure before selection completes
   deletes nothing.

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
- Maintenance runs only on pushes. Between pushes, lifecycle, qualification and
  labelled Full runs on PRs still save entries (about 0.5, 0.4 and 1.6 GB), and
  GitHub can evict caches, including the `vmm` base. Obsolete entries are
  pruned at the next maintenance run; absence of eviction is not promised.
- Each push to `vmm` runs lifecycle, `fast-cache`, qualification and
  maintenance (about 30 minutes wall time, about 50 runner-minutes on
  PR #215's figures). Qualification after each merge is a scheduling effect, not
  a change to #222.

## Acceptance gates

### CYAX-0223 G0 — Specification approval

The owner approves this specification in a durable comment that names its
commit SHA.

### CYAX-0223 G1 — Implementation (pre-merge)

- `scripts/test_ci_cache_workflow.py` checks the wiring of R-001 to R-008 in
  `CI.yml` and, by executing the inline maintenance logic against a fake API,
  the selection rules of R-007: pagination, grouping, newest kept, closed and
  reopened PRs, strict ref pattern, protected `vmm`/`main` entries, logging,
  and failure without unselected deletion. Each test fails before the change.
- All existing Python tests pass, including
  `scripts/test_ci_requires_qualification.py`.
- On the PR's own CI run: the Fast tests job restores, logs a skipped save, and
  no new `job=fast` entry appears on the PR's merge ref; the other jobs behave
  as before.

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

Single-run timings are evidence, not guarantees. The restore-only policy is
revisited if measurements show that repeated compilation makes PR iteration
substantially slower.

## Interfaces and compatibility

CI only. No package, API, schema, environment or version change.

## Open owner decisions

None. The restore-only Fast policy, push-only maintenance and the accepted
limitations were decided by the owner at intake and are recorded in the #223
Issue body.

## Completion criterion

CYAX-0223 G0 and G1 pass and the PR is merged into `vmm`; CYAX-0223 G2
evidence is then recorded on #223.
