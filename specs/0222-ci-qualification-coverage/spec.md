---
spec_id: CYAX-0222
title: Qualification trigger coverage and CYAX-0223 follow-ups
issue: 222
class: S1
status: approved
workstream: CI workflow triggers, the qualification gate and their tests
parent: null
depends_on: []
created: 2026-11-06
last_reviewed: 2026-11-06
review_required: repository-owner approval
approval_ref: "PR #227 comment 6700824360 — approves spec revision 73c9f9b"
version_impact: none (CI only)
---

# Qualification trigger coverage and CYAX-0223 follow-ups

## Objective

A pull request that changes only `LocalPreferences.toml` or the root
`.gitignore` runs CI and scientific qualification before merge, the `CI.yml`
comment describes when qualification runs correctly, and the three accepted
follow-ups from CYAX-0223 are closed.

## Current baseline

Facts at `vmm` `34187c5`, recorded on #222 (comment 6700735948). That comment
corrects the Issue body, and this specification follows the comment.

- `paths` sits only under `pull_request` in `CI.yml`. Pushes and manual runs
  have no filter and always run qualification. A PR to a base other than `vmm`
  runs it whenever the workflow runs, which needs a change matching the filter.
- Julia applies a package's section of `LocalPreferences.toml` only if the
  project lists that package. The root `Project.toml` does not list
  `MPIPreferences`, so the tracked `[MPIPreferences]` section has no effect in
  the root environment (measured on Julia 1.12.6, with a positive control). A
  section for a listed package, such as `HDF5`, does reach the root project.
- `run_fixture.jl` records `clean` from
  `git status --porcelain=v1 --untracked-files=all` and raises an error when
  the source is not clean. CI resolves an untracked root `Manifest.toml`, which
  line 1 of `.gitignore` hides.
- Since PR #224, every push to `vmm` runs qualification. A failure caused by
  either file therefore appears after merge, on the merge push. This work moves
  it to before the merge.

## Scope

`.github/workflows/CI.yml` (the `pull_request` filter and one comment),
`scripts/ci_requires_qualification.py`, their tests,
`scripts/test_ci_cache_workflow.py`, and one accepted limitation in
`specs/0223-ci-cache-lifecycle/spec.md`.

## Non-scope

Removing `LocalPreferences.toml` or listing `MPIPreferences` in `Project.toml`;
the `Manifest.toml` entries in the filter and the gate, although the root file
is untracked; nested `.gitignore` files; the accepted policy that a PR to `vmm`
changing no qualification input does not run the release-pin check; job
conditions, test commands, and cache behaviour.

## Requirements

### R-001 — `LocalPreferences.toml` is a qualification input

The `pull_request` `paths` filter in `CI.yml` and `QUALIFICATION_PATHS` in
`scripts/ci_requires_qualification.py` SHALL include `LocalPreferences.toml`.
The gate's comment SHALL give the reason in the baseline above, not the
`MPIPreferences` dependency.

### R-002 — The `CI.yml` comment states when qualification runs

The comment above the `qualification` job SHALL say that pushes and manual runs
always run qualification, and that a PR to a base other than `vmm` runs it only
when the workflow runs, which depends on the `pull_request` `paths` filter. The
job condition SHALL NOT change.

### R-003 — The root `.gitignore` is a qualification input

The `pull_request` `paths` filter and `QUALIFICATION_PATHS` SHALL include the
root `.gitignore`. Paths such as `docs/.gitignore` SHALL NOT require
qualification: the `/Manifest.toml` rule is in the root file, and the job writes
its evidence to `$RUNNER_TEMP`.

### R-004 — Manual runs in the CYAX-0223 Full-run cost limitation

The accepted limitation on test-only packages in
`specs/0223-ci-cache-lifecycle/spec.md` SHALL list manual runs
(`workflow_dispatch`) among the Full test suite runs that bear the cost, and
SHALL state that a manual run on `main` is a new cost, like a push to `main`.
This amends an approved specification, so it follows the route the owner chose:
an in-place edit with a review-history entry, approved by an owner comment that
names the new commit, which replaces that specification's `approval_ref`.

### R-005 — The placement test counts every spelling it can see

`scripts/test_ci_cache_workflow.py` SHALL count garbage collection calls as
occurrences of the word `gc` in non-comment lines, so `import Pkg as P; P.gc()`,
`Pkg.gc(collect_delay=…)` and two calls on one line are all counted. Julia's
`GC.gc()` is also counted; that fails the test rather than hiding a call. The
test's comment SHALL state this rule instead of "any spelling", because a text
search cannot see a name built at run time.

### R-006 — The restore-only branch uses the same count

The restore-only branch of the same test SHALL assert that this count is zero,
in place of `assertNotIn("Pkg.gc()", block)`.

## Which PRs run what

Changes touching only the named file.

| Change | Event | Workflow runs (now / after) | Qualification (now / after) |
| --- | --- | --- | --- |
| `LocalPreferences.toml` or root `.gitignore` | PR to `vmm` | no / yes | no / yes |
| `LocalPreferences.toml` or root `.gitignore` | PR to any other base | no / yes | no / yes |
| `LocalPreferences.toml` or root `.gitignore` | Push to `vmm`, `main` or a tag | yes / yes | yes / yes |
| `LocalPreferences.toml` or root `.gitignore` | Manual run | yes / yes | yes / yes |
| Nested `.gitignore` only | PR | no / no | no / no |

## Accepted limitations

- The filter is at workflow level, so these PRs start every PR job, not only
  qualification (about 49 minutes on run 42255426058), and a PR to `main` also
  starts the Full test suite. In the last 12 months, 7 of the 9 non-merge
  commits on `vmm` that changed `.gitignore` touched no path in the current
  filter, an upper limit of about seven extra runs a year at that rate. The
  owner accepted this cost (see *Review history and owner decisions*).
- The `paths` filter is applied by GitHub, so tests can check only that the
  entries are present. CYAX-0222 G2 records the real behaviour.
- R-005 is a text check: a name built at run time is not counted.

## Acceptance gates

### CYAX-0222 G0 — Specification approval

The owner approves this specification in a durable comment that names its
commit SHA.

### CYAX-0222 G1 — Implementation (pre-merge)

- One commit per item, each with its test written first. Each test is shown to
  fail on the unfixed code, and to fail again when the fix is broken.
- `scripts/test_ci_requires_qualification.py` covers R-001 to R-003: both files
  in the gate and the filter, through a real Git commit range in its fixture, and
  nested paths excluded.
- `scripts/test_ci_cache_workflow.py` covers R-005 and R-006 with each spelling
  above and with a comment line that must not count. The workflow's current
  steps still pass unchanged.
- All Python tests pass. The PR's own CI run passes, including qualification,
  which the PR triggers because it changes `CI.yml`.

### CYAX-0222 G2 — Post-merge evidence (on #222, not a merge gate)

| Observation | Required record |
| --- | --- |
| First later PR changing only `LocalPreferences.toml` or `.gitignore` | The workflow ran, and the gate reported `qualification=true` |

## Interfaces and compatibility

CI only. No package, API, schema, environment, scientific or version change.

## Review history and owner decisions

- Owner decisions after the corrections in #222 comment 6700735948, made durable
  by the CYAX-0222 G0 approval comment: keep the `LocalPreferences.toml` item on
  the corrected reason; accept the cost of the `.gitignore` entry; amend
  CYAX-0223 by the route in R-004.

## Open owner decisions

None.

## Completion criterion

CYAX-0222 G0 and G1 pass and the PR is merged into `vmm`; CYAX-0222 G2 is
recorded on #222 when such a PR occurs.
