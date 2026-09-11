# CYAX-0157 — Remediate public machine-local path disclosures

Status: Approved for S1 implementation
Issue: #157
Spec class: S1
Version impact: none

## Objective

Remove confirmed personal or machine-local filesystem disclosures from current
public CYAxiverse surfaces while preserving scientific identity, numerical
behavior, persisted schemas, and reproducibility.

The scientific owner approved execution of the reviewed read-only remediation
manifest on 2026-09-10. The private conversation locator and local audit path
are intentionally omitted under the repository publication policy.

## Requirements

### R-001 — Sanitize current GitHub prose

Issue, PR, and comment text must retain public scientific provenance such as
source version, filename, content hash, branch, and commit, while removing
absolute local checkout, archive, worktree, and cache locations.

### R-002 — Configure executable paths at runtime

Tracked executable code must not contain personal home, HPC-account, worktree,
data-root, cache, or fixed local-interpreter paths. Repository locations must
be derived from `@__DIR__`. Data, log, manifest, output, archive, and optional
Python locations must come from explicit arguments or documented environment
variables. A generic container path may remain only when it is an intentional
portable deployment contract.

### R-003 — Preserve scientific behavior and evidence identity

For identical inputs and dependency versions, path sanitization must not alter
geometry selection, numerical algorithms, physical conventions, units,
acceptance criteria, persisted scientific fields, or reported scientific
values. Content hashes and commit identifiers remain public provenance.

When a path string participates in a manifest or generated evidence digest,
the artifact must be regenerated and its dependent hashes reconciled rather
than hand-edited. Path-only digest changes must be identified as such.

If faithful regeneration is impossible because a required bound input is no
longer available, the scientific owner may approve retirement of the complete
bound evidence set. Retirement must remove the set from the current validation
surface, preserve path-free artifact identities and hashes in a ledger, mark
the scientific claims as withdrawn, and require a new evidence version for any
future replacement. The owner approved this route on 2026-09-10 for the
2026-08-25 physical-scaling evidence set.

### R-004 — Preserve operational compatibility deliberately

Legacy data aliases may remain, but personal values must move to deployment-
local environment configuration. Missing configuration must fail with an
actionable message. Changed data, interpreter, and log resolution must have
focused regression coverage; executable notebooks require smoke coverage.

### R-005 — Close every live-reference route before history review

The remediation is incomplete while a current branch, Issue, PR, comment,
review, tracked artifact, or generated document still exposes a prohibited
fragment. Stale public branches must be classified and either removed after a
reachability/preservation check or sanitized if retained.

History rewriting must wait until every live surface is clean and a new scan
has identified the genuine history-only residue.

## Non-scope

- Changing any scientific normalization, population, basis, algorithm,
  acceptance criterion, units contract, or scientific schema.
- Discarding unique or unmerged branch work.
- Rewriting Git history in this implementation slice.
- Removing public source filenames, hashes, commits, or branch names solely
  because they record scientific provenance.
- Treating every generic `/home/$USERNAME` or container example as private
  without checking its intended deployment meaning.

## Implementation and verification

| Requirement | Implementation | Verification |
| --- | --- | --- |
| R-001 | Edit the confirmed Issue, PR, and comment bodies using the approved path-free wording | Read back every edited GitHub object and run a path-pattern scan over its current text |
| R-002, R-004 | Replace tracked personal defaults in `src/`, scripts, notebooks, and docs with explicit arguments, environment variables, or repository-relative resolution | Focused data-directory, Slurm-log, CLI, and optional-Python tests; notebook smoke checks; docs build |
| R-003 | Preserve hashes and scientific identifiers; regenerate any path-bearing, hash-bound evidence as one consistent set | Diff review of scientific values and schemas; bounded replay where an executable scientific driver changes configuration |
| R-003 owner-approved fallback | Retire the complete unavailable 2026-08-25 bound evidence set and add path-free Markdown and JSON retirement ledgers | Verify the ledger covers every removed artifact, reproduces its pre-retirement SHA-256 and Git blob identity, and makes no active scientific claim |
| R-005 | Scan every current public branch and discussion surface, then classify stale refs before deletion | Branch reachability/PR-status inventory plus a clean post-remediation live-surface scan |

## Acceptance

- Confirmed public prose no longer contains the user's local paths.
- Current `vmm` and the reviewed `vmm -> main` candidate contain no personal
  absolute path, local username-bearing topology, named local worktree/archive
  location, or fixed machine-specific interpreter path.
- Focused tests pass for each changed executable boundary.
- Scientific schemas, values, identities, and conventions are unchanged.
- Every surviving public branch has been scanned; branch deletion and any
  history decision are supported by explicit preservation/reachability
  evidence.
