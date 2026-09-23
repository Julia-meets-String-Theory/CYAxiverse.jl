# CYAX-0183 — Deterministic Notion wiki refresh tooling v1

Status: draft  
Work class: S1  
Governing Issue: #183  
Version impact: none

## Objective

Provide a conservative Julia utility that detects material drift between the
CYAxiverse Notion wiki and authoritative GitHub/repository state, without
turning Notion into a second source of truth or allowing unattended scientific
rewrites.

## Authority and safety contract

1. GitHub, version-controlled source/specifications/evidence, and durable owner
   decisions remain authoritative.
2. The wiki is explanatory/derived.
3. The refresh tool MUST NOT make GitHub writes.
4. The unattended path MUST NOT receive a Notion write credential in v1.
5. Scientific, architectural, release, process-policy, and research prose is
   semantic content and MUST NOT be rewritten autonomously.
6. Private Notion page identifiers MUST NOT be committed to this public
   repository. Runtime page IDs come from a gitignored local mapping or an
   environment secret.
7. A semantic reconciliation advances its baseline only after an explicit
   acceptance action.
8. Missing/ambiguous authoritative evidence is a reason to abstain, not infer.

## Required behavior

### Manifest

A versioned manifest maps stable wiki page keys to:
- page title and authority class;
- tracked repository paths;
- relevant Issues and PRs;
- whether branch-head movement itself is material.

The public manifest contains no private Notion locators.

### State

A versioned state file records, per tracked page:
- exact last reconciled `vmm` commit;
- last reconciled tracked Issue snapshots;
- last reconciled tracked PR snapshots.

### `check`

Read-only. Compare current `vmm` state to the page baseline using exact Git
blob identity and tracked Issue/PR state. Report material drift and exit with a
distinct status.

### `verify`

Read-only. Validate current tracked repository paths. If a private page map and
`NOTION_TOKEN` are supplied, also validate Notion page accessibility/title and
Markdown retrieval.

### `reconcile`

For semantic pages, emit a bounded reconciliation packet and require human/agent
review. It MUST NOT rewrite semantic prose.

### `reconcile --accept PAGE`

After semantic reconciliation has occurred, an explicit accept step MAY:
- surgically update an existing `Verified against: vmm@...` marker through the
  Notion Markdown API;
- update local `wiki/state.json`.

It MUST refuse to invent a missing verification marker.

## Exit codes

- `0`: clean / successful operation
- `10`: material drift detected
- `11`: broken canonical source or remote verification failure
- `12`: semantic/owner review required

## CI

The scheduled CI job:
- has read-only repository/Issue/PR permissions;
- receives no Notion credential;
- runs unit tests and read-only drift checks.

PR CI tests the tool but does not run the drift gate as a merge blocker for
unrelated live-state changes.

## Non-scope

- full synchronization of every wiki page;
- autonomous scientific interpretation;
- a knowledge-graph authority layer;
- GitHub mutation;
- package release/version changes;
- root package dependency changes.

## Verification

Required before convergence:
- isolated Julia 1.12 environment instantiates;
- unit tests pass;
- public manifest contains no private Notion IDs;
- tracked source paths verify against current `vmm`;
- read-only `check` behaves correctly on the frozen initial baseline;
- workflow permissions are read-only and contain no Notion token.
