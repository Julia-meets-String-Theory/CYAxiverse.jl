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

## Architecture

The v1 boundary is intentionally split:

```text
Julia refresher
  -> reads GitHub/repository state
  -> detects material drift
  -> emits bounded reconciliation packet
  -> stops

Work / ChatGPT with connected Notion integration
  -> reads the packet and authoritative sources
  -> updates the affected Notion page through the connector
  -> reports the exact reconciled vmm commit

Julia refresher
  -> explicit accept of that exact commit
  -> advances wiki/state.json only
```

The Julia utility has **no direct Notion API client and no Notion credential
handling**.

## Authority and safety contract

1. GitHub, version-controlled source/specifications/evidence, and durable owner
   decisions remain authoritative.
2. The wiki is explanatory/derived.
3. The refresh tool MUST NOT make GitHub writes.
4. The refresh tool MUST NOT hold, read, or use a Notion API credential.
5. Scientific, architectural, release, process-policy, and research prose is
   semantic content and MUST NOT be rewritten autonomously.
6. Private Notion page identifiers and workspace locators MUST NOT be committed
   to this public repository.
7. Notion reads/writes occur through an already-authorized connected Notion
   integration in the human/agent reconciliation step.
8. A semantic reconciliation advances its local baseline only after an explicit
   acceptance action bound to the exact current `vmm` commit.
9. Missing/ambiguous authoritative evidence is a reason to abstain, not infer.

## Required behavior

### Manifest

A versioned public manifest maps stable wiki page keys to:
- page title and authority class;
- tracked repository paths;
- relevant Issues and PRs;
- whether branch-head movement itself is material.

It contains no private Notion locators or credentials.

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

Read-only. Validate current tracked repository paths and manifest/state
integrity. It does not contact Notion.

### `reconcile`

For affected semantic pages, emit a bounded reconciliation packet suitable for
Work/ChatGPT with the connected Notion integration. It MUST NOT rewrite Notion
or scientific prose itself.

### `reconcile --accept PAGE --reconciled-commit SHA`

After Work/ChatGPT has completed and verified the Notion reconciliation, an
explicit accept step MAY advance only local `wiki/state.json`.

Acceptance MUST:
- require the exact reconciled commit as an argument;
- require that commit to equal the current authoritative `vmm` head;
- verify all tracked source paths still exist;
- refresh the tracked Issue/PR snapshots;
- make no GitHub or Notion write.

## Exit codes

- `0`: clean / successful operation
- `10`: material drift detected
- `11`: broken canonical source or remote verification failure
- `12`: semantic/owner review required

## CI

The scheduled CI job:
- has read-only repository/Issue/PR permissions;
- runs unit tests and read-only drift checks;
- has no Notion credential or connector write action.

PR CI tests the tool but does not make external writes.

## Non-scope

- full synchronization of every wiki page;
- autonomous scientific interpretation;
- direct Notion REST/API access from Julia;
- a knowledge-graph authority layer;
- GitHub mutation by the refresh tool;
- package release/version changes;
- root package dependency changes.

## Verification

Required before convergence:
- isolated Julia 1.12 environment instantiates;
- unit tests pass;
- public manifest contains no private Notion IDs;
- source contains no Notion token/client/API path;
- tracked source paths verify against current `vmm`;
- read-only `check` behaves correctly on the frozen initial baseline;
- acceptance refuses a reconciled commit that differs from current `vmm`;
- workflow permissions are read-only.
