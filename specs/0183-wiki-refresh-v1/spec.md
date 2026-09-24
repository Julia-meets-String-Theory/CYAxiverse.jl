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
  -> emits an exact snapshot-bound reconciliation packet
  -> stops

Work / ChatGPT with connected Notion integration
  -> locates exactly one matching wiki page under the CYAxiverse wiki
  -> reads the packet and authoritative sources
  -> updates the affected Notion page through the connector
  -> verifies the edit
  -> explicitly attests that the exact packet was reconciled

Julia refresher
  -> validates the packet digest and replays its GitHub snapshot
  -> requires live GitHub state to match that exact packet
  -> advances wiki/state.json atomically
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
8. A reconciliation acceptance MUST be bound to one exact packet snapshot, not
   merely to a Git commit.
9. Missing/ambiguous authoritative evidence is a reason to abstain, not infer.

## Manifest

A versioned public manifest maps stable wiki page keys to:
- page title and authority class;
- tracked repository paths;
- relevant Issues and PRs;
- whether branch-head movement itself is material.

It contains no private Notion locators or credentials.

## State

A versioned state file records, per tracked page:
- exact last reconciled `vmm` commit;
- last reconciled tracked Issue snapshots;
- last reconciled tracked PR snapshots;
- accepted packet identity when a packet-based reconciliation has occurred.

The state file is operational reconciliation state, not project authority.

## `check`

Read-only. Compare current `vmm` state to the page baseline using exact Git
blob identity and tracked Issue/PR state. Report material drift and exit with a
distinct status.

A requested `--page` key MUST exist in the manifest; an unknown key MUST fail
closed rather than reporting clean.

## `verify`

Read-only. Validate:
- manifest structure and privacy constraints;
- state structure and manifest/state page coverage;
- repository/branch identity;
- tracked source-path existence at current `vmm`.

It does not contact Notion.

## `reconcile`

For each affected page, emit a deterministic packet with a cryptographic
`packet_id` covering at minimum:
- repository owner/name/branch;
- page key/title;
- exact current `vmm` head;
- exact tracked source blob identities;
- exact tracked Issue snapshots;
- exact tracked PR snapshots.

The packet also carries the baseline/change summary and instructions for the
connected-Notion reconciliation. Work MUST require exactly one matching title
under the CYAxiverse wiki hierarchy; zero or multiple matches require
resolution rather than a guessed edit.

## `reconcile --accept`

Acceptance requires:

```text
--accept PAGE
--packet PATH
--reconciled-commit SHA
--attest-notion-reconciled
```

The explicit attestation is the trusted S1 boundary that the connected-Notion
edit occurred. The Julia tool does not independently inspect Notion.

Before state advances, acceptance MUST:

1. validate the packet schema and recompute its digest;
2. require packet page key == `PAGE`;
3. require packet repository/branch == manifest repository/branch;
4. require packet head == `--reconciled-commit`;
5. require that head to equal current authoritative `vmm`;
6. re-fetch current source blobs and Issue/PR snapshots;
7. require the resulting current GitHub snapshot digest to equal the packet ID;
8. persist the packet's exact attested Issue/PR/source baseline, not a silently
   newer snapshot;
9. atomically replace `wiki/state.json` using an expected-old-state digest so
   concurrent writers fail closed.

If GitHub changes after the packet is reconciled but before acceptance, the
accept operation MUST refuse. A fresh packet/reconciliation is then required.

## Exit codes

- `0`: clean / successful operation
- `10`: material drift detected
- `11`: broken canonical source or remote verification failure
- `12`: validation, semantic, or owner review required

Recognized repository-evidence failures, including truncated Git trees, MUST
map to the documented contract rather than escaping as unclassified errors.

## Scheduled operation

GitHub scheduled workflows execute from the default branch. Because CYAxiverse
develops this tool on `vmm` while the repository default branch is `main`,
the scheduler is a **companion default-branch workflow** tracked under Issue
#183. It exists on `main` and explicitly checks out `vmm` before invoking
the refresher.

The `vmm` workflow in this PR is therefore PR/manual verification only; it
does not claim that a schedule declared solely on `vmm` would run.

Issue #183 MUST NOT close until:
- the companion scheduler workflow is merged to `main`;
- the core tool is available on `vmm`; and
- one post-merge default-branch manual/scheduled execution verifies the path.

## State-write safety

State replacement MUST use a temporary file in the same directory plus atomic
rename. Mutating operations MUST compare the current on-disk state digest with
the digest loaded at command start and refuse on concurrent modification.

## Non-scope

- full synchronization of every wiki page;
- autonomous scientific interpretation;
- direct Notion REST/API access from Julia;
- a knowledge-graph authority layer;
- GitHub mutation by the refresh tool;
- package release/version changes;
- root package dependency changes.

## Verification

Before convergence:
- isolated Julia 1.12 environment instantiates;
- focused unit tests cover manifest/state validation, page filtering, packet
  determinism/tampering, exact packet acceptance guards, atomic/CAS state
  writes, and exit-code classification;
- public source contains no Notion token/client/API path or private page ID;
- tracked source paths verify against current `vmm`;
- read-only `check` exercises real drift against the committed baseline;
- the companion default-branch scheduler design is independently reviewed;
- successor candidate receives a fresh exact-state independent review.
