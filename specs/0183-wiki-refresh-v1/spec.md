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

~~~text
Julia refresher
  -> reads GitHub/repository state exactly once per refresh snapshot
  -> derives drift summary from that same frozen snapshot
  -> emits one deterministic reconciliation packet
  -> STOP

Work / ChatGPT with connected Notion integration
  -> locates exactly one matching CYAxiverse wiki page
  -> reconciles the exact packet
  -> verifies the edit
  -> explicitly attests that exact packet_id

Julia refresher
  -> validates the full packet digest
  -> replays the packet's GitHub snapshot against live GitHub
  -> requires exact equality
  -> atomically advances local wiki/state.json under an interprocess lock
~~~

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
8. Reconciliation acceptance MUST be bound to one exact full packet, not merely
   to a commit or a subset of packet fields.
9. Missing or ambiguous authoritative evidence is a reason to abstain, not infer.

## Manifest and state

The public manifest maps stable wiki page keys to:
- title and authority class;
- tracked repository paths;
- relevant Issues and PRs;
- whether branch-head movement itself is material.

The state file records, per page:
- exact last reconciled vmm commit;
- last reconciled Issue snapshots;
- last reconciled PR snapshots;
- accepted packet identity after packet-based reconciliation.

wiki/state.json is operational reconciliation state, not project authority.

## check

Read-only. Compare current vmm state with the page baseline using:
- exact Git blob identity;
- tracked Issue lifecycle state;
- tracked PR lifecycle/head state;
- branch-head movement only where configured.

Unknown --page keys MUST fail closed.

## verify

Read-only. Validate:
- manifest structure/privacy constraints;
- state structure;
- exact manifest/state page-key coverage;
- state repository/branch identity;
- tracked Issue/PR snapshot coverage and required fields;
- tracked current source-path existence.

It does not contact Notion.

## Reconciliation packet

For each affected page, the detector MUST build **one current snapshot once**.
The drift summary and actionable packet MUST both derive from that same snapshot;
packet generation MUST NOT refetch Issue or PR state.

The packet's SHA-256 packet_id MUST bind the complete canonical reconciliation
payload:

- repository owner/name/branch;
- page key/title/authority;
- exact current vmm head;
- exact tracked source blob identities;
- exact tracked Issue snapshots;
- exact tracked PR snapshots;
- baseline verified commit;
- derived source/Issue/PR change summary;
- execution-surface declaration;
- reconciliation instructions.

Changing any actionable field MUST invalidate packet_id.

Work MUST require exactly one exact-title page match within the CYAxiverse wiki
hierarchy. Zero or multiple matches require resolution.

## reconcile --accept

Acceptance requires:

~~~text
--accept PAGE
--packet PATH
--reconciled-commit SHA
--attest-notion-reconciled
~~~

The explicit attestation is the S1 trust boundary that the connected-Notion
edit occurred.

Before advancing state, the tool MUST:

1. validate the full packet schema and digest;
2. require packet page key == requested page;
3. require packet repository/branch == manifest repository/branch;
4. require packet baseline == current per-page state baseline;
5. require packet head == reconciled commit;
6. require reconciled commit == current authoritative vmm;
7. rebuild the live source/Issue/PR snapshot;
8. require live snapshot == packet snapshot exactly;
9. persist the packet snapshot, not a silently newer GitHub snapshot;
10. replace wiki/state.json under an interprocess lock while checking the
    expected pre-mutation state digest.

If GitHub state moves after packet reconciliation, acceptance MUST refuse and a
fresh packet/reconciliation is required.

## State-write safety

Every supported state mutation MUST use the same state-specific interprocess
lock. While holding the lock it MUST:
- verify the expected old digest or expected absence;
- construct the full temporary replacement in the same directory;
- recheck the expected state before replacement;
- atomically rename the completed file.

A competing supported writer MUST fail closed rather than overwrite another
successful reconciliation. A stale lock after abnormal process death is
fail-closed and requires operator inspection/removal.

## Exit contract

- 0: clean / successful operation
- 10: material drift detected
- 11: broken canonical source or remote verification failure
- 12: validation, semantic, or owner review required

Validly parsed but structurally invalid manifest/state/packet inputs MUST map to
validation failure rather than uncaught Julia dispatch/index errors.

## Scheduler

GitHub scheduled workflows execute from the default branch. Therefore the
durable scheduler lives in companion PR #187 targeting main and explicitly
checks out vmm.

Deployment order is:

~~~text
merge #184 -> vmm
merge #187 -> main
run #187 from main
verify/check current vmm
~~~

Once installed, the scheduler MUST fail if the expected wiki-refresh tooling is
absent from vmm; disappearance of the detector must not produce a green
inactive run.

Issue #183 MUST remain open until one real post-merge default-branch manual or
scheduled execution verifies the path.

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
- tests cover manifest/state validation and wrong-root inputs;
- full-payload packet tampering is rejected;
- packet emission remains coherent if GitHub changes after snapshot collection;
- Issue and PR movement after Notion reconciliation reject acceptance;
- vmm movement rejects acceptance;
- wrong page/repository packets reject;
- successful acceptance persists the packet snapshot;
- concurrent supported state writers are serialized/fail closed;
- tracked source verification and real drift detection pass;
- companion scheduler validation passes;
- successor exact-state independent review passes.
