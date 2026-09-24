# CYAxiverse Wiki Refresh

Deterministic freshness checking for the CYAxiverse Notion wiki.

GitHub, source code, approved specifications, validation artifacts, and durable
owner decisions remain authoritative. Notion is a derived human-readable layer.

## v1 architecture

~~~text
GitHub / repository
       |
       v
wiki_refresh.jl
  freeze current source/Issue/PR snapshot once
  derive drift from that snapshot
  hash the full actionable reconciliation packet
       |
       v
Work / ChatGPT + connected Notion integration
  locate exactly one CYAxiverse wiki page
  reconcile that exact packet
  verify the edit
       |
       v
wiki_refresh.jl reconcile --accept ...
  validate full packet identity
  replay live GitHub snapshot
  lock state
  atomically advance local baseline
~~~

The Julia tool has no Notion API client, Notion token, private Notion page IDs,
or GitHub write path.

## Setup

~~~sh
julia --project=wiki -e 'using Pkg; Pkg.instantiate()'
~~~

GITHUB_TOKEN is optional but recommended for rate limits.

## Check

~~~sh
julia --project=wiki scripts/wiki_refresh.jl check
julia --project=wiki scripts/wiki_refresh.jl check --page cytools_boundary
~~~

Unknown page keys fail closed.

## Verify

~~~sh
julia --project=wiki scripts/wiki_refresh.jl verify
~~~

This validates manifest/state integrity and tracked current source paths.

## Generate reconciliation packets

~~~sh
julia --project=wiki scripts/wiki_refresh.jl reconcile
~~~

Packets are written under wiki/.wiki-refresh/packets/.

Each packet_id is a SHA-256 identity of the complete canonical reconciliation
payload, including the exact source/Issue/PR snapshot and the derived actionable
change summary.

Packet creation does not refetch Issue/PR state after drift detection; the
summary and snapshot therefore describe one coherent GitHub observation.

## Accept a completed Notion reconciliation

After Work/ChatGPT has reconciled and verified the exact packet:

~~~sh
julia --project=wiki scripts/wiki_refresh.jl reconcile \
  --accept cytools_boundary \
  --packet wiki/.wiki-refresh/packets/cytools_boundary-<id>.json \
  --reconciled-commit <EXACT_PACKET_VMM_SHA> \
  --attest-notion-reconciled
~~~

Acceptance rejects:
- a tampered packet;
- the wrong page/repository/branch;
- a stale per-page baseline;
- changed source blobs;
- changed Issue or PR state;
- moved vmm.

State replacement is serialized by an interprocess lock and atomically renamed
only after the expected previous state digest is confirmed.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | clean / successful |
| 10 | material wiki drift |
| 11 | broken canonical source / remote verification failure |
| 12 | validation, semantic, or owner review required |

## Scheduling

The durable weekly scheduler is companion PR #187 on default branch main. It
explicitly checks out vmm.

Required deployment order:

~~~text
#184 -> vmm
#187 -> main
one real workflow_dispatch/scheduled run from main
~~~

After activation, absence of the refresh tool on vmm is an error, not a green
no-op.
