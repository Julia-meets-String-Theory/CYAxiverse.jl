# CYAxiverse Wiki Refresh

Deterministic freshness checking for the CYAxiverse Notion wiki.

GitHub, source code, approved specifications, validation artifacts, and durable
owner decisions remain authoritative. Notion is a derived human-readable
knowledge layer.

## v1 architecture

The Julia tool does **not** call the Notion API and does not use a Notion token.

```text
GitHub / repository
       |
       v
wiki_refresh.jl
  detect drift
  emit exact snapshot-bound packet
       |
       v
Work / ChatGPT + connected Notion integration
  require exactly one matching wiki page
  reconcile that exact packet
  verify the edit
       |
       v
wiki_refresh.jl reconcile --accept ...
  validate packet/live GitHub equality
  atomically advance local state
```

## Setup

```sh
julia --project=wiki -e 'using Pkg; Pkg.instantiate()'
```

`GITHUB_TOKEN` is optional but recommended for API rate limits. The tool
requires no Notion credential.

## Commands

### Check — read only

```sh
julia --project=wiki scripts/wiki_refresh.jl check
julia --project=wiki scripts/wiki_refresh.jl check --page cytools_boundary
```

An unknown page key fails closed.

### Verify — read only

```sh
julia --project=wiki scripts/wiki_refresh.jl verify
```

This validates manifest/state integrity and tracked public repository sources.

### Reconcile

```sh
julia --project=wiki scripts/wiki_refresh.jl reconcile
```

Packets are emitted under:

```text
wiki/.wiki-refresh/packets/
```

Each deterministic packet has a `packet_id` bound to the exact repository,
page, `vmm` head, source blobs, Issue snapshots, and PR snapshots presented
to Work.

Work/ChatGPT uses the connected Notion integration to reconcile exactly that
packet. It must find exactly one matching page title under the CYAxiverse wiki;
zero or multiple matches require resolution.

After the Notion edit has been verified:

```sh
julia --project=wiki scripts/wiki_refresh.jl reconcile \
  --accept cytools_boundary \
  --packet wiki/.wiki-refresh/packets/cytools_boundary-<id>.json \
  --reconciled-commit <EXACT_PACKET_VMM_SHA> \
  --attest-notion-reconciled
```

Acceptance re-fetches GitHub and refuses if any tracked source/Issue/PR state
differs from the packet. It then atomically advances `wiki/state.json` to the
packet snapshot. The Julia process still makes no Notion or GitHub write.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | clean / successful |
| 10 | material wiki drift |
| 11 | broken canonical source / remote verification failure |
| 12 | validation, semantic, or owner review required |

## Scheduling

GitHub scheduled workflows run from the repository default branch, which is
`main`. The core `vmm` workflow is therefore PR/manual verification only.

Issue #183 uses a companion workflow on `main` that explicitly checks out
`vmm` before running this tool. The Issue is not complete until that scheduler
is merged and verified after the core tool is available on `vmm`.
