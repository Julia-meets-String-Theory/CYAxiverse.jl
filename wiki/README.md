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
  emit packet
       |
       v
Work / ChatGPT
connected Notion integration
  read authoritative sources
  update affected wiki page
       |
       v
wiki_refresh.jl reconcile --accept ...
  advance local state only
```

This keeps credentials and private page locators out of the public repository
tooling while preserving deterministic drift detection.

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

### Verify — read only

```sh
julia --project=wiki scripts/wiki_refresh.jl verify
```

This validates tracked public repository sources.

### Reconcile

```sh
julia --project=wiki scripts/wiki_refresh.jl reconcile
```

Semantic drift produces packets under:

```text
wiki/.wiki-refresh/packets/
```

The packet is given to Work/ChatGPT, which updates the corresponding Notion
page through the connected Notion integration.

After that update has been checked, advance the local baseline explicitly:

```sh
julia --project=wiki scripts/wiki_refresh.jl reconcile \
  --accept cytools_boundary \
  --reconciled-commit <EXACT_CURRENT_VMM_SHA>
```

The accept step does not contact Notion. It refuses to advance state unless the
provided commit is exactly the current `vmm` head and all tracked source paths
still exist.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | clean / successful |
| 10 | material wiki drift |
| 11 | broken canonical source / remote verification failure |
| 12 | semantic or owner review required |

## Scheduled operation

The GitHub Action is deliberately read-only. It runs the test suite and weekly
drift check. A drift result is a signal to reconcile; the script never makes
GitHub or Notion changes.
