# CYAxiverse Wiki Refresh

Deterministic freshness checking for the CYAxiverse Notion wiki.

GitHub, source code, approved specifications, validation artifacts, and durable
owner decisions remain authoritative. Notion is a derived human-readable
knowledge layer.

## Privacy boundary

The public manifest deliberately contains **no Notion page IDs**. Copy
`pages.local.example.yaml` to `pages.local.yaml` and fill the mapping
locally, or supply the same mapping as JSON through `NOTION_PAGE_MAP_JSON`.
The real mapping is private and gitignored.

## Setup

```sh
julia --project=wiki -e 'using Pkg; Pkg.instantiate()'
```

Optional environment variables:

- `GITHUB_TOKEN` — recommended for API rate limits.
- `NOTION_TOKEN` — required only for private Notion verification / accepted
  metadata writes.
- `NOTION_PAGE_MAP_JSON` — optional private page-key → page-ID mapping.

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

With no Notion credentials this verifies the public repository side only.

### Reconcile

```sh
julia --project=wiki scripts/wiki_refresh.jl reconcile
```

Semantic drift produces packets under `wiki/.wiki-refresh/packets/`; it does
not rewrite the wiki.

After the affected Notion page has been reviewed and reconciled:

```sh
julia --project=wiki scripts/wiki_refresh.jl reconcile --accept cytools_boundary
```

The accept step requires a private page mapping plus `NOTION_TOKEN`. It
updates only an existing `Verified against` marker and the local state file.

## Exit codes

| Code | Meaning |
| ---: | --- |
| 0 | clean / successful |
| 10 | material wiki drift |
| 11 | broken canonical source / remote verification failure |
| 12 | semantic or owner review required |

## Scheduled operation

The GitHub Action is deliberately read-only and receives no Notion token.
It runs the test suite and weekly drift check. A drift result is a signal to
reconcile; the script never makes GitHub changes.
