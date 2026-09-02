# General-L action replacement: bounded operations

This driver is an operational and validation boundary for an owner-approved
bounded run. It consumes immutable source fixtures. It does not run a
population calculation, replay CYTools, write the production database, or
change the `not_validated` gate.

## Fixture input schema

The input manifest is JSON (or zstd JSON) with schema
`cyaxiverse-general-l-action-replacement-input-1.0`. It contains the exact
approval bindings (`task_id`, `program`, `h11_values`, `counting_unit`,
`selection_route`, exact action/terminal conventions, `limits`, `global_limits`,
`seed`, source commit/tree/diff, dependency and environment revisions,
configuration digest, output root, and separate
checkpoint root) plus this list:

```json
{"inputs":[
  {"h11":2,"role":"source_rows","path":"/absolute/source.jsonl",
   "size_bytes":123,"sha256":"...","file_type":"jsonl",
   "source_row_or_partition_identity":"h11=2,row=1",
   "selection_route":"synthetic_fixture","counting_unit":"favorable CY FRST class"},
  {"h11":2,"role":"terminal_ledger","path":"/absolute/ledger.jsonl",
   "size_bytes":456,"sha256":"...","file_type":"jsonl",
   "source_row_or_partition_identity":"h11=2,row=1",
   "selection_route":"synthetic_fixture","counting_unit":"favorable CY FRST class"}
]}
```

There must be one `source_rows` and one `terminal_ledger` partition for each
of h11=2,3,4,5. Each JSONL row must include `h11`, integer `source_row`,
`polytope_id`, `frst_hash`, `frst_class_index`, `record_kind`, terminal status
and reason, `schema_version` equal to
`cyaxiverse-inherited-orientifold-candidate-3.0`, and complete exact action
fields for candidate rows. Matrix and search-summary rows use JSON `null` for
`candidate_id` and `action_digest`, and retain a structural
`source_trilayer_candidate` fallback. The manifest also includes an
`approval_fingerprint` object for the approval file; the CLI checks it before
processing any source row.
The source and ledger files use the same row schema; they are compared as
independent witnesses. Blank or malformed input is counted and blocks
publication.

The manifest must fingerprint the bounded driver and the source formula chain,
including the existing candidate, geometry, ledger, digest, and formula-ledger
files used by the program, plus this driver. It must also bind the global
limits: RSS at most 2 GiB, one worker, one thread, and zero database writes.

## Required gates

Before execution, an owner must provide a new approval record and an immutable
input manifest. The manifest must bind each absolute source path, byte count,
SHA-256, source row or partition identity, code revision, environment revision,
configuration, seed, limits, and a fresh output root. The driver re-fingerprints
the files and refuses a mismatch. Existing final artifacts are never replaced.

The approval JSON must have `status` equal to `approved` or `owner_approved`
and must repeat every manifest binding exactly. The manifest
`approval_fingerprint` object binds the approval file path, byte count, and
SHA-256. The CLI checks it immediately before execution.

Use the bounded command only after those inputs exist:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B scripts/run_general_l_action_replacement_bounded.py \
  --approval /owner/approval.json \
  --input-manifest /owner/input-manifest.json \
  --output-root /owner/fresh-output-root \
  [--resume /owner/checkpoints/checkpoint.json]
```

The command processes one h11 value at a time in deterministic
`source_row::polytope_id::frst_hash::frst_class_index::action_digest` order.
It writes to a same-filesystem staging directory and renames that directory
only after every h11 comparison passes or produces a complete fail-closed
summary. Existing output roots are refused. A checkpoint must bind the input,
code, environment, configuration, seed, limits, output root, and a complete
four-field last-class boundary; a stale, truncated, or tampered checkpoint is
`resume_mismatch`. The `next_class` helper returns only rows after a verified
boundary for an upstream action enumerator. This immutable-source adapter has
no hidden geometry state to skip, so a resumed publication re-reads all
immutable rows to keep the final witness manifests complete and lossless.

## Evidence contracts

Actions retain an integer 4x4 `lattice_matrix`, reduced common-denominator exact
`torus_shift`, and `lambda_f` in `{0,1}`. Their digest is recomputed from those
three fields. `matrix_digest` follows the existing
`build_orientifold_axion_database.py::lattice_matrix_digest` object encoding
and is never used as an action digest. Every terminal record retains its complete evidence and receives
an identity digest plus a digest of the complete record. Matrix validation,
candidate attempts, and `lattice_matrix_search_summary` rows are all counted;
`terminal_rows` must equal the sum of those three counters.

An accepted record is counted as selected only when it carries exact validated
Hodge evidence with `lambda_f=1`, `h11_minus=0`, and `h21_plus=0`. For each
class, the reported representative is the lexicographically smallest such
accepted `action_digest`; all other accepted and rejected actions remain in
the ledger.

Live and ledger witnesses must match in both directions for class keys,
action keys, and terminal identity-plus-digest keys. Equal aggregate counts are
not membership evidence. The only permitted result boundary is a finite,
provenance-bound or fail-closed bounded result with `production_gate` set to
`not_validated` and `scale_status` set to `not_applicable`.

Published files are `run-manifest.json.zst`, five files per h11 using the
`h11-002` through `h11-005` names, and `SHA256SUMS.txt`. The checksum file
lists only the published artifacts and is itself not listed. JSON and JSONL
compression uses zstd level 19.

## Focused checks

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m py_compile \
  scripts/run_general_l_action_replacement_bounded.py \
  scripts/test_run_general_l_action_replacement_bounded.py
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/test_run_general_l_action_replacement_bounded.py
```
