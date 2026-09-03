# General-L action replacement: bounded operations

This driver is an operational and validation boundary for an owner-approved
bounded run. It consumes immutable source fixtures. It does not run a
population calculation, replay CYTools, write the production database, or
change the `not_validated` gate.

## Post-PR-108 continuation

PR #108 merged at `fd1467721ab3e04b91af3cc232cc9cf42644947f`. Freeze that
tree, including its source commit/tree/diff and dependency and environment
revisions, before generating or executing any continuation input. The pilot
artifacts from 20260902 are immutable stale evidence from before PR #108; do
not use them as source rows, terminal-ledger rows, manifests, checkpoints, or
approval evidence.

Treat the continuation as these separate, ordered gates:

1. **Freeze and prepare paths.** Define fresh v2 source output and checkpoint
   roots, fresh bounded output and checkpoint roots, and fresh manifests.
   Every path must be absolute, canonical, absent before creation, distinct
   from every other path, and non-overlapping. Keep the manifests and approval
   file outside all source and bounded roots. All writes are create-only:
   never replace an existing root, manifest, checkpoint, or published artifact.
2. **Generate the bounded source pilot.** Use the source-generation command
   below with `--limit 1`. This pilot still reads all nine frozen partitions
   `05` through `13` for each of `h11=2,3,4,5`; `--limit 1` limits the
   generated FRST-class work, not the frozen partition set. Record the
   observed source and terminal-ledger counts rather than assuming complete
   counts.
3. **Re-fingerprint and reconcile witnesses.** Independently fingerprint the
   source and terminal-ledger files, then compare class, action, and terminal
   identity-plus-digest membership in both directions, including multiplicity.
   Check `terminal_rows` against the matrix-validation, candidate-attempt, and
   `lattice_matrix_search_summary` counters. Blank, malformed, missing,
   duplicated, or unaccounted-for rows fail closed.
4. **Prepare the bounded manifest.** Run `--prepare-bounded-manifest` only
   against the fresh source manifest. Preparation must bind the original
   source-generation roots and the two fresh bounded roots and must pass the
   self-digest, file fingerprints, pilot scope, and path checks. Preparation is
   not owner approval.
5. **Obtain and bind owner approval.** The owner approval must repeat every
   binding exactly: task and program, h11 scope and counting unit, selection
   route, action and terminal conventions, limits, seed, source commit/tree/diff,
   dependency and environment revisions, configuration digest, all four roots,
   and the prepared `input_manifest_sha256`. The approval must have status
   `approved` or `owner_approved`. Create the approval-bound manifest with
   `--bind-approval-manifest`; do not edit either input. Approval binding is a
   separate gate from preparation.
6. **Run the bounded pilot.** Execute only the approval-bound manifest, with the
   exact approval path, byte count, and SHA-256. The driver must re-fingerprint
   inputs immediately before processing, verify the checkpoint boundary on
   resume, and publish only after all witness and terminal-accounting checks
   pass. A mismatch, stale or tampered checkpoint, existing output, root
   overlap, or any attempted database write fails closed.
7. **Perform the independent pre-scale review.** After the bounded pilot, the
   main agent must independently review the frozen-tree identity, path and
   create-only evidence, exact approval bindings, witness membership and
   multiplicity, terminal accounting, and observed counts. Complete source
   generation is permitted only after this review passes. Until then, do not
   scale, infer completeness, or make a population claim.

The source-generation command for the pilot is the existing command in the
next section with `--limit 1` added; keep its nine-partition inputs and fresh
v2 roots. No continuation artifact or run is implied by this procedure.
Whether the pilot is later accepted or fails closed, preserve
`production_gate=not_validated`, `scale_status=not_applicable`, zero database
writes, and no inflation run or population/completeness claim.

## Generate Source Inputs

Use this order: generate the immutable source inputs, prepare fresh bounded
roots, obtain owner approval, bind the approval to the prepared manifest, then
run the bounded driver. The complete sequence is:

```text
source manifest -> prepared bounded manifest -> owner approval
    -> approval-bound manifest -> bounded pilot
```

Generate the source and terminal-ledger JSONL inputs before requesting owner
approval:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/opt/homebrew/Caskroom/miniforge/base/envs/cytools/bin/python -B \
  scripts/generate_general_l_action_source.py \
  --parquet-dir /frozen/calabi-yau-data/polytopes-4d \
  --output-root /fresh/general-l-source \
  --checkpoint-root /fresh/general-l-checkpoints
```

The generator requires all nine frozen partitions `05` through `13`, checks
the physical Hodge mapping, enumerates paper-equivalent FRST classes, and
preserves terminal records from the exact `(L,t,lambda_f)` search. It writes
zstd level-19 source and ledger files, a bounded-driver-compatible
`input-manifest.json.zst`, and `SHA256SUMS.txt` with create-only output
behavior. The generated manifest still requires preparation and a new owner
approval before the bounded driver can execute.

Prepare the generated source manifest with fresh, absolute, distinct output
and checkpoint roots. Both bounded roots must be absent; the source output and
checkpoint roots remain unchanged and are recorded explicitly in the prepared
manifest. The command creates one new manifest and never modifies the source
manifest:

```sh
python3 -B scripts/run_general_l_action_replacement_bounded.py \
  --prepare-bounded-manifest \
  --input-manifest /fresh/general-l-source/input-manifest.json.zst \
  --output-root /fresh/general-l-bounded-pilot \
  --checkpoint-root /fresh/general-l-bounded-checkpoints \
  --output-manifest /owner/prepared-input-manifest.json.zst
```

The preparation step checks the source manifest self-digest, all source file
fingerprints, the pilot scope, and the original roots. It rejects a bound or
stale manifest, malformed or relative paths, existing roots, and roots that
are equal to or aliased with each other or the source roots. Every per-input
bounded-run `output_root` and `checkpoint_root` binding is updated together.

The prepared manifest is intentionally unbound. The owner must review it and
create an approval that repeats its complete binding, including
`source_generation_output_root`, `source_generation_checkpoint_root`,
`output_root`, `checkpoint_root`, and `input_manifest_sha256`. After that owner
decision, create the approval-bound copy with:

```sh
python3 -B scripts/run_general_l_action_replacement_bounded.py \
  --bind-approval-manifest \
  --approval /owner/approval.json \
  --input-manifest /owner/prepared-input-manifest.json.zst \
  --output-manifest /owner/approval-bound-input-manifest.json.zst
```

The binder verifies that the approval file decodes to the supplied approval,
writes one new manifest, and never modifies either input. The bound manifest
excludes only `approval_fingerprint` from its digest. The execution CLI still
requires an exact approval path, byte count, and SHA-256 match, so this binding
is not circular. Keep both the approval file and approval-bound manifest
outside the source-generation and bounded-run roots; the binder rejects any
overlap to keep those roots immutable and create-only.

## Fixture input schema

The input manifest is JSON (or zstd JSON) with schema
`cyaxiverse-general-l-action-replacement-input-1.0`. It contains the exact
approval bindings (`task_id`, `program`, `h11_values`, `counting_unit`,
`selection_route`, exact action/terminal conventions, `limits`, `global_limits`,
`seed`, source commit/tree/diff, dependency and environment revisions,
configuration digest, the original source-generation output and checkpoint
roots, the fresh bounded output root, and separate bounded checkpoint root)
plus this list:

```json
{"inputs":[
  {"h11":2,"role":"source_rows","path":"/absolute/source.jsonl",
   "size_bytes":123,"sha256":"...","file_type":"jsonl",
   "source_row_or_partition_identity":"h11=2,row=1",
   "selection_route":"synthetic_fixture","counting_unit":"favorable CY FRST class",
   "output_root":"/fresh/bounded-output","checkpoint_root":"/fresh/bounded-checkpoints"},
  {"h11":2,"role":"terminal_ledger","path":"/absolute/ledger.jsonl",
   "size_bytes":456,"sha256":"...","file_type":"jsonl",
   "source_row_or_partition_identity":"h11=2,row=1",
   "selection_route":"synthetic_fixture","counting_unit":"favorable CY FRST class",
   "output_root":"/fresh/bounded-output","checkpoint_root":"/fresh/bounded-checkpoints"}
]}
```

There must be one `source_rows` and one `terminal_ledger` partition for each
of h11=2,3,4,5. Each JSONL row must include `h11`, integer `source_row`,
`polytope_id`, `frst_hash`, `frst_class_index`, `record_kind`, terminal status
and reason, `schema_version` equal to
`cyaxiverse-inherited-orientifold-candidate-3.0`, and complete exact action
fields for candidate rows. Matrix and search-summary rows use JSON `null` for
`candidate_id`, `action_digest`, `lattice_matrix`, `torus_shift`, and
`lambda_f`; their structural action evidence remains in the
`source_trilayer_candidate` fallback. The manifest also includes an
`approval_fingerprint` object for the approval file after binding; the
create-only binder adds this object after owner approval and the CLI checks it
before processing any source row. The prepared manifest records the original
source-generation roots as `source_generation_output_root` and
`source_generation_checkpoint_root`.
The source and ledger files use the same row schema; they are compared as
independent witnesses. Blank or malformed input is counted and blocks
publication.

The manifest must fingerprint the bounded driver and the source formula chain,
including the existing candidate, geometry, ledger, digest, formula-ledger,
and `reproduce_fuzzy_axions_h11_4.py` files used by the program, plus this
driver. It must also bind the global limits: RSS at most 2 GiB, one worker, one
thread, and zero database writes.

Without `--limit`, the generator is a complete run and requires these exact
source counts: h11=2 has 36 favorable polytopes and 36 FRST classes; h11=3
has 243 and 274; h11=4 has 1185 and 1760; h11=5 has 4897 and 11713. A failed
FRST classifier cannot be represented by a pseudo class to satisfy a complete
count. Supplying `--limit N` labels the run as a `pilot`; expected complete
counts are recorded as not required and are not enforced.

The generator writes one immutable source/ledger segment and metadata file per
FRST class under `checkpoint_root`. A resumed run verifies both segment files,
their recorded size and SHA-256, class identity, and schema before skipping
the class. A partial or tampered pair fails closed. Source and ledger streams
are concatenated and compressed incrementally; RSS and the declared temporary
and per-h11 output ceilings are checked before loading each h11, before class
spooling, and before publication.

## Required gates

Before execution, prepare the generated source manifest as described above.
Then an owner must provide a new approval record for that prepared manifest.
The manifest must bind each absolute source path, byte count, SHA-256, source
row or partition identity, code revision, environment revision, configuration,
seed, limits, the original source-generation roots, and two fresh bounded roots.
The driver re-fingerprints the files and refuses a mismatch. Existing final
artifacts are never replaced.

The approval JSON must have `status` equal to `approved` or `owner_approved`
and must repeat every manifest binding exactly. The manifest
`approval_fingerprint` object binds the approval file path, byte count, and
SHA-256. The CLI checks it immediately before execution.

Use the bound manifest produced by the binder:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -B scripts/run_general_l_action_replacement_bounded.py \
  --approval /owner/approval.json \
  --input-manifest /owner/approval-bound-input-manifest.json.zst \
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
lists only the published artifacts and is itself not listed.

The terminal-ledger writer emits schema
`cyaxiverse-orientifold-terminal-ledger-1.2`. The reader also accepts legacy
1.1 sidecars for migration and audit; new writers and bounded witnesses require
1.2. JSON and JSONL compression uses zstd level 19.

## Focused checks

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m py_compile \
  scripts/run_general_l_action_replacement_bounded.py \
  scripts/test_run_general_l_action_replacement_bounded.py
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/test_run_general_l_action_replacement_bounded.py
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/test_generate_general_l_action_source.py
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/test_orientifold_terminal_ledger.py
```
