# Bounded h11=4 unaffected-row projection

This ledger defines the read-only comparison gate for the zero-dimensional
fixed-component certificate repair. It does not define a new population
selection or start a scan.

## Immutable source

The baseline artifact is
`../data/orientifold_candidate_replay_h11_4_20260824/replay.jsonl.zst`.
Its SHA-256 is
`b7cb293ea369d52fa22aa01db6b487303b40279f87e40d14a1509bb1a062dfa3`.
The artifact contains 1,146 `record_type="row"` records. Its replay header
records `requested_h11=4`, `max_rows=1146`, `workers=1`, `shard_count=1`, and
`shard_index=0`; its recorded aggregate replay code fingerprint is
`492092e49b20b354659835256d369f5a1da869fd62cc238d06b3c10d598dd4ca`.

The baseline status counts are 1,060 `h21_plus_nonzero`, 28
`accepted_exact_trilayer_action`, and 58
`smoothness_verification_unavailable`.

## Projection contract

The source affected set is exactly the 58 baseline rows whose
`terminal_status` is `smoothness_verification_unavailable`. Include the other
1,088 source row identities. Require the repaired artifact to have the same
complete row-identity set. Project only these fields:

```json
{"row_identity":"...","terminal_status":"..."}
```

Sort by `row_identity` in ascending Python Unicode code-point order. Encode
each object with Python
`json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False)`, encode the result as UTF-8, append one LF byte after every
object including the final object, and hash the concatenated bytes with
SHA-256. The comparison passes only when the source and repaired projection
digests are equal.

The implementation is
[`scripts/orientifold_replay_projection.py`](../scripts/orientifold_replay_projection.py)
and its focused tests are
[`scripts/test_orientifold_replay_projection.py`](../scripts/test_orientifold_replay_projection.py).

## Reproduction command

Generate the repaired bounded replay under `/private/tmp`, then run:

```sh
PYTHONPATH=scripts python3 scripts/orientifold_replay_projection.py \
  --source ../data/orientifold_candidate_replay_h11_4_20260824/replay.jsonl.zst \
  --repaired /private/tmp/cyax-h4-certificate-repair-replay.jsonl.zst \
  --implementation scripts/orientifold_general_l_geometry.py \
  --output /private/tmp/cyax-h4-unaffected-projection-20260824.json.zst
```

The generated report must retain the compressed source and repaired artifact
SHA-256 values, both replay header fingerprints, the repaired implementation
fingerprint, row counts, the projection recipe, and both observed projection
digests. Generated reports and replay outputs remain outside this checkout.

## Observed bounded verification

The repaired replay completed with 1,146 rows, `database_writes=0`, and
`duplicate_count=0`. Its status counts were 1,117 `h21_plus_nonzero`, 28
`accepted_exact_trilayer_action`, and 1
`smoothness_verification_unavailable`. The repaired replay artifact SHA-256
was
`4227a9244e43e719df437126db09520ec7c29fa78937b9906532190791e9e076`.

The compressed comparison report was
`/private/tmp/cyax-h4-unaffected-projection-20260824-final-v2.json.zst`, with
SHA-256
`5450cde311eea24bfbc762c7345ded5dd553bd6d25b5f337de9a6fedd881c838`.
The source and repaired 1,088-row projection digests were both
`ebe14b02d312993fd85ce98b2aa882701b1c14f6aff25065e5d566a2f2ad504b`.
The report records the repaired implementation fingerprint
`ddd3f71b430f32dbbcb7d96dc9dbb64204efe68f095d1e7d8d2fd4cbde9650ad` and
`status_projection_matches=true`.
