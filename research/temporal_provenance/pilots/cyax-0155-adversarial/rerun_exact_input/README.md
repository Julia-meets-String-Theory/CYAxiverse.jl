# CYAX-0163 exact-input rerun harness

This directory is a separate clean-rerun execution area.  The original
contaminated artifacts under `../runs/` and `../scoring/` are not modified.

`dispatch_manifest.json` freezes the common-prompt identity, context
identities, delimiter, full A/B input hashes, and preregistered launch order.
The input bytes are constructed only by `exact_dispatch.py`:

```text
common_prompt_bytes + delimiter_bytes + condition_context_bytes
```

`materialize` also stores six per-run request artifacts under
`runs/requests/`; each A replicate and each B replicate has its own recorded
input hash in `runs/manifest.json`.

Use the following checks before any launch:

```sh
python3 exact_dispatch.py materialize
python3 verify_rerun.py --phase pre-dispatch
```

The single-run launcher uses the local `codex exec` binary directly.  It
passes the already-hashed per-run request artifact as binary stdin with no
shell, uses a fresh empty working directory, pins GPT-5.6 Sol with high
reasoning, and applies `--ephemeral`, `--ignore-user-config`,
`--ignore-rules`, and read-only sandboxing.  It refuses repeated or
out-of-order launches:

```sh
python3 launch_subject.py A1
```

The launcher stores sanitized JSONL events, a byte count/hash for stderr (raw
stderr is not persisted because it can contain host-local paths), and exact
final-response bytes.
It verifies the events contain no tool, shell, browser, command, or source
reopening activity.  A successful run advances the manifest to `CAPTURED`; a
failed or contaminated run is durably marked `FAILED` and cannot be retried.

For inspection or an execution surface that explicitly accepts a binary
stream, emit one input as follows.  The command writes no status text to
stdout:

```sh
python3 exact_dispatch.py emit A1 > request.bin
sha256sum request.bin
```

The emitted hash must equal the A or B full-input hash in
`dispatch_manifest.json`.  Do not paste `request.bin` into a text editor or
chat field that can normalize bytes.  If the execution surface cannot accept
the stream without transformation, stop as `BLOCKED`.

The launcher performs immutable response capture itself.  The lower-level
capture command remains available when an execution surface returns a raw
response separately:

```sh
python3 capture_response.py A1 < response.bytes
```

The capture command refuses an empty response or an existing response and
records response byte count, SHA-256, completion status, and source-reopening
count.  After all six captures, run:

```sh
python3 verify_rerun.py --phase captured
```

Before publishing captured event evidence, run the one-time
`sanitize_captured_artifacts.py` migration.  It refuses to run twice and
rewrites only event JSONL and run records; it does not read or modify inputs,
responses, or scorecards.

Post-capture blind materialization copies each captured response byte-for-byte
to only `scoring/blind_responses/S1.response` through `S6.response`.  The
private `blind_materialization_manifest.json` records source and opaque hashes;
the separate `blind_content_leakage.json` report flags (but never removes)
`graph`, `relational`, and condition-label terms.  These files are audit
artifacts and the mapping-bearing manifest must not be shown to the scorer:

```sh
python3 materialize_blind.py --verify
```

The frozen scorer input is built from `scorer_prompt.md`, the existing
`answer_key.md`, the exact scorecard schema, and the six opaque response bytes.
It contains no condition mapping, run metadata, or dispatch hashes:

```sh
python3 build_scorer_input.py --verify
```

`launch_scorer.py` is a single-use direct `codex exec` launcher pinned to
GPT-5.6 Sol/high reasoning.  It passes `scorer_input.input` as binary stdin to
one fresh empty read-only ephemeral process, captures sanitized JSONL events and
exact final-response bytes, rejects tool/source-reopening events, and splits a
valid six-line JSONL response into immutable `S1`–`S6` scorecards.  It is not
run as part of this implementation; invoke it only after the blind input is
approved.  `scoring/condition_mapping.json` remains withheld until all blind
scorecards are frozen.

Local Codex thread/session identifiers are not published.  Event JSONL and run
records retain identity fields with deterministic `sha256:<64 hex>`
pseudonyms.  Each run record stores the original event byte count/hash as
non-reversible provenance; original event bytes are not retained.  The
sanitizer verifies that all six subject pseudonyms are present and unique.
Response bytes, input artifacts, and scorecards are not changed by this
privacy sanitization.
