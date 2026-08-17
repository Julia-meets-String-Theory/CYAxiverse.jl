# Stage 2 EFT finalization — user guide

This guide covers running [`generate_stage2_eft_reference.py`](./generate_stage2_eft_reference.py)
with `--eft` to build the compact EFT-reference table (`eft_models.parquet`)
from a completed Stage 1 raw-FRST population. For the full flag reference and
scientific contract, see
[`README_generate_geometric_data_multitriangulation.md`](./README_generate_geometric_data_multitriangulation.md).
This guide is narrower: it explains what happens when you run the command,
what to expect while it runs, and gives a ready-to-use command for the
approved 1,400-geometry population.

## What the script does

Stage 2 takes a frozen Stage 1 raw-FRST population (produced by
`generate_stage1_raw_frsts.py`) and, for each retained candidate:

1. Reconstructs the CYTools geometry, validates its topology (using the
   Stage 1 topology cache when available), and applies the canonical QCD
   normalization and QED-aware prefilter.
2. On acceptance, builds the complete, validated, hashed QCD/QED
   assignment pool for that geometry and writes the geometry HDF5 artifact.
3. With `--eft`, **finalization** then samples EFT-reference rows from
   every accepted geometry's assignment pool and writes them to
   `eft_models.parquet`.

Steps 1-2 are still processed one candidate at a time. Step 3 is where this
session's work applies.

## How EFT finalization works, and why it used to be slow

Finalization validates every persisted assignment-pool entry before
sampling — for a geometry with a pool of 1,000 entries, that means scoring
1,000 individual `(QCD, QED)` divisor pairs. Two of the computations involved
in scoring one entry (reconstructing the bounded potential `Q`/`L`, and
classifying which potential terms are "leading" via exact-rational rank
elimination) depend only on the geometry, never on which pair is being
scored — but the original implementation recomputed them from scratch for
every entry. At `h11=491`, each of those recomputations cost on the order of
a minute; multiplied across a large pool, a single geometry's finalization
could run for hours without an unrelated algorithmic inefficiency in the
rank classifier (which rebuilt its whole exact-rational basis from scratch
on every candidate column instead of maintaining it incrementally).

Three fixes, all in this session's changes, address this:

- The geometry-only potential terms and their certificate hashes are now
  computed once per geometry and reused for the rest of that geometry's
  pool.
- The rank classifier now maintains its exact-rational basis incrementally
  instead of rebuilding it per candidate column, and stops as soon as the
  basis reaches full rank (a column can never increase rank past that
  point).
- Because each geometry's pool validation is independent of every other
  geometry's, finalization now runs one worker process per geometry via
  `--eft-workers` (default: all available cores).

A fourth change reduces output size: the per-row certificate used to persist
`ordered_source_indices`, the full priority order over every potential term
for that geometry (up to ~120,000 integers at `h11=491`) — identical for
every row of the same geometry, and reconstructible from the geometry's
`Q`/`L` when needed. It is no longer written per row. Measured effect: the
same 2,749-row output went from 509 MB to 449 KB.

None of this changes what is computed — the same rows, same acceptance
decisions, same terminal statuses. Verified by rerunning a fixed 20-geometry
input population before and after: identical `accepted_geometries`,
`eft_rows`, and dataset status in both cases.

## Prerequisites

- The `cytools` conda environment (`conda run --no-capture-output -n cytools
  python ...` or an activated shell).
- A completed Stage 1 population under `--stage1-root`. The approved
  1,400-geometry plan (`50:500,100:500,200:300,491:100`) lives at
  `/private/tmp/cyaxiverse-glimmers-geometry-staged-20260814-luna4` as of
  this writing; confirm it is still present before using the command below,
  since `/private/tmp` is not guaranteed to survive a reboot.
- An orientifold file (`--orientifold-file`), e.g.
  `o3_o7_involution.json` at the repository root.
- A fresh `--outdir`: the script refuses to write into a directory that
  already contains files.

## Running the full population

```bash
cd /Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl
conda run --no-capture-output -n cytools python scripts/generate_stage2_eft_reference.py \
  --stage1-root /private/tmp/cyaxiverse-glimmers-geometry-staged-20260814-luna4 \
  --outdir /private/tmp/cyax-stage2-full-population-20260817 \
  --eft \
  --orientifold-file /Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl/o3_o7_involution.json \
  --volume-backend auto \
  --seed 20260816 \
  --max-kaehler-attempts 1 \
  --verbose
```

Change the date suffix on `--outdir` if you run this more than once, since a
non-empty output directory is rejected outright.

To cap or disable parallelism, add `--eft-workers N` (or `--eft-workers 1`
for strictly sequential execution — useful when diagnosing a failure, since
sequential output is easier to read top-to-bottom).

## What to expect

**Runtime.** The per-candidate geometry stage (step 1-2 above) is still
sequential; at roughly one second per candidate across the mix of `h11`
values, 1,400 candidates take on the order of 30 minutes. Finalization is
now parallelized across geometries, and `h11=491` dominates its cost: with
around 95-100 accepted `h11=491` geometries spread across the available
cores, finalization is estimated at 45-90 minutes. Total: roughly
**1.5-2 hours**, against several hours for the pre-fix sequential path. This
is an estimate extrapolated from a 20-geometry test, not a measurement of
the full run.

**Storage.** With the certificate fix, `eft_models.parquet` should land
around 16-33 MB even at the full 100,000-200,000 row target — no longer a
practical concern.

**Progress while it runs.** The output directory fills in incrementally:

- `stage2_progress.jsonl` — one record per candidate lifecycle event.
- `stage2_terminal_statuses.partial.jsonl`,
  `stage2_topology_diagnostics.partial.jsonl`,
  `stage2_kaehler_point_diagnostics.partial.jsonl`,
  `stage2_assignment_pool_rejections.partial.jsonl` — flushed incrementally
  during the geometry stage; the `.partial` suffix is dropped once the run
  finishes normally.
- During finalization, there is no per-row progress output — the worker
  processes are computing, not printing, until each geometry's pool is
  fully scored.

**Completion.** A successful run writes `run_manifest.json` and
`eft_models.parquet` (plus `model_terminal_statuses.jsonl`, `charge_factorized_manifest.json`, `summary_by_h11_and_status.json`, `storage_estimate.json`). Check
`run_manifest.json`'s `"eft"` block for `"dataset_status"`:

- `production_complete` — the exact 200,000-row target was reached.
- `diagnostic_partial` — finalization completed normally but validated
  capacity or row generation fell short of the 100,000-row minimum. This is
  a legitimate, fully-computed result, not a failure; check
  `"validated_assignment_capacity"` and `"minimum_shortfall"` alongside it.

## Operational notes

- **No resume.** If the run is interrupted (Ctrl+C, crash, machine sleep),
  the partial files are retained for diagnostics, but the run itself cannot
  be resumed — a rerun starts from scratch into a fresh `--outdir`.
- **Keep the machine awake.** For a run this long, prefix the command with
  `caffeinate -i` to prevent macOS from sleeping mid-run.
- **To stop cleanly**, Ctrl+C in the terminal running the command.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `FileExistsError: output root must be fresh and non-overwriting` | `--outdir` already contains files | Pick a new, empty `--outdir` |
| `RuntimeError: An attempt has been made to start a new process...` (only when calling `expand_eft_reference_rows` from your own script, not from the CLI) | Multiprocessing needs the calling code guarded by `if __name__ == "__main__":` on macOS/spawn | Wrap your script's entry point in that guard, or pass `workers=1` |
| Finalization appears to hang with no new log lines | Expected — finalization does not print per-row progress | Check `ps`/Activity Monitor for active CPU usage in the worker processes; this indicates it is still computing |

## Related documentation

- [`README_generate_geometric_data_multitriangulation.md`](./README_generate_geometric_data_multitriangulation.md) — full flag reference and scientific contract for both stages.
- [`README_generate_h11_491_frsts.md`](./README_generate_h11_491_frsts.md) — `h11=491`-specific Stage 1 sampling notes.
