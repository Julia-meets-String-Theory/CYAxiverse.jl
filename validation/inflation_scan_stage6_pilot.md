# Stage 6 real-geometry inflation pilot

Stage 6 is implemented by
`scripts/inflation_scan_pilot.jl`. It selects evenly spaced geometries within
each available `h11` group, optionally applies a global cap evenly across that
selected range, and runs the bounded Stage 5 scan-prep path. The screening rows
are then grouped by exact `h11`, instanton-count bin, hierarchy class, and
candidate-count bin.

The pilot was run from the inflation-reproduction worktree with:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_pilot.jl \
  --data-dir ../../data \
  --sample-per-h11 1 --max-geometries 20 --max-branches 100000 \
  --shard-dir /private/tmp/inflation-pilot-real.AJpGcG/shards \
  --report /private/tmp/inflation-pilot-real.AJpGcG/report.csv \
  --run-id stage6-real-20
```

Two high-dimensional rows initially exposed an integer-overflow bug in the
screening bookkeeping: `2^h11` was evaluated as an `Int` and became zero for
`h11=150` and `300`. The normalization now uses `BigInt`; the failed rows were
resumed from the shard and both completed successfully.

## Sample and outcome

The deterministic selected `h11` values were:

```text
4, 7, 9, 12, 15, 17, 20, 23, 25, 28, 31, 34, 36, 39, 42, 44, 47, 50, 150, 300
```

| Result | Count |
|---|---:|
| Successful screening rows | 7 |
| Explicit branch-cap rows | 13 |
| Unexpected failures after resume | 0 |
| Total screening wall time | 6.106 s |
| Mean screening wall time | 0.305 s |
| Total stage allocations | 962,636,720 bytes |
| Mean stage allocations | 48,131,836 bytes |
| Maximum streamed branch count | 32,768 |

All 20 rows had zero candidate slow-roll saddles. The only strong-hierarchy
row was the sampled `h11=15` geometry. The `h11=150` and `h11=300` geometries
completed with zero streamed leading branches, but had substantially larger
stage outputs and allocations: approximately 100 MB / 29.6 MB and 589 MB /
226 MB respectively (allocated bytes / output bytes). These are screening
memory signals, not physical-flow results.

The 13 branch-cap rows are expected bounded outcomes, not numerical failures.
They show that a `max_branches=100000` production scan will need an explicit
policy for high-dimensional geometries: reject them at the Float64 screen,
raise the cap selectively, or use a different reduced-branch strategy.

This is a stratified pilot, not an O(10^5) throughput claim. It does not invoke
the arbitrary-precision trajectory solver.
