# Stage 7 candidate-focused inflation pilot

Date: 2026-08-07

This bounded pilot used the expanded geometry corpus while excluding the
high-memory queue:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_pilot.jl \
  --data-dir ../../data --h11-min 4 --h11-max 100 \
  --sample-per-h11 3 --max-geometries 30 --max-branches 100000 \
  --shard-dir /tmp/inflation-stage7-candidate/shards \
  --report /tmp/inflation-stage7-candidate/report.csv \
  --run-id stage7-candidate-low-middle
```

The deterministic selected h11 values were:

```text
4, 5, 7, 9, 11, 13, 14, 16, 18, 20, 21, 23, 25, 27, 29,
30, 32, 34, 36, 38, 39, 41, 43, 45, 46, 48, 50, 70, 90, 100
```

| Measurement | Value |
|---|---:|
| Geometries | 30 |
| Successful screens | 7 |
| Branch-cap rows | 23 |
| Empty-enumeration rows | 0 |
| Unexpected failures | 0 |
| Candidate saddles | 0 |
| Candidate rate | 0 / 30 |
| Cold wall time | 0.313 s |
| Warm wall time | 1.404 s |
| Total stage allocations | 332,142,464 bytes |
| Maximum stage allocation | 190,938,464 bytes |

The successful geometries were h11=4, 5, 7, 9, 11, 13, and 14. The 23
branch-cap outcomes are expected bounded results, not failures. This pilot
does not establish that the expanded corpus contains no physical candidates;
it only reports the bounded Float64 screen on this stratified sample.
