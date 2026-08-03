# User guide

!!! warning
    Under construction

## Physical PQ spectra

`pq_spectrum` provides the fast leading-Hessian spectrum intended for broad
surveys. For a threshold-certified physical sector, use the separate
high-precision APIs:

```julia
potential = CYAxiverse.read.potential(geom_idx)

count = CYAxiverse.generate.pq_physical_mode_count(
    potential.K, potential.L, potential.Q; prec=200)

spectrum = CYAxiverse.generate.pq_hybrid_physical_spectrum(
    potential.K, potential.L, potential.Q;
    prec=200,
    quartics=false,
    label="my geometry",
)
```

The mode count is confirmed at increasing arbitrary precision. A warning marks
any result whose count or refined eigenpairs do not stabilize; such a result is
provisional and should be rerun at higher precision or checked with
`pq_physical_spectrum`.

Set `quartics=false` for large mass-only scans. When quartics are requested,
every input instanton contributes to their contraction even though only the
physical mass eigenvectors are returned.

For resumable ensemble runs, use `scripts/batch_physical_spectrum.jl`. It
discovers indexed or on-disk geometries, flushes one CSV row per geometry, and
writes compact results below `physical_spectrum/`:

```sh
julia --project=. scripts/batch_physical_spectrum.jl \
    --data-dir /path/to/data --h11 200 --limit 100 --mass-only \
    --summary /path/to/data/logs/physical_spectrum_h11_200.csv

julia --project=. scripts/batch_physical_spectrum.jl \
    --data-dir /path/to/data --geometry 10,1,1 --quartics --force
```

The quartic batch mode computes only diagonal self-couplings and derives
`fpert_log10 = m - 0.5 * lambda_self_log10`; mixed quartic arrays and
eigenvectors are not persisted. Existing output is skipped unless `--force` is
given, and per-geometry failures are recorded without stopping the batch.

The hybrid solver accepts `quartic_backend=:auto` (the default). It retains
the dense path for small or dense charge matrices and switches to sparse charge
transforms for large sparse matrices. Use `quartic_backend=:dense` or
`:sparse` to force a backend while benchmarking. With `quartics=true` and
`mixed_quartics=false`, diagonal contractions stream over instantons and do
not allocate the full physical-mode charge matrix.

`pq_spectrum` returns `m` together with the aligned Hessian-eigenvalue signs in
`msign`. Its sequential PQ decay quantity is `f`; the legacy spectrum HDF5
schema stores this array under `decay/fpert` for compatibility with existing
readers.

### Large-geometry scaling checkpoint

The mass-only hybrid solver was validated on ten geometries each at
`h11=150`, `180`, and `200`. Physical-mode counts agreed at 150 and 200 digits
for all 30 inputs, and every 200-digit hybrid solve converged without a
provisional warning.

| `h11` | Instantons per input | Physical modes observed | Mean selector time | Mean hybrid time |
|---:|---:|---:|---:|---:|
| 150 | 11,935 | 4–14 | 0.204 s | 5.83 s |
| 180 | 17,020 | 3–9 | 0.437 s | 9.97 s |
| 200 | 20,910 | 5–8 | 0.649 s | 14.05 s |

The selector uses an allocation-free candidate scan and was checked for exact
ordered-column agreement with the previous repeated-rank implementation on
all 1,000 available `h11=10` inputs and all ten `h11=20` inputs. The
high-precision leading Hessian accumulates only over nonzero charge support.
The hybrid solver also reuses its selected data and requested-precision Hessian
between inertia counting and eigenvector refinement.

At `h11=200`, the selected charge matrix in the profiled input was 0.99% dense.
Sparse Hessian accumulation took 0.078 seconds, while canonical whitening took
3.77 seconds. The whitening makes canonical charges dense; a direct
canonical-charge rank-one construction was tested and was slower, so sparse
storage is intentionally confined to the pre-whitening accumulation stage.
