# h11=491 full geometry generator

[`generate_h11_491_frsts.py`](./generate_h11_491_frsts.py) is a focused
companion to
[`generate_geometric_data_multitriangulation.py`](./generate_geometric_data_multitriangulation.py).
It uses the checked-in
[`h11_491_11_ks.json`](./manifests/h11_491_11_ks.json) manifest for the unique
favorable N-lattice KS polytope with Hodge numbers `(491, 11)`, then delegates
the CY3 construction and HDF5 write to the package's existing
`generate_and_save_geometry` function.

The default run is bounded and reproducible:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
KMP_DUPLICATE_LIB_OK=TRUE \
python scripts/generate_h11_491_frsts.py \
  --outdir /private/tmp/cyax-h11-491-geometry \
  --sampling-scheme ntfe_fast \
  --n 1 --proposal-budget 300 --seed 20260813
```

The output uses the normal package path and HDF5 structure:

```text
OUTDIR/h11_491/np_0000001/cy_0000001/cyax.h5
├── cytools/geometric
│   ├── points, triangulation_points, simplices, glsm, basis, basis_matrix
│   ├── prime_toric_divisors, prime_divisor_charges, tip, tip_prefactor
│   ├── CY_volume, divisor_volumes, prime_divisor_volumes, curve_volumes
│   ├── Kinv, kappa, c2, effective_cone, mori_cone, kahler_hyperplanes
│   └── optional standard_model, orientifold, visible_sector groups
├── cytools/potential
│   ├── L  (2 × N sign/log10 representation)
│   └── Q  (h11 × N charge matrix)
└── construction_metadata
```

The package's physical-generation options are available: `--max-m`,
`--max-kaehler-attempts`, divisor-volume bounds, QCD volume window and target,
`--moduli-policy`, `--qcd-divisor-index`, `--visible-sector-policy`,
`--qed-divisor-index`, `--orientifold-file`, and `--export-kahler-rays`.
Rejected FRSTs and terminal failures are retained in `report.json`; the HDF5
writer itself remains atomic, as in the package generator.

The default `ntfe_fast` path explicitly counts two-face-combination extension
attempts. On the benchmarked polytope, the `fast` two-face sampler may produce
only one proposal per face, so a run can finish with `source_exhausted` before
reaching a larger requested target. This is an intentional finite-support
result, not a claim that the emitted finite set is representative. The
`gnn_ntfe` path is available when the optional `dualgnn` environment is
installed, but its high-h11 pool build is substantially more expensive.
