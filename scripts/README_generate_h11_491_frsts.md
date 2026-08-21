# h11=491 full-geometry generator

[`generate_h11_491_frsts.py`](./generate_h11_491_frsts.py) is the specialist
entry point for the unique favorable N-lattice Kreuzer--Skarke polytope with
Hodge numbers `(h11, h21) = (491, 11)`. It retains the branch’s NTFE and
dualGNN proposal families, then delegates each accepted FRST to the full
CYTools/Kähler/potential HDF5 writer.

The default run is bounded and reproducible. It is a finite proposal experiment,
not an exhaustive or population-representative sample:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
KMP_DUPLICATE_LIB_OK=TRUE \
python scripts/generate_h11_491_frsts.py \
  --database-source manifest \
  --outdir /private/tmp/cyax-h11-491-geometry \
  --sampling-scheme ntfe_fast \
  --n 1 --proposal-budget 300 --seed 20260813
```

The checked-in manifest is the default replay source. A downloaded mirror can
be used instead:

```bash
python scripts/generate_h11_491_frsts.py \
  --database-source mirror \
  --parquet-dir /data/ks-polytopes-4d \
  --outdir /private/tmp/cyax-h11-491-mirror \
  --sampling-scheme ntfe_fast --n 1 --proposal-budget 300 --seed 20260813
```

The mirror is the `calabi-yau-data/polytopes-4d` Parquet dataset. Its Hodge
labels are dual to the physical N-lattice query: physical `h11` is mirror
`h12`, and physical `h21` is mirror `h11`. Published vertex representatives
are consumed directly; the script does not dualize them a second time. The
report and each HDF5 construction record preserve the source kind, dataset
URL or manifest path, partition/row or manifest hash, and the Hodge mapping.
The mirror path requires `pyarrow` in the active CYTools environment.

## Output and provenance

Accepted files use the package layout:

```text
OUTDIR/h11_491/np_0000001/cy_0000001/cyax.h5
├── cytools/geometric
│   ├── points, triangulation_points, simplices, glsm, basis, basis_matrix
│   ├── prime_toric_divisors, prime_divisor_lattice_points, prime_divisor_charges
│   ├── tip, tip_prefactor, CY_volume, divisor_volumes, prime_divisor_volumes
│   ├── Kinv, kappa, c2, effective_cone, mori_cone, kahler_hyperplanes
│   └── optional orientifold, standard_model, visible_sector groups
├── cytools/potential
│   ├── L  (2 × N sign/log10 representation)
│   └── Q  (h11 × N exact integer charge matrix)
└── construction_metadata
```

`report.json` is written atomically and records the command, source identity,
CYTools/GNN environment, sampler controls, proposal budget semantics,
triangulation hashes, physical-selection outcomes, timing/resource data, and
bounded failure records. Existing output slots are resumed unless
`--overwrite` is supplied; HDF5 installation is collision-safe when overwrite
is disabled.

The specialist proposal controls are:

- `--sampling-scheme {fair,fast,ntfe_fast,gnn_ntfe}`;
- `--proposal-budget INT` (explicit two-face extension attempts when
  `--exact-proposals` is enabled);
- `--ntfe-face-sampler`, `--ntfe-max-face-points`, and
  `--ntfe-face-pool-size`;
- `--seed`, `--max-retries`, `--backend`, and the fair-walk controls.

`ntfe_fast` and `gnn_ntfe` are recorded as two-face-inequivalent proposal
families with finite face-pool support. `fast` is a deliberately biased height
proposal and `fair` retains the CYTools secondary-fan semantics. A source
exhaustion or proposal-budget shortfall is a valid terminal result, not a
reason to call the emitted set representative.

## Geometry and visible-sector options

The physical construction controls include `--max-m`,
`--max-kaehler-attempts`, divisor-volume bounds, `--moduli-policy`,
`--qcd-volume-target`, `--qcd-divisor-index`,
`--visible-sector-policy`, `--orientifold-file`, and
`--export-kahler-rays`. The QCD divisor index remains a zero-based internal
CYTools index for the canonical-QCD moduli policy.

The optional `intersecting_d7` policy is a geometry-derived toy assignment.
It requires a validated O3/O7 orientifold, records any computed
`h11_plus`/`h11_minus` split, and selects an
orientifold-invariant prime divisor intersecting QCD, derives its exact
integer divisor-basis charge, and verifies that the corresponding potential
term is present. It is not a global tadpole, matter-spectrum, or physical
Standard Model brane construction.

QED selection is explicit and auditable:

```bash
python scripts/generate_h11_491_frsts.py \
  --visible-sector-policy intersecting_d7 \
  --orientifold-file /path/to/orientifold.json \
  --qed-selection-policy uniform_eligible \
  --qed-selection-seed 17 \
  --qed-volume-filter source_aligned \
  --qed-volume-max 127.5
```

`uniform_eligible` samples uniformly from the stable, pre-volume-filter pool;
`explicit` requires `--qed-divisor-index`. That user-facing QED index is
one-based and is never silently replaced. `--qed-volume-filter
source_aligned` applies the upper bound after the eligible pool is declared,
then records a `qed_volume_rejection` if the selected divisor fails. The
report and `visible_sector` group retain the pool indices/labels, selection
seed/rank, charge hashes, intersection evidence, potential-source match,
leading-rank certificate, and terminal status.

## Interpretation

The generator’s target population is the fixed `(491, 11)` favorable polytope
and the selected FRST proposal family. Retained geometries are additionally
conditioned on FRST validity, Kähler/cone construction, divisor-volume and
potential-control filters, and any visible-sector policy. Compare runs only
when source, sampler, seed controls, backend, proposal budget, and physical
filters are recorded consistently. Provenance enables later conditioning; it
does not correct the proposal or acceptance measure.
