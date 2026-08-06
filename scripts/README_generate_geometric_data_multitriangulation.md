# KS-to-CY3 geometry generator

[`generate_geometric_data_multitriangulation.py`](./generate_geometric_data_multitriangulation.py)
is the CYTools-backed preprocessing step for the CYAxiverse geometry database.
It fetches Kreuzer-Skarke (KS) polytopes, constructs fine regular star
triangulations (FRSTs), builds the corresponding Calabi-Yau threefold
hypersurfaces, selects a controlled point in the toric Kahler cone, and writes a
versioned HDF5 artifact for the Julia pipeline.

The implementation is deliberately CYTools-first: polytope construction,
triangulation, cone extraction, intersection data, and geometric validation are
delegated to public CYTools APIs. The script does not reimplement toric
triangulation algorithms.

## Scientific status

The algorithm is intended to be robust and reproducible under the current
understanding of the KS-to-CY3 construction. It is not a proof that the finite
sample is uniform over all CY3s, nor does its fingerprint prove that two
geometries are diffeomorphic. Every output records its construction and
validation metadata so later statistical or Bayesian analyses can condition on
the actual sampling procedure.

The construction choices are:

1. Fetch a reflexive, full-dimensional four-polytope from the KS database.
2. Use the reflexive default point configuration: all lattice points except
   points strictly interior to facets, while retaining the origin.
3. Generate FRST candidates using CYTools:
   - `fair` (default): secondary-fan random walks and flips following
     Demirtas, McAllister, and Rios-Tascon, arXiv:2008.01730.
   - `fast`: random heights near a Delaunay triangulation. This is useful for
     coverage scans and smoke tests, but is not a fair ensemble.
4. Validate fineness, regularity, the star condition, and triangulation
   validity before constructing the hypersurface.
5. Extract Hodge data, intersection numbers, the divisor basis, the Mori cone,
   the Kahler cone, the effective cone, and the second Chern class.
6. Prefer MOSEK for stretched-cone quadratic optimization when a valid license
   and the qpsolvers MOSEK backend are available.
7. Require positive effective-divisor volumes and search for the smallest
   stretched-cone prefactor satisfying the instanton/potential-control
   criterion used in arXiv:2309.01831.
8. Apply the fuzzy-axion QCD selection from arXiv:2412.12012: every prime
   toric divisor has volume at least one, and at least one prime divisor lies in
   the configurable `[25, 40]` volume window.
9. If an explicit orientifold is supplied, validate its lattice involution,
   polytope preservation, FRST preservation, induced integral H2 action, and
   invariant Kaehler-cone intersection.
10. Write atomically to `cyax.h5`; incomplete temporary files are not retained.

### Literature basis

The construction and its downstream data contract were checked against the
following references:

- *Complete classification of reflexive polyhedra in four
  dimensions*, hep-th/0002240. This is the source classification underlying
  the KS four-dimensional reflexive-polytope database.
- *Bounding the Kreuzer-Skarke
  Landscape*, arXiv:2008.01730. This motivates the secondary-fan random-walk
  and flip sampler used by the `fair` mode.
- *Axion minima in string theory*, arXiv:2309.01831. This
  supplies the stretched-cone and instanton-control criterion used when
  accepting a geometry.
- *Superradiance in String Theory*, arXiv:2103.06812. This
  is a downstream CYAxiverse physics target and informs preservation of the
  divisor, kinetic, and instanton-potential data needed by later analyses.
- *Bayesian inference on Calabi-Yau moduli spaces and the axiverse:
  experimental data meets string theory*, arXiv:2512.00144. This motivates
  immutable geometry metadata, explicit cone/basis conventions,
  construction metadata,
  and serializable arrays for future inference.
- *Orientifolding Kreuzer-Skarke*, arXiv:2305.06363. This
  motivates requiring an explicit involution and invariant triangulation before
  exporting orientifold-even/odd data.
- *Fuzzy Axions and Associated Relics*, arXiv:2412.12012. Equation (3.22) and
  its surrounding construction motivate the prime-divisor lower bound and
  QCD-visible divisor-volume window.

The first four references determine the geometry-generation and explicit
orientifold handoff directly. The remaining references determine which
numerical geometry contract and construction metadata must survive into Julia.
The script records the directly operative construction papers in the HDF5
`construction_metadata_json`.

The QCD-volume filter is a geometry-selection criterion, not a claim that the
output contains a complete Standard Model sector. It is applied to the raw
prime toric divisor volumes returned by CYTools, not to favorable basis
divisors or effective-cone ray volumes.

## Environment

Use the local CYTools environment. In a new shell:

```bash
source activate cytools
python --version
python -c 'import cytools; print(cytools.version)'
```

The environment must provide:

- CYTools >= 1.4.0;
- NumPy, h5py, and python-flint;
- qpsolvers plus at least one usable backend;
- the CYTools triangulation backend selected by `--backend`;
- MOSEK and a valid license if MOSEK optimization is desired.

The script looks for a license in this order:

1. `MOSEKLM_LICENSE_FILE`, if already set;
2. `$HOME/mosek.lic`, if present.

Only the path is exposed to the solver. The license contents are never read,
copied, serialized, or written to the HDF5 output.

The script fetches polytopes through `cytools.fetch_polytopes`, so the first
generation run requires network access to the configured KS endpoint. For Julia
consumers, instantiate the package separately:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Basic invocation

From the repository root:

```bash
source activate cytools
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 4 \
    --h11_max 8 \
    --n 10 \
    --outdir /path/to/database \
    --seed 1234 \
    --cores 4 \
    --verbose
```

This requests ten output geometries for each `h11` from 4 through 8,
distributes them over the favorable polytopes returned by CYTools, and writes
one file per accepted triangulation. `--n` counts accepted output geometries,
not the number of raw FRST attempts and not necessarily the number of distinct
polytopes.

To process only selected values, pass an explicit list:

```bash
python scripts/generate_geometric_data_multitriangulation.py \
    --h11s '[4,10,20,50]' \
    --n 10
```

Alternatively, use an interval with the inclusive minimum and maximum:

```bash
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 10 --h11_max 50 --h11_interval 10 \
    --n 10
```

The batch is parallelized by polytope. Each worker receives only the polytope
vertices and scalar generation options, reconstructs its own CYTools objects,
generates its assigned FRST candidates, and writes distinct output slots. This
keeps large CYTools objects out of inter-process serialization and makes the
parallel path the normal path rather than a special execution mode.

The command exits successfully only after all requested `h11` batches have
completed without worker errors. A shortfall caused by rejected candidates is
reported in the summary; increase the attempt budgets when a full requested
sample is needed.

## Command-line options

### Geometry range and output

| Option | Default | Meaning |
| --- | --- | --- |
| `--h11_min INT` | `4` | First `h11` value, inclusive. |
| `--h11_max INT` | `4` | Last `h11` value, inclusive. Values below `h11_min` are clamped to `h11_min`. |
| `--h11_interval INT` | `1` | Step used between `h11_min` and `h11_max`, inclusive. Must be positive. |
| `--h11s H11 [H11 ...]` | unset | Explicit positive, unique `h11` values. Values may be comma-separated or supplied as a bracketed list; this replaces the min/max range. |
| `--n INT` | `1` | Number of accepted geometries requested per `h11`. Must be positive. |
| `--outdir PATH` | `.` | Root directory for the generated database. |
| `--cores INT` | all available | Number of worker processes. Use `1` for debugging and deterministic logs. |
| `--seed INT` | `0` | Base seed. Worker and candidate seeds are derived from it and recorded in construction metadata. |
| `--overwrite` | off | Replace existing output slots. Without this flag, existing `cyax.h5` files count toward `--n` and are skipped. |
| `--verbose` | off | Print per-worker stages and elapsed times. Recommended for high `h11`. |

For a reproducible rerun, keep the same CYTools version, database endpoint
label, sampler/backend options, seed, and output policy. The random seed does
not make different CYTools or solver versions scientifically interchangeable.

### Parallel execution

`--cores` maps directly to the number of Python worker processes:

```bash
source activate cytools
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 8 --h11_max 8 --n 100 \
    --cores 16 \
    --outdir /scratch/database \
    --seed 1234
```

Use `--cores 1 --verbose` when diagnosing a failure. With multiple workers:

- polytopes are independent jobs and can be processed concurrently;
- each worker activates/configures its own CYTools and solver state;
- MOSEK license discovery is process-safe because workers receive only the
  license path through the environment;
- HDF5 files are written atomically to separate geometry paths;
- the parent process aggregates accepted, rejected, skipped, and failed jobs;
- a worker exception causes the batch to fail loudly instead of producing a
  misleading partial-success result.

Do not launch two independent invocations against the same output directory
with overlapping `h11`/polytope/output slots. Use separate output roots or let
one invocation own the complete batch. The script's internal workers do not
share output files, random-number generators, or mutable CYTools objects.

### KS selection

| Option | Default | Meaning |
| --- | --- | --- |
| `--favorable {true,false,any}` | `true` | Fetch favorable N-lattice polytopes (`true`), non-favorable polytopes (`false`), or do not impose the favorability filter (`any`). |
| `--ks-database-version TEXT` | endpoint label | Provenance label for the KS source. CYTools does not expose the remote database version automatically, so set this explicitly when the endpoint/version is known. |

The script reconstructs each fetched polytope from its vertices in each worker.
This avoids passing a large lattice-point object between processes and is
especially important for the high-`h11` KS cases.

### FRST sampling

| Option | Default | Meaning |
| --- | --- | --- |
| `--sampling-scheme {fair,fast}` | `fair` | Select the secondary-fan sampler or the faster biased height sampler. |
| `--backend {cgal,qhull}` | `cgal` | CYTools backend for triangulation construction. |
| `--max-retries INT` | `50` | Maximum sampler retries for a new triangulation. |
| `--max-tip-attempts INT` | `50` | Maximum FRST candidates tested per polytope before reporting a shortfall. |
| `--n-walk INT` | CYTools default | Fair-mode secondary-fan walk steps per sample. |
| `--n-flip INT` | CYTools default | Fair-mode random flips per sample. |
| `--initial-walk-steps INT` | CYTools default | Fair-mode burn-in walk steps before recording samples. |
| `--fine-tune-steps INT` | `8` | Fair-mode wall-location refinement steps. |
| `--walk-step-size FLOAT` | `0.01` | Fair-mode secondary-fan step size. |
| `--max-steps-to-wall INT` | `25` | Fair-mode maximum steps toward a fine-triangulation wall. |
| `--fast-height-scale FLOAT` | `0.2` | Standard deviation of the Gaussian height perturbation in fast mode. |

`fair` is the production default. For small polytopes, CYTools warns that the
fair random walk may stall because the secondary-fan state space is too small
for its intended Markov-chain procedure. This is not evidence that the
resulting fast sample is fair. Use `fast` for a smoke test or biased coverage
scan, and use `fair` with the default or carefully tuned walk controls for the
large KS targets for which it was designed.

Do not reduce the fair controls aggressively just to make a small fixture run:
very small values can produce `Couldn't find wall` or `There was an error in
the random walk`. If a fair run stalls, try the other backend, restore the
CYTools defaults, increase `--max-steps-to-wall`, or move to a larger target
polytope.

### Kahler-cone and acceptance controls

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-kaehler-attempts INT` | `100` | Number of stretched-cone/angular Kahler points tested per FRST, including the canonical tip. |
| `--min-divisor-volume FLOAT` | `1.0` | Minimum allowed volume for every exported effective-cone ray after rescaling. |
| `--min-prime-divisor-volume FLOAT` | `1.0` | Minimum allowed volume for every raw prime toric divisor. |
| `--qcd-volume-min FLOAT` | `25.0` | Lower edge of the QCD-visible prime-divisor volume window. |
| `--qcd-volume-max FLOAT` | `40.0` | Upper edge of the QCD-visible prime-divisor volume window. |
| `--orientifold-file PATH` | off | JSON file containing an explicit lattice involution and O3/O7 or O5/O9 metadata. |
| `--max-m FLOAT` | `1_000_000` | Maximum stretched-cone prefactor searched for the potential-control criterion. |

The canonical stretched-cone tip is found by minimizing the Euclidean norm
subject to unit distance from the Kahler-cone hyperplanes. When MOSEK is
licensed and available, the script explicitly requests MOSEK for this solve.
For randomized angular projections, licensed MOSEK is tried first; other
qpsolvers backends are fallback candidates. The actual tip and projection
solvers are recorded per artifact.

Candidate acceptance requires:

- a valid FRST and smooth generic CY hypersurface;
- finite topology and cone data;
- a positive physical CY volume and positive curve volumes;
- positive effective-divisor volumes;
- every prime toric divisor volume at least `--min-prime-divisor-volume`;
- at least one prime toric divisor volume in
  `[--qcd-volume-min, --qcd-volume-max]`;
- Kahler-cone hyperplane slack at least one at the final tip;
- the stretched-cone instanton/potential-control inequality to pass.

Rejected candidates are discarded and replaced until the requested count or
the attempt budget is exhausted.

## Useful command recipes

### Small smoke test

The fast mode is appropriate for verifying the installation and HDF5 schema:

```bash
source activate cytools
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 4 --h11_max 4 --n 1 \
    --sampling-scheme fast \
    --cores 1 --seed 17 \
    --max-tip-attempts 5 \
    --max-kaehler-attempts 3 \
    --outdir /tmp/cyaxiverse-smoke \
    --verbose
```

This checks the full geometry-writing path but is intentionally not a fair
ensemble measurement.

### Reproducible fair sample

```bash
source activate cytools
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 8 --h11_max 8 --n 100 \
    --sampling-scheme fair \
    --backend cgal \
    --seed 20260805 \
    --cores 8 \
    --max-tip-attempts 200 \
    --max-kaehler-attempts 100 \
    --ks-database-version "KS/CYTools endpoint, 2026-08-05" \
    --outdir /scratch/database \
    --verbose
```

### High-`h11` exploratory run

Start with a small accepted count and a large attempt budget before scaling up:

```bash
source activate cytools
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 491 --h11_max 491 --n 10 \
    --sampling-scheme fair \
    --seed 491001 \
    --cores 1 \
    --max-tip-attempts 100 \
    --max-kaehler-attempts 100 \
    --outdir /scratch/database/h11_491 \
    --verbose
```

The `h11=491` case is computationally demanding. Expect substantial memory,
CPU time, cone optimization, and HDF5 I/O. A successful run for one
high-`h11` geometry does not establish uniform coverage of the triangulation
ensemble.

### SLURM

The batch script must activate the environment on the compute node, not only
on the login node:

```bash
source activate cytools
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 8 --h11_max 8 --n 100 \
    --cores "${SLURM_CPUS_PER_TASK}" \
    --outdir /scratch/database \
    --seed 1234
```

See [`How_to_slurm.md`](./How_to_slurm.md) for the existing submission notes.

### Explicit orientifold handoff

Orientifolding is opt-in. The generator does not infer an involution, treat the
identity as a physical orientifold, or claim fixed-locus/tadpole results. Pass a
JSON file when a lattice involution has been selected from an external
orientifold analysis:

```json
{
  "label": "example-involution",
  "lattice_matrix": [
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
  ],
  "involution_type": "O3/O7",
  "coefficient_constraints": {}
}
```

Run with `--orientifold-file path/to/involution.json`. The matrix must be an
integer unimodular involution that fixes the origin, preserves every lattice
point of the polytope, and maps the selected FRST to itself. The script then
derives the integral H2 action in the exported divisor basis, computes
`h11_plus`/`h11_minus`, exports invariant and anti-invariant bases, and verifies
that the orientifold-even Kaehler subspace intersects the Kaehler cone. O3/O7
versus O5/O9 and coefficient constraints remain explicit metadata; fixed-locus
topology, tadpole cancellation, fluxes, and phenomenology are downstream
calculations.

## Output layout and HDF5 contract

Each accepted geometry is written to:

```text
OUTDIR/
└── h11_XXX/
    └── np_YYYYYYY/
        └── cy_ZZZZZZZ/
            └── cyax.h5
```

The indices are zero-padded and identify output slots. They are not substitutes
for the stable polytope and triangulation fingerprints stored in construction
metadata.

The file contains:

```text
cytools/geometric/
  points
  triangulation_points
  simplices
  h11, h21
  glsm
  basis, basis_matrix
  tip, tip_prefactor
  CY_volume
  divisor_volumes, prime_divisor_volumes, curve_volumes
  Kinv
  kappa
  c2
  effective_cone
  mori_cone
  kahler_cone
  kahler_hyperplanes
  orientifold/ (only with --orientifold-file)
    lattice_matrix
    h2_involution_matrix
    invariant_kahler_basis
    anti_invariant_h2_basis
    invariant_kahler_point
cytools/potential/
  L, Q
construction_metadata/
  canonical_lattice_points
  face_restriction_dim2
```

Important conventions:

- `kappa` is sparse COO with columns `[i, j, k, value]` and zero-based indices.
- Vectors and matrices are in the CYTools divisor-basis convention recorded in
  `basis_convention`.
- `L` retains the CYAxiverse sign/mantissa and base-10 exponent representation.
- `tip`, `divisor_volumes`, `Kinv`, `curve_volumes`, and `CY_volume` refer to the
  same final rescaled Kahler point.
- `prime_divisor_volumes` follows the order of CYTools
  `prime_toric_divisors()` and is the vector used for the QCD filter; the
  recorded QCD divisor index is zero-based.
- `construction_metadata_json` is stored both as a root attribute and as an
  attribute of the `construction_metadata` group.
- `cy3_fingerprint` is explicitly a conservative topological fingerprint; it
  is not a complete diffeomorphism or birational-equivalence test.

The construction metadata records the CYTools version, KS label, sampler and
all sampler controls, seed, favorability choice, FRST validation flags,
basis/intersection conventions, polytope identity, triangulation identity,
topology fingerprint, MOSEK status, and solver choices.

## Inspecting an artifact

Python:

```bash
source activate cytools
python - <<'PY'
import json
import h5py

path = "/path/to/database/h11_008/np_0000001/cy_0000001/cyax.h5"
with h5py.File(path, "r") as f:
    construction_metadata = json.loads(f.attrs["construction_metadata_json"])
    print("schema:", construction_metadata["schema_version"])
    print("CYTools:", construction_metadata["cytools_version"])
    print("h11/h21:", f["cytools/geometric/h11"][()], f["cytools/geometric/h21"][()])
    print("sampler:", construction_metadata["sampling"]["scheme"])
    print("tip solver:", construction_metadata["mosek_license"]["tip_solver"])
PY
```

Julia:

```bash
julia --project=. -e '
using HDF5
path = "/path/to/database/h11_008/np_0000001/cy_0000001/cyax.h5"
h5open(path, "r") do f
    println(read(f["cytools/geometric/h11"]), "/", read(f["cytools/geometric/h21"]))
end
'
```

## Interpretation and limitations

- `n` is a sample-size request, not a statement about the total number of
  distinct CY3s at a given `h11`.
- `fair` samples triangulations using CYTools' secondary-fan procedure; it does
  not make the KS-polytope selection uniform, and rejection by geometric
  acceptance criteria changes the retained ensemble.
- `fast` is biased by construction and should be labeled as such in downstream
  statistics.
- The favorable filter changes the population being sampled. Record it and do
  not compare `--favorable true` and `--favorable any` as if they were the same
  ensemble.
- The canonical point configuration excludes facet-interior points for
  reflexive polytopes. This is a construction convention, not a claim that all
  other point configurations are invalid in every toric application.
- A successful MOSEK solve improves numerical robustness for large cones but
  does not remove the need for FRST, cone, volume, and physical consistency
  checks.
- Provenance makes future Bayesian conditioning possible; it does not itself
  specify a prior or correct for all selection effects.
