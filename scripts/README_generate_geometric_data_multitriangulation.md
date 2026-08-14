# KS-to-CY3 geometry generator

[`generate_geometric_data_multitriangulation.py`](./generate_geometric_data_multitriangulation.py)
is the CYTools-backed geometry implementation used by the CYAxiverse geometry
database. The schema-1.1 workflow is intentionally split into two commands:
`generate_stage1_raw_frsts.py` collects the approved raw FRST population, and
`generate_stage2_eft_reference.py` reconstructs only those retained FRSTs and
applies the geometry/EFT checks. The older combined `--eft` route is retired.

Schema 1.1 adds a compact, geometry-level factorized charge contract and an
adapted finite model-reuse diagnostic, not a reproduction of the Glimmers model
ensemble.

## Independent stage runs

Stage 1 writes only serializable raw FRST artifacts. It does not choose an
orientifold, search the Kähler cone, apply divisor-volume cuts, or construct EFT
rows:

```bash
python scripts/generate_stage1_raw_frsts.py \
  --h11-plan 50:500,100:500,200:300,491:100 \
  --outdir /path/to/stage1_raw_frsts
```

Stage 2 takes that output as a fixed input population. It never generates a
replacement FRST; missing or corrupt raw files are recorded as unavailable in
the separate `stage2_input_ledger.jsonl`:

```bash
python scripts/generate_stage2_eft_reference.py \
  --stage1-root /path/to/stage1_raw_frsts \
  --outdir /path/to/stage2_eft \
  --orientifold-file /path/to/orientifold.json \
  --eft
```

Use `--dry-run` for a deterministic input-ledger probe. Stage-2 filters may
reduce the number of accepted geometries, but they never replenish or alter the
stage-1 FRST population.

Before a production stage-2 run, confirm the remaining orientifold, divisor,
visible-sector, QCD/QED-pool, potential, and EFT stopping choices. The command
records these choices and any deferrals in `run_manifest.json`; it does not turn
an unconfirmed choice into a physical claim.

The production Stage-2 reference policy is `canonical_qcd`, matching the
Glimmers-style construction: retain the stretched-cone tip, inspect eligible
prime divisors in deterministic index order, skip any whose tip volume already
exceeds 40, and dilate the first remaining eligible choice so its QCD divisor
volume is exactly 40. The radial factor therefore satisfies `m >= 1` by
default. The Stage-2 `--allow-m-below-one` flag is an explicit opt-in to the
unrestricted normalization convention, allowing contraction when the selected
tip divisor already exceeds 40; the choice is recorded in run and HDF5
construction metadata.
`adaptive` remains available only as an explicitly labelled randomized Kähler
diagnostic with a 100-point budget, including the canonical tip.

The stage-2 reconstruction also writes `stage2_topology_diagnostics.jsonl` and
`stage2_kaehler_point_diagnostics.jsonl`.
It contains a compact structural audit for every input: polytope dimensions and
reflexivity, FRST checks, smoothness, CYTools Hodge data, basis convention, and
the shapes/finiteness of intersection, Chern, Mori, and Kähler-cone arrays.
It records the ledger, raw-dataset, and reconstructed-CYTools triangulation
hashes separately. Any disagreement is terminal `input_identity_mismatch`.

The reference run uses `--orientifold-kaehler-policy none`: it validates and
records the orientifold action but does not require an orientifold-even Kähler
subspace intersection. Use `require_even_subspace` for a separate run that
requires that physical orientifold constraint; an infeasible intersection then
terminates as `kaehler_tip_failure`.

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
   - `ntfe_fast`: sample FRTs on the two-faces, then directly extend only
     feasible combinations to two-face-inequivalent (NTFE) FRSTs using the
     algorithm of MacFadden, arXiv:2309.10855. This avoids doing expensive CY
     work repeatedly for 2-face-equivalent FRSTs. Its finite face pools are a
     bounded coverage proposal, not a uniform NTFE or FRST sample.
   - `gnn_ntfe`: use CYTools' optional dualGNN 2-face proposal and the same
     direct NTFE extension. It is a distinct learned-proposal ensemble whose
     model, pool size, and extension failures determine the realised support.
4. Validate fineness, regularity, the star condition, and triangulation
   validity before constructing the hypersurface.
5. Extract Hodge data, intersection numbers, the divisor basis, the Mori cone,
   Kähler-cone hyperplanes, the effective cone, and the second Chern class.
   Kähler-cone rays are an optional export because enumerating them can be
   prohibitively expensive at large `h11`.
6. Prefer MOSEK for stretched-cone quadratic optimization when a valid license
   and the qpsolvers MOSEK backend are available.
7. Under the default `adaptive` policy, require positive effective-divisor
   volumes and search for the smallest stretched-cone prefactor satisfying the
   instanton/potential-control criterion used in arXiv:2309.01831.
8. Apply the fuzzy-axion QCD selection from arXiv:2412.12012. The default
   `adaptive` moduli policy requires every prime toric divisor to have volume
   at least one and places at least one prime divisor in the configurable
   `[25, 40]` volume window. The optional `canonical_qcd` policy keeps the
   canonical stretched-cone tip ray, samples a pairwise-intersecting triple of
   prime toric divisors, and applies a homogeneous radial scaling so its QCD
   member has volume 40 by default.
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
- *Efficient Algorithm for Generating Homotopy Inequivalent Calabi-Yaus*,
  arXiv:2309.10855. This supplies the direct two-face/secondary-cone extension
  used by the NTFE modes.
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
- *Glimmers from the Axiverse*, arXiv:2309.13145. Its geometry-level recipe
  motivates the canonical tip, random intersecting-divisor assignment, and
  QCD divisor volume normalization implemented by `canonical_qcd`.

The first four references determine the geometry-generation and explicit
orientifold handoff directly. The remaining references determine which
numerical geometry contract and construction metadata must survive into Julia.
The script records the directly operative construction papers in the HDF5
`construction_metadata_json`.

The QCD-volume filter is a geometry-selection criterion, not a claim that the
output contains a complete Standard Model sector. It is applied to the raw
prime toric divisor volumes returned by CYTools, not to favorable basis
divisors. In `canonical_qcd`, the sampled triple is a geometric assignment
recorded in `standard_model/`; it does not construct a D7-brane stack, a QED
divisor, or stringy instanton data. The effective-cone, curve-volume, and
Kähler-slack checks remain package-level physical-domain guards in addition to
the paper's raw prime-divisor `< 1` rejection criterion.

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

The schema 1.1 generator intentionally has no GNN, dualGNN, PyTorch, or learned
sampler path. This keeps the requested NTFE run on the native CYTools
implementation only.

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
parallel path the normal path rather than a special execution mode. When
multiple `h11` values are requested, they share one worker pool: a long-running
polytope at one `h11` does not prevent idle workers from processing later
`h11` values. The command still waits for every submitted task before exiting;
it does not automatically cancel a slow task.

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
| `--n INT` | `1` | Number of accepted geometries requested per `h11`; with `--frsts-per-polytope`, number of favorable polytopes to fetch. |
| `--frsts-per-polytope INT` | off | Set the per-polytope FRST target. `--n` then counts polytopes; the combined target is `--n` times this value. |
| `--replace-rejected-polytopes` | off | Refill a polytope's accepted-geometry shortfall with spare favorable polytopes from the same `h11`. |
| `--max-polytope-replacements INT` | `10` | Maximum spare favorable polytopes fetched per `h11` when replacement mode is enabled. |
| `--outdir PATH` | `.` | Root directory for the generated database. |
| `--cores INT` | all available | Number of worker processes. Use `1` for debugging and deterministic logs. |
| `--seed INT` | `0` | Base seed. Worker and candidate seeds are derived from it and recorded in construction metadata. |
| output collision policy | no overwrite | Every run requires a fresh output root; existing geometry or manifest paths are terminal collisions. |
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
- tasks from different requested `h11` values share the same pool, so there is
  no completion barrier after each individual `h11`;
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
| `--polytope-manifest PATH` | unset | Load explicit, validated KS vertices from a local JSON manifest instead of fetching them from the remote endpoint. |

The script reconstructs each fetched polytope from its vertices in each worker.
This avoids passing a large lattice-point object between processes and is
especially important for the high-`h11` KS cases.

### FRST sampling

| Option | Default | Meaning |
| --- | --- | --- |
| `--sampling-scheme {fair,fast,ntfe_fast,gnn_ntfe}` | `fair` | Select fair secondary-fan MCMC, biased random heights, direct NTFE with sampled 2-face FRTs, or optional dualGNN-guided direct NTFE. |
| `--backend {cgal,qhull}` | `cgal` | CYTools backend for triangulation construction. |
| `--max-retries INT` | `50` | Maximum sampler retries for a new triangulation. |
| `--max-tip-attempts/--ntfe-sample-count INT` | `100` | Maximum FRST candidates; for `ntfe_fast`, native `Polytope.ntfe_frts(N=...)`. |
| `--n-walk INT` | CYTools default | Fair-mode secondary-fan walk steps per sample. |
| `--n-flip INT` | CYTools default | Fair-mode random flips per sample. |
| `--initial-walk-steps INT` | CYTools default | Fair-mode burn-in walk steps before recording samples. |
| `--fine-tune-steps INT` | `8` | Fair-mode wall-location refinement steps. |
| `--walk-step-size FLOAT` | `0.01` | Fair-mode secondary-fan step size. |
| `--max-steps-to-wall INT` | `25` | Fair-mode maximum steps toward a fine-triangulation wall. |
| `--fast-height-scale FLOAT` | `0.2` | Standard deviation of the Gaussian height perturbation in fast mode. |
| `--ntfe-face-sampler {fast}` | `fast` | Native `ntfe_frts` `triang_method`; schema 1.1 requires `fast`. |
| `--ntfe-max-face-points INT` | `17` | Native `ntfe_frts(max_npts=...)`. |
| `--ntfe-face-pool-size/--ntfe-face-triangulations INT` | `1000` | Native `ntfe_frts(N_face_triangs=...)`. |
| `--ntfe-as-generator` | `true` | Native `ntfe_frts(as_generator=True)`; disabling it is rejected. |

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

### Schema 1.1 compact EFT-reference mode

Stage 1 uses the approved raw-FRST plan
`50:500,100:500,200:300,491:100`. Stage 2's `--eft` mode requires the approved
`canonical_qcd`, validated intersecting-D7 toy policy, and strict QED bound
`Vol(D_QED) < 127.5`. The h11-specific sampler controls belong to stage 1:
the lower slices use the approved biased fast path, while h11=491 uses native
`ntfe_fast` with the native controls
`N=100`, `max_npts=17`, `N_face_triangs=1000`, `as_generator=True`, and the
`cgal` backend.

Within that fixed plan, the lower slices use the approved biased
`random_triangulations_fast` path with 10 proposals per polytope; only the
unique `h11=491` polytope uses native `ntfe_fast` with `N=100`. The CLI
`--sampling-scheme ntfe_fast` selects this schema-level plan rather than
silently applying NTFE to the lower slices.

The sampling unit is one accepted geometry plus one ordered `(QCD, QED)` pair
from that geometry's complete eligible pool. The pool is persisted before row
sampling and requires distinct divisors, recorded nonzero intersection, exact
QCD normalization to `Vol(D_QCD)=40.0`, every effective and prime-divisor
volume at least one, and strict QED volume below `127.5`. For each geometry,
rows are sampled uniformly without replacement using deterministic row seeds.
After exactly 1400 geometries, the quota allocator assigns 142 rows to 200
geometries and 143 rows to 1200 geometries, ordered by stable geometry-ID
hashes, for exactly 200000 rows. A smaller pool is a model-target shortfall,
not a reason to duplicate assignments.

Each row contains only scalar/index/reference metadata: geometry path/hash,
stable divisor labels and indices, pool rank/size, assignment hash, row seed,
normalization and volume scalars, and schema versions. It contains no dense
`Q`, `L`, `Kinv`, volume vector, or potential array. The single compressed
Parquet table path can be set with `--eft-output-path`; `pyarrow` is required.

This is an adapted finite fresh-favorable geometry reference with compact model
reuse. It is not an exact reproduction of arXiv:2309.13145, its undocumented
200000-model weighting, or a uniform/representative/complete KS sample. The
generator does not run GNN, PyTorch, axion-photon, cosmology, or inflation
analysis.

Stage 1 writes `frst_candidates/`, `frst_terminal_statuses.jsonl`,
`polytope_manifest.json`, and `run_manifest.json`. Stage 2 writes
`stage2_input_ledger.jsonl`, `stage2_terminal_statuses.jsonl`,
`stage2_topology_diagnostics.jsonl`,
`charge_factorized_manifest.json`, `polytope_manifest.json`,
`summary_by_h11_and_status.json`, `storage_estimate.json`, and the geometry
artifacts; `--eft` additionally writes `model_terminal_statuses.jsonl` and
`eft_models.parquet`. Terminal categories
remain separate for sampler, FRST, topology/cone, Kähler, normalization,
divisor, QED-pool, numerical, I/O, model, and storage failures. The manifest
also records the fixed stage boundary, raw identities, proposal/retry/duplicate
counts, and output-collision status.

Relevant options:

| Option | Default | Meaning |
| --- | --- | --- |
| `generate_stage1_raw_frsts.py --h11-plan TEXT` | `50:500,100:500,200:300,491:100` | Approved h11-to-raw-FRST allocation. |
| `generate_stage2_eft_reference.py --eft` | off | Build compact EFT-reference rows after stage-2 geometry acceptance. |
| `--eft-minimum-rows INT` | `100000` | Minimum accepted EFT-reference rows. |
| `--eft-maximum-rows INT` | `200000` | Exact upper row ceiling. |
| `--eft-output-path PATH` | `OUTDIR/eft_models.parquet` | Explicit table path inside the fresh output root. |
| `--materialize-dense-potential` | off | Explicit compatibility opt-in for dense geometry-level `Q/L`; never used by EFT rows. |

To balance the sample across polytopes, use `--frsts-per-polytope`. For
example, the following requests ten favorable polytopes and ten accepted FRSTs
from each, for up to 100 output geometries at that `h11`:

```bash
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 300 --h11_max 300 \
    --n 10 --frsts-per-polytope 10 \
    --replace-rejected-polytopes \
    --sampling-scheme fast \
    --cores 8 --seed 20260807 \
    --outdir /scratch/database/h11_300 \
    --verbose
```

In this mode the combined target is `--n` times `--frsts-per-polytope`. If
fewer than `--n` favorable polytopes are available, the script preserves that
combined target and redistributes it as evenly as possible over the polytopes
that were found. For example, if only four polytopes are available, the
example above requests 25 FRSTs from each. If the total is not divisible by
the number of available polytopes, their targets differ by at most one. A
polytope may still finish below its target if the FRST attempt budget is
exhausted or all candidates fail the physical acceptance filters. Existing
output slots are counted per polytope, so the same command can resume an
interrupted run.

If `--replace-rejected-polytopes` is enabled, the script fetches up to
`--max-polytope-replacements` spare favorable polytopes per `h11`. Whenever an
active polytope finishes below its accepted-geometry target, the next spare is
assigned the missing count. This preserves the combined default-mode target;
in `--frsts-per-polytope` mode it preserves the combined per-polytope target,
although the replacement may cause more than the originally requested number
of polytopes to be used. Replacement is bounded: if the spare pool is
exhausted, the remaining shortfall is reported. A worker exception is still
reported as an error rather than silently replaced.

### h11=491 laptop path

The repository includes the published vertices of the unique favorable
`(h11,h21)=(491,11)` KS polytope in
[`manifests/h11_491_11_ks.json`](./manifests/h11_491_11_ks.json). This avoids
depending on the legacy remote KS endpoint for a replay. First run a bounded
sampler/CY construction probe:

```bash
source activate cytools
XDG_CACHE_HOME=/tmp/cytools-cache \
python scripts/probe_h11_491_sampler.py \
    --sampler ntfe_fast --ntfe-face-sampler fast \
    --ntfe-face-pool-size 1000 --candidate-count 1 \
    --include-cy --include-topology --seed 20260813 \
    --report /tmp/h11_491_ntfe_probe.json
```

The probe emits an atomic JSON report with the fixed polytope identity,
sampler controls, FRST checks, triangulation hash, timing, and (when requested)
CY Hodge/smoothness checks. It is a performance/validity probe, not an HDF5
production run or fairness demonstration.

Schema 1.1 stores the direct charge matrix once per geometry and represents
pairwise terms with `pair_i`, `pair_j`, and the deterministic convention
`Q_pair[:,k] = Q_direct[:,pair_j[k]] - Q_direct[:,pair_i[k]]`. Coefficient and
geometry-reference `L` metadata are stored beside those indices. Dense `Q/L`
materialization is absent by default and is available only through the explicit
`--materialize-dense-potential` compatibility option; it is never repeated in
EFT rows.

For a single full geometry artifact, retain the manifest and sampler controls,
start with one worker and a fresh output root, and let the existing physical
filters account for every rejection:

```bash
source activate cytools
XDG_CACHE_HOME=/tmp/cytools-cache \
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 491 --h11_max 491 --n 1 --cores 1 \
    --polytope-manifest scripts/manifests/h11_491_11_ks.json \
    --sampling-scheme ntfe_fast --ntfe-face-sampler fast \
    --ntfe-max-face-points 17 --ntfe-face-pool-size 1000 \
    --max-tip-attempts 1 --seed 20260813 \
    --outdir /tmp/cyax-h11-491 --verbose
```

The requested schema 1.1 production path does not provide a learned/GNN
alternative. Do not substitute one for `ntfe_fast`.

### Recommended workflow for small geometries

Small-`h11` runs are useful for learning the workflow and checking the HDF5
schema, but they are not automatically representative of the larger KS
ensemble. Their secondary-fan state spaces can be unusually small, so the
fair sampler may report warnings, revisit the same triangulations, or fail to
make a useful random walk. These messages are often expected for a small
fixture rather than evidence of a broken CYTools installation.

Use the following progression:

1. Start with one geometry, one worker, the `fast` sampler, and a fresh output
   directory. This tests installation, CY construction, the acceptance filters,
   and HDF5 writing without making an ensemble claim.
2. Repeat with `fair` at a modest `h11` such as 8--10 when you want to test the
   production sampler. Keep the default fair-sampler controls initially.
3. Increase `--n`, then `--cores`, and only then widen the `h11` range. This
   makes it much easier to identify whether a slowdown comes from triangulation,
   Kähler optimization, physical rejection, or parallel I/O.
4. Save the command, seed, CYTools version, sampler, backend, and rejection
   counts with the run. `fast` results must remain labeled as biased coverage
   samples, not fair samples.

For example, a safe first check is:

```bash
source activate cytools
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 4 --h11_max 4 --n 1 \
    --sampling-scheme fast \
    --cores 1 --seed 17 \
    --max-tip-attempts 5 \
    --max-kaehler-attempts 3 \
    --outdir /tmp/cyaxiverse-small-smoke \
    --verbose 2>&1 | tee /tmp/cyaxiverse-small-smoke.log
```

For a fair-sampler diagnostic, change only `--sampling-scheme fast` to
`--sampling-scheme fair` and use a new output directory. If the run emits
`Couldn't find wall` or a random-walk error at very small `h11`, first try
`--backend qhull`, restore any changed fair-sampler controls, or move to a
slightly larger `h11`; do not silently relabel the resulting `fast` run as
fair.

The attempt counters have different meanings:

- `--max-retries` bounds retries internal to the triangulation sampler;
- `--max-tip-attempts` bounds FRST candidates tested for each polytope;
- `--max-kaehler-attempts` bounds angular Kähler points tested for each FRST;
- `--n` is the number of accepted geometries requested per `h11` in the default
  mode; with `--frsts-per-polytope`, it is the number of favorable polytopes.
  It is never the number of FRST candidates tried.

Messages such as `NoPhysicalKaehlerPoint`, `PrefactorCriterionNotMet`, or
`NoQcdDivisorVolume` mean that a candidate was rejected by a physical filter.
Increasing the attempt budget may find another candidate, but it does not make
the rejected geometry physically viable. Record these counts when comparing
runs.

The default output does not enumerate Kähler-cone rays. The generator only
needs the Kähler-cone hyperplanes for its current stretched-cone optimization
and validation, while CYTools' dual-ray enumeration can become very slow once
the cone dimension is moderately large (CYTools warns around dimensions above
12 and considers it likely impractical above roughly 18). Use
`--export-kaehler-rays` only for a downstream analysis that explicitly requires
cone generators; it is normally unnecessary for small-geometry smoke tests and
should not be enabled casually in high-`h11` scans. The effective-cone rays,
stored under `cytools/geometric/effective_cone`, are distinct and are required
for the direct instanton charge matrix `Q`.

Finally, use a new, empty output directory for every run. Schema 1.1 never
overwrites existing geometry files or accounting artifacts; an output collision
is a terminal error.

### Kahler-cone and acceptance controls

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-kaehler-attempts INT` | `100` | Number of stretched-cone/angular Kahler points tested per FRST, including the canonical tip. |
| `--min-divisor-volume FLOAT` | `1.0` | Minimum allowed volume for every exported effective-cone ray after rescaling. |
| `--min-prime-divisor-volume FLOAT` | `1.0` | Minimum allowed volume for every raw prime toric divisor. |
| `--qcd-volume-min FLOAT` | `25.0` | Lower edge of the QCD-visible prime-divisor volume window. |
| `--qcd-volume-max FLOAT` | `40.0` | Upper edge of the QCD-visible prime-divisor volume window. |
| `--moduli-policy {adaptive,canonical_qcd}` | `adaptive` | Use randomized angular Kähler points with a QCD window, or the canonical stretched-cone ray with radial QCD normalization. |
| `--allow-m-below-one` | off | Stage-2 `canonical_qcd` only: explicitly allow radial contraction (`m < 1`); disabled by default and recorded in metadata. |
| `--qcd-volume-target FLOAT` | `40.0` | Target prime-divisor volume for `canonical_qcd`. |
| `--qcd-divisor-index INT` | first eligible | Optional zero-based CYTools prime-toric-divisor index for `canonical_qcd`; without it, the first eligible candidate in deterministic CYTools divisor order is used. |
| `--orientifold-file PATH` | off | JSON file containing an explicit lattice involution and O3/O7 or O5/O9 metadata. |
| `--export-kaehler-rays` | off | Enumerate and store Kähler-cone rays; expensive at large `h11`. The legacy `--export-kahler-rays` spelling remains accepted. |
| `--max-m FLOAT` | `1_000_000` | Maximum radial prefactor for `canonical_qcd`, or upper bound for the adaptive potential-control search. |

The canonical stretched-cone tip is found by minimizing the Euclidean norm
subject to unit distance from the Kahler-cone hyperplanes. When MOSEK is
licensed and available, the script explicitly requests MOSEK for this solve.
For randomized angular projections, licensed MOSEK is tried first; other
qpsolvers backends are fallback candidates. The actual tip and projection
solvers are recorded per artifact.

At large `h11`, the fallback HiGHS solver may report a SciPy
`SparseConversionWarning` if a caller supplies dense quadratic-program
constraints. The generator constructs these constraints as CSC sparse matrices
so that warning should not normally appear in new runs. If it appears from a
different dependency path, it is a performance warning rather than a failed
geometry; check the selected solver and runtime before changing acceptance
parameters.

With `--moduli-policy adaptive`, candidate acceptance requires:

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

With `--moduli-policy canonical_qcd`, the same FRST and Kähler cone are used,
but the angular sampling loop is skipped. The eligible prime-divisor candidates
are inspected in deterministic CYTools order, or restricted to the explicit
`--qcd-divisor-index` when supplied. Candidates whose tip volume exceeds 40
are skipped so that the selected radial factor obeys `m >= 1`. The Stage-2
`--allow-m-below-one` opt-in disables that skip and permits `m < 1`; the final
Kähler form is then `J = m J_tip`, where `m = sqrt(target / tau_QCD)` for the
first remaining eligible prime divisor. This gives `tau ∝ m²`, `Kinv ∝ m⁴`,
and `Vol(CY) ∝ m³`; the exact assignment and scale are recorded in
`construction_metadata_json` and `cytools/geometric/standard_model`.

The generator imposes `m >= 1` in this mode by default: a tip already above the
target is not contracted and is skipped in deterministic candidate order. With
the Stage-2 `--allow-m-below-one` opt-in, that candidate is retained and may
produce `m < 1`. The final
prime-divisor, effective-cone, curve-volume, and cone-slack checks are then
applied. The adaptive potential-control search is not applied in this mode. This is a radial
normalization of an existing geometry, not a new FRST or a claim that a QCD
divisor has been constructed from a brane model.

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

### Canonical QCD-normalized geometry

To retain the canonical stretched-cone direction, sample the Standard Model
divisor triple, and normalize its QCD member to volume 40, use:

```bash
source activate cytools
python scripts/generate_geometric_data_multitriangulation.py \
    --h11s 15 --n 1 \
    --moduli-policy canonical_qcd \
    --qcd-volume-target 40 \
    --outdir /scratch/database/canonical-qcd \
    --verbose
```

Add `--qcd-divisor-index N` when the zero-based CYTools prime-toric-divisor
index is fixed by a separate model-building input. Without it, the generator
samples a pairwise-intersecting divisor triple and selects its QCD member
uniformly. A triple is interpreted as a triangle in the prime-divisor
intersection graph; this is the explicit, auditable interpretation of the
paper's under-specified phrase "triple of intersecting divisors".

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
calculations. A supplied orientifold that fails polytope, FRST, divisor-image,
or induced-action preservation is a terminal
`orientifold_invariance_failure`; the stage-1 FRST is not replaced.

For the paper-style visible-sector pilot, add
`--visible-sector-policy intersecting_d7`. This requires the orientifold file
to describe a validated O3/O7 involution. The generator records the computed
`h11_plus`/`h11_minus` split but does not reject a nonzero `h11_minus`; the
reference EFT separately records the paper-style all-C₄-axion assumption.
The generator then retains only QCD candidates
with an invariant intersecting prime divisor, chooses an invariant QED divisor
(the lowest-volume eligible one unless `--qed-divisor-index` is supplied), and
records the QCD/QED image map, intersection, divisor-basis charges, and QED
Euclidean-D3 instanton scale. The QED instanton is appended to the potential
when its divisor charge is not already one of the direct effective-cone terms.
This is the paper's stated toy visible-sector assumption, not a complete D7
model: it does not solve D7 tadpoles, engineer matter, or prove E3 zero-mode
conditions.

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
  prime_toric_divisors
  prime_divisor_charges
  tip, tip_prefactor
  CY_volume
  divisor_volumes, prime_divisor_volumes, curve_volumes
  Kinv
  kappa
  c2
  effective_cone
  mori_cone
  kahler_cone (only with --export-kaehler-rays; legacy HDF5 name)
  kahler_hyperplanes
  standard_model/ (only with --moduli-policy canonical_qcd)
    divisor_indices
    qcd_divisor_index
  orientifold/ (only with --orientifold-file)
    lattice_matrix
    h2_involution_matrix
    invariant_kahler_basis
    anti_invariant_h2_basis
    invariant_kahler_point (only when the kaehler subspace check is required)
    prime_divisor_image_indices
    prime_divisor_invariant_indices
  visible_sector/ (only with --visible-sector-policy intersecting_d7)
    qcd_divisor_index, qed_divisor_index
    qcd_image_index, qed_image_index
    qcd_divisor_volume, qed_divisor_volume
    qcd_charge, qed_charge, em_charge
    qed_instanton_index, qed_log10_lambda4
    qcd_qed_intersection, qcd_invariant, qed_invariant
  assignment_pool/ (with canonical_qcd + intersecting_d7; required by stage-2 --eft)
    pool_rank, qcd_divisor_index, qed_divisor_index
    qcd/qed stable labels, normalization/volume scalars, assignment_hash
    intersection_evidence_json
  tip_pre_normalization, divisor/prime/effective/curve volumes_pre_normalization
  CY_volume_pre_normalization, Kinv_pre_normalization
cytools/potential/
  factorized/
    Q_direct, pair_i, pair_j, direct/pair coefficient metadata
    L_direct, L_pairwise (geometry-reference sign/log-scale metadata)
  L, Q (only with --materialize-dense-potential)
construction_metadata/
  canonical_lattice_points
  face_restriction_dim2
```

Important conventions:

- `kappa` is sparse COO with columns `[i, j, k, value]` and zero-based indices.
- Vectors and matrices are in the CYTools divisor-basis convention recorded in
  `basis_convention`.
- `basis_matrix` is `h11 × n_points`; a prime-divisor charge is a selected
  lattice-point column, while `prime_divisor_charges` stores those charges as
  one row per prime divisor.
- `L_direct`/`L_pairwise` retain the CYAxiverse sign/mantissa and base-10
  exponent representation for the immutable geometry reference. `Q_direct` is
  stored once as `h11 × N_direct`; pairwise charges are reconstructed as
  `Q_direct[:, pair_j] - Q_direct[:, pair_i]`.
- Dense `Q`/`L` is absent by default and only appears with the explicit
  `--materialize-dense-potential` compatibility option. EFT rows never contain
  these arrays.
- The current Julia `read.potential` API still expects dense
  `cytools/potential/Q` and `L`. Until a separately reviewed reader adapter
  reconstructs the factorized representation in bounded chunks, Julia
  consumers that need that legacy API must use
  `--materialize-dense-potential`; the schema 1.1 EFT table remains compact.
- The potential charge basis uses unique effective-cone rays. Exact duplicate
  rows returned by CYTools are removed before the stretched-cone control,
  pairwise-term construction, and HDF5 write. The raw GLSM dataset remains
  available for provenance, while `construction_metadata_json` records the
  raw/canonical ray counts and the number removed.
- `tip`, `divisor_volumes`, `Kinv`, `curve_volumes`, and `CY_volume` refer to the
  same final rescaled Kahler point.
- `prime_divisor_volumes` follows the order of CYTools
  `prime_toric_divisors()` and is the vector used for the QCD filter. The
  corresponding `prime_toric_divisors` index array is exported explicitly;
  the recorded QCD divisor index is zero-based.
- `prime_divisor_charges` has one divisor-basis charge row per entry of
  `prime_toric_divisors`. Visible-sector divisor indices and image indices are
  zero-based in HDF5. `qed_instanton_index` is zero-based in the logical
  factorized potential stream; a dense `Q`/`L` column exists only with the
  explicit compatibility option.
- `assignment_pool/` is the complete eligible ordered QCD-QED pool, not one
  permanent QED choice. Its ranks and hashes are the source for deterministic
  EFT row sampling.
- `standard_model/divisor_indices` contains the three zero-based prime-divisor
  indices selected by `canonical_qcd`; all three pairs are edges of the
  triangulated two-face intersection graph. `standard_model/qcd_divisor_index`
  identifies the member whose final volume is normalized to the target.
- `kahler_hyperplanes` is always exported and is sufficient for the generator's
  stretched-cone optimization and physical validation. Use
  `--export-kaehler-rays` only when a downstream sampler explicitly requires a
  generator representation of the Kähler cone.
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
- `ntfe_fast` saves redundant geometry work by changing the sampling unit to
  two-face-inequivalent FRSTs. A finite per-face pool and its FRT proposal are
  selection effects, so it is a labelled coverage sampler, not a fair
  population sample.
- `ntfe_fast` is an adapted finite native-NTFE proposal. Its face pool, direct
  extension failures, duplicate classes, and proposal budget belong in the run
  record; it must not be labelled uniform, representative, or complete.
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
