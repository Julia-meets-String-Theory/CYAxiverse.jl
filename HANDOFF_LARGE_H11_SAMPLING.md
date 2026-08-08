# Handoff: large-\(h^{1,1}\) triangulation sampling investigation

## Mission

Improve the large-\(h^{1,1}\) geometry-generation path so that it is both
computationally usable and scientifically defensible for the CYAxiverse
projects.

The current situation is not solved by choosing between the two existing
samplers:

- `fair` is too slow at the large-\(h^{1,1}\) values of interest;
- `fast` is much quicker, but its physical-acceptance failure rate is too high
  for the current production target, and its ensemble is explicitly biased;
- the slow stage may be triangulation, CYTools topology/cone construction,
  Kähler-space QPs, or downstream physical filters. These must be separated
  before changing the algorithm.

The next agent should produce a measured diagnosis, implement bounded
experiments, and recommend a production sampler only after comparing speed,
failure modes, coverage, and sampling meaning.

This is an investigation handoff, not authorization to launch a large scan,
change the scientific ensemble definition, weaken physical filters, or replace
existing output data.

## Starting state

The main implementation is:

```text
scripts/generate_geometric_data_multitriangulation.py
scripts/README_generate_geometric_data_multitriangulation.md
```

At handoff time the working tree already contains related generator and README
changes. Begin every session with:

```bash
git status --short --branch
git diff --check
git log --oneline -8
```

Preserve unrelated changes. Do not use `git reset --hard`, broad deletion, or
an unreviewed HDF5 overwrite.

The current generator has these important properties:

1. `triangulation_candidates()` uses
   `Polytope.random_triangulations_fair()` for `fair` and, for `fast`,
   yields one deterministic `poly.triangulate()` candidate followed by
   `random_triangulations_fast()` candidates with Gaussian height scale
   `--fast-height-scale`.
2. Both paths request fine, regular, star triangulations, but they do not have
   the same statistical meaning. `fast` is recorded as biased and must not be
   relabeled as fair merely because it has a high acceptance rate.
3. `process_polytope()` bounds the number of candidates with
   `--max-tip-attempts`. Each candidate is passed through the full geometry
   construction and physical acceptance pipeline.
4. `generate_and_save_geometry()` computes topology and cone data, finds a
   stretched Kähler point, searches randomized angular Kähler directions with
   QPs, applies the prefactor criterion, applies prime-divisor/QCD-volume
   filters, constructs potential data, and writes an HDF5 artifact.
5. Kähler-cone ray enumeration is already optional via
   `--export-kahler-rays`; do not reintroduce it into the default large-
   \(h^{1,1}\) path without a specific downstream requirement.
6. `--replace-rejected-polytopes` can refill accepted-geometry shortfalls with
   spare favorable polytopes, but it does not solve a poor FRST sampler or a
   high per-candidate physical rejection rate.
7. A shared process pool permits tasks from different `h11` values to run
   concurrently, but a single pathological polytope still consumes one worker
   until its bounded candidate work completes.

## Scientific distinctions

Do not collapse the following into one “success rate.” Measure them separately:

1. **Polytope selection.** Which favorable polytopes are fetched for a given
   `h11`, and how representative is that selection?
2. **FRST generation.** How many sampler proposals, triangulation retries,
   duplicate triangulations, non-fine proposals, and valid FRSTs are produced?
3. **FRST ensemble.** Is the target uniform over FRSTs, uniform over
   two-face-inequivalent/NTFE classes, or a deliberately biased coverage
   distribution?
4. **Geometry construction.** How much time is spent constructing the CY,
   intersection numbers, Mori/effective/Kähler cones, and other metadata?
5. **Physical acceptance.** Which candidates fail `NoPhysicalKaehlerPoint`,
   `PrefactorCriterionNotMet`, `NoQcdDivisorVolume`, or validation?
6. **Output policy.** How many accepted artifacts are new, skipped from a
   previous run, or rejected after expensive work?

The project may ultimately decide that a uniform NTFE sample is more useful
than a uniform FRST sample, or that a calibrated biased sample is sufficient
for a particular training scan. That is a scientific decision and must be
written down rather than inferred from runtime.

## Literature to read and use

The next agent must read the primary papers and relevant CYTools source/docs,
then map their definitions and algorithms to the current implementation.

### Foundations and calibration

- [Kreuzer and Skarke, *Complete classification of reflexive polyhedra in four dimensions*, hep-th/0002240](https://arxiv.org/abs/hep-th/0002240).
  Use this for the KS database, reflexive four-polytopes, and toric CY
  hypersurface context.
- [Demirtas, McAllister, and Rios-Tascon, *Bounding the Kreuzer-Skarke Landscape*, arXiv:2008.01730](https://arxiv.org/abs/2008.01730).
  Read the secondary-polytope, two-face, and representative-ensemble sections.
  Determine precisely what “fair” means, what mixing evidence is provided, and
  how `initial_walk_steps`, `n_walk`, `n_flip`, wall steps, and fine-tuning
  map to the implementation.
- [Gendler et al., *Counting Calabi-Yau Threefolds*, arXiv:2310.06820](https://arxiv.org/abs/2310.06820).
  Use the exhaustively understood low-\(h^{1,1}\) cases for duplicate,
  two-face-equivalence, and coverage calibration.

### CYTools implementation

- [Demirtas, Rios-Tascon, and McAllister, *CYTools: A Software Package for Analyzing Calabi-Yau Manifolds*, arXiv:2211.03823](https://arxiv.org/abs/2211.03823).
  Read the triangulation-sampling and performance sections.
- [CYTools polytope documentation](https://cy.tools/docs/documentation/polytope/),
  especially `random_triangulations_fair`,
  `random_triangulations_fast`, `random_triangulations_gnn`, and
  `ntfe_frts`.
- [CYTools triangulation documentation](https://cy.tools/docs/documentation/triangulation/),
  especially `random_flips`, regularity checks, and backend behavior.
- [CYTools source repository](https://github.com/LiamMcAllisterGroup/cytools).
  The installed package and source revision are authoritative; online docs may
  describe APIs newer than the local environment.

### Direct and learned alternatives

- [MacFadden, *Efficient Algorithm for Generating Homotopy Inequivalent Calabi-Yaus*, arXiv:2309.10855](https://arxiv.org/abs/2309.10855).
  Study direct NTFE FRST construction from two-face triangulations and its
  regularity/extension tests. Determine whether this changes the target from
  FRSTs to CY topology classes.
- [MacFadden, Schachner, and Sheridan, *The DNA of Calabi-Yau Hypersurfaces*, arXiv:2405.08871](https://arxiv.org/abs/2405.08871).
  Use this for the two-face/“DNA” representation and learned or genetic
  alternatives. Optimization-oriented sampling is biased unless a sampling
  measure and validation procedure are supplied.
- [Berglund et al., *Generating Triangulations and Fibrations with Reinforcement Learning*, arXiv:2405.21017](https://arxiv.org/abs/2405.21017).
  Assess RL as a targeted-generation route, not automatically as a fair
  sampler, and record the demonstrated `h11` range.
- [MacFadden, *Sampling Triangulations and Calabi-Yau Threefolds with Autoregressive GNNs*, arXiv:2605.27770](https://arxiv.org/abs/2605.27770).
  Current CYTools docs describe the optional
  `random_triangulations_gnn`/`ntfe_frts` path as closer to uniform than
  `random_triangulations_fast` and much faster than
  `random_triangulations_fair` at large `h11`. Verify the local package
  version, model/checkpoint, licensing, reproducibility, and whether its output
  is an NTFE FRST sample before considering integration.

### Physical filters

- [*Axion minima in string theory*, arXiv:2309.01831](https://arxiv.org/abs/2309.01831).
  Recheck the stretched-cone/prefactor criterion before tuning Kähler search.
- [*Fuzzy Axions and Associated Relics*, arXiv:2412.12012](https://arxiv.org/abs/2412.12012).
  Recheck the prime-divisor lower bound and QCD-volume window. Do not weaken
  these filters to improve a benchmark rate.

The literature review should end in a mapping table:

```text
literature definition -> current function/option -> diagnostic needed
```

## First actions: safe baseline

Use a fresh temporary output directory and do not run a broad h11 range.
Record:

```bash
source activate cytools
python --version
python -c 'import cytools; print(cytools.version)'
git rev-parse HEAD
git status --short --branch
python - <<'PY'
import numpy as np
np.show_config()
PY
```

Also record `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, MOSEK-license presence (not its contents), backend,
solver availability, and machine/OS information.

Run one candidate at a time with verbose logging:

```bash
python scripts/generate_geometric_data_multitriangulation.py \
    --h11_min 300 --h11_max 300 \
    --n 1 --cores 1 \
    --sampling-scheme fair \
    --max-tip-attempts 1 \
    --outdir /tmp/cyax-large-h11/fair-one \
    --seed 20260807 --verbose \
    2>&1 | tee /tmp/cyax-large-h11/fair-one.log
```

Repeat with `fast`, then with a small bounded candidate count. This is a
timing probe, not evidence that one fair sample has meaningful mixing. If
`h11=300` is unavailable, use a nearby representative and record the actual
polytope ID, point count, facet statistics, and h11.

Build a reproducible polytope manifest containing h11, lattice, favorability,
database query parameters, canonical polytope ID, vertex/lattice-point count,
facet and largest-2-face statistics, seeds, sampler controls, backend, and
CYTools version. Do not compare different polytopes and call it a sampler
comparison.

## Required instrumentation

Add diagnostics behind an explicit flag or profiling harness so normal HDF5
output is not inflated. Prefer JSONL/CSV over human-readable logs alone.

### Sampler metrics

Instrument `triangulation_candidates()` and the worker boundary to record:

- polytope construction and point-label time;
- deterministic-fast candidate time;
- proposal, retry, duplicate, and yielded-candidate counts;
- stable full-triangulation hash and, where possible, two-face/DNA hash;
- non-fine, non-regular, and non-star rejection counts;
- fair walk/flips, wall-finding failures, and time per yielded FRST;
- CPU time and peak RSS if available.

Distinguish duplicate full FRSTs from duplicate two-face classes.

### Geometry/physics metrics

Add stage timers around the existing messages in
`generate_and_save_geometry()`:

1. FRST validation and `get_cy()` construction;
2. Hodge/intersection/divisor-basis extraction;
3. Mori and Kähler-hyperplane construction;
4. stretched-cone tip solve and solver/backend;
5. effective-cone ray extraction;
6. randomized angular-Kähler QP count, dimensions, nonzeros, solver,
   feasible/infeasible/error counts, and time per QP;
7. divisor/effective-ray checks;
8. prefactor search;
9. QCD-window test;
10. potential construction and HDF5 write.

Every candidate must end in one explicit category:

```text
sampler_retry_exhausted
invalid_frst
topology_or_cone_error
kaehler_tip_failure
no_physical_kaehler_point
prefactor_criterion_not_met
no_qcd_divisor_volume
accepted
```

Do not count a worker timeout, CYTools exception, and physical rejection as
the same failure. Preserve exception class and a bounded message.

The previous investigation found that large-\(h^{1,1}\) runs can spend
substantial time in HiGHS QPs while searching randomized angular Kähler
directions. Recheck this with timings rather than assuming fair sampling is
always the bottleneck. The projection QPs should continue to use sparse CSC
matrices.

## Benchmark matrix

Start with one fixed polytope per feasible representative dimension, then
expand to 3--5 polytopes spanning point/facet complexity:

| Factor | Initial values |
| --- | --- |
| h11 | 100, 200, 300, 400, 491 when available |
| sampler | fair, fast; later NTFE/dualGNN if compatible |
| backend | installed default, then cgal/qhull where supported |
| candidates | 1 for profiling, then 5 or 10 for rates |
| workers | 1 first, then bounded 2, 4, 8 |
| fair controls | defaults, then one-factor-at-a-time |
| fast height scale | 0.05, 0.1, 0.2, 0.5, 1.0 |

Report wall/CPU time, RSS, stage percentages, proposals, unique FRSTs,
unique two-face classes, duplicates, terminal-failure counts, solver/warnings,
and two-seed reproducibility. Do not optimize only accepted geometries/hour;
a sampler can look fast by being biased or physically favorable.

## Investigation branches

### A. Prove where the time goes

Compare sampler-only time, CY construction/topology/cones, Kähler tip and
angular QPs, physical filtering, and HDF5 writing. If fair dominates, inspect
walk initialization, wall finding, regularity checks, and repeated work. If
downstream work dominates, sampler choice alone will not solve production
runtime.

### B. Understand fast-mode failures

At fixed polytope and seed family, vary only `--fast-height-scale`. Measure
fine/regular/star yield, retry exhaustion, duplicates, cone/QP errors, and each
physical rejection category. The CYTools docs state that larger height scale
broadens the distribution but increases non-fine proposals with
`only_fine=True`; verify that tradeoff locally. Do not assume a wider height
distribution approaches uniformity over FRSTs.

### C. Tune fair mode only with fairness diagnostics

Vary `initial_walk_steps`, `n_walk`, `n_flip`, `walk_step_size`,
`max_steps_to_wall`, and `fine_tune_steps` one at a time. Compare runtime
with duplicate/two-face coverage, observables from arXiv:2008.01730,
autocorrelation or state distances, seed sensitivity, and a long/default
reference on tractable polytopes. A faster setting that changes the ensemble
must be labeled as a new biased sampler.

### D. Test hybrid walks

After reading the CYTools implementation, test whether one initial
triangulation can seed `random_flips` or secondary-fan segments without
reconstructing state unnecessarily. Verify every result is fine, regular, and
star; record starting triangulation/hash and all walk controls.

### E. Test direct NTFE/dualGNN generation

Determine whether the project needs distinct FRSTs or distinct CY topological
classes; whether NTFE output supplies all downstream Julia data; whether the
local CYTools exposes `ntfe_frts`/ `random_triangulations_gnn`; and what
distribution and validation the paper reports. Test first on low-h11 fixtures
with exhaustive or known two-face data. Measure direct-extension failures
separately from physical failures.

### F. Separate construction from physical selection

Create a bounded diagnostic/replay path that can stop after a valid FRST and
topology object, or replay a saved candidate through physical acceptance. The
replay must preserve the exact triangulation, basis convention, solver settings,
CYTools version, and random seeds. Never infer a sampler problem from a
physical filter that rejects all members of the tested ensemble.

Do not alter the prefactor, prime-divisor, QCD, or Kähler thresholds as a
performance experiment. Scientific changes require review.

## Architecture and caching questions

Investigate, with profiling evidence, whether it is safe to:

- cache polytope points and invariant face metadata while passing compact,
  immutable worker inputs;
- cache topology data that is genuinely invariant across triangulations;
- reuse fair-walk state within one worker without biasing the target;
- batch sampler proposals before expensive CY construction with a memory bound;
- separate sampler and geometry-validation processes;
- cap BLAS/OpenMP threads so workers do not oversubscribe physical cores;
- write diagnostics incrementally outside the main HDF5 artifacts.

Do not cache mutable solver state or reuse candidate-dependent cones across
triangulations without an invariant proof and regression test.

## Required deliverables

The next agent should leave:

1. a literature note mapping definitions and claims to the local CYTools version;
2. a reproducible polytope manifest and benchmark harness;
3. stage timing and terminal-failure instrumentation behind an explicit option;
4. benchmark tables for representative h11/polytope complexity;
5. low-h11 validation against exhaustive or known two-face-equivalence data;
6. fair/fast/direct-NTFE or dualGNN comparisons where compatible;
7. a recommendation stating target ensemble, speed, failure rate, coverage
   evidence, and unresolved bias;
8. regression tests for determinism, FRST validity, failure accounting, and
   provenance;
9. updated README guidance only after behavior is stable.

The recommendation must explicitly say whether to keep fair, introduce a
calibrated fair/NTFE sampler, use fast only for labeled biased scans, or
maintain multiple modes.

## Stop conditions and definition of done

Ask the expert before changing the default sampler, calling a learned/GA/RL/
NTFE sample “uniform FRST,” weakening physical filters, launching a broad scan,
enabling new dependencies/checkpoints, changing the HDF5 schema, overwriting
geometry data, or claiming finite-sample representativeness/completeness.

The investigation is complete when it has measured:

- the time fractions from sampling, topology/cones, Kähler QPs, filters, and I/O;
- why fast fails and how often each failure occurs;
- the scientifically required ensemble;
- whether any faster method preserves or defensibly changes that ensemble;
- scaling with h11, point/face complexity, candidates, and workers;
- a reproducible student-facing benchmark command and failure report.

Before handing work back:

```bash
git diff --check
python -m py_compile scripts/generate_geometric_data_multitriangulation.py
```

Keep generated bytecode, logs, benchmark outputs, and model files out of git
unless intentionally curated.
