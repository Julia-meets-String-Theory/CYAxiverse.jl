# Handoff: inflation reproduction and trajectory validation

## Mission

Continue the inflation-reproduction work from the branch
`agents/inflation-reproduction-instructions`. The immediate scientific task is
to complete the corrected canonical/mass-basis trajectory comparison and derive
the remaining N=5 benchmark quantities. A separate follow-up question is
whether the implemented functions are suitable for scans over O(10^5)
geometries; do not claim scalability until it has been measured.

## Critical starting condition

This handoff was written from checkout `vmm`, whose current HEAD was:

```text
770b09b feat: update Kähler-cone ray export option and schema version in geometry generator
```

The relevant work is on the local branch:

```text
agents/inflation-reproduction-instructions
```

At handoff time that branch pointed to:

```text
53f19cc fix: import linear algebra in author trajectory module
```

It is based on commit `2c297b5 feat: add inflation reproduction benchmarks`.
Switch to or use the worktree for that branch before inspecting or editing the
inflation implementation. Do not merge or reset branches blindly; preserve any
unrelated worktree changes. The current `vmm` checkout had one unrelated
untracked Python bytecode file: `scripts/__pycache__/...pyc`.

## What the branch contains

The branch adds or revises:

- `scripts/inflation_reproduction.jl`: deterministic fixture generation,
  comparison CSV generation, and basis audit printing.
- `src/paper_benchmarks.jl`: N=5 and N=8/poly-102 benchmark models, geometry,
  coordinate maps, hilltop initial conditions, trajectory probes, and the
  author-independent trajectory module.
- `validation/inflation_fixtures.jls`: serialized reproducibility fixture.
- `validation/inflation_comparison.csv`: benchmark comparison table.
- related tests and documentation changes from the benchmark commit.

The package convention is `Q :: (axions, instantons)`, with instanton columns.
The trajectory contract recorded by the fixture is:

- `theta`: raw angle coordinates, radians;
- `chi`: canonical coordinates in reduced Planck units;
- `tangent`: physical tangent in raw-angle coordinates;
- `V = sum(A_i * (1 - cos(Q_i * theta + phase_i)))`;
- `G(k) = G(k_c) * (k_c/k)^2`;
- mass eigenvectors are computed once at the hilltop and are not recomputed
  along the trajectory.

The branch’s final one-line import fix is important: the author trajectory
module uses `LinearAlgebra` directly.

## Recorded benchmark state

The comparison CSV on the branch records:

```text
N5, delta=1e-7       local=27349.0       author=27349.0
N5, delta=6.65e-5    local=60.0          author=60.0
N8, delta=1e-7       local=464970.4928   author=463115.0  independent=464213.5708
N8, delta=1.532e-3   local=63.8574       author=60.0      independent=59.69064
```

Interpret these as comparison diagnostics, not proof that the N=8 local
normal-form trajectory reproduces the author implementation. The N=5 values
are a local reduced model and have no independent author trajectory in the
repository. The former N=8 discrepancy has been narrowed down: the
independent SciPy value `59.690642055250756` uses Float64 BDF with
`max_step=100` across a stiff initial transient. Tighter step caps converge to
the Julia BigFloat result near `60.003`, so that value should not be treated as
a high-precision physical reference. The remaining meaningful difference is
between the local normal-form estimate (`63.8574` at the 60-efold detuning)
and the full nonlinear physical flow (`60.0034`), which is a model-scope
difference.

The author-independent reference is described as a miniforge Python/SciPy
physical-flow calculation with a final-finite-exit event policy. Verify its
coordinate convention, initial tangent, end-event definition, and precision
before changing any model code.

## First commands for the next agent

Run these from the branch worktree:

```sh
git branch --show-current
git status --short --branch
git log --oneline --decorate -8
git diff --check
julia --project=. scripts/inflation_reproduction.jl
julia --project=. test/runtests.jl
```

If the branch is not checked out, stop and move to the correct branch/worktree
before editing. Inspect the generated fixture and CSV after the script runs;
do not silently overwrite published/reference data if the output changes.

## Recommended scientific continuation

1. Read the complete trajectory-related sections of `src/paper_benchmarks.jl`
   and `scripts/inflation_reproduction.jl`.
2. Re-run the basis audit and save stdout/stderr outside the repository or in a
   clearly named validation note. Compare the package-current direction,
   canonical direction, and fixed hilltop mass-eigenbasis direction.
3. Validate the raw-to-canonical maps algebraically: metric normalization,
   gradient transformation, generalized eigenproblem, and tangent projection.
   Use finite-difference checks on the potential and kinetic metric.
4. Reproduce the N=8 author-independent trajectory with exactly the recorded
   tolerances and event policy before tuning tolerances. Record any change in
   `validation/` rather than replacing the reference numbers.
5. Complete the N=5 benchmark quantities using the same explicit contracts;
   distinguish the one-dimensional reduced model from the full eight-term
   potential and state which one each result uses.
6. Add focused regression tests for coordinate orientation, fixed-vs-moving
   mass basis, initial tangent normalization, and event termination.

Ask the expert before changing scientific definitions, including the end
condition, slow-roll thresholds, canonical normalization, mass-basis choice,
potential coefficient model, or interpretation of the author numbers.

## Scalability question: O(10^5) geometries

No scalability claim has been established. The current reproduction code is a
benchmark/validation path, not yet a production scan engine. Before attempting
10^5 geometries, profile representative dimensions and separate:

- geometry/HDF5 loading and allocation;
- `LQtilde` selection and matrix factorization;
- trajectory integration and event detection;
- arbitrary/high precision work;
- serialization and CSV/HDF5 output.

Use bounded pilot runs (1, 10, then a statistically representative sample)
and record wall time, allocations, peak memory, failure rate, and output size.
Avoid one giant in-memory result array; stream summaries and checkpoint each
geometry. Reuse immutable geometry-independent work where valid, but do not
reuse mutable solver state across geometries without proving isolation. Keep
worker count × BLAS threads within physical cores, and use one writer per
output shard followed by a deterministic merge. A scan of 10^5 geometries
requires explicit timeout, retry, resume, and provenance behavior.

## Files and outputs to preserve

Preserve the branch’s fixture schema and comparison CSV unless a deliberate,
reviewed schema change is made. Keep generated bulk scan outputs out of git.
For every discrepancy, save the geometry/model parameters, `Q`, `L`, `K`,
coordinate convention, delta, tolerances, solver status, and event diagnostics.

## Handoff completion criteria

The next agent should leave:

- a reproducible explanation of the N=8 discrepancy or a validated fix;
- completed N=5 benchmark results with model scope clearly labeled;
- regression tests for the coordinate and trajectory contracts;
- a measured, bounded scalability report before any O(10^5) run;
- `git diff --check`, focused tests, and the full package test result;
- a final commit hash and concise list of unresolved scientific caveats.
