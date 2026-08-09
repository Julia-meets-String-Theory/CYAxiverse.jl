# Vacua Stage 4/5 validation boundary

This note defines the compact publication-validation outputs produced by
`scripts/validate_vacua_stage4_5.jl`. The validator is read-only with respect
to geometry HDF5 data: Stage 5 calls `compute_axion_data(...; save=false)` and
records the resulting search status without creating or replacing a vacua
group.

Counts retain the existing meanings. `Nvac`/`N_min` are lower-bound search
counts for finite-start methods. A result is `verified` only for the existing
determinant-certified branch, and `verified_selected_branch_set` is reserved
for the leading branch enumeration. Neither label implies population
coverage. Digitized 2023 values are a separate comparison population and are
not treated as exact geometry-level targets.

Stage 4 covers:

- N=5 critical/minima anchors and N=8 counts `(5, 1)` below/above the
  catastrophe;
- the six initial inflation-screen geometries. The archived `(11,1,1)` row is
  retained as a provenance comparison only: the original geometry label was
  replaced by a physical geometry, so its branch-count difference is expected
  and the current physical geometry's enumeration is authoritative;
- reduced JLM-method `h11=4:11` summaries versus the digitized aggregate; and
- deterministic selected-column ordering at `h11=10` and `h11=20`.

For each nonzero aggregate discrepancy, the validator writes a compact text
reproducer containing geometry identity, Q/L matrices in the established
`Q=(h11, instantons)`, `L=(2, instantons)` orientation, threshold, method,
solver status, and a separately labelled residual diagnostic.

Stage 5 is bounded to available representative geometries near `h11=150,
180,200` and BLAS threads `1,2,4,8,16`. It records wall time, Julia
allocation bytes, process high-water RSS, method, verification/solver status,
mode count, and the read-only one-writer state. Existing spectrum logs are not
used as vacua results. The validator reports `h11=491` as unavailable unless a
geometry file is actually present; no missing potential is inferred or tested.

The validated production path is now vacua-only: it loads `Q`, `L`, and `K`
and does not construct the full spectrum when only vacuum counts are needed.
On the warm real-data pilot, median wall times were 7.67 s, 15.60 s, and
23.94 s for the available `(150,1,1)`, `(180,1,1)`, and `(200,1,1)` geometries,
respectively. The corresponding allocation totals were 9.16 GB, 18.49 GB,
and 28.28 GB. These are bounded resource measurements, not extrapolations to
unavailable geometries or to the eventual full population.

The resume/no-corruption pilot is recorded in
`vacua_stage4_5_results/resume_pilot.txt`. It used a copy of real `(150,1,1)`
data outside the repository: the first run completed with `Nvac=1`, an
identical second run was skipped as a matching completed result, and a
threshold-mismatched rerun was blocked without `--force`. Read-back verified
the completed exact-determinant branch result and byte-level-equivalent
representations of all pre-existing physical-spectrum datasets. The original
source-data file was not written.

Here “JLM” is the initials label for the collaborator who jointly developed
the method; it is not being expanded as an algorithm name.

The digitized aggregate comparison is deliberately labelled as a different
estimand: it is not a pass/fail equality target for finite-search lower
bounds. Any population-level interpretation remains out of scope until the
population, selection route, and completeness convention are harmonized.

Generated manifests and full resource CSVs belong outside Git. Commit only
this boundary note and curated compact comparison/reproducer outputs after
scientific-owner review. The compact resume pilot is an explicit final-state
artifact; its copied HDF5 file, manifests, and full Stage 5 CSVs remain
outside Git.
