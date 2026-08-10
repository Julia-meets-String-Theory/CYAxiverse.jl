# SCI-01: physical-scale derivation and release decision

Date: 2026-08-09
Status: proposed scientific decision; owner approval is required before
integration
Program: KS axiverse inflation only

## Re-evaluation using the draft author's code

This addendum supersedes the earlier statement that a generic physical path
was merely a future possibility. The draft author's Python/CYTools code does
provide an explicit generic effective-theory path. It is materially richer
than the legacy homotopy and is now the pilot's default mode:

```text
tau(k)  = k * tau(1)
Kinv(k) = k^2 * Kinv(1)
Q(k)    = Q(1)
```

It then recomputes the leading and pair/cross instanton coefficients from
`tau(k)`, `Kinv(k)`, `CY_vol`, and the charges. The zero-phase convention
is explicit in the scan. This establishes that the generic deformation is not
mathematically underdetermined in the author's model.

It does **not** authorize relabelling the historical homotopy outputs as
physical. The
author implementation keeps `CY_vol` fixed in `geometric_quantities`, while
the source calls `k` a uniform divisor-volume/overall-volume path and the
potential contains an explicit `V_CY^-2` factor. Under the homogeneous
overall Kähler scaling implied by `tau -> k*tau`, the expected geometric
volume scaling is `V_CY -> k^(3/2)*V_CY`, unless the author has a separately
documented Einstein-frame convention that cancels or replaces it. The archive
does not document that convention or validate the resulting absolute
normalization.

The revised decision is therefore:

1. A **candidate physical-model path exists** in the author code and is
   reproduced coefficient-wise by the pilot.
2. The generic pilot now defaults to `scale_status=physical` with
   `volume_normalization=full`.
3. The author's fixed-`CY_vol` convention remains available explicitly with
   `volume_normalization=fixed`; `homotopy_only` remains available for
   reproducing historical diagnostics.
4. No broad physical inflation claim is made until Kähler-cone/EFT domain
   checks and a bounded physical scan are separately approved.

## Decision

The pilot now contains a validated coefficient/path implementation and
defaults to `scale_status=physical, volume_normalization=full`. Physical-mode
rows must not be treated as an inflation-candidate or trajectory sample
without the separate domain and bounded-scan gate.

The uncommitted pilot is therefore **not approved for integration as a
physical volume scan**. It may be preserved and integrated only as a bounded
mathematical-homotopy diagnostic, with its existing status labels, resource
limits, and claim boundary unchanged.

## Source-supported physical path

The supplied *Catastrophic Inflation in the Axiverse* source defines the
intended control path by uniformly scaling the divisor volumes,

```text
tau_i(k) = k * tau_i(k=1).
```

In the source conventions, `tau_i` are divisor volumes, `K` is the axion
kinetic metric, and the potential coefficients depend on divisor volumes and
the Calabi--Yau volume. The source also states that the saxions are held fixed
while the axion effective theory is studied and that self-consistent Kähler
modulus stabilization at the tuned values is not established. Thus the path
is a source-level effective-theory control path, not by itself a proof that
the tuned point is dynamically realized.

For the potential construction used by the geometry-generation scripts, a
generic physical implementation would have to reconstruct and apply, at
minimum, the following joint transformation:

| quantity | required transformation under the source path | present pilot |
| --- | --- | --- |
| divisor volumes `tau` | `tau -> k * tau` | physical mode |
| Calabi--Yau volume `V` | fixed or `V -> k^(3/2)*V` | `fixed|full` option |
| inverse kinetic metric | `Kinv -> k^2 * Kinv`, equivalently `K -> k^(-2) * K` | physical mode |
| leading amplitudes | reconstruct the prefactor and `exp(-2*pi*q*tau)` dependence | physical mode |
| pair/cross amplitudes | reconstruct both `q_i^T*Kinv*q_j` and `(q_i+q_j)*tau` terms | physical mode |
| charges `Q` | unchanged in the fixed divisor basis | unchanged |
| phases | retain a documented fixed phase convention or transform it | not persisted by the generic potential API |
| normalization | retain the selected volume convention | recorded in output |
| domain | check positive effective-divisor/curve volumes, positive-definite `K`, stretched-cone and EFT-control conditions | not checked by continuation |

The scaling entries above are implemented in `pilot_scaled_inputs`. The
physical mode reconstructs the active author coefficient map from the stored
geometry metadata and records any discrepancy with the persisted HDF5 `L`
values; it does not by itself establish Kähler-cone/EFT validity or authorize
a production physical scan.

## Provenance

The source statements used here are equations (19), (21), (22), and (24) of
the supplied PDF `/Users/vmehta/Downloads/KS_axiverse_inflation (3).pdf`.
The generic coefficient construction is in
`scripts/generate_geometric_data.py` and
`scripts/generate_geometric_data_multitriangulation.py`; the latter stores
`divisor_volumes`, `CY_volume`, `Kinv`, and `effective_cone` in the
geometry HDF5 files. The legacy path reads only `read.oriented_potential`
(`Q`, `L`, and `K`) and applies `pilot_homotopy_scale`; physical mode
additionally reads `read.geometry` and applies `pilot_scaled_inputs`. The
benchmark-only physical helpers in `src/paper_benchmarks/` remain fixed-example
evidence, not the generic map.

The re-evaluation additionally inspected
`/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CN_Axiverse_code/ks_axiverse_python_collaborator/src/cytools_catastrophe_scan.py`
(`geometric_quantities`, `solve_at_k`) and
`Camcode_full_2.py` (`dim_reductor`, `generate_charges_2`), plus
`poly102_efolds_vs_k/poly102_core.wl` and
`poly102_efolds_vs_k/poly102_settings.wl` for the author's trajectory and
normalization convention.

## Why the current pilot is a homotopy

The legacy homotopy applies exactly

```text
L[2, :] -> k * L[2, :]
```

to the base-10 logarithms of already-composed amplitudes while leaving `K`
and all other inputs unchanged. This is not equivalent to scaling divisor
volumes. In physical mode, the pilot reconstructs the leading and pair
coefficients from `tau`, `Kinv`, and the selected volume law, and uses
`K -> K/k^2` for the generalized Hessian.

The source and generator code motivated the geometry-level implementation.
The physical mode records the selected normalization and provenance in its
CSV rows, while historical homotopy outputs remain clearly labelled. The
benchmark-specific N=5/N=8 scale functions are still not generalized merely
because they reproduce the fixed examples.

The author archive provides the executable reference: its `geometric_base`
records `cy_vol`, divisor volumes, effective-cone/GLSM charge rows, and
`kinv`, while `geometric_quantities` applies the `tau` and `kinv`
transformations before `dim_reductor` rebuilds the coefficients. The active
cross coefficient is

```text
(8*pi/V^2) * [pi*q_i^T Kinv q_j + (q_i+q_j).tau]
```

and the package `fixed` path now matches the corrected author-code validation
directory on the second h11=10 comparison geometry to Float64 precision. The fixed/full distinction is
therefore an explicit normalization choice rather than an unresolved source
reconstruction gap.

## Interpretation of the pilot results

The report tested the explicit grid
`0.90, 0.95, 0.99, 1.00, 1.01, 1.05, 1.10`; this is a bounded diagnostic
window, not a complete scale path.

| search | result | authorized interpretation |
| --- | --- | --- |
| complete all-branch rows | `(8,544,1)` has sign-changing generalized-Hessian brackets; no screen-passing corrected branch | `near_catastrophe` under the homotopy only |
| complete all-branch rows | weak `(8,1,1)` has no crossing signal | no crossing in the tested homotopy window |
| `1:1` low-index search | 8 corrected branches pass the numerical screen | `screen_candidate` under an incomplete homotopy search; not a physical or refined candidate |
| `1:2` low-index search | 80 corrected branches pass the numerical screen | `screen_candidate` under an incomplete homotopy search; not a physical or refined candidate |
| weak `(13,1,1)` low-index rows | screen rows occur at homotopy scales `1.01` and `1.10` | detuned numerical screen hits; no reference-scale or physical candidate |

The existing ten-geometry calibration rows remain `homotopy_only`. No row is a
`refined_candidate` or a `trajectory_candidate`; no generic trajectory
refinement was authorized. The existing fixed-potential calibration remains
`0/0`, so candidate recall is unmeasured rather than zero or complete.

The strong `(8,544,1)` crossing is evidence that the numerical deformation
changes the corrected branch/Hessian structure. It is not evidence for a
change of Calabi--Yau volume, moduli, axion dynamics, or inflationary
trajectory.

## Release claim

> The bounded ten-geometry study finds reproducible corrected
> Hessian-crossing and screening structure under an explicitly documented
> `L[2,:]` mathematical homotopy. The pilot also reproduces the author's
> generic coefficient map with selectable fixed/full volume laws, but no physical
> inflation candidate or trajectory has been established.

## Claims not authorized

The current evidence does not authorize any statement that:

- the pilot is a physical scan of the overall Calabi--Yau volume;
- a crossing is a realizable catastrophe in a stabilized compactification;
- any homotopy screen hit is a physical, refined, or trajectory candidate;
- the ten geometries represent the KS population or provide a candidate rate;
- the tested grid is complete or excludes crossings outside its window;
- the fixed benchmarks validate the generic transformation of `L` and `K`;
- the `0/0` fixed-point result measures candidate-recovery efficiency;
- a production high-`h11` scan should proceed before the physical map and
  resource budget are approved.

## Integration and next-step gate

**SCI-01 integration decision: no for production physical inflation
integration.** The pilot implementation is acceptable as a bounded diagnostic
with physical/full default, explicit `fixed|full` provenance, and no placement
of its rows in a physical candidate sample.

The next scientific step is to extend the geometry-level continuation record
with effective-cone/EFT domain checks and a bounded physical calibration. It
must pass the benchmark and generic reference checks for units, joint scaling,
positivity, generalized-Hessian behavior, and complete input provenance before
any generic physical continuation is reported as a scientific result.

The actual author code remains the reference for future regression tests. The
normalization distinction is:

```text
author path:    tau -> k*tau, Kinv -> k^2*Kinv, CY_vol held fixed
source-derived: tau -> k*tau, Kinv -> k^2*Kinv,
                V_CY -> k^(3/2)*V_CY for homogeneous overall scaling
```

At `k != 1`, the two coefficient normalizations differ by the common
`k^(-3)` volume factor for the leading term (and the corresponding factor for
the cross terms). Although a common factor does not change normalized
critical-point signs, it does change absolute inflationary observables such as
the scalar-amplitude normalization; this is why the selected normalization is
recorded in every continuation row.

Owner approval: ____________________    Date: __________
