# Inflation scan call contract

Status: script-level contract only; no generic package scan API is introduced.

The driver is
`scripts/inflation_scan_contract.jl`. It is intentionally the only generic
layer at this stage. The package functions remain responsible for their own
scientific operations.

## Locked sequence

For each `GeometryIndex(h11, polytope, frst)`:

1. `CYAxiverse.read.potential(geom_idx)` returns an `AxionPotential` with
   `L`, `Q`, and `K`. The driver normalizes the orientation to
   `Q :: h11 × n_instantons`, `L :: 2 × n_instantons`, and
   `K :: h11 × h11`.
2. `CYAxiverse.generate.LQtilde(Q, L)` is called exactly once. Its
   `Qtilde`, `Qbar`, `Ltilde`, and `Lbar` are reused downstream.
3. `CYAxiverse.generate.instanton_hierarchy_diagnostics(L)` supplies the
   cheap hierarchy fields.
4. One Cholesky factorization of `K` is reused for canonical-Hessian and
   gradient-norm calculations across all candidate branches.
5. `CYAxiverse.generate.leading_hessian_mass_basis_float64(
   K, selected.Ltilde, selected.Qtilde)` supplies a Float64 mass-basis
   diagnostic. It is a screening diagnostic, not the arbitrary-precision
   trajectory calculation.
6. `CYAxiverse.generate.leading_critical_branches(
   selected; max_branches=...)` enumerates the leading branches only when the
   explicit branch cap permits it.
7. The script evaluates the full `Q/L` potential derivatives on each returned
   branch using log-shifted amplitudes. Classification is local to the script;
   it is not currently a package API.

The generic trajectory/refinement call is deliberately not part of this
contract yet. It must accept a validated geometry-specific representation and
explicitly record solver precision, tolerances, event policy, status, and
failure diagnostics before it is added here.

## Sample contract probe

Command:

```sh
julia --project=. --startup-file=no \
  scripts/inflation_scan_contract.jl \
  --data-dir paper_benchmarks/appendix_c \
  --geometry 8,1,1 --max-branches 100000
```

The checked-in Appendix-C sample returned:

```text
status=:success
Q=(8,78), L=(2,78), K=(8,8)
selected_instantons=8
branch_count=256
leading_minima_count=1
candidate_slowroll_saddles=0
```

The first run includes Julia method compilation. The recorded stage fields are
diagnostic only until a warmup policy is added to the eventual scan driver.
