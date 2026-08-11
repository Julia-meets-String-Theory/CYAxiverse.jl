# Available functions
---
```@meta
CurrentModule = CYAxiverse
```

```@docs
CYAxiverse.CYAxiverse
CYAxiverse.greet_CYAxiverse
```

## `CYAxiverse.filestructure`
```@autodocs
Modules = [CYAxiverse.filestructure]
Pages = ["filestructure.jl"]
```
## `CYAxiverse.generate`
```@autodocs
Modules = [CYAxiverse.generate]
Pages = ["generate.jl"]
```

## `CYAxiverse.structs`
```@autodocs
Modules = [CYAxiverse.structs]
Pages = ["structs.jl"]
```

## `CYAxiverse.inflation_points`
```@autodocs
Modules = [CYAxiverse.inflation_points]
Pages = ["inflation_points.jl"]
```

## `CYAxiverse.minimizer`
```@autodocs
Modules = [CYAxiverse.minimizer]
Pages = ["minimizer.jl"]
```
## `CYAxiverse.read`
```@autodocs
Modules = [CYAxiverse.read]
Pages = ["read.jl"]
```

## `CYAxiverse.plotting`
```@autodocs
Modules = [CYAxiverse.plotting]
Pages = ["plotting.jl"]
```

## `CYAxiverse.jlm_reduced`
```@autodocs
Modules = [CYAxiverse.jlm_reduced]
Pages = ["jlm_reduced.jl"]
```

## `CYAxiverse.paper_benchmarks`
```@autodocs
Modules = [CYAxiverse.paper_benchmarks]
Pages = ["paper_benchmarks.jl", "paper_benchmarks/reduced_models.jl",
    "paper_benchmarks/compatibility.jl"]
```
## `CYAxiverse.paper_benchmarks.author_inflation`
```@autodocs
Modules = [CYAxiverse.paper_benchmarks.author_inflation]
Pages = ["paper_benchmarks/poly102_inflation.jl"]
```
```@docs
CYAxiverse.paper_benchmarks.poly102_inflation
CYAxiverse.axion_benchmarks
```

## Inflation benchmark bases

The paper benchmark helpers are available under
`CYAxiverse.paper_benchmarks`. The fixed poly-102 inflation model is exposed as
`CYAxiverse.paper_benchmarks.poly102_inflation`. Their raw coordinates are angular variables in
radians, with canonical coordinates defined by
`chi = K^(1/2) * theta`.

Pass `basis=:mass_eigenbasis` to
`n8_unstable_direction`, `n8_inflation_initial_condition`,
`n8_hilltop_normal_form`, `n8_efold_gradient_flow`, or
`n8_physical_gradient_flow` to use the fixed hilltop mass basis. Its raw
vectors solve

```text
H_theta * v_i = m_i^2 * K * v_i,
v_i' * K * v_j = delta_ij.
```

The basis is constructed once at the catastrophe/hilltop and is not
recomputed along the nonlinear trajectory. At fixed `K`, the
most-negative canonical-Hessian direction is the corresponding unstable mass
direction; it is distinct from the kinetic eigenbasis.

`n8_physical_gradient_flow` reports the full nonlinear physical-time flow.
`n8_hilltop_normal_form_efolds` reports the separate one-dimensional local
normal-form estimate. If the physical solver reaches `max_time` while still
inflating, it reports `end_event=:tmax` and `terminated=false`.
