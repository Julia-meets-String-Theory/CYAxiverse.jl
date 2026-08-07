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

## `CYAxiverse.axion_benchmarks`
```@autodocs
Modules = [CYAxiverse.axion_benchmarks]
Pages = ["paper_benchmarks.jl"]
```
## `CYAxiverse.cytools_wrapper`
```@autodocs
Modules = [CYAxiverse.cytools_wrapper]
Pages = ["add_functions/cytools_wrapper.jl"]
```

## Inflation benchmark bases

The axion benchmark helpers are available under
`CYAxiverse.axion_benchmarks`. The fixed poly-102 inflation model is exposed as
`CYAxiverse.axion_benchmarks.poly102_inflation`. Their raw coordinates are angular variables in
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
