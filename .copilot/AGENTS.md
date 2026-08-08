# CYAxiverse.jl Architecture & Coding Standards

## 1. Type Stability & Precision Rules
- Parametrize all numerical functions over float precision `T<:AbstractFloat` (e.g., `Float64`, `BigFloat`, or `ArbNumerics`).
- Axion mass matrices and instanton actions span 30+ orders of magnitude: NEVER hardcode `Float64` in internal mass/decay-constant routines.
- Ensure all struct fields for geometry data (intersection numbers, Mori matrix, charge matrix Q) are concrete types (e.g., `SparseVector{Int}`, `Array{T, 3}`).

## 2. Memory & Tensor Efficiency
- Triple intersection numbers $d_{ijk}$ are highly sparse. Always use sparse representations or pre-allocated view contractions for $d_i = d_{ijk} t^j t^k$ and $d_{ij} = d_{ijk} t^k$.
- Avoid heap allocations inside loops that sample Kähler moduli space ($t^i \in \text{Kähler Cone}$).

## 3. Physical Invariants & Domain Boundaries
- Any generated Kähler vector $t^i$ MUST satisfy positive Calabi-Yau volume $\mathcal{V} = \frac{1}{6} d_{ijk} t^i t^j t^k > 0$ and positive 4-cycle volumes $\tau_i > 0$.
- Kinetic matrix $K_{ij} = \frac{1}{2} g_{i\bar{j}}$ MUST be symmetric positive-definite. Throw a domain error if eigenvalues are non-positive.
- Axion decay constants $f_a$ and mass eigenvalues $m_a^2$ must be real and positive.