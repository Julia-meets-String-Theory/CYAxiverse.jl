# CYAxiverse.jl Architecture & Coding Standards

## 0. Julia Execution Environment
- Run every Julia command used for CYAxiverse.jl package development directly in the regular local host environment. Do not run Julia through a sandbox, container, Docker image, remote runner, or other isolated environment.
- If the execution tool defaults to a sandbox, request approved local/unsandboxed execution before running Julia. Do not silently substitute a sandboxed Julia process for tests, audits, benchmarks, package commands, or development scripts.
- Julia project flags such as `--project=...` and `--startup-file=no` remain appropriate; this rule concerns the operating environment in which Julia runs.

## 1. Type Stability & Precision Rules
- Parametrize meaningful numerical kernels over float precision
  `T<:AbstractFloat` where the operation supports it. Do not force `Float64`
  through high-precision paths or add generic type parameters to non-numerical
  helpers.
- Axion mass matrices and instanton actions span 30+ orders of magnitude: NEVER hardcode `Float64` in internal mass/decay-constant routines.
- Ensure all struct fields for geometry data (intersection numbers, Mori matrix, charge matrix Q) are concrete types (e.g., `SparseVector{Int}`, `Array{T, 3}`).

## 2. Memory & Tensor Efficiency
- Triple intersection numbers $d_{ijk}$ are highly sparse. Always use sparse representations or pre-allocated view contractions for $d_i = d_{ijk} t^j t^k$ and $d_{ij} = d_{ijk} t^k$.
- Avoid heap allocations inside loops that sample Kähler moduli space ($t^i \in \text{Kähler Cone}$).

## 3. Physical Invariants & Domain Boundaries
- Any generated Kähler vector $t^i$ MUST satisfy positive Calabi-Yau volume $\mathcal{V} = \frac{1}{6} d_{ijk} t^i t^j t^k > 0$ and positive 4-cycle volumes $\tau_i > 0$.
- Kinetic matrix $K_{ij} = \frac{1}{2} g_{i\bar{j}}$ MUST be symmetric positive-definite. Throw a domain error if eigenvalues are non-positive.
- Axion decay constants $f_a$ and mass eigenvalues $m_a^2$ must be real and positive.

## 4. Boundary evidence

- Validate shape, orientation, units, identity, and provenance at every
  reader/writer or Python/Julia boundary.
- Preserve witness/action identities, selection routes, scale status, and
  failure reasons through generated artifacts.
- Do not promote a filtered, finite, provisional, or structurally complete
  result to a population or physical claim without the applicable source and
  replay gates.
