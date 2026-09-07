---
name: cyaxiverse-julia-quality
description: Implement, debug, refactor, or audit Julia code in CYAxiverse.jl, especially numerical kernels, HDF5 readers, optional PyCall boundaries, JET/Aqua findings, test hygiene, and package regressions. Use for changes under `src/`, `ext/`, `bin/audit.jl`, `scripts/`, or `test/` where Julia numerical or package-quality constraints matter.
---

# CYAxiverse Julia Quality

1. Read `AGENTS.md`, the target code, neighboring tests, `Project.toml`, and a
   comparable implementation when useful. Preserve unrelated work.
2. Keep mechanical reliability separate from scientific interpretation. Stop
   for owner direction before changing physical normalization, scientific
   acceptance, mass/tachyon meaning, persisted scientific fields, units,
   orientation, or a scientific schema when the intended convention is unclear.
3. Preserve precision and exactness intentionally. Parametrize meaningful
   numerical kernels over suitable abstract float types; do not narrow existing
   high-precision, rational, `BigInt`, or `ArbFloat` paths. Prefer concrete
   internal containers and sparse/in-place operations in hot loops.
4. Keep Python optional. Core `using CYAxiverse` and package tests must not
   require CYTools/PyCall; validate and narrow external data at the adapter
   boundary.
5. Preserve HDF5 paths, dimensions, orientation, units, compression, and legacy
   reader compatibility unless the task explicitly changes the contract.
6. Add focused regression coverage for the changed boundary. Fix warnings and
   method-redefinition issues structurally rather than suppressing them.
7. Verify focused behavior first, then run the applicable package tests,
   `bin/audit.jl`, Python-free import, docs build, and `git diff --check` as the
   scope requires. Prefer `scripts/agent_verify.py` for concise evidence.

Report exact commands and observed outcomes, including warnings and unavailable
checks. Use `cyaxiverse-scientific-reproduction` whenever a numerical change
could alter a benchmark, scientific interpretation, or population claim.
