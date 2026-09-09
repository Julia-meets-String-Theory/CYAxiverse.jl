---
name: cyaxiverse-ks-geometry-sampling
description: Investigate or improve CYAxiverse Kreuzer-Skarke/CYTools geometry generation and triangulation sampling. Use for geometry exporters, CYTools adapters, Kähler-cone construction, sampling diagnostics, or KS benchmark design where population, coverage, or selection effects matter.
---

# CYAxiverse KS Geometry Sampling

1. Read `AGENTS.md`, the actual generator/adapter/reader/tests, and the relevant
   scientific source or benchmark definition.
2. Define the target population and sampling unit before optimizing: polytope,
   triangulation, Calabi-Yau geometry, moduli-space point, or a stated hierarchy.
   Record database query parameters, filters, missingness, numerical rejection,
   and how the emitted sample differs from the target population.
3. Preserve a serializable Python boundary. Export arrays, scalars, labels,
   conventions, versions, identities, and validity/failure reasons; never
   serialize live Python objects or make core Julia import require CYTools.
4. Fingerprint the source/query before a bounded reproducible probe. Record
   source and tool versions, seed, manifest, proposal/acceptance counts,
   duplicate/coverage diagnostics, stage timings, terminal failures, and output
   size. Stop if the realized source or counting unit differs from the declared
   one.
5. Label sample properties only with evidence. Do not call a filtered, finite,
   or implementation-biased sample representative, complete, or unbiased.
6. Add regression coverage for deterministic identifiers, interchange round
   trips, validity checks, failure accounting, and cache invariants. Keep bulk
   generated data outside Git unless it is an intentionally reviewed fixture.
7. When delegated, own ordinary sampling diagnostics, implementation
   corrections, and retests. Keep detailed missingness, coverage, rejection,
   and selection diagnostics in durable artifacts while returning a compact
   manager-facing summary with references.

Finish with a measured recommendation stating target population, realized
sample, source fingerprint, selection effects, runtime/resources, coverage,
unresolved bias, and verification evidence. Use
`cyaxiverse-scientific-reproduction` for scientific claims beyond engineering
sampling diagnostics.
