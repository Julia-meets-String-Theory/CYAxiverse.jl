---
name: cyaxiverse-vacua-pipeline
description: Build, repair, validate, or run CYAxiverse vacuum/minima pipelines and HDF5-backed batch jobs. Use for `vacua_pipeline.jl`, reduced-JLM, leading-branch enumeration, resumable scans, vacuum counts, or geometry-data writes.
---

# CYAxiverse Vacua Pipeline

1. Read `AGENTS.md`, the target script, its readers/writers, and relevant tests.
   Preserve dirty/unrelated work.
2. Treat the task as engineering unless the user explicitly authorizes a model
   change. Do not alter orientation, units, normalization, scientific
   accept/reject criteria, mass/tachyon semantics, persisted scientific fields,
   or `scale_status` under this skill alone.
3. Preserve the geometry/action contract. Validate `Q`, `L`, HDF5 dimensions,
   orientation, units, and identifiers at the boundary; keep path helpers,
   zero-padding, group/dataset names, compression, and compatible readers.
4. Make writes recoverable: no implicit overwrite; explicit force for
   replacement; validate targets; use temporary outputs and atomic finalization
   where practical; do not infer completion merely from file existence.
5. Persist replayable operational state: `GeometryIndex`, immutable input
   identity, schema/pipeline versions, configuration, seed, starts/limits,
   tolerances, solver status, residual diagnostics, runtime, Julia version, and
   Git revision. Distinguish verified, estimated, failed, invalid, unavailable,
   skipped, and timeout states.
6. Add focused tests for every changed boundary, including malformed dimensions,
   orientation, legacy layouts, metadata round trips, and a minimal fixture.
7. Run focused reproduction first, then the applicable package/audit/diff gates.

Report commands/outcomes, schema compatibility, source/input fingerprint,
resource impact, warnings, and any scientific decision deliberately deferred.
