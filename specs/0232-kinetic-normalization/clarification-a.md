# CYAX-0232 Clarification A — Decisions OD-1 to OD-7

**Status:** Draft (focused G0 review pending; owner approval pending)
**Amends:** `spec.md` approved at `f097536` (`approval_ref` in `spec.md`)
**Origin:** the seven open points in `plan.md` (*Owner decisions required
before the affected tasks*) and an independent scientific review of the plan
at `dceecbf`, which the owner adopted on 2026-11-12. That review accepted
OD-2, OD-4 and OD-5, accepted OD-1 with an added phase requirement, accepted
OD-6 with an explicit non-certification statement, and modified OD-3 and OD-7.

This is a normative amendment to CYAX-0232. Where it conflicts with wording in
`spec.md`, this clarification supersedes that wording. `spec.md` is not
rewritten. File:line citations refer to `10aa3dc`.

## A1 — F2 cubic tensor: argument orientation and evaluation phase (OD-1)

**Implementation facts.**
- `generate.cubic(x, L, Q)` (`generate.jl:567–581`) requires `L` as N×2
  (sign, log10 scale per instanton) and `Q` as N×n, with `length(x) == n`, and
  evaluates `sin(Q*x)`.
- `hp_spectrum_save` (`generate.jl:1073`) passes `Ltilde` (2×N) and `Qtilde`
  (h11×N), both transposed. The assertions pass only when N = h11 = 2. The
  `:hp_effective` selection re-appends merged subleading charges
  (`_hp_selected_potential`, `generate.jl:821`), so N can exceed h11, and the
  call then throws even at h11 = 2.
- `pq_spectrum_save` (`generate.jl:3013–3015`) passes `Ltilde` (2×N,
  transposed) and `Qmass = Q̃ᵀ·C·basis` (N×h11, correct). Its docstring defines
  `phase` as the evaluation point "in the selected charge basis", but the code
  uses `phase` directly as the mass-basis coordinate. The two agree only at
  `phase = 0`, where every tensor element vanishes.
- Neither writer has a test.

**Requirements.**
- **A1.1** Both F2 writers SHALL call `cubic` with `L` as N×2 and `Q` as N×n.
- **A1.2** The `phase` argument of both writers is the charge-basis evaluation
  point ϑ₀, in the coordinates in which `potential_factored` defines `Q`.
  `hp_spectrum_save` SHALL evaluate the charge-basis tensor at ϑ₀.
  `pq_spectrum_save` SHALL evaluate the canonical-mass-basis tensor at
  x₀ = basisᵀ·C⁻¹·ϑ₀, with `C` the canonical factor of the current convention,
  so that `Qmass·x₀ = Q̃ᵀ·ϑ₀`. This is the transformation that the R-005 cubic
  row already assumes.
- **A1.3** `spectrum/cubic/phase` SHALL store ϑ₀ for both writers.
- **A1.4** Tests SHALL cover:
  - both writers at a fixed nonzero ϑ₀;
  - `hp_spectrum_save` with an `:hp_effective` selection where N > h11;
  - `pq_spectrum_save` at h11 ≥ 3;
  - an independent check of each tensor against finite differences of the
    potential's Hessian, in the writer's own basis;
  - the R-005 cubic rows (scale 2^{-3/2} in the mass basis at fixed ϑ₀, and 1
    in the charge basis).
- **A1.5** The conventions page (R-009) SHALL state that F2 cubic tensors
  written at or before the implementation base with a nonzero phase are
  unreliable (mis-oriented or wrong evaluation point). R-007 already makes
  legacy F2 report `cubic_basis = :unknown`, and consumers that need the basis
  throw. No regeneration is required.

## A2 — F5 legacy versions 1.0 and 1.1 (OD-2)

R-007 *Inference* is extended: F5 at `…-reproduction-1.0` or
`…-reproduction-1.1` with no label SHALL also be inferred as `:legacy_c_half`,
on one condition. `plan.md` SHALL first record, from the commit that wrote each
version (`ec73c15` for 1.0, `b191af1` for 1.1), the fields that depend on the
kinetic matrix (at least the `model_stage` block). A reader SHALL verify that a
file has the recorded layout for its version before inferring, and SHALL throw
otherwise. Each accepted version has a focused test.

## A3 — Historical inflation CSV versions (OD-3)

This supersedes approval note 5's limit to the pre-change constants.

- **A3.1** `plan.md` SHALL record every combination of version columns that
  merged code wrote to a persisted inflation CSV row. Each entry names the
  family, the full tuple of version columns present in the row (for example
  `contract_version`, `diagnostic_schema_version`, and the shard, pilot,
  refinement or flow schema column), and the commit that wrote it.
- **A3.2** A legacy row is inferred as `:legacy_c_half` only if its full
  version tuple matches one recorded combination for that family.
- **A3.3** No version column takes priority over another. A row whose version
  columns contradict each other, or match no recorded combination, or match
  more than one family, SHALL be rejected with the R-007 error naming the file,
  the row and the conflicting values.
- **A3.4** One test per recorded combination, and one rejection test for a
  contradictory tuple.

## A4 — Reproduction functions that use no kinetic matrix (OD-4)

A reproduction function (R-008 classification rule) that consumes no kinetic
matrix, directly or through another function, has no label duty. Its outputs
are not K-dependent under the R-006 general rule. The R-008 search record in
`plan.md` lists each such function as *K-free*. A result that mixes K-free and
K-dependent parts follows A7.

## A5 — Scale-continuation resume (OD-5)

R-007 *Batch resume* also covers `run_scale_continuation`
(`inflation_scale_continuation.jl:2023`), with the `inflation_scan_prep.jl`
rule:
- a row labelled or inferred other than `"sugra_c1"` is not complete;
- `--resume` on a report that holds such rows refuses, naming the stored label;
- the remedy is a new report path, because the runner has no overwrite option;
- `_pilot_completed_scales` (`:1474`) SHALL let parse and R-007 errors
  propagate (today it suppresses them at 1488–1494).

## A6 — Script with an untracked dependency (OD-6)

`scripts/scan_fresh_relaxed_qcd_candidates.py` passes `kinv` to a module that
is not in the repository. Its outputs cannot be classified from repository
source. The script is out of scope:
- it is not changed, and it SHALL NOT write a `kinetic_normalization` label;
- the conventions page (R-009) and the writer inventory SHALL state that
  CYAX-0232 does not certify its outputs, under either convention.

## A7 — Reproduction results that combine contracts (OD-7)

This extends the R-008 field duty. A reproduction result struct or `NamedTuple`
whose fields come from more than one contract (P96, A96 or K-free):
- SHALL NOT carry a single top-level `kinetic_normalization` field;
- SHALL carry `kinetic_normalization_by_field`, a `NamedTuple` that maps each
  field depending on a kinetic matrix to its label
  (`:p96_reproduction_c_half` or `:a96_reproduction_raw_radian`), or to a
  sorted tuple of both labels for a field derived from both contracts. K-free
  fields are omitted.

`pilot_benchmark_regression` (`inflation_scale_continuation.jl:1829–1894`) is
such a result:
- `n8_detuned_negative_modes` and `n8_detuned_eigenvalues` come from the P96
  `paper_benchmarks.n8_hilltop` (1862);
- the other `n8_*` fields come from the A96 `poly102_inflation`
  `n8_degenerate_point` (1830, 1861);
- the `n5_*` fields are K-free;
- `passed` combines the A96 and P96 checks, so it maps to both labels.

A test asserts the map. Single-contract results keep the R-008 top-level
field.

## Effect on other requirements

- R-005, R-006, R-009 and the Interfaces section are unchanged except as
  stated above. `kinetic_normalization_by_field` is an additive field.
- Downstream (PR #219) is unchanged: its consumer adaptation is tracked
  separately and is not part of this specification.
