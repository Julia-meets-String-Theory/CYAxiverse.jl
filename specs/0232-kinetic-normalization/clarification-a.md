# CYAX-0232 Clarification A — Decisions OD-1 to OD-7

**Status:** Draft, revision 2 (focused G0 round 1 of `f6c75bd`: REQUEST_CHANGES,
resolved here; round 2 pending; owner approval pending)
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
  transposed) and `Qmass = Q̃ᵀ·C·basis` (N×h11, correct). `LQtilde` selects
  h11 leading charges (`generate.jl:3173–3176`), so N = h11: the call throws
  for every h11 ≠ 2 and is mis-indexed at h11 = 2. Its docstring defines
  `phase` as the evaluation point "in the selected charge basis"
  (`generate.jl:2998–2999`), but the code uses `phase` directly as the
  mass-basis coordinate. The two agree in general only at `phase = 0`, where
  every tensor element vanishes.
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
  - both writers at a fixed nonzero ϑ₀, pinned in `plan.md` for every
    fixture (including the h11 = 3 fixture);
  - `hp_spectrum_save` with an `:hp_effective` selection where N > h11;
  - `pq_spectrum_save` at h11 = 3, with non-degenerate leading masses (the
    R-005 sign-only allowance does not hold under rotation inside a
    degenerate subspace);
  - an independent check of each stored tensor against central finite
    differences of the Hessian of V(x) = Σₐ Λₐ(1 − cos(Qₐ·x)), built from
    exactly the `(L, Q)` passed to `cubic` in that writer's basis: the
    selected (merged) set for `hp_spectrum_save`, and only the leading
    `LQtilde` set in mass coordinates for `pq_spectrum_save`, not the full
    potential. Step and relative tolerance are pinned in `plan.md`. If the
    test uses `generate.hessian` (`generate.jl:543`), it SHALL pass that
    function's own argument orientation (L N×2, Q n×N);
  - the R-005 cubic rows (scale 2^{-3/2} in the mass basis at fixed ϑ₀, and 1
    in the charge basis).
- **A1.5** The conventions page (R-009) SHALL state that a cubic tensor in the
  legacy F2 layout (no `spectrum/metadata`) whose stored `spectrum/cubic/phase`
  is nonzero is unreliable (mis-oriented or wrong evaluation point). R-007 already makes
  legacy F2 report `cubic_basis = :unknown`, and consumers that need the basis
  throw. No regeneration is required.

## A2 — F5 legacy versions 1.0 and 1.1 (OD-2)

R-007 *Inference* is extended: F5 at `…-reproduction-1.0` or
`…-reproduction-1.1` with no label SHALL also be inferred as `:legacy_c_half`,
on one condition.

Each version was written by a range of commits, not one:
- `…-1.0`: `ec73c15` through `e7a8f51`. `ec73c15` writes no `model_stage`
  block; `aee4c1a` and later 1.0 commits do.
- `…-1.1`: `b191af1`, `31cf465`, `f45ea31`, `68df2a2` and `239a3a7`.

`plan.md` SHALL record every distinct layout written under each version over
its full commit range, each with the commits that wrote it and the fields that
depend on the kinetic matrix. A reader SHALL infer only a file that matches
one recorded layout for its version, and SHALL throw otherwise. One test per
recorded layout.

## A3 — Historical inflation CSV versions (OD-3)

This supersedes approval note 5's limit to the pre-change constants.

Version values alone do not identify a family. The refinement, flow and
pilot CSVs all use the generic column `schema_version`
(`inflation_candidate_refinement.jl:21, 42`; `inflation_scale_continuation.jl:56`);
refinement (`c7482f5`) and pilot (`10aa058`) both first wrote `"3"`; and the
refinement file (`point` and `summary` rows, 168, 391) and the separate flow
file (`flow` rows, 285, 551) use the same column name. Only the shard family
has its own column, `shard_schema_version`.

- **A3.1** A legacy row's *layout key* is:
  - the exact header of its file (the ordered column list);
  - its `row_type`, where the header has one;
  - the values of that family's version columns.
  The version columns are enumerated per family in `plan.md` (for example
  `schema_version`, `shard_schema_version`, `contract_version`,
  `diagnostic_schema_version`). The pilot columns
  `domain_certificate_version` and `conversion_policy_version` version other
  contracts, not the CSV layout; they are not part of the key.
- **A3.2** `plan.md` SHALL record every layout key that merged code wrote to a
  persisted inflation CSV, each with its family and the commits that wrote it.
- **A3.3** A legacy row is inferred as `:legacy_c_half` only if its layout key
  matches exactly one recorded entry. No column takes priority over another.
  A row whose key matches no entry, or matches entries of more than one
  family, SHALL be rejected with the R-007 error naming the file, the row and
  the values.
- **A3.4** One test per recorded layout key, and rejection tests for an
  unknown key and for a contradictory version value under a known header.

## A4 — Reproduction functions that use no kinetic matrix (OD-4)

A reproduction function (R-008 classification rule) that consumes no kinetic
matrix, directly or through another function, has no label duty. Its outputs
are not K-dependent under the R-006 general rule. The R-008 search record in
`plan.md` lists each such function as *K-free*. A result with K-free fields
and K-dependent fields from a single contract keeps the R-008 top-level field;
only a result with K-dependent fields from both P96 and A96 follows A7.

## A5 — Scale-continuation resume (OD-5)

R-007 *Batch resume* also covers `run_scale_continuation`
(`inflation_scale_continuation.jl:2023`), with the `inflation_scan_prep.jl`
rule:
- a row labelled or inferred other than `"sugra_c1"` is not complete;
- `--resume` on a report that holds such rows refuses, naming the stored label;
- the remedy is a new report path, because the runner has no overwrite option;
- `_pilot_completed_scales` (`:1474`) SHALL raise instead of silently
  skipping. Today it skips in three places: missing required columns return an
  empty set (`:1482`), a field-count mismatch skips the row (`:1486`), and a
  `try`/`catch` drops parse errors (`:1488–1494`). R-007 errors SHALL also
  propagate.

## A6 — Script with an untracked dependency (OD-6)

`scripts/scan_fresh_relaxed_qcd_candidates.py` passes `kinv` to a module that
is not in the repository. Its outputs cannot be classified from repository
source. The script is out of scope and exempt from the R-006 general labelling duty:
- it is not changed, and it SHALL NOT write a `kinetic_normalization` label;
- physical readers treat its outputs as an unknown layout and throw (R-007);
- the conventions page (R-009) and the writer inventory SHALL state that
  CYAX-0232 does not certify its outputs, under either convention.

## A7 — Reproduction results that combine contracts (OD-7)

This extends the R-008 field duty. It applies only to a reproduction result
struct or `NamedTuple` with K-dependent fields from both P96 and A96. Such a
result:
- SHALL NOT carry a single top-level `kinetic_normalization` field;
- SHALL carry `kinetic_normalization_by_field`, a `NamedTuple` that maps each
  K-dependent field to a sorted tuple of labels
  (`(:a96_reproduction_raw_radian,)`, `(:p96_reproduction_c_half,)`, or both).
  Tuple values keep the type stable. K-free fields are omitted.

Classification is by provenance: a field depends on every contract whose
function produced a value it was computed from, even where the value is
mathematically independent of K (for example an inertia count, or a
degeneracy scale).

`pilot_benchmark_regression` (`inflation_scale_continuation.jl:1829–1897`) is
such a result:
- `n8_detuned_negative_modes` and `n8_detuned_eigenvalues` come from the P96
  `paper_benchmarks.n8_hilltop` (1862), evaluated at `n8.k + 1e-7`, where
  `n8.k` comes from the A96 solve: both labels;
- the other `n8_*` fields come from the A96 `poly102_inflation`
  `n8_degenerate_point` (1830, 1861): A96;
- the `n5_*` fields are K-free;
- `passed` combines the A96 and P96 checks: both labels.

A test asserts the map. A physical consumer SHALL reject any result that
carries `kinetic_normalization_by_field` (R-007, R-008). Results with one
contract, with or without K-free fields, keep the R-008 top-level field.

## Effect on other requirements

- R-005, R-006, R-009 and the Interfaces section are unchanged except as
  stated above. `kinetic_normalization_by_field` is an additive field.
- Downstream (PR #219) is unchanged: its consumer adaptation is tracked
  separately and is not part of this specification.
