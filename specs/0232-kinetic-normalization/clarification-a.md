# CYAX-0232 Clarification A — Decisions OD-1 to OD-7

**Status:** Draft, revision 4 (focused G0 rounds 1–3 (`f6c75bd`, `98060aa`,
`ea93938`): REQUEST_CHANGES, resolved here; round 4 pending; owner approval
pending)
**Amends:** `spec.md` approved at `f097536` (`approval_ref` in `spec.md`)
**Origin:** the seven open points in `plan.md` (*Owner decisions required
before the affected tasks*) and an independent scientific review of the plan
at `dceecbf`, which the owner adopted on 2026-11-12. That review accepted
OD-2, OD-4 and OD-5, accepted OD-1 with an added phase requirement, accepted
OD-6 with an explicit non-certification statement, and modified OD-3 and OD-7.

*Merged code* and *merged commits* below mean commits that are ancestors of the
implementation base `10aa3dc`. Outputs written by other commits match no
recorded layout or entry, and readers throw.

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
- **A1.4** Tests SHALL use dedicated cubic fixtures, pinned in `plan.md` with
  every `Kinv`, `Q`, `L` and ϑ₀. In each fixture, all potential coefficients
  Λₐ = 10^{L[2,a]} (the coefficients `cubic` uses, not physical scales) lie
  within a factor 10³ of one another. For the two R-005 cubic rows these
  fixtures replace the R-004 and R-003 fixture inputs. (The R-004 fixture, with
  Λ₂⁴/Λ₁⁴ ≈ 3·10⁻⁵¹, is unsuitable: Float64 rounding in the mass basis exceeds
  the subleading content.) The tests SHALL cover:
  - both writers at a fixed nonzero ϑ₀;
  - an h11 = 2 fixture for both writers;
  - `hp_spectrum_save` with an `:hp_effective` input whose subleading charge
    passes both `αmatrix` tests: the `valid_cols` filter
    (log10 Λ̄ ≥ min log10 Λ̃ + log10 0.5, `generate.jl:3602–3609`) and the
    per-pair keep test (log10 Λ̄ − log10 Λ̃ⱼ > log10 0.5 for some j with
    |αᵢⱼ| > 1e-3, `:3621–3631`), so that it reaches the `Canonicalα` branch
    with N > h11;
  - `pq_spectrum_save` at h11 = 3;
  - non-degenerate leading masses in every `pq_spectrum_save` fixture,
    asserted in the test as a relative gap of at least 0.5 between adjacent
    leading Hessian eigenvalues (the R-005 sign-only allowance does not hold under
    rotation inside a degenerate subspace);
  - the test computes x₀ itself from the stored ϑ₀ and asserts
    `Qmass·x₀ ≈ Q̃ᵀ·ϑ₀`, so that the check is not circular with respect to
    A1.2;
  - an independent check of each stored tensor against central finite
    differences of the Hessian of V(x) = Σₐ Λₐ(1 − cos(Qₐ·x)), built from
    exactly the `(L, Q)` passed to `cubic` in that writer's basis: the
    selected (merged) set for `hp_spectrum_save`, and only the leading
    `LQtilde` set in mass coordinates for `pq_spectrum_save`, not the full
    potential. Step and relative tolerance are pinned in `plan.md`. If the
    test uses `generate.hessian` (`generate.jl:543`), it SHALL pass that
    function's own argument orientation (L N×2, Q n×N);
  - the R-005 cubic rows (scale 2^{-3/2} in the mass basis at fixed ϑ₀, and 1
    in the charge basis), compared normwise (Frobenius norm) after aligning
    the sign of each mass-basis mode, at relative tolerance 1e-10.
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
depend on the kinetic matrix.

A file's *normalized path set* is built from its JSON as follows:
- object keys form paths at every depth; a key whose value is `null` is
  present;
- an array collapses to one path segment `[]`, and the keys of its elements
  are the union over all elements;
- an object whose keys are data values rather than field names (a count by
  reason, a map keyed by h11, polytope, class, matrix or component, for
  example `reason_counts_by_polytope` at `e7a8f51:957–976`) is recorded as a
  map node `<map>`; its keys are not compared, and its values are normalized
  as one element.

A recorded layout is a required set R and an optional set O of normalized
paths. Each optional path cites the source lines that make it optional (for
example `model_stage.convention.prefactor_P`, absent when there are no records,
`aee4c1a:491–590`, or status-dependent `details[]` keys). Runtime flags change
the layout within one commit (for example `--model-stage` and `--keep-details`
at `aee4c1a`, lines 722 and 797); each flag variant is a separate recorded
layout. A file matches a layout when R ⊆ (its normalized path set) ⊆ R ∪ O.
A reader SHALL infer only a file that matches exactly one recorded layout for
its version, and SHALL throw otherwise. One test per recorded layout, plus a
rejection test for a file with an unrecorded path.

## A3 — Historical inflation CSV versions (OD-3)

This supersedes approval note 5's limit to the pre-change constants.

Version values alone do not identify a family. The refinement, flow and
pilot CSVs all use the generic column `schema_version`
(`inflation_candidate_refinement.jl:21, 42`; `inflation_scale_continuation.jl:56`);
refinement (`c7482f5`) and pilot (`10aa058`) both first wrote `"3"`; and the
refinement file (`point` and `summary` rows, 168, 391) and the separate flow
file (`flow` rows, 285, 551) use the same column name. A row can also carry
several version columns at once: scan-prep rows carry `contract_version` and
`diagnostic_schema_version` (`inflation_scan_prep.jl:21`), and shard rows add
`shard_schema_version` to the scan-prep fields
(`inflation_scan_shards_common.jl:15–16`).

- **A3.1** A *recorded entry* is one historical file layout:
  - the exact header (the ordered column list);
  - the `row_type` value, where the header has one;
  - the values of the version columns listed for that entry (for example
    `schema_version`, `shard_schema_version`, `contract_version`,
    `diagnostic_schema_version`).
  The pilot columns `domain_certificate_version` and
  `conversion_policy_version` version other contracts, not the CSV layout;
  they are not listed in any entry.
- **A3.2** `plan.md` SHALL record every entry that merged code wrote to a
  persisted inflation CSV, each with the commits and the writer functions that
  wrote it. Entries with the same header SHALL list the same version columns,
  so that no row can match two entries by construction.
- **A3.3** A legacy row is inferred as `:legacy_c_half` only if it matches
  exactly one recorded entry and every writer of that entry is a physical
  function (R-008 classification). No column takes priority over another. A
  row that matches no entry, or more than one entry, SHALL be rejected with
  the R-007 error naming the file, the row and the values. An entry with a
  reproduction writer follows the R-007 rules for legacy reproduction outputs
  and mixed legacy files; for example, `inflation_refinement_common.jl:74`
  writes A96 rows in the diagnostic layout.
- **A3.4** One test per recorded entry, and rejection tests for an
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
- `--resume` on a report or shard file that holds such rows refuses, naming
  the stored or inferred label. This check SHALL run before the existing
  header comparison in `pilot_prepare_csv` (`:1456`, called at
  `:2039–2040`), for both the report and the shard file before either call,
  so that a legacy file is reported by its label rather than as a "CSV schema
  mismatch";
- the remedy is a new report path and a new shard directory, because the
  runner has no overwrite option;
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
  K-free fields are omitted.

For such a result this also supersedes the R-006 *In memory* requirement of a
single `kinetic_normalization` field.

Classification is by provenance: a field depends on every contract whose
function produced a value it was computed from, even where the value is
mathematically independent of K (for example an inertia count, or a
degeneracy scale). K-free functions (A4) contribute no contract, even when
they belong to a reproduction module.

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
