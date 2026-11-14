# F5 legacy layouts: `…-reproduction-1.0` (CYAX-0232 T041)

Recorded layouts of the reproduction JSON written by
`scripts/reproduce_fuzzy_axions_h11_4.py` under schema
`cyaxiverse-fuzzy-axions-h11-4-reproduction-1.0`, for Clarification A2. The
record is in two parts that join at `47b1d74` → `fc66261`: part A covers
`ec73c15`–`47b1d74` (its groups 1–4), part B covers `fc66261`–`e7a8f51` (its
groups 1–2, called 1.0 groups 5–6 in `plan.md`). The shared conventions are in
`plan.md` (*Historical layout records (T041)*). Every commit of the 1.0 range is
assigned to a group: part A lists its commits, and part B lists the commits
that change only imported modules, with the diff evidence that they add no
path.

## Part A: `ec73c15`–`47b1d74`

**Import-only commits in part A's range.** `d36c51b`, `c2b5851`, `8b87ea8`
and `e24064f` have the same blobs as `4d68038` for the writer and every module
in its transitive local-import closure, so they write group 3's layouts.
`d36c51b` changes only `scripts/fuzzy_axion_model_stage_driver.jl` (the
superpotential, a value change; the writer reads the same fixed datasets),
`c2b5851` changes only a test, and the other two change no file under
`scripts/`.

Commits in this record (in order): `ec73c15`, `aee4c1a`, `4d68038`, `efdcbce`, `47b1d74`.
The other 1.0 commits (after `47b1d74`, up to `e7a8f51`) are not in this record.

### Method and conventions

- Sources were read with `git cat-file -p <commit>:<path>`. Writer:
  `scripts/reproduce_fuzzy_axions_h11_4.py`. The JSON is the `summary` dict in
  `reproduce()`, passed through `_jsonable` and written by
  `json.dumps(result, indent=2, sort_keys=True)` + `args.output.write_text`
  (`ec73c15`:648,652; `aee4c1a`:864,868; `4d68038`:1076,1080; `efdcbce`:1064,1068;
  `47b1d74`:1088,1092). No code mutates `summary` after it is built. The only
  later change is to `model_stage["diagnostic_reason"]`, before `summary` is built
  (`aee4c1a`:723-740), and it changes a value, not a key.
- Imported local modules whose dicts reach the JSON:
  - `scripts/generate_geometric_data_multitriangulation.py` (`GEN`): this file is
    byte-identical at all five commits. It supplies:
    - `details[].provenance`, from `load_mirror_polytopes` (`GEN`:4075-4094; the
      dict is at 4078-4092);
    - `details[].kaehler_point_export_per_class[].diagnostic`, from
      `evaluate_kaehler_point` (`GEN`:1472-1666).
  - `scripts/inherited_orientifold_candidates.py`: nothing from it is persisted.
    At `ec73c15`/`aee4c1a`/`4d68038` it supplies only involution matrices and shift
    dicts. These are used internally by `_orientifold_action_audit`. At
    `efdcbce`/`47b1d74` it supplies `enumerate_orientifold_candidates` records,
    which that function reduces to three counts. The persisted audit dict is
    always built in the writer itself.
  - `scripts/fuzzy_axion_model_stage_driver.jl` (from `aee4c1a`) writes an
    HDF5 file. The writer reads selected datasets from it into `model_stage`
    (see the K-dependent fields).
- Path notation follows the shared T041 convention (see `plan.md`). An array-valued key `k` gives the path `k`.
  The keys of its object elements give `k[].key`. **I do not list a separate
  bare `k[]` path** (see Open point 1).
- **R/O convention for `details[]`:** when `--keep-details` is set,
  `details` is a list with one element per loaded polytope. Every element has the
  same top-level keys (dict literal at `details.append`). Those keys are put in
  **R**. If the population is empty (`records == []`, which happens only when the
  Parquet mirror has no favorable polytope for the requested h11;
  `load_mirror_polytopes` always returns at least one record when one exists, even
  for `--limit 0`, `GEN`:4095-4096), `details == []`. That file's path set is then
  identical to the matching non-`--keep-details` layout, so it matches that
  layout (see Open point 3).
- **Map nodes: none in this range.** Every object's keys are fixed field names.
  Candidates I checked and rejected: `paper_targets` (target names from a fixed
  literal), `counts`, `claim_status`, `input`, `details[].provenance`,
  `...diagnostic` and `...diagnostic.checks` (check names from fixed literals),
  `model_stage.convention`. `PAPER_TARGETS_BY_H11` is keyed by h11, but only one
  value of it is persisted, never the outer dict.

---

### Named path sets (used by all groups)

Each set is listed once here. The groups below combine these sets.

#### `BASE_1` (top level, `ec73c15`; 36 paths)
```
schema_version
paper
orientifold_source
input
input.source
input.parquet_dir
input.requested_h11
input.favorable_lattice
input.record_count
input.population_complete
counts
counts.favorable_polytopes
counts.raw_frsts
counts.frst_classes
counts.raw_trilayer_polytopes
counts.raw_trilayer_frst_classes
counts.nonfrozen_trilayer_frst_classes
counts.identity_torus_action_count
counts.identity_torus_action_cy_count
counts.identity_valid_o3o7_action_cy_count
counts.source_vertex_evidence_inherited_orientifold_cys
counts.source_vertex_evidence_h11_minus_zero_orientifold_cys
counts.kaehler_point_export_accepted_count
counts.kaehler_point_export_rejected_count
paper_targets
paper_targets.favorable_polytopes
paper_targets.frst_classes
paper_targets.inherited_orientifold_cys
paper_targets.h11_minus_zero_orientifold_cys
paper_targets.h11_minus_zero_h21_plus_zero_orientifold_cys
paper_targets.models
claim_status
claim_status.favorable_polytopes
claim_status.frst_classes
claim_status.h21_plus_zero
details
```
The four `counts.source_vertex_evidence_*` and `counts.kaehler_point_export_*`
keys are always present. Their values are `null` when the related flag is off.
`details` is `null` when `--keep-details` is off.

#### `BASE_2_ADD` (added from `aee4c1a`; 2 paths)
```
claim_status.model_count
model_stage
```
Both keys are always present. Each is `null` when `--model-stage` is off.

#### `BASE_3_ADD` (added from `4d68038`; 1 path)
```
counts.h21_plus_zero_trilayer_frst_classes
```

#### `PT_CHILDREN` (6 paths; these are in R unless `paper_targets` is `null`)
```
paper_targets.favorable_polytopes
paper_targets.frst_classes
paper_targets.inherited_orientifold_cys
paper_targets.h11_minus_zero_orientifold_cys
paper_targets.h11_minus_zero_h21_plus_zero_orientifold_cys
paper_targets.models
```
(`PT_CHILDREN` is a subset of `BASE_1`. This set is named only so that the
`efdcbce`/`47b1d74` `--h11` delta can remove it.)

#### `KD_R_1` (keep-details element keys, `ec73c15`-`47b1d74`; 23 paths)
```
details[].polytope_index
details[].provenance
details[].provenance.source_kind
details[].provenance.dataset
details[].provenance.dataset_url
details[].provenance.parquet_file
details[].provenance.row_index
details[].provenance.mirror_h11
details[].provenance.mirror_h12
details[].provenance.physical_h11
details[].provenance.physical_h21
details[].provenance.vertex_count
details[].provenance.favorable_checked_by
details[].raw_frst_count
details[].frst_class_count
details[].trilayer
details[].frozen_conifold
details[].frozen_conifold_per_class
details[].nonfrozen_class_count
details[].identity_torus_action_numerators
details[].identity_valid_o3o7_action_numerators
details[].orientifold_action_audit
details[].kaehler_point_export_per_class
```
`identity_torus_action_numerators` and `identity_valid_o3o7_action_numerators`
are lists of integer lists (`actions.tolist()`). They add no path below them.

#### `KD_O_1` (keep-details optional paths, `ec73c15`-`47b1d74`; 23 paths)
```
details[].trilayer.p0
details[].trilayer.q0
details[].trilayer.lattice_matrix
details[].trilayer.torus_shift_numerator
details[].trilayer.torus_shift_denominator
details[].trilayer.lambda_f
details[].trilayer.criterion
details[].frozen_conifold.status
details[].frozen_conifold.reason
details[].frozen_conifold.surface_records
details[].frozen_conifold.surface_records[].rays
details[].frozen_conifold.surface_records[].frozen_conifold_count
details[].frozen_conifold.surface_records[].formula
details[].frozen_conifold.unavailable_cones
details[].frozen_conifold.frozen_surface_count
details[].frozen_conifold_per_class[].status
details[].frozen_conifold_per_class[].reason
details[].frozen_conifold_per_class[].surface_records
details[].frozen_conifold_per_class[].surface_records[].rays
details[].frozen_conifold_per_class[].surface_records[].frozen_conifold_count
details[].frozen_conifold_per_class[].surface_records[].formula
details[].frozen_conifold_per_class[].unavailable_cones
details[].frozen_conifold_per_class[].frozen_surface_count
```
Why each path is optional (citations per commit are in each group):
- `trilayer.*`: `_trilayer_candidate` returns `None` when no vertex passes the
  test. Otherwise it returns the 7-key dict.
- `frozen_conifold` and `frozen_conifold_per_class` stay `None` unless a trilayer
  exists. `frozen_conifold` is also `None` when the polytope has no FRST class
  (`frozen_per_class[0] if frozen_per_class else None`).
- `status` and `surface_records` are present in both return variants of
  `_frozen_conifold_diagnostic`. They are in O only because their parent can be
  `null` in every element.
- `reason` comes only from the "tensor does not match fan rays" variant.
- `unavailable_cones` and `frozen_surface_count` come only from the main return.
- `surface_records[].*` exist only when the list is not empty. It is `[]` in the
  mismatch variant, and also when no smooth parity-passing 2-cone exists.
- `unavailable_cones` is a list of ray-point lists, so it adds no path below it.

#### `KD_R_3_ADD` (from `4d68038`; 2 paths)
```
details[].h21_plus_zero_per_class
details[].h21_plus_zero_class_count
```

#### `KD_O_3_ADD` (from `4d68038`; 10 paths)
```
details[].h21_plus_zero_per_class[].status
details[].h21_plus_zero_per_class[].reasons
details[].h21_plus_zero_per_class[].chi_F_I
details[].h21_plus_zero_per_class[].h21_minus
details[].h21_plus_zero_per_class[].h21_plus
details[].h21_plus_zero_per_class[].components
details[].h21_plus_zero_per_class[].components[].dim
details[].h21_plus_zero_per_class[].components[].rays
details[].h21_plus_zero_per_class[].components[].chi
details[].h21_plus_zero_per_class[].components[].kind
```
- `h21_plus_zero_per_class` is `None` unless a trilayer exists. When it exists, it
  is a list with one element per class, which is empty if there are no classes.
- Both return variants of `_h21_plus_zero_diagnostic` carry all six keys.
- `reasons` is a list of strings.
- `components` can be `[]` (early unavailable return), or a partial or full list
  of 4-key dicts. `components[].rays` is a tuple of ints, which becomes a list of
  scalars.

#### `AUDIT_R` (`--orientifold-audit` with `--keep-details`; 3 paths)
```
details[].orientifold_action_audit.inherited
details[].orientifold_action_audit.h11_minus_zero
details[].orientifold_action_audit.h11_minus_zero_classes
```
- The audit runs for every polytope, not only trilayer polytopes. Both return
  variants (empty-classes early return and main return) carry all three keys.
  For this reason these paths are R.
- `h11_minus_zero_classes` is a list of ints.

#### `EXPORT_O` (effective Kähler export with `--keep-details`; 50 paths)
```
details[].kaehler_point_export_per_class[].status
details[].kaehler_point_export_per_class[].reason
details[].kaehler_point_export_per_class[].h11
details[].kaehler_point_export_per_class[].point
details[].kaehler_point_export_per_class[].cy_volume
details[].kaehler_point_export_per_class[].prime_divisor_volumes
details[].kaehler_point_export_per_class[].inverse_metric
details[].kaehler_point_export_per_class[].glsm_charge_matrix
details[].kaehler_point_export_per_class[].potential_matrix_convention
details[].kaehler_point_export_per_class[].potential_matrix_convention.Q
details[].kaehler_point_export_per_class[].diagnostic
details[].kaehler_point_export_per_class[].diagnostic.attempt_index
details[].kaehler_point_export_per_class[].diagnostic.point_kind
details[].kaehler_point_export_per_class[].diagnostic.point_seed
details[].kaehler_point_export_per_class[].diagnostic.attempted
details[].kaehler_point_export_per_class[].diagnostic.solver
details[].kaehler_point_export_per_class[].diagnostic.point_status
details[].kaehler_point_export_per_class[].diagnostic.checks
details[].kaehler_point_export_per_class[].diagnostic.divisor_volume_tolerance
details[].kaehler_point_export_per_class[].diagnostic.prime_divisor_volume_lower_bound
details[].kaehler_point_export_per_class[].diagnostic.effective_divisor_volume_lower_bound
details[].kaehler_point_export_per_class[].diagnostic.divisor_volume_lower_bounds_enforced
details[].kaehler_point_export_per_class[].diagnostic.volume_backend
details[].kaehler_point_export_per_class[].diagnostic.coordinate_dimension
details[].kaehler_point_export_per_class[].diagnostic.point_norm
details[].kaehler_point_export_per_class[].diagnostic.point_sha256
details[].kaehler_point_export_per_class[].diagnostic.minimum_kaehler_slack
details[].kaehler_point_export_per_class[].diagnostic.cy_volume
details[].kaehler_point_export_per_class[].diagnostic.minimum_curve_volume
details[].kaehler_point_export_per_class[].diagnostic.minimum_basis_divisor_volume
details[].kaehler_point_export_per_class[].diagnostic.minimum_prime_divisor_volume
details[].kaehler_point_export_per_class[].diagnostic.minimum_effective_divisor_volume
details[].kaehler_point_export_per_class[].diagnostic.minimum_metric_eigenvalue
details[].kaehler_point_export_per_class[].diagnostic.failure_reason
details[].kaehler_point_export_per_class[].diagnostic.checks.finite_coordinates
details[].kaehler_point_export_per_class[].diagnostic.checks.cone_membership
details[].kaehler_point_export_per_class[].diagnostic.checks.finite_cy_volume
details[].kaehler_point_export_per_class[].diagnostic.checks.positive_cy_volume
details[].kaehler_point_export_per_class[].diagnostic.checks.finite_curve_volumes
details[].kaehler_point_export_per_class[].diagnostic.checks.positive_curve_volumes
details[].kaehler_point_export_per_class[].diagnostic.checks.finite_basis_divisor_volumes
details[].kaehler_point_export_per_class[].diagnostic.checks.positive_basis_divisor_volumes
details[].kaehler_point_export_per_class[].diagnostic.checks.finite_prime_divisor_volumes
details[].kaehler_point_export_per_class[].diagnostic.checks.positive_prime_divisor_volumes
details[].kaehler_point_export_per_class[].diagnostic.checks.prime_divisor_volume_lower_bound
details[].kaehler_point_export_per_class[].diagnostic.checks.finite_effective_divisor_volumes
details[].kaehler_point_export_per_class[].diagnostic.checks.positive_effective_divisor_volumes
details[].kaehler_point_export_per_class[].diagnostic.checks.effective_divisor_volume_lower_bound
details[].kaehler_point_export_per_class[].diagnostic.checks.finite_inverse_metric
details[].kaehler_point_export_per_class[].diagnostic.checks.positive_inverse_metric
```
Why every path is optional:
- `kaehler_point_export_per_class` is `None` unless a trilayer exists. When it
  exists, it has one element per class. An element is `None` (no path) for a
  class that fails the population gate, an "accepted" dict, or a "rejected" dict.
- An accepted element has the keys `status`, `h11`, `point`, `cy_volume`,
  `prime_divisor_volumes`, `inverse_metric`, `glsm_charge_matrix`,
  `potential_matrix_convention{Q}` and `diagnostic`.
- A rejected element has the keys `status`, `reason` and `diagnostic`.
- `diagnostic` keys (`GEN`):
  - Always present: the 12 initial keys at `GEN`:1496-1511.
  - Set only when the try block reaches them: `coordinate_dimension`,
    `point_norm`, `point_sha256` (`GEN`:1515-1517) and `minimum_kaehler_slack`
    (`GEN`:1523), plus `cy_volume` and the five `minimum_*` keys
    (`GEN`:1553-1580).
  - `failure_reason` is set only on exception (`GEN`:1662-1665).
- `checks.*`:
  - `finite_coordinates` and `cone_membership` are set at `GEN`:1524-1528.
  - The other 14 are set at `GEN`:1581-1636 only after the volume geometry
    succeeds.
  - On an early exception `checks` stays `{}` (`GEN`:1664-1665).
- An accepted element always has the full diagnostic minus `failure_reason`.
  Because every element can be `None`, every path is still O.
- `point`, `prime_divisor_volumes` and `inverse_metric` are numeric lists, so
  they add no path below them.

#### `MS_R` (`--model-stage`; 11 paths)
```
model_stage.input_record_count
model_stage.total_model_count
model_stage.model_count_per_record
model_stage.models
model_stage.convention
model_stage.convention.gs
model_stage.convention.gs_justification
model_stage.convention.w0_real
model_stage.convention.w0_imag
model_stage.convention.w0_justification
model_stage.diagnostic_reason
```

#### `MS_O` (`--model-stage`; 1 path)
```
model_stage.convention.prefactor_P
```
This path is absent in the early return for zero model-stage input records, and
added only after the Julia run.

#### `MS_KD_O` (`--model-stage` with `--keep-details`; 6 paths)
```
model_stage.models[].record_index
model_stage.models[].axion_index
model_stage.models[].qcd_divisor_index
model_stage.models[].lambda
model_stage.models[].mass_reference_log10_ev
model_stage.models[].tau_reference
```
- `models` is `None` without `--keep-details`.
- With `--keep-details`, `models` is `[]` in the zero-record early return, or a
  list built from the driver output. That list is empty when the driver finds no
  models.
- `model_count_per_record` is a list of ints and adds no path below it.

---

#### 1.0 group 1: commits ec73c15

Writer: `ec73c15:scripts/reproduce_fuzzy_axions_h11_4.py`. `summary` is built at
578-624 and returned through `_jsonable` (460-469) at 625.

**Base R** = `BASE_1` (36 paths). Citations:
- `schema_version`, `paper`, `orientifold_source`: `ec73c15:scripts/reproduce_fuzzy_axions_h11_4.py:579-581`
- `input.*`: `ec73c15:scripts/reproduce_fuzzy_axions_h11_4.py:582-589`
- `counts.*`: `ec73c15:scripts/reproduce_fuzzy_axions_h11_4.py:590-612`
- `paper_targets.*`: `ec73c15:scripts/reproduce_fuzzy_axions_h11_4.py:613` (literal `PAPER_TARGETS`, 41-48)
- `claim_status.*`: `ec73c15:scripts/reproduce_fuzzy_axions_h11_4.py:614-622`
- `details`: `ec73c15:scripts/reproduce_fuzzy_axions_h11_4.py:623` (`details if args.keep_details else None`)

**Base O**: none. Without `--keep-details`, every key is unconditional.

**Map nodes**: none.

**Flag deltas**
- `--keep-details` (`ec73c15`:633; gate at 623):
  - adds `KD_R_1` to R. The element literal is at `ec73c15:...:553-568`, and
    `provenance` comes from `GEN`:4078-4092.
  - adds `KD_O_1` to O. Optionality citations:
    - `trilayer` `None` return: `ec73c15`:113. The dict is at 104-112.
    - `frozen` and `frozen_per_class` are initialised to `None` at 504-505 and
      set only inside `if trilayer is not None`, 508-530. Line 530 is
      `frozen_per_class[0] if frozen_per_class else None`.
    - Frozen-diagnostic variants: mismatch return 172-176 (`status`, `reason`,
      `surface_records=[]`), main return 226-231. Records are at 211-217.
- `--orientifold-audit` (`ec73c15`:634; gate 531-534): adds `AUDIT_R` to R.
  Both returns of `_orientifold_action_audit` have the three keys (310 and
  336-340). The flag has no path effect without `--keep-details`:
  `counts.source_vertex_evidence_*` (600-605) are always present.
- `--export-kaehler-points` (`ec73c15`:635-644; gate 536-552): adds `EXPORT_O`
  to O.
  - The gate is `args.export_kaehler_points and frozen_per_class is not None`.
    Non-`no_frozen_conifold_obstruction` classes get a `None` element (544-546).
  - Element variants: rejected 438-444, accepted 445-457.
  - The flag has no path effect without `--keep-details`:
    `counts.kaehler_point_export_*` (606-611) are always present.

**Interactions**
- `--orientifold-audit` and `--export-kaehler-points` change the path set only
  together with `--keep-details`. The argparse help (639-643) says this for
  export.
- The two flags are independent of each other, and neither excludes the other.

**Structure-neutral flags**: `--parquet-dir`, `--limit`, `--progress`, `--output`.
Without `--keep-details`, `--orientifold-audit` and `--export-kaehler-points` are
also neutral. (`--limit` changes only the value of `input.population_complete`,
at 588.)

**K-dependent fields**: none. This commit has no `model_stage` block and no
`claim_status.model_count`. The stored `details[].kaehler_point_export_per_class[].inverse_metric`
is the CYTools `Kinv` and is K-free under A2.

**Layout count: 5**

| id | flags | R | O |
|---|---|---|---|
| 1.0-a-L1 | no `--keep-details` (any audit/export) | `BASE_1` | none |
| 1.0-a-L2 | `--keep-details` | `BASE_1` + `KD_R_1` | `KD_O_1` |
| 1.0-a-L3 | `--keep-details --orientifold-audit` | `BASE_1` + `KD_R_1` + `AUDIT_R` | `KD_O_1` |
| 1.0-a-L4 | `--keep-details --export-kaehler-points` | `BASE_1` + `KD_R_1` | `KD_O_1` + `EXPORT_O` |
| 1.0-a-L5 | `--keep-details --orientifold-audit --export-kaehler-points` | `BASE_1` + `KD_R_1` + `AUDIT_R` | `KD_O_1` + `EXPORT_O` |

---

#### 1.0 group 2: commits aee4c1a

Writer: `aee4c1a:scripts/reproduce_fuzzy_axions_h11_4.py`. `summary` is at 742-798
and returned through `_jsonable` (586-595) at 799. The new Julia driver is
`aee4c1a:scripts/fuzzy_axion_model_stage_driver.jl`.

**What changed from group 1**
- `claim_status.model_count` (786-794) and `model_stage` (796) are new
  always-present keys (`BASE_2_ADD`).
- There is a new `--model-stage` flag (819-832), which implies Kähler export:
  `export_kaehler_points = args.export_kaehler_points or args.model_stage` (618).
- New `model_stage` children (`_run_model_stage`, 491-583).
- The `details[]` element literal (692-707) and all helper return shapes are
  unchanged.

**Base R** = `BASE_1` + `BASE_2_ADD` (38 paths). Citations:
- top scalars: `aee4c1a:scripts/reproduce_fuzzy_axions_h11_4.py:743-745`
- `input.*`: `aee4c1a:...:746-753`
- `counts.*`: `aee4c1a:...:754-776`
- `paper_targets.*`: `aee4c1a:...:777` (literal 44-51)
- `claim_status.*`: `aee4c1a:...:778-795`
- `model_stage`: `aee4c1a:...:796` (`null` unless `--model-stage`, 722)
- `details`: `aee4c1a:...:797`

**Base O**: none.

**Map nodes**: none.

**Flag deltas**
- `--keep-details` (807; 797): adds `KD_R_1` to R (element 692-707) and
  `KD_O_1` to O. Citations:
  - trilayer `None` 116, dict 107-115;
  - `None` initialisation 632-633;
  - trilayer block 636-658;
  - frozen variants 175-179 and 229-234, records 214-220.
- `--orientifold-audit` (808; 659-662): adds `AUDIT_R` to R (returns 313 and
  339-343). This applies only with `--keep-details`.
- Effective export E = `--export-kaehler-points` (809-818) OR `--model-stage`
  (618). With `--keep-details` it adds `EXPORT_O` to O. Citations: gate 664-691,
  `None` element 672-674, export returns 441-447 (rejected) and 448-460
  (accepted).
- `--model-stage` (819-832; 722):
  - adds `MS_R` to R. Citations: zero-record return 518-528, main return
    576-583, convention dict 500-516.
  - adds `MS_O` to O. `prefactor_P` is set only at 575, after the Julia run.
    The early return 518-528 omits it.
  - With `--keep-details` it also adds `MS_KD_O` to O. Citations: `models`
    `[]`/`None` at 523, list at 563-573, `None` without keep-details at 555-556.
  - It also triggers E (see Interactions).

**Interactions**
- `--model-stage` implies E (618). So `--model-stage --keep-details` adds
  `EXPORT_O`, whether `--export-kaehler-points` is set or not.
  `--model-stage --export-kaehler-points` = `--model-stage`.
- `--orientifold-audit` and E change paths only with `--keep-details`.
- `MS_KD_O` needs both `--model-stage` and `--keep-details`.
- No flags are mutually exclusive.

**Structure-neutral flags**: `--parquet-dir`, `--limit`, `--progress`, `--output`,
`--gs` (833-839), `--w0-real` (840-847), `--w0-imag` (848), `--julia-binary`
(849-853), `--julia-project` (854-860). Without `--keep-details`,
`--orientifold-audit` and `--export-kaehler-points` are also neutral.

**K-dependent fields** (written only with `--model-stage`):
- Computed by the Julia driver, which reads `inverse_metric` from HDF5
  (`aee4c1a:scripts/fuzzy_axion_model_stage_driver.jl`:67,75):
  - `model_stage.total_model_count` (552)
  - `model_stage.model_count_per_record` (553)
  - `model_stage.models[].record_index`, `.axion_index`, `.qcd_divisor_index`,
    `.lambda`, `.mass_reference_log10_ev`, `.tau_reference` (557-573). These
    are only present with `--keep-details`.
- Derived from them:
  - `claim_status.model_count` (786-794)
  - `model_stage.diagnostic_reason` when there is at least one input record
    (723-740; it quotes `total_model_count`). In the zero-record early return it
    is a fixed K-free string (525-527).
- Julia-computed but K-free: `model_stage.convention.prefactor_P` (554, 575).
  The driver computes it as `fuzzy_axion_prefactor_P(gs)` (driver:45), which
  is `gs^4/128` (see Open point 7).
- K-free: `model_stage.input_record_count`, the `convention.gs`/`w0_*` inputs
  and their justification strings, all `counts`, the other `claim_status`
  entries, and all of `details` (this includes the stored `inverse_metric`).

**Layout count: 8**

| id | flags | R | O |
|---|---|---|---|
| 1.0-a-L6 | no `--keep-details`, no `--model-stage` | `BASE_1`+`BASE_2_ADD` | none |
| 1.0-a-L7 | `--model-stage`, no `--keep-details` | `BASE_1`+`BASE_2_ADD`+`MS_R` | `MS_O` |
| 1.0-a-L8 | `--keep-details` | `BASE_1`+`BASE_2_ADD`+`KD_R_1` | `KD_O_1` |
| 1.0-a-L9 | `--keep-details --orientifold-audit` | `BASE_1`+`BASE_2_ADD`+`KD_R_1`+`AUDIT_R` | `KD_O_1` |
| 1.0-a-L10 | `--keep-details --export-kaehler-points` | `BASE_1`+`BASE_2_ADD`+`KD_R_1` | `KD_O_1`+`EXPORT_O` |
| 1.0-a-L11 | `--keep-details --orientifold-audit --export-kaehler-points` | `BASE_1`+`BASE_2_ADD`+`KD_R_1`+`AUDIT_R` | `KD_O_1`+`EXPORT_O` |
| 1.0-a-L12 | `--keep-details --model-stage` (± `--export-kaehler-points`) | `BASE_1`+`BASE_2_ADD`+`KD_R_1`+`MS_R` | `KD_O_1`+`EXPORT_O`+`MS_O`+`MS_KD_O` |
| 1.0-a-L13 | `--keep-details --model-stage --orientifold-audit` (± export) | `BASE_1`+`BASE_2_ADD`+`KD_R_1`+`AUDIT_R`+`MS_R` | `KD_O_1`+`EXPORT_O`+`MS_O`+`MS_KD_O` |

---

#### 1.0 group 3: commits 4d68038

Writer: `4d68038:scripts/reproduce_fuzzy_axions_h11_4.py`. `summary` is at
951-1010 and returned at 1011.

**What changed from group 2**
- New R key `counts.h21_plus_zero_trilayer_frst_classes` (972; `BASE_3_ADD`).
- New `details[]` keys `h21_plus_zero_per_class` and `h21_plus_zero_class_count`
  (908-909; `KD_R_3_ADD`), with children `KD_O_3_ADD`. These are built in
  `_h21_plus_zero_diagnostic` (382-420) and `_fixed_locus_euler_characteristic`
  (315-379).
- The Kähler-export gate changed from frozen-conifold status to h21 status
  (870-897; 878: `h21_per_class[class_index]["status"] != "h21_plus_zero"`).
  This changes values, not paths.
- The `model_stage` and `claim_status` shapes are unchanged.

**Base R** = `BASE_1` + `BASE_2_ADD` + `BASE_3_ADD` (39 paths). Citations:
- top scalars: `4d68038:scripts/reproduce_fuzzy_axions_h11_4.py:952-954`
- `input.*`: `4d68038:...:955-962`
- `counts.*`: `4d68038:...:963-988`
- `paper_targets.*`: `4d68038:...:989` (literal 48-55)
- `claim_status.*`: `4d68038:...:990-1007`
- `model_stage`: `4d68038:...:1008` (931)
- `details`: `4d68038:...:1009`

**Base O**: none.

**Map nodes**: none.

**Flag deltas**
- `--keep-details` (1019; 1009): adds `KD_R_1` + `KD_R_3_ADD` to R (element
  898-915). Adds `KD_O_1` + `KD_O_3_ADD` to O. Citations:
  - trilayer `None` 120, dict 111-119;
  - `None` initialisation 819-822;
  - trilayer block 825-864, with `frozen_per_class` at 838-841, `frozen` at 848
    and `h21_per_class` at 857-860;
  - frozen variants 179-183 and 233-238, records 218-224;
  - h21 variants 398-405 (unavailable) and 413-420;
  - `components` lists: `[]` at 337, partial list at 360-365, full list at 379.
    Components dicts are at 355, 372 and 376.
- `--orientifold-audit` (1020; 865-868): adds `AUDIT_R` to R (returns 499 and
  525-529). This applies only with `--keep-details`.
- Effective export E (1021-1030 OR `--model-stage`, 805): with `--keep-details`
  it adds `EXPORT_O` to O. Citations: gate 870-897, `None` element 878-880,
  returns 627-633 and 634-646.
- `--model-stage` (1031-1044; 931): adds `MS_R` to R and `MS_O` to O.
  Citations: convention 686-702, early return 704-714 (no `prefactor_P`),
  `prefactor_P` 761, return 762-769. With `--keep-details` it adds `MS_KD_O`
  (models 749-759; `[]` at 709).

**Interactions**: the same as group 2 (E = export OR model-stage, 805).

**Structure-neutral flags**: the same as group 2 (`--gs` 1045-1051, `--w0-real`
1052-1059, `--w0-imag` 1060, `--julia-binary` 1061-1065, `--julia-project`
1066-1072, `--parquet-dir`, `--limit`, `--progress`, `--output`).

**K-dependent fields**: the same paths as group 2. Line citations at this
commit:
- `total_model_count` / `model_count_per_record` / `prefactor_P`: 738-740
- `models[]`: 742-759
- `diagnostic_reason`: derivation 932-949
- `claim_status.model_count`: 998-1006

The new `details[].h21_plus_zero_per_class[]` fields and
`counts.h21_plus_zero_trilayer_frst_classes` are K-free: they are topological
(χ, h21).

**Layout count: 8.** The table matches group 2 with these replacements:
`BASE_1`+`BASE_2_ADD` → `BASE_1`+`BASE_2_ADD`+`BASE_3_ADD`, `KD_R_1` →
`KD_R_1`+`KD_R_3_ADD`, and `KD_O_1` → `KD_O_1`+`KD_O_3_ADD`.

| id | flags | R | O |
|---|---|---|---|
| 1.0-a-L14 | no `--keep-details`, no `--model-stage` | B3 | none |
| 1.0-a-L15 | `--model-stage`, no `--keep-details` | B3+`MS_R` | `MS_O` |
| 1.0-a-L16 | `--keep-details` | B3+K3R | K3O |
| 1.0-a-L17 | `--keep-details --orientifold-audit` | B3+K3R+`AUDIT_R` | K3O |
| 1.0-a-L18 | `--keep-details --export-kaehler-points` | B3+K3R | K3O+`EXPORT_O` |
| 1.0-a-L19 | `--keep-details --orientifold-audit --export-kaehler-points` | B3+K3R+`AUDIT_R` | K3O+`EXPORT_O` |
| 1.0-a-L20 | `--keep-details --model-stage` (± export) | B3+K3R+`MS_R` | K3O+`EXPORT_O`+`MS_O`+`MS_KD_O` |
| 1.0-a-L21 | `--keep-details --model-stage --orientifold-audit` (± export) | B3+K3R+`AUDIT_R`+`MS_R` | K3O+`EXPORT_O`+`MS_O`+`MS_KD_O` |

where B3 = `BASE_1`+`BASE_2_ADD`+`BASE_3_ADD` (39), K3R = `KD_R_1`+`KD_R_3_ADD`
(25) and K3O = `KD_O_1`+`KD_O_3_ADD` (33).

---

#### 1.0 group 4: commits efdcbce, 47b1d74

Writers: `efdcbce:scripts/reproduce_fuzzy_axions_h11_4.py` (`summary` at 923-991,
returned at 992) and `47b1d74:scripts/reproduce_fuzzy_axions_h11_4.py` (`summary`
at 947-1015, returned at 1016).

**What changed from group 3**
- A new `--h11` flag (`efdcbce`:998-1004, `47b1d74`:1022-1028; default 4).
  `targets = PAPER_TARGETS_BY_H11.get(args.h11)` (`efdcbce`:752, `47b1d74`:776),
  and `"paper_targets": targets` (`efdcbce`:961, `47b1d74`:985). So
  `paper_targets` is `null` when h11 has no recorded target, and `PT_CHILDREN`
  are then absent.
- `claim_status` now has two literals (`efdcbce`:962-988, `47b1d74`:986-1012).
  Both have the same four keys, so no path changes.
- `input.requested_h11` is now `args.h11` (value only).
- `_orientifold_action_audit` was rewritten (`efdcbce`:430-496,
  `47b1d74`:454-520). Its persisted return shape is unchanged.
- No other key changed. The `details[]` literal is at `efdcbce`:866-883 and
  `47b1d74`:890-907.
- **The only difference between `efdcbce` and `47b1d74`** is the target table.
  At `efdcbce`:53-77 it has keys {3, 4, 5}. At `47b1d74`:53-102 it has keys
  {2, 3, 4, 5, 6, 7}. This changes which `--h11` values trigger the
  "no targets" delta, not any path set. The `--h11` help text at `47b1d74`
  still says "3, 4, and 5" (1026-1027); this is stale and has no effect.

**Base R** (h11 has a target) = `BASE_1` + `BASE_2_ADD` + `BASE_3_ADD` (39
paths). This is identical to group 3. Citations (efdcbce / 47b1d74):
- top scalars: `efdcbce:...:924-926` / `47b1d74:...:948-950`
- `input.*`: `efdcbce:...:927-934` / `47b1d74:...:951-958`
- `counts.*`: `efdcbce:...:935-960` / `47b1d74:...:959-984`
- `paper_targets.*`: `efdcbce:...:961` (table 53-77) / `47b1d74:...:985` (table 53-102)
- `claim_status.*`: `efdcbce:...:962-988` / `47b1d74:...:986-1012`
- `model_stage`: `efdcbce:...:989` (900) / `47b1d74:...:1013` (924)
- `details`: `efdcbce:...:990` / `47b1d74:...:1014`

**Base O**: none.

**Map nodes**: none.

**Flag deltas** (citations efdcbce / 47b1d74)
- `--h11 N` with `N` not a key of `PAPER_TARGETS_BY_H11` (efdcbce: N ∉ {3,4,5};
  47b1d74: N ∉ {2,3,4,5,6,7}): **removes `PT_CHILDREN` from R**. `paper_targets`
  stays in R with value `null`. Citations: `efdcbce`:752,961 / `47b1d74`:776,985.
  The `claim_status` keys are unchanged; all four are `null` (`efdcbce`:963-968
  / `47b1d74`:987-992).
- `--keep-details` (1007 / 1031): adds K3R to R and K3O to O. Citations:
  - element 866-883 / 890-907;
  - trilayer 134-142, `None` 143 / 158-166, `None` 167;
  - `None` initialisation 787-790 / 811-814;
  - trilayer block 793-832 / 817-856;
  - frozen variants 186-190 and 240-245, records 225-231 / 210-214 and
    264-269, records 249-255;
  - h21 variants 405-412 and 420-427 / 429-436 and 444-451;
  - components `[]` 344, partial 367-372, full 386 / 368, 391-396, 410.
- `--orientifold-audit` (1008 / 1032; gate 833-836 / 857-860): adds `AUDIT_R`
  to R (returns 469, 492-496 / 493, 516-520). This applies only with
  `--keep-details`.
- Effective export E (`--export-kaehler-points` 1009-1018 / 1033-1042 OR
  `--model-stage`; 773 / 797): with `--keep-details` it adds `EXPORT_O` to O.
  Citations: gate 838-865 / 862-889, returns 594-600 and 601-613 / 618-624 and
  625-637.
- `--model-stage` (1019-1032 / 1043-1056): adds `MS_R` to R and `MS_O` to O.
  Citations: convention 653-669 / 677-693, early return 671-681 / 695-705,
  `prefactor_P` 728 / 752, return 729-736 / 753-760. With `--keep-details` it
  adds `MS_KD_O` (models 716-726 / 740-750).

**Interactions**
- The same as group 2: E = export OR model-stage, and audit/export need
  `--keep-details`.
- The `--h11` "no targets" delta is independent of all other flags. It is the
  only delta that removes paths from R.
- `--h11` also changes `load_mirror_polytopes(h11=args.h11)` (`efdcbce`:753-758
  / `47b1d74`:777-782). This changes the population and values only.

**Structure-neutral flags**: `--parquet-dir`, `--limit`, `--progress`, `--output`,
`--gs`, `--w0-real`, `--w0-imag`, `--julia-binary`, `--julia-project`. Also
`--h11` with any value that has a target. Without `--keep-details`,
`--orientifold-audit` and `--export-kaehler-points` are also neutral.

**K-dependent fields**: the same paths as group 2. Citations (efdcbce /
47b1d74):
- `total_model_count`, `model_count_per_record`, `prefactor_P`: 705-707 /
  729-731
- `models[]`: 709-726 / 733-750
- `diagnostic_reason`: derivation 901-921 / 925-945
- `claim_status.model_count`: 978-986 / 1002-1010

When targets are `null`, `claim_status.model_count` is `null` even with
`--model-stage`. This is a value change, not a path change. The Julia driver is
unchanged since `aee4c1a`.

**Layout count: 16** = 8 (targets present) + 8 (targets absent).
- **Targets present (T):** the 8 layouts have path sets identical to group 3's
  L14-L21. Under A2 ("identical path sets are one layout listing all their
  commits and flags"), these are **the same layouts as 1.0-a-L14…L21**. Those
  layouts then list commits `4d68038`, `efdcbce`, `47b1d74`; for the last two
  the condition is `--h11` ∈ the commit's target set.
- **Targets absent (N):** 8 new layouts. Each is the matching T layout with
  `PT_CHILDREN` removed from R:

| id | flags (all with `--h11` lacking a target) | R | O |
|---|---|---|---|
| 1.0-a-L22 | no `--keep-details`, no `--model-stage` | B3 − `PT_CHILDREN` (33) | none |
| 1.0-a-L23 | `--model-stage` | B3 − PT + `MS_R` | `MS_O` |
| 1.0-a-L24 | `--keep-details` | B3 − PT + K3R | K3O |
| 1.0-a-L25 | `--keep-details --orientifold-audit` | B3 − PT + K3R + `AUDIT_R` | K3O |
| 1.0-a-L26 | `--keep-details --export-kaehler-points` | B3 − PT + K3R | K3O + `EXPORT_O` |
| 1.0-a-L27 | `--keep-details --orientifold-audit --export-kaehler-points` | B3 − PT + K3R + `AUDIT_R` | K3O + `EXPORT_O` |
| 1.0-a-L28 | `--keep-details --model-stage` (± export) | B3 − PT + K3R + `MS_R` | K3O + `EXPORT_O` + `MS_O` + `MS_KD_O` |
| 1.0-a-L29 | `--keep-details --model-stage --orientifold-audit` (± export) | B3 − PT + K3R + `AUDIT_R` + `MS_R` | K3O + `EXPORT_O` + `MS_O` + `MS_KD_O` |

**Distinct layouts over this record's range (`ec73c15`…`47b1d74`): 29**
(group 1: 5, group 2: 8, groups 3+4 (T): 8, group 4 (N): 8).

---

### Open points

1. **Bare `[]` path.** A2 says "an array collapses to one path segment `[]`" and
   that empty or scalar arrays add "no path below `[]`". It does not say whether
   `k[]` is itself a path. I list `k` (the key) and `k[].child`, never a bare
   `k[]`. If the reader emits `k[]`, it must also be added. For arrays of
   objects, `k[]` would be present exactly when some `k[].child` is present, so
   it goes in the same set (R or O) as those children. For arrays of scalars,
   the rule must decide whether `k[]` appears when the array is not empty.
2. **Map nodes.** I found none. `paper_targets`, `counts`, `claim_status`,
   `details[].provenance` and `...diagnostic.checks` have fixed field-name keys.
   A reviewer who classifies `diagnostic.checks` as a map (its keys are check
   names) would replace the 16 `checks.*` paths with one
   `...diagnostic.checks.<map>` path.
3. **Empty population with `--keep-details`** (`details == []`, and with
   `--model-stage`, `models == []`). The path set equals that of the matching
   non-`--keep-details` layout (L1, L6/L7, L14/L15, L22/L23), so such a file
   matches only that layout. I put the `details[]` element keys in R, not O. The
   other choice (put them in O) gives the same accept/reject result, because the
   reader accepts any match.
4. **Files that match more than one layout in this range.**
   - Within each group, the export-on layouts (L4/L5, L10/L11, L18/L19, L26/L27)
     have the same R as their export-off pair (L2/L3, L8/L9, L16/L17, L24/L25)
     and a superset O. Any file that matches the export-off layout also matches
     the export-on layout. This happens for example when no class passes the
     population gate, so every `kaehler_point_export_per_class` is `null` or
     all-`null`.
   - The `--model-stage` layouts differ from the non-model-stage layouts by
     `MS_R` in R, so they are disjoint.
   - The audit-on and audit-off layouts are disjoint for non-empty `details`.
   - Group 1 is disjoint from all later layouts: `model_stage` and
     `claim_status.model_count` are in every later R and in no group-1 R∪O.
   - Group 2 is disjoint from groups 3 and 4: `counts.h21_plus_zero_trilayer_frst_classes`
     is in their R and not in group 2's R∪O. The `details[].h21_*` keys are
     also R in groups 3 and 4 with `--keep-details`.
   - Group 3 and the targets-present half of group 4 are the **same** layouts
     (identical R and O).
   - The targets-absent layouts are disjoint from the others: `PT_CHILDREN` is
     in every other R and not in their R∪O.
   - All 1.0 layouts give the same inferred label, so these overlaps do not
     change the reader's result.
5. **Grouping of `efdcbce` and `47b1d74`.** Both commits have identical path
   sets, deltas and interactions. They differ only in which `--h11` values
   select the targets-absent delta (efdcbce {3,4,5}; 47b1d74 {2,…,7}). I put
   them in one group. If the reviewers want the trigger condition to be part of
   the "delta", split them into two groups with identical layouts.
6. **Non-strict JSON numbers.** `json.dumps` is called with the default
   `allow_nan=True`. Some diagnostic fields can be `Infinity` or `NaN`:
   - `...diagnostic.minimum_*` fall back to `math.inf` for empty arrays
     (`GEN`:1522, 1555-1579);
   - `minimum_kaehler_slack` can be `inf`;
   - non-finite volumes can appear in rejected records.

   A strict JSON parser in the reader could fail on these files before the
   layout is checked. This is not a layout question, but the reader must parse
   such tokens.
7. **`model_stage.convention.prefactor_P`** is read from the Julia output
   (`aee4c1a`:554). The driver computes it from `gs` only
   (`fuzzy_axion_prefactor_P(gs)`, driver:45; help text "P = gs^4/128"). I
   list it under "Julia-computed" but mark it K-free. The owner must decide
   whether the label's "K-dependent fields" list includes it.
8. **`model_stage.models[].record_index`, `.axion_index` and
   `.qcd_divisor_index`** are integers. Which `(record, axion, divisor)` tuples
   appear depends on the K-dependent model enumeration, so I list them as
   K-dependent. A narrower reading would list only `lambda`,
   `mass_reference_log10_ev` and `tau_reference`, plus the counts.
9. **`model_stage.diagnostic_reason`** is K-dependent only when at least one
   input record exists. In that case it quotes `total_model_count`
   (`aee4c1a`:734-738). In the zero-record case it is a fixed string. I list it
   as "derived, partially K-dependent".
10. **Crash paths.** A failed Julia subprocess (`check=True`) or a CYTools
    exception stops the run before `write_text`, so it writes no file. I did
    not record any layout for these paths.
11. **`details[].frozen_conifold` vs `frozen_conifold_per_class[]`.** These have
    the same shape. I list both subtrees in full, not as a shared reference.

## Part B: `fc66261`–`e7a8f51`

Writer: `scripts/reproduce_fuzzy_axions_h11_4.py`, the JSON written by `main()`
(`json.dumps(result, indent=2, sort_keys=True)` then `args.output.write_text`,
`fc66261:scripts/reproduce_fuzzy_axions_h11_4.py:1128-1133`,
`e7a8f51:scripts/reproduce_fuzzy_axions_h11_4.py:1356-1360`). `result` is
`_jsonable(summary)` (`fc66261:...:775-784`, `:1040`), which only converts numpy
values, stringifies dict keys and turns tuples into lists. It adds no keys and
removes none.

Normalization rules: Clarification A2.

### Notation and conventions used in this record

- A path is dot-joined object keys. `[]` is attached to the key whose value
  is an array (`details[].polytope_index`). `<map>` is the key level of a map
  node. It appears only as a prefix of the paths of the map's values, and in
  the name of a nested map node
  (`orientifold_reason_diagnostics.reason_counts_by_h11.<map>`).
- Every object key gives a path, at every depth, including the parent
  (`counts` and `counts.raw_frsts`). A key whose value is `null` gives a path.
- A `[]` level never gives a path of its own. An array value gives only the
  path of its key (`model_stage.model_count_per_record`). The keys of its
  object elements give `key[].k`. Arrays of scalars, arrays of arrays of
  scalars (for example `inverse_metric`, `lattice_matrix`, `sigma_rays`),
  empty arrays and `null` elements add nothing below the key.
- A `<map>` level never gives a path of its own (revision 3, the T061 API
  convention of `f5_legacy_layout_api.normalize`). A map node `X` gives the path of
  its key `X`, and the values of the map are normalized under `X.<map>`. In
  this range every map has scalar values, except `reason_counts_by_h11`, whose
  values are maps of scalars. So no path below a map node exists, and the
  inner level `reason_counts_by_h11.<map>` is itself listed as a map node.
  Revision 1 and revision 2 emitted a bare `X.<map>` for non-empty maps. Those
  paths are removed.
- Encoding: `scripts/f5_legacy_layouts_v10b.py` (T061) builds these layouts by composition. Its
  self-check confirms, for every layout, that R and O are disjoint and that
  `normalize(synthesize(S), map_nodes) == S` for S = R and S = R ∪ O.
- R = paths present in every file of the layout. O = paths that a file of the
  layout may or may not contain. A file matches when R ⊆ S ⊆ R ∪ O.
- **R convention for `details[]` (revision 2, aligned with the other 1.0
  record).** If a layout has `--keep-details` on, it describes files where
  `details` is non-empty. In such a layout, the following are in **R**:
  - the 14 first-level element keys `details[].*`;
  - the 11 `details[].provenance.*` keys;
  - with `--orientifold-audit`: the 3 `details[].orientifold_action_audit.*`
    keys;
  - with `--orientifold-audit` and `--orientifold-reason-diagnostics`:
    `details[].orientifold_action_audit.reason_diagnostics` and its 4 keys.

  Each of these is written in every element whenever its flags are on
  (`fc66261:REPRO:902-919`, `fc66261:GEN:4078-4092`, `fc66261:REPRO:492-493`,
  `:516-520`, `e7a8f51:REPRO:499-508`, `:650-656`).

  A file written with `--keep-details` whose `details` is `[]` has none of
  these paths. It is covered by the layout with `--keep-details` **off** and
  the same other flags. The R ∪ O of that layout includes every path such a
  file can hold: with no polytope there are no Kähler records, no models and
  no reason rows. I checked this mechanically for every K-on layout (the mapping is in the "Empty-details coverage" lines under
  each layout table).

  I found no case where this convention misses a real file:
  - K-on files with a non-empty `details` always carry these keys;
  - K-on files with an empty `details` always match the K-off layout named in
    the coverage lines.

  The keys *below* a value that can be `null` or `[]` in a non-empty
  `details` stay in **O**: `trilayer`, `frozen_conifold`, `*_per_class`,
  `kaehler_point_export_per_class`, and the reason rows. So do `model_stage.models[]`
  (it can be `[]` with records present, when the driver finds 0 models) and
  the top-level reason rows and maps (they can be empty under D ∧ A). For
  these, no other layout covers the empty case, or the other layout's R
  differs.
- Abbreviations in citations: `REPRO` = `scripts/reproduce_fuzzy_axions_h11_4.py`,
  `GEN` = `scripts/generate_geometric_data_multitriangulation.py`,
  `IOC` = `scripts/inherited_orientifold_candidates.py`. All citations are
  `commit:path:line` and were read with `git cat-file -p <commit>:<path>`.

### Commit coverage

The 1.0 range in this part is `fc66261` .. `e7a8f51` (parent of `fc66261` is
`47b1d74`, covered by the other F5 1.0 record). Nine commits are in this
range. The schema string is the same in all of them. I also checked the five
commits that were not assigned to me, because they also write 1.0 output
(A2 gives the 1.0 range as `ec73c15` through `e7a8f51`). The blob ids below
are from `git rev-parse <commit>:<path>`.

| commit | REPRO blob | IOC blob | GEN blob | Touches a serialized layout? |
|---|---|---|---|---|
| `fc66261` | `c19fed7c` | `38b30ea9` | `630c3631` | yes (`model_stage.convention`) |
| `f5f397c` | `9ebdb27f` | `459ccbbd` | `630c3631` | no (adds `lambda_f == 1` filter, `f5f397c:REPRO:507-516`; values only) |
| `4d2fc3b` (not assigned) | `9ebdb27f` | `459ccbbd` | `61fb9ba6` | no (GEN `validate_orientifold` internals only) |
| `4c3f7c5` (not assigned) | `9ebdb27f` | `2b4088b3` | `61fb9ba6` | no (IOC internals; nothing from IOC is serialized in group 1) |
| `1620b5a` (not assigned) | `9ebdb27f` | `2c326581` | `61fb9ba6` | no (same) |
| `f3bf677` (not assigned) | `9ebdb27f` | `2c326581` | `61fb9ba6` | no (validation docs only) |
| `4fcc3fa` | `7a209e59` | `12db3e66` | `61fb9ba6` | no (sets `topology["compute_general_fixed_surface_n_s"] = True`, `4fcc3fa:REPRO:503`; `topology` is not serialized) |
| `de1220b` (not assigned) | `7a209e59` | `12db3e66` | `61fb9ba6` | no (validation doc only) |
| `e7a8f51` | `60d74840` | `3266e9e5` | `61fb9ba6` | yes (new top-level key, new flag) |

`scripts/geometry_charge_conventions.py` (blob `a05bf4d4`) is the same in all
nine commits. It produces no serialized keys.

In group 1, the only data from IOC that reaches the JSON is the three-key
`_orientifold_action_audit` result. It is built in REPRO, and its values are
counts. The IOC records are filtered but never serialized
(`fc66261:REPRO:454-520`). So the IOC changes in `f5f397c`, `4c3f7c5`, `1620b5a`
and `4fcc3fa` change values only, not paths.

#### Transitive-import check for the unassigned commits (revision 2)

The local-import closure of REPRO changes at `4d2fc3b`, `4c3f7c5` and
`1620b5a`/`f3bf677`. For each group I ran
`git diff <prev> <commit> -- scripts/<module>.py` against the previous analysed
commit. None of the changes adds or removes a key in a dict that reaches the
persisted summary. In detail:

- **`4d2fc3b` vs `f5f397c`: GEN `630c3631` → `61fb9ba6` (REPRO and IOC
  unchanged).**
  - The diff has three hunks: `@@ -464 +464,37`, `@@ -466 +502`,
    `@@ -476 +512`. All three are inside `validate_orientifold`
    (`4d2fc3b:GEN:392`).
  - The change swaps the `basis_matrix` selector for the GLSM `charge_matrix`
    in the H2-action solve. It adds two new `OrientifoldValidationFailure`
    raises.
  - `validate_orientifold` is called only from IOC's
    `enumerate_orientifold_candidates`. Its result and failures go into IOC
    candidate records, which REPRO does not serialize (only the three-key
    audit result is written, `fc66261:REPRO:516-520`).
  - The GEN functions whose output does reach the JSON have no hunks:
    `load_mirror_polytopes` (`details[].provenance`, `4d2fc3b:GEN:4068`),
    `evaluate_kaehler_point` (`...diagnostic` and `...diagnostic.checks`,
    `4d2fc3b:GEN:1508`), `extract_topology` (`:660`),
    `sample_stretched_kaehler_points` (`:1324`) and `configure_mosek_license`
    (`:149`). They only move by +36 lines.
  - Paths: unchanged.
- **`4c3f7c5` vs `4d2fc3b`: IOC `459ccbbd` → `2b4088b3` (GEN `61fb9ba6`
  unchanged, REPRO unchanged).**
  - The diff has four hunks: `@@ -536,0 +537,33` adds the helper
    `_cone_has_smooth_star` (`4c3f7c5:IOC:537`). `@@ -547,3 +580,3`,
    `@@ -556,0 +590,10` and `@@ -570,0 +614,2` change the docstring of
    `identity_fixed_surface_n_s_table` (`4c3f7c5:IOC:570`) and add one
    `continue` that skips table entries.
  - That table goes into `topology["fixed_surface_n_s"]`
    (`fc66261:REPRO:500-502`), which is never serialized.
  - The IOC functions REPRO imports and whose output reaches values are
    unchanged: `_ambient_intersection_tensor` (numbers only, in
    `_frozen_conifold_diagnostic` / `_fixed_locus_euler_characteristic`),
    `_triangulation_cones` and `facets_with_non_smooth_cones`.
  - Paths: unchanged.
- **`1620b5a` (= `f3bf677`) vs `4c3f7c5`: IOC `2b4088b3` → `2c326581` (GEN
  unchanged, REPRO unchanged).**
  - The diff has six hunks:
    - `@@ -43,0 +44` adds `"torus_shift_not_involution"` to
      `TERMINAL_STATUSES`.
    - `@@ -462,0 +464`, `@@ -464 +466,11`, `@@ -473 +484,0`,
      `@@ -483 +494,15` and `@@ -488,0 +514` are in
      `_dual_vertex_parity_evidence` (`1620b5a:IOC:459`). They make `2t`
      exact and add a key `noninteger_pairings` to its returned dict.
    - `@@ -857,6 +883,37` is in `enumerate_orientifold_candidates`
      (`1620b5a:IOC:823`). It adds a `torus_shift_rejected_noninvolution`
      record with keys `record_kind`, `matrix_candidate_id`,
      `torus_shift_binary_source`.
  - Both new keys live only inside IOC candidate records (via
    `classify_smoothness` → `smoothness`). REPRO reads those records only
    through `record.get("terminal_status")`, `record.get("lambda_f")` and
    `record.get("h11_minus")` (`f5f397c:REPRO:512-521`), and serializes only
    the counts.
  - Paths: unchanged.
- `f3bf677` has no change under `scripts/`. `de1220b` has the same closure as
  `4fcc3fa`.

So these commits stay in group 1, and the group and layout counts do not
change.

#### Line maps for group 1 citations

All group 1 REPRO citations below are given at `fc66261`. To convert a line L:
- to blob `9ebdb27f` (`f5f397c`, `4d2fc3b`, `4c3f7c5`, `1620b5a`, `f3bf677`):
  L ≤ 506 → L; 507 ≤ L ≤ 510 → L+5; L > 510 → L+6;
- to blob `7a209e59` (`4fcc3fa`, `de1220b`): L ≤ 502 → L; 503 ≤ L ≤ 506 →
  L+1; 507 ≤ L ≤ 510 → L+6; L > 510 → L+7.
- GEN citations are given for blob `630c3631` (`fc66261`, `f5f397c`). For blob
  `61fb9ba6` (`4d2fc3b` onward), L > 464 → L+36.

Anchors to check the map: `summary = {` is at 969 / 975 / 976.
`_run_model_stage` is at 668 / 674 / 675. `details.append(` is at 902 / 908 / 909.
`evaluate_kaehler_point` is at GEN 1472 / 1508.

---

#### 1.0 group 5 (part B group 1): commits fc66261, f5f397c, 4d2fc3b, 4c3f7c5, 1620b5a, f3bf677, 4fcc3fa, de1220b

(Assigned commits in this group: `fc66261`, `f5f397c`, `4fcc3fa`. The other
five commits write the same layouts, as shown in the coverage table.)

**What `fc66261` changes relative to `47b1d74`.** One change to the serialized
layout, and one new flag that does not change the structure:
- `_run_model_stage` adds two keys to `convention`:
  `model_stage.convention.qcd_divisor_domain` and
  `model_stage.convention.qcd_divisor_domain_justification`
  (`fc66261:REPRO:677-688`). Both are in R whenever `--model-stage` is on,
  with and without records. `convention` is built before the `if not
  records` branch (`:706`), so both returns carry the two keys.
- New flag `--qcd-divisor-domain {all_prime,leading_nonself}` (`:1097-1112`,
  default `all_prime`). It changes values only: the content of
  `convention.qcd_divisor_domain`, one extra `diagnostic_reason` clause
  (`:940-948`), and `claim_status.model_count` (`:1029-1031`). The Julia driver
  writes `qcd_divisor_domain` to its HDF5 output
  (`fc66261:scripts/fuzzy_axion_model_stage_driver.jl:132`), but REPRO does
  not read that dataset back (`:740-762`).
- Nothing else in the path set changes. The `47b1d74` → `fc66261` diff of REPRO
  is only the above plus the help text. So for `--model-stage` **off**, the
  `fc66261` layouts have the same path sets as the `--model-stage`-off layouts
  of `47b1d74`, if the `47b1d74` record uses the same conventions. When the two
  records are joined, those layouts can be merged into one layout that lists
  both commit ranges. For `--model-stage` **on**, the layouts differ by exactly
  the two R paths above.

##### Base R

The base is all flags off, with the default `--h11 4`. That value has a Table 1
target, so `paper_targets` is an object.

Top-level scalars (`fc66261:scripts/reproduce_fuzzy_axions_h11_4.py:969-972`):
- `schema_version`
- `paper`
- `orientifold_source`

`input` (`fc66261:scripts/reproduce_fuzzy_axions_h11_4.py:973-980`):
- `input`
- `input.source`
- `input.parquet_dir`
- `input.requested_h11`
- `input.favorable_lattice`
- `input.record_count`
- `input.population_complete`

`counts` (`fc66261:scripts/reproduce_fuzzy_axions_h11_4.py:981-1006`). The four
flag-gated counts are always present and are `null` when their flag is off
(`:994-1005`):
- `counts`
- `counts.favorable_polytopes`
- `counts.raw_frsts`
- `counts.frst_classes`
- `counts.raw_trilayer_polytopes`
- `counts.raw_trilayer_frst_classes`
- `counts.nonfrozen_trilayer_frst_classes`
- `counts.h21_plus_zero_trilayer_frst_classes`
- `counts.identity_torus_action_count`
- `counts.identity_torus_action_cy_count`
- `counts.identity_valid_o3o7_action_cy_count`
- `counts.source_vertex_evidence_inherited_orientifold_cys`
- `counts.source_vertex_evidence_h11_minus_zero_orientifold_cys`
- `counts.kaehler_point_export_accepted_count`
- `counts.kaehler_point_export_rejected_count`

`paper_targets` (`fc66261:scripts/reproduce_fuzzy_axions_h11_4.py:1007`. The
value is `PAPER_TARGETS_BY_H11.get(args.h11)` at `:788`. The table is at
`:53-102` and has rows for h11 = 2, 3, 4, 5, 6, 7, each with the same six
keys):
- `paper_targets`
- `paper_targets.favorable_polytopes`
- `paper_targets.frst_classes`
- `paper_targets.inherited_orientifold_cys`
- `paper_targets.h11_minus_zero_orientifold_cys`
- `paper_targets.h11_minus_zero_h21_plus_zero_orientifold_cys`
- `paper_targets.models`

`claim_status` (`fc66261:scripts/reproduce_fuzzy_axions_h11_4.py:1008-1036`.
Both branches, `targets is None` `:1009-1014` and with targets `:1016-1035`,
have the same four keys):
- `claim_status`
- `claim_status.favorable_polytopes`
- `claim_status.frst_classes`
- `claim_status.h21_plus_zero`
- `claim_status.model_count`

`model_stage`. It is `null` without `--model-stage`
(`fc66261:scripts/reproduce_fuzzy_axions_h11_4.py:936`, `:1037`):
- `model_stage`

`details`. It is `null` without `--keep-details`
(`fc66261:scripts/reproduce_fuzzy_axions_h11_4.py:1038`):
- `details`

Base R count: 39 paths.

##### Base O

None. Every base key is always written (`summary` literal,
`fc66261:REPRO:969-1039`), and no later code mutates `summary`. The only later
mutation of a `summary` value is `model_stage["diagnostic_reason"]`
(`:941-967`), which changes a value, not a key.

##### Map nodes

None in group 1. `paper_targets`, `claim_status`, `counts`, `input`,
`model_stage.convention`, `details[].provenance`,
`details[].kaehler_point_export_per_class[].potential_matrix_convention`,
`...diagnostic` and `...diagnostic.checks` have fixed field-name keys. See
Open point 3 for `diagnostic.checks`.

##### Flag deltas

Letters used below: **T** = `--h11` value with a Table 1 row (2-7).
**M** = `--model-stage`. **K** = `--keep-details`. **A** = `--orientifold-audit`.
**E** = `--export-kaehler-points`. **E′** = E or M (`export_kaehler_points =
args.export_kaehler_points or args.model_stage`, `fc66261:REPRO:809`).

**Δ¬T: `--h11` value with no Table 1 row (not in {2..7}; the default 4 is in the set).**
This removes the 6 paths below from R. `paper_targets` stays in R with value `null`.
(`fc66261:REPRO:53-102`, `:788`, `:1007`, `:1046-1052`.) `claim_status` keeps its
four keys, with value `null` (`:1009-1014`).
- `paper_targets.favorable_polytopes`
- `paper_targets.frst_classes`
- `paper_targets.inherited_orientifold_cys`
- `paper_targets.h11_minus_zero_orientifold_cys`
- `paper_targets.h11_minus_zero_h21_plus_zero_orientifold_cys`
- `paper_targets.models`

**ΔM: `--model-stage` (`fc66261:REPRO:1067-1080`).** `model_stage` becomes the
dict from `_run_model_stage` (`:668-772`, called at `:936`).
Adds to R (13 paths; `convention` `:677-704`; no-records return
`:706-716`; with-records return `:765-772`):
- `model_stage.input_record_count`
- `model_stage.total_model_count`
- `model_stage.model_count_per_record`
- `model_stage.models`
- `model_stage.convention`
- `model_stage.diagnostic_reason`
- `model_stage.convention.qcd_divisor_domain`
- `model_stage.convention.qcd_divisor_domain_justification`
- `model_stage.convention.gs`
- `model_stage.convention.gs_justification`
- `model_stage.convention.w0_real`
- `model_stage.convention.w0_imag`
- `model_stage.convention.w0_justification`

Adds to O:
- `model_stage.convention.prefactor_P`. It is set only after the driver ran
  (`fc66261:REPRO:764`). It is absent when there are no exported records,
  because the early return at `:706-716` comes first.

`model_stage.models` is `null` without K (`:711`, `:744-745`). Its element keys
are in interaction I3 below. `model_stage.model_count_per_record` is an array
of ints (`:742`). `model_stage.diagnostic_reason` can be changed by `:941-967`
(value only).

**ΔK: `--keep-details` (`fc66261:REPRO:1055`).** `details` becomes the list
built at `:811`, `:902-919`. Under the R convention in the header, this layout
describes files with a non-empty `details`. When `details` is non-empty, each
element has all 14 first-level keys (`:902-919`) and all 11 `provenance` keys.
`details` is `[]` only when `load_mirror_polytopes` returns no polytope
(`fc66261:GEN:4032-4097`; for example an `--h11` with no favorable polytope in
the mirror). That file matches the K-off layout with the same T and M.

Adds to R (25 paths):
- `details[].polytope_index`
- `details[].provenance`
- `details[].provenance.source_kind`
- `details[].provenance.dataset`
- `details[].provenance.dataset_url`
- `details[].provenance.parquet_file`
- `details[].provenance.row_index`
- `details[].provenance.mirror_h11`
- `details[].provenance.mirror_h12`
- `details[].provenance.physical_h11`
- `details[].provenance.physical_h21`
- `details[].provenance.vertex_count`
- `details[].provenance.favorable_checked_by`
- `details[].raw_frst_count`
- `details[].frst_class_count`
- `details[].trilayer`
- `details[].frozen_conifold`
- `details[].frozen_conifold_per_class`
- `details[].nonfrozen_class_count`
- `details[].h21_plus_zero_per_class`
- `details[].h21_plus_zero_class_count`
- `details[].identity_torus_action_numerators`
- `details[].identity_valid_o3o7_action_numerators`
- `details[].orientifold_action_audit`
- `details[].kaehler_point_export_per_class`

Adds to O (33 paths; each is under a value that can be `null` or `[]`).
The sources of the sub-blocks are:
- `provenance`: `fc66261:GEN:4078-4092`. Always 11 keys.
- `trilayer`: `null` when there is no trilayer candidate (`fc66261:REPRO:167`,
  `:816`), else 7 keys (`:158-166`). `p0`, `q0`, `torus_shift_numerator` are
  int arrays. `lattice_matrix` is an array of int arrays.
- `frozen_conifold`: `null` with no trilayer or no classes (`:823`, `:852`),
  else one `_frozen_conifold_diagnostic` dict. The early-return shape is
  `{status, reason, surface_records: []}` (`:210-214`). The normal shape is
  `{status, surface_records, unavailable_cones, frozen_surface_count}`
  (`:264-269`). `surface_records[]` elements: `:249-255`.
  `unavailable_cones` is an array of int-matrix arrays (`:217`, `:223`, `:231`).
- `frozen_conifold_per_class`: `null` with no trilayer (`:824`), else a list of
  the same dicts (`:842-845`).
- `h21_plus_zero_per_class`: `null` with no trilayer (`:826`), else a list of
  `_h21_plus_zero_diagnostic` dicts. Both returns have the same 6 keys
  (`:429-436`, `:444-451`). `components[]` elements are
  `{dim, rays, chi, kind}` (`:386`, `:403`, `:407`). `components` is `[]` at
  `:368` and is partial at `:391-396`. `reasons` is a string array, and `rays`
  is an int array (tuple → list).
- `identity_torus_action_numerators`, `identity_valid_o3o7_action_numerators`:
  arrays of int arrays (`:180-187`, `:914-915`).
- `orientifold_action_audit`: `null` without A (`:828`). See I1.
- `kaehler_point_export_per_class`: `null` without E′ or with no trilayer
  (`:873-874`). See I2.

- `details[].trilayer.p0`
- `details[].trilayer.q0`
- `details[].trilayer.lattice_matrix`
- `details[].trilayer.torus_shift_numerator`
- `details[].trilayer.torus_shift_denominator`
- `details[].trilayer.lambda_f`
- `details[].trilayer.criterion`
- `details[].frozen_conifold.status`
- `details[].frozen_conifold.reason`
- `details[].frozen_conifold.surface_records`
- `details[].frozen_conifold.unavailable_cones`
- `details[].frozen_conifold.frozen_surface_count`
- `details[].frozen_conifold.surface_records[].rays`
- `details[].frozen_conifold.surface_records[].frozen_conifold_count`
- `details[].frozen_conifold.surface_records[].formula`
- `details[].frozen_conifold_per_class[].status`
- `details[].frozen_conifold_per_class[].reason`
- `details[].frozen_conifold_per_class[].surface_records`
- `details[].frozen_conifold_per_class[].unavailable_cones`
- `details[].frozen_conifold_per_class[].frozen_surface_count`
- `details[].frozen_conifold_per_class[].surface_records[].rays`
- `details[].frozen_conifold_per_class[].surface_records[].frozen_conifold_count`
- `details[].frozen_conifold_per_class[].surface_records[].formula`
- `details[].h21_plus_zero_per_class[].status`
- `details[].h21_plus_zero_per_class[].reasons`
- `details[].h21_plus_zero_per_class[].chi_F_I`
- `details[].h21_plus_zero_per_class[].h21_minus`
- `details[].h21_plus_zero_per_class[].h21_plus`
- `details[].h21_plus_zero_per_class[].components`
- `details[].h21_plus_zero_per_class[].components[].dim`
- `details[].h21_plus_zero_per_class[].components[].rays`
- `details[].h21_plus_zero_per_class[].components[].chi`
- `details[].h21_plus_zero_per_class[].components[].kind`

**ΔA alone and ΔE alone (without K): no change to the path set.** A and E′ set
only the values of the four flag-gated `counts` keys (`:994-1005`). The keys are
already in base R.

##### Interactions

- **I1 = K ∧ A.** Adds to R (`fc66261:REPRO:869-872`, `:916`; the dict is built
  at `:492-493` and `:516-520`; both returns have the same 3 keys;
  `h11_minus_zero_classes` is an int array):
- `details[].orientifold_action_audit.inherited`
- `details[].orientifold_action_audit.h11_minus_zero`
- `details[].orientifold_action_audit.h11_minus_zero_classes`
- **I2 = K ∧ E′.** Adds to O (50 paths). Sources:
  `fc66261:REPRO:873-901`, `:917`; `_export_kaehler_point` `:523-637` (rejected
  shape `{status, reason, diagnostic}` `:617-624`; accepted shape `:625-637`);
  `diagnostic` from `fc66261:GEN:1472-1666` (initial 12 keys `:1496-1511`;
  `:1515-1527` adds 4 keys and 2 `checks` keys; `:1553-1580` adds 6 keys;
  `:1581-1636` adds 14 `checks` keys; `failure_reason` only on exception
  `:1662-1665`). The list has `null` elements for classes that are not
  `h21_plus_zero` (`:882-884`). `point`, `prime_divisor_volumes` are float
  arrays. `inverse_metric`, `glsm_charge_matrix` are arrays of arrays.
- `details[].kaehler_point_export_per_class[].status`
- `details[].kaehler_point_export_per_class[].reason`
- `details[].kaehler_point_export_per_class[].h11`
- `details[].kaehler_point_export_per_class[].point`
- `details[].kaehler_point_export_per_class[].cy_volume`
- `details[].kaehler_point_export_per_class[].prime_divisor_volumes`
- `details[].kaehler_point_export_per_class[].inverse_metric`
- `details[].kaehler_point_export_per_class[].glsm_charge_matrix`
- `details[].kaehler_point_export_per_class[].potential_matrix_convention`
- `details[].kaehler_point_export_per_class[].potential_matrix_convention.Q`
- `details[].kaehler_point_export_per_class[].diagnostic`
- `details[].kaehler_point_export_per_class[].diagnostic.attempt_index`
- `details[].kaehler_point_export_per_class[].diagnostic.point_kind`
- `details[].kaehler_point_export_per_class[].diagnostic.point_seed`
- `details[].kaehler_point_export_per_class[].diagnostic.attempted`
- `details[].kaehler_point_export_per_class[].diagnostic.solver`
- `details[].kaehler_point_export_per_class[].diagnostic.point_status`
- `details[].kaehler_point_export_per_class[].diagnostic.checks`
- `details[].kaehler_point_export_per_class[].diagnostic.divisor_volume_tolerance`
- `details[].kaehler_point_export_per_class[].diagnostic.prime_divisor_volume_lower_bound`
- `details[].kaehler_point_export_per_class[].diagnostic.effective_divisor_volume_lower_bound`
- `details[].kaehler_point_export_per_class[].diagnostic.divisor_volume_lower_bounds_enforced`
- `details[].kaehler_point_export_per_class[].diagnostic.volume_backend`
- `details[].kaehler_point_export_per_class[].diagnostic.coordinate_dimension`
- `details[].kaehler_point_export_per_class[].diagnostic.point_norm`
- `details[].kaehler_point_export_per_class[].diagnostic.point_sha256`
- `details[].kaehler_point_export_per_class[].diagnostic.minimum_kaehler_slack`
- `details[].kaehler_point_export_per_class[].diagnostic.cy_volume`
- `details[].kaehler_point_export_per_class[].diagnostic.minimum_curve_volume`
- `details[].kaehler_point_export_per_class[].diagnostic.minimum_basis_divisor_volume`
- `details[].kaehler_point_export_per_class[].diagnostic.minimum_prime_divisor_volume`
- `details[].kaehler_point_export_per_class[].diagnostic.minimum_effective_divisor_volume`
- `details[].kaehler_point_export_per_class[].diagnostic.minimum_metric_eigenvalue`
- `details[].kaehler_point_export_per_class[].diagnostic.failure_reason`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.finite_coordinates`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.cone_membership`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.finite_cy_volume`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.positive_cy_volume`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.finite_curve_volumes`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.positive_curve_volumes`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.finite_basis_divisor_volumes`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.positive_basis_divisor_volumes`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.finite_prime_divisor_volumes`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.positive_prime_divisor_volumes`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.prime_divisor_volume_lower_bound`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.finite_effective_divisor_volumes`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.positive_effective_divisor_volumes`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.effective_divisor_volume_lower_bound`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.finite_inverse_metric`
- `details[].kaehler_point_export_per_class[].diagnostic.checks.positive_inverse_metric`
- **I3 = K ∧ M.** Adds to O (`fc66261:REPRO:745-762`; `[]` when there are no
  records `:711`, and also when the driver finds 0 models):
- `model_stage.models[].record_index`
- `model_stage.models[].axion_index`
- `model_stage.models[].qcd_divisor_index`
- `model_stage.models[].lambda`
- `model_stage.models[].mass_reference_log10_ev`
- `model_stage.models[].tau_reference`
- **M ⇒ E′** (`:809`). So K ∧ M always includes I2. M with or without E gives
  the same path set.
- No flag is rejected in combination with another. There are no mutually
  exclusive flags, and no requirement between flags is enforced.
  `--export-kaehler-points`'s help says "Requires --keep-details to appear in
  the output" (`:1060-1065`). That is the I2 interaction, not a check.

##### Structure-neutral flags

`--parquet-dir`, `--limit`, `--progress`, `--output`, `--gs`, `--w0-real`,
`--w0-imag`, `--qcd-divisor-domain`, `--julia-binary`, `--julia-project`. Also
`--orientifold-audit` without K, `--export-kaehler-points` without K or when M
is on, and the exact `--h11` value within each class (with target / without
target). `--limit` changes only `input.population_complete`, the counts and the
`diagnostic_reason` text (`:930`, `:949-950`). See Open point 6 for
`--limit ≤ 0`.

##### K-dependent fields

These are computed by the Julia driver from `inverse_metric` (Kinv; Cholesky at
`fc66261:src/paper_benchmarks/fuzzy_axions_model_stage.jl:83-85`, model loop at
`fc66261:scripts/fuzzy_axion_model_stage_driver.jl:95-111`), read back at
`fc66261:REPRO:740-762`:
- `model_stage.total_model_count`
- `model_stage.model_count_per_record` (each per-record count)
- `model_stage.models[].record_index`, `.axion_index`, `.qcd_divisor_index`,
  `.lambda`, `.mass_reference_log10_ev`, `.tau_reference`. Which models
  exist, and which leading axion and tau are the reference, depends on K.

These are derived from those fields:
- `model_stage.diagnostic_reason` (`:961-965` puts `total_model_count` into the
  text)
- `claim_status.model_count` (`:1024-1034`)

These are under `model_stage` but are K-free:
- `model_stage.input_record_count` (Python record count, `:766`)
- `model_stage.convention.*`, including `prefactor_P`. `prefactor_P` comes back
  from the driver but is `fuzzy_axion_prefactor_P(gs)`, a function of `gs`
  only (`fc66261:scripts/fuzzy_axion_model_stage_driver.jl:55`).

These are K-free (A2): all `counts.*`, the other `claim_status.*` entries,
`input.*`, `paper_targets.*`, all `details[].*` geometry, and
`details[].kaehler_point_export_per_class[].inverse_metric` (the stored CYTools
Kinv). `...diagnostic.minimum_metric_eigenvalue`,
`...diagnostic.checks.finite_inverse_metric` and `.positive_inverse_metric` are
properties of that stored Kinv (`fc66261:GEN:1551`, `:1574-1578`, `:1630-1634`),
so they are also K-free in the A2 sense.

##### Layout count

16 distinct layouts: 2 (T) × [2 with K off (M off/on; A and E have no effect)
+ 6 with K on (A off/on × {none, E only, M})]. Each layout is:
- R = base R (minus Δ¬T if no target) ∪ ΔM.R if M ∪ ΔK.R if K ∪ I1 if K∧A;
- O = ΔM.O if M ∪ ΔK.O if K ∪ I2 if K∧E′ ∪ I3 if K∧M.

| # | `--h11` has target | `--keep-details` | export (none / `--export-kaehler-points` / `--model-stage`) | `--orientifold-audit` | `--orientifold-reason-diagnostics` | \|R\| | \|O\| |
|---|---|---|---|---|---|---|---|
| 1.1 | yes | off | none or E only | any | n/a | 39 | 0 |
| 1.2 | yes | off | M (E any) | any | n/a | 52 | 1 |
| 1.3 | yes | on | none | off | n/a | 64 | 33 |
| 1.4 | yes | on | none | on | n/a | 67 | 33 |
| 1.5 | yes | on | E only | off | n/a | 64 | 83 |
| 1.6 | yes | on | E only | on | n/a | 67 | 83 |
| 1.7 | yes | on | M (E any) | off | n/a | 77 | 90 |
| 1.8 | yes | on | M (E any) | on | n/a | 80 | 90 |
| 1.9 | no | off | none or E only | any | n/a | 33 | 0 |
| 1.10 | no | off | M (E any) | any | n/a | 46 | 1 |
| 1.11 | no | on | none | off | n/a | 58 | 33 |
| 1.12 | no | on | none | on | n/a | 61 | 33 |
| 1.13 | no | on | E only | off | n/a | 58 | 83 |
| 1.14 | no | on | E only | on | n/a | 61 | 83 |
| 1.15 | no | on | M (E any) | off | n/a | 71 | 90 |
| 1.16 | no | on | M (E any) | on | n/a | 74 | 90 |

Empty-details coverage, checked mechanically: a K-on file with `details: []`
matches layout 1.1 (for 1.3-1.6), 1.2 (for 1.7-1.8), 1.9 (for 1.11-1.14) and
1.10 (for 1.15-1.16).

---

#### 1.0 group 6 (part B group 2): commits e7a8f51

**What changed from group 1.** REPRO blob `7a209e59` → `60d74840`. IOC blob
`12db3e66` → `3266e9e5`.
- New top-level key `orientifold_reason_diagnostics`, always written: `null`
  unless the new flag is on (`e7a8f51:REPRO:1248-1257`). It is in base R.
- New flag `--orientifold-reason-diagnostics` (**D**, `e7a8f51:REPRO:1277-1284`).
  With D, `orientifold_reason_diagnostics` becomes the
  `_orientifold_reason_diagnostics_summary` object (`:933-981`, the dict at
  `:958-981`). With D ∧ A, `_orientifold_action_audit` also adds
  `reason_diagnostics` to each per-polytope result (`:499-508`, `:650-656`).
  That result goes into `details[].orientifold_action_audit` (`:1126`). Its rows
  are copied into the top-level lists (`:1078-1082`).
- Rows contain dicts built in IOC at `e7a8f51`:
  - `fixed_component` of surface rows is from
    `_general_fixed_surface_n_s_table` (`e7a8f51:IOC:1527-1533`,
    `:1561-1567`).
  - `surface_data` is the surface diagnostic dict minus `fixed_components`
    (`e7a8f51:REPRO:586-590`; dicts at `e7a8f51:IOC:1534-1543` (unavailable)
    and `:1556-1575` (certified, `**diagnostic_data` from `:1949-1987`)).
  - `fixed_component` of unresolved rows is from `_fixed_component_records`
    (`e7a8f51:IOC:408-423`).
  - `nu` and `torus_shift` are `_fraction_vector_to_json` objects
    `{numerator, denominator}` (`e7a8f51:IOC:133-145`, `:2425`).
  - The Chern terms are `_fraction_to_json`: an int, or `{numerator,
    denominator}` (`e7a8f51:IOC:722-728`).
- Everything else in the path set is the same as group 1. The `4fcc3fa` →
  `e7a8f51` REPRO diff adds only `Counter`, the audit's reason-diagnostic
  collection, the summary helpers, the new key and the new flag.

##### Base R

The same 39 paths as group 1 base R, plus `orientifold_reason_diagnostics`. The
citations at `e7a8f51` are:
- top-level scalars `e7a8f51:scripts/reproduce_fuzzy_axions_h11_4.py:1179-1182`
- `input` `e7a8f51:scripts/reproduce_fuzzy_axions_h11_4.py:1183-1190`
- `counts` `e7a8f51:scripts/reproduce_fuzzy_axions_h11_4.py:1191-1216`
- `paper_targets` `e7a8f51:scripts/reproduce_fuzzy_axions_h11_4.py:1217`
  (`:985`, table `:54-103`)
- `claim_status` `e7a8f51:scripts/reproduce_fuzzy_axions_h11_4.py:1218-1246`
- `model_stage` `e7a8f51:scripts/reproduce_fuzzy_axions_h11_4.py:1247`
- `orientifold_reason_diagnostics`
  `e7a8f51:scripts/reproduce_fuzzy_axions_h11_4.py:1248-1257`
- `details` `e7a8f51:scripts/reproduce_fuzzy_axions_h11_4.py:1258`

- `schema_version`
- `paper`
- `orientifold_source`
- `input`
- `input.source`
- `input.parquet_dir`
- `input.requested_h11`
- `input.favorable_lattice`
- `input.record_count`
- `input.population_complete`
- `counts`
- `counts.favorable_polytopes`
- `counts.raw_frsts`
- `counts.frst_classes`
- `counts.raw_trilayer_polytopes`
- `counts.raw_trilayer_frst_classes`
- `counts.nonfrozen_trilayer_frst_classes`
- `counts.h21_plus_zero_trilayer_frst_classes`
- `counts.identity_torus_action_count`
- `counts.identity_torus_action_cy_count`
- `counts.identity_valid_o3o7_action_cy_count`
- `counts.source_vertex_evidence_inherited_orientifold_cys`
- `counts.source_vertex_evidence_h11_minus_zero_orientifold_cys`
- `counts.kaehler_point_export_accepted_count`
- `counts.kaehler_point_export_rejected_count`
- `paper_targets`
- `paper_targets.favorable_polytopes`
- `paper_targets.frst_classes`
- `paper_targets.inherited_orientifold_cys`
- `paper_targets.h11_minus_zero_orientifold_cys`
- `paper_targets.h11_minus_zero_h21_plus_zero_orientifold_cys`
- `paper_targets.models`
- `claim_status`
- `claim_status.favorable_polytopes`
- `claim_status.frst_classes`
- `claim_status.h21_plus_zero`
- `claim_status.model_count`
- `model_stage`
- `orientifold_reason_diagnostics`
- `details`

Base R count: 40 paths.

##### Base O

None (same reasoning as group 1; `summary` literal `e7a8f51:REPRO:1179-1259`).

##### Map nodes

Present only with D, all under `orientifold_reason_diagnostics`
(`e7a8f51:REPRO:957-976`):
- `orientifold_reason_diagnostics.reason_counts`
- `orientifold_reason_diagnostics.reason_counts_by_h11`
- `orientifold_reason_diagnostics.reason_counts_by_h11.<map>`
- `orientifold_reason_diagnostics.reason_counts_by_polytope`
- `orientifold_reason_diagnostics.reason_counts_by_frst_class`
- `orientifold_reason_diagnostics.reason_counts_by_matrix`
- `orientifold_reason_diagnostics.reason_counts_by_fixed_component`
- `orientifold_reason_diagnostics.unresolved_candidate_reason_counts`

- `reason_counts` (`:957`, `:964`) and `unresolved_candidate_reason_counts`
  (`:974-976`) are keyed by reason code.
- `reason_counts_by_h11` (`:965`) is keyed by `str(h11)`. Its single value is
  the `reason_counts` map, so the inner level `reason_counts_by_h11.<map>` is
  a map node too (nested map).
- Under the API convention (revision 3), none of these maps gives any path
  below its own key: all their values are scalars, or maps of scalars.
- `reason_counts_by_polytope` (`:966-968`) is keyed by polytope index.
  `reason_counts_by_frst_class` (`:969-971`) by `"poly:class"`.
  `reason_counts_by_matrix` (`:972`, key from `:943-944`) by
  `"poly:class:matrix_id"`. `reason_counts_by_fixed_component` (`:973`, key from
  `:946-955`) by a JSON string.
- The values are ints (`_count_reason_rows` `:924-930`). Despite the names,
  the `*_by_<scope>` maps count skipped rows per scope, not per reason.

No map nodes in `details[]`.

##### Flag deltas

Δ¬T, ΔM, ΔK, ΔA-alone and ΔE-alone are the same as group 1, at the `e7a8f51`
lines:
- `_run_model_stage` `:805-903`: `convention` `:814-841`, no-records return
  `:843-853`, `prefactor_P` `:901`, models `:881-899`.
- `details.append` `:1112-1129`.
- `_trilayer_candidate` `:137-168`.
- `_frozen_conifold_diagnostic` `:191-270`.
- `_fixed_locus_euler_characteristic` `:347-411`.
- `_h21_plus_zero_diagnostic` `:414-452`.
- `_export_kaehler_point` `:660-774`.
- GEN at blob `61fb9ba6`: provenance `:4114-4128`, `evaluate_kaehler_point`
  `:1508-1702`.
- flags: `--h11` `:1266-1272`, `--keep-details` `:1275`, `--orientifold-audit`
  `:1276`, `--export-kaehler-points` `:1285-1294`, `--model-stage` `:1295-1308`.

They add the same path lists as in group 1, plus the new delta below.

**ΔD: `--orientifold-reason-diagnostics`.** Adds to R (16 paths;
`e7a8f51:REPRO:958-981`). Revision 3 removed the bare
`reason_counts_by_h11.<map>` path, which revision 2 had in R. With D but
without A, the three row lists are `[]`, and all maps other than
`reason_counts_by_h11` are `{}` (`:1009-1011` are never extended without A,
`:1078`):
- `orientifold_reason_diagnostics.schema_version`
- `orientifold_reason_diagnostics.h11`
- `orientifold_reason_diagnostics.surface_attempt_count`
- `orientifold_reason_diagnostics.skipped_surface_count`
- `orientifold_reason_diagnostics.certified_surface_count`
- `orientifold_reason_diagnostics.reason_counts`
- `orientifold_reason_diagnostics.reason_counts_by_h11`
- `orientifold_reason_diagnostics.reason_counts_by_polytope`
- `orientifold_reason_diagnostics.reason_counts_by_frst_class`
- `orientifold_reason_diagnostics.reason_counts_by_matrix`
- `orientifold_reason_diagnostics.reason_counts_by_fixed_component`
- `orientifold_reason_diagnostics.unresolved_candidate_component_count`
- `orientifold_reason_diagnostics.unresolved_candidate_reason_counts`
- `orientifold_reason_diagnostics.surface_attempts`
- `orientifold_reason_diagnostics.unresolved_components`
- `orientifold_reason_diagnostics.certified_surfaces`

##### Interactions

The group 1 interactions apply unchanged: I1 = K ∧ A, I2 = K ∧ E′, I3 = K ∧ M,
M ⇒ E′. New in group 2:

- **I4 = D ∧ A.** Adds to O the row element paths of the top-level summary
  (127 paths). Revision 3 dropped the 7 bare `….<map>` entries that revision 2
  listed here: maps can now be non-empty without adding a path. All are
  optional:
  the row lists are empty when no non-identity involution reaches the
  general-L surface computation (`e7a8f51:IOC:2320-2339`), and maps are `{}`
  when there are no skipped rows.
  - Rows of `surface_attempts[]` and `certified_surfaces[]` are built at
    `e7a8f51:REPRO:577-607`. `certified_surfaces` holds the same dicts as the
    certified `surface_attempts`.
  - In both, the 18 paths from `polytope_index` to `surface_data.sigma_dimension`
    are written in every row. `surface_data.c2_ambient_restricted` through
    `surface_data.restricted_divisor_coefficients[].coefficients[].coefficient`
    are certified-only (`e7a8f51:IOC:1556-1575`, `:1949-1987`).
  - `candidate_context.*` is only on certified rows that have a matching
    `lambda_f = 1` candidate (`e7a8f51:REPRO:556-563`, `:592-604`).
  - `.numerator` / `.denominator` under `c2_ambient_restricted` and
    `c1_surface_squared` appear only when that value is not an integer
    (`e7a8f51:IOC:722-728`).
  - Rows of `unresolved_components[]` are built at `e7a8f51:REPRO:628-644`. All
    UC keys are written in every row. `fixed_component` has the 8
    `_fixed_component_records` keys (`e7a8f51:IOC:408-423`).
- `orientifold_reason_diagnostics.surface_attempts[].polytope_index`
- `orientifold_reason_diagnostics.surface_attempts[].frst_class_index`
- `orientifold_reason_diagnostics.surface_attempts[].matrix_id`
- `orientifold_reason_diagnostics.surface_attempts[].lattice_matrix`
- `orientifold_reason_diagnostics.surface_attempts[].fixed_component`
- `orientifold_reason_diagnostics.surface_attempts[].fixed_component.sigma_rays`
- `orientifold_reason_diagnostics.surface_attempts[].fixed_component.nu`
- `orientifold_reason_diagnostics.surface_attempts[].fixed_component.nu.numerator`
- `orientifold_reason_diagnostics.surface_attempts[].fixed_component.nu.denominator`
- `orientifold_reason_diagnostics.surface_attempts[].status`
- `orientifold_reason_diagnostics.surface_attempts[].reason_code`
- `orientifold_reason_diagnostics.surface_attempts[].reason`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.status`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.reason_code`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.reason`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.sigma_rays`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.sigma_dimension`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.c2_ambient_restricted`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.c2_ambient_restricted.numerator`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.c2_ambient_restricted.denominator`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.c2_surface`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.c1_surface_squared`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.c1_surface_squared.numerator`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.c1_surface_squared.denominator`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.n_s`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.invariant_lattice_basis`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.quotient_annihilator`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.quotient_surface_rays`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.quotient_surface_cones`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.surface_cone_provenance`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.surface_cone_provenance[].quotient_cone`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.surface_cone_provenance[].ambient_cones`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.boundary_ray_scales`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.boundary_ray_scales[].quotient_ray`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.boundary_ray_scales[].scale`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.reference_ambient_cone`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.restricted_divisor_coefficients`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.restricted_divisor_coefficients[].ambient_ray`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.restricted_divisor_coefficients[].coefficients`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.restricted_divisor_coefficients[].coefficients[].quotient_ray`
- `orientifold_reason_diagnostics.surface_attempts[].surface_data.restricted_divisor_coefficients[].coefficients[].coefficient`
- `orientifold_reason_diagnostics.surface_attempts[].candidate_context`
- `orientifold_reason_diagnostics.surface_attempts[].candidate_context.candidate_id`
- `orientifold_reason_diagnostics.surface_attempts[].candidate_context.torus_shift`
- `orientifold_reason_diagnostics.surface_attempts[].candidate_context.torus_shift.numerator`
- `orientifold_reason_diagnostics.surface_attempts[].candidate_context.torus_shift.denominator`
- `orientifold_reason_diagnostics.surface_attempts[].candidate_context.lambda_f`
- `orientifold_reason_diagnostics.surface_attempts[].candidate_context.h11_plus`
- `orientifold_reason_diagnostics.surface_attempts[].candidate_context.h11_minus`
- `orientifold_reason_diagnostics.surface_attempts[].candidate_context.candidate_terminal_status`
- `orientifold_reason_diagnostics.unresolved_components[].polytope_index`
- `orientifold_reason_diagnostics.unresolved_components[].frst_class_index`
- `orientifold_reason_diagnostics.unresolved_components[].matrix_id`
- `orientifold_reason_diagnostics.unresolved_components[].candidate_id`
- `orientifold_reason_diagnostics.unresolved_components[].lattice_matrix`
- `orientifold_reason_diagnostics.unresolved_components[].torus_shift`
- `orientifold_reason_diagnostics.unresolved_components[].torus_shift.numerator`
- `orientifold_reason_diagnostics.unresolved_components[].torus_shift.denominator`
- `orientifold_reason_diagnostics.unresolved_components[].lambda_f`
- `orientifold_reason_diagnostics.unresolved_components[].h11_plus`
- `orientifold_reason_diagnostics.unresolved_components[].h11_minus`
- `orientifold_reason_diagnostics.unresolved_components[].candidate_terminal_status`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component.sigma_rays`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component.sigma_dimension`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component.nu`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component.nu.numerator`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component.nu.denominator`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component.nu_binary_source`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component.integrality_vector`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component.fixed_toric_dimension`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component.f_vanishes_identically`
- `orientifold_reason_diagnostics.unresolved_components[].fixed_component.hypersurface_component_dimension`
- `orientifold_reason_diagnostics.unresolved_components[].reason_code`
- `orientifold_reason_diagnostics.unresolved_components[].reason`
- `orientifold_reason_diagnostics.certified_surfaces[].polytope_index`
- `orientifold_reason_diagnostics.certified_surfaces[].frst_class_index`
- `orientifold_reason_diagnostics.certified_surfaces[].matrix_id`
- `orientifold_reason_diagnostics.certified_surfaces[].lattice_matrix`
- `orientifold_reason_diagnostics.certified_surfaces[].fixed_component`
- `orientifold_reason_diagnostics.certified_surfaces[].fixed_component.sigma_rays`
- `orientifold_reason_diagnostics.certified_surfaces[].fixed_component.nu`
- `orientifold_reason_diagnostics.certified_surfaces[].fixed_component.nu.numerator`
- `orientifold_reason_diagnostics.certified_surfaces[].fixed_component.nu.denominator`
- `orientifold_reason_diagnostics.certified_surfaces[].status`
- `orientifold_reason_diagnostics.certified_surfaces[].reason_code`
- `orientifold_reason_diagnostics.certified_surfaces[].reason`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.status`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.reason_code`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.reason`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.sigma_rays`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.sigma_dimension`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.c2_ambient_restricted`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.c2_ambient_restricted.numerator`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.c2_ambient_restricted.denominator`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.c2_surface`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.c1_surface_squared`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.c1_surface_squared.numerator`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.c1_surface_squared.denominator`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.n_s`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.invariant_lattice_basis`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.quotient_annihilator`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.quotient_surface_rays`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.quotient_surface_cones`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.surface_cone_provenance`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.surface_cone_provenance[].quotient_cone`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.surface_cone_provenance[].ambient_cones`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.boundary_ray_scales`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.boundary_ray_scales[].quotient_ray`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.boundary_ray_scales[].scale`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.reference_ambient_cone`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.restricted_divisor_coefficients`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.restricted_divisor_coefficients[].ambient_ray`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.restricted_divisor_coefficients[].coefficients`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.restricted_divisor_coefficients[].coefficients[].quotient_ray`
- `orientifold_reason_diagnostics.certified_surfaces[].surface_data.restricted_divisor_coefficients[].coefficients[].coefficient`
- `orientifold_reason_diagnostics.certified_surfaces[].candidate_context`
- `orientifold_reason_diagnostics.certified_surfaces[].candidate_context.candidate_id`
- `orientifold_reason_diagnostics.certified_surfaces[].candidate_context.torus_shift`
- `orientifold_reason_diagnostics.certified_surfaces[].candidate_context.torus_shift.numerator`
- `orientifold_reason_diagnostics.certified_surfaces[].candidate_context.torus_shift.denominator`
- `orientifold_reason_diagnostics.certified_surfaces[].candidate_context.lambda_f`
- `orientifold_reason_diagnostics.certified_surfaces[].candidate_context.h11_plus`
- `orientifold_reason_diagnostics.certified_surfaces[].candidate_context.h11_minus`
- `orientifold_reason_diagnostics.certified_surfaces[].candidate_context.candidate_terminal_status`
- **I5 = D ∧ A ∧ K.** `reason_diagnostics` and its 4 keys are written in every
  element (`e7a8f51:REPRO:499-508`, `:650-656`, `:1126`).
  Adds to R (5 paths):
- `details[].orientifold_action_audit.reason_diagnostics`
- `details[].orientifold_action_audit.reason_diagnostics.polytope_index`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces`

  Adds to O (127 paths). These are the rows, which can be empty. The row
  layouts are the same as in I4:
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].polytope_index`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].frst_class_index`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].matrix_id`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].lattice_matrix`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].fixed_component`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].fixed_component.sigma_rays`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].fixed_component.nu`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].fixed_component.nu.numerator`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].fixed_component.nu.denominator`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].status`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].reason_code`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].reason`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.status`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.reason_code`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.reason`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.sigma_rays`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.sigma_dimension`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.c2_ambient_restricted`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.c2_ambient_restricted.numerator`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.c2_ambient_restricted.denominator`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.c2_surface`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.c1_surface_squared`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.c1_surface_squared.numerator`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.c1_surface_squared.denominator`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.n_s`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.invariant_lattice_basis`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.quotient_annihilator`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.quotient_surface_rays`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.quotient_surface_cones`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.surface_cone_provenance`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.surface_cone_provenance[].quotient_cone`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.surface_cone_provenance[].ambient_cones`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.boundary_ray_scales`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.boundary_ray_scales[].quotient_ray`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.boundary_ray_scales[].scale`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.reference_ambient_cone`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.restricted_divisor_coefficients`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.restricted_divisor_coefficients[].ambient_ray`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.restricted_divisor_coefficients[].coefficients`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.restricted_divisor_coefficients[].coefficients[].quotient_ray`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].surface_data.restricted_divisor_coefficients[].coefficients[].coefficient`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].candidate_context`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].candidate_context.candidate_id`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].candidate_context.torus_shift`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].candidate_context.torus_shift.numerator`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].candidate_context.torus_shift.denominator`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].candidate_context.lambda_f`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].candidate_context.h11_plus`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].candidate_context.h11_minus`
- `details[].orientifold_action_audit.reason_diagnostics.surface_attempts[].candidate_context.candidate_terminal_status`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].polytope_index`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].frst_class_index`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].matrix_id`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].candidate_id`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].lattice_matrix`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].torus_shift`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].torus_shift.numerator`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].torus_shift.denominator`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].lambda_f`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].h11_plus`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].h11_minus`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].candidate_terminal_status`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component.sigma_rays`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component.sigma_dimension`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component.nu`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component.nu.numerator`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component.nu.denominator`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component.nu_binary_source`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component.integrality_vector`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component.fixed_toric_dimension`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component.f_vanishes_identically`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].fixed_component.hypersurface_component_dimension`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].reason_code`
- `details[].orientifold_action_audit.reason_diagnostics.unresolved_components[].reason`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].polytope_index`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].frst_class_index`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].matrix_id`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].lattice_matrix`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].fixed_component`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].fixed_component.sigma_rays`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].fixed_component.nu`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].fixed_component.nu.numerator`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].fixed_component.nu.denominator`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].status`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].reason_code`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].reason`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.status`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.reason_code`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.reason`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.sigma_rays`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.sigma_dimension`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.c2_ambient_restricted`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.c2_ambient_restricted.numerator`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.c2_ambient_restricted.denominator`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.c2_surface`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.c1_surface_squared`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.c1_surface_squared.numerator`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.c1_surface_squared.denominator`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.n_s`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.invariant_lattice_basis`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.quotient_annihilator`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.quotient_surface_rays`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.quotient_surface_cones`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.surface_cone_provenance`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.surface_cone_provenance[].quotient_cone`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.surface_cone_provenance[].ambient_cones`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.boundary_ray_scales`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.boundary_ray_scales[].quotient_ray`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.boundary_ray_scales[].scale`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.reference_ambient_cone`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.restricted_divisor_coefficients`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.restricted_divisor_coefficients[].ambient_ray`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.restricted_divisor_coefficients[].coefficients`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.restricted_divisor_coefficients[].coefficients[].quotient_ray`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].surface_data.restricted_divisor_coefficients[].coefficients[].coefficient`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].candidate_context`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].candidate_context.candidate_id`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].candidate_context.torus_shift`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].candidate_context.torus_shift.numerator`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].candidate_context.torus_shift.denominator`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].candidate_context.lambda_f`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].candidate_context.h11_plus`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].candidate_context.h11_minus`
- `details[].orientifold_action_audit.reason_diagnostics.certified_surfaces[].candidate_context.candidate_terminal_status`
- **D without A.** The help says "Requires --orientifold-audit"
  (`e7a8f51:REPRO:1281-1282`), but nothing checks it (`main` `:1263-1361`,
  `reproduce` `:984-1260`). D alone gives the ΔD R paths with all lists empty
  and all maps except `reason_counts_by_h11` empty. `details[].orientifold_action_audit`
  stays `null`, so D ∧ K without A adds nothing under `details`.
- **D without K.** Only the top-level summary. `details` stays `null`.
- No mutually exclusive flags.

##### Structure-neutral flags

Same as group 1: `--parquet-dir`, `--limit`, `--progress`, `--output`, `--gs`,
`--w0-real`, `--w0-imag`, `--qcd-divisor-domain`, `--julia-binary`,
`--julia-project`, the exact `--h11` value within each class, A without K and
without D, E without K or when M is on.

##### K-dependent fields

The same as group 1 (at `e7a8f51:REPRO:877-899`, `:1171-1175`, `:1234-1244`).
All `orientifold_reason_diagnostics.*` and `details[].orientifold_action_audit.*`
paths are K-free: they are toric and lattice data from IOC.

##### Layout count

36 distinct layouts: 2 (T) × [6 with K off: M off/on × {D off (A has no effect),
D ∧ ¬A, D ∧ A} + 12 with K on: {none, E only, M} × (A, D) ∈ {off,on}²]. Each
layout is the group 1 formula with these additions:
- base R ∪ {`orientifold_reason_diagnostics`};
- R ∪ ΔD.R if D;
- O ∪ I4 if D ∧ A;
- R ∪ I5.R and O ∪ I5.O if D ∧ A ∧ K.

| # | `--h11` has target | `--keep-details` | export (none / `--export-kaehler-points` / `--model-stage`) | `--orientifold-audit` | `--orientifold-reason-diagnostics` | \|R\| | \|O\| |
|---|---|---|---|---|---|---|---|
| 2.1 | yes | off | none or E only | any | off | 40 | 0 |
| 2.2 | yes | off | none or E only | off | on | 56 | 0 |
| 2.3 | yes | off | none or E only | on | on | 56 | 127 |
| 2.4 | yes | off | M (E any) | any | off | 53 | 1 |
| 2.5 | yes | off | M (E any) | off | on | 69 | 1 |
| 2.6 | yes | off | M (E any) | on | on | 69 | 128 |
| 2.7 | yes | on | none | off | off | 65 | 33 |
| 2.8 | yes | on | none | off | on | 81 | 33 |
| 2.9 | yes | on | none | on | off | 68 | 33 |
| 2.10 | yes | on | none | on | on | 89 | 287 |
| 2.11 | yes | on | E only | off | off | 65 | 83 |
| 2.12 | yes | on | E only | off | on | 81 | 83 |
| 2.13 | yes | on | E only | on | off | 68 | 83 |
| 2.14 | yes | on | E only | on | on | 89 | 337 |
| 2.15 | yes | on | M (E any) | off | off | 78 | 90 |
| 2.16 | yes | on | M (E any) | off | on | 94 | 90 |
| 2.17 | yes | on | M (E any) | on | off | 81 | 90 |
| 2.18 | yes | on | M (E any) | on | on | 102 | 344 |
| 2.19 | no | off | none or E only | any | off | 34 | 0 |
| 2.20 | no | off | none or E only | off | on | 50 | 0 |
| 2.21 | no | off | none or E only | on | on | 50 | 127 |
| 2.22 | no | off | M (E any) | any | off | 47 | 1 |
| 2.23 | no | off | M (E any) | off | on | 63 | 1 |
| 2.24 | no | off | M (E any) | on | on | 63 | 128 |
| 2.25 | no | on | none | off | off | 59 | 33 |
| 2.26 | no | on | none | off | on | 75 | 33 |
| 2.27 | no | on | none | on | off | 62 | 33 |
| 2.28 | no | on | none | on | on | 83 | 287 |
| 2.29 | no | on | E only | off | off | 59 | 83 |
| 2.30 | no | on | E only | off | on | 75 | 83 |
| 2.31 | no | on | E only | on | off | 62 | 83 |
| 2.32 | no | on | E only | on | on | 83 | 337 |
| 2.33 | no | on | M (E any) | off | off | 72 | 90 |
| 2.34 | no | on | M (E any) | off | on | 88 | 90 |
| 2.35 | no | on | M (E any) | on | off | 75 | 90 |
| 2.36 | no | on | M (E any) | on | on | 96 | 344 |

Row 2.N is layout `1.0-G6-LNN` in `scripts/f5_legacy_layouts_v10b.py` (row 1.N in group 5 is
`1.0-G5-LNN`). Revision 3 changed only the D-on rows: |R| is 1 lower (no
`reason_counts_by_h11.<map>`), and |O| is 7 lower when A is also on (no bare
map levels). The layout count, 36, is unchanged.

Empty-details coverage, checked mechanically: a K-on file with `details: []`
matches these layouts:
- 2.1 (for 2.7, 2.9, 2.11, 2.13);
- 2.2 and 2.3 (for 2.8, 2.10, 2.12, 2.14);
- 2.4 (for 2.15, 2.17);
- 2.5 and 2.6 (for 2.16, 2.18);
- 2.19, 2.20/2.21, 2.22, 2.23/2.24 in the same pattern for 2.25-2.36.

---

### Open points

1. **Is `[]` a path level of its own?** A2 does not say. This record emits no
   path that ends in `[]`. An array key gives only `key`, and its object
   elements give `key[].k`. If the reader's normalizer also emits `key[]` for
   non-empty arrays (or for all arrays), add that path wherever the record has
   `key[].…`, and make it optional where `key` can be `[]` or `null`. Whichever
   choice is made, use it for all F5 records.
2. **Is `<map>` a path level of its own?** Resolved in revision 3: no.
   - This follows the T061 API `normalize`. Revisions 1 and 2 emitted
     `X.<map>` for non-empty maps; revision 3 drops those paths (1 in R, 7 in
     O) and lists `reason_counts_by_h11.<map>` as a nested map node.
   - The layout counts (16, 36) and the overlapping pairs in point 5 are
     unchanged.
   - D ∧ ¬A and D ∧ A (K off) still differ by the 127 row paths in O.
3. **`diagnostic.checks` keys.** The check names in
   `details[].kaehler_point_export_per_class[].diagnostic.checks` are fixed
   names from code (`fc66261:GEN:1524-1526`, `:1581-1636`), so this record
   treats them as fields, not a map. A reviewer could see them as "a count by
   reason"-like map of name → bool. If so, make `...diagnostic.checks` a map
   node and replace the 16 `checks.*` paths with `checks.<map>`.
4. **Nested arrays.** `lattice_matrix`, `inverse_metric`, `glsm_charge_matrix`,
   `unavailable_cones`, `sigma_rays`, `quotient_surface_cones`,
   `surface_cone_provenance[].ambient_cones`, `invariant_lattice_basis`,
   `quotient_annihilator` and the numerator arrays are arrays of arrays (or
   deeper) of scalars. A2 names "arrays of scalars" only. This record treats
   any array that reaches no object as adding nothing below its key. No array
   of arrays of **objects** occurs in this range.
5. **Overlapping layouts in this range (a file can match more than one).** All
   layouts give the same inferred label, so this does not change the result,
   but a reader that wants a unique layout cannot get one.
   - I computed the pairs mechanically: two layouts overlap when
     R1 ∪ R2 ⊆ (R1 ∪ O1) ∩ (R2 ∪ O2). Under the revision 2 R convention the
     overlapping pairs are:
     - group 1: 1.3~1.5, 1.4~1.6, 1.11~1.13, 1.12~1.14. These are K-on with no
       export vs `--export-kaehler-points`: a file with no exported class
       matches both, because `kaehler_point_export_per_class` is a list of
       `null` or `null` in both.
     - group 2: 2.7~2.11, 2.8~2.12, 2.9~2.13, 2.10~2.14 and 2.25~2.29 …
       2.28~2.32. These are the same K-on export pairs.
     - group 2: 2.2~2.3, 2.5~2.6, 2.20~2.21, 2.23~2.24. These are K-off
       D ∧ ¬A vs D ∧ A: a file whose reason lists are all empty matches both.
       Under revision 3, map contents never add paths. Rechecked on the
       `scripts/f5_legacy_layouts_v10b.py` encoding; the pairs are the same.
   - Each empty-`details` K-on file matches the K-off layout(s) listed under
     the tables. That is intended under the R convention.
   - A ∧ K vs ¬A ∧ K no longer overlap: `details[].orientifold_action_audit.*`
     is R with A.
   - Group 1 and group 2 do not overlap: every group 2 file has
     `orientifold_reason_diagnostics` (base R), which is in no group 1 R ∪ O.
   - Group 1 M-on and `47b1d74` M-on layouts do not overlap: the two new
     `convention` keys are in group 1 R and, if the other record is right,
     absent from the `47b1d74` R ∪ O.
   - Group 1 M-off layouts are the same path sets as the `47b1d74` M-off
     layouts (see the group 1 intro). A file from either commit matches both.
     That is correct and needs no tie-break.
6. **`--limit ≤ 0`.** `load_mirror_polytopes` appends first and then checks
   `len(records) >= limit` (`fc66261:GEN:4075-4096`). So `--limit 0` still
   writes one polytope when any exists. `details` is `[]` only when the
   mirror has no favorable polytope for that `--h11`. Under the revision 2 R
   convention, that file is matched by the K-off layout (see the header).
   Revision 1 of this record had all `details[].*` paths in O.
7. **Value-dependent object vs scalar.** `surface_data.c2_ambient_restricted`
   and `surface_data.c1_surface_squared` are an int or a
   `{numerator, denominator}` object (`e7a8f51:IOC:722-728`), so their
   `.numerator` / `.denominator` paths are O.
   - `surface_data.c2_surface` is `Fraction(len(surface_cones))`, always an int
     (`e7a8f51:IOC:1880`).
   - `surface_data.n_s` is always an int, because a non-integer `n_s` is
     recorded as unavailable instead (`:1895-1902`).
   - `restricted_divisor_coefficients[].coefficients[].coefficient` is always
     an int, because a non-integer coefficient raises (`:1864-1867`).
   - I found no `.numerator` path for these three. If a reviewer reads the
     code differently, add them to O.
8. **`--h11` as a "flag".** I treat the class of the `--h11` value (with or
   without a Table 1 row) as a structure-changing flag, because it changes R
   (6 `paper_targets.*` paths). This doubles the counts (8 → 16, 18 → 36). The
   other choice is to put the 6 paths in O in one layout. That accepts a
   strict superset of files: a partial `paper_targets`, which the code cannot
   write. Both records of the 1.0 version should make the same choice.
9. **Not assigned, but in range.** `4d2fc3b`, `4c3f7c5`, `1620b5a`, `f3bf677`
   and `de1220b` were not assigned to me. They are 1.0 writers under A2's
   range rule. I put them in group 1 after checking their blobs (see the
   coverage table). If plan.md lists only the assigned commits, these five are
   missing from it.
10. **IOC fields that are not serialized.** IOC candidate records have many
    more keys (`auxiliary_fan`, `smoothness`, `fixed_point_set`, …;
    `e7a8f51:IOC:2416-2457`). Only the keys copied explicitly into
    `candidate_context`, the unresolved rows and `fixed_component` reach the
    JSON. In group 2, rows store `fixed_component` by reference, but no
    `classify_smoothness` code changes component dicts (it builds new dicts at
    `e7a8f51:IOC:2157-2161`). I did not run the writer. All layouts here come
    from reading the source.
