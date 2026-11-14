# CYAX-0232 Clarification B — G1 owner decisions

**Status:** Approved
**Approved:** 2026-11-14
**Approval provenance:** GitHub Issue #232 comment `6871292308`
(https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/232#issuecomment-6871292308),
approving the text posted in comment `6871256546`
(https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/232#issuecomment-6871256546)
**Review:** no separate G0 review. B1, B3, B5 and B6 ratify behaviour that
was already implemented and verified at G1 (code candidate `90f6f43`). B2
and B4 add labels only. The G2 fresh-context review (T093) of the whole PR
covers this clarification and its implementation.
**Amends:** `spec.md` approved at `f097536` and Clarification A at `1ada61d`.
Where they conflict, this clarification takes precedence.
**Origin:** `plan.md`, *Open owner decisions at G1* (G1-1 to G1-7), and the
plan's R-005 fixture substitution.

The text below is the approved comment, unchanged.

**B1 — Ratified as implemented.**
- G1-1: `orientifold_population_preflight.py` accepts an inferred legacy
  `shard-merge-1.0` merge, because it reads only population counts, which do
  not depend on K. It returns the label in its result.
- G1-2: the R-002 and R-005 harness dumps are verification tools and carry no
  label.
- G1-5: `trajectory_observables` carries no label; the trajectory it is given
  carries its own label.

**B2 — Comparison CSV labelled per field (extends A7 to a persisted file; G1-3).**
`scripts/inflation_reproduction.jl` `write_comparison` gains one label column
per K-dependent value column. Each label column names the contract of that
value in that row:

| Value column | Label column | N5 rows | N8_poly102 rows |
|---|---|---|---|
| `reduced_model_efolds` | `reduced_model_efolds_kinetic_normalization` | `a96_reproduction_raw_radian` (author normal form) | `p96_reproduction_c_half` (`n8_local_hilltop_efolds`) |
| `normal_form_efolds` | `normal_form_efolds_kinetic_normalization` | `a96_reproduction_raw_radian` | `a96_reproduction_raw_radian` |
| `independent_physical_flow_efolds` | `independent_physical_flow_efolds_kinetic_normalization` | empty (no value) | `a96_reproduction_raw_radian` |
| `reference_efolds` | `reference_efolds_kinetic_normalization` | `a96_reproduction_raw_radian` | `a96_reproduction_raw_radian` |

The file has no row-level `kinetic_normalization` column. Any physical reader
must reject it, as A7 requires for in-memory results.

**B3 — Normal-form functions (G1-6).** `n5_hilltop_efolds`,
`poly102_inflation.n8_hilltop_efolds` and `reference_efolds` remain K-free
under A4 and return no label. Their docstrings name the A96 author source of
their constants.

**B4 — Committed `validation/` files (G1-4).** These three files are
regenerated with the current writers once B2 is implemented:
- `validation/inflation_fixtures.jls`
- `validation/inflation_comparison.csv`
- `validation/author_poly102_reference_miniforge.json`

The regeneration must show that every value is unchanged and that only the
labels are added. This is reproduction output, so it falls outside the spec's
non-scope on regenerating spectra and scans.

**B5 — R-005 inflation and F6 fixture.** The R-005 inflation and F6 rows are
verified on the `infl` fixture (`Q = [1 0 1; 0 1 1]`, `L = (−1, −2, −3)`,
`Kinv = [3 1; 1 2]`), not on the R-004 or R-003 pins. Those pins have leading
scales 50 decades apart, so the Float64 eigen route of `run_geometry` returns
a light mass dominated by roundoff in both conventions. The gated factors on
`infl` match the R-005 table (`specs/0232-kinetic-normalization/evidence/r005_report.md`).

**B6 — Out of scope (G1-7).** The default tolerances of
`n8_author_trajectory` (a `reltol` below the working precision) are filed as a
separate Issue (#234). This PR does not change them.

## Implementation note (B4)

The committed `validation/inflation_comparison.csv` predates commit `828d2ff`,
which renamed the writer's value columns. Its header is
`package_local_efolds, author_efolds, author_independent_efolds, draft_efolds`.
The current names are `reduced_model_efolds, normal_form_efolds,
independent_physical_flow_efolds, reference_efolds`. The B4 regeneration
therefore also renames these four columns, which is pre-existing drift. The
value-unchanged check compares the columns in order under this mapping.
