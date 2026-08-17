# Tolerance-aware Hessian mode counts — measured output change

**Date:** 2026-08-17
**Change:** `generate.spectrum_mode_counts`, applied at four classification sites
**Screen:** `scripts/analyze_inflation_candidates.jl`, `reduction=:leading_branches`, default six geometries
**Determinism:** `minimizer.critical_points` seeds from a radical-inverse (Halton) sequence, not `rand`, so the screen is exactly reproducible and every difference below is attributable to this change alone.

---

## 1. What changed

Previously, at four sites:

```julia
negative_modes = count(<(0), eigenvalues)
zeroish_modes  = count(x -> abs(x) <= 1e-10 * scale, eigenvalues)
positive_modes = count(>(0), eigenvalues)
```

`negative_modes` and `positive_modes` were untoleranced sign tests, so an eigenvalue zero to within rounding was assigned to whichever side its last bit fell on. Only `zeroish_modes` knew about a tolerance, and it **overlapped** the other two rather than partitioning with them.

Now all three come from one helper that partitions the spectrum:

```julia
modes = CYAxiverse.generate.spectrum_mode_counts(eigenvalues)   # negative + zeroish + positive == length
```

Sites updated:

| file | function |
|---|---|
| `scripts/analyze_inflation_candidates.jl` | `classify_point` |
| `scripts/inflation_scan_common.jl` | `_classify_point` |
| `scripts/inflation_scan_common.jl` | `_classify_point!` (hot path) |
| `scripts/inflation_scale_continuation.jl` | `_continuation_classification` |

---

## 2. The old counts were inconsistent in 100% of recorded output

Checked against the committed artifact `paper_benchmarks/2023_minima/inflation_screen/candidate_critical_points.csv`, which stores all three counts per point:

| check | result |
|---|---|
| rows where `negative + zeroish + positive > h11` | **40,608 of 40,608 (100%)** |
| rows with `zeroish_modes > 0` | **40,608 of 40,608 (100%)** |
| rows where `negative_modes` exceeds the resolvable count `h11 - zeroish` | **37,737 of 38,006 saddle rows (99.29%)** |

For `h11 = 11` the modal case is **10 of 11 eigenvalues inside the noise band** while 6 are reported as tachyons — at most 1 could be genuine. Summing over saddle rows, at least **151,305 of 204,886 (73.8%)** reported tachyonic modes were provably numerically zero.

This is measured on real recorded output, not synthetic fixtures.

---

## 3. Measured effect on the screen

### Mode counts

| quantity | before | after | |
|---|---|---|---|
| rows satisfying `negative + zeroish + positive == h11` | **0 / 32,416** | **32,416 / 32,416** | invariant now holds |
| summed `negative_modes` over all points | 172,728 | **30,448** | **17.6%** retained |
| points where `negative_modes` decreased | — | **32,416 of 32,416** | every point |
| points where `negative_modes` increased | — | **0** | |

**82% of reported tachyonic modes were numerical noise.** The direction is uniform: not one point gained a negative mode.

### Saddle classification

| geometry | branches | saddles before | saddles after | retained |
|---|---|---|---|---|
| 5, 1, 1 | 160 | 159 | 140 | 88.1% |
| 9, 1, 1 | 2,560 | 2,559 | 2,520 | 98.5% |
| 10, 1, 1 | 5,120 | 5,100 | 3,840 | 75.3% |
| 11, 1, 1 | 2,048 | 2,040 | 1,024 | 50.2% |
| 11, 2, 1 | 10,240 | 10,235 | 7,680 | 75.0% |
| 11, 7, 1 | 12,288 | 12,282 | 6,144 | 50.0% |
| **total** | | **32,375** | **21,348** | **65.9%** |

The old filter was nearly vacuous — 159 of 160 branches were "saddles".

### The flatness metric was pure noise

`abs_min_eta = minimum(abs.(eta_values))` is the flatness figure of merit, and any curvature-free direction drives it to zero. Comparing it against the same quantity restricted to resolvable directions:

| geometry | `best_abs_min_eta` | `best_abs_min_eta_resolvable` |
|---|---|---|
| 5, 1 | 1.86e-11 | 3.91 |
| 9, 1 | 1.48e-07 | 9.97e+05 |
| 10, 1 | 1.20e-38 | 1.91e-03 |
| 11, 1 | 3.56e-11 | 9.05e+06 |
| 11, 2 | **6.32e-161** | 75.5 |
| 11, 7 | 5.54e-94 | 1.99e+07 |

Median over all 32,416 points: `abs_min_eta` = **1.00e-94**, resolvable = **1.99e+07**. That is a **101-order-of-magnitude** gap.

Read literally, the old column said essentially every critical point was perfectly flat and therefore an ideal inflation candidate. The resolvable value says median |η| ≈ 2e7 — nowhere near slow roll.

---

## 4. No candidate is gained or lost

The headline candidate counter is **unchanged in all six geometries**:

| geometry | `candidate_slowroll_saddles` before | after | `leading_minima_count` before | after |
|---|---|---|---|---|
| 5, 1 | 0 | 0 | 5 | 5 |
| 9, 1 | 0 | 0 | 5 | 5 |
| 10, 1 | 0 | 0 | 5 | 5 |
| 11, 1 | 0 | 0 | 1 | 1 |
| 11, 2 | 0 | 0 | 5 | 5 |
| 11, 7 | 0 | 0 | 6 | 6 |

`candidate_slowroll_saddles` gates on `epsilon < 1 && abs(min_eta) < 1`, and `min_eta` is large and negative throughout, so no point passed either before or after. **No published candidate list changes.** What changes is `saddle_count`, `negative_modes`, and the selected `least_tachyonic`/`best`/`flattest` rows in 2 of 6 geometries.

---

## 5. Columns

Added, both additive:

- `abs_min_eta_resolvable` (points and summary as `best_abs_min_eta_resolvable`) — `min |η|` over directions with resolvable curvature.
- `zeroish_mode_points` (summary) — points with at least one curvature-free direction. Currently **100% of points in all six geometries**.

`abs_min_eta`, `min_eta` and `max_eta` keep their existing definitions. Redefining η is a physics decision — whether a curvature-free direction is a genuine flat modulus or noise is not decidable from the spectrum — so both quantities are reported side by side rather than one silently replacing the other.

**Resolution (2026-08-17, DECISION-ETA):** the maintainer decided `min_eta`/`abs_min_eta` should exclude directions with no resolvable curvature. `fix/eta-resolvable-metric` redefines both accordingly and drops the now-redundant `abs_min_eta_resolvable`/`best_abs_min_eta_resolvable` columns; `max_eta` is unchanged. The measurements above remain the historical record of the side-by-side comparison that motivated the decision.

---

## 6. The committed artifact is not a clean baseline

`paper_benchmarks/2023_minima/inflation_screen/*.csv` records `enumeration = leading_half_integer_branches`, a name the current script no longer emits, and its `h11=11, polytope=1` row has `branch_count = 10240` where the current code produces `2048`. Four of six rows reproduce exactly; two do not.

**These files were therefore not regenerated by this change**, to avoid folding a pre-existing provenance drift into this diff. The before/after above was generated fresh from the parent commit and this branch with identical settings. The drift is worth a separate look.

---

## 7. Reproduction

```
CYAXIVERSE_DATA_DIR=<corpus> CYAXIVERSE_INFLATION_REDUCTION=leading_branches \
  julia --project=. scripts/analyze_inflation_candidates.jl
```

Run on the parent commit for "before" and on this branch for "after". Deterministic, so the CSVs are byte-comparable. Run unsandboxed on the local host per `.copilot/AGENTS.md` section 0.
