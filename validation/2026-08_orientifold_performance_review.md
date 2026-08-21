# Inherited-orientifold audit — performance review and optimisation

**Date:** 2026-08-20
**Branch / worktree:** `codex/orientifold-overcount-20260819` (`CYAxiverse-orientifold-overcount`), base HEAD `2ae6bc8`.
**Scope:** make the inherited-orientifold Table 1 audit
(`scripts/reproduce_fuzzy_axions_h11_4.py` +
`scripts/inherited_orientifold_candidates.py`) fast enough to run the
**exhaustive** population for `h11 = 2 … 10`. Large `h11` is to be **sampled**
(implementation deferred; see §F7).

**Method — measured, not estimated.** A 2- and 6-polytope `h11 = 4` slice
under `cProfile` in the `cytools` conda env, plus the completed full runs
(`h11 = 4`: 1760 FRST classes, ~2 h 26 m single-core; 86,156 terminal records,
61,106 = 71% `fixed_point_set_non_smooth`). Every optimisation below was
verified to leave the terminal ledger **byte-for-byte identical** on real
KS geometries (the audit is exact integer/rational arithmetic; none of these
changes touch a result).

---

## Baseline hot path (pre-optimisation, 2-polytope `h11=4`, 61.4 s audit)

| Function | cumtime | note |
|---|---|---|
| `_fixed_component_records` | 32.1 s (52%) | via `_half_ray_shortcut_proof` |
| `_exact_determinant` (sympy `.det()`) | 23.9 s | 64,744 calls, ≤4×4 int matrices |
| `classify_smoothness` + `_positive_component_section_certificate` | 11.5 s | sympy exact arithmetic |
| `enumerate_polytope_involutions` | 9.7 s | **4 calls** — n⁴ sympy, recomputed per FRST class |
| `_exact_rank` (sympy `.rank()`) | 9.2 s | 22,465 calls |

Signature of the problem: **306 M calls for two polytopes**, almost all sympy
exact-matrix internals on tiny integer matrices — avoidable without changing
any result.

---

## Findings and status

### F0 — Fast exact integer determinant / rank — **DONE, verified**
`_exact_determinant` and `_exact_rank`
(`inherited_orientifold_candidates.py:99`, `:88`) called sympy on ≤4×4 integer
matrices. Replaced with a pure-Python-int **Bareiss** determinant and a
**Fraction** Gaussian-elimination rank. Bit-for-bit identical to sympy;
measured **54×** (det) and **11×** (rank) faster. Roughly halves total runtime.

### F2 — Automorphism-group involutions — **DONE, verified**
`enumerate_polytope_involutions` brute-forced `itertools.product(points,
repeat=4)` (n⁴) with sympy per iteration, recomputed per FRST class. Replaced
with `poly.automorphisms(square_to_one=True, action='left')` when a CYTools
`poly` is available (the production path); the brute force is retained as the
fixture/reference path. Verified to yield the **identical involution set** on
12 real `h11=4` polytopes, **121×** faster (1292 ms → 11 ms). This also removes
the n⁴ wall that would dominate at large `h11` (point count grows with `h11`).

### F3 — Per-matrix memoisation of shift-invariant Cartier checks — **DONE, verified**
`_half_ray_shortcut_proof` (18.3 s, 31,672 calls) and
`_positive_component_section_certificate` (32.7 s) depend only on
`(auxiliary_fan, matrix, sigma_rays[, nu])` — invariant across a matrix's 16
torus shifts and both `lambda_f`. Added per-matrix caches
(`half_ray_cache`, `section_cache` in `enumerate_orientifold_candidates`,
threaded into `_fixed_component_records` and `classify_smoothness`).
Bit-for-bit identical.

**Cumulative F0+F2+F3 (6-polytope `h11=4`, unprofiled, byte-identical output):
65.0 s → 25.8 s wall (≈2.5×); audit compute ≈ 61 s → 22 s (≈2.8×).** The
speedup grows with `h11` as F2's automorphism win scales with point count.

### F5 — Process-level parallelism (external shard + merge) — **DONE, verified**
The `reproduce()` loop is embarrassingly parallel over favorable polytopes
(each independent; `load_mirror_polytopes` provenance carries
`parquet_file`+`row_index`). Added `--shard-count N` / `--shard-index i`:
a deterministic **strided** partition over the global favorable-polytope index,
so each shard processes a disjoint subset and writes its own auto-suffixed
`*.shardNNN-of-MMM.*` output and terminal ledger (immutability model
untouched). `scripts/merge_orientifold_shards.py` recombines the per-shard
summaries — additive population counts, union of the geometry-keyed
(`polytope_normal_form_id`, `frst_class_index`) class funnels — and
re-evaluates completeness and Table 1 claim status against the merged totals.
`--shard-count 1` (default) is a byte-for-byte no-op. Works on one multicore
box (`&`) and on SLURM job arrays.

### F4 — Vectorise the O(n²) tensor loops — **pending**
`_n_s_for_two_ray_cone` (`:863`) and `_frozen_conifold_diagnostic`
(`reproduce…:568`) sum the ambient intersection tensor with pure-Python
`for r: for s:`. Replace with `tensor[p,q].sum()` etc. Low priority at
`h11 ≤ 10` (small tensors); becomes relevant with F7.

### F6 — Integer HNF membership — **pending, minor**
`_integer_lattice_membership` (`:205`) uses sympy HNF (~1.3 s). Fold into the
F0 integer-arithmetic pass if convenient.

### F7 — Large-`h11` scaling ceilings — **deferred to a sampling mode**
For the exhaustive `h11 ≤ 10` target these are handled by F0/F2/F5. They are
genuine algorithmic walls for large `h11` and are **not** removed by any
constant-factor work; the plan is a separate **sampling** mode (deferred):
1. `_frst_classes` (`reproduce…:446`) materialises all FRSTs
   (`all_triangulations(as_list=True)`) then does O(raw²) pairwise
   `is_equivalent`. The raw FRST count explodes with `h11`.
2. Dense n⁴ ambient intersection tensor (`_ambient_intersection_tensor`):
   ~68 MB at `h11=50`, infeasible at `h11=491`. Switch to sparse intersection
   numbers + the F4 vectorised sums.

---

## Verification protocol used

For each of F0/F2/F3/F5: (a) `python -m unittest` on the three orientifold
test modules stays green (66 tests); (b) a real `h11=4` slice run produces a
terminal-ledger JSONL that is **byte-identical** to the pre-change run
(`diff` empty); F5 additionally verified by an `h11=2` single-run vs
2-shard-plus-merge equivalence. Changes are performance-only and carry **no
version impact** to scientific results; the reproduction/ledger schema
versions are unchanged.
