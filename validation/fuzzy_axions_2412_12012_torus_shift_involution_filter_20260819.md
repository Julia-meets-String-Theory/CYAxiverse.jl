# Two cancelling non-identity-L torus-shift bugs; the h11=3 "gap" was an artifact

**Program:** KS axiverse / arXiv:2412.12012 Table 1 reproduction (inherited
orientifold row).
**Worktree/branch:** `../CYAxiverse-orientifold-overcount`,
`codex/orientifold-overcount-20260819`, on top of `4c3f7c5`
(post-A/B/C + Stream-A + eq.4.50 fixes).
**Environment:** conda `cytools` (Python 3.14, CYTools 1.4.12); use `python`,
not `python3`.
**Primary source:** `validation/fuzzy_axions_supp/paper_source_2305_06363/KS_orientifolds.tex`
(Moritz, arXiv:2305.06363) — the orientifold construction. Line citations below
are to that file.

## 0. Result in one table (h11=3, full population, 274 FRST classes)

| machinery state | inherited O3/O7 CYs | vs target 253 |
| --- | --- | --- |
| broken (pre-A/B/C) | 274 | +21 |
| A/B/C + Stream-A + eq.4.50 (= `4c3f7c5`) | 226 | −27 |
| + eq.4.45 parity truncation fix (this session) | 272 | +19 |
| **+ 2t∈N involution filter (this session)** | **203** | **−50** |

Lower bound (`identity_valid_o3o7_action_cy_count`) 198 and the trilayer
`h21_plus_zero_trilayer_frst_classes` 66 are **unchanged** by both fixes
(identity path and trilayer path both untouched). Target 253 sits **between**
the two individual corrections (203 and 272) — the signature of two bugs of
opposite sign that were partially cancelling.

## 1. The two bugs

Both live in the non-identity-`L` torus-shift sector of
`scripts/inherited_orientifold_candidates.py`. Neither can be seen with
`L=identity` (identity's `2t` is always integral, and its parity never
truncates), which is why identity-only validation — the only kind prior
sessions did against CYTools — never surfaced them. They only became reachable
at all once the Stream-A fix (`4d2fc3b`) stopped killing non-identity-`L`
candidates upstream at `nonintegral_h2_action`.

### Bug 1 — eq. (4.45) parity `int()` truncation (spurious rejection)

`_dual_vertex_parity_evidence` computed
`two_t = [int(2 * value) for value in torus_shift]`. After the Defect-B fix,
`torus_shift = (I+L)·bits / 4`, so for a non-identity `L` `2t = 2·torus_shift`
can be a genuine half-integer vector (e.g. `2t = (1/2,1/2,0,0)` for a
coordinate-swap `L`). `int(·)` truncated each half-integer component toward
zero, so `2t` collapsed to `(0,0,0,0)` and the eq. (4.45) parity
`(⟨2t,q⟩ + λ_f) mod 2` became `λ_f`. For O3/O7 (`λ_f = 1`) that is `1` at
every fixed dual vertex — an automatic violation. **Every** non-identity-`L`
O3/O7 candidate was auto-rejected. This is what dragged the count down to 226
and made it look like a plausible undershoot.

Fixed by keeping `2t` exact (`Fraction`) and contracting against the integer
dual vertex before any rounding.

### Bug 2 — no `2t ∈ N` involution filter (spurious acceptance)

Source line ~426-428 is explicit: requiring `L∘φ_[t]` to square to the identity
gives `φ_[2t] = id`, **i.e. `2t ∈ N`**. Line 434 restates it: "the inequivalent
classes `[2t]` are the **integer points** in the coset
`H_+^L := P_+^L(N) / 2P_+^L(N)`." `enumerate_projected_lattice_representatives`
enumerates the *full* quotient `P_+^L(N)/2P_+^L(N)`, which for a non-identity
`L` also contains non-integer cosets (`2t ∉ N`); those are **not `Z2`
involutions at all** and the source excludes them. `enumerate_orientifold_candidates`
never applied that filter, so 2,212 non-involution reps (h11=3) flowed into the
smoothness checks and some were accepted — inflating the count to 272 once
Bug 1 stopped masking them.

Fixed by skipping any shift whose representative `2t = shift["vector"]` is
non-integral, emitting a single `torus_shift_not_involution` record per
excluded coset (new terminal status; source citation in the code comment).

## 2. The 203 is exact, not over-filtered (proved and checked)

The filter tests whether the *stored* coset representative is integral. That is
equivalent to "the coset contains an integer point" because integrality is a
**coset invariant**: two representatives differ by an element of
`2P_+^L(N) = (I+L)·N`, and `I+L` is an integer matrix, so the difference lies
in `N` and is integral. Hence a coset is either all-integral or all-non-integral;
testing one rep decides the whole coset.

Verified empirically across **81 distinct real non-identity `L`** (40 h11=3
polytopes): grouping all 16 `bits` into cosets via the code's own
`_same_projected_class`, **zero** cosets had non-constant integrality
(299 integer cosets, 217 non-integer). So the filter drops exactly the
non-involution cosets and never a valid one. The 203 is the exact output of the
machinery as implemented.

## 3. What the residual 203 → 253 gap is (and is not)

Post-fix acceptance breakdown (h11=3): 198 via identity, and non-identity `L`
**rescues only 5 classes** on its own (62 classes have a valid non-identity
candidate but 57 of those also pass via identity). The paper's 253 needs 55
beyond the identity lower bound of 198.

The residual 50-class gap is **not** a bug in the implemented machinery. It is
the general-`L` orientifold construction that does not exist anywhere in this
codebase: the `L`-symmetric (possibly non-simplicial) toric fan `Σ_L` from
symmetrized heights (source §4.3, eq. ~471-489), the general-`L` `n_S` on
non-identity fixed surfaces (source line 649-654; the current
`identity_fixed_surface_n_s_table` is identity-only by construction), and the
fixed-locus smoothness of source line 656-657 for `L ≠ id`. This matches, from
the opposite direction, the independent finding of
`fuzzy_axions_2412_12012_symmetric_subdivision_route_scaled_20260819.md`
(the symmetric-subdivision route adds at most ~1 class at h11=3 as currently
reachable) — the missing acceptances need machinery, not a numeric correction.

**Consequence for the prior narrative:** the "226 vs 253" undershoot every
earlier session chased was an artifact of Bugs 1 and 2 cancelling. The original
`HANDOFF_inherited_orientifold_overcount_20260819.md` primary hypothesis
(non-identity-`L` over-acceptance) was *correct* — it read as "falsified" only
because Bug 1 was silently zeroing out the very sector being measured.

## 4. Everything else audited this session — clean

- **HuggingFace parquet transcoding** (`load_mirror_polytopes`,
  `generate_geometric_data_multitriangulation.py:4068`): self-verifying — maps
  `physical_h11 = mirror h12` (dual convention) then hard-asserts
  `poly.h11() == requested h11`, so a transcode error raises rather than
  silently passing. `row_index` is provenance only.
- **Model-count gap** (`src/paper_benchmarks/fuzzy_axions_model_stage.jl`):
  structural, robust to any numeric-factor bug — the `models/CY ≤ h11·(h11+4)`
  ceiling (exceeded at h11=6,7) is pure Table-1 arithmetic, and the C-sweep
  bounds the count over the whole prefactor range. λ-solver
  `λ² = 1 + ln(m_ref/m_target)/(π τ_a)` is consistent with `m ∝ exp(−π τ_a)`.
- **eq. (4.34/4.45/4.48/4.50)** map faithfully to KS_orientifolds.tex lines
  627/645/653; the `n_S` expansion `D_p·D_q·(K⁻¹−D_p)·(K⁻¹−D_q)` and the
  canonical-index +1 offset are correct.

## 5. Files changed / tests / scope

- `scripts/inherited_orientifold_candidates.py`: Bug 1 (exact `2t` in
  `_dual_vertex_parity_evidence`), Bug 2 (`2t∈N` filter + new
  `torus_shift_not_involution` terminal status in
  `enumerate_orientifold_candidates`).
- `scripts/test_inherited_orientifold_candidates.py`: two fixtures updated to
  post-filter counts, empirically determined, with source citations
  (`test_fully_symmetric_frst_enumerates_the_full_triple_space` now filter-aware
  on the record count; `test_partial_frst_rejects_candidates_that_move_the_excluded_point`
  74 records / 36 accepted / 6 frst-failed / 12 non-involution). 20/20 pass
  (incl. `test_h21_plus_zero_fixed_locus`).
- **Version impact** (per `AGENTS.md`): changes the numeric output of a
  diagnostic script (`--orientifold-audit`), not a persisted-data contract or
  public API. No `Project.toml` bump on this feature branch.

## 6. Next work

Implementing the missing general-`L` machinery is the only path to 253. It is
scoped in `HANDOFF_general_L_smoothness_machinery_20260819.md` (this session).
