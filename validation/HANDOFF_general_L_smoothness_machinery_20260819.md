# HANDOFF: implement the general-L orientifold smoothness machinery

**For:** a fresh agent. You have no prior context;
everything you need is here or cited by path.
**Written by:** the session that found and fixed the two cancelling
non-identity-`L` torus-shift bugs, 2026-08-19, working from
`fuzzy_axions_2412_12012_torus_shift_involution_filter_20260819.md` (read it
first — it is short and establishes why this work is now the *only* remaining
path to Table 1).
**Status:** a build brief. This is a feature-implementation task, not a
bug hunt. The gap it targets is proven-genuine missing physics, not a defect.

---

## 1. The goal, precisely

Reproduce arXiv:2412.12012 Table 1's "CYs with inherited involutions" row by
implementing the general-`L` (L ≠ identity) parts of Moritz arXiv:2305.06363
§4.3-§4.6 that are absent from this codebase. Concretely, close the residual:

| h11 | machinery now (both bug fixes in) | lower bound | **Table 1 target** |
| --- | --- | --- | --- |
| 2 | 25 (re-measure) | 25 | **32** |
| 3 | **203** | 198 | **253** |
| 4 | ~1,2xx (re-measure) | 1,153 | **1,559** |
| 5 | (re-measure) | 6,294 | **9,530** |

At h11=3 the machinery accepts 198 via identity `L` and only **5** more via
non-identity `L`. The missing 50 classes all require acceptances that need
general-`L` structure the code does not have. Work at **h11=3** while iterating
(274 classes, ~70 s per `--orientifold-audit` run), confirm at h11=2 (fast) and
h11=4/5 once a mechanism firms up.

## 2. What is DONE — do not re-investigate

All of the following are fixed and validated on `codex/orientifold-overcount-20260819`
(uncommitted as of this handoff unless the writing session committed them —
check `git diff`). Full detail in
`fuzzy_axions_2412_12012_torus_shift_involution_filter_20260819.md`.

- **eq. (4.45) parity truncation** (Bug 1): `_dual_vertex_parity_evidence`
  now keeps `2t` exact. Do not reintroduce `int(2*value)`.
- **`2t ∈ N` involution filter** (Bug 2): `enumerate_orientifold_candidates`
  now skips torus-shift cosets whose representative `2t` is non-integral
  (source line ~428; `torus_shift_not_involution` terminal status). Proven
  exact (integrality is a coset invariant; 0 violations across 81 real
  non-identity `L`). Do not "loosen" it to recover count — the excluded reps
  are genuinely not `Z2` involutions.
- **Parquet transcoding, model-count gap, eq. 4.34/4.45/4.48/4.50 formulas**:
  all audited clean this session (see the writeup's §4). Not your concern.

The identity path (`identity_valid_o3o7_action_cy_count = 198`) and the trilayer
path (`h21_plus_zero_trilayer_frst_classes = 66`) are **separate code paths**;
whatever you build must leave both exactly where they are.

## 3. Where the 50 missing classes actually are (h11=3, both fixes in)

Measured over the 71 rejected classes (274 − 203):

```
rejected classes:                                                     71
  with a non-identity O3/O7 stuck 'unavailable'
      (general-L n_S, §4.1, would resolve):                           28
  with non-identity O3/O7 all hard-rejected
      ('fixed_point_set_non_smooth' — stricter/other physics or
       genuine rejection):                                            41
  non-identity L exists but only frst_not_preserved
      (needs Σ_L subdivision, §4.2):                                   0
  no non-identity L structurally at all (genuinely un-rescuable):      2

non-identity terminal_status tally across rejected classes:
  smoothness_verification_unavailable  242
  fixed_point_set_non_smooth           841
  torus_shift_not_involution           601   (correctly filtered, §2 Bug 2)
  torus_shift_search_exhausted         158
  frst_not_preserved                    12
  accepted_verified_orientifold          3   (rescued a class already counted)
```

**Read the sequencing off this, and note two things.** (1) The paper itself
rejects ~21 of the 274 at h11=3 (accepts 253), so only ~50 of the 71 are
genuinely "missing" — do not aim to accept all 71. (2) The single tractable
lever is **§4.1 (general-`L` n_S)**: 28 rejected classes have a non-identity
O3/O7 candidate stuck at `smoothness_verification_unavailable` purely for want
of `n_S` evidence — the machinery already reaches them. The `Σ_L` subdivision
(§4.2) rescues **0** additional classes at h11=3 (no rejected class is blocked
solely at `frst_not_preserved` with a structurally-available non-identity `L`),
so it is *not* the low-h11 lever despite being the source's headline
construction — its payoff is at large h11 (source line 617). **Build §4.1
first; re-measure before touching §4.2.**

## 4. What is missing, from the primary source

Primary source: `validation/fuzzy_axions_supp/paper_source_2305_06363/KS_orientifolds.tex`.
Three pieces, in rough order of tractability.

### 4.1 General-`L` `n_S` on non-identity fixed surfaces (source line 649-654)

`classify_smoothness`'s 2-dimensional-fixed-component branch
(`inherited_orientifold_candidates.py`, the `component["fixed_toric_dimension"] == 2`
case) looks up `n_S` from `topology["fixed_surface_n_s"]`, which is populated
**only** by `identity_fixed_surface_n_s_table` (identity-only by construction —
its `nu` coset `H_-^{id}` is trivial and its `Σ_L` reduces to the ambient fan).
For a non-identity `L` no evidence is supplied, so every such candidate returns
`smoothness_verification_unavailable` and is not accepted. Deriving and
supplying the general-`L` `n_S = ∫_S c_2(O(K_V^{-1})|_S ⊗ N^*_{S/V})` (source
eq. 649) for a non-identity fixed surface `F_L(σ,ν)` would let these candidates
resolve. The identity formula `n_S = D_p·D_q·(K⁻¹−D_p)·(K⁻¹−D_q)` assumes the
fixed component is a 2-cone `(p,q)` of the ambient fan; for `L ≠ id` the fixed
component's own toric structure is the fixed sub-fan `Σ_L` (§4.4 of the source),
generally not a plain 2-divisor intersection — this is exactly why the identity
table was never extended. **Derive it from the source, do not extrapolate the
identity formula.** Cross-check any candidate `n_S` against an independent
method (e.g. a direct nodal-point count on a small explicit example) before
trusting it, the same discipline `identity_fixed_surface_n_s_table` records.

### 4.2 The `L`-symmetric toric fan `Σ_L` / auxiliary variety `V_s` (source §4.3, eq. ~466-489)

For non-identity `L` at larger h11, valid orientifolds generically live on a
non-simplicial `L`-symmetric fan built from symmetrized heights
`h_p := ½(h'_p + h'_{L(p)})` (source eq. ~473), not on any `L`-invariant FRST.
The singular fan `Σ_L` is obtained from an FRST by **removing the 3-cones
`σ_{p,p',p''}` associated to degenerate curves and pairwise-gluing the cones
that met along them** (source line 489). Classes that reach `frst_not_preserved`
today have no `L`-invariant FRST at all and can only be reached this way. The
prior report
`fuzzy_axions_2412_12012_symmetric_subdivision_route_scaled_20260819.md` built a
from-scratch, out-of-package reimplementation of the *height-symmetrization +
subdivision* step (via CYTools `vc().subdivide(... backend="ppl",
make_fine=False, cure_heights=True)`) and validated it against the prior
report's own examples — reuse its recipe as the starting point, but note §3
measures **0** rejected classes at h11=3 that this would rescue (none is blocked
solely at `frst_not_preserved` with a structurally-available non-identity `L`),
consistent with that report's own "closes at most ~1" upper bound. This is the
*large-h11* lever, not the low-h11 one (source line 617: "at large h11 all
examples known to us feature non-simplicial `L`-symmetric toric fans"); defer it
until §4.1 is built and h11=4/5 are re-measured. The existing `build_auxiliary_fan` (eq. 4.26) and
`classify_smoothness` are written entirely in terms of a `Triangulation`'s
simplices, **not** a general `Fan` — supporting a non-simplicial `Σ_L` means
non-simplicial cell/face records throughout, the removal/gluing step, and a
subdivision-aware auxiliary fixed-locus fan.

### 4.3 Fixed-locus smoothness for `L ≠ id` (source line 656-657)

"We impose that the fixed point set `F_I` is smooth itself": the toric
varieties `F(σ,ν)` on which `f` vanishes identically must be smooth, and for
the others the generic section must be nef with no orbifold singularities
meeting the hypersurface. The identity path handles the vanishing-surface case
via `_cone_has_smooth_star`; the nef/orbifold-intersection condition for
general `L` is not implemented and may be needed to *reject* some of the
currently-over-… (n.b. post-fix the machinery under-accepts, so this condition
is more likely a refinement than a gap-closer — build 4.1 and 4.2 first and
re-measure before investing here).

## 5. Current code touchpoints

- `scripts/inherited_orientifold_candidates.py`
  - `classify_smoothness` — the accept/reject gate. The
    `fixed_toric_dimension == 2` and `> 0` branches are where general-`L`
    evidence must land.
  - `identity_fixed_surface_n_s_table` / `_n_s_for_two_ray_cone` /
    `_cone_has_smooth_star` — the identity-only evidence to generalize (4.1).
  - `build_auxiliary_fan` / `_fixed_component_records` — eq. (4.26)/(4.34-4.35),
    already general-`L` but simplicial-only.
  - `facets_with_non_smooth_cones` — the eq. (4.45) extension fan; today uses
    `triangulation.fan()` (correct for FRST-preserving `L`, wrong once `Σ_L`
    differs — revisit under 4.2).
- `scripts/reproduce_fuzzy_axions_h11_4.py`
  - `_orientifold_action_audit` — assembles topology evidence per class and
    counts; where new `topology[...]` evidence keys get wired in.
  - `PAPER_TARGETS_BY_H11` — the Table 1 targets to validate against.

## 6. Ordered plan with validation at each step

1. **Re-measure the baseline** at h11=2,3,4 with both bug fixes in (the numbers
   in §1 for h11=2,4,5 are pre-fix placeholders). Record the exact
   [lower_bound, current, target] bracket per h11. Falsification of the whole
   plan: if the machinery already ≈ target somewhere, the gap is smaller than
   believed.
2. **Instrument §3's distribution at h11=4** to confirm the rescue-path mix
   generalizes (h11=3 alone can mislead).
3. **Build 4.1 (general-`L` `n_S`)** first — it reuses existing plumbing
   (`topology["fixed_surface_n_s"]`) and targets the `unavailable` classes.
   Validate: the count moves toward target and the identity/trilayer numbers do
   not regress. Cross-check `n_S` on a hand-workable example.
4. **Re-measure.** If a large residual remains and it is `frst_not_preserved`
   classes, **build 4.2 (`Σ_L` subdivision)** — the larger effort. Reuse the
   prior report's subdivision recipe; add non-simplicial fan support to
   `build_auxiliary_fan`/`classify_smoothness`.
5. **Only then** consider 4.3 as a refinement.

At every step: run `--orientifold-audit` at h11=2 and 3, assert
`identity_valid_o3o7_action_cy_count` and `h21_plus_zero_trilayer_frst_classes`
unchanged, and update `test_inherited_orientifold_candidates.py` /
`test_h21_plus_zero_fixed_locus.py` empirically (never hand-derive expected
counts — run the fixture, read the number, comment the physics/source line).

## 7. Setup, run, gotchas

```bash
cd /Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse-orientifold-overcount
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate cytools     # use `python`, NOT python3; pytest not installed, use `python -m unittest`
```

```bash
python scripts/reproduce_fuzzy_axions_h11_4.py --h11 3 \
  --parquet-dir /private/tmp/cyax-ks-mirror-h11-4 \
  --orientifold-audit --keep-details \
  --julia-binary /Users/vmehta/.juliaup/bin/julia --output out.json
```

- Runtimes (full pop): h11=2 ~20 s, h11=3 ~70 s, h11=4 ~2 min, h11=5 ~5 min.
- The Bash cwd may reset to the main tree after each command in some sandboxes —
  chain `cd <worktree> && …` in one invocation.
- Standalone scans (importing the machinery via `sys.path.insert`, not the CLI)
  recompute topology per class and are ~2-4× slower than the CLI — run them in
  the background.
- `argparse` %-expands help strings; a literal `%` breaks the parser — double
  it to `%%`.

## 8. Scope boundaries (per AGENTS.md)

- Work in a git worktree; stage only files you changed, by explicit path; never
  `git add -A`. No `Project.toml` bump on a feature branch — state version
  impact at the integration boundary.
- **Do not touch the model-count chain** (`src/paper_benchmarks/fuzzy_axions_model_stage.jl`,
  the `qcd_divisor_domain` flag). Separate, settled investigation.
- `validation/` is **tracked** in this worktree (contrary to some older
  handoffs' claim that it is gitignored — `git check-ignore` says it is not).
  Commit your writeups by explicit path like any other deliverable.
- Do not send the author-note draft anywhere; that is the user's action alone.

## 9. Deliverables

1. A `validation/` writeup: what you implemented, the source derivation, the
   before/after Table 1 bracket at each h11, and an honest statement of any
   residual. A partial close ("general-`L` n_S recovers N of 50 at h11=3") is a
   real result.
2. Commit your own files by explicit path on the worktree branch.
3. A checkpoint per `AI_POLICY.md` if you hit the context threshold.

## 10. References

- `fuzzy_axions_2412_12012_torus_shift_involution_filter_20260819.md` — the two
  bug fixes and why this is the only remaining path (read first).
- `fuzzy_axions_2412_12012_symmetric_subdivision_route_scaled_20260819.md` — the
  `Σ_L` subdivision route, reachability at scale, and a working reimplementation
  recipe (for 4.2).
- `HANDOFF_orientifold_smoothness_gap_streams_20260819.md` §4/§5 — Stream B
  (eq. 4.50 `n_S`) and Stream C (eq. 4.45 `Σ` fan) framing, now partly
  superseded but still the clearest source-linked description of the
  smoothness checks.
- `validation/fuzzy_axions_supp/paper_source_2305_06363/KS_orientifolds.tex` —
  Moritz. §4.3 (`Σ_L`, symmetrized heights), §4.4 (fixed loci), §4.6
  (smoothness: lines 619-659). Your primary source.
- `checkpoints/2026-08-19/093329-claude-sonnet-5-checkpoint.md` — the A/B/C fix
  session, for the lower-bound/target brackets at all h11.
