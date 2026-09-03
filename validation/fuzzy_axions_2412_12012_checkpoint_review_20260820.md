# Post-checkpoint scientific review: inherited orientifolds

Date: 2026-08-20
Scope: read-only review of checkpoint `085119-Codex-checkpoint.md`, the
three final audit artifacts, the validation reports, and the vendored Moritz
and Sheridan source files. No population rerun was performed and no revised
population count is claimed here.

## Review status

The checkpoint artifacts remain structurally valid, and the base-population
and trilayer diagnostics remain usable. The review found one direct formula
notation error and one unresolved source-convention issue in the general-
`L` fixed-component path. Therefore the old general-`L` surface totals are
diagnostic, conditional evidence rather than source-validated counts pending
a convention repair and rerun.

## Update (2026-08-20, after the h11 = 2 and h11 = 3 rerun)

The convention repair described in the findings below was subsequently
implemented and validated by an h11=2 and h11=3 rerun with the corrected code:

- Fixed components are now labelled by the original-`Sigma` pointwise-invariant
  cones with their original ray scale (function
  `_pointwise_invariant_cone_keys`), and the eq. (4.45) parity condition uses
  the corrected `any`-pairing facet test. A nonprimitive-invariant-pair
  regression test was added.
- At h11=2 no accepted orientifold changed (inherited and \(h^{1,1}_-=0\) stay
  10/10; the accepted set is identical polytope-by-polytope). At h11=3 exactly
  one previously accepted inherited orientifold is now correctly rejected
  (polytope 229, via the eq. (4.45) extension), so the inherited and
  \(h^{1,1}_-=0\) lower bounds move from 81 to 80.
- General-`L` diagnostic evidence-row counts fell (h11=2: 1362 → 1164 attempts;
  h11=3: 6314 → 5510) because the corrected enumeration uses the smaller,
  source-faithful original-`Sigma` cone universe.
- New artifacts (reproduction schema 1.1, general-`L` diagnostic schema 1.1):
  h11=2 `c4cc9bc93a8d590d42a2ef98fdb58156da442493f07d5854b8ca8e4873f9ebb0`,
  h11=3 `05ce23cc5fa0819a83ecbde0d8784628e5e540a874e8e6ba291ce2ed30f63e42`.
  h11=4 was **not** rerun; its rows remain the superseded 2026-08-19 values
  pending a corrected-code rerun.

The findings recorded below are preserved as the original review. Their
"conditional/superseded" language is resolved for h11=2 and h11=3 by this
rerun and remains open only for h11=4.

## Confirmed findings

1. **The fixed-surface bundle is conormal.** Moritz’s eq. (4.50) uses
   \(\mathcal O(K_V^{-1}|_S)\otimes N^*_{S/V}\), not the normal bundle
   \(N_{S/V}\) (Moritz [arXiv:2305.06363](https://arxiv.org/html/2305.06363),
   eq. (4.50); source `KS_orientifolds.tex` lines 643–655).
   The implementation’s expanded expression
   \(c_2(T_V)|_S-c_2(T_S)+c_1(T_S)^2\) is the corresponding conormal
   expression. The detailed report’s normal-bundle notation was corrected.

2. **The original-cone versus auxiliary-ray convention is unresolved.** The
   source defines the auxiliary fan using intersections with the fixed space
   (Moritz [arXiv:2305.06363](https://arxiv.org/html/2305.06363), eq. (4.26);
   source `KS_orientifolds.tex` lines 494–498), but its fixed-component formula labels a
   pointwise-`L`-invariant cone `sigma` in the original `Sigma` and sums the
   original rays with their lattice scale (same source, lines 541–555). At
   review time the code/report path labelled records by primitive cones/rays of
   `Sigma_L` (now corrected — see the update above). These are not
   automatically interchangeable: primitive normalization can change the
   half-ray sum and hence the eq. (4.35) integrality test. A non-axis-aligned
   invariant ray pair gives a concrete scale discrepancy.

   This is a scientific validation boundary, not proof that every current
   record is wrong. Identity-involution results do not use this general-`L`
   replacement and remain unaffected. General-`L` records and any accepted
   class whose evidence depends on them are conditional until original-cone
   provenance/scale is preserved or equivalence is proved.

3. **Kähler smoothness is an acceptance boundary.** Moritz separately
   requires an invariant Kähler hyperplane/interior point and discusses the
   distinction between a smooth hypersurface and a singular/non-simplicial
   symmetric ambient fan (same source, lines 610–617, with the fan
   construction at lines 483–489). The audit reports must keep this condition
   explicit. An invariant FRST may admit an invariant interior point by
   averaging, but that implication is not a substitute for documenting the
   check when the auxiliary fan is non-simplicial or non-unimodular.

4. **The source does not prescribe a generic “crepant repair.”** The next
   implementation step should be source-compatible Cartier/intersection and
   orbifold data on non-unimodular provenance cones. Non-simplicial symmetric
   fan subdivision/removal/gluing is a separate §4.3 task, not an assumed
   crepant completion.

5. **General-`L` rows are evidence rows, not fixed components of every `t`.**
   The diagnostic table assigns surface evidence to `(sigma, nu)` records;
   it does not attach eq. (4.35) integrality to every specific torus shift.
   Until that provenance is attached, use “diagnostic evidence rows,” not
   “certified fixed surfaces.” The candidate schema
   `cyaxiverse-inherited-orientifold-candidate-2.2` applies to standalone
   candidate manifests; it is not the schema of the nested general-`L`
   diagnostic table.

## Results that remain valid from the checkpoint

- The three final JSON artifacts remain the referenced artifacts, with valid
  JSON structure, complete favorable-polytope detail coverage, and unchanged
  SHA-256 hashes:

  | `h11` | artifact | SHA-256 |
  |---:|---|---|
  | 2 | `/private/tmp/cyax-orientifold-final-reviewed-h11-2-20260819.json` | `27e7b18148bc6730089c86fabb99bd2dae0ff8c91ed721ce302f60106feb647a` |
  | 3 | `/private/tmp/cyax-orientifold-final-reviewed-h11-3-20260819.json` | `9de41a8967895c3f0de36317b83ce44bb978ae8971b84c573dc30795e2c6dc9f` |
  | 4 | `/private/tmp/cyax-orientifold-final-reviewed-h11-4-20260819.json` | `a5849fa7c02040dc4986653168262eac4d299a492d58fe7728e948afcd875769` |

- The favorable populations and FRST-class populations remain the checkpoint
  values: `(36, 36)`, `(243, 274)`, and `(1185, 1760)` for `h11=2,3,4`.
- The identity-sector diagnostics remain valid under this review: identity
  actions `(251, 2296, 13218)` and identity-valid O3/O7 classes
  `(25, 198, 1153)`.
- The independent trilayer \(h^{2,1}_+=0\) benchmark remains exact:
  `(11, 66, 267)`. This is a separate diagnostic population, not a proof of
  the general-`L` fixed-component convention.

## Results that are conditional or superseded pending rerun

The old general-`L` diagnostic totals were `(1362, 6314, 25398)` attempts,
`(862, 3934, 17024)` rows labelled certified, and `(500, 2380, 8374)` rows
skipped for `h11=2,3,4`, respectively. They must now be read as diagnostic
evidence from the unchanged artifacts, not as source-validated general-`L`
surface counts; those labels are superseded pending a rerun with the original
`Sigma` cone/ray provenance and shift-integrality check repaired. No revised
general-`L`, inherited, or model population counts are supplied by this note.

The inherited lower-bound rows `(10, 81, 435)` and the corresponding
`h11_- = 0` rows `(10, 81, 429)` were conservative outputs of the old run. The
h11=2 and h11=3 rerun resolves them to `10`/`80` inherited and `10`/`80`
`h11_- = 0` (the single h11=3 change is the polytope-229 overcount removal
described in the update above); the h11=4 rows `435`/`429` remain superseded
pending a corrected-code rerun.

## Required follow-up

Done in this pass: the original pointwise-invariant `Sigma` cone/ray-scale
labelling is implemented, a nonprimitive invariant-pair regression test was
added, and `h11=2,3` were rerun with the hashes and tables updated. Remaining:
rerun `h11=4` with the corrected code, and — for a full mapping proof rather
than the current source-faithful reimplementation — independently revalidate a
nonidentity surface (the existing independent note used a primitive ray only).
Until the h11=4 rerun is complete, retain the conservative acceptance statuses
and do not infer a published-population discrepancy from the gaps.

## Source anchors

- Moritz source: [arXiv:2305.06363 (HTML)](https://arxiv.org/html/2305.06363), §§4.3–4.6, especially eqs. (4.26), (4.33)–(4.35), (4.45), (4.50) (source `KS_orientifolds.tex` lines 483–498, 541–555, 610–659).
- Sheridan source: [arXiv:2412.12012 (HTML)](https://arxiv.org/html/2412.12012), especially Table 1 and the inherited/trilayer population definitions.
- [Checkpoint detailed report](./fuzzy_axions_2412_12012_orientifold_audit_detailed_20260820.md).
