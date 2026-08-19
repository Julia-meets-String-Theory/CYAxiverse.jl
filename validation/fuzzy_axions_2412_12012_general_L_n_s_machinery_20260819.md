# Validation: general-$L$ fixed-surface smoothness machinery

Date: 2026-08-19
Branch: `codex/orientifold-overcount-20260819`
Source: Moritz, [arXiv:2305.06363](https://arxiv.org/html/2305.06363), §§4.3--4.6

## Outcome

The general-$L\ne\mathbb{1}$ fixed-surface calculation required by eq. (4.50) is implemented and wired into the inherited-orientifold audit. It is conservative: it records evidence only when the fixed quotient fan is a complete smooth toric surface and the restricted ambient line bundles have certified integral Cartier data.

The implementation does not close the remaining Table 1 population gap at h11=2, 3, or 4. The unchanged counts are a result of the source condition itself: a certified nonzero $n_S$ rejects a candidate, while an uncertified component remains `smoothness_verification_unavailable`; only certified $n_S=0$ can promote a candidate.

## Implementation

`scripts/inherited_orientifold_candidates.py` now provides:

- a Smith-normal-form integer kernel basis, so invariant and quotient lattices are saturated rather than rational nullspaces with independently cleared denominators;
- exact coordinates in the invariant lattice and a saturation check for the fixed cone;
- the two-dimensional quotient fan for each fixed surface in the existing auxiliary fan;
- smooth complete-surface checks and toric divisor intersection arithmetic;
- local ambient Cartier support functions and their restrictions to the fixed surface;
- the Chern-class evaluation

  ```text
  n_S = int_S c_2(T_V)|_S - c_2(T_S) + c_1(T_S)^2,
  ```

  obtained from
  `c_2(O(K_V^-1)|_S tensor N*S)` by the toric Euler sequence and adjunction.

The resulting evidence is stored in `fixed_surface_n_s_evidence` on candidate records. The candidate manifest schema is now `cyaxiverse-inherited-orientifold-candidate-2.1` because this is an additive persisted field. The package version remains `0.2.0`; any package-version bump is deferred to the reviewed release boundary.

The audit driver enables this calculation for nonidentity matrices. The identity path remains supplied by its existing independently validated intersection-tensor implementation.

## Independent checks

1. The general helper reduces exactly to the identity formula. Across all 273 h11=2 FRST-class identity surface tables, the direct identity implementation and the general fixed-fan implementation produced 273 comparable entries with zero mismatches.

2. A hand-computed test uses $V=(\mathbb{P}^1)^4$ with rays $\pm e_i$ and $L=\operatorname{diag}(1,1,-1,-1)$. Each fixed surface is $S=\mathbb{P}^1\times\mathbb{P}^1$, with trivial two-dimensional normal bundle and $K_V^{-1}|_S=\mathcal{O}(2,2)$. Therefore

   ```text
   int_S c_2(O(2,2) tensor O_S^2) = int_S (2 H_1 + 2 H_2)^2 = 8.
   ```

   The implementation returns four fixed components, each with $n_S=8$.

3. The focused Python suite passes 21/21 tests, including the saturated-kernel regression and the explicit $(\mathbb{P}^1)^4$ fixture. The fixed-locus regression passes 1/1 test.

## Population measurements

All audits used the local `cytools` environment, the KS mirror at `/private/tmp/cyax-ks-mirror-h11-4`, `--orientifold-audit`, and `--keep-details`.

| h11 | lower bound | before | after general-$L$ $n_S$ | Table 1 target | identity O3/O7 after | trilayer after |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 25 | 25 | 25 | 32 | 25 | 11 |
| 3 | 198 | 203 | 203 | 253 | 198 | 66 |
| 4 | 1,153 | 1,169 | 1,169 | 1,559 | 1,153 | 267 |

The h11=3 component diagnostic covered the full 243-polytope population. Among nonidentity O3/O7 records with an identically vanishing two-dimensional fixed component, it found 202 certified nonzero values, 117 components with missing evidence, and no certified zero values. These are candidate-component counts, not distinct CY-class counts.

h11=5 was not run in this validation pass.

## Verification commands and warnings

Focused checks:

```text
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate cytools && PYTHONDONTWRITEBYTECODE=1 python scripts/test_inherited_orientifold_candidates.py
Ran 21 tests in 1.519s — OK

source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate cytools && PYTHONDONTWRITEBYTECODE=1 python scripts/test_h21_plus_zero_fixed_locus.py
Ran 1 test in 0.808s — OK
```

Population audits wrote their JSON results before exit:

```text
python scripts/reproduce_fuzzy_axions_h11_4.py --h11 2 ... --orientifold-audit ...
python scripts/reproduce_fuzzy_axions_h11_4.py --h11 3 ... --orientifold-audit ...
python scripts/reproduce_fuzzy_axions_h11_4.py --h11 4 ... --orientifold-audit ...
```

Each CYTools process emitted the same non-fatal exit warning while trying to save its cache under `/Users/vmehta/Library/Caches/CYTools/`; the sandbox denied that cache write. The audit JSON files were successfully written to `/private/tmp` and their counts were read after completion.

## Residual and next boundary

The §4.1 machinery is now present, but it is not the missing acceptance lever for the measured low-dimensional populations. The remaining work is outside this implementation boundary:

- construct and propagate the non-simplicial symmetrized fan $\Sigma_L$ described in §4.3, where an invariant FRST does not exist;
- add the source's remaining general fixed-locus smoothness and generic-hypersurface checks if the $\Sigma_L$ measurements show they are needed;
- rerun h11=5 only after a mechanism changes the h11=4 residual.

The current result is therefore a validated, additive general-$L$ $n_S$ capability with an honestly unchanged population bracket, not a claim that Table 1 has been fully reproduced.
