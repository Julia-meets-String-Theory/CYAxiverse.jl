# HANDOFF: follow-up work for general-$L$ orientifold smoothness

**For:** the next coding agent working on the CYAxiverse orientifold audit.
**Date:** 2026-08-19
**Starting commit:** `4fcc3fa` (`Implement general-L fixed-surface smoothness evidence`)
**Primary source:** Moritz, [arXiv:2305.06363](https://arxiv.org/html/2305.06363), §§4.3--4.6

## 1. Current state

The §4.1 general-$L$ fixed-surface $n_S$ machinery is implemented in
`scripts/inherited_orientifold_candidates.py` and enabled by
`scripts/reproduce_fuzzy_axions_h11_4.py`.

The implementation computes

```text
n_S = int_S c_2(O(K_V^-1)|_S tensor N*S)
    = int_S c_2(T_V)|_S - c_2(T_S) + c_1(T_S)^2
```

using the auxiliary fixed fan, saturated integer lattices, smooth toric
surface intersections, and restricted ambient Cartier data. It records
conservative evidence only when these checks succeed. Candidate manifests use
schema `cyaxiverse-inherited-orientifold-candidate-2.1`; the package version
remains `0.2.0` and must not be bumped on this feature branch.

Validation already completed:

- the general implementation agrees with the identity implementation on all
  273 comparable h11=2 identity surface entries, with zero mismatches;
- a hand-computed $(\mathbb{P}^1)^4$ example returns $n_S=8$ for all four
  fixed surfaces;
- `scripts/test_inherited_orientifold_candidates.py`: 21/21 passed;
- `scripts/test_h21_plus_zero_fixed_locus.py`: 1/1 passed.

The full-population counts did not change after adding general-$L$ $n_S$:

| h11 | current | Table 1 target | identity O3/O7 | trilayer |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 25 | 32 | 25 | 11 |
| 3 | 203 | 253 | 198 | 66 |
| 4 | 1,169 | 1,559 | 1,153 | 267 |

This is an honest residual, not a failed test. Equation (4.50) accepts only
certified $n_S=0$ surfaces. A certified nonzero value is a smoothness
rejection, and missing evidence remains `smoothness_verification_unavailable`.

The h11=3 diagnostic found, among nonidentity O3/O7 records with an
identically vanishing two-dimensional fixed component, 202 certified nonzero
surface values, 117 components without a certificate, and zero certified
zeros. These are candidate-component counts, not distinct CY-class counts.

## 2. Priority 1: explain unresolved $n_S$ evidence

Instrument `_general_fixed_surface_n_s_table` so every skipped surface carries
a machine-readable reason. At minimum, distinguish:

- nonsaturated fixed-cone lattice;
- missing or non-simplicial full-dimensional auxiliary cone;
- incomplete quotient surface fan;
- non-smooth ambient cone or surface fan;
- missing, nonintegral, or inconsistent restricted Cartier data;
- nonintegral final $n_S$.

Aggregate these reasons by h11, polytope, FRST class, matrix, and fixed
component. The immediate question is whether the 117 missing components are
legitimate source-boundary rejections or implementation gaps. In particular,
identify the previously reported h11=3 classes whose O3/O7 candidates were
stuck at `smoothness_verification_unavailable` and show the reason for each.

Do not convert an unresolved reason directly into acceptance. Add evidence or
an explicit source-matched rejection only after the corresponding geometric
condition is established.

Recommended deliverables for this step:

1. a focused diagnostic function or optional audit field rather than an
   always-on verbose printout;
2. an h11=3 reason table;
3. an h11=4 reason table to determine whether the residual changes character
   at larger h11.

## 3. Priority 2: independently validate a real CYTools example

Before using any new $n_S=0$ result to change population counts, select one
real nonidentity candidate from the h11=3 or h11=4 audit and publish its:

- lattice matrix $L$, shift, and fixed component $(\sigma,\nu)$;
- auxiliary quotient fan and primitive quotient rays;
- restricted ambient divisor coefficients;
- computed $c_2(T_V)|_S$, $c_2(T_S)$, $c_1(T_S)^2$, and $n_S$.

Recompute the same value with an independent method, such as a direct toric
Chow-ring calculation or an explicit generic-section/nodal-point count on the
fixed surface. The existing $(\mathbb{P}^1)^4$ fixture is a useful unit test,
but it is not a substitute for a CYTools geometry with a nontrivial quotient
fan.

Record the example and the independent calculation in a new validation note.
If the two methods disagree, stop population work and resolve the convention
or lattice issue first.

## 4. Priority 3: implement non-simplicial $\Sigma_L$ if measurements justify it

The next structural gap is the non-simplicial $L$-symmetric fan from source
§4.3. Reuse the earlier symmetrized-height/subdivision investigation in this
repository, but extend the machinery end to end:

- represent non-simplicial cells and their faces;
- remove and glue the cones associated with degenerate curves;
- build the subdivision-aware auxiliary fixed-locus fan;
- propagate the fan through smoothness, parity, integrality, and $n_S$
  evidence;
- preserve provenance from the original FRST and record terminal failures.

Start this work only if the h11=4 reason table shows a material population in
`frst_not_preserved` or another class that cannot be represented by the
current invariant FRST path. The h11=3 handoff data showed no rejected class
blocked solely at `frst_not_preserved` with a structurally available
nonidentity matrix, so this is a larger-h11 lever rather than the first low-h11
fix.

Do not replace the current conservative FRST result with an unverified
non-simplicial approximation.

## 5. Population gates

Run the following gates after each coherent change:

1. Run the focused Python tests.
2. Run the h11=2 and h11=3 orientifold audits.
3. Confirm that identity O3/O7 and trilayer counts remain unchanged unless a
   source-matched reason explains the change.
4. Run h11=4 after the mechanism is plausible and compare both the population
   count and terminal-status distribution.
5. Run h11=5 only after h11=4 shows a measurable, independently validated
   mechanism or after an explicit scope decision to perform the large scan.

Use the local `cytools` environment and `python`, not `python3` or pytest:

```bash
cd /Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse-orientifold-overcount
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate cytools
PYTHONDONTWRITEBYTECODE=1 python scripts/test_inherited_orientifold_candidates.py
PYTHONDONTWRITEBYTECODE=1 python scripts/test_h21_plus_zero_fixed_locus.py
PYTHONDONTWRITEBYTECODE=1 python scripts/reproduce_fuzzy_axions_h11_4.py \
  --h11 3 \
  --parquet-dir /private/tmp/cyax-ks-mirror-h11-4 \
  --orientifold-audit --keep-details \
  --julia-binary /Users/vmehta/.juliaup/bin/julia \
  --output /private/tmp/cyax-orientifold-followup-h11-3.json
```

CYTools may emit its known non-fatal cache-save `PermissionError` at process
exit. Treat it as a warning only when the audit JSON was written successfully.
Report the warning and the output path.

## 6. Scope and handoff requirements

- Do not modify the model-count chain in this follow-up.
- Do not run h11=5 merely because the lower-dimensional count remains below
  target.
- Preserve the existing handoff-file edit in the worktree unless its owner
  requests otherwise.
- Preserve no-overwrite behavior and candidate provenance.
- Do not change scientific claims without a validation note and an explicit
  source reference.
- Keep package-version changes for the reviewed release boundary. Any commit
  containing AI-assisted work must include a `Co-Authored-By:` trailer.

The next agent is finished when the unresolved-reason table and one real
independent $n_S$ validation are documented, the h11=4 decision about
non-simplicial $\Sigma_L$ is evidence-based, and all reported population
changes have focused tests and reproducible audit outputs.
