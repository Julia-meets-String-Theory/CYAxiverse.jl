# Independent general-$L$ fixed-surface validation

Date: 2026-08-19

This note independently recomputes one real h11=3 CYTools surface from the
machine-readable Priority-1 diagnostic output. It validates the toric
surface calculation only. It does not change the population count.

## Selected record

- Diagnostic input: `/private/tmp/cyax-orientifold-reason-h11-3.json`
- Polytope: `58`
- FRST class: `0`
- Matrix ID: `2f37841e536f16c965fce8a3314fa0e9f118a6ae8393a71a11dbcc8903aa7c28`
- Candidate ID: `84825af4a582f1544948fcfa79f6cc0e71c3a9f9b626a06bde6f816f82de80f5`
- Matrix:

  ```text
  [[1, 0, 0, 0],
   [0, 0, 1, 0],
   [0, 1, 0, 0],
   [0, 0, 0, 1]]
  ```

- Shift: `t = (1, 0, 0, 0)/2`
- Coefficient parity: `lambda_f = 1`
- Fixed component: `sigma = ((1, 0, 0, 0),)`, `nu = (0, 0, 0, 0)`
- Candidate terminal status: `smoothness_verification_unavailable`

The selected fixed surface itself is certified and has `n_S=0`. The
candidate-level unresolved status comes from another fixed component. This
record is therefore not an accepted-population claim.

## Exported toric data

The quotient surface rays and cones are

```text
rays = [(-2,-1), (-1,0), (0,1), (1,0)]
cones = [((-2,-1),(-1,0)),
         ((-2,-1),(1,0)),
         ((-1,0),(0,1)),
         ((0,1),(1,0))]
boundary scales = [1, 1, 1, 1]
reference ambient cone =
  ((0,0,0,1), (0,0,1,0), (0,1,0,0), (1,0,0,0))
```

The restricted divisor coefficients, in ambient-ray order, are

```text
(-1, 0, 0, 0)       -> [ 0,  0, 0, 0]
( 0, 0, 0, 1)       -> [ 1,  0, 0, 0]
( 0, 0, 1, 0)       -> [ 2,  1, 0, 0]
( 0, 1, 0, 0)       -> [ 2,  1, 0, 0]
( 1, 0, 0, 0)       -> [-4, -2, 0, 0]
( 2,-1,-1, 0)       -> [ 0,  1, 0, 0]
( 4,-2,-2,-1)       -> [ 1,  0, 0, 0]
```

Each coefficient vector is ordered by the quotient rays above.

## Independent Chow-ring calculation

The standalone calculation used only the exported quotient rays, quotient
cones, and restricted Cartier coefficients. It did not import the
implementation's private intersection helper.

For the smooth complete toric surface, adjacent boundary divisors intersect
once and a boundary divisor's self-intersection is minus the determinant of
its two neighboring rays. I used the toric Euler sequence and adjunction in
the source convention

```text
n_S = c2(T_V)|S - c2(T_S) + c1(T_S)^2.
```

The exact comparison is:

| term | audit row | independent Chow calculation |
|---|---:|---:|
| `c2(T_V)|S` | `-4` | `-4` |
| `c2(T_S)` | `4` | `4` |
| `c1(T_S)^2` | `8` | `8` |
| `n_S` | `0` | `0` |

All four values match exactly. The matrix is nonidentity, the quotient fan
is smooth and complete, and the independent result satisfies the required
`n_S=0` check.

Source formula: Moritz, arXiv:2305.06363, eq. (4.50), around lines
647--654 in `validation/fuzzy_axions_supp/paper_source_2305_06363/`.
