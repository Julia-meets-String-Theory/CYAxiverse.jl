# Issue 148 N=8 metric-normalization boundary audit

## Status: BLOCKED on one combined scientific-owner contract

Manager disposition: accept this Sol/xhigh audit as evidence of an unresolved
owner boundary. The recommendation below is **not approved or implemented**.
G0 and G1 remain PASS; G2 implementation is not dispatched pending the owner's
explicit coordinate and matrix-authority decision. G3 is not started.

The source record does **not** uniquely determine which N=8 canonical
normalization Issue 148 should reproduce.  The paper defines period-one axions
and gives the corresponding canonicalization rule, while the executable author
artifact uses raw-radian coordinates with the same numerical metric matrix.
Those contracts describe the same periodic potential and the same catastrophe
location, but they do not give the same canonical distance, Hessian, local
normal form, slow-roll parameters, or e-fold count.

The smallest decision required before G2 must fix both the coordinate map and
the numerical-matrix authority.  The recommended single approval is:

> **Approve P96 for scientific G2 outputs:** use period-one
> `theta in R^8/Z^8`, `V(theta)=V(2pi theta)`, and
> `K_theta(k)=M96/k^2`, where `M96` is the reconstructed numerical matrix whose
> eigenvalues are reported in equation (96).  In raw radians use
> `G_x=M96/[k^2(2pi)^2]`.  Retain **A96** only for explicitly labeled author
> reproduction.  This approval consciously chooses equation (96)/CYTools as
> the benchmark matrix authority over the `2M96` obtained by literally
> differentiating displayed equations (17) and (23); G2 must record that
> source inconsistency and must not call `M96` an equation-(23) derivation.

Here `M96` is the numerical N=8 matrix represented with more retained digits by the
N=8 HDF5 reconstruction in the stale repository path `appendix_c`, and at four
significant figures by the author program.  Contract P96 implies
`G_x=M96/(k^2 (2pi)^2)` after the coordinate
change `x=2pi theta`.  Contract A96 instead implies a period-one metric
`K_theta=(2pi)^2 M96/k^2`.  A third possible contract, **P23**, uses period-one
coordinates but `K_theta=2M96/k^2` to follow the literal displayed derivative
formula; it does not reproduce the equation-(96) spectrum.

**Recommendation:** approve P96 for new scientific and canonical claims.  Keep
A96 only as an explicitly named author-reproduction contract.  P96 follows the
paper's stated field period, Fourier argument, and canonical map; A96 is valuable
evidence for how the published numerical trajectory workflow was executed, but
it cannot simultaneously satisfy those three statements.  Choosing `M96`
keeps the paper's reported N=8 eigenspectrum and the independently reconstructed
geometry fixed.  The factor-two equation inconsistency remains explicit rather
than being silently “resolved.”  This recommendation uses source definitions
rather than fitting the published e-fold curve.

No production code or scientific convention was changed in this audit.

## Scope and immutable evidence

This audit is against repository commit
`1eef936e1908959db50a969e717b16a9c9e414bf`.  G0 and G1 remain accepted and are
not reopened.  The scope is the N=8 metric/coordinate boundary needed before
G2.  It does not implement continuation, survey other branches, change phases,
or make an off-ray claim.

The evidence identities are:

| Evidence | Identity |
|---|---|
| Paper | `arXiv:2608.14780v1`, local PDF SHA-256 `b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6` |
| Owner clarification | [Issue 148 comment 5609698767](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/148#issuecomment-5609698767) |
| Author settings | `poly102_settings.wl`, SHA-256 `2c49a27ad06ed7da707a348324029da48fd062109244e28b55237c99992a9fcc` |
| Author trajectory core | `poly102_core.wl`, SHA-256 `558df6893b62d574e3f50901c56bd0d455943703d3b427ca262cd6f10c71ffd0` |
| Author Figure-3 driver | `make_figure3_full_numerical.wl`, SHA-256 `92c11166470952c7d7d53a67ec02a776a47d61eb597831f1a1b03161e46d74af` |
| N=8 HDF5 (repository's stale `appendix_c` path) | `paper_benchmarks/appendix_c/h11_008/np_0000001/cy_0000001/cyax.h5`, SHA-256 `8fc483bc71a7e3f356b512adcfd44555f86a86bb765566c3f650984bf0ef1f4a` |

The owner comment explicitly requires this boundary to be resolved before
canonical claims.  It does not select P96, A96, or a numerical-matrix authority.

## Source facts

These are statements directly present in the paper, without implementation
interpretation.

1. Equation (14) defines the complex fields as `T^i=tau^i+i theta_i`.
2. Equation (18) uses `exp(-2pi q.T)`.  Equations (19) and (20) therefore use
   `cos(2pi q.theta + delta)`.
3. The paragraph below equation (19) states that the `theta_i` are
   dimensionless, have period one, and have a noncanonical kinetic term.
4. Equations (22) and (23) write the real kinetic term as
   `1/2 K_ij partial(theta_i) partial(theta_j)` and define `K_ij` from the
   Kahler potential.  The following paragraph states that calculations are
   performed in the canonically normalized basis and gives
   `phi ~ K^(1/2) theta` while retaining `V ~ cos(2pi q.theta)`.
5. Equation (25) sets `tau(k)=k tau(1)`.  Homogeneity gives
   `K(k)=K(1)/k^2` on this radial ray.
6. Appendix **D**, printed pages 30-31 (PDF pages 31-32), gives the N=8
   GLSM-basis divisor volumes in equation (95), the twelve Table-1 charges, and
   the eight kinetic eigenvalues in equation (96):
   `8.20e-4, 6.35e-4, 5.97e-4, 3.13e-4, 1.24e-4, 9.15e-5,
   8.30e-5, 5.84e-5` in descending order.

The immutable v1 table of contents identifies Appendix C as the N=5 geometry
and Appendix D as the N=8 geometry.  Repository names such as
`appendix_c`, `generate_appendix_c_geometry.py`, and the `n8_geometry`
docstring are stale labels and are not source-section identities.

The paper does not state that the equation-(96) matrix has already been divided
by `(2pi)^2`, and it does not introduce a second raw-radian metric.

## Executable-author facts

These are facts from the downloaded Mathematica artifacts, independent of the
paper prose.

1. `poly102BestXAtKCat` has entries near `pi/2`, `3pi/2`, and `2pi`, so its
   coordinate is a raw angle `x`, not a period-one `theta`.  The retained
   ten-term potential evaluates `Cos[chargeMatrix[[i]].x+phase]`; it has no
   `2pi` multiplier (`poly102_settings.wl:24-36` and
   `poly102_core.wl:287-313`).
2. `rawMatrix` is the four-significant-figure matrix whose eigenvalues match
   equation (96).  The code sets `G=k^-2 rawMatrix`,
   `S=MatrixPower[G,-1/2]`, `critPtCan=Inverse[S].bestX`, and evaluates the
   potential at `S.varsChi` (`poly102_core.wl:315-343`).
3. The same canonical variables determine the unstable mode, the fixed
   canonical displacement `1e-8`, gradient flow, epsilon, eta, field range,
   scalar-amplitude estimate, and cumulative turn
   (`poly102_core.wl:343-388, 466-493, 557-589, 687-699, 1094-1117,
   1413-1463`).
4. The settings label the analytic coefficients `alpha/Vc=93.86` and
   `G/Vc=1.52e7` as fitted/invariant parameters and use the same canonical
   displacement (`poly102_settings.wl:150-163`).  They are consequently A96
   quantities.
5. `SetPrecision[...,50]` is applied after the metric entries have already
   been entered as machine-precision decimal literals.  It pads those inputs;
   it does not recover digits that were not present in the artifact.

There is no metadata in the author artifacts that says `rawMatrix` is a
pre-rescaled raw-radian matrix.  Its entries and spectrum are the published
numbers themselves.  The artifact therefore supplies authoritative evidence
for executable reproduction, but conflicts with the paper's coordinate
definition if both use the same `M96`.

## Repository implementation facts at `1eef936`

The repository currently contains both contracts.  Namespace alone is not a
sufficient discriminator because the root derivative function changes
coordinate meaning when `trajectory=true`.

| Path | Coordinates and potential | Numerical metric actually used | Precision/source boundary |
|---|---|---|---|
| Root `n8_geometry`, `n8_kinetic_matrix`, default `n8_potential_derivatives` | period-one GLSM `theta`; `2pi Q'theta`; 12 Table-1 rows by default | precise reconstructed `M96/k^2` | hard-coded Float64 matrix agreeing with HDF5 |
| Root `n8_hilltop`, `n8_inflation_initial_condition`, `n8_local_hilltop_coefficients`, root gradient/valley/stiff flows | period-one; derivatives explicitly include powers of `2pi` | precise reconstructed `M96/k^2` | P96 implementation; Float64 |
| Root compatibility `n8_coordinate_maps`, `n8_mass_eigenbasis`, `n8_unstable_direction` | default root derivatives, hence period-one | precise reconstructed `M96/k^2` | P96 implementation; Float64 |
| Root compatibility aliases `n8_physical_gradient_flow`, `n8_efold_gradient_flow`, `n8_slow_roll_trajectory`, and `n8_hilltop_normal_form` | forward to `author_inflation` | rounded author `M96/k^2` | A96 implementation despite root-level names |
| Root `n8_potential_derivatives(...; trajectory=true)` | forwards its input unchanged to the author module; input is raw radians despite the root default using period-one | caller-dependent | mixed API boundary; no conversion is performed |
| `paper_benchmarks.author_inflation` N=8 geometry, coordinate maps, mass basis, initial condition, trajectories, and observables | raw-radian `x`; `Q'x`; 10 rows for trajectories | rounded author `M96/k^2` | A96 implementation; stored inputs originate as Float64 |
| `n8_catastrophe_diagnostic` | author raw-radian point and `argument_scale=1` | rounded author `M96/k^2` | A96 implementation; high-precision option promotes stored values |
| `benchmark_manifest` | declares raw angles in radians | rounded author `M96` | A96 metadata |
| N=8 HDF5 in stale repository path `appendix_c`, plus `read._kinetic_matrix` | HDF5 itself does not bind a periodic coordinate; generic readers invert stored `Kinv` | precise reconstructed `M96` | all stored real arrays are Float64 |
| `scripts/author_poly102_reference.py` | independent raw-radian transcription, `arguments=Q@x` | rounded author `M96/k^2` | independently confirms A96 behavior |

Relevant anchors are `reduced_models.jl:122-147, 252-281, 311-449,
500-625, 708-780`; `compatibility.jl:47-111`;
`poly102_inflation.jl:174-197, 590-609, 676-839, 1111-1192`;
`catastrophe_diagnostics.jl:220-244, 275-294`; and
`author_poly102_reference.py:24-115, 122-150`.

The root 12-term augmented catastrophe solve is metric-independent: it uses a
hierarchy preconditioner rather than the kinetic matrix.  The author 10-term
augmented solve uses a canonical Hessian to formulate its null equation, but a
positive constant rescaling of the metric only rescales that Hessian and does
not change its zero set.  Agreement in `k_c` is therefore expected and is not
evidence that P96 and A96 have equivalent canonical normalization.

## Coordinate, basis, and metric derivation

Let `theta` be the paper's dimensionless period-one GLSM coordinate and let

```text
x = 2pi theta,        theta = x/(2pi).
```

`x` is also dimensionless, with angular period `2pi`.  Restore reduced Planck
units by writing the equation-(96) numerical matrix as `M_Pl^2 M96`.  On the
radial ray under P96,

```text
K_theta(k) = M_Pl^2 M96/k^2.
```

Because `dtheta=dx/(2pi)`, coordinate covariance of
`ds^2=dtheta' K_theta dtheta=dx' G_x dx` requires

```text
G_x(k) = K_theta(k)/(2pi)^2
       = M_Pl^2 M96/[k^2 (2pi)^2].
```

For a symmetric square root, the same physical canonical field can be written
in either coordinate:

```text
phi_P = K_theta^(1/2) theta
      = K_theta^(1/2) x/(2pi)
      = G_x^(1/2) x.                         [mass dimension one]
```

The executable author field is instead

```text
phi_A = K_theta^(1/2) x = 2pi phi_P.
```

A Cholesky factor produces a canonically equivalent field related by an
orthogonal rotation.  Distances, Hessian spectra, projected derivative
magnitudes, and slow-roll scalars below do not depend on choosing the symmetric
root or Cholesky basis.

Let `H_x` be the raw Hessian of the raw-radian potential.  At corresponding
points,

```text
gradient_theta = 2pi gradient_x,
H_theta        = (2pi)^2 H_x.
```

The two correct coordinate representations of P96 agree:

```text
Hcanon_P
  = K_theta^(-1/2) H_theta K_theta^(-1/2)
  = G_x^(-1/2) H_x G_x^(-1/2)
  = (2pi)^2 M96_k^(-1/2) H_x M96_k^(-1/2),

Hcanon_A
  = M96_k^(-1/2) H_x M96_k^(-1/2),

Hcanon_P = (2pi)^2 Hcanon_A,
```

where `M96_k=M96/k^2`.  Therefore every canonical mass-squared eigenvalue scales
by `(2pi)^2`, and every mass magnitude scales by `2pi`.  Eigenvalue signs,
inertia, nullity, and the geometric eigendirections remain unchanged.  A raw
direction normalized with `G_x` is `2pi` times the same direction normalized
with `M_k`.

For a fixed canonical direction and the same raw point, the `n`th directional
derivative obeys

```text
D_P^n V = (2pi)^n D_A^n V.
```

Thus a normalized quadratic detuning coefficient scales by `(2pi)^2`, and a
quartic normal-form coefficient scales by `(2pi)^4`.  If the author settings'
numbers were converted algebraically at the same raw point, `93.86` would
become approximately `3705.4443`, and `1.52e7` would become approximately
`2.3689891e10`.  Those converted values are implications of the coordinate
contract, not newly validated fits.

The same global factor commutes with any fixed linear charge or divisor basis
change.  It cannot be repaired by choosing GLSM, leading-charge, mass, or
Cholesky coordinates differently.

## Empirical replay

The tracked audit script
`scripts/audit_issue_148_n8_metric_boundary.jl`
compares P96 and A96 at the same ten-term N=8 point, verifies the precise HDF5/root
matrix identity, and checks the tensor transformations without using the
published e-fold curve as a target.

Run:

```console
julia --startup-file=no --project=. scripts/audit_issue_148_n8_metric_boundary.jl
```

Observed results at this commit:

```text
author augmented k                         0.6745063700033650
metric precise/rounded relative Frobenius  1.024655e-04
2pi                                         6.2831853071795862
(2pi)^2                                     39.4784176043574320
sample canonical distance, author contract  4.2664078892523443e-04
sample canonical distance, paper contract   6.7901990482077026e-05
distance ratio                              6.2831853071795853
canonical Hessian matrix relative residual  4.286844e-16
epsilon ratio, paper/author                 39.4784176043574391
eta ratio, paper/author                     39.4784176043574604
scalar-amplitude ratio, author/paper        6.2831853071795871
third directional derivative ratio          248.0502134423984444
fourth directional derivative ratio         1558.5454565440400074
```

The third- and fourth-derivative ratios are `(2pi)^3` and `(2pi)^4`.
The script also reproduces the equation-(96) eigenspectrum for both the precise
and rounded matrices.  This is an empirical check of the algebra and path
identities.  It does not adjudicate which contract the owner wants.

The final replay environment was Julia 1.12.6 on Darwin 25.6.0 arm64 at exact
HEAD `1eef936e1908959db50a969e717b16a9c9e414bf`.  The script exited zero;
`git diff --check`, the focused trailing-whitespace scan, and
`scripts/agent_verify.py diff-check` passed.  The exact-intersection source
check used Python 3.14.6, CYTools 1.4.12, SymPy 1.14.0, and NumPy 2.4.6.  It
reconstructed `V=126`, the equation-(95) divisor volumes, and `M96` with a
maximum absolute Float64 comparison error below `8.2e-20`.

## Inferences and limits

The following conclusions combine the source and implementation facts; they
are not quotations or independent source claims.

1. The author point is a raw-radian representation because its entries and
   cosine arguments are related to the paper coordinate by `x=2pi theta`.
2. Because the author metric entries equal the reported equation-(96) values,
   the author workflow did not visibly apply the tensor factor `(2pi)^-2`.
3. The P96 paper contract and A96 author contract cannot both be the same canonical
   normalization.  Their agreement on the raw potential, catastrophe point,
   and `k_c` does not remove the contradiction.
4. Existing evidence cannot tell whether Issue 148 should privilege literal
   paper normalization or numerical reproduction of the author figures.  That
   is why the result is BLOCKED rather than a source-fixed correction.
5. P96 is recommended because three independent paper statements align on it:
   period-one fields, `2pi` Fourier arguments, and `phi~K^(1/2)theta`.  No
   benchmark fit is used to support that recommendation.
6. Neither contract establishes saxion stabilization, off-ray validity, an
   exhaustive catastrophe population, or physical predictivity beyond the
   fixed-saxion benchmark.

## Observable and acceptance impact

The following ratios hold at the same raw-radian point and with the same
potential normalization:

| Quantity | P96 relative to A96 | Consequence |
|---|---:|---|
| Canonical distance / field range | `1/(2pi)` | A96 reports fields and path lengths `2pi` larger |
| Canonical gradient magnitude | `2pi` | local slope changes |
| Canonical Hessian / mass squared | `(2pi)^2` | eigenvalue signs are unchanged, values are not |
| Canonical mass magnitude | `2pi` | absolute mass diagnostics change |
| `n`th canonical directional derivative | `(2pi)^n` | quartic coefficient changes by `(2pi)^4` |
| `epsilon_V` | `(2pi)^2` | the `epsilon=1` boundary moves along a trajectory |
| `eta_parallel` | `(2pi)^2` | the `abs(eta)=1` boundary moves |
| `r=16 epsilon` at the same point | `(2pi)^2` | same-point tensor ratio changes |
| scalar amplitude proportional to `sqrt(V/epsilon)` | `1/(2pi)` | A96 is `2pi` larger at the same point |
| E-fold integral between the same raw endpoints | `1/(2pi)^2` | A96 is `(2pi)^2` larger |
| Cumulative turn angle for the same geometric raw curve | `1` | a global field rescaling cancels from angles |
| `k_c`, stationary point, inertia, nullity, catastrophe type | `1` | these cannot select P96 versus A96 |

The fixed initial displacement in the author workflow is a canonical number.
If both contracts use the numerical value `1e-8 M_Pl`, their raw starting
points differ by a factor `2pi`.  Full trajectory observables then do not follow
only the same-point ratios in the table.  Likewise, quantities evaluated “60
e-folds before the end” occur at different raw points.  G2 must bind the
coordinate contract before it constructs the initial condition, end event, or
observable slice.

Constant metric rescaling preserves the slow-roll path as an unparameterized
raw curve when the raw start and endpoint are held fixed.  It changes the
gradient-flow parameter and e-fold accumulation.  Matching the same
catastrophe `k` or the same raw path is therefore insufficient evidence for
canonical equivalence.

## Precision and source-construction boundary

There are two numerical N=8 matrices in the current tree:

1. **Precise reconstructed matrix.** `reduced_models.n8_geometry()` and the
   HDF5 `Kinv` inversion agree to Float64 rounding.  Its eigenvalues in
   ascending order are
   `5.8372911720e-5, 8.3012530049e-5, 9.1523923188e-5,
   1.2406738401e-4, 3.1297658750e-4, 5.9740905741e-4,
   6.3485841597e-4, 8.1968936710e-4`.
2. **Rounded author matrix.** `author_inflation.N8_K_RAW` reproduces the
   four-significant-figure Mathematica matrix.  Its relative Frobenius
   difference from the precise matrix is `1.024655e-4`.

Both are stored as Float64.  Calling `BigFloat.(M)`, `T(N8_KC)`, or
`T.(N8_BEST_X)` carries the exact binary Float64 values into a wider type; it
does not create new source precision.  This occurs in
`n8_author_trajectory` (`poly102_inflation.jl:1133-1158`) and in the
high-precision catastrophe diagnostic
(`catastrophe_diagnostics.jl:226-240`).  G2 may use those values as seeds, but
must not describe the resulting metric or seed data as independently sourced
at 100 or 120 bits.

The potential admits a genuine high-precision construction because its charges
are exact integers, its listed `q.tau` values are integers or half-integers,
and its zero phases are exact.  A BigFloat augmented solve should construct
those values from integers/rationals and parse only decimal seeds as seeds;
the converged `k`, point, residuals, and derivatives can then carry genuine
working precision.

The metric also has a possible exact reconstruction route, but it is not
implemented in the current N=8 paths.  The stored triangulation gives exact
integer intersections, and the stretched-cone tip is numerically
`(1,4,4,-2,4,3,3,3)`, yielding exactly `V=126` and the equation-(95) divisor
volumes.  With `A_ij=kappa_ijk t^k`, exact rational arithmetic reconstructs the
CYTools numerical matrix.  This bounded audit found that CYTools 1.4.12 uses

```text
Kinv_CYTools = 4 (tau tau' - V A),
M_CYTools    = inverse(Kinv_CYTools).
```

The exact rational reconstruction agrees with the HDF5/root matrix to
`8.2e-20` maximum absolute Float64 error.  G2 can therefore avoid pretending
that Float64 promotion adds precision: either keep a declared 53-bit metric
source bound, or reconstruct the matrix from exact intersections and the exact
tip in a separately verified arbitrary-precision path.

There is a secondary source-consistency caveat.  Direct differentiation of
the paper's equations (17) and (23), with `K=-2 log V` and the paper's real
`tau`, gives

```text
H_tau = -A^(-1)/V + t t'/(2 V^2),
K_equation23 = H_tau/2,
M_CYTools     = H_tau/4.
```

Thus the equation-(96) numbers, the author matrix, and the repository matrix
match `M_CYTools`, while a literal evaluation of the displayed equation (23)
is larger by a factor two.  This audit classifies that as an observed source
normalization inconsistency, not as authority to rewrite `M96`.  The primary G2
decision above deliberately holds the equation-(96)/benchmark numerical `M96`
fixed.  Any G2 claim that it has newly derived the absolute kinetic
normalization from equations (17) and (23), rather than replayed the reported
matrix, would require the owner to resolve this additional factor two.

## Bounded G2 inputs after the owner decides

G2 implementation can remain bounded to the following inputs:

1. Record `coordinate_contract = P96` or `A96` in every N=8 canonical certificate.
2. Keep the potential coordinate and metric paired:
   - P96: period-one `theta`, `argument_scale=2pi`, metric `M96/k^2`;
   - A96: raw-radian `x`, `argument_scale=1`, metric `M96/k^2`.
   A raw-radian representation of P96 must use `M96/[k^2(2pi)^2]`.
3. Do not call the root `trajectory=true` derivative path without an explicit
   coordinate conversion; it changes the root coordinate contract.
4. Choose and record `metric_source = precise_reconstruction` or
   `rounded_author_artifact`.  Use the precise reconstruction for P96 and the
   rounded matrix only when reproducing A96 unless the owner says otherwise.
5. Construct integer charges, rational action data, exact zero phases, and the
   radial `k^-2` law in the target precision.  Treat Float64 `k_c`, point, and
   matrices only as seeds or source-limited fixtures.
6. Report raw stationarity and null residuals separately from canonical
   Hessian and higher-derivative values.  Use scale-aware tolerances so the
   `(2pi)^n` factors cannot change acceptance by accident.
7. Bind canonical displacement before solving a trajectory.  Never compare P96
   and A96 e-folds using the same numeric canonical displacement as though they
   were the same raw start.
8. Retain the G0 distinction between the twelve-row Table-1 potential and the
   ten-row executable-author trajectory truncation.  Metric normalization does
   not resolve that separate source choice.

An owner response approving the P96 block at the start of this audit resolves
both the coordinate and numerical-matrix boundary and follows the evidence-
based recommendation.  An owner response of “A96 for all G2 canonical outputs”
is also implementable, but those outputs must be labeled as executable-author
normalization and must not be described as the literal period-one
canonicalization in equations (19), (22), and (23).  An owner response of
“P23” selects the displayed equation-(23) normalization `2M96`; it will not
reproduce equation (96) and requires corresponding factor-two updates in every
canonical quantity.
