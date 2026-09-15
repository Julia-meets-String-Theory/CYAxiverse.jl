# P0 numerical semantics and pathological fixtures

Status: evidence-only archaeology and fixture corpus.  This artifact does not
change a production algorithm.  The numerical source is pinned to
`7a40285bb5c313f7e8746b90644d5f45bb67be44` (`vmm` at the P0 reference point).
The fixture manifest and raw fixture hashes are in
`validation/p0_numerical_equivalence/fixtures/manifest.toml`.

## Evidence classes and replay boundary

This record distinguishes four kinds of statement:

* **Source fact** — directly read from the pinned Julia source, with a
  repository-relative line anchor.
* **Implementation fact** — the route-level contract inferred by following
  calls between pinned source files.
* **Empirical fixture replay** — the bounded, dependency-free replay emitted by
  `scripts/fixture_replay.py`.
* **Inference/requirement** — a future P2 acceptance rule or an unresolved
  scientific decision.  It is not a claim about a source result.

The full package could not be loaded in this checkout: Julia 1.12.6 attempted
to write the user-level compiled-package cache and was denied by the execution
sandbox; with compiled modules disabled, package dependencies were not
instantiated.  Therefore this worker did not report package-level Julia
benchmark output as executed.  The helper replays only the small arithmetic
policies visible in the source, and marks the F11 named benchmark as a source
fixture rather than pretending to have run its package route.

Executed focused check:

```text
python3 validation/p0_numerical_equivalence/scripts/fixture_replay.py
```

Observed result: exit 0; all 13 fixture payloads parsed, all manifest SHA-256
entries matched, and all checks emitted by the helper passed.  The helper's
output is intentionally a diagnostic stream, not a new scientific oracle.

## Historical encoded input

For a term `a`, preserve the two stored fields separately:

```text
Lambda[a] = c[a] * 10^ell[a]
L[1,a]    = c[a]       (signed coefficient factor / mantissa)
L[2,a]    = ell[a]     (base-10 logarithmic exponent)
```

The source documentation calls `L[1,:]` signs in some routes, but the
critical-point and derivative implementations multiply that value directly
(`src/minimizer.jl:133-146`, `src/generate.jl:395-428`).  F3 therefore uses
`2.5` and `-0.25`; a future implementation must not silently canonicalize
these values to `+/-1`.

The canonical in-memory package orientation is `Q :: h11 x N` and
`L :: 2 x N`, one instanton per column (`docs/src/pipelines.md:63-81`,
`src/read.jl:463-477`).  Older generic helpers in `generate.jl` also expose a
row-oriented legacy convention (`L` has one term per row in
`src/generate.jl:522-549`); this is a route boundary, not permission to
transpose all persisted data.  Every fixture records the canonical
axion-by-instanton form unless its purpose is explicitly source metadata.

`c=0` is an encoded exact zero and does not remove the term.  A zero coefficient
can still affect a selected logarithmic reference because reference selection
uses stored `ell` before multiplication.  `ell=-Inf` is represented in TOML as
the string `"-Inf"` to avoid a non-standard TOML numeric token.  It is decoded
to IEEE negative infinity by the helper.

## Term-order and orientation boundaries

The historical route is not one universal “raw HDF5 order” rule:

| Route | Historical boundary | Order consequence |
| --- | --- | --- |
| `read.potential` / `read.potential_factored` | Read the stored HDF5 potential and return raw arrays; `potential` selects either `cytools/potential` or `cytools/hilbert/potential` (`src/read.jl:241-296`). | No generated-term deduplication is applied here. |
| `read.oriented_potential` | Repairs either matrix orientation, validates dimensions/finite values, and by default calls `_canonicalize_generated_potential` (`src/read.jl:393-432`). | Only triangular generated potentials with duplicate leading charges are canonicalized; the first representative and its coefficient are retained, and redundant pairwise terms are removed/rebuilt. Non-triangular data are left unchanged. `canonicalize_charge_rows=false` preserves the oriented order. |
| `generate.LQtilde` | Sorts by descending `L[2,:]`, then scans the sorted columns for independent charges (`src/generate.jl:3140-3174`). | Selected and remaining terms follow the descending-log selection route, not persisted order. |
| `generate.reduced_critical_points` | Calls `LQtilde`, then passes `[selected; remaining]` to `critical_points` and supplies equation scales (`src/generate.jl:3189-3205`). | The critical-point term order is the route's ordered selection output. |
| `generate.LQtildebar` and vacua routes | Sort by descending logs and apply route-specific relevance/alpha filtering (`src/generate.jl:3748-3808`). | Filtering and alpha construction may change the scientific input population. |
| `jlm_reduced._oriented_potential_matrices` | Repairs orientation but does not call the generated-charge canonicalizer (`src/jlm_reduced.jl:63-76`). | Its order is the input order to its own `LQtilde` path. |
| `inflation_points.prepare_geometry_context` | Calls `read.oriented_potential` with its default canonicalization (`src/inflation_points.jl:194-200`). | Inflation context sees the oriented/canonicalized route. |
| direct `minimizer.critical_points` | Uses the supplied `q`, `phase`, `signs`, and `logscale` in their supplied column order (`src/minimizer.jl:128-185`). | No term sort occurs inside this function. |

**Future input-contract boundary:** preserve the route-specific transformation
before constructing a future `PotentialDefinition`.  Do not spread generated
charge deduplication into raw-reader, direct critical-point, or unrelated
legacy routes unless a separate scientific decision authorizes it.

## Zero, global scaling, row scaling, and support

There are several distinct normalization policies.

* **Global/log-shifted evaluator:** `log_shift = maximum(L[2,:])`, including
  terms with `c=0`; normalized amplitudes are
  `c[a] * 10^(ell[a]-log_shift)` (`src/generate.jl:374-384`,
  `src/inflation_points.jl:102-112`).  F4 makes a zero term carry the dominant
  stored log scale.
* **Critical-point row scale:** first transform
  `q = coordinate_basis \ Q` (or materialize `Q` directly), convert
  `L/phases/q` to Float64, and for each row choose the largest `ell` among
  entries satisfying `!iszero(q[i,j])` (`src/minimizer.jl:128-146`).  A zero
  coefficient remains in support if its transformed charge is nonzero, so its
  `ell` can set the row reference.  F7 and F12 cover this boundary.
* **Equation scales:** optional caller-provided scales are divided by their
  maximum absolute value and must remain strictly positive
  (`src/minimizer.jl:148-150`).  These are separate from `row_logscale`.
* **Float64 floor:** a scaled contribution is used only when its log delta is
  at least `log10(floatmin(Float64))`; otherwise the term is set aside in the
  critical-point gradient/Hessian routes (`src/minimizer.jl:136-146`,
  `src/minimizer.jl:153-185`).

For an empty support row the historical rule is explicit:
`row_logscale[i] = maximum(logscale)` (`src/minimizer.jl:137-141`).  It is not
`-Inf`, zero, equation deletion, or an error.  F12 freezes this behavior.

For all-`-Inf` logs, `-Inf - (-Inf)` is `NaN`.  In the direct critical-point
path, the scaled-amplitude comparison is false but the separately assembled
classification matrix can receive `NaN` because `NaN < floor` is also false
(`src/minimizer.jl:136-161`).  Generic workspace and inflation context
validators reject non-finite `L` (`src/generate.jl:376-381`,
`src/inflation_points.jl:88-99`).  F13 records this as a historical defect and
cross-route ambiguity; P0 does not repair or harmonize it.

## H1, H2, and H3

The three matrices must stay named separately in future equivalence tests.

* **H1 — physical Hessian.**  The actual second derivative of the chosen
  physical/log-shifted potential in the stated coordinates.  Examples are
  `generate.hessian` for its legacy radian route
  (`src/generate.jl:539-549`), the benchmark derivative Hessian
  (`src/paper_benchmarks/reduced_models.jl:248-258`), and
  `inflation_points._derivatives` (`src/inflation_points.jl:203-233`) after
  global log shifting.  It is not row-normalized.
* **H2 — stationarity/Newton Jacobian.**  `critical_points.hessian!` uses
  row-local `scaled_amplitudes`, multiplies by `(2pi)^2`, and divides each row
  by `row_scales` (`src/minimizer.jl:177-185`).  It may be nonsymmetric and is
  the Jacobian consumed by `nlsolve` (`src/minimizer.jl:207-211`).
* **H3 — classification/congruence matrix.**  The historical
  `scaled_physical_hessian!` uses the geometric-mean row log reference for
  each pair and then a symmetric factor `(2pi)^2`
  (`src/minimizer.jl:153-164`).  The returned
  `critical_points.hessian_eigenvalues` are eigenvalues of H3
  (`src/minimizer.jl:226-238`), not unconditional eigenvalues of H1.

`inflation_points.mass_eigenbasis` separately solves the generalized physical
problem `H1*v = m^2*K*v` by Cholesky whitening
(`src/inflation_points.jl:146-169`); its `hessian_eigenvalues` field in
`PointDiagnostics` means those generalized H1 eigenvalues.  A future contract
must record which of H1/H2/H3 each observable refers to.

## Seed versus final thresholds

The `critical_points` thresholds are distinct and both are frozen:

1. **Seed displacement mode threshold:** seed H3 eigenvalues are selected by
   `value < -100*eps(Float64)` (`src/minimizer.jl:193-203`).  With IEEE
   Float64 this boundary is `-2.220446049250313e-14`.
2. **Final inertia threshold:** for a root, set
   `scale=max(maximum(abs, values),1.0)` and
   `zero_tolerance=100*residual_tolerance*scale`; classify below
   `-zero_tolerance`, within the closed band, or above it
   (`src/minimizer.jl:226-235`).  The default residual tolerance is `1e-10`.

F10 places a cancellation-derived mode at this final band boundary while the
positive mode fixes the scale.  A migration must not reuse final inertia logic
for seed generation.  The `inflation_points.diagnose` keyword
`zero_tolerance=1e-10` is another diagnostic threshold and is not a substitute
for either critical-point threshold (`src/inflation_points.jl:247-265`).

## Phase, displacement, and evaluation order

These operations are distinct contracts:

* `critical_points` computes
  `arguments = (2pi .* (q' * theta)) .+ phase` in that order
  (`src/minimizer.jl:153-180`).  `phase` has one entry per instanton and is in
  radians; it is not a coordinate vector.
* The benchmark derivative route uses the same ordered expression
  (`src/paper_benchmarks/reduced_models.jl:248-258`).
* The legacy `generate.V/jacobian/hessian` route uses `Q' * x` / `sum(x .* Q)`
  without the critical-point `2pi` factor (`src/generate.jl:522-549`); those
  coordinates are a separate radian route.
* `inflation_points.gradient_flow` forms a raw-coordinate displacement
  `periodic(hilltop + sign*displacement*direction)`, then maps it to the
  canonical Cholesky chart `chi = L' * theta`
  (`src/inflation_points.jl:315-342`).  F6 freezes this as a displacement, not
  a phase offset.
* The convenience `generate.cubic(x, phase, L, Q)` overload implements
  coordinate addition `cubic(x + phase, L, Q)` (`src/generate.jl:579-587`),
  which is another route-specific convention and must not be conflated with
  the per-instanton argument phase.

No reassociation, phase absorption, or `@fastmath` is allowed in a numerical
parity implementation until a separate tolerance review covers the changed
evaluation order.

## Float64, arbitrary precision, and concurrency

* `critical_points` accepts `AbstractMatrix{<:Real}`/real phases but immediately
  materializes `Q`, `L`, and phases as Float64 (`src/minimizer.jl:116-146`).
  This is historical Float64 behavior even when callers pass BigFloat inputs.
* The log-shifted workspace is concretely Float64 and requires finite `L`
  (`src/generate.jl:100-112`, `src/generate.jl:374-384`).
* Legacy minimizer helpers convert inputs to `ArbFloat` and preserve concrete
  arbitrary precision (`src/minimizer.jl:44-82`).
* High-precision spectrum/Hessian routes use `ArbFloat` and call global
  `setprecision(ArbFloat; digits=prec)` (`src/generate.jl:897-918`,
  `src/generate.jl:1466-1483`).
* `inflation_points.prepare_context` supports Float64 by default and BigFloat
  at requested precision; it scopes `setprecision(BigFloat, bits)` around
  context construction and each context evaluation
  (`src/inflation_points.jl:115-139`).

**Future concurrency requirement:** a path that changes global/default
BigFloat or ArbFloat precision is not considered safely concurrently
executable across threads without separate evidence.  P0 does not redesign
precision state or claim thread-safe BigFloat execution.

## Workspace and result aliasing

`generate.logshifted_derivatives!` returns a named tuple whose `gradient` and
`hessian` fields are the exact reusable arrays stored in the workspace
(`src/generate.jl:390-429`).  The next call begins with `fill!`, so a retained
result is invalidated after workspace reuse.  This is a borrowed internal
result, not a durable/public ownership guarantee.

By contrast, `inflation_points._derivatives` allocates fresh gradient and
Hessian arrays and returns an immutable `PointDerivatives` value
(`src/inflation_points.jl:203-233`).  Future P2 APIs must state ownership at
the boundary: an internal `!` function may borrow storage only with an
explicit lifetime, while a public/durable result must copy before workspace
reuse.  Required regression pattern: retain result A, evaluate result B using
the same workspace, and verify the promised borrowed/copied behavior.

## F1--F13 fixture catalogue

Each payload includes its identity, pinned source revision and line anchors,
construction provenance, purpose, expected historical result, precision and
environment.  The manifest records the SHA-256 hash of each raw TOML payload.

| ID | Payload | Historical boundary frozen |
| --- | --- | --- |
| F1 | `fixtures/F01_disjoint_support_hierarchy.toml` | Disjoint `[0,-400]` terms select row scales `[0,-400]`; each local row retains its own term at normalized amplitude 1. |
| F2 | `fixtures/F02_mixed_support_hierarchy.toml` | Shared `-400` term falls below the Float64 delta floor where its row reference is dominant; mixed support still selects row scales independently. |
| F3 | `fixtures/F03_nonunit_coefficient_factor.toml` | Factors `2.5` and `-0.25` are multiplied directly; no sign-only canonicalization. |
| F4 | `fixtures/F04_exact_zero_and_minus_inf.toml` | Zero coefficients remain encoded; finite dominant zero logs affect references; `-Inf` is accepted by direct critical-point code but rejected by finite-input validators. |
| F5 | `fixtures/F05_argument_phase_offsets.toml` | Per-instanton radian offsets are added after `2pi*(q' * theta)` and remain separate from coordinates. |
| F6 | `fixtures/F06_coordinate_displacement.toml` | Raw-coordinate mass-mode displacement is periodicized before canonical-chart integration; it is not a phase vector. |
| F7 | `fixtures/F07_transformed_near_zero_support.toml` | `A\\Q` is Float64 materialized before support detection; `8.881784197001252e-16` is nonzero and supported. |
| F8 | `fixtures/F08_transformed_noninteger_charges.toml` | Noninteger transformed charges are accepted under the current real-valued dispatch domain. |
| F9 | `fixtures/F09_signed_cancellation.toml` | Opposite signed terms cancel without pruning, preserving encoded term count and accumulation order. |
| F10 | `fixtures/F10_near_degenerate_classification.toml` | Dominant cancellation places one H3 mode at the final inertia band; seed and final thresholds remain separate. |
| F11 | `fixtures/F11_governed_n5_source_fixture.toml` | Bounded named N=5 source reconstruction (`specs/0148-catastrophe-continuation` / paper benchmark), with no population claim. |
| F12 | `fixtures/F12_empty_support_row.toml` | Empty support uses `maximum(logscale)` and keeps the zero equation present. |
| F13 | `fixtures/F13_zero_only_all_minus_inf.toml` | All-`-Inf` support reaches `-Inf - (-Inf) => NaN` in the direct classification path; cross-route validator rejection is retained as a defect. |

F11's named source identity is `arXiv:2608.14780v1`, with the source digest
recorded by `src/paper_benchmarks/catastrophe_diagnostics.jl:3-12`; its bounded
N=5 arrays are copied from `src/paper_benchmarks/reduced_models.jl:11-35`.
The fixture records the governed `k_c` checks (two reduced minima at
`k_c-1e-4`, one at `k_c+1e-4`) but does not promote them to a population claim.
The legacy `src/paper_benchmarks/poly102_inflation.jl:158-164` route still
defines `n5_critical_scale() = N8_KC` (approximately `0.674506370003365`).
The accepted G1 record identifies that value as a pre-existing N=5 fixture
defect; F11 intentionally follows the source-faithful N=5 value from
`reduced_models.jl:150-153`.  This is a recorded route conflict, not a P0
production change.

## Open scientific/implementation decisions for later review

The source facts are sufficiently determinate to freeze the F1--F13 input and
route semantics.  The following are not silently resolved in P0:

1. whether the `-Inf` zero-only state should remain reachable in a future
   contract or become a single explicit rejection policy;
2. whether any later route may use canonical generated-charge deduplication
   beyond `read.oriented_potential`'s existing boundary;
3. the tolerances for future P2 equivalence tests, which must be precommitted
   from historical repeated behavior, analytic fixtures, and high-precision
   diagnostics before observing P2 output;
4. the public ownership/copy policy for future workspace-backed result types;
5. whether any changed BigFloat/ArbFloat precision policy can be evidenced as
   thread-safe.

These are stop/owner-review conditions, not defects to fix in P0.
