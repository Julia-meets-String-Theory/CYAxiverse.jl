# CYAxiverse numerical-equivalence contract v1

Status: **owner-approved CYAX-0170 P0 baseline; evidence G2 PASS; exact synchronization candidate pending normative-fidelity review**

Contract version: `cyaxiverse-numerical-equivalence-v1`

Scientific reference: `7a40285bb5c313f7e8746b90644d5f45bb67be44`

Governing specification: `specs/0170-p0-numerical-equivalence/spec.md`

This contract freezes historical observations for later differential tests.  It
does not declare every historical result physically desirable, does not repair
known defects, and does not authorize P1/P2/P3A/P3B work. Owner approval is
recorded in Issue #170 comment `5685945127`.

The scientific reference remains
`7a40285bb5c313f7e8746b90644d5f45bb67be44`. Independently reviewed P0 evidence
is identified by candidate `ddc304f25040ff36bc84e7897d5b6dc5d16344f6`;
`6f2acaef2a181937f63d8f085c4317e91a08267b` is mechanical bookkeeping only.

## 1. Evidence classes

- **Source fact:** directly established by the pinned source.
- **Implementation fact:** observable behavior of that source at the recorded
  route boundary.
- **Empirical evidence:** an executed result tied to fixture and environment.
- **Owner-approved contract:** a P0 comparison rule approved for later
  differential implementation, subject to the route-specific dispositions below.
- **Open owner decision:** normative intent P0 must not infer.

Evidence sources are:

- `environment_and_load.md` and the retained `environment/` lockfiles;
- `compatibility_inventory.md`;
- `numerical_semantics_and_fixtures.md` and `fixtures/manifest.toml`;
- `benchmark_baseline.md` and `benchmark_results/`.

## 2. Input and route identity

A differential comparison is valid only when both sides identify:

1. source/candidate revision and exact resolved environment;
2. fixture content hash and construction/source provenance;
3. selected public/internal route and its pre-route transformations;
4. matrix orientation, coordinate/basis convention, phase units, and term order;
5. arithmetic type/precision, thread/BLAS state, seed/start set, and solver options;
6. warm-up/sample/allocation method for performance comparisons.

A result from one route is not an oracle for a different route merely because
the mathematical expression can be rewritten into a similar form.

## 3. Stored potential and term-order contract

For each term `a`, preserve the encoded pair

```text
Lambda_a = c_a * 10^ell_a
```

with signed coefficient/mantissa `c_a` and stored base-10 exponent `ell_a`
separate.  Equality of only the multiplied real number is insufficient.

The comparison SHALL be exact for:

- encoded coefficient and exponent arrays, including signed/exact zero;
- number of terms and their order at the selected route boundary;
- charge orientation and entries before/after the named route transformation;
- phase vector units and order;
- source indices, aliases, return field names, status values, and HDF5 path,
  shape, orientation, units, and schema.

Route boundaries remain distinct:

- raw `read.potential` order;
- `read.oriented_potential` orientation and generated-term canonicalization;
- `generate.LQtilde` / `LQtildebar` sorting and selection;
- `jlm_reduced` orientation and its own selection;
- `axion_photon` descending-scale order/source indices;
- direct `minimizer.critical_points` caller-supplied order;
- fixed paper-benchmark routes.

Canonicalization in one route SHALL NOT silently spread to another route.

## 4. Zero, `-Inf`, normalization, and support

Representation validity is distinct from policy admissibility.

- Zero-coefficient terms remain in the encoded input and may affect the global
  log shift or row reference before their mathematical contribution becomes zero.
- Physical/global scaling uses the historical maximum stored log over encoded
  terms for routes that implement that policy.
- Stationarity support uses the historical coordinate transform, Float64
  materialization, and exact `iszero` test on transformed charge entries.
- A supported zero-coefficient term can set `row_logscale` through its stored log.
- An empty-support row uses `maximum(logscale)` and remains a zero equation.
- `row_logscale`, support masks, included/underflowed term masks, and empty-row
  presence compare exactly.

Current routes conflict for non-finite log inputs: generic workspaces and the
inflation context reject them, while direct critical-point construction can
reach `-Inf - (-Inf) => NaN`. F13 freezes both observations. The owner-approved
P0 disposition is route-specific preservation: validator routes retain
historical non-finite rejection, while direct critical-point non-finite
propagation/failure remains a historical parity defect. P0 performs no
harmonization.

## 5. Argument and coordinate semantics

The following are separate operations and compare at the named boundary:

- per-instanton argument offset added after the route's charge/coordinate map;
- coordinate displacement applied to coordinates, including periodicization;
- unit-torus realization, including multiplication by `2pi` at its historical
  position;
- legacy-radian realization.

Algebraic equivalence does not permit changing floating-point evaluation or
reduction order when exact historical-order parity is required.  Argument maps
and periodicized coordinates compare bitwise in the retained environment for an
unchanged route; an approved replacement kernel uses the numeric envelopes in
section 9 only after its route identity and order policy are explicit.

## 6. Physical derivatives and precision

Compare separately:

- physical potential value;
- physical gradient in its stated coordinates;
- H1 physical Hessian;
- mass/generalized-Hessian quantities and basis maps;
- epsilon and eta-related quantities;
- serial Float64 and currently supported BigFloat behavior.

BigFloat precision is part of the test identity.  A path that changes global or
default BigFloat precision is not concurrently thread-safe by assumption; any
thread-safety claim requires separate evidence.  Higher precision is a
diagnostic and is not automatically the historical Float64-parity oracle.

## 7. Stationarity and classification

The following objects SHALL remain separate:

- H1: physical Hessian;
- H2: row-scaled stationarity/Newton Jacobian, potentially nonsymmetric;
- H3: symmetric congruence-scaled classification matrix.

Historical `critical_points.hessian_eigenvalues` are H3 eigenvalues.  A field or
test SHALL NOT relabel them unconditionally as physical-Hessian eigenvalues.

Seed displacement and final inertia are separate gates:

- Seed selection uses the historical seed-mode threshold and determines which
  displaced starts exist.
- Final inertia uses
  `100 * residual_tolerance * max(maximum(abs, eigenvalues), 1)` as its zero band
  in the direct critical-point route.

Compare the start set, seed modes/displacements, converged/discarded statuses,
periodic root matching, unique root set, residuals, H3 eigenvalues, and inertia.
The present public result does not expose every discarded-start status; this is
an explicit observability gap, not a license to infer success from root count.

## 8. Root-set equivalence

For a fixed start set and solver configuration:

1. A candidate root must meet the route's historical residual gate (default
   `1e-10` for direct `critical_points`).
2. Coordinate distance is periodic on the unit torus using
   `max_i min(abs(delta_i), 1-abs(delta_i))`.
3. Two roots match when that distance is at most the predeclared merge tolerance
   (default `1e-7` for direct `critical_points`).
4. Matching is one-to-one; counts alone do not establish set equality.
5. Status/failure class and inertia tuple compare exactly after matching.

If the selected route uses different governed residual/merge defaults, those
values must be recorded in the fixture identity and applied symmetrically.

## 9. Precommitted numeric envelopes

These envelopes are fixed before any P2 output is observed.  A route-specific
historical test or solver gate overrides a generic row when it is stricter and
is recorded in the fixture identity.

| Object | Approved comparison envelope | Historical basis |
|---|---|---|
| Encodings, masks, order, indices, statuses, counts, inertia, events | exact | Discrete compatibility/semantic contract |
| Same route/order/environment deterministic replay | `isequal` where promised by the existing API/test; otherwise the route-specific row below | Detect unintended order changes |
| Normalized/log-shifted Float64 value, gradient, H1/H2/H3 components | componentwise `isapprox(rtol=1e-13, atol=1e-13)` | Existing structured/generic derivative tests in `test/runtests.jl` and B3 predeclared parity probe |
| Direct critical-point residual | `<= 1e-10` unless fixture records another existing value | Current default in `src/minimizer.jl` |
| Periodic root match | distance `<= 1e-7` unless fixture records another existing value | Current default in `src/minimizer.jl` |
| Final direct-critical-point inertia | exact tuple after historical residual-dependent band | Current formula in `src/minimizer.jl` |
| `inflation_points` Float64 correction | residual `<= 1e-10`; exact terminal status | Current default |
| `inflation_points` BigFloat correction | residual `<= 1e-40` at recorded precision; exact status | Current `compare_precision` default |
| Mass/diagnostic inertia | exact tuple after `1e-10 * max(maxabs(eigs),1)` band | Current `diagnose` default |
| Spectrum mass logs and quartic logs on established fixtures | `atol=1e-10`, with exact signs/indices/counts | Repeated existing spectrum regression envelope |
| Analytic N=5 scale/ratio fixture | `atol=64eps(Float64)` for branch decision; stored source constant at its existing test envelope | `reduced_models.jl` analytic gate |
| Basis/metric identities | `rtol=1e-12`, `atol=1e-15`, unless governed fixture is stricter | Existing package regression envelope |
| Inflation/correction coordinate states | route's declared solver tolerance; compare periodic coordinates one-to-one, status/event exactly, derived scalars at `rtol=1e-8` unless a governed fixture specifies stricter | Existing bounded inflation regression envelopes |

The `1e-13` normalized derivative envelope is not valid as an absolute tolerance
for raw unnormalized quantities near `1e-40`.  Such routes must compare a shared
log-shifted/normalized representation or use a fixture-specific scale before
applying the table.  Zero and underflow classification remain exact semantic
checks.

Performance is not numerical equivalence.  B1-B7 timings are protection
baselines with environment/method identity; later performance acceptance must
predeclare its statistic and regression budget separately.

## 10. Inflation, correction, and spectra

For stationary correction compare seed, precision, tolerance, status,
iterations, periodic final coordinate, residual, diagnostics, and failure text
class.  Hessian-evaluation and line-search counts are unavailable from the
current API and remain explicit gaps.

For inflation compare initial condition/basis/displacement, solver policy,
trajectory samples at named independent variables, entry into the slow-roll
interval, interval duration/efolds, terminal event/status, and failure class.
RHS/Hessian counters are unavailable in current bounded routes and remain gaps.
Local normal-form diagnostics and nonlinear flow are distinct routes.

For protected spectra compare retained mode indices, mass logs, signs,
eigenvectors up to the approved sign/degenerate-subspace convention, quartic
indices/signs/logs, diagnostics, provisional/fallback/certified statuses, and
threshold/window identity. B6 shows a material current Float64/high-precision
light-mode disagreement; v1 records it as route-specific historical behavior.
The Float64 and high-precision baselines remain separate; numerical/physical
authority is unresolved, and no future migration may silently choose one route
as an oracle.

## 11. Workspace/result ownership acceptance

Future internal mutating APIs may return borrowed workspace storage only when
the lifetime is explicit.  The required regression pattern is:

```text
evaluate and retain result A
reuse the same workspace for result B
verify A changes only if the API explicitly promises borrowed storage
verify every public/durable result remains stable
```

Public results must not unknowingly mutate after workspace reuse.  The exact
borrowed-versus-copied policy for each future result type must be approved in
the later implementation spec before that API is accepted.

## 12. P2 Julia acceptance gate

A later numerical representation passes only when:

- hot plan/workspace fields and inner-loop containers are concrete/parametric,
  or an exception has measured justification;
- dispatch selects the numerical kernel before the hot instanton loop;
- dynamic selection is behind a function barrier and avoidable repeated dynamic
  dispatch is absent from hot loops;
- targeted inference/JET checks and method-ambiguity/Aqua checks pass;
- runtime values are not indiscriminately lifted into `Val` specialization;
- Float64, supported high-precision, optional-Python, HDF5, and compatibility
  domains frozen by P0 remain covered;
- the complete F1-F13 and protected B2-B7 differential evidence passes this
  contract, including exact failure/status behavior.

## 13. P3 migration acceptance gate

Each migrated consumer must identify its historical route and pass the relevant
input/order/derivative/result/failure comparisons before the old path can be
removed.  Inflation and critical-point migrations are reviewed as distinct S2
scientific changes.  A matching aggregate root, minimum, or spectrum count is
insufficient without identity/matching and diagnostic parity.

## 14. Approved route-specific dispositions

The owner approved the P0 baseline and this contract in Issue #170 comment
`5685945127`, subject only to mechanical synchronization of these dispositions:

1. **F13 all-minus-infinity behavior.** Validator routes retain historical
   non-finite rejection. Direct critical-point non-finite propagation/failure
   remains a historical parity defect. No P0 harmonization is permitted.
2. **N=5 authority.** The governed/source-faithful top-level N=5 route is
   authoritative for future modularisation and equivalence gates. Stale nested
   legacy N=5 behavior is historical only and is not the scientific oracle.
3. **B6 spectrum authority.** Float64 and high-precision baselines remain
   separately preserved. Numerical/physical authority is unresolved; no future
   migration may silently choose one route as oracle.
4. **Phase/volume Hessian factor.** Observed `2pi^2 I` is historical defect
   evidence, not intended behavior. The `4pi^2 I` correction is separate Issue
   #172 work. Issue #173 remains separate.

The underlying facts remain evidence: the stale
`poly102_inflation.n5_critical_scale()` route, unavailable failed-start and
internal RHS/Hessian/line-search counters, and the pre-existing package-test
failure are not repaired or reinterpreted here. The exact future
workspace-result borrowing/copying policy remains a later implementation
decision. This contract does not authorize production changes or any P1/P2/P3A/P3B
work.

## 15. Review identity

The G2 verdict must name the exact Git candidate revision, this contract version,
fixture-manifest hash, retained environment hashes, benchmark-result hashes, and
reviewer identity.  Any material change to a scientific/numerical claim,
tolerance, fixture, or expected result after review invalidates that verdict and
requires fresh review of the changed state.
