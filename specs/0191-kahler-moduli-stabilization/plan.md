# Implementation Plan — CYAX-0191

## Governing specification

Canonical specification: `specs/0191-kahler-moduli-stabilization/spec.md`

Spec revision for this plan: draft on base `74fea608684eae746f25ae45b18512f89b862fb0`.
No final owner approval exists yet. This plan is subordinate to the draft
specification and does not authorize implementation.

## Coverage

| Requirement / gate | Planned implementation | Planned verification |
| --- | --- | --- |
| CYAX-0191 Gate A (recorded entry PASS) | Carry the owner-confirmed exact Issue #191 contract into the approved SDD; do not rerun or imply new source rendering certification in this preparation. | Confirm the pinned source/body remains the governing input and that its mathematical, convention, anomaly, and benchmark requirements are represented without semantic change. |
| R-001 | Keep the first scientific deliverable within the specified retained O3/O7 EFT and research arm. | Claim-boundary review of every result/report; reject any global or omitted-sector inference. |
| R-002 | Build versioned Julia geometry and convention records from exact source geometry and basis data. | Volume/divisor identities, basis checks, domain inequalities, coordinate-map fixture. |
| R-003 | Store BBHL metadata independently of active correction switches. | B1 disabled-correction fixture checks metadata and all active evaluation layers. |
| R-004 | Implement the source model as independently testable terms using the prescribed heavy-field Schur-complement effects. | Independent one-modulus inverse-metric discriminator and limiting tests. |
| R-005 | Encode the adopted unequal-index phase in the quadratic term. | Gate B direct-complex discriminator with unequal phases. |
| R-006 | Provide distinct parent-metric and retained-kinetic-metric reports. | Positive-definiteness/domain fixtures for each metric and separate result fields. |
| R-007 | Generalize divisor terms through `Q` as an explicit CYAxiverse extension. | Identity-charge reduction, independent oracle, nontrivial charge and basis-covariance checks. |
| R-008 | Make CYTools an importer/fixture boundary producing canonical native Julia geometry. | Gate D native-vs-imported fixtures, geometry identity, tensor, basis, and cone provenance checks. |
| R-009 | Separate physics model from potential derivatives and critical-point/search algorithms; keep AD and solver adapters replaceable. | Interface-level review and pointwise model checks across supported precision/backend choices. |
| R-010 | Add structured geometry, model, run, benchmark, and result provenance. | Replay manifest completeness review and exact source/configuration identity checks. |
| R-011 | Represent critical points, retained kinetic data, covariant Hessian, generalized masses, and zero modes separately. | Synthetic mass/flat-direction fixtures and Gate E classification. |
| R-012 | Freeze scales and search policy in manifests; return structured control states. | Pre-run manifest review, scaled-residual checks, bounded-search replay and failure accounting. |
| R-013 / Gate C | Reproduce B1 with its BBHL-off active model and retained geometric BBHL metadata. | Publisher-PDF source comparison at printed precision and independent residual. |
| R-014 / Gate C | Reproduce B2 location and equation-defined value while retaining the source table anomaly. | Separate source-observation and equation-derived manifest fields; independent refinement and residual. |
| R-015 / Gate C | Reproduce publisher-PDF B3 E5 with its imported inequalities and exact flat axion. | Refined critical point, source-precision comparisons, domain and flat-direction evidence. |
| R-016 / CYAX-0191 Gate B | Run local analytic/synthetic oracles before named source data. | No-scale limits, inverse metric, derivative, unequal phase, generic `Q`, identity reduction, basis covariance, frozen scaling. |
| R-017 / CYAX-0191 Gate C | Run B1, B2, B3 in named-source order with independent residuals. | Complete manifests and source/equation comparisons; stop at first failure. |
| R-018 / CYAX-0191 Gate D | Validate the geometry import boundary independently of solver output. | Exact geometry identity, CYTools revision, cone provenance, tensor/index, duality, basis, native-fixture checks. |
| R-019 / CYAX-0191 Gate E | Classify each named retained-sector result from recorded evidence. | Stationarity, fluctuations, flat directions, BF where applicable, both metrics, controls, and claim-boundary review. |
| R-020 / SDD review and approval | Freeze one exact SDD candidate for independent Spec and Standards review; return it for owner approval. | Fresh reviewers use the capability pins in the governing review request and bind identical commit/tree and file hashes; block if a pin is unavailable; record owner approval separately before any implementation handoff. |

## Existing architecture

The package contract in `AGENTS.md` keeps CYTools/Python optional for core
Julia use, preserves numeric precision and sparse intersection structure, and
separates physical-domain checks from saddle/tachyon diagnostics. Issue #191
places the work in a research arm rather than the stable public API. Exact
production file paths and concrete APIs are not frozen by the Issue.

The production-planning comment requires a native versioned geometry artifact,
a physics/numerics boundary, replaceable AD and solver backends, generic
numeric and sparse-friendly design, replay provenance, and serialization-based
future interoperability. It allows CYTools at import and fixture-generation
time only.

## Proposed approach

### 1. Canonical geometry and convention boundary

Import CYTools output or checked-in fixtures into a versioned Julia-side
geometry representation. Include the intersection tensor, Euler characteristic,
divisor ordering and basis maps, dual curve-basis map, toric inequalities and
construction/provenance, and exact or numeric precision identity. Retain
unknown completeness as a structured status; a toric inference is not proof of
the full cone.

Keep frame, length and `alpha'` units, Planck normalization, `K_cs`, paper/Julia
complex-coordinate map, phases, periodicity/condensate branch, and active or
frozen fields in a model/convention record associated with each evaluation.
Domain membership uses imported inequalities. A cone margin cannot be
serialized or described as a physical curve volume without an integral curve
normalization, frame, and units.

### 2. Potential model separated from numerical methods

Implement the 2020 model as a validated model consumed by generic evaluation
and differentiation interfaces. Preserve separately testable BBHL,
non-perturbative linear, non-perturbative quadratic, and optional uplift
contributions. Keep uplift off by default and make correction switches
consistent across the Kähler potential, metric/inverse metric, and potential.
Keep stored geometry-derived values separate from active switch state.

Encode the declared full-metric Schur-complement prescription rather than
freezing and reinverting only the retained metric block. Use the adopted
unequal phase. Leave the generic-`Q` extension behind explicit extension
metadata and validate it before named benchmarks.

### 3. Differentiation, search, and fluctuations

Keep potential evaluation, gradient/Hessian construction, optimizer/root
search, and fluctuation analysis as replaceable interfaces. Avoid tying the
physics contract to one AD or solver package. Preserve sparse/exact structure
where practical and avoid narrowing precision to `Float64` when the model and
backend support a higher-precision representation.

Return structured results for coordinates, stationarity, metrics, geometry,
fluctuations, controls, heavy-sector assessment, and global-consistency
assessment. Store physical masses as generalized eigenvalues relative to the
retained kinetic metric. Represent exact kernel modes from active `Q`
separately from numerical near-zeros and lifted modes.

### 4. Provenance and replay

Every geometry fixture and benchmark result links to immutable source and
artifact identities. The run manifest records model and basis convention,
units, schema, code revision, runtime/backend versions, numeric precision,
search strategy, seed/budget, tolerances, and benchmark identity. Keep source
rounded data, independently refined values, and equation-defined quantities
as distinct fields. Store failure and rejection accounting for bounded search.

Future cross-package comparisons use common serialized geometry/model inputs
and compare `V`, `grad V`, and later Hessian/mass data pointwise. No runtime
dependency or call into the unreleased KahlerJAX package is introduced.

### 5. Progressive evidence order

1. Carry the Issue #191 Gate A PASS entry evidence and exact equations into the
   approved SDD; do not rerun or claim new Gate A certification here.
2. After SDD approval and a separate implementation dispatch, validate the
   analytic/synthetic local cases in CYAX-0191 Gate B.
3. Only after Gate B passes, run named B1, B2, and B3 fixtures in Gate C.
4. Validate geometry import identity and conventions in Gate D.
5. Classify accepted points and controls in Gate E.

Any failed gate stops later scientific execution. A correction that changes
scientific meaning returns to this specification for renewed independent
review and owner approval before the affected implementation continues.

## Alternatives considered

- **Live CYTools object in model evaluation:** rejected by the approved
  architecture requirement because geometry and optional Python would then
  enter numerical hot paths. Import to a versioned native artifact instead.
- **Physics embedded in a single optimizer:** rejected because it couples the
  source model to one search strategy and makes model, derivative, and solver
  evidence difficult to replace or compare.
- **One fixed AD/optimizer backend in the scientific contract:** deferred to
  implementation selection; the requirement is replaceable interfaces, not a
  prescribed package.
- **Treating generic `Q` as source-paper behavior:** rejected. It is a separate
  CYAxiverse extension with independent Gate B evidence.
- **Tuning B2 to match the printed energy:** rejected by Issue #191; preserve
  source observation and equation-defined result as distinct evidence and
  leave the anomaly unresolved.

## Data/API/schema impact

This SDD draft has no API, source, dependency, data artifact, persisted schema,
environment, or package-version change. Later Phase-0 implementation is
planned in the research arm and must keep geometry/model/run provenance
explicit. No stable public API, reader/writer schema, or package-version
adoption is authorized or implied. If a future implementation requires a
scientific schema or observable change, stop for owner direction and amend the
specification before implementation.

## Verification strategy

### SDD preparation

- Review all requirements and each CYAX-0191 Gate A-E item against Issue #191.
- Verify all three Markdown artifacts are present, well-formed, and mutually
  consistent; verify that every requirement and gate maps to a planned task
  and evidence item.
- Run repository document checks that apply to these files and
  `git diff --check` only. No scientific code, benchmark, geometry scan, or
  Gate B/C execution belongs to this preparation slice.
- Obtain fresh comprehensive Spec and Standards reviews of the same frozen
  candidate. A material normative-byte change requires renewed affected review.

### Later scientific verification

- **CYAX-0191 Gate B:** no-scale/correction limits; full-versus-frozen metric
  discriminator; independent derivative oracle; unequal-phase direct-complex
  check; generic-`Q` oracle; identity reduction; nontrivial basis covariance;
  numerical scale/tolerance freeze. A convention-matched Schachner comparison
  is additional only.
- **CYAX-0191 Gate C:** manifest and named reproduction for B1, B2, and B3;
  compare rounded source observations at their precision; independently refine
  the equation-defined point and residual; preserve the B2 energy anomaly.
- **CYAX-0191 Gate D:** imported and native geometry identity; CYTools revision;
  cone provenance; tensor index/permutation, divisor/curve duality, basis, and
  fixture checks.
- **CYAX-0191 Gate E:** scaled stationarity; retained fluctuation and
  generalized mass results; exact/near-zero/lifted mode classification; BF
  where applicable; separate metric and domain reports; structured controls;
  explicit global and heavy-sector status.

Do not run Gate C before Gate B passes. Do not treat a benchmark count or
rounded equality as proof that populations or physical claims match.

## Migration / compatibility

No migration or compatibility effect is part of this documentation change.
Future code remains isolated from stable APIs, preserves optional Python and
precision behavior, and must define compatibility for any later geometry or
result schema before persistence is adopted. Potential observable, unit,
normalization, basis, scientific acceptance, and schema changes require owner
direction and an SDD amendment/review.

## Risk and stop conditions

- Stop for owner direction if any source normalization, phase, basis, heavy
  reduction, domain, acceptance, mass interpretation, or schema is ambiguous.
- Stop if an issue/source identity changes materially or Issue #191 no longer
  contains the approved contract bound to this SDD.
- Stop later gates after an earlier gate failure. Never tune a parameter,
  normalization, convention, threshold, or source selection to force a match.
- Stop if CYTools geometry cannot be identified or if tensor/basis/domain
  facts conflict; do not infer missing geometry.
- Stop before consequential scientific implementation unless exact SDD review
  passes and the owner has separately approved the frozen revision.
- Stop on any need to edit a public path outside the three authorized SDD
  files in the present preparation slice.
- The missing final owner SDD approval is an intentional current boundary; it
  is not inferred from Issue approval, Gate A, Manager review, or reviewer
  verdicts.

## Task decomposition

`tasks.md` separates the current documentation/review/owner-approval work from
the future scientific programme. The first evidence tranche is exact SDD
candidate review. Scientific implementation tasks remain planned and gated by
both final SDD approval and a separate implementation dispatch. Gate B
precedes Gate C; Gate D confirms imported geometry; Gate E classifies the
accepted named results.
