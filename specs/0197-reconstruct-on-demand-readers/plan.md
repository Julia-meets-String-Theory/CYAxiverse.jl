# Implementation Plan — CYAX-0197

## Governing specification

Canonical candidate specification:

\`specs/0197-reconstruct-on-demand-readers/spec.md\`

The specification is currently **draft**. No production/source implementation
is authorized until CYAX-0197 G0 passes independent review and owner approval is
recorded in the spec.

## Architectural intent

Keep all storage-schema branching inside \`src/read.jl\` (or a narrowly scoped
reader helper owned by the same module).

The implementation should converge on one internal component boundary:

~~~julia
_components = (; L, Q, Kinv)
~~~

with two inputs:

~~~text
legacy dense HDF5
    -> validate/read dense Q/L/Kinv

schema-1.1 compact HDF5
    -> validate schema/metadata
    -> reconstruct tau, V, Kinv
    -> reconstruct Q
    -> reconstruct L
    -> verify counts/hashes/witnesses
~~~

After this point existing public readers retain their current semantics:

~~~text
read.potential
    -> _kinetic_matrix
    -> _validate_kinetic_matrix
    -> AxionPotential

potential_factored
    -> symmetric Kinv
    -> Cholesky
    -> (; L,Q,Kinv,C)
~~~

\`oriented_potential\` should require no storage-specific change.

## Coverage mapping

| Requirement / Gate | Planned implementation | Planned verification |
| --- | --- | --- |
| R-001 | Add one shared component reader/reconstructor | Dense/compact unit tests; public-reader tests |
| R-002 | Strict schema/metadata parser and validator | malformed/partial/conflicting metadata fixtures |
| R-003 | Native COO reconstruction with distinct permutations | independent iii/iij/ijk analytic fixtures |
| R-004 | Exact \`4*(tau*tau' - V*kappa(t))\` reconstruction + symmetrization | analytic matrices; normalization/sign/parentheses mutation tests |
| R-005 | Strict integral unique effective-cone validation and transpose | malformed numeric fixtures + exact dense-oracle equality |
| R-006 | Deterministic lexicographic pair-index generation | exact pair-index/Q fixtures |
| R-007 | Direct/pair L reconstruction and signed-log encoding | independent coefficient oracles + zero-amplitude rejection |
| R-008 | Source counts, canonical JSON hashes, persisted-volume replay | frozen cross-language hash vectors + corrupted witness tests |
| R-009 | Preserve legacy dense path | baseline regression tests on legacy fixtures |
| R-010 | Compare all public potential readers on matched artifacts | dense/compact reader equivalence |
| R-011 | Existing vacuum-only pipeline through \`:auto\` | exact classification/count comparison |
| R-012 | One direct \`AxionPotential\` consumer | matched-output comparison without \`read.geometry\` |
| R-013 | Read-only compact handling | before/after HDF5 digest / no-write assertion |
| R-014 | Fail closed on malformed state | explicit failure matrix |
| R-015 | No Python/runtime-transitive JSON dependency | package import check + dependency inspection |
| R-016 | Exact evidence/review | candidate commit/tree + independent review |
| G0 | Approve exact S2 package | Spec + Scientific/Numerical review + owner approval |
| G1 | Component reconstruction correctness | focused tests + malformed-state suite |
| G2 | Matched real-oracle equivalence | h11=4,10,50,+frozen higher-dimensional fixture |
| G3 | Downstream equivalence | vacua-only + one direct potential consumer |
| G4 | Exact-candidate acceptance | full verification + fresh independent review |

## Proposed implementation slices

### Slice A — schema/metadata boundary

Add a strict dispatcher that examines schema markers before attempting any
scientific read.

The implementation must encode the full historical dense inventory from
F-003A rather than treating "marker-free" as synonymous with legacy:

~~~text
marker-free
cyaxiverse-ks-cy3-v2
cyaxiverse-ks-cy3-v3
cyaxiverse-ks-cy3-v4
cyaxiverse-ks-cy3-v5
cyaxiverse-ks-cy3-v8
cyaxiverse-ks-cy3-v8-qed-assignment
transitional dense cyaxiverse-ks-cy3-v9-schema-1.1
~~~

The same v9 top-level marker is also used by the compact schema. Therefore the
potential sub-schema markers are part of dispatch identity:

- v9 + exact \`reconstruct_on_demand\` + exact reconstruction schema => compact;
- v9 + complete dense \`Q/L/Kinv\` + both compact potential markers absent =>
  transitional legacy dense;
- partial/unknown/hybrid forms => fail closed.

For all dense classes the reader preserves stored \`Q/L\` orientation. It does
not canonicalize row-oriented v2-v5 data at the component boundary.

For compact artifacts:

1. parse \`reconstruction_metadata_json\` inside Julia;
2. require the exact literal source-dataset list from F-003C;
3. compare JSON normative fields to duplicated HDF5 attributes;
4. if construction metadata duplicates a normative field, require agreement;
5. verify every declared source path exists;
6. enforce the exact basis/intersection/\`kappa\` convention strings;
7. enforce \`GeometryIndex.h11 == persisted h11 == tip/effective-cone/metric/Q\`
   dimensions;
8. validate compatible \`glsm\`, \`basis_matrix\`, and
   \`prime_toric_divisors\` dimensions.

A direct pure-Julia JSON dependency may be added if needed. Prefer an established
small parser rather than a bespoke parser. If added, it must be a direct
\`Project.toml\` dependency with compatibility metadata.

### Slice B — compact geometric reconstruction

Implement a private helper that accepts only already-validated arrays and
returns:

~~~julia
(; tau, volume, Kinv)
~~~

Algorithm:

1. validate \`tip\`;
2. validate COO shape and zero-based exact-integral indices;
3. for every COO row, enumerate **distinct** index permutations;
4. accumulate \`volume\`, \`tau\`, and \`kappa_matrix\` exactly as frozen in
   the spec;
5. compute \`Kinv = 4*(tau*tau' - volume*kappa_matrix)\`;
6. symmetrize with \`0.5*(Kinv + Kinv')\`;
7. require finite quantities and positive finite volume;
8. compare \`volume\` to persisted \`CY_volume\` at the schema replay tolerance.

Keep this helper allocation-conscious but prioritize exact contract fidelity over
premature optimization.

### Slice C — charge, coefficient, and geometry-level QED reconstruction

From validated \`effective_cone\`:

1. require exact integral entries before conversion;
2. require unique rows;
3. transpose to \`Q_direct\`;
4. generate \`pair_i/pair_j\` with the frozen lexicographic order;
5. materialize \`Q_pair = Q_direct[:,pair_j] - Q_direct[:,pair_i]\`;
6. concatenate direct then pair blocks;
7. reconstruct direct/pair coefficients with the frozen equations;
8. encode row 1 as signs and row 2 as log10 magnitudes plus exponent;
9. reject zero/non-finite raw amplitudes;
10. verify source counts and canonical hashes.

Then apply the exact compact geometry-level visible-sector rule:

- null \`qed_source_index\` + no visible-sector group => direct+pair only;
- direct QED source => verify charge/source identity and append nothing;
- \`appended_prime_divisor_e3\` => append exactly one charge/coefficient column
  after the pair block using the single-instanton coefficient formula.

For an EFT assignment-pool geometry:

- require no geometry-level visible-sector selection;
- require null \`qed_source_index\`;
- return direct+pair only;
- never select an assignment or append an assignment-specific QED source.

Contradictory visible-sector/assignment-pool states fail closed.

### Slice D — public readers

Refactor only the common **input/component** portion.

\`read.potential\` keeps its current kinetic-matrix construction and validation.

\`potential_factored\` keeps its current symmetric-\`Kinv\` Cholesky behavior.

\`oriented_potential\` should continue calling \`potential\`; no compact-schema
branch belongs there.

### Slice E — evidence and downstream qualification

Freeze matched dense/compact fixtures for the same FRST and final Kähler point.

The higher-dimensional G2 fixture must be selected and recorded before its
comparison output is observed.

Then run:

- reader-equivalence checks;
- vacuum-only \`:auto\` comparison;
- one direct potential consumer that does not require \`read.geometry\`.

Do not use \`compute_axion_data\` as G3 evidence because that route remains
outside the #197 contract.

## Test/oracle design

### Synthetic analytic fixtures

At minimum:

1. **iii** — one stored \`(i,i,i)\` term;
2. **iij** — one repeated-index term requiring multiplicity 3;
3. **ijk** — one all-distinct term requiring multiplicity 6.

Each fixture independently computes expected:

- \`tau\`;
- \`V\`;
- \`kappa(t)\`;
- \`Kinv\`.

Add mutations/oracles that would fail for:

- omitted factor 4;
- \`+\mathcal V\kappa\` instead of minus;
- no symmetrization;
- historical \`0.5*Kinv + Kinv'\` parenthesis error.

### Hash fixtures

Freeze tiny values with Python-reference digests for:

- \`Q_direct\`;
- \`{"pair_i":[...],"pair_j":[...]}\`.

Julia must produce the exact lower-case SHA-256 hex.

The test must independently build the canonical JSON bytes; it must not compare
only against another Julia helper using the same implementation path.

### Matched real fixtures

Minimum cells:

~~~text
h11 = 4
h11 = 10
h11 = 50
one higher-dimensional cell frozen before comparison
~~~

For each, retain enough fixture identity to prove dense and compact data refer
to the same FRST and final Kähler point.

The raw \`read.potential\` equality oracle must be a canonical column-oriented
dense artifact. Historical row-oriented fixtures are separate compatibility
regressions and are compared after \`oriented_potential\`.

In addition freeze:

- one matched geometry-level visible-sector case whose QED charge is not a
  direct effective-cone ray, so compact reconstruction must append the same
  single QED source as the dense v8-QED predecessor;
- one compact EFT assignment-pool artifact proving geometry-level
  \`read.potential\` remains assignment-independent.

Compare:

- exact \`Q\` for the canonical column-oriented oracle;
- exact coefficient signs;
- \`tau\`;
- \`V\`;
- \`Kinv\`;
- row 2 of \`L\`;
- public-reader outputs;
- QED source index/charge/term count when geometry-level QED is present.

### Downstream fixtures

Vacua comparison must hold all run settings fixed and compare:

- count;
- classification;
- selected auto method;
- determinant/branch metadata where present.

The non-vacua consumer should use \`AxionPotential\` data directly and not
smuggle in schema-1.1 \`read.geometry\` support.

## Files expected to change during implementation

Likely:

- \`src/read.jl\`;
- focused test file(s) under \`test/\`;
- \`Project.toml\` / \`Manifest.toml\` only if a direct pure-Julia JSON parser is
  added;
- bounded test-fixture/support scripts if needed to freeze oracle data;
- SDD evidence/status updates.

Do not modify the schema-1.1 writer merely to simplify the reader.

## Verification strategy

### Contract phase

Before implementation:

- independent Spec Review;
- independent Scientific/Numerical Review;
- owner approval of exact normative revision.

### Focused implementation phase

Run the smallest tests proving:

- dispatch;
- COO mathematics;
- metric normalization;
- charge ordering/integrality;
- coefficients;
- canonical hashing;
- malformed failures;
- no mutation of compact files.

### Real-oracle phase

Run matched dense/compact fixtures and store concise comparison evidence.

### Downstream phase

Run vacuum-only and direct-consumer equivalence.

### Broader verification

Required before exact-candidate review:

~~~bash
julia --project=. bin/audit.jl
python3 scripts/agent_verify.py diff-check
git diff --check
~~~

plus the full local Julia package tests required by current repository policy.

Report commands actually run and their observed status. Unavailable checks remain
unverified.

## Migration and compatibility

No persisted-data migration is required.

Historical dense artifacts remain readable under their current representation.

Regression fixtures must cover the materially distinct legacy classes:

- marker-free row-oriented;
- marker-free column-oriented;
- v5 row-oriented;
- v8 column-oriented;
- v8-QED-assignment with appended QED;
- transitional dense v9.

The dense component path preserves raw orientation; canonicalization remains an
\`oriented_potential\` responsibility.

Schema-1.1 artifacts remain compact and immutable. Geometry-level visible-sector
QED lineage is reconstructed when explicitly persisted. EFT assignment-pool
augmentation remains a separate row-level operation and is not performed by
\`read.potential(GeometryIndex)\`.

No writer rewrite is part of #197.

## Performance considerations

Potential reconstruction is naturally O(N_direct²) because the physical
potential contains all pairwise difference terms.

The first implementation should avoid obviously unnecessary copies and repeated
pair generation, but performance optimizations must not change ordering,
precision, validation, or scientific semantics.

Any caching must remain in-memory/read-only and non-authoritative.

## Risks and stop conditions

Stop and return to the spec/owner if:

- reconstruction cannot reproduce the accepted metric normalization from
  persisted schema-1.1 data;
- implementation requires Python/CYTools at runtime;
- a writer/schema change appears necessary;
- exact hash replay requires changing the accepted stable-hash bytes;
- legacy-reader behavior must change;
- a downstream equivalence failure implies a scientific rather than reader bug;
- \`read.geometry\` support becomes necessary to satisfy a proposed acceptance
  claim;
- the higher-dimensional oracle must be replaced after results are observed.

## Review topology

G0 requires fresh independent:

1. Spec Review — requirement/scope/schema determinism;
2. Scientific/Numerical Review — reconstruction equations, normalization,
   oracle/tolerance sufficiency.

The implementation worker must not serve as the final G4 scientific reviewer.

Changed normative spec bytes after a passing G0 review require fresh G0 review.
Changed implementation/evidence bytes after a passing G4 review require fresh
exact-candidate review.
