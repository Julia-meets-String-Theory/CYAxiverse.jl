---
spec_id: CYAX-0197
title: Native Julia schema-1.1 reconstruct-on-demand potential readers
issue: 197
class: S2
status: draft
workstream: Sampling / Ensembles
parent: null
depends_on: []
created: 2026-10-01
last_reviewed: null
review_required: Spec Reviewer + Scientific/Numerical Reviewer
approval_ref: N/A while draft
version_impact: pre-1.0 minor reader capability; defer package bump to reviewed release boundary
---

# Native Julia schema-1.1 reconstruct-on-demand potential readers

## Objective

Make the existing Julia potential-reader boundary consume both:

1. legacy CYAxiverse HDF5 artifacts that persist dense \`Q\`, \`L\`, and
   \`Kinv\`; and
2. schema-1.1 artifacts that persist a compact, deterministic
   reconstruct-on-demand representation,

while presenting the same downstream contracts through
\`read.potential\`, \`potential_factored\`, and \`oriented_potential\`.

Storage-schema branching belongs entirely inside the reader/data boundary.
Vacua, spectrum kernels, and other consumers of \`AxionPotential\` must not
learn HDF5 schema details.

This is S2 work because it crosses a durable scientific schema boundary and
reconstructs the numerical normalization of a scientific potential.

## Motivation

The schema-1.1 geometry pipeline deliberately stopped persisting dense
potential arrays. It instead stores the final accepted Kähler point,
intersection data, canonical effective-cone rays, reconstruction metadata,
and integrity witnesses needed to recreate the axion potential without a live
Python/CYTools object.

The current Julia reader still assumes:

~~~text
cytools/potential/L
cytools/potential/Q
cytools/geometric/Kinv
~~~

This prevents a schema-1.1 artifact from flowing directly into the
vacuum-only qualification pipeline even though the accepted producer already
contains a deterministic reconstruction path.

The correct repair is a general reader capability, not a vacua-specific
adapter and not a change to the scientific solvers.

## Current baseline

The specification is bound to the following audit baseline:

- \`vmm\`:
  \`a5eacd8cd4a5905161cab239a461ba252c64e8e0\`
- tree:
  \`d419416514a44d943ddd20290885ac8ba4090999\`
- \`src/read.jl\` blob:
  \`9c54eb4f42da69c716a7f358800084e106f53d52\`
- schema-1.1 writer
  \`scripts/generate_geometric_data_multitriangulation.py\` blob:
  \`52774fc2cab228af29b82a3a0b96797c1500bed5\`
- schema helper \`scripts/glimmers_schema11.py\` blob:
  \`49f4bbd206c08c73095ab3bcb9ceddb198430ba2\`
- PR #195 is already included in the baseline.

### Source / schema facts

The accepted schema-1.1 writer identifies current compact artifacts with:

~~~text
file schema_version:
  cyaxiverse-ks-cy3-v9-schema-1.1

cytools/potential/storage_schema:
  reconstruct_on_demand

cytools/potential/schema_version:
  cyaxiverse-potential-reconstruction-1.0
~~~

The potential group also records:

- orientation;
- difference convention;
- pair ordering;
- \`reconstruction_metadata_json\`.

That reconstruction metadata contains at least:

- direct source count;
- pair source count;
- \`q_direct_sha256\`;
- \`pair_source_index_sha256\`;
- replay \`rtol = 1e-10\`;
- replay \`atol = 1e-10\`;
- the source-dataset list.

The file also persists \`cytools/geometric/CY_volume\`, which is an
independent numerical witness for the reconstruction from \`kappa\` and
\`tip\`.

### Current implementation facts

The current \`read.potential\` and \`potential_factored\` paths read dense
datasets directly.

The accepted generator lineage contains several historically valid dense
representations. The known dense classes on the bound history are:

- marker-free dense artifacts;
- \`cyaxiverse-ks-cy3-v2\`;
- \`cyaxiverse-ks-cy3-v3\`;
- \`cyaxiverse-ks-cy3-v4\`;
- \`cyaxiverse-ks-cy3-v5\`;
- \`cyaxiverse-ks-cy3-v8\`;
- \`cyaxiverse-ks-cy3-v8-qed-assignment\`;
- a transitional dense form carrying the top-level
  \`cyaxiverse-ks-cy3-v9-schema-1.1\` marker but no reconstruct-on-demand
  potential markers.

The v2-v5 generator family writes row-oriented potential arrays
(\`Q: N x h11\`, \`L: N x 2\`). The v8 family writes the now-canonical
column-oriented arrays (\`Q: h11 x N\`, \`L: 2 x N\`). Marker-free artifacts
exist in both historical row-oriented and later compatibility-generator
column-oriented forms. The transitional dense v9 form is column-oriented.

The v8 QED-assignment generator may append one geometry-level QED instanton
column after the direct and pair blocks when the selected QED divisor charge is
not already a direct effective-cone charge.

The accepted Python schema-1.1 reconstruction already operates without a live
CYTools object once the compact reference data have been read. Therefore this
work is a cross-language deterministic reconstruction of an existing accepted
contract, not a new physical model.

## Scientific claim boundary

### This specification may establish

- deterministic native-Julia reconstruction of the schema-1.1 potential
  representation;
- numerical equivalence of reconstructed geometry/potential quantities to
  the pinned accepted reference implementation and matched dense oracle
  fixtures;
- transparent use of compact and legacy artifacts through the same potential
  reader APIs;
- direct use of schema-1.1 artifacts by the existing vacuum-only pipeline.

### This specification must not establish

- a new Kähler-point prescription;
- a new Eq. 21 control criterion;
- a new sampling/population interpretation;
- a new axion potential normalization;
- a new vacua/minimizer algorithm;
- correctness of arbitrary schema-1.1 geometry readers;
- full \`compute_axion_data\` compatibility;
- general schema-1.1 \`read.geometry\` support;
- a physical Standard Model, orientifold, or EFT claim.

A consumer that needs only the returned \`AxionPotential\` may be used as a
downstream equivalence test. The full geometry/spectrum pipeline remains a
separate follow-on because current \`read.geometry\` expects dense geometric
datasets that schema 1.1 does not persist.

## Scope

This specification includes:

- one shared internal \`L,Q,Kinv\` reader/reconstruction boundary that
  preserves historical dense raw orientation;
- deterministic schema dispatch;
- strict schema-1.1 metadata validation;
- native-Julia reconstruction of \`tau\`, \`V\`, \`Kinv\`, \`Q\`, and \`L\`;
- exact reconstruction-integrity checks;
- transparent support in \`read.potential\`;
- transparent support in \`potential_factored\`;
- unchanged operation of \`oriented_potential\`;
- analytic/synthetic tests;
- matched dense/compact oracle tests;
- vacuum-only downstream equivalence;
- at least one non-vacua \`AxionPotential\` consumer equivalence check.

## Non-scope

This specification does not authorize:

- changing \`read.geometry\`;
- making \`compute_axion_data\` schema-1.1 compatible;
- re-running Eq. 21 or searching for a new radial scale while reading;
- selecting or repairing a Kähler point while reading;
- changing the schema-1.1 writer;
- persisting reconstructed dense arrays back into schema-1.1 HDF5;
- changing legacy dense-generator semantics;
- changing \`:auto\` vacua search ordering;
- changing solver thresholds/tolerances;
- changing Stage-1 FRST sampling;
- changing Stage-2 Kähler selection;
- changing spectrum physics;
- unrelated reader refactors.

## Fixed conventions and invariants

### F-001 — Reader reconstruction is pure/read-only

The schema-1.1 reader reconstructs scientific arrays transiently from the
persisted final artifact. It performs no geometry optimization, Eq. 21 search,
Kähler-point repair, resampling, or HDF5 mutation.

PR #195 is regression/provenance context for the accepted conventions. The
reader does not reproduce the PR #195 control-search procedure.

### F-002 — No live Python/CYTools dependency

Core Julia reading and \`using CYAxiverse\` remain independent of Python,
PyCall, and CYTools.

The reconstruction metadata JSON must be parsed in-process in Julia. The
implementation may add one explicit pure-Julia JSON dependency if needed; it
must not rely on an undeclared transitive dependency or invoke Python.

### F-003 — Exact schema-1.1 identity

The only compact schema accepted by this specification is:

~~~text
file schema_version = cyaxiverse-ks-cy3-v9-schema-1.1
potential storage_schema = reconstruct_on_demand
potential schema_version = cyaxiverse-potential-reconstruction-1.0
~~~

The accepted potential attributes are:

~~~text
orientation =
  h11 x N_instanton; charge vectors are columns

difference_convention =
  q_pair[:, k] = q_direct[:, pair_j[k]] - q_direct[:, pair_i[k]]

pair_ordering =
  lexicographic_i_then_j_with_i_less_than_j
~~~

Unknown or contradictory schema markers fail closed.

### F-003A — Supported historical dense schemas

The reader SHALL preserve current behavior for the following known dense
top-level schema classes:

~~~text
no schema_version attribute
cyaxiverse-ks-cy3-v2
cyaxiverse-ks-cy3-v3
cyaxiverse-ks-cy3-v4
cyaxiverse-ks-cy3-v5
cyaxiverse-ks-cy3-v8
cyaxiverse-ks-cy3-v8-qed-assignment
cyaxiverse-ks-cy3-v9-schema-1.1   [historical dense_opt_in form]
~~~

A dense artifact is complete only when it contains all three historical
datasets:

~~~text
cytools/potential/Q
cytools/potential/L
cytools/geometric/Kinv
~~~

For the historical dense v9 class, the exact accepted identity is:

~~~text
file schema_version = cyaxiverse-ks-cy3-v9-schema-1.1
cytools/potential/storage_schema = dense_opt_in
dense Q/L/Kinv present
no cytools/potential schema_version identifying
  cyaxiverse-potential-reconstruction-1.0
~~~

The same historical writer also emitted:

~~~text
cytools/potential/storage_schema = factorized_canonical
~~~

when dense materialization was not requested. That older factorized format is
**not** the current \`reconstruct_on_demand\` schema governed by CYAX-0197 and
is not supported by this specification. It SHALL fail closed rather than being
silently interpreted as either legacy dense or current compact reconstruction.

The absence of a top-level schema marker is itself a recognized historical
legacy class when the dense datasets are complete. An unknown nonempty
top-level schema marker is not legacy and fails closed.

### F-003B — Preserve legacy raw orientation

The common component boundary SHALL NOT normalize the stored orientation of
historical dense artifacts.

For a dense legacy artifact, \`read.potential\` and \`potential_factored\`
continue returning \`Q\` and \`L\` in the exact raw HDF5 orientation they
currently expose.

Known generator layouts are:

~~~text
marker-free: may be row- or column-oriented; preserve stored orientation
v2-v5:       Q = N x h11, L = N x 2
v8 family:   Q = h11 x N, L = 2 x N
dense v9:    Q = h11 x N, L = 2 x N
~~~

\`oriented_potential\` remains the normalization boundary that converts either
historical raw layout to canonical \`Q: h11 x N\`, \`L: 2 x N\`.

Consequently, raw \`read.potential\` equality between a compact artifact and an
arbitrary row-oriented historical artifact is not a requirement. Matched
raw-reader equality in G2 must use a column-oriented dense oracle. Historical
row-oriented fixtures are instead required to preserve their exact existing raw
outputs and to agree after \`oriented_potential\`.

### F-003C — Literal schema-1.1 source and convention metadata

For the supported compact reconstruction schema, the accepted
\`source_datasets\` list is exactly, in this order:

~~~text
cytools/geometric/kappa
cytools/geometric/glsm
cytools/geometric/basis_matrix
cytools/geometric/prime_toric_divisors
cytools/geometric/effective_cone
cytools/geometric/tip
~~~

The accepted literal geometric conventions are:

~~~text
basis_convention =
  CYTools divisor_basis(include_origin=True); all numerical vectors in basis

intersection_convention =
  CYTools CalabiYau.intersection_numbers(in_basis=True, format='coo')

kappa_format = coo
kappa_index_base = 0
~~~

All declared source datasets must exist even when a particular array is used
only as an integrity/convention witness by the geometry-level potential reader.

For consumed identity arrays, the following dimensional consistency is
required:

~~~text
GeometryIndex.h11
= cytools/geometric/h11
= length(cytools/geometric/tip)
= size(cytools/geometric/effective_cone, 2)
= size(reconstructed Kinv, 1)
= size(reconstructed Kinv, 2)
= size(Q_direct, 1)
~~~

In addition:

- \`glsm\` is two-dimensional with first dimension \`h11\`;
- \`basis_matrix\` is two-dimensional with first dimension \`h11\`;
- \`prime_toric_divisors\` has length equal to \`size(glsm, 2)\`;
- every \`kappa\` index lies in \`0:(h11-1)\`.

Any disagreement is a corrupt-artifact condition and fails closed.

### F-004 — Sparse intersection representation

Schema-1.1 \`kappa\` is a COO array with four columns:

~~~text
i, j, k, value
~~~

Indices are zero-based and must be finite exact integers within
\`0:(h11-1)\`. Values must be finite.

Each stored row represents a symmetric tensor entry. For each stored
\`(i,j,k,value)\`, define \`P(i,j,k)\` to be the set of its distinct
permutations.

The reconstruction SHALL use every distinct permutation exactly once. Thus
the multiplicity is:

- 1 for \`iii\`;
- 3 for \`iij\`;
- 6 for \`ijk\` with all indices distinct.

For \`t = tip\`, reconstruct:

~~~math
\mathcal V
=
\sum_{(i,j,k,v)}
\frac{|P(i,j,k)|}{6}
v\,t_i t_j t_k .
~~~

For every distinct permutation \`(a,b,c)\` of each COO row, accumulate:

~~~math
\kappa_{ab}(t) \mathrel{+}= v\,t_c,
~~~

and

~~~math
\tau_a \mathrel{+}=
\frac{1}{2}v\,t_b t_c .
~~~

The resulting quantities must also satisfy the conventional identities

~~~math
\tau_i=\frac{1}{2}\kappa_{ijk}t^jt^k,
\qquad
\mathcal V=\frac{1}{6}\kappa_{ijk}t^it^jt^k .
~~~

### F-005 — Exact inverse-Kähler normalization

The schema-1.1 inverse metric is reconstructed exactly as:

~~~math
K^{-1}
=
4\left(
\tau\tau^T-\mathcal V\,\kappa(t)
\right).
~~~

After construction it is symmetrized as:

~~~math
K^{-1}
\leftarrow
\frac{1}{2}
\left(
K^{-1}+(K^{-1})^T
\right).
~~~

No alternative normalization may be selected by the implementer.

The component helper must reject non-finite reconstructed values and
non-positive/non-finite \`V\`. Existing public-reader post-processing remains
authoritative for the physical kinetic-matrix/factorization checks.

### F-006 — Effective-cone charge contract

The persisted \`cytools/geometric/effective_cone\` dataset has shape:

~~~text
N_direct x h11
~~~

and is the canonical set of unique effective-cone rays.

Before integer conversion the reader SHALL require:

- a nonempty two-dimensional array;
- finite entries;
- every entry exactly equal to its nearest integer;
- every integer representable by the selected Julia integer type;
- exactly \`h11\` columns;
- unique rows.

Silent truncation or tolerance-based acceptance is forbidden at the persisted
reader boundary.

The geometry attributes SHALL be internally consistent:

~~~text
potential_charge_convention = unique_effective_cone_rays

canonical_effective_cone_ray_count = N_direct
raw_effective_cone_ray_count
  = canonical_effective_cone_ray_count
    + duplicate_effective_cone_rows_removed
~~~

The direct charge matrix is:

~~~math
Q_{\rm direct}
=
(\text{effective_cone})^T
~~~

with shape \`h11 x N_direct\`.

### F-007 — Pair-source contract

Let \`N=N_direct\`. Construct zero-based source-index arrays in the exact
Python/\`itertools.combinations(range(N),2)\` order:

~~~text
(0,1), (0,2), ..., (0,N-1),
(1,2), ..., (N-2,N-1)
~~~

Thus:

~~~math
N_{\rm pair}=\frac{N(N-1)}{2}.
~~~

For each source pair \`i<j\`:

~~~math
q_{ij}=q_j-q_i .
~~~

The final matrix is direct columns followed by pair columns:

~~~math
Q=[Q_{\rm direct}\;Q_{\rm pair}] .
~~~

### F-008 — Potential coefficient contract

Define:

~~~math
P=\frac{8\pi}{\mathcal V^2},
\qquad
d_i=q_i\cdot\tau .
~~~

For each direct source:

~~~math
A_i=P\,d_i,
\qquad
e_i=-2\pi\log_{10}(e)\,d_i .
~~~

For each pair \`i<j\`:

~~~math
A_{ij}
=
P\left[
\pi\,q_i^T K^{-1}q_j+d_i+d_j
\right],
~~~

~~~math
e_{ij}
=
-2\pi\log_{10}(e)(d_i+d_j).
~~~

Encode each coefficient \`a\` as:

~~~math
L_{1a}=\operatorname{sign}(A_a),
\qquad
L_{2a}=\log_{10}|A_a|+e_a .
~~~

Row 1 is specifically the sign, not an arbitrary mantissa. Every raw
amplitude and exponent must be finite, and a zero raw amplitude is an error
before \`log10\`.

Direct columns precede pair columns in exactly the \`Q\` ordering above.

### F-008A — Geometry-level visible-sector/QED extension

Schema-1.1 preserves the dense v8-QED scientific lineage for a selected
geometry-level visible-sector assignment.

The reconstruction metadata field \`qed_source_index\` has the following exact
meaning:

1. **No geometry-level visible-sector QED term**
   - \`qed_source_index == null\`;
   - \`cytools/geometric/visible_sector\` is absent;
   - the geometry potential is exactly the direct+pair potential from F-007 and
     F-008.

2. **Selected QED charge already appears among direct effective-cone charges**
   - \`cytools/geometric/visible_sector\` is present;
   - its \`qed_charge\` is an exact integral vector of length \`h11\`;
   - \`qed_potential_source == "direct_effective_cone"\`;
   - \`qed_instanton_index == qed_source_index\`;
   - \`0 <= qed_source_index < N_direct\`;
   - \`qed_charge == Q_direct[:, qed_source_index]\`;
   - no extra potential column is appended.

3. **Selected QED charge is not a direct effective-cone charge**
   - \`cytools/geometric/visible_sector\` is present;
   - its \`qed_charge\` is an exact integral vector of length \`h11\`;
   - \`qed_potential_source == "appended_prime_divisor_e3"\`;
   - \`qed_instanton_index == qed_source_index\`;
   - \`qed_source_index == N_direct + N_pair\` using zero-based source indexing;
   - the QED charge must not equal any direct charge;
   - append exactly one QED column after all direct and pair columns.

For the appended case, define

~~~math
d_{\rm QED}=q_{\rm QED}\cdot\tau ,
~~~

and reconstruct its coefficient with the same single-instanton formula:

~~~math
A_{\rm QED}=P\,d_{\rm QED},
\qquad
e_{\rm QED}=-2\pi\log_{10}(e)\,d_{\rm QED}.
~~~

Encode it with the same signed/log10 rule from F-008 and append it to \`L\`
after the pair block.

Where persisted, \`qed_charge_exact_match\` must be true. Inconsistent,
partial, out-of-range, or contradictory QED metadata fails closed.

This geometry-level extension is part of \`read.potential(GeometryIndex)\`
because it is the scientific term that the dense v8-QED predecessor persisted
for that selected geometry-level assignment.

### F-008B — EFT assignment-pool semantics

An EFT assignment pool is a different layer from the geometry-level
\`read.potential(GeometryIndex)\` contract.

For an accepted schema-1.1 artifact carrying
\`cytools/geometric/assignment_pool\`:

- \`cytools/geometric/visible_sector\` must be absent;
- reconstruction metadata \`qed_source_index\` must be null;
- \`read.potential(geom_idx)\` returns the assignment-independent
  direct+pair geometry potential only;
- no assignment is selected implicitly;
- no assignment-specific QED column is appended by the geometry reader.

Assignment-specific QED augmentation remains an EFT-row reconstruction
operation and requires an explicit assignment identity outside the scope of
CYAX-0197.

A compact artifact carrying both a geometry-level visible-sector selection and
an assignment pool, or an assignment pool with non-null geometry-level
\`qed_source_index\`, is contradictory and fails closed.

### F-009 — Numerical replay tolerance

Schema-1.1 potential reconstruction freezes:

~~~text
replay_rtol = 1e-10
replay_atol = 1e-10
~~~

These are schema values, not implementation-tuned tolerances.

The independently reconstructed \`V\` must be compared to persisted
\`cytools/geometric/CY_volume\` with these tolerances.

Matched dense/reference-oracle numerical comparisons use the same replay
tolerances unless a test has a stricter independently justified exact
criterion.

### F-010 — Canonical hash serialization

The schema hash algorithm is the accepted Python \`stable_hash\` contract:

1. convert arrays/tuples to JSON arrays and dictionary keys to strings;
2. require finite JSON values;
3. serialize JSON with keys sorted lexicographically;
4. use separators \`,\` and \`:\` with no additional whitespace;
5. encode the JSON bytes as UTF-8;
6. SHA-256 those exact bytes;
7. compare lower-case hexadecimal digests.

Normative witnesses:

\`q_direct_sha256\` is the hash of the nested JSON array representation of
\`Q_direct\` in \`h11 x N_direct\` orientation.

\`pair_source_index_sha256\` is the hash of:

~~~json
{"pair_i":[...],"pair_j":[...]}
~~~

with the zero-based arrays defined by F-007.

Cross-language tests must compare Julia-produced digests against frozen
Python-reference digests.

### F-011 — Public-reader post-processing remains distinct

The new common boundary supplies \`L,Q,Kinv\`.

After that boundary:

- \`read.potential\` retains its existing
  \`_kinetic_matrix\` plus \`_validate_kinetic_matrix\` semantics and returns
  \`AxionPotential(L,Q,K)\`;
- \`potential_factored\` retains its existing symmetric-\`Kinv\` Cholesky
  semantics and returns \`(; L,Q,Kinv,C)\`.

This task does not unify or alter those two existing post-processing/failure
contracts.

## Deterministic schema dispatch matrix

The reader SHALL classify an artifact before reading scientific arrays.

| File/potential state | Required behavior |
| --- | --- |
| Exact current schema-1.1 file marker + \`storage_schema=reconstruct_on_demand\` + exact reconstruction schema + complete consistent metadata + no forbidden dense arrays | reconstruct current compact representation |
| Marker-free artifact + complete dense \`Q/L/Kinv\` + no potential schema/storage markers | use legacy dense path unchanged, preserving stored orientation |
| Exact legacy marker v2/v3/v4/v5/v8/v8-QED-assignment + complete dense \`Q/L/Kinv\` + no potential schema/storage markers | use legacy dense path unchanged, preserving stored orientation |
| Top-level \`cyaxiverse-ks-cy3-v9-schema-1.1\` + \`storage_schema=dense_opt_in\` + complete dense \`Q/L/Kinv\` + no current reconstruct-on-demand reconstruction schema | use historical dense-v9 legacy path unchanged, preserving stored orientation |
| Top-level \`cyaxiverse-ks-cy3-v9-schema-1.1\` + \`storage_schema=factorized_canonical\` | unsupported historical compact format; fail closed |
| Exact schema-1.1 file marker + unknown/partial potential schema or storage marker | fail closed |
| Exact current compact schema markers + unexpected dense \`Q\` or \`L\` | fail closed as contradictory hybrid |
| Exact current compact schema markers + unexpected persisted dense \`Kinv\` or \`divisor_volumes\` | fail closed as contradictory hybrid |
| Reconstruct-on-demand marker on an unrecognized file schema | fail closed |
| Complete dense datasets plus an unknown top-level schema marker | fail closed |
| Complete dense datasets plus any unknown potential storage/schema marker | fail closed |
| Partial reconstruction metadata or missing declared source dataset | fail closed |
| Conflicting normative values between potential attributes and \`reconstruction_metadata_json\` | fail closed |
| If duplicate construction metadata is present, conflicting overlapping normative reconstruction fields | fail closed |
| Compact visible-sector / QED state violating F-008A | fail closed |
| Compact EFT assignment-pool state violating F-008B | fail closed |
| Neither complete supported legacy dense nor exact supported current compact schema | fail closed |

The top-level v9 marker alone is therefore not sufficient to classify an
artifact. The potential storage marker is part of the identity:

~~~text
dense_opt_in          -> historical dense-v9 compatibility path
factorized_canonical  -> unsupported historical compact path / fail closed
reconstruct_on_demand -> current compact reconstruction path, only with the
                         exact current reconstruction schema
~~~

A malformed compact artifact must never be relabeled as legacy merely because
some dense datasets happen to exist.

## Requirements

### R-001 — One shared component boundary

The reader SHALL provide one internal storage-dispatch/reconstruction path that
returns:

~~~julia
(; L, Q, Kinv)
~~~

for both supported storage representations.

For compact schema-1.1 inputs, \`Q\` and \`L\` use the schema's canonical
column orientation. For historical dense inputs, \`Q\` and \`L\` remain in
their exact stored raw orientation according to F-003B.

\`read.potential\` and \`potential_factored\` SHALL share this input boundary.
No vacua-specific reconstruction reader is permitted.

### R-002 — Strict metadata/schema validation

Before compact reconstruction, the reader SHALL validate:

- exact F-003 schema identities;
- exact F-003C source-dataset list and literal basis/intersection conventions;
- F-003C \`h11\` and dimensional identities;
- potential orientation;
- difference convention;
- pair ordering;
- \`kappa_format == "coo"\`;
- \`kappa_index_base == 0\`;
- source counts;
- F-009 replay tolerances;
- F-010 integrity hashes;
- F-006 raw/canonical/duplicate count consistency;
- F-008A geometry-level QED state when \`qed_source_index\` is non-null;
- F-008B assignment-pool state when present.

The implementation SHALL parse metadata within Julia and fail closed when a
required field is absent or malformed.

Before taking a dense legacy path, it SHALL validate that the top-level marker
belongs to F-003A (or is absent), the three dense datasets are complete, and
the marker-specific storage contract is satisfied. In particular,
historical dense v9 requires \`storage_schema=dense_opt_in\`; v2-v8/marker-free
dense classes require the historical absence of compact potential markers.
Any marker-specific historical layout fixture must remain compatible with the
stored arrays.

### R-003 — Exact sparse-\`kappa\` reconstruction

The reader SHALL implement F-004 exactly.

Focused tests SHALL contain independent \`iii\`, \`iij\`, and \`ijk\`
fixtures so that 1/3/6 multiplicity errors are separately detectable.

### R-004 — Exact metric reconstruction

The reader SHALL implement F-005 exactly.

Tests must detect at least:

- missing factor 4;
- sign reversal of the \(\mathcal V\kappa\) term;
- missing final symmetrization;
- the historical PR #195 parenthesis/symmetrization mistake.

### R-005 — Strict direct-charge reconstruction

The reader SHALL implement F-006 exactly and reject malformed charge data
before integer conversion.

No tolerance-based rounding is permitted at the persisted schema-1.1 reader
boundary.

### R-006 — Exact pair reconstruction

The reader SHALL implement F-007 exactly.

The reconstructed \`Q\` must be integer-identical to the matched dense oracle,
including column ordering.

### R-007 — Exact \`L\` reconstruction

The reader SHALL implement F-008 exactly.

Tests SHALL separately exercise direct and mixed terms, positive and negative
raw amplitudes where physically/mathematically valid for the fixture, and the
zero-amplitude failure path.

### R-008 — Reconstruction integrity

The reader SHALL recompute and require exact equality of:

- \`q_direct_sha256\`;
- \`pair_source_index_sha256\`.

It SHALL require exact agreement of recorded source counts and conventions.

It SHALL independently reconstruct \(\mathcal V\) and require agreement with
persisted \`cytools/geometric/CY_volume\` under F-009.

When a geometry-level visible-sector QED state is present, it SHALL additionally
enforce F-008A and include the persisted selected QED term exactly once. When
an EFT assignment pool is present, it SHALL enforce F-008B and SHALL NOT append
an assignment-specific term.

A hash/count/tolerance/QED-state mismatch is a terminal read error, not a
warning.

### R-009 — Legacy behavior preservation

For every supported dense class in F-003A, existing \`read.potential\`,
\`potential_factored\`, and \`oriented_potential\` scientific outputs and
failure semantics SHALL remain unchanged.

The common component helper may reorganize code, but it must not reinterpret or
reorient historical dense artifacts.

Regression coverage SHALL include every materially distinct historical layout
class, including:

- marker-free row-oriented dense;
- marker-free column-oriented dense;
- v5 row-oriented dense;
- v8 column-oriented dense;
- v8-QED-assignment with an appended QED column;
- historical dense v9 with the schema-1.1 top-level marker,
  \`storage_schema=dense_opt_in\`, and complete dense \`Q/L/Kinv\`;
- historical v9 \`storage_schema=factorized_canonical\` as an explicit
  fail-closed regression.

### R-010 — Public potential-reader equivalence

For matched dense and compact representations of the same FRST at the same
final Kähler point:

- when the dense oracle is canonical column-oriented, \`read.potential\` raw
  outputs SHALL agree under the approved numerical tolerances;
- when the dense oracle is canonical column-oriented, \`potential_factored\`
  raw \`Q/L\` outputs SHALL agree under the approved numerical tolerances;
- \`oriented_potential\` SHALL require no schema-specific logic and SHALL agree
  for both row- and column-oriented historical dense layouts;
- at least one matched geometry-level visible-sector fixture SHALL exercise an
  appended QED source and prove compact reconstruction matches the corresponding
  dense v8-QED scientific potential;
- at least one assignment-pool fixture SHALL prove
  \`read.potential(GeometryIndex)\` returns only the assignment-independent
  geometry potential and does not select or append an EFT-row QED source.

### R-011 — Vacuum-only downstream equivalence

Using the same vacua configuration and matched dense/compact fixture, the
existing vacuum-only path:

~~~text
compute_vacua_data
  -> _vacua_core
  -> read.potential
~~~

SHALL produce identical:

- vacuum count;
- \`search_classification\`;
- \`auto_selected_method\`;
- determinant/branch metadata where applicable.

This requirement authorizes no vacua algorithm change.

### R-012 — Non-vacua \`AxionPotential\` consumer equivalence

At least one downstream consumer that takes the returned potential components
directly SHALL demonstrate equivalent results for matched dense/compact reader
outputs.

This evidence must not depend on \`read.geometry\` and must not imply that the
full \`compute_axion_data\` path is schema-1.1 compatible.

### R-013 — Read-only persistence boundary

Compact reconstruction SHALL NOT write dense \`Q\`, \`L\`, \`Kinv\`, \`K\`,
or any cache back into the source HDF5 artifact.

No derived cache becomes authoritative under this specification.

### R-014 — Fail-closed malformed-data behavior

Focused tests SHALL cover at least:

- unknown file schema;
- unknown potential reconstruction schema;
- unknown storage marker;
- contradictory compact+dense hybrid;
- supported dense-v9 \`storage_schema=dense_opt_in\` dispatch;
- unsupported historical v9 \`storage_schema=factorized_canonical\` failure;
- missing reconstruction metadata;
- conflicting duplicated metadata;
- missing or reordered literal source-dataset metadata;
- basis/intersection convention mismatch;
- \`GeometryIndex.h11\` versus persisted-\`h11\` mismatch;
- tip/effective-cone/reconstructed-metric dimension mismatch;
- malformed COO shape;
- nonintegral/out-of-range COO index;
- malformed/duplicate/nonintegral/out-of-range direct charge;
- source-count mismatch;
- hash mismatch;
- persisted-volume replay mismatch;
- non-finite reconstructed quantities;
- zero raw potential amplitude;
- visible-sector/QED source-index, source-kind, charge, or group inconsistency;
- contradictory visible-sector plus assignment-pool state;
- assignment pool with non-null geometry-level \`qed_source_index\`;
- singular/corrupt metric behavior at each existing public-reader boundary.

### R-015 — No hidden runtime dependency

The implementation SHALL keep package import Python-free.

If a new JSON library is required, it SHALL be a declared direct pure-Julia
dependency with normal package compatibility metadata. Reliance on a transitive
JSON package is not acceptable.

### R-016 — Exact evidence and review

The implementation candidate SHALL record:

- exact base commit/tree;
- exact spec approval reference;
- changed paths;
- focused commands and observed results;
- oracle fixture identities;
- matched dense/compact reconstruction evidence;
- downstream equivalence evidence;
- final commit/tree.

The exact implementation candidate SHALL receive fresh independent
Scientific/Numerical Review after implementation verification.

## Acceptance gates

### CYAX-0197 G0 — Approve the S2 reconstruction contract

**Objective:** establish the durable reader/scientific contract before source
implementation.

**Acceptance:**

- exact \`spec.md\` receives independent Spec Review;
- exact mathematical/schema reconstruction contract receives independent
  Scientific/Numerical Review;
- blocking findings are repaired and changed normative bytes are re-reviewed;
- owner explicitly approves the reviewed revision;
- \`status\` becomes \`approved\`;
- \`approval_ref\` binds the exact reviewed normative revision/content.

**Stop condition:** unresolved normalization, COO semantics, basis convention,
schema-dispatch behavior, integrity semantics, or public-reader scope.

### CYAX-0197 G1 — Analytic and schema-bound reconstruction

**Objective:** implement the canonical compact reconstruction boundary.

**Acceptance:**

- R-001 through R-008 and R-013 through R-015 pass;
- independent \`iii/iij/ijk\` tests pass;
- canonical hash replay passes;
- compact artifacts fail closed on every required malformed-state fixture.

**Stop condition:** implementation requires a new scientific convention,
writer/schema change, Python runtime, or undocumented recovery behavior.

### CYAX-0197 G2 — Matched real-oracle reader equivalence

**Objective:** prove the new reader reproduces the accepted scientific
representation.

Use matched dense/compact representations of the same FRST and final Kähler
point. Representative oracle cells SHALL include at least:

~~~text
h11 = 4
h11 = 10
h11 = 50
~~~

plus one bounded higher-dimensional case chosen before observing comparison
results.

**Acceptance:**

- R-009 and R-010 pass;
- the matched raw-reader oracle used for compact/raw equality is explicitly
  frozen as a canonical column-oriented dense oracle;
- \`Q\` is exactly equal as integers for that canonical dense oracle;
- \`tau\`, \`V\`, \`Kinv\`, and \`L\` satisfy the frozen replay tolerance;
- signs and source ordering agree exactly;
- historical row-oriented fixtures preserve their raw reader outputs and agree
  after \`oriented_potential\`;
- one matched appended-QED visible-sector fixture reproduces the dense v8-QED
  scientific potential;
- one assignment-pool fixture confirms geometry-level reading remains
  assignment-independent;
- no post-hoc tolerance tuning or fixture replacement is used.

**Stop condition:** disagreement not attributable to a defect in the candidate
reader or an independently demonstrated corrupt fixture.

### CYAX-0197 G3 — Downstream qualification

**Objective:** show the general potential reader unlocks the intended
qualification path without changing downstream science.

**Acceptance:**

- R-011 vacuum-only equivalence passes;
- R-012 non-vacua direct-consumer equivalence passes;
- no \`read.geometry\` or full \`compute_axion_data\` claim is made.

**Stop condition:** downstream equality requires changing vacua/spectrum
scientific behavior.

### CYAX-0197 G4 — Exact-candidate independent acceptance

**Objective:** independently review the exact implementation and evidence.

**Acceptance:**

- required local verification passes or unavailable checks are explicitly
  reported;
- exact candidate commit/tree and evidence receive fresh independent
  Scientific/Numerical Review with no blocking finding;
- Control Desk/owner reconciliation accepts the candidate for the next
  integration decision.

A passing G4 review does not itself authorize merge or Issue closure.

## Verification requirements

Required evidence includes:

1. synthetic \`iii\`, \`iij\`, \`ijk\` COO fixtures;
2. analytic volume/tau/\`Kinv\` oracles;
3. direct/pair \`Q\` ordering oracle;
4. direct/pair \`L\` coefficient oracle;
5. geometry-level appended-QED coefficient/source oracle;
6. assignment-pool geometry-reader non-augmentation oracle;
7. zero-amplitude and malformed-data failures;
8. canonical JSON/SHA-256 cross-language hash fixtures;
9. persisted-\`CY_volume\` replay check;
10. exact \`h11\`/dimension-identity failure fixtures;
11. literal source/basis/intersection metadata fixtures;
12. matched dense/compact real fixtures at the G2 cells;
13. public reader equivalence;
14. legacy dense dispatch/orientation regression fixtures for every material
    historical class in R-009, including an actual dense-v9
    \`storage_schema=dense_opt_in\` artifact and an explicit
    \`factorized_canonical\` fail-closed fixture;
15. vacuum-only downstream equivalence;
16. one direct \`AxionPotential\` consumer equivalence check;
17. focused Julia tests;
18. full local package tests;
19. \`julia --project=. bin/audit.jl\`;
20. \`python3 scripts/agent_verify.py diff-check\`;
21. \`git diff --check\`;
22. exact-candidate independent Scientific/Numerical Review.

Unobserved checks are not PASS.

## Interfaces and compatibility

### Public API

No public function removal is intended.

Existing reader entry points retain their signatures. Schema-1.1 support is a
new accepted input representation behind those interfaces.

### Persisted schema

No schema-1.1 writer or artifact layout change is authorized.

### Runtime

Core Julia remains Python-free. A declared pure-Julia JSON parser dependency is
permitted only if needed for the persisted reconstruction metadata.

### Package version

This is a pre-1.0 durable reader capability and may warrant a minor package
version at the reviewed release boundary. No feature-branch package bump is
authorized by this specification.

## Dependencies and blockers

- PR #195 is included in the bound baseline.
- The schema-1.1 writer/reference reconstruction is already present on the
  bound baseline.
- No \`read.geometry\` follow-on is required to complete the vacuum-only
  qualification enabled by this specification.

## Open owner decisions

None are intentionally left open in the reconstruction mathematics or schema
dispatch.

If review or implementation finds that exact reconstruction requires changing
the accepted schema, normalization, basis convention, population definition,
or public scientific behavior, stop and return to the owner for a revised and
re-reviewed specification.

## Completion criterion

CYAX-0197 is complete when:

1. G0 is approved with durable exact-review provenance;
2. the approved spec is implemented without changing its scientific boundary;
3. G1-G3 evidence passes;
4. G4 independent exact-candidate review has no blocking finding;
5. spec/plan/tasks/implementation/tests/evidence/PR scope are converged;
6. Control Desk/owner reconciliation accepts the result.

Completion of CYAX-0197 establishes the potential-reader boundary needed for
the planned vacua qualification dataset. It does not establish general
schema-1.1 geometry-reader compatibility.
