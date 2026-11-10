---
spec_id: CYAX-0232
title: Adopt the supergravity axion kinetic normalization
issue: 232
class: S2
status: draft
workstream: axion spectrum physical conventions
parent: null
depends_on: []
created: 2026-11-10
last_reviewed: null
review_required: Spec Reviewer + Scientific/Numerical Reviewer
approval_ref: N/A while draft
---

# Adopt the supergravity axion kinetic normalization

## Objective

Make every **physical** axion kinetic path in CYAxiverse use the kinetic matrix
$K_{\rm kin} = 2\,\partial_T\partial_{\bar T}\mathcal{K} = 2\,$`Kinv`$^{-1}$ in
$\mathcal{L}_{\rm kin} = -\tfrac12 M_{\rm pl}^2 K_{\rm kin}\,\partial\vartheta\,\partial\vartheta$,
with period-one $\vartheta$. This is the literal $\mathcal{N}=1$ supergravity
normalization, $c = 1$.

The named arXiv:2608.14780 **reproduction paths** are excluded. They keep
$c = \tfrac12$ under the approved P96/A96 contracts and carry their own
labels (R-008).

Every persisted output is labelled with its normalization, readers fail closed
on ambiguous labels, and the convention is documented. The potential, the
stored `Kinv`, and the vacuum locations and counts do not change.

## Motivation

Issue #232 shows that the physical kinetic paths in `vmm` use $c = \tfrac12$.

- **Sources that define the term completely with period-one axions** all
  display $c = 1$: arXiv:2412.12012, arXiv:2309.13145, arXiv:2512.00144
  (Eq. C.18) and arXiv:2608.14780 (Eq. 23). arXiv:2412.12012 also uses
  $c = 1$ in its published numbers.
- **arXiv:2103.06812** displays $c = 2$, with period-2π axions. It is a
  documented exception, not part of the consensus.

Compared with the $c = 1$ sources, `vmm` decay constants are low by √2,
masses high by √2, quartics high by 4, cubics high by $2^{3/2}$ and
$g_{a\gamma\gamma}$ high by √2. PR #219 consumes these spectra and waits on
this specification.

## Current baseline

### Source facts

Common measure: $\mathcal{L}_{\rm kin} = -c\,M_{\rm pl}^2 g\,\dot\vartheta\dot\vartheta$,
where $g = \partial_T\partial_{\bar T}\mathcal{K}$, $\mathcal{K} = -2\log\mathcal{V}$,
$\vartheta$ has period one and $\tau_i = \tfrac12\kappa_{ijk}t^jt^k$.

| Source | Displayed $c$ | Basis |
|---|---|---|
| arXiv:2412.12012 | 1 | Eqs. 3.1, 3.11 |
| arXiv:2309.13145 | 1 | Eqs. 3.6, 3.9; its inverse kinetic matrix is $2(\tau\tau^T - \mathcal{V}\kappa t)$ (Eq. 3.7) |
| arXiv:2512.00144 | 1 | Eq. C.18 |
| arXiv:2608.14780 | 1 | Eqs. 22–23 |
| arXiv:2103.06812 | 2 | Eqs. 1.1, 2.4; period-2π axions |
| arXiv:2112.04503, arXiv:1808.01282 | undetermined | $K$ not defined precisely |

In arXiv:2309.01831 Eq. 9, the potential uses $K^{a\bar b} = g^{-1}$.

### Implementation facts (`vmm` `10aa3dc`)

Every cited path is unchanged between `34187c5`, where these facts were
first verified, and `10aa3dc`.

**CYTools and storage.** CYTools `compute_inverse_kahler_metric` returns
$g^{-1} = 4(\tau\tau^T - \mathcal{V}\kappa t)$, and every generator stores it
as `Kinv`.

**Physical kinetic paths at $c = \tfrac12$:**

| Path | What it does |
|---|---|
| `src/read.jl` `potential_factored` | `C = chol(Kinv)` |
| `src/read.jl` `_kinetic_matrix` and `_validate_kinetic_matrix` | `K = inv(Kinv)`; the docstring calls it "`K = (1/2) g`" with $g$ undefined |
| `src/generate.jl` `_canonical_factor_from_K` and its callers | the `hp_spectrum`, `pq_*`, `pq_hybrid_physical_spectrum` and leading-Hessian families, and `scripts/batch_physical_spectrum.jl` |
| `src/axion_photon.jl` `_canonical_frame` and the QED kinetic norm | `chol(Kinv)` |
| `src/paper_benchmarks/fuzzy_axions_model_stage.jl` | `chol(Kinv)` |

**Reproduction paths at $c = \tfrac12$ by contract:**
`src/paper_benchmarks/n8_g3_control.jl` (P96),
`src/paper_benchmarks/poly102_inflation.jl` (A96 author reproduction) and
`src/paper_benchmarks/reduced_models.jl` (the arXiv:2608.14780 N = 5 and
N = 8 models).

**Vacuum search.** It uses only $Q$ and $L$: `vacua_id`, `vacua_estimate`
and the `vacua_pipeline` searches. `scripts/vacua_pipeline.jl` reads $K$ only
to check its shape and finiteness.

**Persisted layouts:**
- `spectrum/physical/metadata/*` scalar datasets, with
  `schema_version = "cyaxiverse-physical-spectrum-3"`;
- unversioned `hp_spectrum_save` groups `spectrum/{masses,decay,quartdiag,quart31,quart22,cubic}`;
- `spectrum/axion_photon` with identity group `identity`, whose
  `schema_version` is the integer 1;
- an axion–photon CSV summary.

### Empirical evidence (Issue #232)

- arXiv:2412.12012 Sec. 4.2.1: Eq. 4.3 is reproduced to two figures only with
  $c = 1$. The inverse metric was rebuilt independently in CYTools.
- arXiv:2608.14780: the Eq. 96 eigenvalues equal eig($g$) within 0.07% on the
  reconstructed N = 8 geometry. The published numbers therefore use
  $c = \tfrac12$.
- The F-term cross term with $K^{a\bar b} = g^{-1}$ reproduces
  arXiv:2309.01831 Eq. 9 analytically.

### Owner-approved conventions

- P96 (Issue #148, 2026-09-10).
- The owner decisions of 2026-11-10 recorded on Issue #232:
  1. adopt $c = 1$ and document it;
  2. new schema plus a label, no regeneration;
  3. retain P96;
  4. version impact `minor`.

## Scope

- One shared kinetic-factor function (R-001), used by every physical kinetic
  path listed above.
- Labelled persisted schemas for the three output families (R-006).
- Fail-closed readers (R-007) and a rule for combining outputs (R-007).
- Labels on the reproduction paths (R-008).
- Tests, docstrings, a conventions documentation page, and a README note.

## Non-scope

- Regenerating persisted spectra or axion–photon scans.
- Changing the potential, `Kinv` storage, charges, phases or vacuum algorithms.
- Changing the P96/A96 numerics or any Issue #148 gate.
- Conversion utilities between normalizations.
- Determining the arXiv:2103.06812 numerics.
- Re-interpreting published results.
- Releasing or adopting a package version.
- Updating PR #219's readers. That is the consumer's own follow-up.

## Scientific claim boundary

### This specification may establish

- That the CYAxiverse physical kinetic outputs follow the displayed convention
  of arXiv:2412.12012, arXiv:2309.13145, arXiv:2512.00144 and
  arXiv:2608.14780 Eq. 23.
- That those outputs reproduce arXiv:2412.12012 Eq. 4.3.

### This specification must not establish

- Population-level or physical claims from legacy outputs that are relabelled
  but not recomputed.
- That any past paper's results are wrong.
- Any statement about the arXiv:2103.06812 numerics.

## Fixed conventions and invariants

- **Physical kinetic matrix:** $K_{\rm kin} = 2\,$`Kinv`$^{-1}$, with the ½
  prefactor and period-one $\vartheta$.
- **Canonical factor:** $B$ is lower triangular with positive diagonal and
  $BB^T = $`Kinv`$/2 = K_{\rm kin}^{-1}$.
- **Coordinate map:** $\vartheta = B\psi$, with
  $\mathcal{L}_{\rm kin} = -\tfrac12 M_{\rm pl}^2\dot\psi^T\dot\psi$.
- **Transformation rules:** a charge column $q$ maps to $\hat q = B^Tq$, and a
  ϑ-Hessian $H$ maps to $B^THB$.
- **Potential:** uses $K^{a\bar b} = $`Kinv` unchanged.
- **Reproduction contracts:** P96/A96 keep $c = \tfrac12$.

## Requirements

### R-001 — Single physical kinetic convention

Every physical kinetic path SHALL obtain its kinetic factor from one shared
function, `kinetic_factor(Kinv)`. That function SHALL return
$B = $ `cholesky(Symmetric(Kinv ./ 2)).L`, satisfying the *Fixed conventions*.

Every physical path that forms $K_{\rm kin}$ explicitly SHALL use
$2\,$`inv(Hermitian(Kinv))`, which includes `_kinetic_matrix`.

The three reproduction paths named in *Current baseline* are exempt. They are
governed by R-008.

No other physical path SHALL factor or invert `Kinv` for kinetic purposes.

### R-002 — Invariance of potential and vacua

The following SHALL be unchanged:

(a) the potential data $(Q, L)$, including the pair-term amplitudes, and the
stored `Kinv`;

(b) the vacuum locations, in period-one $\vartheta$ coordinates of the GLSM
basis, as returned by `vacua_id` and `vacua_estimate` and as stored in the
`vacua_pipeline` group datasets (`estimate`, `issquare`, `verified`,
`threshold`, and the coordinate datasets);

(c) the vacuum counts;

(d) the Hessian inertia in $\vartheta$ (by Sylvester's law of inertia).

**Bitwise comparison.** These are compared bitwise against baseline artifacts
generated at `vmm` `10aa3dc`. The baseline is pinned by its sha256 in
`plan.md`, and the comparison uses the same Julia 1.12 patch release, OS image
and BLAS as the CI job that runs the fixtures.

**Tolerance comparison.** A tolerance comparison is allowed only where the PR
lists a changed floating-point schedule in a K-independent path. It uses the
periodic torus distance at most $\max(10^{-12},$ `merge_tolerance`$)$.

**Quantities that are not invariant.** Canonical-coordinate quantities and
mass-threshold counts (for example, modes with $m \ge H_0$) do not fall under
this requirement. They change by R-005, and the changes SHALL be reported.

### R-003 — Source reproduction

WHEN the arXiv:2412.12012 Sec. 4.2.1 inputs are evaluated, CYAxiverse SHALL
reproduce Eq. 4.3 to two significant figures:
- $f_a = (3.2\cdot10^{15}, 1.2\cdot10^{16})$ GeV;
- $m_a = (1.8\cdot10^{-9}, 5.0\cdot10^{-20})$ eV.

### R-004 — Factor contract and orientation

`kinetic_factor` SHALL satisfy:
- $\lVert BB^T - $`Kinv`$/2\rVert \le 10^{-14}\lVert$`Kinv`$\rVert$;
- $B^TK_{\rm kin}B = I$ to $10^{-12}$;
- $B$ is lower triangular with positive diagonal.

**Non-diagonal fixture.** One non-diagonal positive-definite fixture SHALL be
tested, for example `Kinv` $= \begin{pmatrix}3&1\\1&2\end{pmatrix}$ with
hierarchical charges $q_1 = (1,0)^T$ and $q_2 = (1,1)^T$. The test SHALL
assert, to $10^{-10}$ relative:
- $f_1 = M_{\rm pl}/(2\pi\sqrt{G_{11}})$;
- $f_2 = M_{\rm pl}/(2\pi\sqrt{\det G/G_{11}})$, where $G = Q^T($`Kinv`$/2)Q$;
- $m_i^2 = \Lambda_i^4/f_i^2$.

**Validity guard.** The test SHALL also assert that substituting $B^T$ for
$B$, or reversing the order of multiplication, changes at least one asserted
value by more than $10^{-3}$ relative. This guard shows the fixture can detect
orientation errors.

**Scalar fixture.** The one-axion scalar fixture
$f = M_{\rm pl}\sqrt{2/(q^2k)}/(2\pi)$ is retained.

### R-005 — Scaling contract

On the R-004 non-diagonal fixture and on one stored geometry fixture, every
physical output compared with an evaluation at $c = \tfrac12$ of the same
inputs SHALL equal the factor below to $10^{-10}$ relative.

| Output | Factor |
|---|---|
| $f_K$, $f_{\rm pert}$ | $\sqrt2$ |
| masses | $1/\sqrt2$ |
| $\lambda_{\rm self}$, $\lambda_{31}$, $\lambda_{22}$ | $1/4$ |
| cubic couplings | $2^{-3/2}$ |
| axion–photon $f$ | $\sqrt2$ |
| $g_{a\gamma\gamma}$ | $1/\sqrt2$ |
| QED threshold mass | $1/\sqrt2$ |

The axion–photon decay constants SHALL equal arXiv:2309.13145 Eq. 2.15, with
the Gram–Schmidt step taken in the Eq. 3.7 inverse metric.

### R-006 — Persisted labelling contract

**Label.** The label is a scalar UTF-8 string HDF5 dataset named
`kinetic_normalization` (and a CSV column of the same name). It has exactly
these allowed values:

| Value | Meaning | Written by |
|---|---|---|
| `"sugra_c1"` | $K_{\rm kin} = 2\,$`Kinv`$^{-1}$ | every physical writer |
| `"p96_reproduction_c_half"` | the P96 reproduction contract | P96 reproduction writers only |
| `"a96_reproduction_raw_radian"` | the A96 author reproduction | A96 reproduction writers only |

`"legacy_c_half"` is never written. Readers produce it only by inference
(R-007).

**Output families:**

| Family | Location | Version |
|---|---|---|
| F1 physical spectrum | `spectrum/physical/metadata/kinetic_normalization` | `spectrum/physical/metadata/schema_version = "cyaxiverse-physical-spectrum-4"` |
| F2 `hp_spectrum_save` | `spectrum/metadata/kinetic_normalization` | new `spectrum/metadata/schema_version = "cyaxiverse-hp-spectrum-1"` |
| F3 axion–photon | `spectrum/axion_photon/identity/kinetic_normalization` | identity `schema_version = 2` |
| F3 CSV summary | column `kinetic_normalization` on every row | — |

**In memory.** Every result type or `NamedTuple` returned by a physical
kinetic path SHALL expose `kinetic_normalization::Symbol`, equal to
`:sugra_c1` for computed results. Readers SHALL return the stored or inferred
value as a `Symbol`.

### R-007 — Readers and combination (fail closed)

**Inference is limited to recognized legacy layouts:**
- F1 with `schema_version == "cyaxiverse-physical-spectrum-3"` and no label;
- F2 with `spectrum/masses/log10` present and no `spectrum/metadata` group;
- F3 with identity `schema_version == 1` and no label.

These SHALL be read as `:legacy_c_half`, and the result SHALL record that the
value was inferred.

**Readers SHALL throw an `ArgumentError`** naming the file and family for:
- a `-4`, `hp-spectrum-1` or identity-2 output whose label is missing;
- a label with a value outside R-006;
- a legacy version that carries a label;
- a physical schema (F1 `-4`, F2 `hp-spectrum-1`, F3 identity-2) whose label
  is not `"sugra_c1"`;
- an unknown schema version or layout.

**Combination.** Any CYAxiverse function or script that combines outputs from
more than one file or call SHALL reject inputs whose `kinetic_normalization`
values differ, with an `ArgumentError` naming both values. No implicit
conversion is provided.

### R-008 — Reproduction paths retained and distinguishable

The three named reproduction paths and their tests SHALL be numerically
unchanged. Their results SHALL carry `kinetic_normalization` =
`:p96_reproduction_c_half` or `:a96_reproduction_raw_radian`, and any
persisted output SHALL carry the matching R-006 string.

Physical readers (F1–F3) SHALL reject reproduction labels, as R-007 requires.

### R-009 — Documentation

A documentation page on conventions SHALL state:
- the definition, units and factor contract;
- the cross-paper table with equation references;
- the P96/A96 exception;
- the arXiv:2608.14780 Eq. 23 / Eq. 96 and arXiv:2103.06812 discrepancies;
- the R-005 scaling table;
- the schema and label history.

The affected docstrings SHALL point to it, including the `_kinetic_matrix`
"`K = (1/2) g`" text, which is to be replaced. So SHALL the README.

### R-010 — Certification boundary

IF a quantity in `test/fixtures/scientific_certification` is built from
`Kinv` through a path covered by R-001, the work SHALL stop for owner
approval before that fixture is re-certified. Fixtures that supply $K$
directly are out of scope and SHALL be shown to be unchanged.

## Acceptance gates

### CYAX-0232 G0 — Specification approval

**Objective:** an approved specification revision.

**Acceptance:** an independent specification review in a fresh context,
followed by an owner approval comment on Issue #232 that names the reviewed
revision.

**Stop condition:** a normative gap is found.

### CYAX-0232 G1 — Implementation candidate

**Objective:** R-001 to R-010 implemented.

**Acceptance:**
- the fast and full suites, the Python regressions, the docs build and
  `agent_verify.py` all pass at the exact SHA;
- every changed test is listed with its reason;
- the R-002 baseline hashes are recorded.

**Stop condition:** R-002 fails, or R-010 requires re-certification.

### CYAX-0232 G2 — Independent review

**Objective:** an independent judgement of the whole PR.

**Acceptance:** a fresh-context scientific and numerical review of the whole
PR at the exact SHA, plus CI green.

**Stop condition:** a blocking finding, which produces a new candidate.

## Verification requirements

| Evidence | Requirement |
|---|---|
| Factor identity, orientation guard and analytic fixtures | R-004 |
| Named source fixture | R-003 |
| Scaling table | R-005 |
| Pinned bitwise baseline | R-002 |
| Round-trip per family; legacy inference; each rejection case; mixed-label combination | R-006, R-007 |
| Reproduction paths unchanged and labelled | R-008 |
| Docs build | R-009 |
| Certification audit | R-010 |

## Interfaces and compatibility

**Public API.** Signatures are unchanged, and the result fields gain
`kinetic_normalization`. The physical outputs change by the R-005 factors.

**Persisted schemas.**
- F1 moves to `-4`.
- F2 gains `hp-spectrum-1` metadata.
- F3 identity moves to version 2.
- Legacy outputs stay readable as `:legacy_c_half`.

**Downstream.** PR #219's readers, which accept `-3` only, need their own
update.

**Package-version impact:** `minor`. The version is not bumped on the feature
branch.

## Dependencies and blockers

- PR #219 depends on this specification.

## Open owner decisions

None at drafting. The durable `approval_ref` will be the owner comment on
Issue #232.

## Completion criterion

R-001 to R-010 pass at an approved, reviewed candidate merged into `vmm` by the
owner.
