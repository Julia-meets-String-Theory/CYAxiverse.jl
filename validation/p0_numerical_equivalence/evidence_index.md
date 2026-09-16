# CYAX-0170 P0 evidence index

Status: **CYAX-0170 G0 approved; G2 evidence PASS; synchronization candidate pending fresh normative-fidelity review**

## Identity and topology

- Scientific source: `7a40285bb5c313f7e8746b90644d5f45bb67be44`.
- Governing Issue: #170.
- Governing specification: `specs/0170-p0-numerical-equivalence/spec.md`.
- Owner approval: Issue #170 comment `5685945127`, approving the P0 baseline and
  `cyaxiverse-numerical-equivalence-v1` subject only to mechanical
  synchronization of the four approved dispositions.
- Independently reviewed evidence candidate:
  `ddc304f25040ff36bc84e7897d5b6dc5d16344f6`.
- `6f2acaef2a181937f63d8f085c4317e91a08267b` is mechanical bookkeeping only;
  it is not the reviewed evidence candidate.
- Contemporary `vmm` integration/base revision:
  `4523ab090c70ba4fcf3b1711a028d19f2b8e2a62`.
- Initial draft revision: `99d702c8b28e674d34e29566f8ecc483d76c7f7f`.
- Topology used: Manager; Worker A environment/load; Worker B compatibility;
  Worker C semantics/F1-F13; Worker D B2-B7; independent numerical reviewer.
  The first review of candidate `8dab6e6185867c62d962b44ca4e664f749df55db`
  returned **revise/BLOCKED** and drove bounded evidence corrections.  A fresh
  v2 review drove the provenance correction.  Independent Numerical Reviewer
  v3 reviewed exact candidate `ddc304f25040ff36bc84e7897d5b6dc5d16344f6`
  and returned **PASS for evidence integrity**. The later synchronization
  candidate requires a fresh normative-fidelity review.
- Worker write sets were disjoint.  The Manager owns this index, the versioned
  contract, convergence, candidate revision, and final return.

No `src/`, `test/`, `ext/`, `Project.toml`, or production schema change is part
of P0.

## Owner-approved route-specific dispositions

Issue #170 comment `5685945127` approves the P0 baseline and
`cyaxiverse-numerical-equivalence-v1`, subject only to mechanical recording of
these four dispositions:

1. **F13:** validator routes retain historical non-finite rejection; direct
   critical-point non-finite propagation/failure remains a historical parity
   defect. No harmonization is performed.
2. **N=5:** the governed/source-faithful top-level route is authoritative for
   future modularisation and equivalence gates. Stale nested legacy behavior is
   historical only and is not the scientific oracle.
3. **B6:** Float64 and high-precision baselines remain separately preserved.
   Numerical/physical authority is unresolved; no future migration may silently
   choose one route as an oracle.
4. **Phase/volume factor:** observed `2pi^2 I` is historical defect evidence,
   not intended behavior. The `4pi^2 I` correction is separate Issue #172 work;
   Issue #173 remains separate.

These are normative P0 dispositions, not evidence regeneration or reinterpretation.
P1/P2/P3A/P3B remain unauthorized.

## Requirement-to-evidence map

| Requirement | Primary durable evidence | Candidate status |
|---|---|---|
| R-001 exact replay identity | `execution_provenance.md`, `environment_and_load.md`, `environment/`, `environment_benchmarks/`, benchmark metadata | execution bases, exact harness/result hashes, later candidate relationship, two lockfiles, and shared dependency equivalence are explicit |
| R-002 distinct load baselines | `environment_and_load.md` | present: five warm samples and one fresh compiled-cache/build run |
| R-003 compatibility inventory | `compatibility_inventory.md` | present, including named modules, consumers, failures, persistence, and conditional surfaces |
| R-004 encoding/order | `numerical_semantics_and_fixtures.md`, compatibility inventory, F1-F13 | present; route boundaries remain separate |
| R-005 zero/`-Inf` | semantics report, F4/F13 | present as conflicting route behavior; owner-approved route-specific F13 preservation, with no harmonization |
| R-006 support/empty rows | semantics report, F7/F12/F13 | present; transformed Float64 exact-zero and empty-row rule frozen |
| R-007 H1/H2/H3 and thresholds | semantics report, contract v1, F10 | present and separately named |
| R-008 argument/displacement | semantics report, F5/F6 | present and operation-order-sensitive |
| R-009 F1-F13 | `fixtures/`, `fixtures/manifest.toml`, `fixture_julia_replay.jl`, `fixture_replay.py` | 13/13 hashes pass; the retained Julia environment executes the pinned package routes, while Python is non-oracular support |
| R-010 physical/precision | B2, B5, B6 plus semantics report | bounded evidence present; B6 exposes material Float64/high-precision disagreement |
| R-011 ownership/dispatch | contract v1 sections 11-12 | owner-approved future gate; exact per-type copy/borrow policy remains a later implementation decision |
| R-012 B1-B7 | environment/load and benchmark reports/results | present where observable; missing internal counters explicitly unavailable |
| R-013 versioned contract | `numerical_equivalence_contract-v1.md` | owner-approved P0 contract; exact synchronization review pending |
| R-014 precommitted tolerances | contract v1 section 9 | owner-approved baseline for later differential implementation; no silent precision/oracle choice |
| R-015 identity/privacy | `execution_provenance.md`, retained hashes, relative paths, privacy scans | execution revision is distinguished from first integration and exact review candidate; SHA256SUMS is converged |
| R-016 independent review/stop | `independent_review_v3.md` reviews exact evidence candidate `ddc304f25040ff36bc84e7897d5b6dc5d16344f6` | G2 evidence PASS; fresh normative-fidelity review is required for the exact synchronization candidate; no later phase is authorized |

## Key empirical results

- Warm/precompiled import median: `3.829197833 s` and `516067920` allocated
  bytes across five fresh processes in the recorded audit environment.
- Cold fresh compiled-cache/build run: instantiate `0.958732625 s`, strict
  precompile `189.847255291 s`, post-precompile import `4.393729166 s`; full
  process real time `232.39 s`.
- B2-B7 harness: 23/23 recorded rows completed.  Structured/generic derivative
  parity passed its predeclared `rtol=atol=1e-13` probe for h11 4 and 8.
- B4 bounded critical points: the 24-point deterministic start set, four root
  coordinates, one-to-one periodic replay witnesses, residuals, and inertia are
  recorded; per-start failure statuses are not exposed by the public result.
- B5 stationary correction converged in Float64 and BigFloat-128 with residual
  and inertia agreement at the precommitted `1e-40` high-precision gate. Final
  coordinates and diagnostics are recorded; internal evaluation/trial counters
  and the bounded full-flow final state are not exposed.
- B6 N=5 spectra differ materially between current Float64 and high-precision
  light modes.  Both routes now retain eigenvectors, signed quartic logs, and
  mass/quartic/hierarchy diagnostics; no physical interpretation or repair is
  inferred.
- B7 synthetic HDF5 scan, query, enrichment, and spectrum routes completed.

## Manager verification actually executed

1. `julia --project=validation/p0_numerical_equivalence/environment
   validation/p0_numerical_equivalence/scripts/fixture_julia_replay.jl` ran
   under the recorded writable depot prefix and completed all F1-F13 direct
   routes/probes with `package_load=ok` and pinned-source parity true.
2. `python3 validation/p0_numerical_equivalence/scripts/fixture_replay.py`
   completed with all 13 manifest hashes and declared support checks true; it is
   explicitly non-oracular.
3. `python3 -m py_compile validation/p0_numerical_equivalence/scripts/fixture_replay.py`
   completed; generated cache was removed from the candidate.
4. Both retained normalized environments loaded CYAxiverse `0.2.0` with a
   writable depot prefix; exit `0`.
5. `python3 scripts/agent_verify.py diff-check` passed.
6. `Pkg.test("CYAxiverse")` was executed in the retained audit environment.
   Earlier reported testsets passed, then the run stopped with one baseline
   failure at `test/runtests.jl:1252`: observed identity-fixture Hessian
   `2pi^2 I`, expected `4pi^2 I`.  The source helper at
   `scripts/phase_volume_detuning_scan.jl:113` supplies the observed factor.
   P0 does not fix the helper or test.
7. `git diff --name-only <scientific-reference> -- src Project.toml test ext`
   produced no P0 changes to those paths before candidate integration.

The Makie precompile attempted an optional texture-atlas download and emitted a
network-resolution warning, then reconstructed its cache and continued.  This
warning did not cause the package-test failure.

## Evidence hashes before synchronization

These hashes identify unchanged empirical inputs at the pre-synchronization
checkpoint. The final `SHA256SUMS` updates only the authorized lifecycle/index
entries after synchronization.

| Artifact | SHA-256 |
|---|---|
| `fixtures/manifest.toml` | `1eaea4a58a369590a9c20908aa6ee518d9977eb2165cea9ec113bb20c35e4659` |
| `environment/Manifest.toml` | `a421591181011d15d08918f0f5e49a9f7537143a43de8bdec19506c702bccdcf` |
| `environment_benchmarks/Manifest.toml` | `5370a09f21ad09488d6cf81aab67bb648e016d5fbb850881786ab145609c8b84` |
| `benchmark_results/b2_b7.tsv` | `274ba44db46e394f53cdf47c07155048c9772891d348ecded7b0ff0ce3d7692a` |
| `benchmark_results/b2_b7_metadata.txt` | `81541ae468698e900f25b422ae527504054b23767fa47cfbdf11a271fe53a22a` |

## G1 evidence findings and approved boundaries

The evidence records these facts without repairing or reinterpreting them:

1. all-`-Inf` direct critical-point NaN behavior versus modern validator
   rejection;
2. stale legacy N=5 scale versus the governed source-faithful N=5 fixture;
3. material Float64/high-precision N=5 light-spectrum disagreement;
4. the phase/volume-detuning Hessian factor test failure;
5. unavailable failed-start and internal RHS/Hessian/line-search counters;
6. exact future per-result workspace borrow/copy promises;

The owner-approved dispositions above resolve the P0 policy treatment of the
first four facts without changing their empirical identity. The unavailable
observability gaps and future workspace policy remain later implementation
decisions.

Under the P0 boundary these are preserved evidence and explicit boundaries.
They do not authorize production correction or later-phase work.

## Final independent disposition

Independent Numerical Reviewer v3 reviewed exact evidence candidate
`ddc304f25040ff36bc84e7897d5b6dc5d16344f6` and returned **G2 evidence PASS**.
The review independently passed checksum, privacy, source-parity, direct Julia
F1-F13, and bounded B2-B7 coverage checks.  It found no remaining correctable
evidence blocker.  Its post-review record is `independent_review_v3.md` and is
part of the later mechanical synchronization, not the reviewed commit.

The owner-approved P0 baseline is now synchronized into the normative/lifecycle
surfaces. A fresh independent normative-fidelity review must bind to the exact
synchronization commit produced by this handoff; no merge or later phase is
authorized by that review requirement.

## Synchronization preservation record

The pre-synchronization validation tree was compared by SHA-256 and exact
content. The empirical fixtures, retained environments, benchmark outputs,
semantic report, compatibility inventory, execution provenance, and prior
independent review remain byte-identical. Only the authorized lifecycle/index
records changed: the spec/plan/tasks status and dispositions, this evidence
index, the contract status/dispositions, and the corresponding entries in
`SHA256SUMS`. No fixture manifest was regenerated and no empirical result was
altered or relabelled. The final manifest is verified with
`sha256sum -c validation/p0_numerical_equivalence/SHA256SUMS`.
