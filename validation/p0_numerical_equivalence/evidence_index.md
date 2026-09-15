# CYAX-0170 P0 evidence index

Status: **candidate evidence; G0 approval and G2 independent review pending**

## Identity and topology

- Scientific source: `7a40285bb5c313f7e8746b90644d5f45bb67be44`.
- Governing Issue: #170.
- Governing specification: `specs/0170-p0-numerical-equivalence/spec.md`.
- Initial draft revision: `99d702c8b28e674d34e29566f8ecc483d76c7f7f`.
- Topology used: Manager; Worker A environment/load; Worker B compatibility;
  Worker C semantics/F1-F13; Worker D B2-B7; fresh independent numerical
  reviewer reserved for G2 after candidate integration.
- Worker write sets were disjoint.  The Manager owns this index, the versioned
  contract, convergence, candidate revision, and final return.

No `src/`, `test/`, `ext/`, `Project.toml`, or production schema change is part
of P0.

## Requirement-to-evidence map

| Requirement | Primary durable evidence | Candidate status |
|---|---|---|
| R-001 exact replay identity | `environment_and_load.md`, `environment/`, `environment_benchmarks/`, benchmark metadata | present; two execution lockfiles retained and shared dependency equivalence checked |
| R-002 distinct load baselines | `environment_and_load.md` | present: five warm samples and one fresh compiled-cache/build run |
| R-003 compatibility inventory | `compatibility_inventory.md` | present, including named modules, consumers, failures, persistence, and conditional surfaces |
| R-004 encoding/order | `numerical_semantics_and_fixtures.md`, compatibility inventory, F1-F13 | present; route boundaries remain separate |
| R-005 zero/`-Inf` | semantics report, F4/F13 | present as conflicting route behavior; owner/reviewer disposition open |
| R-006 support/empty rows | semantics report, F7/F12/F13 | present; transformed Float64 exact-zero and empty-row rule frozen |
| R-007 H1/H2/H3 and thresholds | semantics report, contract v1, F10 | present and separately named |
| R-008 argument/displacement | semantics report, F5/F6 | present and operation-order-sensitive |
| R-009 F1-F13 | `fixtures/`, `fixtures/manifest.toml`, `fixture_replay.py` | 13/13 hashes and expected checks pass |
| R-010 physical/precision | B2, B5, B6 plus semantics report | bounded evidence present; B6 exposes material Float64/high-precision disagreement |
| R-011 ownership/dispatch | contract v1 sections 11-12 | proposed future gate; exact per-type copy/borrow policy remains later owner decision |
| R-012 B1-B7 | environment/load and benchmark reports/results | present where observable; missing internal counters explicitly unavailable |
| R-013 versioned contract | `numerical_equivalence_contract-v1.md` | proposed candidate present |
| R-014 precommitted tolerances | contract v1 section 9 | proposed from existing defaults/tests before P2 output; independent review pending |
| R-015 identity/privacy | retained hashes, relative paths, privacy scans | candidate present; final SHA256SUMS generated at commit convergence |
| R-016 independent review/stop | not yet produced | blocked behind G0 and exact candidate commit |

## Key empirical results

- Warm/precompiled import median: `3.829197833 s` and `516067920` allocated
  bytes across five fresh processes in the recorded audit environment.
- Cold fresh compiled-cache/build run: instantiate `0.958732625 s`, strict
  precompile `189.847255291 s`, post-precompile import `4.393729166 s`; full
  process real time `232.39 s`.
- B2-B7 harness: 23/23 recorded rows completed.  Structured/generic derivative
  parity passed its predeclared `rtol=atol=1e-13` probe for h11 4 and 8.
- B4 bounded critical points: 24 starts, four unique roots, one minimum, with
  recorded periodic/residual/inertia outputs.
- B5 stationary correction converged in Float64 and BigFloat-128 with residual
  and inertia agreement; internal evaluation/trial counters are not exposed.
- B6 N=5 spectra differ materially between current Float64 and high-precision
  light modes; no physical interpretation or repair is inferred.
- B7 synthetic HDF5 scan, query, enrichment, and spectrum routes completed.

## Manager verification actually executed

1. `python3 validation/p0_numerical_equivalence/scripts/fixture_replay.py`
   completed with all 13 manifest hashes and declared expected checks true.
2. `python3 -m py_compile validation/p0_numerical_equivalence/scripts/fixture_replay.py`
   completed; generated cache was removed from the candidate.
3. Both retained normalized environments loaded CYAxiverse `0.2.0` with a
   writable depot prefix; exit `0`.
4. `python3 scripts/agent_verify.py diff-check` passed.
5. `Pkg.test("CYAxiverse")` was executed in the retained audit environment.
   Earlier reported testsets passed, then the run stopped with one baseline
   failure at `test/runtests.jl:1252`: observed identity-fixture Hessian
   `2pi^2 I`, expected `4pi^2 I`.  The source helper at
   `scripts/phase_volume_detuning_scan.jl:113` supplies the observed factor.
   P0 does not fix the helper or test.
6. `git diff --name-only <scientific-reference> -- src Project.toml test ext`
   produced no P0 changes to those paths before candidate integration.

The Makie precompile attempted an optional texture-atlas download and emitted a
network-resolution warning, then reconstructed its cache and continued.  This
warning did not cause the package-test failure.

## Candidate hashes before final commit

These hashes identify the integrated inputs at this checkpoint.  A full
`SHA256SUMS` is generated after the final Manager edits and before G2 review.

| Artifact | SHA-256 |
|---|---|
| `fixtures/manifest.toml` | `1eaea4a58a369590a9c20908aa6ee518d9977eb2165cea9ec113bb20c35e4659` |
| `environment/Manifest.toml` | `a421591181011d15d08918f0f5e49a9f7537143a43de8bdec19506c702bccdcf` |
| `environment_benchmarks/Manifest.toml` | `5370a09f21ad09488d6cf81aab67bb648e016d5fbb850881786ab145609c8b84` |
| `benchmark_results/b2_b7.tsv` | `e6b276317e9016325cb2b8898122aa37ff47d025ec8056337833adab18e7aaf7` |
| `benchmark_results/b2_b7_metadata.txt` | `1b79b1a01b95512d95b05d964d9abd41c1a2100831f57f601c45cfa440420c31` |

## G1 stop findings and owner/reviewer decisions

The candidate records but does not resolve:

1. all-`-Inf` direct critical-point NaN behavior versus modern validator
   rejection;
2. stale legacy N=5 scale versus the governed source-faithful N=5 fixture;
3. material Float64/high-precision N=5 light-spectrum disagreement;
4. the phase/volume-detuning Hessian factor test failure;
5. unavailable failed-start and internal RHS/Hessian/line-search counters;
6. exact future per-result workspace borrow/copy promises;
7. approval or amendment of the proposed route-specific tolerance table.

Under the P0 boundary these are evidence and explicit decision points.  They do
not authorize production correction or later-phase work.
