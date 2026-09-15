# CYAX-0170 P0 evidence index

Status: **G2 evidence PASS; overall P0 REVISE/BLOCKED pending G0 owner decisions**

## Identity and topology

- Scientific source: `7a40285bb5c313f7e8746b90644d5f45bb67be44`.
- Governing Issue: #170.
- Governing specification: `specs/0170-p0-numerical-equivalence/spec.md`.
- Initial draft revision: `99d702c8b28e674d34e29566f8ecc483d76c7f7f`.
- Topology used: Manager; Worker A environment/load; Worker B compatibility;
  Worker C semantics/F1-F13; Worker D B2-B7; independent numerical reviewer.
  The first review of candidate `8dab6e6185867c62d962b44ca4e664f749df55db`
  returned **revise/BLOCKED** and drove bounded evidence corrections.  A fresh
  v2 review drove the provenance correction.  Independent Numerical Reviewer
  v3 reviewed exact candidate `ddc304f25040ff36bc84e7897d5b6dc5d16344f6`
  and returned **PASS for evidence integrity**, with the overall checkpoint
  still blocked on G0 owner decisions.
- Worker write sets were disjoint.  The Manager owns this index, the versioned
  contract, convergence, candidate revision, and final return.

No `src/`, `test/`, `ext/`, `Project.toml`, or production schema change is part
of P0.

## Requirement-to-evidence map

| Requirement | Primary durable evidence | Candidate status |
|---|---|---|
| R-001 exact replay identity | `execution_provenance.md`, `environment_and_load.md`, `environment/`, `environment_benchmarks/`, benchmark metadata | execution bases, exact harness/result hashes, later candidate relationship, two lockfiles, and shared dependency equivalence are explicit |
| R-002 distinct load baselines | `environment_and_load.md` | present: five warm samples and one fresh compiled-cache/build run |
| R-003 compatibility inventory | `compatibility_inventory.md` | present, including named modules, consumers, failures, persistence, and conditional surfaces |
| R-004 encoding/order | `numerical_semantics_and_fixtures.md`, compatibility inventory, F1-F13 | present; route boundaries remain separate |
| R-005 zero/`-Inf` | semantics report, F4/F13 | present as conflicting route behavior; owner/reviewer disposition open |
| R-006 support/empty rows | semantics report, F7/F12/F13 | present; transformed Float64 exact-zero and empty-row rule frozen |
| R-007 H1/H2/H3 and thresholds | semantics report, contract v1, F10 | present and separately named |
| R-008 argument/displacement | semantics report, F5/F6 | present and operation-order-sensitive |
| R-009 F1-F13 | `fixtures/`, `fixtures/manifest.toml`, `fixture_julia_replay.jl`, `fixture_replay.py` | 13/13 hashes pass; the retained Julia environment executes the pinned package routes, while Python is non-oracular support |
| R-010 physical/precision | B2, B5, B6 plus semantics report | bounded evidence present; B6 exposes material Float64/high-precision disagreement |
| R-011 ownership/dispatch | contract v1 sections 11-12 | proposed future gate; exact per-type copy/borrow policy remains later owner decision |
| R-012 B1-B7 | environment/load and benchmark reports/results | present where observable; missing internal counters explicitly unavailable |
| R-013 versioned contract | `numerical_equivalence_contract-v1.md` | proposed candidate present |
| R-014 precommitted tolerances | contract v1 section 9 | proposed from existing defaults/tests before P2 output; independent review pending |
| R-015 identity/privacy | `execution_provenance.md`, retained hashes, relative paths, privacy scans | execution revision is distinguished from first integration and exact review candidate; SHA256SUMS is converged |
| R-016 independent review/stop | `independent_review_v3.md` reviews exact candidate `ddc304f25040ff36bc84e7897d5b6dc5d16344f6` | G2 evidence PASS; no correctable evidence blocker remains; G0 and owner-only decisions keep overall P0 blocked |

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
  and inertia agreement at the proposed `1e-40` high-precision gate.  Final
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

## Candidate hashes before final commit

These hashes identify the integrated inputs at this checkpoint.  A full
`SHA256SUMS` is generated after the final Manager edits and before G2 review.

| Artifact | SHA-256 |
|---|---|
| `fixtures/manifest.toml` | `1eaea4a58a369590a9c20908aa6ee518d9977eb2165cea9ec113bb20c35e4659` |
| `environment/Manifest.toml` | `a421591181011d15d08918f0f5e49a9f7537143a43de8bdec19506c702bccdcf` |
| `environment_benchmarks/Manifest.toml` | `5370a09f21ad09488d6cf81aab67bb648e016d5fbb850881786ab145609c8b84` |
| `benchmark_results/b2_b7.tsv` | `274ba44db46e394f53cdf47c07155048c9772891d348ecded7b0ff0ce3d7692a` |
| `benchmark_results/b2_b7_metadata.txt` | `81541ae468698e900f25b422ae527504054b23767fa47cfbdf11a271fe53a22a` |

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

## Final independent disposition

Independent Numerical Reviewer v3 reviewed exact candidate
`ddc304f25040ff36bc84e7897d5b6dc5d16344f6` and returned **G2 evidence PASS**.
The review independently passed checksum, privacy, source-parity, direct Julia
F1-F13, and bounded B2-B7 coverage checks.  It found no remaining correctable
evidence blocker.  Its post-review record is `independent_review_v3.md` and is
part of the later mechanical synchronization, not the reviewed commit.

The overall P0 checkpoint remains **REVISE/BLOCKED** because the required G0
approval and owner dispositions listed above are absent.  No later phase is
authorized.
