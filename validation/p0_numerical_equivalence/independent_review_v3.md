# CYAX-0170 G2 independent numerical review v3

G2 evidence verdict: **PASS** — no correctable evidence blocker remains in the
reviewed candidate.

Overall P0 checkpoint: **REVISE / BLOCKED** pending the absent G0 owner approval
and the owner-only scientific or baseline decisions listed below.  This G2
verdict does not approve the proposed contract or authorize P1, P2, P3A, or
P3B.

Reviewer: `Independent Numerical Reviewer v3` (`/root/independent_review_v3`).
I did not define the P0 contract or fixture corpus.

Reviewed candidate: `ddc304f25040ff36bc84e7897d5b6dc5d16344f6` on
`research/p0-numerical-equivalence-20260915`.

Scientific source: `7a40285bb5c313f7e8746b90644d5f45bb67be44`.

Contract under review: `cyaxiverse-numerical-equivalence-v1` from
`numerical_equivalence_contract-v1.md`.

This verdict file is a post-review mechanical record.  It was not present in
the reviewed candidate commit, is not included in that candidate's
`SHA256SUMS`, and does not claim self-referential evidence.

## Identity and artifact checks

The reviewed candidate has no changes to `src/`, `test/`, `ext/`, or
`Project.toml` relative to the scientific source.  Repository-relative P0
paths contain no machine-local provenance.  The candidate was clean before
this post-review record was written.

Key artifact identities from the reviewed candidate's checksum manifest:

| Artifact | SHA-256 |
|---|---|
| `environment/Project.toml` | `ef6dabfa1ded67f26de09b9931c9b56969ef39a762af35da48efa78a4c7a67d9` |
| `environment/Manifest.toml` | `a421591181011d15d08918f0f5e49a9f7537143a43de8bdec19506c702bccdcf` |
| `environment_benchmarks/Project.toml` | `f42fcc77faac53ac2ce271605b59ef3e677c018e7510dfe306f04ebb5fa95cec` |
| `environment_benchmarks/Manifest.toml` | `5370a09f21ad09488d6cf81aab67bb648e016d5fbb850881786ab145609c8b84` |
| `fixtures/manifest.toml` | `1eaea4a58a369590a9c20908aa6ee518d9977eb2165cea9ec113bb20c35e4659` |
| `numerical_semantics_and_fixtures.md` | `e8ed83809a5c52156878d3925725bcfddceb61603694b09461b04e56d63bd326` |
| `benchmark_baseline.md` | `cee00e665f1cd9b6eb685b6146c5f7e3875d7db8f7d64eef409bd61e272c9f44` |
| `benchmark_results/b2_b7.tsv` | `274ba44db46e394f53cdf47c07155048c9772891d348ecded7b0ff0ce3d7692a` |
| `benchmark_results/b2_b7_metadata.txt` | `81541ae468698e900f25b422ae527504054b23767fa47cfbdf11a271fe53a22a` |
| `execution_provenance.md` | `df2fefd63cc7d69a5d59951ee77f97dff4eb888c4b8ec342a2588cfe2e4f2bdc` |
| `numerical_equivalence_contract-v1.md` | `dfe406c21353de51bab7c3db57b1f6f9bfbdbbcfd254eb42587b9c54072b9f4b` |

## Commands and outcomes

1. `git rev-parse HEAD` returned the exact reviewed SHA
   `ddc304f25040ff36bc84e7897d5b6dc5d16344`.
2. `shasum -c validation/p0_numerical_equivalence/SHA256SUMS` passed for every
   listed artifact, including the provenance correction and prior review.
3. `git diff --name-status
   7a40285bb5c313f7e8746b90644d5f45bb67be44 -- src test ext Project.toml`
   returned no output.  `git diff --check` and
   `python3 scripts/agent_verify.py diff-check` passed.
4. The direct Julia replay was independently run against the reviewed
   candidate with the retained normalized environment and a writable depot:

   ```text
   env JULIA_DEPOT_PATH=<retained-writable-depot>:<host-depot> \
     JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_PROGRESS=0 \
     julia --startup-file=no --history-file=no \
     --project=validation/p0_numerical_equivalence/environment \
     validation/p0_numerical_equivalence/scripts/fixture_julia_replay.jl
   ```

   It exited `0` under Julia `1.12.6` and reported
   `current_commit=ddc304f25040ff36bc84e7897d5b6dc5d16344`,
   `source_revision=7a40285bb5c313f7e8746b90644d5f45bb67be44`,
   `source_tree_equal=true`, and `package_load=ok`.  All F1-F13 direct rows
   completed.  The replay independently confirms the route-specific
   acceptance/rejection behavior, transformed support, argument/displacement
   probes, source-faithful F11 route, aliasing probe, and F13 non-finite
   failure without changing production code.
5. `python3 validation/p0_numerical_equivalence/scripts/fixture_replay.py`
   passed all 13 manifest/hash and declared support checks.  It remains
   correctly labeled non-oracular support; the Julia replay is the package
   route evidence.
6. A separate attempt with a newly empty temporary Julia depot was interrupted
   during dependency precompilation after the bounded review time.  It exited
   `130` on interrupt before reaching the harness.  This was a review-run
   setup timeout only; the retained-depot exact-candidate replay above
   completed successfully.  No transient `__pycache__` remained.

## Provenance assessment

The prior G2 provenance blocker is corrected.  `execution_provenance.md`
explicitly separates:

- the Git execution base (`99d702c...` for environment/load and the working
  tree based on `8dab6e6...` for the F1-F13 and B2-B7 runs);
- the SHA-256 of each executed harness and retained result bytes;
- the first integration candidate (`5c90bfe...`); and
- the later exact candidate under this review (`ddc304f...`).

It also states that the Manager and v2 reviewer reran the Julia fixture
harness at `5c90bfe...`, and that a generated result cannot contain the hash of
the future commit that first adds it.  The retained reports therefore keep
their historical execution bases, while the content hashes and candidate
relationship make the integration chain reconstructible.  The provenance
record does not pretend that the future `ddc304f...` hash was present in an
earlier measurement.

## Coverage and findings

- R-001-R-003: environment/load identity, normalized lockfiles, CPU/OS/BLAS/
  thread state, compatibility surfaces, consumers, persistence, and
  conditional paths are present and checksummed.
- R-004-R-009: the actual Julia replay covers F1-F13.  The evidence separates
  source formula probes from callable package routes and records direct
  errors, support/zero behavior, order/offset/displacement semantics, and
  route boundaries.
- R-010-R-012: B2-B7 retains 23/23 completed bounded rows.  B4 has roots,
  coordinates, residuals, inertia, deterministic starts, and one-to-one
  periodic replay witnesses.  B5 includes trajectories and correction
  diagnostics with the `1e-40` high-precision gate.  B6 retains eigenvectors,
  signed quartic logs, and diagnostics.  B7 is explicitly synthetic.
  Unavailable API counters/statuses are stated as unavailable rather than
  inferred.
- R-013-R-014: the versioned contract and proposed tolerance table are
  present and traceable, but remain proposed until G0 approval.
- R-015: checksum and privacy checks pass, and provenance distinguishes the
  execution base, integration candidate, and reviewed candidate.
- R-016: this fresh review is independent of the contract/fixture authors and
  reviewed the exact candidate named above.

No correctable evidence blocker remains.  The following are deliberately
separate owner or checkpoint decisions, not evidence corrections:

1. G0 durable approval or amendment of the draft S2 specification and proposed
   contract is absent.
2. F13 all-`-Inf` route-specific behavior versus validator rejection requires
   an owner policy decision.
3. F11 governed source-faithful N=5 scale versus the legacy stale route
   requires an owner decision.
4. B6 material Float64/high-precision spectrum disagreement requires a
   numerical-authority disposition; no physical interpretation is inferred.
5. The pre-existing phase/volume-detuning Hessian-factor package-test failure
   requires an owner decision or separately governed correction.

These decisions keep the overall P0 checkpoint blocked and preserve the phase
stop.  They do not identify a correctable defect in the reviewed evidence
candidate.

