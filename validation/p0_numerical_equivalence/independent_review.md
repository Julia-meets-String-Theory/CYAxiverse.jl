# CYAX-0170 G2 independent numerical review

Verdict: **REVISE / BLOCKED**

Reviewer: `Independent Numerical Reviewer v2` (`/root/independent_review_v2`),
fresh reviewer; I did not define the P0 contract or fixtures.

Reviewed candidate: `5c90bfe3ce91dcdca442af9df72cf1f32390aea6`
(`research/p0-numerical-equivalence-20260915`)

Scientific source: `7a40285bb5c313f7e8746b90644d5f45bb67be44`

Contract: `cyaxiverse-numerical-equivalence-v1` from
`validation/p0_numerical_equivalence/numerical_equivalence_contract-v1.md`.

This file is a post-review mechanical record. It was not present in the
reviewed candidate commit and is not included in the reviewed state.

## Identity and artifact checks

The candidate has no changes to `src/`, `test/`, `ext/`, or `Project.toml`
relative to the scientific source. The repository-relative privacy scan found
no machine-local paths in the P0 specification or validation artifacts.

Key SHA-256 identities recorded in the candidate:

| Artifact | SHA-256 |
|---|---|
| `environment/Project.toml` | `ef6dabfa1ded67f26de09b9931c9b56969ef39a762af35da48efa78a4c7a67d9` |
| `environment/Manifest.toml` | `a421591181011d15d08918f0f5e49a9f7537143a43de8bdec19506c702bccdcf` |
| `environment_benchmarks/Project.toml` | `f42fcc77faac53ac2ce271605b59ef3e677c018e7510dfe306f04ebb5fa95cec` |
| `environment_benchmarks/Manifest.toml` | `5370a09f21ad09488d6cf81aab67bb648e016d5fbb850881786ab145609c8b84` |
| `fixtures/manifest.toml` | `1eaea4a58a369590a9c20908aa6ee518d9977eb2165cea9ec113bb20c35e4659` |
| `benchmark_results/b2_b7.tsv` | `274ba44db46e394f53cdf47c07155048c9772891d348ecded7b0ff0ce3d7692a` |
| `benchmark_results/b2_b7_metadata.txt` | `81541ae468698e900f25b422ae527504054b23767fa47cfbdf11a271fe53a22a` |
| `numerical_equivalence_contract-v1.md` | `dfe406c21353de51bab7c3db57b1f6f9bfbdbbcfd254eb42587b9c54072b9f4b` |

Commands and outcomes:

1. `git diff --name-status 7a40285bb5c313f7e8746b90644d5f45bb67be44 -- src test ext Project.toml` — no output; source parity is clean.
2. `shasum -c validation/p0_numerical_equivalence/SHA256SUMS` — all listed artifacts `OK`.
3. The retained normalized consumer environment was loaded with Julia 1.12.6 and a writable depot prefix. `using CYAxiverse` succeeded, reported version `0.2.0`, and `Pkg.status` reported Aqua 0.8.16 and JET 0.12.1.
4. The direct Julia replay was run as:

   ```text
   env JULIA_DEPOT_PATH=<writable-depot>:<host-depot> \
     JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_PROGRESS=0 \
     julia --startup-file=no --history-file=no \
     --project=validation/p0_numerical_equivalence/environment \
     validation/p0_numerical_equivalence/scripts/fixture_julia_replay.jl
   ```

   It exited `0`, reported `current_commit=5c90bfe3ce91dcdca442af9df72cf1f32390aea6`, `source_tree_equal=true`, and `package_load=ok`. F1-F13 all produced a Julia replay row. The Python helper also passed all 13 manifest/hash checks; it is correctly labeled non-oracular support.
5. `python3 scripts/agent_verify.py diff-check` and `git diff --check` passed. The temporary Python bytecode created during review was removed.
6. The retained full-package-test record was inspected. It stops at `test/runtests.jl:1252`: the identity fixture observes `2pi^2 I` from `scripts/phase_volume_detuning_scan.jl:113`, while the test expects `4pi^2 I`. Source inspection confirms the mismatch is pre-existing; no production or test repair is part of this review.

## Coverage assessment

The retained evidence is strong for the bounded P0 scope:

- R-001/R-002: Julia/build/host/BLAS/thread/RNG identity, normalized lockfiles,
  and distinct warm versus cold load measurements are present.
- R-003: the compatibility inventory covers the named modules, aliases,
  dispatch domains, optional extensions, scripts/consumers, SLURM behavior,
  and HDF5 paths and fields.
- R-004-R-009: coefficient/log separation, route-specific term order,
  zero/`-Inf`, transformed-Float64 support, H1/H2/H3 distinctions, thresholds,
  and F1-F13 are documented. Julia route calls are separated from source-level
  formula probes.
- R-010/R-012: B2-B7 contains 23/23 completed bounded rows. B4 includes the
  deterministic 24-start set, root coordinates, residuals, inertia, and
  one-to-one periodic replay witnesses. B5 contains trajectory samples,
  correction coordinates/diagnostics, and the `1e-40` high-precision gate. B6
  retains eigenvectors, signed quartic logs, and diagnostics. B7 is explicitly
  synthetic and bounded.
- R-011/R-013/R-014: the future ownership/JET/Aqua gate and versioned
  comparison/tolerance contract are specified as proposed, not treated as
  already approved.
- R-015: the checksum manifest passes and the committed evidence uses
  repository-relative paths.

## Findings

### F-1 — BLOCKER, correctable evidence identity

The exact reviewed candidate is `5c90bfe3...`, but substantive evidence files
identify earlier execution states: `environment_and_load.md` names
`99d702c...`, while `numerical_semantics_and_fixtures.md`,
`benchmark_baseline.md`, `b2_b7_metadata.txt`, and the evidence index refer to
`8dab6e6...` or earlier candidate topology. The benchmark script itself would
record the current revision when rerun, but the retained output does not.

This is a correctable R-001/R-015/G2 identity gap, not evidence that the source
code changed. The Manager must either regenerate the retained evidence at the
exact candidate or explicitly record the pre-commit working-tree identity and
script/artifact content hashes, then refresh the index and checksum manifest.
Any substantive regenerated result requires review of the changed state.

### F-2 — BLOCKER, G0 owner decision is absent

`spec.md` remains `status: draft` with `approval_ref: N/A`, and the contract is
proposed. The required durable owner approval for G0 is not present. Therefore
the proposed tolerances, route boundaries, and scientific interpretation limits
cannot be treated as an approved normative contract.

### F-3 — OWNER DECISION, cross-route `-Inf` policy

F4/F13 correctly show that direct `critical_points` can reach the hostile
all-`-Inf` classification edge, while the generic workspace and inflation
context reject non-finite `L`. The evidence is useful and the implementation
was not repaired. The owner must decide whether v1 preserves route-specific
reachability or requires a single rejection policy.

### F-4 — OWNER DECISION, governed N=5 scale conflict

F11 correctly exposes the source-faithful `reduced_models.jl` N=5 scale and
the legacy `author_inflation` route's stale N8 value. The owner must identify
which route is normative for later equivalence. No production correction is
authorized by this review.

### F-5 — BLOCKER, material B6 precision disagreement

The N=5 Float64 and high-precision spectra disagree materially in three light
modes and in the mass signs. The retained Float64 mass-basis residuals also
reach about `0.2593`, while high-precision residuals reach about `0.1388`.
Eigenvectors, quartic logs, and diagnostics are retained, and the evidence does
not infer a physical answer. Before a future parity gate can use these spectra,
the owner/numerical authority must decide the route-specific oracle and
disposition of the disagreement. Higher precision is not automatically the
historical Float64 oracle.

### F-6 — ACCEPTED GAP, unavailable status/counter observability

The current APIs do not expose B4 per-start discarded statuses, B5a RHS/Hessian
counters or final integrated state, or B5b Hessian/line-search trial counts.
The candidate states these gaps instead of inferring values. This satisfies the
evidence requirement for an explicit gap, but the owner must decide whether the
future contract needs new observability before implementation work.

### F-7 — MINOR, F9 aliasing witness could be stronger

The Julia F9 replay proves that retained gradient/Hessian fields are the same
workspace objects after reuse, but the second input happens to produce the same
numeric arrays, so the output does not visibly demonstrate changed values. The
source and pointer identity support the borrowed-storage conclusion; a future
revision may add a deliberately differing second evaluation if an explicit
mutation witness is required.

### F-8 — OWNER DECISION, pre-existing full-test defect

The package test failure at `test/runtests.jl:1252` is a real baseline defect in
the pinned source/test pair. P0 correctly records it and does not change either
side. The owner must decide whether to accept it as a frozen baseline defect or
open a separately governed correction; it must not be silently treated as a
clean package gate.

## Final G2 disposition

The candidate is valuable and largely reconstructible, but it does not pass G2.
The stale exact-candidate identity, missing G0 approval, unresolved cross-route
and B6 scientific choices, and known package-test defect require revision or
explicit owner decisions. The Manager must stop at the P0 checkpoint and must
not start P1, P2, P3A, or P3B from this verdict.
