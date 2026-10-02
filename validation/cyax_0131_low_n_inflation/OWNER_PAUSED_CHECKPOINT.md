# CYAX-0131 r5 owner pause checkpoint

Historical status at the owner pause: **OWNER_PAUSED**. This records a pause,
not a scientific failure or pass. The owner later explicitly resumed bounded
diagnostics; current status is in `execution_record.md` and
`trajectory-diagnostics-report.md`. The full replay remains terminated.

## Candidate and durable state

- Branch: `research/cyax-0131-low-n-inflation-r5`.
- Base: `a5eacd8cd4a5905161cab239a461ba252c64e8e0`; initial tree:
  `d419416514a44d943ddd20290885ac8ba4090999`.
- Four modified tracked files (no other tracked changes):
  - `scripts/inflation_diagnostics_common.jl` — SHA-256
    `b0223ed28b446c4489a40b69542f0846b77732b7badb28f21250e132dbf30e2d`
  - `scripts/inflation_refinement_common.jl` — SHA-256
    `e70fad6e4f8a397708641a220e9f9ae02fae628185d20eb3477481c0ad18978b`
  - `src/paper_benchmarks/poly102_inflation.jl` — SHA-256
    `578a6b44877f82082f50699dedc2830bc2d34f4efedc93394e3b2b255ac9a564`
  - `test/runtests.jl` — SHA-256
    `04d10d42cac81ff8e75d3891832c68aed1cc3def4a341c0e6d9dbd3859d1de24`
- Candidate dependency files remain unchanged. No source or settings were
  changed for this pause. Validation files are ignored by Git; at a later
  authorized freeze, explicitly stage only approved evidence files.
- The active historical `execution_record.md` still says the replay was
  running. This checkpoint supersedes that status; update the record only
  after resumption.

## Completed verification

- Focused regressions: 30/30 passed in 16.8 s. Command:
  `repo_root="$(git rev-parse --show-toplevel)"; p0_env="${repo_root}/validation/p0_numerical_equivalence/environment"; JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_LOAD_PATH="@:${p0_env}:@stdlib" julia --threads=1 --startup-file=no --project=. validation/cyax_0131_low_n_inflation/focused_regressions.jl`.
  Output: `focused-regressions.log`, SHA-256
  `fe53f9f3cc7c8c7d71088dd340498e81e09337958488fe295df9f852951ed4dd`.
- Full `Pkg.test`: 55 testsets, 2,081/2,081 assertions passed using an
  external test environment copied from the pinned P0 Project/Manifest and
  extended only with declared extras under preserve-all. External Project SHA
  `8e26eb84a345c68f887f719dfb6c99645ce30a70aa8a438a4aa6a6aea7331bf6`,
  Manifest SHA
  `b515317bcb0d126c9070f761135073ee1c6ff2463194ca3600aed3a53a997f43`.
  Exact invocation template and warnings/skips are in
  `execution_record.md`; output `full-tests.sanitized.log`, SHA-256
  `bafa0a79e4f2dda00ca84474966dc9445c7d748358f6949b592e7c63bcbb2e53`.
  Four existing Float64 seed-truncation warnings and two missing-real-data
  fixture skips were reported.
- `julia --project=. bin/audit.jl`: passed; JET found no errors across 1,263
  top-level definitions, Aqua and physics sanity checks passed. Sanitized
  output `bin-audit.sanitized.log`, SHA-256
  `e458dcf0a1a361c709248b323175091ddb1238dc73edd175a839b8b3a4c74c50`.
- Python-free import/optional-extension assertions passed; the candidate
  source was loaded, Python environment variables were unset, and neither
  PyCall nor its extension was loaded. Exact command/output are in
  `execution_record.md`.
- `git diff --check` and `python3 scripts/agent_verify.py diff-check` passed.

## Interrupted full replay

The only full replay ran Julia 1.12.6 with default compilation and default
thread settings, candidate project plus pinned P0 environment on
`JULIA_LOAD_PATH`, via:

```sh
repo_root="$(git rev-parse --show-toplevel)"
JULIA_LOAD_PATH="@:${repo_root}/validation/p0_numerical_equivalence/environment:@stdlib" \
  julia --startup-file=no --project=. \
  validation/cyax_0131_low_n_inflation/replay.jl \
  > validation/cyax_0131_low_n_inflation/replay.log 2>&1
```

The owner ordered SIGINT after diagnostic preservation. Peer process evidence
confirmed the interrupted replay worker had exited; this implementer launched
no bounded-probe process. A shell process-list query here was denied by the
host, so no global process inventory is claimed. Do not restart the replay.

The sanitized peer bundle `cyax-0131-replay-20261001` reports 101m51s elapsed,
99m51s CPU, and 98.6% CPU before interruption. Its flushed log is
`replay-after-interrupt.log`, SHA-256
`a2b63069f15753a643f0b8b6c7f1b981d725500abb6939241b4bddb148680c1816`;
replay source SHA-256
`9d83d558011af4be3ff9fdb9afac9fefb20bba2da24d465f2ea6facdb9c1ce8c`;
relevant source SHA-256
`578a6b44877f82082f50699dedc2830bc2d34f4efedc93394e3b2b255ac9a564`.
Process-wide exit diagnostics: 112,801,083,728 allocations (112,801,082,254
pool; 1,474 big) and 3,137 GCs. Stack samples showed 159/181 and later
274/274 main-thread samples in GC; footprint reached 14.2 GB. The interrupt
trace locates the active work in `n8_author_trajectory`, Rosenbrock
`perform_step!`, LinearSolve generic LU, BigFloat arithmetic/allocation, and
GC. These observations support allocation/GC pressure during the ODE; they do
not establish a scientific result or full-run resource breakdown.

Flushed partial results: zero-phase N8 calibration completed; 400/400 row-2
phase steps and all 12 stationary points converged; N5 shifted-fold checks
completed; all seven P96 cross-check points converged. The physical trajectory
has no recovered terminal time, accepted/rejected-step counters, final
e-folds, or result. Its outcome is **UNESTABLISHED** because the owner
terminated the run, neither pass nor solver failure. Later gates were not
reached. The raw `replay.log` is retained locally and must not be force-added;
use sanitized records only.

## Diagnostic concern and pending work

At 100-bit precision, source formulas request `reltol=1e-50` and
`abstol=1e-66`; source also enables `save_everystep=true` and `dense=true`.
This precision/tolerance and retention combination is a concern to measure,
not an established root cause or permission to alter accepted settings. No
runtime epsilon ratio, short-probe timing, solver counter sample, warm-run
comparison, or precision comparison has been measured.

`trajectory_default_probe.jl` is present and was **not run**. Its SHA-256 is
`2ec9b6c5e6c1cc677cf5050d4913b407ba62ed5fa5b61e40d428c11ae1eb1965`. After
the owner explicitly resumes, first inspect/revalidate it, then run its two
bounded 100-bit default-tolerance probes (max_time 10, maxiters 1,000) with
the exact author route and no changes to source or accepted settings. Record
timing, allocations, GC, retcode and available solver stats; the author API
does not expose partial simulation time on an unsuccessful return. Capture
and sanitize output in a separate new validation file. Any later precision
comparison is diagnostic only; the API is BigFloat-based, so 64-bit BigFloat
must not be called Float64.

Resume command from repository root, only after explicit resume:

```sh
repo_root="$(git rev-parse --show-toplevel)"
p0_env="${repo_root}/validation/p0_numerical_equivalence/environment"
JULIA_LOAD_PATH="@:${p0_env}:@stdlib" \
  julia --startup-file=no --project=. \
  validation/cyax_0131_low_n_inflation/trajectory_default_probe.jl \
  > validation/cyax_0131_low_n_inflation/trajectory-default-probe.raw.log 2>&1
```

Do not resume the full replay. Preserve the r5 packet’s exact gates and
scientific conventions. Before any eventual freeze, update R-001–R-011
statuses, record all later fields as `NOT_REACHED` where applicable, sanitize
evidence, rerun required final diff checks, and include intended ignored
validation evidence explicitly.
