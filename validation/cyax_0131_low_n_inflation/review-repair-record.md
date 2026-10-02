# CYAX-0131 r5 fresh-review repairs

## Reviewed candidate and findings

The fresh SPEC review artifact has SHA-256
`b8d1b3829620d799502ad7e6a35e65bb561e1840d81cbc144e8be6403acc5005` and the
fresh SCIENTIFIC review artifact has SHA-256
`186542898117afdb654a50efdfe5e6623b2a9faf9802cc14b13b0abc4e255f97`. The
SPEC review reported one P3 table omission. The SCIENTIFIC review reported
two P2 gate defects. These dispositions apply to the prior reviewed commit
only; the repaired successor requires both fresh review axes.

The R-008 table now lists five deferred categories: `A_s` conversion and
acceptance window, `n_s` acceptance window, tensor-to-scalar `r`, the
observational stretch goal, and the full physical N5 trajectory.

Physical diagnostic helpers now require a completed author10 catastrophe
witness. They compare exact phase encodings, refined coordinates, raw-radian
coordinate convention, reconstructed-author-metric convention, canonical
Hessian basis, and critical `k` at the trajectory precision. A missing or
altered witness cannot produce `physical_k_reestablished` or sample
diagnostics. The row-2 refinement now lifts the exact returned Float64 phase
vector to its existing 128-bit precision and returns that represented vector
with the witness. No phase-equivalence tolerance was introduced.

An open `:tmax` slow-roll interval now returns only explicitly censored
exploratory duration, solver status, and counters. It has no completed
`N_e`, top-level entry/end coordinates, or samples. The refinement summary
classifies it as `:censored`; replay exits through its bounded-null path and
records all remaining `NOT_REACHED` fields. A completed finite-exit window is
required before sample diagnostics can pass the gate.

## Historical and repaired phase identities

The owner-terminated full replay and 100-bit probes used the pre-review
trajectory source SHA-256
`578a6b44877f82082f50699dedc2830bc2d34f4efedc93394e3b2b255ac9a564`, whose
128-bit row-2 refinement used decimal `BigFloat("0.04")` and returned
`critical_k=0.5080234603138255193960987179021912891477`. Those results remain
historical and are not merged with the repaired row-2 calibration. The full
run remains `OWNER_TERMINATED`, numerical outcome `UNESTABLISHED`; it was not
restarted.

The repaired trajectory source SHA-256 is
`f252962376de39784207f3d56f6db5271a0e80ac4649b383d10087a24af0dcaa`. The
input Float64 phase vector is `[0.0, 0.04, 0, 0, 0, 0, 0, 0, 0, 0]`; at 128-bit
precision the exact lifted phase vector is
`[0, 0.04000000000000000083266726846886740531772, 0, 0, 0, 0, 0, 0, 0, 0]`.
The focused run measured `critical_k=0.5080234603138255175832528546556562960302`,
gradient residual `1.926019943173470684443306798976058396723e-37`, null
residual `6.078716386972213161207930381283676849136e-34`, and minimum absolute
Hessian eigenvalue `1.648814618691634454827763903154989763316e-34`. All 400
phase increments and 12 stationary points converged. The existing two-sided
stationary-branch bracket is `[0.5080134603057957, 0.5080334603057957]`, width
`1.999999999990898e-5`; endpoint minimum-Hessian values are
`+9.202600373622222e-5` and `-9.212145745994419e-5`. No phase-equivalence
tolerance was introduced.

The earlier `n8_shifted_fold_audit.md` independently records the same
Float64-lifted 128-bit fold identity and retains its unsuccessful displaced
seed probes. The `replay-partial.sanitized.log` and
`trajectory-default-probe.sanitized.log` retain the older decimal-BigFloat
identity. Neither historical record has been rewritten to resemble the
repaired result.

## Verification on the repaired source

The final focused regression passed **45/45 assertions in 16.0 seconds** on
Julia 1.12.6. It checks phase provenance, the source-computed null residual against
the minimum absolute eigenvalue, the physical witness gate, open-window
suppression, and negative identity/status cases. Its positive completed
window is an explicitly synthetic gate fixture; no physical values are
attributed to it. Exact command:

```sh
JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_LOAD_PATH="@:@stdlib" \
  julia --threads=1 --startup-file=no \
  --project=validation/p0_numerical_equivalence/environment \
  validation/cyax_0131_low_n_inflation/focused_regressions.jl
```
Sanitized output is retained in
`focused-regressions-review-repair.sanitized.log`.

The same focused run's bounded **64-bit BigFloat** route case used Rodas5P,
`max_time=10`, `maxiters=1000000`, `reltol=1e-8`, and `abstol=1e-10`. Solver
retcode was `Success`; it accepted 41 steps, rejected zero, and reported 328
RHS and 41 Jacobian evaluations. It reached `:tmax` with `terminated=false`,
reported only censored duration `17.8028961864886779848`, and returned no
completed `N_e` or samples. This is a gate regression, not a physical
trajectory or observable.

The full Julia package suite passed **55 test summaries and 2,097/2,097
assertions** in the existing external test environment. The environment was
identified by Project SHA-256
`8e26eb84a345c68f887f719dfb6c99645ce30a70aa8a438a4aa6a6aea7331bf6` and
Manifest SHA-256
`b515317bcb0d126c9070f761135073ee1c6ff2463194ca3600aed3a53a997f43`.
Exact successful invocation:

```sh
repo_root="$(git rev-parse --show-toplevel)"
test_env_dir="$(python3 - <<'PY'
from pathlib import Path
import hashlib
project_sha = "8e26eb84a345c68f887f719dfb6c99645ce30a70aa8a438a4aa6a6aea7331bf6"
manifest_sha = "b515317bcb0d126c9070f761135073ee1c6ff2463194ca3600aed3a53a997f43"
for project in Path("/tmp").rglob("Project.toml"):
    try:
        project_hash = hashlib.sha256(project.read_bytes()).hexdigest()
        manifest = project.with_name("Manifest.toml")
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else ""
    except OSError:
        continue
    if project_hash == project_sha and manifest_hash == manifest_sha:
        print(project.parent)
        raise SystemExit(0)
raise SystemExit("approved external test environment not found by its recorded hashes")
PY
)"
JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 \
  julia --threads=1 --startup-file=no --project="${test_env_dir}" \
  -e 'using Pkg; Pkg.test(Pkg.PackageSpec(name="CYAxiverse", uuid=Base.UUID("e5e45d93-5055-4eab-878b-2e484be3f951"), path=ARGS[1]); allow_reresolve=false)' \
  "${repo_root}"
```

The full suite retained four existing Float64-seed truncation warnings and
two skips because real-data round-trip fixtures were unavailable. The first
suite attempt stopped after 207 assertions because a test fixture paired a
zero-phase route with the shifted witness; the exact gate correctly rejected
it. The test route was corrected to use the shifted phase, critical `k`, and
basis, then the full suite passed.

The repaired-source audit command `julia --startup-file=no --project=.
bin/audit.jl` passed. JET analyzed 1,263 definitions with no errors; Aqua and
the potential, derivative, and positive-definiteness checks passed. Audit
Project/Manifest SHA-256 values were
`ef6dabfa1ded67f26de09b9931c9b56969ef39a762af35da48efa78a4c7a67d9` and
`75be439157e143f111261868819fb99ca60a79bb4225c899f401e1094dfc4d51`.
Python-free import assertions also passed under the unchanged P0 environment:
candidate source binding was exact, Python variables were unset, and neither
PyCall nor its extension loaded. `git diff --check` and
`python3 scripts/agent_verify.py diff-check` passed with no whitespace errors.

## Frozen source and evidence hashes

| Artifact | SHA-256 |
|---|---|
| `scripts/inflation_diagnostics_common.jl` | `45e7f1b757dcb0d375a79687439855d1488639c7b9fa93d57533e3f671497acf` |
| `scripts/inflation_refinement_common.jl` | `1b5e20c2e02ccf8aaa549b9fc4fe7e35d5a9d4138de6929b1fefe94c0bd690d5` |
| `src/paper_benchmarks/poly102_inflation.jl` | `f252962376de39784207f3d56f6db5271a0e80ac4649b383d10087a24af0dcaa` |
| `test/runtests.jl` | `66f44688566d5c67702bedbfbf44a4f30656101fb663c67901f76f831dc0df93` |
| `focused-regressions-review-repair.sanitized.log` | `6f9479e60871a0c1610a87899427784c07edc838582401511a922c39bda80c77` |
| `full-tests-review-repair.sanitized.log` | `a899e28b71fc991519f3d6f9b8364eb65e3a1a1ce1cfdf7bc3f8b892a260291c` |
| `bin-audit-review-repair.sanitized.log` | `92171a6b5dbfe0513494cb385559c6d4c7cd5dcf22626db8684fc63c28c9426f` |
| `python-free-review-repair.sanitized.log` | `d7dc6cc59bb31f6247d3d1b0452c53cca3f8ce0514db2f48ccd5a97b83d54a4f` |

`EVIDENCE_SHA256SUMS.txt` contains hashes for all retained authorized
source/test and sanitized evidence files. Raw host-local logs are excluded.

The repaired source/test hashes are recorded in `EVIDENCE_SHA256SUMS.txt`.
Candidate `Project.toml` remains SHA-256
`38275eaf04c8f9ae28541542326f6c791c3a6a40d2a9ca3916088ca0652c368c`; no
candidate Manifest exists. Tracked P0 Project/Manifest SHA-256 values remain
`ef6dabfa1ded67f26de09b9931c9b56969ef39a762af35da48efa78a4c7a67d9` and
`a421591181011d15d08918f0f5e49a9f7537143a43de8bdec19506c702bccdcf`.
No 100-bit full replay was restarted, no physical observable is claimed, and
the repaired candidate still requires fresh independent SPEC and SCIENTIFIC
reviews.
