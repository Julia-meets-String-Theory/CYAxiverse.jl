# Distributed test worker coverage

Status: S1 bounded engineering contract
Governing issue: #207
Baseline: `fb8139ca18cc5a91c59c6b7d693e5174abb3bc27` (`vmm`)

## Requirements

- Identify the coverage-writing subprocess by matching observed worker PIDs to
  per-source coverage filenames after executing the existing Full-only vacua
  persistence test, including its dispatched job and worker shutdown.
- Permit the test to pass additional Julia worker flags through the batch runner
  to `addprocs`. Keep the default worker flags and CLI behavior unchanged.
- Use the existing test coverage helper to detach worker coverage outside the
  checkout. Preserve the inherited coverage filter and scientific/job behavior.
- Assert that the batch test adds no per-source coverage files after workers
  exit. Verify with coverage enabled and disabled, then the Full package suite.
- Allow a `full-tests` PR label to opt into Full CI on development-line PRs;
  retain the existing automatic Full triggers.

## Boundaries

This is test/process infrastructure with patch-level impact; no package version
is adopted. Numerical algorithms, scientific acceptance, persisted data, and
the `vmm -> main` release boundary are unchanged.

## Verification

Record baseline/candidate reproduction and Full-suite results in the PR. Local
coverage artifacts are diagnostic output, not tracked source. The follow-up
shared-helper consolidation remains issue #206, after #207 closes.
