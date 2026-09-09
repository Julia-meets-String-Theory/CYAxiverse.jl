# Issue 148 G1 pause checkpoint — 2026-09-09

Paused at the user's request before a location change. Do not resume until the
user asks. G0 is accepted PASS. G1 is not accepted; the independent review of
`b35cb74781513601cc4079e37060bf75a3e39e0e` recommends FAIL with bounded
implementation corrections. G2 has not started.

The same Spark/xhigh implementation worker has begun a further correction.
The two changed files in this WIP checkpoint are
`src/paper_benchmarks/poly102_inflation.jl` and
`scripts/inflation_scale_continuation.jl`. Their changes are **unverified**;
the associated regression updates and evidence are not complete. The manager
ran `git diff --check`, which passed; no numerical acceptance follows from it.

The in-flight Julia run of `/tmp/check_n5_nonconv.jl` was interrupted during
precompilation, with exit 130/SIGINT (execution session 60667). The worker
confirmed it stopped editing and launching tests. There is no pending worker
lease and no automatic resume scheduled.

## Resume packet

- Objective: finish only G1's remaining corrections from the latest section of
  `issue_148_g1_independent_review.md`; keep the same deliverable branch.
- Acceptance: iteratively refine the continuation event and enforce gradient,
  Hessian, and scale tolerances; reject incorrectly identified satellite
  seeds; preserve source precision and mixed input types; validate finite
  controls and numerical nonconvergence; add tracked replay and exact evidence.
- Inputs: accepted G0 `62c5a61`, independent review record `317abd4`, this WIP,
  Issue 148, and draft PR 149.
- Constraints: source-reduced zero-phase N5 only; no N8/off-ray work or
  scientific/public-interface/schema change. Retain the later N8 metric owner
  boundary before canonical-distance or observable claims.
- Worker ownership: same Spark/xhigh worker owns diagnose/edit/test/correct/
  retest. Finish tests in `test/runtests.jl`, add a tracked replay script,
  commit code, run replay, then record the tested code SHA in evidence.
- Escalation: genuine source/convention/interface/owner boundaries only.
- Lease: on explicit resume, renew a 20-minute lease; one checkpoint and one
  evidence-based extension. After DONE, Sol/xhigh independently re-reviews
  without implementation ownership; manager then adjudicates G1.

The full-suite phase-volume assertion is independently proven pre-existing;
do not repair it in this scope. Documentation CI passed at `b35cb74`; other
CI and the new WIP must be checked on resume. Keep PR 149 draft.
