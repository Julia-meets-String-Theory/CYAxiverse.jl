Historical pause record; work resumed and final G2 acceptance is recorded in
`issue_148_g2_final_manager_decision.md`.

Issue 148 G2 repair pause record (2026-09-10)

Phase: bounded implementation repair after independent review FAIL. Worktree is at clean HEAD e1b35fb before repair edits; no new commit made. Current uncommitted edits are limited to src/paper_benchmarks/n8_continuation.jl and scripts/issue_148_g2_continuation_evidence.jl.

Changes in progress:
- Corrected the scaled-radial bordered system: Jacobian correction now updates k through k_arc_scale and the arc residual uses the same transformed coordinate.
- Added per-step corrector_method, bordered_rank, bordered_condition, and rejected_steps records; fallback and failure provenance are recorded.
- Added exact-source BigFloat P96 diagnostic with explicit metric_source_precision_bits=53 boundary.
- Began replacing the fixed-k precision ladder with target-constructed 128/256 BigFloat augmented event solves and unconditional assertions.
- Began loading the existing inflation_scale_continuation script so the actual pilot_match_records! can be exercised.

Checks: git diff --check passed after edits. Before these edits, the candidate replay at 9fe32eb passed 321/321 but was rejected by review for fixed-k fallback dominance, synthetic matcher records, fixed-k precision ladder, hidden Float64 canonical widening, weak status records, and mixed P96/A96 comparison. No Julia replay has been run after the current edits.

Outstanding defects / next action:
- Complete evidence script with independently solved adjacent pilot slices and branch_match_id comparison; log all discrepancies.
- Add genuine accepted bordered-step and disabled-corrector/fallback/status regression assertions.
- Finish like-for-like ten-term P96/A96 tensor-factor checks and source12/author10 separation.
- Run focused replay, diagnose failures, commit coherent code/tests, then write revision-specific evidence with tested SHA.
Making progress: yes; paused by user request before a new long test loop.
