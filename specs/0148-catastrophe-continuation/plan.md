# Implementation Plan — CYAX-0148

## Governing specification

Canonical specification:
`specs/0148-catastrophe-continuation/spec.md`

Spec revision used to derive this plan:
`81ff48c460eec6d4d7dcbb87a739fddcdb0dd075` (draft brownfield migration
revision).

Approval state: **draft / owner approval pending**. This plan does not authorize
new scientific behavior while the governing S2 specification remains draft.

Migration context: CYAX-0151 G3 reconstructs Issue #148/PR #149 as SDD without
changing scientific intent. The migration base is
`vmm@6434e133af4b91db81babfafa024fc3cbada901a`; scientific PR #149 was frozen
for migration purposes at
`e991495f1bb58edd7a7043dfc90771b6666c717c`.

## Coverage

New requirement IDs describe the migrated current contract. Historical work is
mapped to them for traceability only; the IDs did not govern that historical
execution.

| Requirement / Gate | Current or planned realization | Verification / durable evidence |
| --- | --- | --- |
| R-001 Outcome-neutral objective | G3 investigates the fate of the radial catastrophe under one valid non-radial deformation; no persisted-curve requirement | Owner clarification: Issue #148 comment `5609698767`; final G3 evidence and independent review |
| R-002 Claim boundary | Preserve local/replayable continuation claims and explicit non-claims | Issue #148 claim boundary; G0 audit; per-gate review |
| R-003 Source/representation identity | Track paper radial `k`, period-one source coordinates, source12/P96 vs author10/A96, homotopy-only controls | `issue_148_g0_baseline_audit.md`; `issue_148_n8_metric_boundary_audit.md`; evidence provenance fields |
| R-004 N=5 source continuation/precision | Existing accepted G1 implementation in `poly102_inflation.jl` plus focused G1 scripts/tests | G1 tested code `792a02f...`; 63/63 focused review; 128/256-bit replay; manager decision `1eef936...` |
| R-005 P96 N=8 metric | P96 scientific path in N8 diagnostics; A96 retained as labelled reproduction | Owner comment `5623902670`; `issue_148_n8_approved_metric_contract.md` at `d1a0b70...`; like-for-like checks |
| R-006 Well-posed N8 continuation | `src/paper_benchmarks/n8_continuation.jl` plus radial evidence script; repaired bordered/fallback semantics are a current candidate, not yet accepted | First candidate independent FAIL `091cc9c...`; repair candidate `5b8daff...`; fresh independent re-review still required at migration cutoff |
| R-007 N8 independent validation/precision | Twelve-term augmented comparison, actual matcher comparison, status/conditioning/failure evidence, target-constructed 128/256-bit event solves | Current repair evidence `docs/src/issue_148_g2_repair_evidence.md`; acceptance depends on fresh independent review, not worker replay alone |
| R-008 Valid non-radial deformation | Future G3: cone-adapted/two-cycle parameterization, consistent recomputation of dependent quantities | Preflight geometry/Kahler checks; representative-point replay; independent review |
| R-009 Independent-control sensitivity | Future G3 local rank/sensitivity diagnostic before expensive continuation | Saved numerical sensitivity/rank evidence with source/geometry identity |
| R-010 Local discriminant fate | Future G3 continuation/local analysis from accepted radial catastrophe, allowing persistence/termination/splitting/class change/additional nullity | Reproducible local result plus independent checks at representative points/witnesses |
| R-011 Symmetry/higher derivatives | Future G3 zero-phase cubic test plus projected D3/D4, transverse Hessian/nullity diagnostics | Focused diagnostic replay and independent scientific review |
| R-012 Replayable provenance | Existing G0/G1/G2 evidence pattern; extend to G3 geometry/source/code/environment identity | Exact commands, revisions, source SHA, precision, tolerances, geometry/witness IDs in durable evidence |
| R-013 Scientific ambiguity stop | Manager/worker must stop on unresolved normalization/basis/acceptance/physical interpretation | Owner-decision record when needed; no silent implementation choice |
| R-014 Gate sequencing/evidence status | G0/G1 accepted; G2 earlier candidate rejected; repair candidate pending independent review; G3 blocked until G2 PASS | GitHub Issue #148 gate records + review artifacts; live state remains in GitHub |
| R-015 Negative outcomes valid | G3 completion is outcome-neutral within the claim/control boundary | Scientific review of whichever local discriminant outcome is observed |
| R-016 Optional physical probe separation | Optional G4 only after G3 PASS and without model change | Separate exploratory evidence clearly labelled non-core |
| G0 | Historical baseline and contract audit | PASS: `docs/src/issue_148_g0_baseline_audit.md`, audit script, Issue comment `5608088958` |
| G1 | Historical N=5 bug repair + genuine continuation + precision/failure validation | PASS: code `792a02f...`, evidence `668ef25...`, final independent acceptance and manager decision `1eef936...` |
| G2 | Current radial N8 continuation/validation work on PR #149 | First candidate FAIL; later repair candidate at `5b8daff...` with evidence through frozen head `e991495...`; fresh independent review required before any PASS |
| G3 | Future first non-radial discriminant investigation | Blocked on explicit G2 PASS; no G3 implementation present at migration cutoff |
| G4 | Optional bounded physical probe | Not started; conditional on G3 PASS and unchanged physical model |

## Existing architecture

### Scientific benchmark layer

`src/paper_benchmarks/poly102_inflation.jl`
: Contains the N=5 source benchmark/continuation path affected by G1 and the
  executable-author benchmark material. G1 corrected the N=5 critical-scale
  reuse defect and added accepted continuation behavior.

`src/paper_benchmarks/n8_continuation.jl`
: Contains the N=8 radial continuation machinery developed for G2, including
  pseudo-arclength/bordered-corrector and target-precision support. The current
  post-FAIL repair is not accepted merely because it exists in this file.

`src/paper_benchmarks.jl`
: Integrates benchmark submodules/functions into the package benchmark surface.

### Existing orchestration / comparison layer

`scripts/inflation_scale_continuation.jl`
: Contains the pre-existing scale-continuation/pilot behavior and the old
  post-hoc `pilot_match_records!` path used as a comparison target rather than
  as the new intrinsic continuation algorithm.

The G0 audit distinguishes the physical/source radial `k` from other numerical
homotopies. G2/G3 must not use homotopy-only controls as though they were the
source radial or Kahler coordinates.

### Focused evidence layer

- `scripts/audit_issue_148_g0.jl`
- `scripts/issue_148_g1_n5_regression_tests.jl`
- `scripts/issue_148_g1_replay_checks.jl`
- `scripts/audit_issue_148_n8_metric_boundary.jl`
- `scripts/issue_148_g2_continuation_evidence.jl`
- focused assertions in `test/runtests.jl` where appropriate

Durable narrative evidence and independent review records live under
`docs/src/issue_148_*.md`. Those records are revision-specific: later evidence
may supersede an earlier candidate's acceptance implication without erasing the
historical record.

## Brownfield state reconstruction

### Accepted historical work

#### G0

The baseline audit established source identity, scale/coordinate dictionaries,
existing radial/homotopy/matcher behavior, augmented-solver behavior, and the
N=5 defect classification. It made no scientific convention change.

#### G1

Accepted G1 corrected the N=5 source critical scale and demonstrated genuine
branch continuation, literal failure/event boundaries, and target-constructed
128/256-bit precision behavior. The accepted scope remains one known regular
zero-phase source-reduced N=5 branch; it does not establish N=8 or off-ray
claims.

#### P96 owner decision

The N=8 canonical metric boundary is resolved. Scientific N8 outputs use
period-one GLSM coordinates with `K_theta=M96/k^2`; raw-radian equivalence uses
`M96/[k^2(2pi)^2]`. Eq.96/CYTools matrix authority is deliberate and the
factor-two displayed-equation discrepancy remains documented. Author raw-radian
`M96/k^2` is A96 reproduction-only.

### Rejected and current G2 work

The first G2 implementation candidate (`9fe32eb...`) reproduced useful radial
facts but failed scientific acceptance because the evidence did not establish
the claimed mechanism: the accepted path could be reproduced with the bordered
corrector disabled, matcher evidence was synthetic, the precision ladder held
`k` fixed, canonical comparisons mixed term sets, and failure/status/tolerance
coverage was inadequate. The independent review and manager decision at
`091cc9c...` therefore supersede any candidate-language implying G2 acceptance.

A later repair implementation at `5b8daff...` and evidence recorded through the
migration-frozen PR head `e991495...` address those findings by reporting, among
other things:

- consistent transformed radial coordinate in the bordered system;
- explicit corrector/fallback/failure provenance and conditioning;
- a disabled-corrector trace that differs from the normal trace;
- actual use of the old `pilot_match_records!` on independently solved slices;
- target-constructed 128/256-bit augmented event solves;
- explicit 53-bit boundary for reconstructed P96 metric input;
- source12/author10 separation and like-for-like P96/A96 tensor-factor checks;
- preserved G1 focused replay.

This is **candidate repair evidence only** until a fresh independent reviewer
checks the repaired revision against the original G2 contract and the previous
minimum-correction list. The plan does not pre-adjudicate that review.

## Proposed approach

### Phase A — Finish G2 from the existing scientific branch

Continue G2 only on Issue #148 / PR #149's scientific deliverable branch. Do
not move scientific implementation onto the SDD migration branch.

1. Freeze the exact repaired revision for review.
2. Have a fresh independent scientific reviewer replay and inspect the repaired
   implementation against R-003/R-005/R-006/R-007/R-012/R-014 and G2.
3. Correct any concrete implementation/evidence defects found by that review on
   the same scientific deliverable branch.
4. Record the eventual gate decision durably on Issue #148. Only a durable PASS
   unblocks scientific G3.
5. If valid implementation evidence still leaves a normative catastrophe
   classification/interpretation choice, return only that remaining question
   to the scientific owner; do not change a cutoff to obtain a desired label.

The migration artifacts record this plan but do not themselves change PR #149
or adjudicate G2.

### Phase B — Prepare the G3 scientific slice after G2 PASS

Do not preselect a deformation or claim a catastrophe topology before G2 is
accepted. Once unblocked:

1. Identify the benchmark geometry and exact accepted radial catastrophe
   witness/revision.
2. Determine the independent Kahler-coordinate representation available for
   that geometry and the applicable cone/control checks.
3. Compute a local sensitivity/rank diagnostic mapping independent Kahler
   directions into the relevant instanton actions/amplitudes. Use it to select
   one genuinely non-radial direction rather than an arbitrary direction that
   may reproduce radial scaling.
4. Define a bounded cone-valid deformation range and recompute divisor volumes,
   total volume, kinetic data, instanton actions, potential coefficients and
   other dependent quantities consistently.
5. Test the zero-phase cubic normal-form coefficient locally before assuming
   any symmetry protection persists off ray.
6. Starting at the independently accepted radial catastrophe, use the
   appropriate augmented/local continuation analysis to determine its fate.
   Do not force a continuous curve if the equations instead terminate, unfold,
   split, change class, or gain nullity.
7. Independently verify representative points or termination/splitting
   witnesses, including stationarity/degeneracy, projected derivatives,
   transverse Hessian/nullity, cone/domain and EFT-control diagnostics.
8. Preserve replayable geometry/source/code/environment/precision identity and
   run focused regression plus applicable broader checks.
9. Obtain fresh independent scientific review before G3 acceptance.

The detailed G3 numerical algorithm remains intentionally open until G2 PASS
and the geometry/sensitivity preflight establish the actual local problem.
Technical choices may evolve without spec re-approval if R-001/R-008-R-012 and
the scientific claim boundary remain unchanged.

### Phase C — Optional G4 physical probe

Only after G3 PASS, and only if no physical-model change is needed, select a
small number of well-defined off-ray witnesses and apply the existing
catastrophic-inflation detuning/diagnostics. Keep the evidence labelled
exploratory and separate from the G3 discriminant claim.

## Alternatives considered

### Treat the original Issue body as the canonical specification

Rejected. Later durable owner decisions changed the precision requirements,
fixed the P96 N8 metric contract, and superseded the original G3 requirement
that a nontrivial catastrophe curve persist.

### Fold SDD migration commits into PR #149

Rejected. PR #149 is the scientific implementation/evidence branch. The
migration is a CYAX-0151 process deliverable based on current `vmm`; coupling
them would make SDD migration acceptance depend unnecessarily on the current
G2 repair and would blur historical evidence boundaries.

### Mark the repaired G2 candidate accepted because its local replay passes

Rejected. `AGENTS.md`, CYAX-0151 and the Issue #148 gate history require fresh
independent scientific review. Worker test counts are evidence, not gate
adjudication.

### Require a persisted off-ray catastrophe curve

Rejected as superseded. Owner clarification explicitly makes G3
outcome-neutral: persistence, termination, splitting/unfolding, class change or
additional nullity are all admissible scientific outcomes.

## Data / API / schema impact

- No persisted scientific schema change is planned or authorized.
- G1 corrected N5 scientific benchmark behavior and added continuation
  capability; no intentional compatibility break was approved.
- G2/G3 should preserve existing public/persisted interfaces unless the
  scientific task demonstrates a reviewed need for change.
- P96 scientific diagnostics and A96 author reproduction must remain explicitly
  distinguishable; do not silently change legacy A96 behavior.
- No dependency change is planned by this migrated specification.
- No feature-branch `Project.toml` version bump. Any reviewed package version
  bump belongs at the normal `vmm -> main` integration/release boundary.

## Verification strategy

Use progressive verification under `AGENTS.md`:

1. **G1 historical oracle:** analytic N5 source fixture and its accepted focused
   replay remain the regression baseline for the N5 changes.
2. **G2 source fixture:** published N8 source12 radial benchmark under the P96
   contract.
3. **Mechanism checks:** prove the continuation/matcher/precision mechanism
   being claimed, not just the final catastrophe number.
4. **Precision:** source-construct target precision; disclose source-precision
   boundaries such as the reconstructed 53-bit P96 metric input where
   applicable.
5. **Failure/conditioning:** retain truthful status, rejected-step, rank,
   condition and fallback behavior and test bounded failures.
6. **G3 geometry preflight:** validate cone-adapted coordinates, independent
   control rank, and all dependent physical quantities before expensive
   continuation.
7. **G3 local evidence:** stationarity/degeneracy plus D3/D4/transverse
   diagnostics and cone/EFT controls at representative points/witnesses.
8. **Focused regression first**, then applicable package/audit/docs/CI checks.
9. **Fresh independent scientific review** before G2/G3 acceptance. A worker's
   DONE state or a large passing-test count is insufficient.

## Migration / compatibility

This `spec.md`/`plan.md`/`tasks.md` set is a documentation/control-plane
migration. It does not move existing scientific commits between branches,
rewrite Issue #148 or PR #149 history, or declare the current G2 repair
accepted.

After the migrated S2 specification receives independent semantic-fidelity
review and durable scientific-owner approval, Issue #148 should receive a
concise pointer to the canonical spec. PR #149 may receive the same pointer.
The historical Issue/PR bodies remain intact.

Live PR draft/ready/merge state, Issue closure, Project status and current
review ownership remain GitHub state rather than fields to synchronize in
`tasks.md`.

## Risk and stop conditions

Stop and return to the governing specification/scientific owner if:

- two durable sources appear to contain conflicting normative intent whose
  precedence cannot be established from the chronology;
- a future repair/deformation requires changing P96, basis, physical
  normalization, catastrophe acceptance criterion, Kahler-coordinate meaning,
  reported observable or scientific schema;
- source12/P96 and author10/A96 cannot be kept like-for-like and explicitly
  distinguished in a claimed comparison;
- a proposed G3 deformation cannot be shown to supply independent Kahler
  control or satisfy the applicable cone/physical-domain checks;
- a desired scientific outcome would require weakening a gate or tuning a
  classifier/tolerance merely to obtain a preferred label;
- investigation changes the intended scientific behavior rather than only its
  technical realization.

A negative but well-controlled G3 result is not itself a reason to change the
specification.

## Task decomposition

`tasks.md` separates:

- historically accepted G0/G1 evidence;
- completed P96 owner-decision work;
- the current G2 repaired implementation/evidence candidate from the still
  required independent review/acceptance boundary;
- future G3 work blocked on G2 PASS;
- conditional G4 work.

Tasks end at observable implementation/evidence/review-readiness boundaries.
Current gate, PR merge and Project workflow state remain in GitHub.
