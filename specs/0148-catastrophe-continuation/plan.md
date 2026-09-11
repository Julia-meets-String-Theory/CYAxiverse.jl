# Implementation Plan — CYAX-0148

## Governing specification

Canonical specification:
`specs/0148-catastrophe-continuation/spec.md`

Spec revision used to derive this plan: originally
`eff7798ca65f932a0c6cf635bca7f49f4c5d4517` (P96-fidelity-corrected brownfield
migration revision); updated to incorporate post-cutoff G2 and G3 acceptance
evidence from PR #149.

Approval state: **draft / owner approval pending**. This plan does not authorize
new scientific behavior while the governing S2 specification remains draft.

Migration context: CYAX-0151 G3 reconstructs Issue #148/PR #149 as SDD without
changing scientific intent. The migration base is
`vmm@6434e133af4b91db81babfafa024fc3cbada901a`; scientific PR #149 was
originally frozen for migration purposes at
`e991495f1bb58edd7a7043dfc90771b6666c717c`, with post-cutoff accepted evidence
through `f33ba768f53fee749352a82c04aedc09919d1053` (G3 PASS).

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
| R-005 P96 N=8 metric | P96 scientific path in N8 diagnostics; `M96` reconstructed in the relevant GLSM basis with reconstruction/source identity verified at working precision; any other-basis representation uses explicit metric congruence transformation; A96 retained as labelled reproduction | Owner comment `5623902670`; `issue_148_n8_approved_metric_contract.md` at `d1a0b70...`; working-precision reconstruction/source-identity checks; explicit basis-transform checks where applicable; like-for-like P96/A96 checks |
| R-006 Well-posed N8 continuation | `src/paper_benchmarks/n8_continuation.jl` plus radial evidence script; repaired bordered/fallback semantics accepted at `cc8ac73...` | First candidate FAIL `091cc9c...`; repair candidate `5b8daff...`; further repair `cc8ac73...`; independent review and G2 PASS at `f9b04ed...` |
| R-007 N8 independent validation/precision | Twelve-term augmented comparison, actual matcher comparison, status/conditioning/failure evidence, target-constructed 128/256-bit event solves | `issue_148_g2_final_independent_review_cc8ac73.md`; `issue_148_g2_final_repair_evidence.md`; G2 PASS `f9b04ed...` |
| R-008 Valid non-radial deformation | G3 two-cycle parameterization `t=sqrt(k)(t_ref+alpha*u)` with `u=(0,1,2,-1,1,1,1,1)`, consistent recomputation of all dependent quantities | `issue_148_g3_control_audit.md`; `issue_148_g3_repair_evidence.md`; G3 PASS `f33ba76...` |
| R-009 Independent-control sensitivity | G3 rank diagnostic: rank two for `[log(k),alpha]` action controls and full-log-amplitude controls; off-ray action derivative has relative residual `0.317` after best radial fit | `audit_issue_148_g3_controls.jl`; `issue_148_g3_control_audit.md`; G3 independent review `f33ba76...` |
| R-010 Local discriminant fate | G3 determines persistence along one positive-alpha branch; 46 stored states through `alpha≈1.435e-4`, `k≈0.5007`; stops at scale guard; opposing-seed failure inconclusive | `issue_148_g3_repair_evidence.md`; `issue_148_g3_final_independent_review_4abd9c3.md`; G3 PASS `f33ba76...` |
| R-011 Symmetry/higher derivatives | G3 tested zero-phase cubic; not protected along this control; projected D3/D4, transverse Hessian/nullity diagnostics reported | `issue_148_g3_repair_evidence.md`; G3 independent review `f33ba76...` |
| R-012 Replayable provenance | Existing G0/G1/G2 evidence pattern; extend to G3 geometry/source/code/environment identity | Exact commands, revisions, source SHA, precision, tolerances, geometry/witness IDs in durable evidence |
| R-013 Scientific ambiguity stop | Manager/worker must stop on unresolved normalization/basis/acceptance/physical interpretation | Owner-decision record when needed; no silent implementation choice |
| R-014 Gate sequencing/evidence status | G0/G1 accepted; G2 first candidate rejected, later repair accepted (PASS `f9b04ed...`); G3 accepted (PASS `f33ba76...`) | GitHub Issue #148 gate records + review artifacts; live state remains in GitHub |
| R-015 Negative outcomes valid | G3 completion is outcome-neutral within the claim boundary | Scientific review of whichever local discriminant outcome is observed |
| R-016 Optional physical probe separation | Optional G4 only after G3 PASS if G3 yields a suitable off-ray catastrophe locus and the existing physical model can be used unchanged | Separate exploratory evidence from selected points along that locus; otherwise G4 is N/A |
| G0 | Historical baseline and contract audit | PASS: `docs/src/issue_148_g0_baseline_audit.md`, audit script, Issue comment `5608088958` |
| G1 | Historical N=5 bug repair + genuine continuation + precision/failure validation | PASS: code `792a02f...`, evidence `668ef25...`, final independent acceptance and manager decision `1eef936...` |
| G2 | Radial N8 continuation/validation on PR #149 | PASS: first candidate FAIL `091cc9c...`; repair at `5b8daff...`; further repair `cc8ac73...` accepted after fresh independent review; durable acceptance `f9b04ed...` |
| G3 | First non-radial discriminant investigation on PR #149 | PASS: implementation `4abd9c3...`, evidence `c6d2291...`, fresh independent review, durable acceptance `f33ba76...` |
| G4 | Optional bounded physical probe | Not started; applicable because G3 yielded a suitable off-ray catastrophe locus; requires that the existing physical model be usable unchanged |

## Existing architecture

### Scientific benchmark layer

`src/paper_benchmarks/poly102_inflation.jl`
: Contains the N=5 source benchmark/continuation path affected by G1 and the
  executable-author benchmark material. G1 corrected the N=5 critical-scale
  reuse defect and added accepted continuation behavior.

`src/paper_benchmarks/n8_continuation.jl`
: Contains the N=8 radial and off-radial continuation machinery developed for
  G2 and G3, including pseudo-arclength/bordered-corrector and target-precision
  support. The accepted G2 revision is `cc8ac73...`; the accepted G3 revision
  is `4abd9c3...`.

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
`M96/[k^2(2pi)^2]`. `M96` is the precise reconstructed Eq.96/CYTools reference
metric in the relevant GLSM basis, and its reconstruction/source identity must
be verified at working precision. Any representation in another basis must use
the corresponding explicit metric congruence transformation. Eq.96/CYTools
matrix authority is deliberate and the factor-two displayed-equation
discrepancy remains documented. Author raw-radian `M96/k^2` is A96
reproduction-only.

### G2 candidate history and acceptance

The first G2 implementation candidate (`9fe32eb...`) reproduced useful radial
facts but failed scientific acceptance because the evidence did not establish
the claimed mechanism: the accepted path could be reproduced with the bordered
corrector disabled, matcher evidence was synthetic, the precision ladder held
`k` fixed, canonical comparisons mixed term sets, and failure/status/tolerance
coverage was inadequate. The independent review and manager decision at
`091cc9c...` therefore supersede any candidate-language implying G2 acceptance
for that revision.

A repair at `5b8daff...` addressed those findings, and a further repair at
`cc8ac73668ac488a492dd16008c0b790a4e4ef3b` received fresh independent
scientific review and was accepted. The durable G2 PASS decision at
`f9b04ed74bc30cc8e0071fcbe18179a4f85016e2` explicitly supersedes the earlier
FAIL for the older revision. The accepted G2 scientific boundary: bounded
zero-phase, fixed-saxion, source-twelve radial N8 one-null degeneracy under
P96; positive transverse modes; projected classifier `:unresolved`; no
numerically resolved quartic-cusp label or stronger catastrophe classification.

### Accepted G3 work

After G2 PASS, the G3 investigation established geometric control and
sensitivity (`b562780...`), implemented bounded local-control continuation
(`1c57ed2...`), underwent an initial independent review identifying corrections
(`17d28b9...`), and was repaired at `4abd9c31aba1d387bac27769730c4e0dc7d0c4a0`
with evidence at `c6d2291e5c90c8890f48a9574343c14808a5a507`. Fresh independent
scientific review accepted the result, with durable G3 PASS at
`f33ba768f53fee749352a82c04aedc09919d1053`.

The accepted G3 scientific boundary: the source-twelve, zero-phase, fixed-saxion
P96 degeneracy persists from the radial event onto one positive-alpha branch on
the audited two-cycle slice `t=sqrt(k)(t_ref+alpha*u)` with
`u=(0,1,2,-1,1,1,1,1)`; 46 stored states reaching approximately
`alpha=1.435e-4`, `k=0.5007`; stops at the imposed scale guard (persistence to
a numerical boundary, not physical termination); opposing-seed failure does not
establish absence of another branch; zero-phase cubic cancellation tested and
not protected along this control; all stored states retain one near-null
canonical mode and positive transverse spectrum. No global continuation,
exhaustive-branch, G4, inflation, stabilized-moduli, population, negative-alpha,
or resolved G2 cusp claim is established.

## Proposed approach

### Phase A — Finish G2 from the existing scientific branch (completed)

Phase A was completed on PR #149. The repaired implementation at `cc8ac73...`
received fresh independent scientific review and was accepted at `f9b04ed...`.
No owner decision was required. The projected higher-derivative diagnostic
remained `:unresolved` and was not forced to a cusp label. The accepted G2
boundary is recorded in the brownfield state reconstruction above.

### Phase B — G3 scientific slice after G2 PASS (completed)

Phase B was completed on PR #149. The G3 investigation followed the planned
approach: geometry/sensitivity preflight at `b562780...`; bounded local-control
continuation at `1c57ed2...`; initial independent review identifying corrections
at `17d28b9...`; repair at `4abd9c3...`; updated evidence at `c6d2291...`; fresh
independent scientific review and acceptance at `f33ba76...`. No owner decision
was required. The accepted G3 boundary is recorded in the brownfield state
reconstruction above.

### Phase C — Optional G4 physical probe (applicable, not started)

G3 PASS yielded a bounded off-ray catastrophe locus (46 stored states along the
positive-alpha branch), so G4 is applicable in principle. G4 is not started.

Where applicable, and only if no physical-model change is needed, select a
small number of well-defined points along the validated off-ray catastrophe
locus and apply the existing catastrophic-inflation detuning/diagnostics. Keep
the evidence labelled exploratory and separate from the G3 discriminant claim.

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

### Mark a G2 candidate accepted without fresh independent review

Rejected at the migration cutoff. The eventual G2 PASS was obtained only after
a fresh independent scientific reviewer accepted the further-repaired revision
`cc8ac73...`. Worker test counts are evidence, not gate adjudication.

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
   contract, including working-precision verification of the reconstructed
   `M96` metric/source identity and explicit metric congruence transformation
   for any other-basis representation.
3. **Mechanism checks:** prove the continuation/matcher/precision mechanism
   being claimed, not just the final catastrophe number.
4. **Precision:** source-construct target precision; disclose source-precision
   boundaries such as the reconstructed 53-bit P96 metric input where
   applicable. The current G2 repair plan uses 128/256-bit stages as an
   implementation choice, not as immutable canonical intent.
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
migration. It does not move existing scientific commits between branches or
rewrite Issue #148 or PR #149 history. It records the post-cutoff G2 and G3
acceptance decisions from their durable PR #149 evidence.

After the migrated S2 specification receives independent semantic-fidelity
review and durable scientific-owner approval, Issue #148 should receive a
concise pointer to the canonical spec. PR #149 may receive the same pointer.
The historical Issue/PR bodies remain intact.

Live PR draft/ready/merge state, Issue closure, Project status and current
review ownership remain GitHub state rather than fields to synchronize in
`tasks.md`.

## Risk and stop conditions

G0–G3 are now accepted. For any future work (optional G4 or extensions), stop
and return to the governing specification/scientific owner if:

- two durable sources appear to contain conflicting normative intent whose
  precedence cannot be established from the chronology;
- a future extension requires changing P96, basis, physical normalization,
  catastrophe acceptance criterion, Kahler-coordinate meaning, reported
  observable or scientific schema;
- source12/P96 and author10/A96 cannot be kept like-for-like and explicitly
  distinguished in a claimed comparison;
- a desired scientific outcome would require weakening a gate or tuning a
  classifier/tolerance merely to obtain a preferred label;
- investigation changes the intended scientific behavior rather than only its
  technical realization.

## Task decomposition

`tasks.md` separates:

- historically accepted G0/G1 evidence;
- completed P96 owner-decision work;
- the G2 candidate history (rejected first candidate, accepted repair) and
  its completed independent review/acceptance;
- completed G3 scientific work and its accepted independent review;
- conditional, locus-applicable G4 work (not started).

Tasks end at observable implementation/evidence/review-readiness boundaries.
Current gate, PR merge and Project workflow state remain in GitHub.
