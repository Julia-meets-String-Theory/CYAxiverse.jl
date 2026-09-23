# Implementation Plan — CYAX-0176

## Governing specification

Candidate specification:
`specs/0176-cytools-mosek-initialization/spec.md`

Implementation is stacked on exact PR #158 head
`8f6a9c28ac4b778b07f244dbbbc0076c86dc1a45`.
No source mutation may begin until **both** CYAX-0176 G0 and G0.5 pass:
the exact S1 contract must clear Spec+Standards review, then a fresh exact
Manager-facing implementation handoff must clear Handoff Review, be reconciled
`READY_TO_DISPATCH`, and be manually dispatched by the owner.

## Coverage

| Requirement / Gate | Planned implementation | Planned verification |
| --- | --- | --- |
| R-001 | Preserve optional extension boundary | Core Python-free import test |
| R-002 | Remove eager MOSEK side effects from wrapper initialization | Load extension and assert no activation calls/messages |
| R-003 | Consolidate CYTools readiness under `enable_cytools!()` | enable/idempotence/basic wrapped call tests |
| R-004-R-007 | Separate solver state/config/fallback | state-machine + fallback fixtures |
| R-008 | Instrument/classify transitive consultation | operation capability matrix tests |
| R-009-R-010 | Distinguish Hilbert paths and pre-enable policy | targeted Hilbert tests |
| R-011 | Implement refresh/restart behavior | corrected-license/restart fixture |
| R-012 | Phase-specific errors | failure-message assertions |
| R-013 | Preserve PR #158 guarantees | exact-base/diff/privacy checks |
| G1-G3 | Focused execution evidence | exact commands/results |
| G4 | Independent exact-candidate review | Spec + Standards verdicts |

## Existing architecture

Touched implementation surfaces are expected to be bounded primarily to:

- `ext/CYAxiversePyCallExt.jl`;
- `add_functions/cytools_wrapper.jl`;
- focused optional-integration tests;
- only the smallest additional helper needed for explicit solver state/config.

`src/python_interpreter.jl` is inherited from PR #158 and should remain
unchanged unless a separately reviewed interpreter-contract defect is found.

## Proposed approach

1. Rebind exact PR #158 head and privacy/interpreter files.
2. Freeze the reviewed S1 contract and, after G0 passes, prepare a fresh exact
   Manager-facing implementation handoff bound to those reviewed bytes and the
   exact PR #158/equivalent base.
3. Obtain comprehensive Handoff Review of that exact packet, reconcile the
   unchanged passing packet to `READY_TO_DISPATCH`, and wait for owner manual
   dispatch. Control Desk does not launch Manager.
4. Manager re-runs exact preflight. Only then begin source mutation.
5. Remove MOSEK setup/license calls from wrapper `__init__()`.
6. Make `enable_cytools!()` establish CYTools readiness without requiring
   active MOSEK.
7. Introduce a small explicit solver-state/configuration layer that:
   - never assumes HOME;
   - calls `mosek_is_activated()` rather than returning the function object;
   - distinguishes enabled-active, enabled-inactive-license-failed, and
     restart-required;
   - permits supported upstream fallback.
8. Guard all CYTools-wrapper public entry points behind the enable boundary,
   including `hilbert_save`, while ensuring a rejected pre-enable
   `hilbert_save` performs zero writes.
9. Build an operation capability matrix from executed/inspected upstream call
   paths rather than name-based inference.
10. Test direct and transitive solver consultation:
   fair triangulation, stored-simplices reconstruction, standard geometry
   generation, and Hilbert paths.
11. Freeze the implementation candidate and obtain independent Spec/Standards
   review.
12. Return to Control Desk; do not merge automatically.

## License-path strategy

Default: do not call a CYTools custom path setter merely to guess a location.

If a bounded explicit override interface is added, pass an actual license-file
path to CYTools. The implementation detail may be a keyword/config helper, but
must be documented and tested without persisting private path text.

## Recovery strategy

Treat upstream activation as potentially cached.

After an inactive result, a later configuration change may attempt only an
upstream-supported fresh check. If a fresh in-process state cannot be proved,
return `RESTART_REQUIRED`.

## Data/API/schema impact

No scientific persisted schema change.

No core package dependency on Python.

Optional-extension helper signatures may be extended in a backward-compatible
way when needed for explicit license configuration/state reporting.

## Verification strategy

Use mocked/controlled Python-side state for deterministic activation-state tests
where possible, plus one real CYTools smoke environment for actual import and
basic wrapped-call evidence.

Do not require a live MOSEK license for the full test suite. Tests must verify
the inactive/fallback behavior explicitly.

Run focused optional-path tests first, then appropriate package checks and
`scripts/agent_verify.py diff-check`.

## Migration / compatibility

No data migration.

Existing users with valid default CYTools/MOSEK discovery should continue to
work after explicit enable.

Users without MOSEK must retain supported non-MOSEK CYTools functionality.

## Risk and stop conditions

Stop/rebind if:

- G0.5 is not satisfied before source mutation;
- the reviewed implementation packet or its exact base drifts before execution;
- PR #158 base changes materially;
- implementation requires weakening interpreter/privacy guarantees;
- actual CYTools behavior contradicts the reviewed operation matrix assumptions;
- a scientific algorithm would need to change;
- a public API break is required;
- upstream has no reliable way to refresh an inactive activation state and the
  proposed behavior would claim otherwise.

## Task decomposition

One integration worker should own initialization/state changes and focused tests.
A separate bounded investigation worker may trace upstream operation/backend
consultation, but shared-file mutation remains serialized.

Final reviewers are independent of implementation workers.
