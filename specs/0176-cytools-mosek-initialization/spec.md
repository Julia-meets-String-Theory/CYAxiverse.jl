---
spec_id: CYAX-0176
title: Correct CYTools and MOSEK initialization boundaries
issue: 176
class: S1
status: draft
workstream: optional CYTools integration
parent: null
depends_on: [157]
created: 2026-09-22
last_reviewed: null
review_required: Spec Reviewer + Standards Reviewer
approval_ref: N/A while draft
---

# Correct CYTools and MOSEK initialization boundaries

## Objective

Make the optional CYTools/PyCall integration obey one explicit initialization
contract while separating three states that current code conflates:

1. **CYTools readiness** — the configured Python interpreter is valid and the
   CYTools wrapper may be used;
2. **MOSEK activation state** — whether a fresh license check establishes usable
   MOSEK in the current process;
3. **optimizer/license consultation** — whether a particular downstream CYTools
   operation actually needs to inspect optimizer availability or select a
   backend.

The correction must remove eager/duplicate MOSEK side effects from extension
loading, invoke upstream activation APIs correctly, preserve supported
non-MOSEK workflows, and retain the privacy/interpreter guarantees introduced
by PR #158.

## Motivation

Issue #176 records that the existing optional integration does not match its
documented explicit-enable boundary:

- `ext/CYAxiversePyCallExt.jl::enable_cytools!()` performs MOSEK setup/checks;
- including `add_functions/cytools_wrapper.jl` runs its module `__init__()`,
  which performs the same setup/checks before `enable_cytools!()`;
- both helper functions return the `config.mosek_is_activated` callable rather
  than invoking `config.mosek_is_activated()`;
- the current code assumes `$HOME` is a valid argument to
  `config.set_mosek_path`, although CYTools expects an actual license-file
  path when an explicit override is supplied.

The repaired Issue #176 design was independently reviewed as PASS in the durable
Issue history. This specification migrates those reviewed decisions into the
repository's SDD hierarchy before source implementation.

## Current baseline

### Chosen implementation base

Source implementation is based on exact PR #158 head:

`8f6a9c28ac4b778b07f244dbbbc0076c86dc1a45`

PR #158 supplies the required privacy/interpreter behavior:

- `src/python_interpreter.jl` exists and validates optional
  `CYAXIVERSE_PYTHON` against the effective PyCall interpreter;
- no automatic `Pkg.build("PyCall")` occurs;
- `CYAxiversePyCallExt.enable_cytools!()` calls
  `CYAxiverse.python_interpreter.check_configured_python(PyCall.python)`.

This specification is intentionally stacked on that exact head. If PR #158
changes or equivalent interpreter/privacy guarantees are adopted on another
base, implementation must rebind and re-verify before mutation.

### Current initialization defect on that base

At PR #158 head:

- `ext/CYAxiversePyCallExt.jl` blob
  `ea9006a50ba9d5796abc86fc542a371c9c8cac4d`;
- `add_functions/cytools_wrapper.jl` blob
  `0b19cee9b9bff5a3570e3ff2680c71eaa9d45362`;
- `src/python_interpreter.jl` blob
  `85c03c862feb66f8f8457b11cd2494bc253bd410`.

The extension still performs MOSEK setup inside `enable_cytools!()`, and the
wrapper still performs eager MOSEK setup in `__init__()`.

## Scope

This specification includes:

- the optional PyCall extension initialization state;
- CYTools wrapper initialization;
- MOSEK license-path handling and activation checks;
- explicit distinction between CYTools readiness and MOSEK activation;
- downstream operation classification by optimizer/license consultation;
- recovery after failed/inactive MOSEK activation;
- focused tests for load/enable/wrapped-call/solver-boundary behavior;
- exact rebinding to the privacy/interpreter guarantees from PR #158.

## Non-scope

This work does not authorize:

- changing core `using CYAxiverse` to depend on Python or CYTools;
- automatic rebuilding of PyCall;
- making MOSEK an unconditional package prerequisite;
- changing mathematical/scientific algorithms or accepted physics results;
- changing persisted HDF5 scientific schemas;
- broad CYTools geometry/database scans;
- changing PR #158's privacy policy or public-surface remediation contract;
- closing Issue #157 or merging PR #158;
- package-version/release-boundary changes.

## Fixed contract

### State separation

The implementation must preserve the following distinctions.

#### CYTools disabled

Before explicit `enable_cytools!()` succeeds:

- CYTools-backed wrapper operations are not available;
- no MOSEK activation/license check occurs as a side effect of extension or
  submodule loading;
- no wrapper operation may silently self-enable.

#### CYTools enabled with active MOSEK

A fresh activation check establishes MOSEK as available.

State label for evidence:

`ENABLED_ACTIVE`

#### CYTools enabled with inactive/failed MOSEK license

CYTools import/readiness is valid but a fresh MOSEK activation check is false.

State label for evidence:

`ENABLED_INACTIVE_LICENSE_FAILED`

This state does **not** invalidate CYTools readiness. Workflows for which CYTools
supports a non-MOSEK backend may continue using that supported backend.

#### Restart required

If a prior inactive/cached state cannot be refreshed reliably after license
configuration changes, the implementation must not claim that MOSEK became
active.

State label:

`RESTART_REQUIRED`

The error must tell the user to restart/re-enable rather than silently retain a
stale activation claim.

## License-path semantics

CYAxiverse must not synthesize `$HOME` as a MOSEK license path.

- With no explicit override, use CYTools' normal license discovery.
- If CYAxiverse supplies an explicit MOSEK override to CYTools, that value must
  be the actual license-file path expected by the CYTools path setter/input.
- The implementation may choose an engineering interface for supplying this
  override, but it must be explicit, documented, testable, and must not publish
  or persist private machine-local paths.
- A custom path must not become durable/public provenance.

## Workflow capability matrix

The implementation must maintain a tested matrix with separate fields for:

- operation/workflow;
- whether CYTools readiness is required;
- whether optimizer/license state may be consulted directly or transitively;
- selected/eligible optimizer backend;
- whether MOSEK is mandatory for that operation;
- supported fallback, if any;
- failure behavior when required capability is unavailable.

The matrix must include at least the following paths.

### Basic CYTools readiness

Examples such as version/query operations may require CYTools but must not make
MOSEK activation a package-wide precondition unless the specific upstream
operation actually needs it.

### Fast triangulation paths

Fast-generation behavior must be tested for the actual upstream calls it uses.
Do not infer optimizer requirements merely from the enclosing module.

### Fair triangulation paths

`random_triangulations_fair` and associated regularity checks must be traced
transitively. If upstream automatic selection consults MOSEK activation before
choosing another supported backend such as HiGHS, record that as
**consultation**, not as "MOSEK mandatory".

### Stored-simplices reconstruction

`cy_from_poly(...)` calls `triangulate(simplices=...)`. Any transitive
regularity/optimizer validation performed by that upstream call must be included
in the capability matrix.

### Standard geometry generation

`geometries_generate` calls `tip_of_stretched_cone`; optimizer capability
must be classified from the actual CYTools behavior. Inactive optional MOSEK
must not block the workflow when a supported alternative backend is valid.

### Hilbert workflow

The following operations are distinct:

1. `hilbert_basis(rays)` computes a Hilbert basis through CYTools/Normaliz;
2. `hilbert_save(geom_idx, basis)` persists a caller-supplied basis to HDF5;
3. `geometries_generate_hilbert` reconstructs a CY from stored simplices and
   stored geometry data, including an already stored tip.

The contract must not attribute `tip_of_stretched_cone` to
`geometries_generate_hilbert`: that path uses the stored tip. It still may
consult optimizer state transitively through stored-simplices reconstruction,
so the whole path must be traced rather than assumed solver-free.

## Deliberate pre-enable policy for `hilbert_save`

Although `hilbert_save` is persistence-only at the function-body level, the
reviewed Issue #176 design intentionally keeps all functions exposed through
the CYTools-backed wrapper behind the explicit enable boundary.

Therefore:

> `hilbert_save` SHALL reject before successful `enable_cytools!()`.

Before enable, the rejected call must perform:

- zero HDF5 writes;
- zero optimizer/license checks;
- zero implicit enable attempts.

After CYTools enable, `hilbert_save` itself must not require MOSEK activation
because its operation is persistence-only.

## Requirements

### R-001 — Core package remains Python-free

`using CYAxiverse` without loading PyCall must not import Python, CYTools, or
MOSEK and must not require any Python environment.

### R-002 — Extension load has no MOSEK activation side effect

Loading the optional PyCall extension and its submodules must not call:

- `set_mosek_path`;
- `check_mosek_license`;
- `mosek_is_activated()`;

and must not write CYTools/HDF5 data as an initialization side effect.

### R-003 — One explicit CYTools readiness boundary

`enable_cytools!()` is the explicit transition from disabled to CYTools-ready.

It must:

- validate the PR #158 interpreter contract;
- import/initialize the required CYTools wrapper objects exactly once;
- not call `Pkg.build`;
- cache only state that is safe to cache;
- be idempotent after successful enable.

### R-004 — MOSEK activation is independent state

Successful CYTools readiness must not require active MOSEK.

The implementation must record/test `ENABLED_ACTIVE`,
`ENABLED_INACTIVE_LICENSE_FAILED`, and `RESTART_REQUIRED` distinctly.

### R-005 — Invoke activation API correctly

Where activation state is checked, call the upstream activation function and use
its boolean result.

The implementation must not treat the `mosek_is_activated` callable object as
the activation result.

### R-006 — Correct license override semantics

No `$HOME` assumption is permitted.

An explicit override, when used, must be an actual license-file path supplied to
the upstream CYTools mechanism. With no override, CYTools normal discovery must
remain available.

### R-007 — Supported fallback is not failure

If an operation consults MOSEK state but supports an eligible non-MOSEK backend,
inactive MOSEK alone must not cause the operation to fail.

If an operation explicitly requires MOSEK or no supported fallback exists, the
failure must identify the missing solver/license capability at the operation
boundary.

### R-008 — Transitive optimizer consultation is observable

The capability matrix and tests must cover direct and transitive consultation,
including fair triangulation and stored-simplices reconstruction.

### R-009 — Hilbert path attribution is exact

The implementation/tests must distinguish `hilbert_basis`,
`hilbert_save`, and `geometries_generate_hilbert` as specified above.

### R-010 — `hilbert_save` pre-enable contract

Before enable, `hilbert_save` must reject with zero HDF5 writes and zero
optimizer/license consultation.

After enable, its persistence-only behavior must not require active MOSEK.

### R-011 — Recovery after inactive activation

A failed/inactive MOSEK check must not permanently poison CYTools readiness.

When license configuration is corrected:

- if a fresh activation state can be established safely in-process, record the
  fresh result;
- if the upstream state cannot be refreshed reliably, return
  `RESTART_REQUIRED` and require process restart/re-enable.

Do not promote stale cached state to `ENABLED_ACTIVE`.

### R-012 — Focused phase-specific diagnostics

Failures must identify which phase failed:

1. extension/submodule load;
2. explicit CYTools enable/interpreter/import;
3. MOSEK activation/configuration;
4. downstream wrapped CYTools operation/backend selection.

### R-013 — Privacy/interpreter base is preserved

The implementation must retain the exact PR #158 interpreter/privacy guarantees
or a separately reviewed equivalent.

Any material change to PR #158 head before implementation requires rebind and
fresh verification.

## Acceptance gates

### CYAX-0176 G0 — S1 contract review

**Acceptance:**

- independent Spec Review has no blocking finding;
- independent Standards Review has no blocking finding;
- exact reviewed spec/plan/tasks are bound to the chosen implementation base.

A passing review permits **preparation only** of a fresh exact Manager-facing
implementation handoff bound to the reviewed S1 bytes and the exact
privacy/interpreter-safe base. It does not authorize dispatch, source mutation,
merge, or Issue closure.

### CYAX-0176 G0.5 — Reviewed implementation handoff and owner dispatch

**Objective:** preserve the Control Desk execution boundary between reviewed S1
intent and source mutation.

**Acceptance:**

- a fresh exact Manager-facing implementation packet is prepared from the
  reviewed S1 contract and exact PR #158/equivalent base;
- that packet receives independent comprehensive Handoff Review with no blocking
  finding;
- Control Desk reconciles the unchanged reviewed packet to `READY_TO_DISPATCH`;
- the owner manually dispatches that exact packet to Manager;
- Manager re-runs exact preflight before any source mutation.

**Stop condition:** packet/base drift, unresolved Handoff Review finding, missing
owner dispatch, or any need to change reviewed S1 intent.

G0.5 does not authorize merge or Issue closure. Control Desk does not launch
Manager.

### CYAX-0176 G1 — Initialization boundary

**Acceptance:**

- extension load is solver/license-side-effect-free;
- `enable_cytools!()` is the single explicit CYTools readiness boundary;
- repeated enable is idempotent;
- core CYAxiverse import remains Python-free.

### CYAX-0176 G2 — Solver/license state and recovery

**Acceptance:**

- activation helper is invoked correctly;
- no HOME license-path assumption remains;
- active/inactive/restart-required states are distinguished;
- supported fallback behavior is verified;
- corrected-license recovery follows the reviewed state machine.

### CYAX-0176 G3 — Workflow matrix

**Acceptance:**

Focused tests establish the consultation/backend/fallback contract for the
required operation matrix, including fair triangulation, stored-simplices
reconstruction, geometry generation, Hilbert operations, and one deterministic
wrapped basic CYTools call.

### CYAX-0176 G4 — Exact-candidate independent review

The exact implementation candidate must receive independent Spec and Standards
review with no blocker before Control Desk acceptance.

## Verification requirements

At minimum verify separately:

1. core `using CYAxiverse` without PyCall;
2. loading CYAxiverse + PyCall and resolving the extension without explicit
   enable;
3. pre-enable wrapper rejection, including zero-write `hilbert_save`;
4. successful `enable_cytools!()` with interpreter check;
5. repeated enable idempotence;
6. CYTools-ready + inactive MOSEK state;
7. active MOSEK state where an appropriate test environment is available;
8. supported non-MOSEK fallback;
9. explicit/mandatory MOSEK failure behavior;
10. corrected-license refresh or deterministic `RESTART_REQUIRED`;
11. `cytools_version()` as a deterministic basic wrapped call;
12. fair-generation and stored-simplices consultation behavior;
13. Hilbert-basis, Hilbert-save, and stored-tip Hilbert-generation distinctions;
14. package/core tests and `scripts/agent_verify.py diff-check`;
15. privacy scan showing no machine-local path is published as durable evidence.

Environment-specific solver absence is not itself a test failure when the
contracted outcome is an inactive/fallback/restart state.

## Interfaces and compatibility

The optional extension contract may gain bounded configuration/state helpers,
but core CYAxiverse public behavior must remain Python-independent.

No persisted scientific schema or numerical scientific convention is changed.

## Dependencies and blockers

Implementation is stacked on PR #158 head
`8f6a9c28ac4b778b07f244dbbbc0076c86dc1a45`.

This dependency is on the **interpreter/privacy guarantees**, not on Issue #157
being closed. A later rebased/merged equivalent may replace the pinned head only
after exact rebind establishes equivalent guarantees.

## Open owner decisions

None are intentionally required for the reviewed r5 design direction.

If implementation discovers that a workflow requires making MOSEK universally
mandatory, changing scientific/numerical semantics, changing PR #158 privacy
policy, or introducing a materially different state model, stop for owner
decision and re-review.

## Completion criterion

CYAX-0176 is complete when the reviewed S1 contract has passed G0 and G0.5,
has been implemented on an exact privacy/interpreter-safe base, G1-G3 evidence
is satisfied, exact candidate Spec/Standards review has no blocker, and Control
Desk reconciles the result.
