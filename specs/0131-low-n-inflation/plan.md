# Implementation Plan — CYAX-0131

## Governing specification

`specs/0131-low-n-inflation/spec.md`

The specification is currently **draft**. This plan is subordinate to it.
The owner-approved scientific contract authorizes drafting and review only.
No scientific execution or production implementation is authorized until the
exact S2 specification is finally approved by the owner and a new,
current-schema implementation handoff is separately reviewed and dispatched.

## Coverage and gates

| Requirement / gate | Planned work | Required evidence or verification |
| --- | --- | --- |
| R-001 / CYAX-0131 calibration | Use the source-authoritative 10-row author trajectory model and identify `author_inflation.n8_author_trajectory(...)` plus returned samples | Exact source revision, route, row count, model identity, and sample provenance |
| R-002 / CYAX-0131 calibration | Apply `phase[2] = 0.04` radians with all other 10-row phases zero; identify both critical points on the relevant sides and refine the catastrophe in the same model | Phase vector and units; source-consistent point identities; refined-location evidence; acceptance criteria from the approved implementation handoff |
| R-003 / CYAX-0131 cross-check | Run/report the separate 12-row P96/Table-1 continuation only as a cross-check | Separate model labels, row count, source route, and result identity; no certification of the 10-row model |
| R-004 / CYAX-0131 N5 replay | Use the reduced two-cosine light-direction model, set the second cosine to `π/4`, and re-solve/refine the shifted fold | Model and phase identity plus re-solved fold evidence; absent an explicit full-map source, mark the full eight-row mapping `NOT_VERIFIABLE` |
| R-005 / CYAX-0131 physical gate | Keep homotopy `k` in discovery/calibration; independently re-establish physical `k` before physical trajectory diagnostics | Distinct recorded scale categories and evidence for the physical-k re-establishment |
| R-006 / CYAX-0131 diagnostics | After physical-k re-establishment, report only `N_e`, `n_s`, `paper_delta_H` scalar amplitude, and cumulative turning through the concrete N8 trajectory/sample route; label as the 10-row author trajectory using the raw-radian coordinate convention and identify the rounded/reconstructed author metric where applicable | Per-quantity sample/source identity, units and diagnostic provenance; implementation handoff pins the source field/interval and window for `N_e` plus its sample/index identity; no new acceptance window |
| R-007 / CYAX-0131 catastrophe diagnostic | Attach transverse-Hessian eigenvalues to a named catastrophe point with its model, phase, coordinate, scale, metric/basis, and source/code identity | Point identity and Hessian basis; explicit statement that values are neither pivot-scale nor along-trajectory spectra |
| R-008 / CYAX-0131 scope boundary | Keep the five observational items `NOT_REACHED` | Explicit scope/status record for each item; no optimization result |
| R-009 / CYAX-0131 claim boundary | State fixed-saxion effective-theory scope in every result summary | Claim review against prohibited population, stabilization, string-cosmology, and full-KS language |
| R-010 / CYAX-0131 currentness | Preserve integrated Eq. 19 and Issue #172 correction ancestry | Current-source/diff review confirms the `q·tau`/cross-coefficient repair and `4π²` ancestry remain intact |
| R-011 / contract gate | Obtain specialist review, reconciliation, exact owner approval, then a new implementation handoff before execution | Same-candidate independent SPEC and SCIENTIFIC reviews; final exact-content owner approval; separate reviewed handoff |

## Execution stages

### Stage 0 — Prepare and review this S2 contract

1. Draft `spec.md`, `plan.md`, and `tasks.md` using only the owner-approved
   scientific contract and bound source/currentness records.
2. Check that every requirement R-001–R-011 maps to a plan row and task.
3. Run document-only checks, freeze one exact candidate, and obtain fresh
   independent SPEC and SCIENTIFIC reviews of that same candidate.
4. Repair only findings that clarify or faithfully express existing approved
   choices. If a finding requires a new scientific choice, stop and return it
   to the owner. Any normative-byte repair requires both specialist reviews
   again on the same revised candidate.
5. Return the exact reviewed candidate for Control Desk reconciliation and
   separate final owner approval. Drafting, review, or reconciliation alone
   does not authorize numerical work.

### Stage 1 — Establish a separately reviewed implementation handoff

After exact final S2 owner approval, rebind the implementation packet to the
then-current source tree, Issue/dependency state, current schemas, and exact
execution capability. Submit that handoff for its own review and dispatch.
The implementation handoff must carry any numerical acceptance criteria,
solver tolerances, witness identity details, permitted command surface, and
output schema required for the authorized implementation. Before `N_e` is
reported, it must pin which source field or interval defines the reported
value and window, plus the sample/index identity used. The source distinguishes
total `efolds`, `slow_roll_efolds`, and sample `n`; this plan selects none of
them. This plan does not fill in scientific choices that the owner has not
approved.

### Stage 2 — Conditional calibration and bounded discovery

Only to the extent explicitly authorized by the new implementation handoff:

1. Reproduce the N8 author model's zero-phase calibration and its source
   identity.
2. Apply the row-2 `0.04` radian phase benchmark and establish source-consistent
   two-sided critical-point identities and the refined catastrophe location in
   that same 10-row model.
3. Record the 12-row P96 continuation separately as a cross-check.
4. Re-solve the N5 reduced two-cosine `π/4` phase-shifted fold. Do not create a
   full eight-row phase vector without explicit source authority.
5. Classify all homotopy `k` results as discovery/calibration evidence. Before
   any physical trajectory diagnostic, independently re-establish physical
   `k` under the reviewed handoff's convention and evidence rules.
6. After the physical-k gate and after the implementation handoff pins the
   `N_e` field/interval, window, and sample/index identity, report only the
   permitted diagnostics. Label them as the 10-row author trajectory, retain
   the raw-radian coordinate convention, and identify the rounded/reconstructed
   author metric where applicable. Name the catastrophe point for
   transverse-Hessian eigenvalues.
7. Keep all five observational items `NOT_REACHED` unless a later, separate
   owner-approved specification governs them.

## Traceability and evidence rules

- Preserve the distinction between source facts, implementation facts,
  empirical evidence, owner-approved conventions, and inference.
- Identify each replay by source revision, model route and row count, phase
  vector and units, k category, coordinate and metric/basis, named witness,
  units, code revision, and relevant environment/tool versions.
- Do not use matching aggregate counts as a substitute for matching model and
  witness identities.
- Do not invent an N5 eight-row phase mapping, numerical tolerance, physical
  normalization, observable, population definition, or scientific acceptance
  criterion.
- Keep the integrated Eq. 19 repair and Issue #172 `4π²` normalization
  ancestry intact.

## Data, API, and version impact

This document-preparation tranche creates only the three S2 documents. It
does not change Julia APIs, production code, persisted scientific schemas,
dependencies, or package version. Any later implementation handoff must state
its own API/schema/version impact before execution.

## Verification strategy

For the current tranche, verify only the documents and their scoped diff:

- required three paths exist as newly created files;
- every R-001–R-011 requirement has a consistent plan and task mapping;
- phase values/row counts, model separation, k hierarchy, diagnostic set,
  deferred observational items, and fixed-saxion boundary agree across all
  three documents;
- no prose authorizes execution before exact owner approval and the new
  implementation handoff;
- `git diff --check` passes; and
- the public diff contains only the three requested files.

Numerical execution and scientific verification belong to a later authorized
implementation stage and are not run for this plan preparation.
