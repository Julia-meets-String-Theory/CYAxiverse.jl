# Implementation Plan — CYAX-NNNN

## Governing specification

Canonical specification: `specs/.../spec.md`

Spec revision reviewed for this plan: <commit/revision>

## Coverage

| Requirement / Gate | Planned implementation | Planned verification |
| --- | --- | --- |
| R-001 | ... | ... |
| G1 | ... | ... |

Every normative in-scope requirement should appear in the coverage mapping.

## Existing architecture

Identify only the current code/process paths and contracts the change touches.

## Proposed approach

Describe the technical/numerical design. For algorithms include representation, precision strategy, convergence/failure behaviour, resource considerations, and provenance/persistence implications as relevant.

## Alternatives considered

Record alternatives only when they materially affected the decision.

## Data/API/schema impact

...

## Verification strategy

Use progressive verification appropriate to the governing spec and `AGENTS.md`: focused analytic/synthetic checks, named source fixture, bounded replay, broader package/audit/docs/CI, and independent review where required.

## Migration / compatibility

...

## Risk and stop conditions

Identify implementation discoveries that require returning to the specification or owner rather than continuing autonomously.

## Task decomposition

Explain only non-obvious dependency structure. Detailed execution units live in `tasks.md`.
