# CYAX-0182 Stage-0 execution tasks

Status: draft execution decomposition. This file is evidence readiness, not live PR, merge, Project, review ownership, or Issue status.

| ID | Gate | Observable outcome | Verification / escalation |
| --- | --- | --- | --- |
| T-01 | P0 | Exact packet, READY record, two heads/trees, live Issue, architecture, governance, source identities and 547-path pinned exchange manifest checked | Hash/blob equality and no allowed-path collision. Material drift: `BLOCKED_FOR_REBIND`. |
| T-02 | S1 | Public `spec.md`, `plan.md`, `tasks.md` translate accepted r4 without new normative intent | Freeze exact bytes and get independent Spec Review. Blocking finding: bounded repair and fresh review. |
| T-03 | I1 | Lossless new private rule inventory with exact source/fragment/dependency/context/deferral identities | Freeze and obtain independent Standards Review. Unreviewed inventory cannot feed assembler. |
| T-04 | I2 | New private V0 request/state and TASK_AUTHORITY/currentness preflight modules | Positive and adversarial R-01–R-03 tests. Materializer/authority change: stop/rebind. |
| T-05 | I2 | New private reviewed-inventory-bound module planner, recovery validator, frame emitter/parser, versioned Stage-0 manifest schema, JCS manifest and reconstructor | Schema identity and independent review, validation and invalid-schema tests, integer-domain checks, R-04–R-07 tests and clean byte-identical reconstruction. Existing-source edit: stop/rebind. |
| T-06 | V3 | Focused and existing exchange suites pass or baseline failures are isolated; public spec diff checks pass | Record exact commands, exit codes, counts, warnings and source-preservation proof. |
| T-07 | E4 | Controlled paired historical replay across required outcome classes, with distinct cases relying on Spec/Standards evidence and Scientific evidence if each is available | Fresh shared request/state per case; frozen reviewer prompt/template and substantive inputs; exact baseline prompt, modular runtime and manifest identities; same model family/version where controllable, same reasoning effort, equivalent tool configuration; byte/token measurements, environment differences and forensic reconciliation. Historical authority never marked current. |
| T-08 | R5 | Exact two-repository candidate independently reviewed on Spec and Standards axes | Repair bounded defects and re-review changed bytes; unresolved architecture returns to owner. |
| T-09 | Return | Frozen evidence-bearing Manager handback | Status `READY_FOR_CONTROL_DESK_HANDOFF_REVIEW`, `REQUEST_CHANGES`, or `BLOCKED_FOR_OWNER_DECISION`, with all r2 return fields and non-authority statement. |

Scientific Review: `NOT_APPLICABLE` for the authorized infrastructure-only slice. Any scientific semantics discovered in this work stops for rebind.
