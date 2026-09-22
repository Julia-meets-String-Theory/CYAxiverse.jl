# CYAX-0182 Stage-0 implementation plan

Status: draft with the exact `spec.md`; implementation is gated on independent Spec Review. Scientific Review is `NOT_APPLICABLE` because this is review infrastructure only.

## Dependency order

| Gate | Contract | Planned artifact and evidence | Exit condition |
| --- | --- | --- | --- |
| P0 | Scope/authority | Live two-repository head/tree and Issue hashes; accepted r4 review/reconciliation; governing source blob/hash checks; pinned-source file manifest; absent spec and allowlist collisions | `PRECHECK_BOUND` or `BLOCKED_FOR_REBIND` before source edits |
| S1 | R-01–R-08 | Exact public `spec.md`, `plan.md`, `tasks.md`; independent Spec Review record under adopted rubric | No blocking Spec finding on exact bytes |
| I1 | R-02–R-04 | New private `prototypes/review_runtime_v0/rule-inventory-v1.json`, detached hash/identity, independent Standards Review | Exact inventory reviewed with no blocking finding before fragment-consuming assembler code |
| I2 | R-01–R-07 | New private `cyax_exchange/review_runtime_v0/**`, standalone `tools/review_runtime_v0.py`; versioned compiler/fact and manifest identities | Deterministic non-mutating compile and clean reconstruction |
| V3 | R-01–R-08 | New focused test/fixture paths; existing validator/test suite; CYAxiverse `git diff --check` and `scripts/agent_verify.py diff-check`; immutable-source proof | Exact commands, exit codes, counts, warnings, and attributable failures recorded |
| E4 | R-08 | New private replay fixtures/reports with paired fresh non-authorizing request/state | Required outcome classes and controlled byte/token measurements, with reconciled differences |
| R5 | All | One frozen two-repository candidate/evidence set; independent Spec and Standards reviews | No blocking finding; exact identities and residual limits handed back |

## Technical approach

The compiler accepts explicit exact inputs; it does not discover governing policy by interpreting prose. Preflight validates V0 request and state through read/import interfaces, checks the accepted TASK_AUTHORITY issuer and binding, checks finite currentness roots, and resolves only inventory-reviewed source fragments. The module and frame plan is frozen before task/evidence bytes enter serialization. Recoverability is checked for every byte-bearing input. The emitter writes canonical frames and RFC 8785 manifest bytes; a separate clean reconstruction path consumes only manifest recovery data. Local output remains outside production publication paths.

The first slice consumes durable exact state bytes/fixtures. If correctness needs a state materializer or an edit to any pinned pre-existing exchange file, stop with the packet's rebind reason rather than changing the technical contract. All exchange output paths are new files under the five reviewed locations. The pinned source file manifest is the write guard and the final preservation proof.

## Requirement-to-verification map

| Requirement | Focused verification | Evidence to preserve |
| --- | --- | --- |
| R-01 exact request/state | Valid comprehensive request; wrong scope, status, request/candidate/revision/scope/check/lineage, pre/post-result and conflicting/missing state fixtures | Exact request/state/validator identities and test results |
| R-02 authority/modality | Valid Control Desk bound closure; candidate/self-asserted/unbound/wrong issuer/source mismatch; `UNKNOWN`/`CONFLICT`; modality preservation | TASK_AUTHORITY fixture identities, inventory fragments, failure outputs |
| R-03 currentness | Three terminal roots; stale/superseded unchanged bytes; unknown/conflicting roots; discovery-only rejection | Root/evidence identities and adversarial fixture results |
| R-04 modules/inventory | Fixed Handoff-only module plan, dependency/context closure, excluded reviewer-role instructions, reviewed inventory hash equality | Frozen inventory bytes, independent review and exact compiler input identity |
| R-05 recovery | Three closed modes, strict Base64, mutable/unrecoverable and privacy-rejected inputs | Recovery fixture hashes and clean recovery result |
| R-06 frames | Emit/parse round trip; fake `@FRAME`/`@END`, Markdown/XML/JSON/role labels, multibyte UTF-8, CRLF, bad order/ID/ordinal/hash/length/terminator/trailing data | Runtime/frame hashes and focused test results |
| R-07 manifest | RFC 8785 vectors, no newline, all identity fields, clean manifest-only reconstruction | Canonical manifest hash, compiler/validator identities, reconstructed runtime hash |
| R-08 non-mutation/replay | Instrumented absence of writes/launch; existing suite; allowlist/source preservation; paired outcome-class replay | Commands, counts, warnings, replay controls and forensic differences |

## Review and handback

The independent S1 Spec Reviewer receives exact spec/plan/tasks bytes, accepted r4 architecture identity, r2 handoff, execution bases, permitted/forbidden paths, and adopted rubric identity. The I1 Standards Reviewer receives exact frozen inventory bytes and source identities before assembler implementation. Final Spec and Standards reviewers receive one exact frozen cross-repository candidate, all test/reconstruction/replay evidence, and the same source/authority bindings. Reviews remain distinct; changed normative or implementation bytes receive fresh review on affected axes.

The Manager return includes exact start/final commits and trees; changed paths; exact spec, inventory, compiler, manifest, frame, and state identities; requirement evidence; commands and observed outcomes; replay controls and measurements; independent review verdicts/findings; and explicit residual limitations. No success claim may imply production adoption, merge, protocol adoption, or Issue closure.
