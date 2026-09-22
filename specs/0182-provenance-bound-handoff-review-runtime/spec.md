# CYAX-0182 Stage-0 provenance-bound Handoff Review runtime

Status: **draft, pending independent Spec Review**
Risk class: **S1 bounded infrastructure engineering**
Scientific Review: **NOT_APPLICABLE**. This slice changes review infrastructure only; it cannot change Julia behavior, scientific meaning, public APIs, or persisted scientific data.

## Authority and provenance

This specification translates the accepted [Issue #182 architecture](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/182) into an implementation contract. The accepted r4 Issue snapshot has SHA-256 `de1db39823e04380464940e54916121f215cded9e1b2d906ccd9c15018dcbafe` (65,124 UTF-8 bytes). Its architecture review result has SHA-256 `848c2a4ac6a1b4976aca629e2a2c6096429d0b41d37d16bafeab4404325341eb`, verdict `PASS_WITH_NONBLOCKING_FINDINGS`, applicability `CURRENT`. Control Desk accepted that architecture in `handoff-reviews/cyax-0182-modular-review-runtime-architecture/r4/review-001/evidence/control-desk-reconciliation-v1.json` in the exchange repository. The r2 Manager handoff is SHA-256 `b9f308756b88a78a251339a46b5a48d1a7744de02e64ba74068d92a01944b2f2` and is the execution boundary for this slice.

Issue #182 is intake and architecture history. This document is the S1 governing feature contract. It incorporates the accepted r4 architecture by exact identity. If a clause here appears to weaken, extend, or contradict the accepted r4 text, implementation stops for rebind; this document does not silently supersede r4.

The governing bases are CYAxiverse `vmm` commit `995163f0058488ea183ac645045ed8b1636bef4a` and exchange `main` source snapshot `6fbe1fc2141ff5abfa7791e2d3fb2e13a5d73b4c`. Later review artifacts on exchange `main` do not change that source snapshot. The adopted Spec/Standards rubric is SHA-256 `418f2d5a276cbdb74b8ad331b532d59219ef33b4e9fabc2b3d4a21c55bc06c72`.

## Goal and boundaries

Build a deterministic, non-mutating prototype that assembles **only** an ordinary-Chat V0 **Comprehensive Handoff Reviewer** runtime from exact authoritative fragments and exact task/evidence inputs. It must emit a canonical framed runtime and a content-bound manifest from which a clean environment can reconstruct identical bytes. It performs no substantive LLM review, publishes no result, invokes no Review Desk or downstream agent, and confers no review, dispatch, merge, adoption, rollout, or Issue-closure authority.

The public CYAxiverse change is limited to this `spec.md`, `plan.md`, `tasks.md`, and `evidence.md`. Private implementation is create-new only in the exchange repository under `cyax_exchange/review_runtime_v0/**`, `tools/review_runtime_v0.py`, `tests/test_review_runtime_v0.py`, `tests/fixtures/review_runtime_v0/**`, and `prototypes/review_runtime_v0/**`. Every file that existed at exchange source snapshot `6fbe1fc2...` remains byte-identical. In particular, the V0 validator, schemas, protocol, ordinary-Chat procedure, rubrics, profiles, existing tools/tests, and historical review artifacts are read-only. A required edit to any existing source path blocks this slice.

## Requirements

### R-01 — Exact supported request and preflight

The compiler shall consume an exact validated `CYAX-HANDOFF-REVIEW/0` request, exact candidate and normative references, exact task/evidence objects, and exact durable `CYAX-HANDOFF-REVIEW-STATE/0` bytes. It shall accept only `review.scope=comprehensive`; other scopes return `UNSUPPORTED_STAGE0` with no runtime. Discovery is locator metadata only. The pinned V0 state schema (`23518a43f9fb98dc969615168f696afc0c551b8112e849b8783fb28b1b9166a6`) and validator version `0.3.2` / implementation SHA-256 `71ea92ddcb3dc7a55884d5adf1c9477073fd3f7bc8e145117da5c206bd39e01d` establish the active request and candidate. Invalid, missing, stale, or conflicting state blocks emission.

Before a result exists, accepted state must have `active_request.status=PUBLISHED`, `publication_complete=true`, matching `active_request_sha256`, exact candidate handoff ID/revision/SHA-256, exact scope/checks/lineage, `candidate_current=true`, `request_current=true`, `selected_result_sha256=null`, `result_not_superseded=false`, and validated `exchange_state=PENDING`. `current_applicability=NOT_APPLICABLE` is valid before result publication. Requiring `APPLICABLE` before a result exists is incorrect. No deterministic state materializer is part of this slice.

### R-02 — Task authority and source modality

The compiler shall require a content-pinned `CYAX-TASK-AUTHORITY/0` closure whose sole accepted issuer class is `CONTROL_DESK_REVIEW_REQUEST`. The artifact must be an immutable normative reference of the exact active Control Desk request (`sender_role=control-desk`, `recipient_role=reviewer`) and match handoff ID, revision, exchange ID, candidate SHA-256, governing source identities, and validated active state. `COMPLETE` may continue; `UNKNOWN` or `CONFLICT` blocks. Candidate/evidence self-assertions or unbound connector claims cannot establish completeness. The compiler checks declared bindings; it does not infer omitted relevant skills/specs from prose.

Every instruction fragment retains exact source provenance and original `MUST`/`SHOULD`/`MAY` modality. Compiler predicates inspect only already-governed structured facts and return exactly `TRUE`, `FALSE`, `UNKNOWN`, or `CONFLICT`. Unresolved substantive applicability, authority, or currentness yields no runtime. No compiler predicate creates or changes policy.

### R-03 — Finite currentness roots

Each currentness proof must terminate at `LIVE_GIT_REF`, `LIVE_GITHUB_TARGET`, or `VALIDATED_V0_ACTIVE_STATE`. The first binds repository instructions, skills, specs, and exchange policy beneath a live canonical ref; the second binds mutable target state such as the live Issue; the third alone establishes exact active review request/candidate currentness. Content identity and governance currentness are separate. `STALE`, `UNKNOWN`, `CONFLICT`, unavailable roots, recursive registry authority, or unchanged predecessor bytes after canonical supersession block emission. Discovery cannot establish currentness. The manifest records terminal-root identities and decisions.

### R-04 — Fixed modules and reviewed inventory

Only these Handoff-facing instruction modules may be emitted: `HANDOFF_CORE`, `EXCHANGE_V0`, `ORDINARY_CHAT_PROCEDURE`, `TASK_AUTHORITY`, `COMPREHENSIVE_PROFILE`, `SUPPORTING_REVIEW_EVIDENCE`, and `PUBLICATION`; `OUTPUT_CONTRACT` is the separate final frame class. The comprehensive profile is required for the accepted scope. Supporting Spec/Standards/Scientific evidence may be checked for identity, currentness, applicability, and sufficiency, but the compiler cannot run those reviewer roles or load their full role policies as executable instructions. No `SPEC_REVIEW`, `STANDARDS_REVIEW`, or `SCIENTIFIC_REVIEW` module exists.

Before assembler code consumes extracted rules, a lossless rule-to-module inventory shall be frozen to exact bytes and independently reviewed on the Standards axis. Each operative fragment records repository/revision/path, source Git blob/SHA-256/byte length, deterministic locator, fragment SHA-256, source modality, module, dependencies and surrounding context, duplication/inheritance, independent extractability, enabled/disabled reason, and whether omission blocks. Classification includes the seven emitted modules plus `OUTPUT_CONTRACT`, `DEFERRED_ROLE_POLICY`, `DEFERRED_LESSON`, and `TASK_ONLY`. The inventory is a compiler input, never authority. Changed inventory bytes require new independent review before consumption.

### R-05 — Closed input recoverability

Each serialized source, task, evidence, or preflight input must use exactly one mode: `IMMUTABLE_REPOSITORY` (repository, immutable commit, relative path, Git blob, SHA-256, bytes, type), `IMMUTABLE_SNAPSHOT` (immutable publication/retrieval and content identity, locator, SHA-256, bytes, type), or privacy-permitted `EMBEDDED_BYTES` (strict bounded RFC 4648 Base64, decoded SHA-256, bytes, type). Exact bytes must be independently recoverable. Mutable URLs, UI/connector-only state, ephemeral output, and live GitHub evidence without an immutable snapshot or permitted exact embedding block emission. The compiler does not create snapshots or publish data.

### R-06 — Exact framed runtime

The sole runtime framing identity is `CYAX-REVIEW-RUNTIME-FRAMES/0`. The byte stream is UTF-8, begins with `CYAX-REVIEW-RUNTIME-FRAMES/0\n`, has no BOM, and uses LF only for framing. Each frame is exactly `@FRAME <ordinal> <class> <logical-id> <utf8-byte-length> <sha256>\n` followed by exactly that many unchanged UTF-8 payload bytes and `\n@END <same-ordinal>\n`. The parser consumes by byte length, never searches payload for delimiters, and accepts no trailing bytes. At least one frame is required.

Ordinals match `[1-9][0-9]*`, begin at 1, and are contiguous without leading zeros. Logical IDs match `[a-z0-9][a-z0-9._-]{0,127}`, are compiler assigned, ASCII sorted, and globally unique. Lengths match `0|[1-9][0-9]*` and obey versioned exact-integer/resource bounds. Hashes are lower-case `[0-9a-f]{64}` SHA-256 of exact payload bytes. Classes are only `INSTRUCTION`, `TASK_STATE`, `EVIDENCE`, `PREFLIGHT`, `OUTPUT_CONTRACT`. Canonical class order is `INSTRUCTION`, `TASK_STATE`, `PREFLIGHT`, `EVIDENCE`, `OUTPUT_CONTRACT`. Instruction module order is `HANDOFF_CORE`, `EXCHANGE_V0`, `ORDINARY_CHAT_PROCEDURE`, `TASK_AUTHORITY`, `COMPREHENSIVE_PROFILE`, `SUPPORTING_REVIEW_EVIDENCE`, `PUBLICATION`; within a module or other class, logical IDs sort by ascending ASCII bytes. Invalid UTF-8, CRLF, order, ordinal, ID, hash, length, terminator, or trailing byte rejects the entire runtime without recovery parsing.

Authority roots, module/dependency plan, frame classes/IDs/order, and output contract freeze before untrusted task/evidence payload serialization. This gives structural isolation; it does not claim complete semantic prompt-injection immunity.

### R-07 — Canonical manifest and reconstruction

The Stage-0 manifest shall be a JSON value that validates against a versioned, independently reviewed Stage-0 manifest schema whose exact identity is bound into implementation evidence. It is serialized **only** with RFC 8785/JCS to UTF-8 JSON without extra whitespace or trailing newline. Its detached SHA-256 covers exact canonical bytes. Ordinals, byte lengths, and other integer identity fields remain in the exact interoperable domain required by that reviewed schema; implementations may neither round nor reformat them. Changing canonicalization requires new reviewed format identity. It binds format/framing/ordering versions; task authority and currentness roots; compiler implementation, predicate table, reviewed inventory; source fragments and dependencies; every runtime input and its recovery information; every frame identity/linkage; validator and deterministic facts; and final runtime byte length/SHA-256. The manifest never becomes an authority source.

A clean environment must reconstruct every bound input using only its manifest recovery information and immutable artifacts, reassemble byte-identical runtime output, and verify the final hash. A final hash comparison without clean reconstruction does not satisfy this requirement.

### R-08 — Non-mutation, verification, and replay

Compiler execution may emit local/stdout/test artifacts but shall make no durable repository, GitHub, exchange, or Review Desk write and shall launch no downstream agent. Focused verification must cover successful and adversarial frames, multibyte UTF-8 and fake delimiters, task-authority mismatches, pre/post-result state and currentness failures, all recovery modes/privacy rejection, RFC 8785 canonicalization, inventory dependency/context completeness, clean reconstruction, and absence of publication/launch. Existing exchange tests and validator suite must run. Every exchange write in the implementation branch must be absent from the pinned source file manifest and inside the create-new allowlist; all pre-existing bytes must remain identical.

Historical replay is evidence only. Cases span `PASS`, `PASS_WITH_NONBLOCKING_FINDINGS`, `REQUEST_CHANGES`, `BLOCKED`, an exact-state/currentness-sensitive case, a case relying on separately produced Spec/Standards review evidence if available, and a distinct case relying on separately produced Scientific review evidence if available. Replay does not execute those reviewer functions. Each case uses one fresh isolated non-authorizing comprehensive request and exact V0 state fixture shared byte-identically between baseline and modular arms. Historical production authority is comparison evidence, never asserted current. Paired runs use fresh contexts, the same model family/version where controllable, the same reasoning effort, equivalent tool configuration, and frozen reviewer prompt/template and substantive candidate/evidence inputs. Bind exact baseline prompt, modular runtime, and manifest identities; record material environment differences and repeat pairs where feasible. Record baseline/modular controlled runtime UTF-8 bytes and tokens, tokenizer identity, observable platform context separately, opaque or uncontrolled context explicitly, mandatory coverage, recovered evidence, findings, verdicts, and forensic explanations for differences. Do not attribute a one-off model difference to prompt architecture without reconciliation.

## Gates and completion

1. **P0:** Rebind live heads, Issue/architecture/governance identities, absent public spec, exact writable path status, and pinned exchange file manifest. Material drift returns `BLOCKED_FOR_REBIND`.
2. **S1:** Freeze spec/plan/tasks and obtain independent Spec Review with no blocking finding. Prototype implementation cannot start sooner.
3. **I1:** Freeze the exact inventory and obtain independent Standards Review with no blocking finding. Fragment-consuming assembler code cannot start sooner.
4. **I2/V3/E4:** Create-new prototype, focused and broad verification, and controlled replay under R-01–R-08.
5. **R5:** Freeze one cross-repository candidate and obtain independent Spec and Standards implementation reviews. Repair changed affected bytes and re-review until no blocking findings remain.

Success is `READY_FOR_CONTROL_DESK_HANDOFF_REVIEW`: a frozen evidence-bearing candidate returned for Control Desk reconciliation and ordinary-Chat Handoff Review. It does not authorize merge, production Review Desk adoption, rollout, protocol adoption, dispatch of additional governance, or Issue #182 closure. A need for existing-source edits, a state materializer, new normative intent, V0/role/scientific change, unrecoverable or privacy-unsafe input, historical-currentness laundering, or material scope increase stops for owner decision/rebind under the r2 packet's reason codes.
