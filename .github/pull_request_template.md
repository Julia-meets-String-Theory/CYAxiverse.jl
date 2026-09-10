## Specification

Issue: #
Spec: `specs/.../spec.md`
Spec revision: <commit/revision or N/A for S0>

Implements:
- R-...
- G...

## Change

Concise description of what was implemented.

## Explicit non-scope

What remains outside this PR/spec gate.

## Scientific / contract impact

- Scientific behaviour:
- Physical convention:
- Public API:
- Persisted schema:
- Compatibility:
- Version impact:

## Verification

| Requirement / gate | Evidence | Result |
| --- | --- | --- |
| R-... | `test/...` / validation artifact / command | PASS / FAIL / BLOCKED |

Include exact commands and observed results where material. Keep large evidence in durable artifacts rather than the PR body.

## Convergence

- [ ] Implementation checked against the approved `spec.md` where SDD applies.
- [ ] Every requirement/gate claimed by this PR has evidence.
- [ ] No unmet requirement is being silently declared complete.
- [ ] Any changed scientific assumption was returned to the spec/owner before implementation.
- [ ] Independent review completed where the governing contract requires it.

## Remaining work

List later gates/tasks without implying that a partial-gate PR completes its parent specification.
