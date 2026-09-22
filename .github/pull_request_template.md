## Specification

Issue: # / N/A for PR-only S0
Spec: `specs/.../spec.md` / N/A for S0
Spec revision: <commit/revision or N/A for S0>
Approval ref: <durable approval reference where required / N/A>

Implements:
- R-... / N/A for S0
- G... / N/A for S0

Target iteration: `<stable identity>` / N/A for S0
Package-infrastructure SemVer impact: `patch` / `minor` / `major` / `none`
Package-version adoption: `none` / `<exact version and reason>`

> For S0 work, omit or mark `N/A` any SDD section that is not relevant. Do not invent specification metadata merely to satisfy the template.

## Change

Concise description of what was implemented.

## Explicit non-scope

What remains outside this PR/spec gate. N/A for trivial S0 changes when there is no meaningful separate non-scope.

For lifecycle work, state explicitly whether historical designation,
production `-DEV`, closure, public-tag creation, publication, or `vmm -> main`
reconciliation is outside this PR. Gate A infrastructure does not perform those
actions.

## Scientific / contract impact

- Scientific behaviour: N/A if none
- Physical convention: N/A if none
- Public API: N/A if none
- Persisted schema: N/A if none
- Compatibility: N/A if none
- Version impact: N/A if none

## Verification

| Requirement / gate | Evidence | Result |
| --- | --- | --- |
| R-... / S0 check | `test/...` / validation artifact / command | PASS / FAIL / BLOCKED |

For release work, include the verified ref/manifest context, exact candidate and
certification tree identities, and the release evidence location. Documentation
deployment must be selected from that context without a release-time tracked
source edit.

Include exact commands and observed results where material. Keep large evidence in durable artifacts rather than the PR body.

## Lessons / corrections

- Reusable failure mode discovered: Yes / No / N/A
- Owner redirection that should persist: Yes / No / N/A
- Lesson added/updated: L-____ / N/A
- Normative rule change required: Yes / No / N/A

For S0 work, omit this section or mark all fields `N/A` when lessons and
corrections are irrelevant.

## Convergence

- [ ] Implementation checked against the approved `spec.md`/amendments where SDD applies, or N/A for S0.
- [ ] Every requirement/gate claimed by this PR has evidence, or N/A for S0.
- [ ] No unmet requirement is being silently declared complete.
- [ ] Any changed scientific assumption was returned to the spec/owner before implementation, or N/A.
- [ ] Independent review completed where the governing contract requires it, or N/A.

## Remaining work

List later gates/tasks without implying that a partial-gate PR completes its parent specification. N/A when there is no separately tracked remaining work.
