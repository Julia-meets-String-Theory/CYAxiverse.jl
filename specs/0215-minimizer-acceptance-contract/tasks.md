# Tasks — CYAX-0215

`tasks.md` records evidence readiness, not live PR, Issue or Project state.

- [ ] **T001 [G1] Record implementation evidence on the implementation candidate**
  - Input: implementation candidate `a6a07da`; R-001 to R-010.
  - Output: suite results and break checks in the PR #215 description.
  - Verify: each requirement row in `spec.md` *Verification* has the listed
    evidence. R-007 has integration evidence only. Evidence recorded on
    `ab48726`, `dc42633`, `b309a55` and `dbd7b62` is superseded; it is
    re-recorded on `a6a07da`.
  - Escalate if: a requirement has no evidence.

- [ ] **T002 [G0] Independent Scientific/Numerical Review of the specification**
  - Input: `spec.md` at a named draft commit SHA; implementation candidate
    `a6a07da`. Earlier reviews did not pass: `444fcc3` (against `ab48726`),
    `6b033b5` (against `dc42633`), `0442ead` (against `b309a55`), and `92f4c71`
    and `898029d` (against `dbd7b62`), and `033f673` (against `4abbd60`).
    They stay
    recorded as not passed; their records are not published on GitHub. The
    next revision needs a fresh review.
  - Output: a review record, as a PR #215 comment, that names the reviewer
    role and the reviewed SHA, with no unresolved blocking finding.
  - Verify: the reviewed SHA contains the `spec.md` that the owner approves in
    T003.
  - Escalate if: review finds an unlisted scientific decision, or a blocking
    finding needs an implementation change (update the implementation
    candidate and repeat T001, T002, T003 and T005).

- [ ] **T003 [G0] Owner approval event**
  - Input: the `spec.md` revision reviewed in T002.
  - Output: a durable owner comment on PR #215 that approves that revision by
    its commit SHA.
  - Verify: the comment names the SHA reviewed in T002.
  - Escalate if: the owner amends a requirement (update the implementation
    candidate if needed, and repeat T001, T002, T003 and T005 after the
    change).

- [ ] **T004 Record the approval**
  - Input: the T003 comment.
  - Output: a commit touching only this directory that sets `status: approved`,
    `last_reviewed` and `approval_ref`. Its result is the final merge candidate.
  - Verify: `git diff` from the reviewed SHA changes only those fields.
  - Escalate if: any other change to `spec.md` is needed (return to T002).

- [ ] **T005 [G1] Reconcile on the final merge candidate**
  - Input: implementation candidate `a6a07da`; final merge candidate.
  - Output: confirmation in the PR #215 description.
  - Verify: `git diff --stat a6a07da..<final merge candidate>` lists only this
    directory; CI passes on the final merge candidate.
  - Escalate if: any implementation or test file differs.

- [ ] **T006 [G2] Independent review of the final merge candidate**
  - Input: the final merge candidate; the approved spec; the PR #215
    description.
  - Output: a review record bound to the final merge candidate commit,
    covering the entire PR #215 diff against `vmm`.
  - Verify: no blocking finding.
  - Escalate if: a blocking finding needs an implementation change (update
    the implementation candidate and repeat from T001).
