# Tasks — CYAX-0215

`tasks.md` records evidence readiness, not live PR, Issue or Project state.

- [ ] **T001 [G1] Record implementation evidence on the implementation
  candidate**
  - Input: implementation candidate `4cca53d`; R-001 to R-010.
  - Output: suite results and break checks in the PR #215 description.
  - Verify: each requirement row in `spec.md` *Verification* has the listed
    evidence. R-007 has integration evidence only. Evidence recorded on
    `ab48726`, `dc42633`, `b309a55`, `dbd7b62`, `4abbd60` and `a6a07da` is
    superseded; it is re-recorded on `4cca53d`.
  - Escalate if: a requirement has no evidence.

- [ ] **T002 [G0] Independent Scientific/Numerical Review of the specification**
  - Input: `spec.md` at a named draft commit SHA; implementation candidate
    `4cca53d`. Earlier reviews that did not pass: `444fcc3` (against
    `ab48726`), `6b033b5` (against `dc42633`), `0442ead` (against `b309a55`),
    `92f4c71` and `898029d` (against `dbd7b62`), `033f673` (against
    `4abbd60`), and `94a61f7` (against `a6a07da`). They stay recorded as not
    passed; their records are not published on GitHub. The review of
    `043cc3f` (against `a6a07da`) passed; its record is not published on
    GitHub. The review of `f88be82` (against `a6a07da`) passed; it is the
    review recorded on PR #215. By owner direction, the provenance and
    editorial corrections after it were not re-reviewed, and the owner approved
    `96169a1` (`spec.md` G0). That approval is superseded by the candidate
    change after the G2 review of `f658144`. The review of `6916f0d` (against
    `4cca53d`) passed; it is recorded on PR #215. Its text findings are
    corrected, and the corrected draft receives a focused review.
  - Output: a review record, as a PR #215 comment, that names the reviewer
    role and the reviewed SHA, with no unresolved blocking finding.
  - Verify: the owner approves in T003 the SHA reviewed here. (The G0
    exception in `spec.md` applied only to the superseded approval of
    `96169a1`.)
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
  - Verify: `git diff` from the approved SHA changes only those fields.
  - Escalate if: any other change to `spec.md` is needed (return to T002).

- [ ] **T005 [G1] Reconcile on the final merge candidate**
  - Input: implementation candidate `4cca53d`; final merge candidate.
  - Output: confirmation in the PR #215 description.
  - Verify: `git diff --stat 4cca53d..<final merge candidate>` lists only this
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
