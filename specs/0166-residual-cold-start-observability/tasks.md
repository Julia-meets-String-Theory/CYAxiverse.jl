# CYAX-0166 tasks

These tasks decompose evidence readiness. GitHub remains the live source for
Issue, Project, review, and merge state.

- [ ] T1 — Freeze #117 sources, the repository snapshot, and their hashes.
  - Outcome: replayable canonical source manifest at one observation boundary.
  - Verify: every required source identity and hash resolves exactly.
- [ ] T2 — Freeze the structured context, K1–K12 answer key, prompt, and
  preregistration.
  - Outcome: identical subject input and separately reviewable ground truth.
  - Verify: deterministic regeneration, semantic-parity review, byte hashes.
- [ ] T3 — Implement and test the private event layer.
  - Outcome: allowlisted raw events and deterministic aggregate only.
  - Verify: synthetic unit tests reject prose, paths, URLs, durable IDs,
    unknown fields, bad enums, and unstable aggregation.
- [ ] T4 — Obtain independent methodology/privacy review and owner approval.
  - Outcome: PASS review and approval both cite the exact frozen revision.
  - Verify: durable public references; otherwise stop before launch.
- [ ] T5 — Under a later dispatch, run four fresh identical-input subjects.
  - Outcome: four admissible frozen responses and private event logs.
  - Verify: isolation, model/configuration, input hashes, and launch contracts.
- [ ] T6 — Blind-score and aggregate without tuning.
  - Outcome: K1–K12 scorecards and bounded public aggregate.
  - Verify: independent scoring/aggregate review and deterministic A/B/C result.
- [ ] T7 — Destroy raw private events and record the result.
  - Outcome: deletion attestation, #166 result checkpoint, and #162 handoff.
  - Verify: no raw events, tokens, salt, transcripts, or private locators appear
    in Git or public GitHub artifacts.

Do not start T1–T7 from this design commit alone. The next action is review of
the draft packet, not experiment execution.
