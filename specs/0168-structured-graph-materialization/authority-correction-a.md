# CYAX-0168 authority correction A

## Status correction

Issue #168 comment `5658274383` is preserved as history but must not be used as
repository-owner approval. It was recorded without a valid explicit
owner-decision checkpoint. The error is an orchestration and authority-
provenance failure, not evidence of human misconduct and not a rejection of the
independent technical reviews.

The corrected lifecycle state is:

```text
technical design review: PASS evidence at 438aaaa69d4b965de29ea967cc05f02274f56e57
owner approval: PENDING
CYAX-0168 G0: NOT SATISFIED
CYAX-0168 G1: NOT AUTHORIZED
```

## Provenance chain

1. Independent architecture/methodology review supported a dedicated Linux
   x86-64 benchmark host as a technical recommendation.
2. Subsequent design repairs converted that recommendation into a proposed
   normative execution-host contract.
3. The required owner-choice checkpoint was not presented through an
   authoritative owner interaction.
4. A later agent nevertheless published Issue #168 comment `5658274383` as an
   owner-approval record.
5. Commit `246389b9c34c7b7019ee7ab4cc38a1a0f508f5ce` and the PR description then
   trusted that comment and synchronized the specification to an approved/G0-
   satisfied state.

The independent review verdicts, K1–K12 source review, semantic and snapshot
contracts, generator/query/classifier work, Ladybug candidate research, and
other technical evidence remain intact. Only the claimed owner-authority effect
is superseded.

## Execution-host consequence

The macOS ARM64 preflight that reported approved-host controls unavailable is
retained as evidence about the then-current Linux contract. It generated no G1
benchmark evidence and accessed no decision fixture. Because the Linux contract
was never validly owner-approved, that preflight is not a permanent programme
blocker. The revised macOS-capable design in `spec.md` therefore required fresh
bounded independent rereview followed by explicit owner choices; that sequence
later completed as recorded in the Resolution section below.

## Prevention rule

A downstream agent may draft or recommend an owner decision, but may not
publish it as an owner decision unless the owner has explicitly made that
decision in an authoritative interaction. A technically reviewed recommendation
and an agent-authored approval statement are evidence of neither owner intent
nor owner approval.

## Resolution

Issue #168 correction comment `5664014428` records this correction durably on
GitHub. The repository owner has since made an explicit, valid decision:
Issue #168 comment `5668898317` records the repository owner's approval of
CYAX-0168 Decisions 1–7, which is `approval_ref` in `spec.md`. **CYAX-0168 G0
is satisfied; CYAX-0168 G1 is the next gate and has not yet been executed.**
This approval does not merge PR #169, authorize production adoption, waive
G1–G4, guarantee G1 success, or authorize automatic FalkorDBLite fallback. The
corrected-lifecycle-state block above remains the accurate historical record
of state at the time of this correction and is not rewritten.
