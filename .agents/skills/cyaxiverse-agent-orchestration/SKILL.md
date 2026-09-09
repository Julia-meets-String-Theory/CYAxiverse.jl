---
name: cyaxiverse-agent-orchestration
description: >
  Coordinate multiple AI agents on a CYAxiverse deliverable. Use when the
  current agent will spawn, supervise, recover, or integrate subagent work. Do
  not load for ordinary single-agent implementation.
---

# CYAxiverse Agent Orchestration

Use this skill only when coordinating delegated work. It is an on-demand
manager workflow, not an additional repository-wide policy source.

## 1. Manager role

The manager owns:

- task decomposition;
- scientific and architectural decisions;
- dependency ordering;
- worker assignment;
- contradiction resolution;
- integration;
- final acceptance.

The manager is not a continuous progress monitor. Detailed implementation,
routine debugging, routine testing, formatting, and ordinary corrections belong
to workers. Before a supervisor continuation, ask:

> Does the manager need to make a decision now?

If the answer is no, prefer continued waiting over another manager reasoning
continuation. Wake for completion, a declared checkpoint, a decision or
escalation, a contradiction, or an integration/final-acceptance decision.

Scientific acceptance, evidence, and independent-review standards remain
unchanged by this workflow.

## 2. Delegation task packet

Every delegated task should state the following:

- **Objective** — one bounded outcome.
- **Acceptance** — observable completion criteria.
- **Inputs** — relevant files, artifacts, issue/PR references, and source
  material.
- **Constraints** — scientific, compatibility, schema, or repository
  invariants that matter.
- **Worker ownership** — ordinary failures and corrections the worker is
  expected to resolve without escalation.
- **Escalation** — decisions that must return to the manager or scientific
  owner.
- **Lease** — expected time before one health checkpoint becomes appropriate.

Do not delegate a large open-ended programme when it can be split into
independently observable phases. A task packet should make the worker's
durable starting state and the expected handoff location clear.

## 3. Worker return contract

Use four states:

### DONE

Acceptance criteria are met. Return a concise evidence packet with the result,
changed files or artifacts, exact checks and outcomes, relevant assumptions or
conventions, and durable references.

### BLOCKED

Progress requires a manager or scientific-owner decision. State the smallest
decision or question that unlocks the task, along with the durable state and
evidence that led to it.

### FAILED

The worker cannot produce a viable next action after bounded recovery. State
what failed and preserve enough durable state, logs, and references for
resumption or replacement.

### CHECKPOINT

Use only when a predeclared task lease expires while the worker is still active.
Keep it short, with a maximum around 150 words. Include:

- current phase;
- concrete progress;
- last verified result;
- current obstacle, if any;
- next intended action;
- `making_progress: yes|no`.

Routine test failures, normal debugging, and normal implementation corrections
are not `BLOCKED` states. The worker should normally continue through them.
The detailed final scientific evidence may be as long as required; the roughly
150-word limit applies only to a health `CHECKPOINT`.

## 4. Waiting and stall detection

Prefer long, completion-driven waiting. Do not repeatedly wake the manager just
to ask whether a worker is still running.

At lease expiry:

1. Obtain or inspect one `CHECKPOINT`.
2. If concrete progress exists, extend the lease once and return to waiting.
3. If progress is absent, circular, or no longer narrowing, issue one recovery
   instruction.
4. If recovery also fails, interrupt or replace the worker from its durable
   checkpoint/state.

Evidence of a slow but healthy worker includes changed test state, a new
artifact, a new verified output, a narrowed error diagnosis, a successful
command that advances the task, an eliminated scientific hypothesis, a smaller
remaining search space, or a concrete next action grounded in new evidence.

Probable stall indicators include repeated identical failures without narrowing,
repeated rereading with no new evidence, no changed artifact/test/diagnosis
across the lease, circular uncertainty without a decision criterion, repeated
retries without meaningful state change, or no activity plus failure to respond
to the health checkpoint.

**Do not infer a stall solely from elapsed time.** A healthy scientific worker
may be quiet while performing a bounded, expensive computation; use evidence
and the one-checkpoint recovery procedure.

## 5. Worker reuse versus independent agents

Prefer the same implementation worker for:

```
implementation -> test -> diagnose -> correction -> retest
```

Do not automatically spawn a replacement for each failure. Use a fresh worker
when the existing worker is demonstrably stalled, the approach needs a clean
restart, the work has become independently mergeable, deliberate independent
reasoning is desired, or independent scientific verification materially
strengthens the claim.

Preserve genuinely independent scientific review. Do not optimise away
independent review merely to reduce agent count. Record the reason for each
replacement and distinguish a recovered genuine stall from a healthy worker
that was allowed to continue.

## 6. Manager context discipline

Manager-visible worker evidence should normally contain only:

- status/result;
- files or artifacts changed;
- exact commands, tests, or checks and observed outcomes;
- relevant scientific assumptions or conventions;
- the unresolved manager/scientific decision, if any;
- commit, SHA, path, artifact, or other durable reference.

Do not send the manager full command transcripts, large raw logs, whole
derivations, repeated repository summaries, or duplicated historical context
unless required to resolve a specific contradiction. Store detailed evidence
in durable files or artifacts and return their paths. Collapse completed tasks
to short durable records: the manager should know where evidence is, not carry
all of it in active context.

## 7. Suggested initial task leases

These are heuristics, not correctness criteria:

- **MECHANICAL/check-only:** ~5 minutes
- **bounded IMPLEMENTATION:** ~10–15 minutes
- **substantial implementation/verification:** ~15–20 minutes
- **deep SCIENTIFIC reproduction/reasoning:** ~20–30 minutes

Use judgment. Some healthy scientific workers need longer. Elapsed time alone
is not a stall signal, and a longer healthy worker lease is preferable to
frequent expensive manager resumptions.

## 8. Task classification

Classify delegated work as one of:

- `MECHANICAL`
- `IMPLEMENTATION`
- `SCIENTIFIC`
- `INDEPENDENT_REVIEW`

The class may guide model/effort choice, expected lease, escalation rules, and
reviewer requirements. Do not hard-code a particular model into repository-wide
policy; model availability changes over time.
