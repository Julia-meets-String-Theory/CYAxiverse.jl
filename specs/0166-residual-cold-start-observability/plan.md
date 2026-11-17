# CYAX-0166 implementation plan

## Status

Draft design only. No experiment or instrumentation implementation is
authorized until the exact packet revision receives independent
methodology/privacy review and repository-owner approval.

## Approach

1. Freeze the #117 public source bundle and repository revision named in
   R-001. Record byte sizes, SHA-256 values, observation time, and source roles.
2. Materialize one concise structured context from those sources and a separate
   citation-bearing K1–K12 answer key. Do not optimize either from subject
   output.
3. Implement a narrow private event writer, schema validator, aggregator, and
   deletion check for the R-004/R-005 fields only. Use synthetic fixtures for
   tests; never commit raw run logs or salts.
4. Obtain independent methodology/privacy review of the full freeze, including
   semantic parity between context and answer key, source availability,
   scoring, instrumentation validity, and privacy.
5. Obtain repository-owner approval tied to the exact preregistration revision.
6. Only under a later dispatch, launch four fresh identical-input subjects,
   freeze responses/events before scoring, score blindly, aggregate, verify,
   destroy raw events, and apply the A/B/C gate.
7. Post one bounded public result to #166 and return to #162. Open no additional
   Issue or mechanism experiment unless a later owner decision follows a
   preregistered Outcome B failure.

## Requirement mapping

| Requirement | Planned artifact or action | Verification |
| --- | --- | --- |
| R-001 | source/context/answer-key/preregistration manifests | hash and identity validator |
| R-002 | four-run launch manifest | exact input/configuration and isolation checks |
| R-003 | blinded K1–K12 scorecards | score schema and independent scoring review |
| R-004 | minimal event writer/validator/aggregator | allowlist and deterministic aggregate tests |
| R-005 | private raw-log lifecycle | semantic-payload rejection, public-output scan, deletion attestation |
| R-006 | frozen A/B/C analysis rules | deterministic result classification |
| R-007 | design and dependency review | diff/dependency/non-scope scan |

## Escalation conditions

Return to the owner before continuing if source precedence is ambiguous,
Issue #117 changes, an event field beyond the approved schema appears necessary,
the model/configuration must change, a run sees non-identical inputs, privacy
cannot be verified, or the proposed conclusion would authorize a new mechanism.

## Version impact

None for the bounded experiment. Any public/package behavior requires a new
reviewed scope and the normal `vmm` → `main` release boundary.
