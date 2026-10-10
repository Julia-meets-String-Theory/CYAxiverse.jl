# Implementation self-check

Trigger: Preparing an implementation handback. Inputs: exact governing requirements, authorized baseline, frozen candidate, applicable standards, and existing verification evidence.

Inspect the complete actual change from the authorized baseline to the candidate. Include in-scope generated/external artifacts through their separate identities. A merge-base diff alone is insufficient when it omits differences from the authorized baseline.

Report two distinct dimensions:

- Specification: map every applicable requirement to implementation and evidence, or identify missing/partial/blocked coverage and scope additions. Do not infer a replacement specification or silently waive a required one.
- Engineering: inspect correctness, compatibility, Julia constraints, precision, data boundaries, and adequacy of verification. Tool-enforced rules require actual applicable tool results; tool availability is not evidence that the rule passed.

Classify concrete defects separately from optional design suggestions. Preserve all material findings; use linked detail rather than truncating findings to fit a word budget. Identify any scientific interpretation boundary and follow the existing scientific process.

Correct ordinary defects within authorized scope and rerun affected checks before handback. A changed candidate requires refreshed identity and applicable evidence. Return this as an internal assessment, without a formal independent-review verdict or inferred readiness, dispatch, merge, or scientific acceptance. Staffing and independent review remain governed elsewhere.
