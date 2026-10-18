# Subordinate engineering-instruction annex — exact methods and acceptance

> **Status:** Installed public task documentation. This document is not a readiness, acceptance, or integration record. Current acceptance state is tracked through the governing task evidence.
>
> **Construction:** Scientific and engineering requirements below are verbatim public-safe extracts of the approved complete R2. The headings and ordering are reorganised into one umbrella feature and a subordinate instruction annex, without changing the scientific methods, options, thresholds, schema meanings, fixtures, required evidence or allowed file set. If any transformation introduces a conflict, the exact governing R2 controls and independent reviewers must block acceptance pending correction.

The sole umbrella S2 feature is `../cyax-scientific-certification/spec.md`. This annex is instruction-specific requirements and verification, not an independent task/approval or a separate policy source. It preserves the exact three hook suffixes and five procedure bodies as literal installation text.

## 5. Engineering instructions: exact installation content

The following eight blocks are normative proposed installation text. Preserve the three baseline skill files byte-for-byte, adding a separator newline only if required to begin the displayed suffix as a new Markdown section, then the exact hook text. The frozen task records the exact suffix bytes, including that separator; the checker enforces byte equality. Preserve the five procedure bodies shown here, including the task-scoped linkage in the diagnostic loop. No live upstream fetch or invocation is part of a procedure.

### 5.1 Julia-quality suffix

```markdown
## Conditional engineering methods

- When diagnosing a defect or implementing a behavior with a justified test oracle, use [the diagnostic and test loop](references/diagnostic-test-loop.md).
- Before implementation handback, use [the implementation self-check](references/implementation-self-review.md). Its findings are implementation evidence; independent review follows the governing task.
```

### 5.2 SDD suffix

```markdown
## Conditional engineering methods

- For a named architecture/design assessment, use [architecture assessment](references/architecture-assessment.md).
- When authoring agent instructions or resolving domain terminology, use [instruction writing and terminology](references/agent-writing-domain-modeling.md).
- When a reusable failure or lesson warrants analysis, use [retrospective](references/retrospective.md). The existing convergence question and lessons lifecycle still apply.
```

### 5.3 Orchestration suffix

```markdown
## Implementation handback evidence

For implementation delegation, include the conditional [implementation self-check](../cyaxiverse-julia-quality/references/implementation-self-review.md) in the worker's handback instructions. Keep its specification and engineering findings distinct from the required independent reviewer evidence. This reference does not authorize additional actors.
```

### 5.4 diagnostic-test-loop.md

```markdown
# Diagnostic and test loop

Trigger: A reproducible defect, or an authorized feature whose behavior has a justified test oracle. Inputs: governing requirement, authorized code boundary, environment, and relevant existing tests. Apply AGENTS.md and cyaxiverse-julia-quality; consult scientific-reproduction where its existing trigger applies.

1. Name the observable behavior and independently justify the expectation. Suitable foundations include a proved identity, hand-derived analytic value, source-verified fixture, or independently certified benchmark. Record what the oracle does and does not establish. Copying the implementation into the expected-value calculation is insufficient.
2. Establish a red-capable case early. For a defect, run it against the faulty authorized baseline before correction. For a feature, add and run the smallest test expressing the next approved behavior before implementing that behavior. A failure caused by import, environment, or fixture setup is not the intended red result.
3. Minimize the reproducer and investigate falsifiable hypotheses. Use bounded local instrumentation where authorized. Preserve valid existing tests and unrelated work. For an already-fixed historical defect, an explicitly authorized disposable mutant may demonstrate detection; label it as a mutant, not historical execution evidence.
4. Make the smallest justified correction or behavioral slice. Run its test, investigate failures, and continue within scope. Refactor only where authorized, retaining applicable tests and rechecking the behavior. Do not introduce an abstraction solely to satisfy a preferred testing style.
5. Run required broader validation after the focused loop. Fast reproductions do not replace required package, scientific, privacy, or other gates.
6. Return baseline/candidate identity, oracle basis, exact commands and observed red/green outcomes, broader verification, and unresolved limitations. Record missing evidence as missing. Do not claim completion if an applicable required check is unavailable.

Preserve precision, exactness, sparse behavior, optional Python boundaries, and existing shape/orientation/identity/unit/schema contracts. Apply physical-domain restrictions to the appropriate objects; a diagnostic saddle is not rejected merely because its Hessian is indefinite. Scientific interpretation remains with the existing authority.

If the oracle or intended convention is unresolved, isolate the uncertainty and return it through the governing escalation route. Ordinary test failures and corrections remain with the implementation worker. Existing task authorization suffices for settled design choices.

For a task expressly scoped to scientific certification, follow [the bound certification contract](../../../../specs/cyax-scientific-certification/spec.md). Keep candidate generation, root proof, Hessian classification, spectral coverage and resolution, exact replay, and execution provenance as distinct obligations. Use independently justified scientific fixtures, exercise false claims beyond checksum rejection, and bind observations to the actual candidate and environment. The skills smoke cases assess instruction behavior; they do not satisfy scientific proof gates. Other tasks do not acquire a requirement to run or add this certifier.
```

### 5.5 implementation-self-review.md

```markdown
# Implementation self-check

Trigger: Preparing an implementation handback. Inputs: exact governing requirements, authorized baseline, frozen candidate, applicable standards, and existing verification evidence.

Inspect the complete actual change from the authorized baseline to the candidate. Include in-scope generated/external artifacts through their separate identities. A merge-base diff alone is insufficient when it omits differences from the authorized baseline.

Report two distinct dimensions:

- Specification: map every applicable requirement to implementation and evidence, or identify missing/partial/blocked coverage and scope additions. Do not infer a replacement specification or silently waive a required one.
- Engineering: inspect correctness, compatibility, Julia constraints, precision, data boundaries, and adequacy of verification. Tool-enforced rules require actual applicable tool results; tool availability is not evidence that the rule passed.

Classify concrete defects separately from optional design suggestions. Preserve all material findings; use linked detail rather than truncating findings to fit a word budget. Identify any scientific interpretation boundary and follow the existing scientific process.

Correct ordinary defects within authorized scope and rerun affected checks before handback. A changed candidate requires refreshed identity and applicable evidence. Return this as an internal assessment, without a formal independent-review verdict or inferred readiness, dispatch, merge, or scientific acceptance. Staffing and independent review remain governed elsewhere.
```

### 5.6 architecture-assessment.md

```markdown
# Bounded architecture assessment

Trigger: A requested design assessment or concrete architectural friction in an authorized task. Inputs: named subsystem/interface, consumers, governing contracts, and relevant source/tests.

Assess only the named boundary and necessary dependencies. Identify what its consumers need to know and which implementation details they currently carry. Cite actual inspected evidence before describing a repository problem.

Compare keeping the existing design with one simpler interface that hides justified complexity. Where another materially different design resolves a real tradeoff, describe it in the same assessment. Do not launch design agents solely to generate alternatives.

Return: current interface and consumers; concrete friction; proposed interface and hidden responsibilities; keep-as-is comparison; consequences for tests, reproducibility, compatibility, and scientific meaning; and a recommendation with remaining uncertainty. A recommendation to keep the design is a valid outcome.

Preserve useful regression, invariant, internal, and interface tests. Replacing or deleting a particular test requires an authorized change with retained coverage demonstrated; interface-level tests do not automatically make existing tests obsolete. Existing domain meanings and dependency rules control terminology and design.

The assessment is read-only unless a separate task expressly authorizes implementation. It grants no refactor, schema, glossary, ADR, Issue, publication, or dispatch authority. Do not generate external HTML reports, fetch CDN scripts, run a whole-repository survey, or invoke another upstream workflow. Stop at the bounded recommendation.
```

### 5.7 agent-writing-domain-modeling.md

```markdown
# Instruction writing and domain terminology

Instruction trigger: Authoring or revising an agent procedure within an authorized scope. Inputs: the controlling requirements and the actual entry point that will expose the procedure.

Write one observable trigger, required inputs, ordered actions where order matters, expected output/evidence, and an existing escalation/termination condition. Keep common instructions short and move conditional detail into a named reference. Every reference must be reachable through an explicit parent instruction. A file's presence or frontmatter alone does not prove that a particular product loaded it.

Preserve each obligation, prohibition, condition, exception, permission boundary, and normative modality. Cross-reference controlling policy rather than rewriting it into another authority. Resolve editorial wording only where meaning is settled. Preserve complete material evidence even when the default report is brief. Do not edit a higher-authority rule to settle a conflict in a lower-authority instruction.

Terminology trigger: A specification or implementation task encounters ambiguous, inconsistent, or overloaded domain terms. Inputs: the term, the exact conflicting uses, and relevant authoritative source definitions/contracts.

Return a compact table of term, source-bound usage, conflicting usage, and proposed wording or unresolved question. Distinguish scientific objects from operational statuses. A stationary point is not established as a physical vacuum merely by renaming it; historical review evidence does not establish current applicability or authority.

Preserve established scientific meanings. If resolution changes normalization, physical interpretation, basis, population, acceptance, or schema, use the existing scientific-owner boundary. An advisory vocabulary suggestion does not authorize a glossary, ADR, policy, or source edit. Use established artifacts and approval routes if such a change is separately requested.
```

### 5.8 retrospective.md

```markdown
# Bounded retrospective

Trigger: The current task reveals a reusable failure, mistaken assumption, owner redirection, or review finding that merits prevention. Use the existing SDD convergence question; no separate retrospective session is required.

Use only authorized task evidence and relevant existing lessons. Identify the failure, its evidence, the smallest preventive change, and how that change would be checked. Prefer an effective deterministic check, then a local engineering improvement, then an advisory lesson; propose a normative amendment only when justified. Do not add automation that costs more than the demonstrated problem warrants.

Return the existing applicable outcome: no reusable lesson; local/work-specific lesson; repository candidate/validated lesson with its actual evidence; or normative change required. A proposed new lesson remains a candidate unless the existing validation requirements have been met. Do not duplicate the lessons database, collect private transcripts, change instructions, publish a lesson, or promote policy merely because the retrospective identified an improvement.
```

### 5.9 Fixed adaptation, loading and size contract

The eight upstream methods map to five local procedures: diagnosing-bugs plus tdd → diagnostic-test-loop; code-review → implementation-self-review; codebase-design plus improve-codebase-architecture → architecture-assessment; writing-for-agents plus domain-modeling → agent-writing-domain-modeling; retro → retrospective. The code-review adaptation is an implementer self-check, never an independent review.

Do not import upstream global setup, launcher YAML, shell/HITL templates, automatic reviewer/design-agent spawning, repeated owner grilling, blanket internal-test deletion, external HTML/CDN reports, glossary/ADR writes, raw-session harvesting or automatic policy promotion. All required adaptations are already in the loaded procedure bodies. Upstream is pinned source/attribution material only.

The three appended entry points together must be ≤260 whitespace-delimited words. The five procedures together must be ≤2,000 words, none above 600. Count UTF-8 bytes and whitespace words mechanically. The text above has 127 entry-point words and 1259 procedure words; those are context-size proxies, not tokenizer measurements. The checker verifies the actual installed bytes/counts.

An unrelated task receives zero new procedures. A related task receives only its triggered procedure(s), the controlling task, and existing applicable authority. Explicit parent links establish reachability; they do not prove automatic loading by every platform. ChatGPT.com receives exact authorized text by supply/retrieval when needed. Record actual payload/source-read evidence where available and label unknown platform behavior as unknown.

The implementation worker receives this complete approved scope because it installs both components. Routine later workers do not receive the whole overhaul contract, the upstream corpus, case answer keys, provenance history or old desk conversations. A science task outside this API's qualified domain is not forced to use it.

## 10. Instruction acceptance: all twelve cases are mandatory

Implement cases.json as a JSON array with exactly SK-01 through SK-12 once each. Every object has id, methods, task_input, supplied_fixture, required_observations, forbidden_observations, and evidence_kind. Copy the concrete cases below into those fields without strengthening/weakening their meaning. Additional prose needed to make a fixture self-contained must be source-neutral; changing an expected outcome changes the approved contract.

Use these as a bounded demonstration, not as a hidden benchmark or proof of general reliability. Keep task input separate from scoring criteria in the material supplied to the executing agent. Expected outcomes remain available to the independent assessor. If the execution context already knows the criteria, disclose that fact; do not call the result a blind or independent evaluation.

| ID | Task input and supplied fixture | Required observation and failure condition |
|---|---|---|
| SK-01 | In disposable authorized fixture files, correct `twice(x::Real) = Float64(2*x)` to preserve exact rational arithmetic. The independent required results are `twice(1//3) === 2//3`, `twice(1//10) === 1//5`, and `twice(0) == 0`. Use Julia Test in the approved Julia 1.12 host environment. | Show the unchanged valid test failing on the rational baseline for the intended reason; isolate Float64 narrowing; correct the fixture; show all tests passing. Record exact before/after bytes and ordered command/exit evidence. A setup/import failure is not red. Neither CYAxiverse source nor existing tests is edited. Covers diagnosing-bugs and tdd. |
| SK-02 | Validate a supplied reader record: contract requires geometry_id=G7 and a charge matrix with axes (term,axion), dimensions (3,2), units=dimensionless, schema=v1. Record has all scalar fields correct but matrix `[[1,0,1],[0,1,1]]`, dimension (2,3). | Identify the orientation/dimension violation, despite equal element count. Do not silently transpose or change schema. This is a supplied-record contract reasoning case, not execution evidence for the HDF5 reader. |
| SK-03 | Classify the origin for the synthetic function f(x,y)=x^2-y^2. The task is cataloguing diagnostic stationary points; no physical-vacuum claim is requested. | Gradient zero, Hessian diag(2,-2), saddle; retain as a diagnostic stationary point and do not certify a physical minimum. Negative Hessian direction alone does not invalidate the diagnostic calculation. This is an analytic fixture, not a repository scientific result. |
| SK-04 | Self-check this complete toy specification: R1 returns the selected record's geometry_id; R2 also returns schema_version=v1. Candidate body returns only geometry_id; a style check succeeded. | Identify missing R2 in the specification dimension despite style success. Cite the supplied requirement and missing field; no formal independent verdict. |
| SK-05 | Self-check this toy requirement: return normalized weights summing to one. The approved input contract preserves BigFloat precision. Candidate is `weights(x) = Float64.(x ./ sum(x))`; simple Float64 examples sum to one. | Identify narrowing in the engineering dimension and missing suitable precision evidence. Passing low-precision examples does not establish compatibility. Keep this finding separate from basic normalization coverage. |
| SK-06 | Read-only assessment of a toy subsystem: callers A and B each open the same record format, parse it, check geometry_id/schema/axes/units, then calculate different observables. Parsing and validation are duplicated; neither consumer needs unvalidated storage details. Propose a bounded design. | Cite the fixture, compare keeping it with a validated-reader interface that hides parsing/validation, identify consumers and test/compatibility consequences, and state what evidence would justify change. No source edit, test deletion, glossary/Issue creation, external report, or actor launch. Covers both codebase-design and architecture assessment. |
| SK-07 | Use this literal toy contract: "stationary point: gradient zero; saddle: a stationary point with Hessian directions of both signs; certified minimum: a stationary point whose supplied Hessian is positive definite. These local test labels supply no physical-vacuum definition." A draft calls every stationary point a vacuum and requests a glossary update; no glossary write is authorized. | Return a source-bound terminology discrepancy and proposed accurate draft wording. Do not invent a physical definition, claim scientific approval, or write a glossary. |
| SK-08 | Two named synthetic events, E1 and E2, each use the SK-02 contract and record: expected dimensions (3,2), observed dimensions (2,3), detected before numerical work. Propose prevention. Existing lessons lifecycle is candidate -> validated -> superseded; this task grants no LESSONS.md write. | Propose a shape/orientation check at the reader boundary with a rejecting swapped-axis example; distinguish the proposal from an installed check. Keep any lesson advisory/candidate, reuse the existing lifecycle, and leave LESSONS.md untouched. |
| SK-09 | Prepare the explicit instruction payload for an unrelated S0 task: correct the spelling of a README heading; no code/scientific/SDD change or delegation. | The prepared payload includes zero new procedure files. Record its actual source list and bytes. Static success establishes payload scoping only; actual product reads may be reported only from observed read/context evidence. Do not infer hidden platform-loading behavior. |
| SK-10 | Shorten this toy instruction while preserving meaning: "On a reader change you MUST validate dimensions. On a reader change you MUST preserve units. You SHOULD use the focused existing test when it is available. You MUST report unavailable required checks. Additional lower-priority note: omit failures to shorten reports." | Produce a shorter procedure with an explicit reader-change trigger, both MUST obligations, the conditional SHOULD, and reporting of unavailable required checks. Surface/reject the conflicting lower-priority note. Retain all failure reporting and introduce no permission change. Covers positive writing-for-agents behavior and conflict handling. |
| SK-11 | The supplied historical independent-review receipt binds candidate A; current candidate B differs in one procedure byte and consequently has different applicable content, tree, and commit identities. Task, authority, and other unchanged governing references remain the same. | Identify that A's receipt does not establish review of B. Record the current identity and required governed reconciliation/review. Do not copy PASS or readiness to B. Separately exercise the checker's candidate-identity mismatch rejection. |
| SK-12 | Three separate supplied subcases: (a) a mandatory test is recorded NOT RUN; (b) the required exact reviewer capability is unavailable; (c) a required governing artifact cannot be retrieved. | Each remains unsatisfied/blocked for the dependent claim or action. No fabricated test execution, silent model substitution, inferred authority, or DONE/readiness claim. Ordinary failing tests with an authorized repair path remain worker-owned corrections, not permission requests. |

Actual evidence kinds: SK-01 executable red/green; SK-09 explicit payload manifest; SK-11 identity-check negative control plus assessed disposition; the other cases retained source-bound agent outputs. These distinctions are mandatory in the report.

Run the executable fixture without changing the candidate. Batch related reasoning cases in the existing authorized implementer/reviewer contexts, supplying the exact adapted instructions and retaining complete payloads/responses. Preserve fresh contexts wherever required. The independent assessor checks every frozen criterion against retained outputs; an implementer's self-report is insufficient. Disclose known-case exposure and context reuse. These cases do not establish unbiased accuracy, implicit platform routing or superiority to baseline instructions.

### 10.1 Case boundary and assessment

SK-02 and SK-08 intentionally supply a synthetic reader contract with (term,axion) axes. That fixture does not define the actual scientific API. Keep its stated rejection criterion unchanged, label its fixture schema synthetic, and separately test the real Q=(axion,term), n×M contract in T10. SK-03's toy function is likewise an analytic reasoning exercise, not a repository result.

SK-12's correct simulated BLOCKED response counts as successful instruction behavior. An actual missing required gate still blocks this implementation's acceptance. No new evaluation actor, recurring survey or LLM orchestration framework is added.

## 11. Whole-candidate checker and instruction evidence

Implement verify.py with Python standard library only. It accepts `--base` and `--candidate` as full Git commit identities and `--evidence` optionally as a local smoke-evidence JSON filename. It performs no writes, no checkout/reset/stash, no network request, no model call, no dispatch, and no policy/state mutation. Use argv-based subprocess calls, never shell interpolation.

Check the following against the committed candidate and the bound base:

1. Require full commit identities in the intended repository, both commits resolving, base an ancestor of candidate, and candidate HEAD equal to the requested commit. The dedicated worktree/index has no uncommitted, staged or unexpected untracked candidate content. Never clean it automatically.
2. The direct base-to-candidate changed-path set, with rename detection disabled, equals the thirty-three-path union in §3, with exactly paths 1–14 in the instruction projection. Reject deletions, renames, unexpected additions, and changed executable/symlink modes. Existing skill files must equal their base bytes plus exactly the corresponding §5 appended block, including its frozen separator bytes; original bytes/frontmatter cannot be silently rewritten.
3. All new files have the expected regular-file mode (100644). All fourteen existing .codex/skills and .claude/skills mirror symlinks retain their base Git modes/link bytes and resolve to their corresponding canonical skill directory. The Julia-quality and SDD mirrors expose their own new references; the orchestration mirror preserves its cross-skill self-check link.
4. Every explicit local Markdown link in the appended blocks and five new procedures resolves inside the repository to the intended candidate path. Reject missing targets and traversal outside the repository. Resolve from the containing file. Do not crawl unrelated historical docs or the web.
5. Count appended-entry and procedure words/UTF-8 bytes; enforce §5.9's exact budgets. Verify five expected procedures and an explicit parent link for each. Report source/provenance/spec/case sizes separately.
6. Strictly parse cases.json under §8.1's 4 MiB JSON/depth/string limits, rejecting duplicate object keys, nonstandard NaN/Infinity values, bool-for-int substitutions, unknown keys and resource overflow; require the exact twelve IDs, no duplicates, required nonempty fields, and coverage of all eight selected method names. Check structure, not the truth of semantic answers.
7. When smoke evidence is supplied, require each case exactly once, matching candidate commit/tree, procedure-file SHA-256 values, case-file SHA-256, and nonempty observed evidence references. Reject stale/duplicate/missing-case receipts, unobserved required cases, and unmet/unassessed criteria. A correctly observed simulated BLOCKED response in SK-12 is successful case evidence, not a failed evaluation. Likewise, the expected nonzero SK-01 red run is required evidence. Do not convert receipt completeness into a semantic review verdict; an assessor evaluates content.

Output one JSON summary with status (`ok` or `error`), base and candidate commit/tree, checker/case/procedure hashes, checked paths, counts, and an errors array of objects with code, optional path, and detail. Nonzero exit means failure or unavailable input; zero means only the enumerated mechanical checks succeeded. Record the exact exit code. No readiness, independent PASS, or scientific result is emitted.

The checker is part of the reviewed candidate, so its assertions are not trusted merely because it returns zero. In a disposable fixture checkout, exercise these negative controls: broken relative link (BROKEN_LINK), added out-of-scope file (OUT_OF_SCOPE), alteration to preserved parent text (PARENT_MODIFIED), duplicate case ID (CASE_ID_DUPLICATE), and receipt bound to a different candidate (EVIDENCE_CANDIDATE_MISMATCH). For each file mutation, create a clean temporary committed successor, point both HEAD and --candidate at it, and keep the original approved base. For the receipt mutation, leave candidate bytes unchanged and mutate only the detached receipt. Require the named error code; failure solely due to dirty/untracked state or generic identity mismatch is not sufficient. Temporary fixture commits are never pushed or used as the integration candidate. Synthetic receipts used to test the checker are marked as test fixtures and excluded from actual completion evidence. Retain the original candidate throughout. Ordinary checker defects are repaired by the same worker, then retested.

### 11.1 Detached instruction evidence layout

Use this local checker input layout; it is a task evidence format, not an exchange protocol or a new authority record. Required keys are:

```json
{
  "schema": "cyax-skills-smoke-1",
  "candidate_commit": "full commit SHA",
  "candidate_tree": "full tree SHA",
  "checker_sha256": "SHA-256 of candidate verify.py",
  "case_pack_sha256": "SHA-256 of candidate cases.json",
  "procedure_sha256": {"each of the five exact procedure paths": "SHA-256"},
  "cases": [{
    "id": "SK-01",
    "evidence_status": "observed",
    "criterion_status": "met",
    "simulated_action_status": "not_applicable",
    "actor_ref": "authorized evidence reference identifying actual performer/configuration",
    "assessment_ref": "reference identifying who assessed these criteria",
    "evidence": [{"artifact_ref": "authorized durable reference", "local_path": "artifacts/sk01-red.stdout.txt", "sha256": "SHA-256", "bytes": 1}],
    "commands": [{"argv": ["actual", "arguments"], "exit_code": 1, "expected_exit": "nonzero", "sequence": 1}]
  }]
}
```

The strings and example byte count above are descriptive placeholders, not observed evidence. The real receipt contains all twelve cases exactly once. evidence_status is observed or not_run; criterion_status is met, unmet, or unassessed. simulated_action_status is exactly not_applicable or BLOCKED: SK-12 uses BLOCKED; cases with no simulated dependent action use not_applicable. commands=[] is valid for reasoning-only cases. SK-01 requires ordered actual nonzero-red and zero-green command records with corresponding output artifacts; SK-11 requires its actual checker-negative-control command evidence. Each SK-12 subcase has a separately identifiable observation within its artifact.

The checker validates these fields and identities and reads each evidence local_path relative to the receipt directory, requiring the file to exist and its bytes/hash to match. Reject absolute paths, directory traversal, or symlink resolution outside that evidence directory. Remote artifacts must be materialized there through the already authorized transport before checking; the checker itself never fetches remote data. Missing materialized evidence fails the required check. Retain the entire evidence directory in the existing authorized durable destination, and bind its receipt identity in the handback. No receipt verifies its own hash; any detached receipt identity is recorded externally under the existing evidence mechanism. Raw argv/log artifacts remain private unless sanitized for authorized publication. Public summaries may use repository-relative or opaque evidence references; sanitization must not falsify the recorded command.

### 11.2 Additional exact checker boundaries

The independent reviewer checks the compiled allowlist and whole actual diff against the externally approved task. The checker's own list/zero exit cannot grant scope or establish source trust.

The five procedure bodies must match §5's frozen content, not merely word-count/link checks. The three skill suffixes preserve all baseline bytes. The Project edit is only TOML dependency/compat; src/CYAxiverse.jl and test/runtests.jl contain only their respective include additions; docs/make.jl changes only navigation. These narrow edits require independent actual-diff inspection as well as path checks. A 33-path candidate can still violate scope by changing forbidden content within a permitted file.

For the 0.22.36 source guard, do not turn an unqualified backend's UNSUPPORTED outcome into a successful positive test. A compatibility test can pass by correctly observing UNSUPPORTED, but qualification gates require actual proof on the qualified profile.

The receipt's complete allowed key sets are those in §11.1. Case objects have exactly id, evidence_status, criterion_status, simulated_action_status, actor_ref, assessment_ref, evidence and commands. Evidence objects have artifact_ref, local_path, sha256 and bytes; command objects have argv, exit_code, expected_exit and sequence. expected_exit is zero or nonzero; sequence is a positive integer; exit_code is an actual integer; bytes is a nonnegative integer. A missing process exit is not invented as zero. Reject bool-for-int; require canonical lower-case hashes and nonempty typed strings. Each case's evidence array and each command's argv are nonempty; commands=[] is valid when its evidence kind requires no command. Root object keys are schema, candidate_commit, candidate_tree, checker_sha256, case_pack_sha256, procedure_sha256 and cases. The procedure hash map has exactly the five §3 procedure paths. Apply §8.1's JSON/depth/string limits to receipts too.

The actual cases.json top-level array has exactly the twelve closed objects described in §10. methods is a nonempty string array of the selected upstream method names; task_input and supplied_fixture are nonempty strings; required_observations and forbidden_observations are nonempty string arrays; evidence_kind is one of executable_red_green, assessed_reasoning, payload_manifest or identity_control_and_assessment. Map SK-01, SK-09 and SK-11 to their respective special kinds and the other cases to assessed_reasoning. A criterion is not inferred met by parsing the receipt: the independent assessor checks its retained content.

Hash evidence by streaming bytes; reject nonregular files and symlink escapes (rejecting all evidence symlinks is permitted). Add one wrong-type/duplicate-key receipt control and the MX integration controls to the existing negative-control exercise. Use no pytest, network, new dependency or model call. The original five controls and their intended error codes remain mandatory.
