# Implementation Plan — CYAX-0170

## Governing specification

Canonical specification: `specs/0170-p0-numerical-equivalence/spec.md`

Spec revision reviewed for this plan: owner-approved CYAX-0170 P0 baseline,
approved in Issue #170 comment `5685945127`. Scientific reference remains
`7a40285bb5c313f7e8746b90644d5f45bb67be44`; independently reviewed evidence
candidate remains `ddc304f25040ff36bc84e7897d5b6dc5d16344f6`.

## Coverage

| Requirement / Gate | Planned implementation | Planned verification |
| --- | --- | --- |
| R-001, R-002 | Environment/load evidence and replay script | Re-run identity/load commands; hash artifacts |
| R-003 | Source and consumer inventory | Repository-wide `rg`; Julia method/module probes; spot-check persistence readers/writers |
| R-004-R-008 | Source archaeology and focused numerical probes | Line-linked source evidence plus executed analytic/pathological fixtures |
| R-009 | Versioned F1-F13 fixture corpus | Deterministic generation, content hashes, historical expected-output capture |
| R-010 | Physical/precision evidence | Float64 and currently supported BigFloat focused runs |
| R-011 | Future ownership/dispatch contract | Contract review against current APIs and JET/Aqua infrastructure |
| R-012 | B1-B7 benchmark suite | Bounded repeated runs with explicit warm-up/stat/allocation method |
| R-013, R-014 | `numerical_equivalence_contract-v1.md` | Trace each comparison/tolerance to source, repeatability, analytic, or high-precision evidence |
| R-015 | Evidence index and hashes | Privacy scan; repository-relative references; exact source/candidate identity |
| R-016, G2 | Fresh independent numerical review | Reviewer verdict tied to exact candidate commit |
| G0 | Owner review of draft | **Met:** approval comment `5685945127` recorded in spec metadata |
| G1 | Manager convergence | Spec-plan-task-evidence-diff checklist |
| G3 | Manager return packet | Exact reviewed revision and explicit later-phase stop |

## Existing architecture

The work observes current module-loading, numerical potential, critical-point,
inflation, spectrum, geometry/filesystem, persistence, and optional-extension
paths at the pinned revision. It does not move ownership or change these paths.

## Proposed approach

1. Record the source/worktree identity and a disjoint worker topology.
2. Draft and obtain approval of the S2 contract while read-only historical
   evidence is gathered; record the four route-specific dispositions without
   changing historical evidence.
3. Capture environment/load evidence and compatibility inventory independently
   from numerical-semantics archaeology and performance measurement.
4. Build deterministic validation-only fixtures/scripts. Record historical
   results, including failures, without patching production code.
5. Integrate evidence into a versioned numerical-equivalence contract and
   precommit tolerances before any later implementation is observed.
6. Run focused and repository-level verification appropriate to artifact-only
   changes; inspect the exact diff.
7. Commit the candidate, then dispatch a fresh independent reviewer against
   that exact commit. Correct and re-review any material changes.
8. Return the reviewed checkpoint and stop.

## Data/API/schema impact

No production API, HDF5 schema, dependency, or numerical behavior change is
planned. New content is limited to specification and validation/evidence
artifacts. Fixture files use validation-only versioned schemas.

## Verification strategy

- Static source/consumer inventory and module/method probes.
- Deterministic analytic and hostile F1-F13 runs.
- Named existing governed fixtures where bounded and available.
- Repeated B1-B7 measurements with environment and methodology identity.
- `scripts/agent_verify.py diff-check` and focused commands; package/audit/import
  checks where the validation scripts exercise package loading or Julia code.
- Hash manifest and privacy scan for durable evidence.
- Fresh independent numerical review of the exact candidate revision.

## Migration / compatibility

None. P0 records the historical contract and defines future acceptance gates.

## Approved P0 synchronization state

The owner-approved dispositions are synchronized mechanically across the spec,
contract, evidence index, and execution tasks:

1. F13 remains route-specific: validator routes retain historical non-finite
   rejection; direct critical-point propagation/failure remains a historical
   parity defect; no harmonization is performed.
2. The governed/source-faithful top-level N=5 route is authoritative for future
   modularisation/equivalence gates. Stale nested legacy N=5 behavior is
   historical only and is not the scientific oracle.
3. Float64 and high-precision B6 baselines remain separate. Numerical/physical
   authority is unresolved; no silent oracle choice is allowed.
4. Observed `2pi^2 I` remains historical defect evidence, not intended behavior.
   The `4pi^2 I` correction is separate Issue #172 work, and Issue #173 remains
   separate.

The synchronization reconciles this artifact-only branch with contemporary
`vmm` revision `4523ab090c70ba4fcf3b1711a028d19f2b8e2a62` as an integration/base
update. It does not rebase the scientific reference, alter empirical evidence,
or authorize P1/P2/P3A/P3B. PR #171 metadata is a separate lifecycle surface;
the manager should confirm the live PR description records the same state. The
contemporary-vmm `.gitignore` tightening was cleanly accounted for: this branch
retains its existing P0 validation allow-list so authorized validation artifacts
remain tracked, and no `.gitignore` change is included in the candidate.

## Risk and stop conditions

Return to the owner if routes contradict the assumed contract, support/order or
normalization is scientifically ambiguous, tolerances require later output,
fixture creation requires production changes, external evidence cannot be
identified durably, or scope crosses existing scientific ownership.

## Task decomposition

Four evidence workers use disjoint write sets. The Manager owns all normative
artifacts, integration, exact-diff review, candidate commit, and return. The
independent reviewer is created only after contributing workers finish and does
not edit the candidate.
