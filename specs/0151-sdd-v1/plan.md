# Implementation Plan — CYAX-0151

## 1. Governing specification

Canonical path: `specs/0151-sdd-v1/spec.md`

Status: **Approved**

Owner approval: 2026-09-10, durably recorded on GitHub Issue #151.

The first implementation change materializes the approved specification as a tracked repository file. No repository changes preceded the approved planning stage.

## 2. Implementation strategy

Implement CYAX-0151 as independently reviewable gates:

```text
G0  Approve SDD contract              PASS
 |
 v
G1  Repository SDD layer              process-only PR
 |
 v
G2  GitHub Project                    project configuration
 |
 v
G3  #148/#149 migration               separate process-only PR
 |
 v
G4  Six-week development backfill     Project metadata/reconciliation
 |
 v
G5  Usage evaluation                  after 6–8 weeks
```

This keeps the control-plane change small and prevents the SDD pilot or historical backfill from becoming prerequisites for landing the core templates.

## 3. Requirement coverage

| Requirement | Planned realization | Verification |
| --- | --- | --- |
| R-001 Constitution precedence | Minimal SDD section in `AGENTS.md`; SDD skill explicitly defers to it | Diff review for duplicated/conflicting policy |
| R-002 Spec precedence | `AGENTS.md`, SDD skill, spec template and PR convergence checklist | Cross-artifact review |
| R-003 No competing ledger | No active-work `specs/README`; Project is live map | Repository/file review |
| R-004 Risk beats line count | S0–S3 rules in skill + intake form | Template/skill inspection |
| R-005 Intent changes are spec changes | Agent stop/re-review rules | Skill + AGENTS review |
| R-006 Tasks not automatically Issues | Explicit task/sub-issue policy | Skill/template review |
| R-007 Scientific claim boundary | Required S2/S3 spec-template section | Template inspection |
| R-008 Normative ambiguity | Explicit escalation rule | AGENTS + skill review |
| R-009 Negative results | Scientific completion guidance | Spec template + skill |
| R-010 Partial gate PRs | PR template supports requirement/gate subsets | PR-template inspection |
| R-011 Unmet requirements visible | Convergence checklist/process | PR template + skill |
| R-012 Minimal AGENTS amendment | Small normative section only | Diff-size/content review |
| R-013 On-demand SDD skill | New canonical skill + existing adapter pattern | Symlink/content verification |
| R-014 Preserve #148 contract | Dedicated migration after G1/G2 | Manual scientific diff/convergence |
| R-015 Preserve gate history | G0/G1 represented retrospectively, not rewritten | #148/#149 migration review |
| R-016 No pilot science changes | Process-only #148 migration | Changed-file + semantic review |

## 4. G1 — Repository SDD layer

### 4.1 Branch boundary

Create one focused process branch from the then-current `vmm`:

`chore/sdd-v1-151`

The branch SHALL contain only control-plane/process artifacts.

No `src/`, scientific implementation, persisted-data, dependency or package-version change belongs in this PR.

### 4.2 Materialize the approved specification

Add:

```text
specs/0151-sdd-v1/
├── spec.md
├── plan.md
└── tasks.md
```

`spec.md` SHALL contain the owner-approved CYAX-0151 contract with `status: approved` and otherwise preserve its normative meaning.

### 4.3 Add the canonical SDD skill

Add:

```text
.agents/skills/cyaxiverse-sdd/
├── SKILL.md
├── agents/openai.yaml
└── templates/
    ├── spec.md
    ├── plan.md
    └── tasks.md
```

The skill SHALL be concise and procedural. It SHALL cover S0–S3 classification, discovery of an existing spec, drafting/clarification, owner-approval boundaries, requirement IDs, scientific claim boundaries, plan/task derivation, escalation, consistency analysis, convergence and GitHub coordination.

It SHALL NOT restate the full Julia/scientific/release policies already owned by `AGENTS.md` and existing specialist skills.

### 4.4 Preserve the consolidated skill architecture

PR #128 established `.agents/skills/` as the single editable source for project skills, mirrored into tool-specific directories with tracked symlinks. The new SDD skill SHALL follow that architecture.

Add:

```text
.codex/skills/cyaxiverse-sdd
.claude/skills/cyaxiverse-sdd
```

as tracked symlinks to the canonical `.agents/skills/cyaxiverse-sdd` directory.

### 4.5 Amend `AGENTS.md`

Add only the minimal normative SDD boundary:

- substantial work is classified S0–S3;
- S1–S3 work looks for a governing spec;
- S2/S3 consequential implementation requires approved intent;
- unresolved normative scientific choices return to the owner;
- changed intent returns to the spec;
- completion requires convergence;
- Issues/Projects do not supersede the spec;
- tasks do not automatically become Issues.

Detailed procedure remains in `cyaxiverse-sdd`.

### 4.6 Replace the generic feature intake

Add `.github/ISSUE_TEMPLATE/sdd_work.yml` for substantial new work and keep the bug-report path.

Retire the generic `feature_request.md` once the SDD form provides its replacement, so contributors do not see competing feature-entry routes.

The form SHALL collect intake information only; it SHALL NOT require a completed specification at issue creation.

### 4.7 Add the PR traceability template

Add `.github/pull_request_template.md` with governing Issue, spec path/revision, requirement/gate coverage, concise change, non-scope, scientific/API/schema/version impact, evidence matrix, convergence checklist and remaining work.

The template SHALL explicitly support partial-gate PRs.

### 4.8 Update the human workflow guide

Update `docs/AI_AGENT_AND_GIT_WORKFLOW.md` only enough to explain the new layer:

```text
AGENTS.md
    ↓
feature spec
    ↓
plan/tasks
    ↓
branch/PR/evidence
```

and point users to the SDD skill/templates. Do not duplicate the complete SDD handbook in the human guide.

### 4.9 G1 verification

Required checks:

1. `git diff --check`
2. `python3 scripts/agent_verify.py diff-check`
3. inspect changed-file scope;
4. validate Issue-form YAML syntax;
5. inspect all new skill mirrors as tracked symlinks to `.agents/skills/cyaxiverse-sdd`;
6. confirm no package/scientific/runtime files changed;
7. compare final `AGENTS.md` amendment against CYAX-0151 for contradiction or duplicated policy;
8. compare all templates against R-001–R-013;
9. final diff review before PR-ready state.

Package tests are not intrinsically required for process-only files that cannot affect package execution; remote CI remains the clean-checkout gate. If changed paths unexpectedly trigger or affect package behaviour, use the broader verification required by `AGENTS.md`.

### 4.10 G1 PR

Open one focused PR to `vmm`.

The PR SHALL reference #151 and state G0 approved, G1 is the PR's scope, G2–G5 remain future work, process-only/no version impact, and no #148 scientific migration occurs in this PR.

## 5. G2 — GitHub Project

After G1 lands, create **CYAxiverse Research & Development** and configure the approved workflow, fields, WIP policy, views and low-risk automation. Project state must remain a view over Issues/PRs rather than a competing project database.

## 6. G3 — #148/#149 pilot migration

After the SDD repository layer exists, use a separate focused branch/PR to create `specs/0148-catastrophe-continuation/{spec.md,plan.md,tasks.md}`. Preserve historical evidence and gate boundaries; do not rewrite history or alter the scientific contract.

## 7. G4 — Six-week Project backfill

Backfill approximately 2026-07-30 through 2026-09-10. Classify recent merged PRs for history, reconcile active work/dependencies/duplicates, and create only strategically useful migration specs.

## 8. G5 — Evaluation

After approximately 6–8 weeks of real use, assess traceability, Project usability, agent continuation burden, PR evidence mapping, stale/duplicate work, owner-review bottlenecks, specification overhead and S0 burden. Only then consider additional SDD automation.

## 9. Compatibility

No changes are planned to scientific behaviour, numerical behaviour, package APIs, persisted scientific schemas, dependencies, Julia support or `Project.toml` version.

## 10. Implementation stop conditions

Return to CYAX-0151 before continuing if implementation would require changing S0–S3 semantics, the authority hierarchy, task→Issue policy, scientific-owner approval semantics, the #148 scientific contract, the Project's non-authoritative role, or adopting Spec Kit wholesale; or if it would introduce scientific/package behaviour changes.

Ordinary formatting, template wording and technical layout choices that preserve the approved contract do not require re-approval.
