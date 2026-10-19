# CYAxiverse Agent Surface Allocation Policy

**Status:** Adopted  
**Scope:** CYAxiverse development, research, review, documentation, and agent orchestration  
**Purpose:** Define how execution surfaces are selected for CYAxiverse work without conflating interface choice with role, authority, model configuration, permissions, or agent topology.  
**Product capability assumptions last reviewed:** 2026-09-23

## 1. Core model

CYAxiverse treats the following as separate dimensions:

~~~text
role
  → determines responsibility

task packet
  → determines authority and scope

model / configuration
  → determines agent capability

permissions
  → determine allowed effects

execution mode
  → determines the appropriate surface

context isolation
  → determines review independence
~~~

These dimensions MUST NOT be inferred from one another unless an applicable governing policy explicitly establishes such a relationship.

In particular:

> **A surface is an execution environment, not an authority level, role assignment, model selection, permission profile, or worker topology.**

Surface allocation is therefore based on:

~~~text
role + dominant execution mode + required controls
~~~

rather than on role alone.

## 2. Terminology and canonical surface allocation

This section is the **single normative source** for surface allocation.

### 2.1 Product terminology

For this policy:

#### Chat

The conversational ChatGPT environment used for:

- reasoning;
- discussion;
- task refinement;
- lightweight investigation;
- specification work;
- Control Desk activity;
- Review Desk activity.

#### Work

The ChatGPT environment used for substantial:

- research;
- analysis;
- file and document workflows;
- cross-source reconciliation;
- cross-application workflows;
- evidence-bearing research;
- production of finished research or knowledge artifacts.

#### Codex app

The dedicated Codex software-development environment.

It is generally preferred for:

- autonomous repository work;
- worktree-based execution;
- parallel agent execution;
- repository-scale changes;
- longer-running software tasks;
- Manager orchestration;
- implementation that does not require continuous editor interaction.

#### Codex IDE extension (VS Code)

The Codex integration operating within VS Code.

It is generally preferred for implementation dominated by:

- direct editor interaction;
- selected/open source context;
- Julia REPL use;
- debugger interaction;
- rapid edit/test/debug cycles;
- interactive inspection of local diffs and test output.

The term **surface** refers to one of these execution environments.

These product-specific defaults MAY be revised as product capabilities evolve without changing the underlying role, authority, permission, or independence architecture of this policy.

### 2.2 Canonical allocation table

| Role / activity | Dominant execution mode | Default surface |
|---|---|---|
| Control Desk | reasoning, task refinement, reconciliation | **Chat** |
| Control Desk | substantial multi-source or cross-application workflow | **Work** |
| Manager | repository orchestration, worker coordination, worktrees, autonomous execution | **Codex app** |
| Implementer | autonomous, worktree-based, parallel, or longer-running repository implementation | **Codex app** |
| Implementer | editor-, REPL-, debugger-, or rapid-feedback-heavy implementation | **Codex IDE extension (VS Code)** |
| Independent code reviewer | repository/candidate inspection | **fresh Codex context**, using Codex app or IDE according to execution mode |
| Independent architecture / handoff reviewer | document/specification reasoning | **fresh Chat context** |
| Independent scientific / methodological reviewer | source-bearing or analytical review | **fresh Chat or Work context**, according to research depth |
| Scientific discussion / question formulation | conversational reasoning | **Chat** |
| Lightweight scientific investigation | bounded search and reasoning | **Chat** |
| Literature survey / source reconciliation | evidence-bearing research | **Work** |
| Finished research artifact | sustained research and synthesis | **Work** |
| Repository experiment arising from research | software execution | **Codex app or IDE**, according to execution mode |
| Issue / specification / architecture drafting | reasoning and refinement | **Chat** |
| Durable knowledge / wiki architecture | synthesis and project knowledge | **Chat or Work** |
| Code-coupled documentation | implementation | **Codex app or IDE**, according to execution mode |

This table defines defaults, not authority.

A task-specific packet MAY override a default surface choice when doing so remains consistent with higher-order governance, review, safety, permission, and independence requirements.

## 3. Surface allocation is orthogonal to agent configuration

Surface allocation MUST NOT implicitly determine:

- model;
- model family;
- reasoning effort;
- context budget;
- permission profile;
- available skills;
- reviewer configuration;
- implementer configuration;
- Manager configuration;
- number of workers;
- worker hierarchy;
- degree of concurrency;
- use of subagents;
- delegation topology.

For example:

~~~text
Manager → Codex app
~~~

means only that the Codex app is normally the appropriate execution environment for Manager-style repository orchestration.

It does **not** mean:

~~~text
Codex app
    → selects the Manager model
    → authorizes delegation
    → determines worker models
    → determines number of workers
    → determines permissions
~~~

Those choices MUST be governed independently.

Where model/configuration symmetry or freezing is required for controlled comparisons or independent review, that requirement MUST be specified separately from surface allocation.

## 4. Dispatch dimensions

Every execution dispatch MUST establish enough information to determine the agent's actual authority.

At minimum, an execution packet or governing source MUST determine:

1. **role;**
2. **task scope;**
3. **authorized repository or artifact state;**
4. **execution surface or surface-selection rule;**
5. **write permissions;**
6. **prohibited actions;**
7. **required validation;**
8. **required review state, if any.**

Where relevant it SHOULD additionally specify:

- model or acceptable model class;
- reasoning configuration;
- skills;
- allowed tools;
- worker topology;
- concurrency;
- worktree requirements;
- branch constraints;
- external side-effect permissions.

Surface selection cannot supply missing authority.

## 5. Execution permissions

Permissions are a mandatory and independent execution-control dimension.

### 5.1 Implementers

An Implementer MUST modify only:

- the authorized repository/worktree;
- the authorized paths or scope;
- the candidate state permitted by the task packet.

An Implementer MUST NOT infer permission to:

- merge;
- push;
- create or modify pull requests;
- write GitHub issues;
- modify external services;
- publish artifacts;
- change unrelated files;

merely because its execution surface technically permits those actions.

Such actions require authority under the governing task and project policies.

### 5.2 Reviewers

Independent reviewers SHOULD normally operate read-only with respect to the reviewed candidate.

A reviewer MAY execute **candidate-preserving** inspection and validation commands where permitted.

Such commands MAY create:

- temporary files;
- caches;
- compilation artifacts;
- precompilation state;
- build products;
- coverage output;
- benchmark data;
- diagnostic state;

provided they do not alter the reviewed candidate.

If validation alters candidate-relevant state, the original candidate identity MUST be restored and verified before a verdict is issued.

An independent reviewer MUST NOT intentionally modify the candidate being reviewed.

If a reviewer makes a candidate-changing edit:

~~~text
Reviewer role ends
        ↓
candidate changes
        ↓
successor candidate exists
        ↓
fresh independent review required
~~~

The resulting assessment MUST NOT be described as an independent review of the modified candidate unless that successor candidate is independently reviewed.

### 5.3 External side effects

Actions such as:

- merge;
- push;
- PR creation;
- issue/comment writes;
- releases;
- publication;
- mutation of external services;

MUST follow the authorization rules of the governing project policy and task packet.

Execution capability is not execution permission.

## 6. Role-specific guidance

### 6.1 Control Desk

The Control Desk is responsible for:

- task refinement;
- scientific and architectural discussion;
- reconciliation with durable project state;
- issue/specification preparation;
- preparation of complete Manager-facing handoffs;
- incorporation of returned implementation/review evidence;
- maintenance of coherent project understanding.

Its default surface is **Chat**.

Use **Work** when the task materially depends on:

- substantial source reconciliation;
- large document sets;
- cross-application activity;
- extensive research;
- production of evidence-bearing artifacts.

The Control Desk MUST NOT become the downstream Implementer merely because source-control or coding tools are available.

The Control Desk MUST NOT launch or create downstream Manager, Implementer, or Reviewer agents as part of its normal dispatch role.

Its normal terminal dispatch state is:

~~~text
READY TO DISPATCH
~~~

The owner performs the initial dispatch of that packet.

### 6.2 Manager

The Manager is responsible for turning an approved Manager-facing handoff into controlled downstream execution.

Typical responsibilities include:

- verifying repository state;
- validating baseline identity;
- selecting worker topology;
- determining whether delegation is warranted;
- deriving bounded worker packets;
- launching authorized downstream workers;
- coordinating parallel execution when justified;
- maintaining scope discipline;
- ensuring validation occurs;
- freezing candidate identity where required;
- arranging required independent review;
- collecting evidence;
- producing a structured handback.

The **Codex app** is the default Manager surface because Manager work commonly involves:

- worktrees;
- multiple workers;
- repository-wide state;
- autonomous commands;
- long execution chains;
- parallel work.

This default does not prescribe Manager model, worker models, topology, or permissions.

A Manager MAY dispatch Implementers and Reviewers only where the governing handoff or higher-order policy grants bounded delegation authority.

Such downstream delegation MUST NOT expand:

- task scope;
- write permissions;
- scientific authority;
- external side effects;
- review authority;
- worker topology beyond the authorized bounds.

### 6.3 Implementer

The Implementer performs bounded repository changes authorized by its task packet.

There is **no single mandatory Implementer surface**.

Surface selection depends on implementation mode.

#### Use Codex app when implementation is predominantly:

- autonomous;
- longer-running;
- worktree-based;
- repository-scale;
- parallelizable;
- multi-file;
- command-driven;
- naturally delegable.

#### Use Codex IDE extension (VS Code) when implementation is predominantly:

- editor-coupled;
- interactive;
- REPL-driven;
- debugger-driven;
- selection/context-driven;
- rapid-feedback-heavy;
- dependent on repeated local edit/test/inspect cycles.

For Julia development, VS Code will often be preferable when the inner loop involves:

~~~text
inspect
  ↓
edit
  ↓
Julia REPL
  ↓
test
  ↓
debug
  ↓
inspect diff
  ↺
~~~

The Codex app may remain preferable for substantial autonomous implementation even when the assigned role is Implementer.

## 7. Independent review

Independent review is defined by **context and role separation**, not by application separation.

### 7.1 Independence invariant

When the governing task requires **independent review**, the reviewer MUST operate in a fresh review context.

If fresh context is not used, the resulting assessment MUST NOT be described as independent review.

A different application window alone is insufficient.

#### Definition: fresh review context

**Fresh review context** means that the reviewer does not inherit the Implementer's:

- conversational context;
- implementation-agent state;
- private scratch context;
- self-authored justification;
- mutable working assumptions.

Fresh context does **not** require:

- a different application;
- a different model;
- a different repository clone;
- a different worktree.

The reviewer MAY inspect the same frozen candidate state.

Thus:

> **fresh epistemic context ≠ fresh repository copy**

unless a governing policy independently requires repository isolation.

### 7.2 Reviewer inputs

An independent reviewer MUST receive or independently retrieve the authoritative inputs required to judge the candidate.

Where they exist and govern the review, these include:

~~~text
governing specification / handoff
+
exact candidate identity
+
applicable review rubric
+
required evidence
+
authorized repository state
~~~

A rubric need not be invented merely because this policy lists one; the requirement applies where such a rubric exists and governs the review.

The reviewer MUST inspect the candidate itself.

The reviewer MUST NOT rely solely on:

- implementation conversation/context;
- Implementer self-description;
- self-authored justification;
- mutable working assumptions;
- claims of successful validation that can independently be checked.

### 7.3 Candidate identity

Before any **verdict-bearing independent review**, the candidate MUST be bound to an exact identity sufficient to reconstruct all reviewed content.

Verdict-bearing review includes assessments such as:

~~~text
PASS
PASS_WITH_NONBLOCKING_FINDINGS
REQUEST_CHANGES
READY_TO_MERGE
~~~

or any equivalent state-bound verdict.

Candidate identity MAY be established through:

- commit identity;
- tree identity;
- blob identity;
- cryptographic hash;
- exact artifact bytes;
- another immutable mechanism defined by the governing review protocol.

The identity mechanism MUST cover **all in-scope reviewed state**.

A Git commit alone is insufficient where the reviewed candidate also includes:

- uncommitted worktree changes;
- untracked in-scope files;
- external generated artifacts;
- separately bound evidence;
- other candidate-relevant state not represented by that commit.

Before issuing a verdict, the reviewer MUST be able to determine what exact state the verdict applies to.

### 7.4 Successor candidates

Independent review evaluates a defined candidate.

If review causes or requests any candidate-changing modification:

~~~text
Candidate N
    ↓
independent review
    ↓
change requested
    ↓
candidate modified
    ↓
Candidate N+1
~~~

then Candidate N+1 is a new review object.

Any review requirement MUST be applied to the successor candidate according to the governing review policy.

A prior verdict MUST NOT silently transfer to changed candidate state.

## 8. Normative lifecycle

CYAxiverse distinguishes **pre-dispatch review** from **post-implementation review**.

The standard lifecycle is:

~~~text
Control Desk
     ↓
governing issue / specification / handoff candidate
     ↓
[required pre-dispatch independent review]
     ↓
READY TO DISPATCH
     ↓
owner initial dispatch
     ↓
Manager
     ↓
authorized Manager-dispatched Implementer(s)
     ↓
validation / evidence collection
     ↓
freeze exact implementation candidate identity
     ↓
[required implementation independent review]
     ↓
Manager handback
     ↓
Control Desk reconciliation
~~~

Bracketed review stages occur **when required by the governing protocol**.

Not every task requires both review stages.

The two review classes answer different questions.

### Pre-dispatch independent review

Asks:

> **Is this the correct, sufficiently specified, properly authorized task to execute?**

Typical review objects include:

- governing issues;
- specifications;
- architectural plans;
- handoffs;
- experiment protocols.

### Post-implementation independent review

Asks:

> **Does this exact implementation candidate satisfy the governing task?**

Typical review objects include:

- commits;
- diffs;
- source trees;
- generated artifacts;
- validation evidence.

Not every task requires every role, but stages MUST NOT be implicitly collapsed where governing policy requires separation.

Where single-agent implementation is explicitly authorized:

~~~text
Manager + Implementer responsibility
~~~

may be combined.

Where independent review is mandatory:

~~~text
Implementer + Independent Reviewer
~~~

MUST NOT be combined for the same candidate.

## 9. Dispatch authority

Dispatch occurs at two distinct levels.

### 9.1 Initial owner dispatch

The owner remains the authority for initial dispatch of a Control-Desk-produced:

~~~text
READY TO DISPATCH
~~~

packet.

The Control Desk stops at that boundary.

The normal initial flow is:

~~~text
Control Desk
    ↓
READY TO DISPATCH
    ↓
Owner
    ↓
Manager
~~~

### 9.2 Bounded Manager downstream dispatch

After initial owner dispatch, an authorized Manager MAY launch and coordinate:

- Implementers;
- Reviewers;
- other explicitly permitted downstream workers;

within the delegation authority granted by the governing packet or higher-order policy.

The resulting topology is:

~~~text
Owner
  ↓
Manager
  ├── Implementer A
  ├── Implementer B
  └── Independent Reviewer
~~~

where such topology is authorized.

Manager delegation MUST NOT independently expand:

- task scope;
- repository scope;
- permissions;
- external side effects;
- scientific claims;
- review authority;
- prohibited actions.

The Manager remains accountable for preserving the governing execution boundary across all derived worker packets.

## 10. Surface transitions and role transitions

A **role transition** and a **surface transition** are different events.

### 10.1 Role transition

Example:

~~~text
Manager
   ↓
Implementer
~~~

This changes responsibility and potentially authority.

It requires explicit authorization or a valid downstream task packet.

### 10.2 Surface migration with unchanged role

Example:

~~~text
Implementer in Codex app
        ↓
Implementer in VS Code
~~~

where the agent moves to VS Code because Julia debugging becomes REPL-heavy.

This is an **execution migration**, not automatically a new authority handoff.

The existing bounded packet continues to govern unless:

- the role changes;
- authority changes;
- repository state changes outside the packet's assumptions;
- the governing protocol requires a new handoff.

Likewise:

~~~text
Chat → Work
~~~

does not automatically change the agent's role.

### 10.3 Authority does not travel implicitly

Changing surfaces never expands authority.

The following is invalid:

~~~text
VS Code exposes GitHub action
        ↓
therefore GitHub write is authorized
~~~

Capabilities are constrained by the governing packet and permissions.

## 11. Research allocation

Research surface selection depends on whether the activity is conversational or evidence-bearing.

### 11.1 Chat

Use **Chat** for:

- scientific discussion;
- hypothesis formation;
- question formulation;
- conceptual reasoning;
- lightweight investigation;
- deciding what research is required.

### 11.2 Work

Use **Work** for:

- literature surveys;
- systematic source collection;
- arXiv / INSPIRE reconciliation;
- methodology comparison;
- evidence-bearing research;
- analysis across multiple files or sources;
- finished research briefs;
- research artifacts intended to become durable project evidence.

### 11.3 Research-driven repository experiments

Research role and execution surface need not be identical.

A research investigation may produce a need for:

- numerical experiment;
- Julia script;
- benchmark;
- test implementation;
- repository inspection.

That execution MAY move to Codex app or the IDE without changing the scientific purpose of the task.

The governing research question and evidential requirements MUST remain explicit.

## 12. Context discipline

Surface allocation is also a context-management mechanism.

### 12.1 Avoid unnecessary repository duplication

Where a downstream agent has authorized repository access, prompts SHOULD generally identify:

- paths;
- symbols;
- commits;
- tests;
- requirements;

rather than embedding large quantities of repository source text.

For example:

~~~text
Inspect src/Foo.jl and its tests.
Implement requirement R3 from the governing packet.
~~~

is preferable to copying the full files into the packet when the files are directly available.

### 12.2 Preserve semantic context

Direct repository access does not replace scientific, architectural, or governance context.

Task packets MUST explicitly preserve requirements that cannot safely be reconstructed from source alone, including:

- scientific invariants;
- physical conventions;
- normalization assumptions;
- owner decisions;
- prohibited behavior;
- scope boundaries;
- review requirements;
- validation criteria.

### 12.3 No implicit conversational memory requirement

A downstream packet MUST NOT depend on statements such as:

~~~text
as discussed earlier
~~~

when the referenced information is required to perform the task correctly.

Necessary information must exist in the packet or in explicitly identified authoritative sources.

## 13. Surface-selection decision procedure

When selecting a surface, determine in order:

### A. What is the role?

Examples:

- Control Desk;
- Manager;
- Implementer;
- Reviewer;
- Researcher.

### B. What is the dominant execution mode?

Examples:

- conversational reasoning;
- multi-source research;
- autonomous repository work;
- parallel worktree execution;
- editor interaction;
- REPL/debugger loop;
- exact-state review.

### C. What controls are mandatory?

Determine:

- independence;
- write permissions;
- candidate immutability;
- review requirements;
- scientific risk;
- external side-effect restrictions.

### D. What is the smallest sufficient execution environment?

Choose the:

> **smallest sufficient environment consistent with required independence, permissions, evidence, risk controls, and execution mode.**

Convenience alone MUST NOT override a required control.

## 14. Risk-based role combination

Role combination MUST be based on risk and governance rather than file count.

A Manager MAY also perform implementation when the governing task explicitly permits a single-agent topology and the work is sufficiently controlled.

Relevant factors include:

- semantic risk;
- scientific risk;
- scope clarity;
- reversibility;
- validation quality;
- blast radius;
- independence requirements;
- external side effects;
- explicit authorization.

A one-line numerical or scientific change MAY require greater separation and review rigor than a large mechanical edit.

Examples of potentially high-risk small changes include changes affecting:

- normalization;
- basis conventions;
- Hessians;
- physical units;
- instanton actions;
- numerical tolerances with scientific meaning;
- sign conventions.

File count MAY inform complexity assessment but MUST NOT determine review rigor.

## 15. Owner-controlled initial dispatch

The owner remains the authority for initial dispatch from the Control Desk into the downstream execution hierarchy.

The standard sequence is:

~~~text
1. Control Desk and owner refine the task.

2. Governing issue / specification / handoff undergoes any
   required pre-dispatch review.

3. Control Desk produces the complete governing handoff.

4. Handoff reaches READY TO DISPATCH.

5. Control Desk stops.

6. Owner dispatches the Manager packet.

7. Manager exercises only such downstream delegation authority
   as the governing packet permits.

8. Implementers perform authorized work.

9. Required validation and candidate binding occur.

10. Required independent review occurs in fresh context.

11. Manager produces handback evidence.

12. Control Desk reconciles the result into durable project state.
~~~

No interface automatically grants downstream dispatch authority merely because it technically supports spawning or coordinating agents.

## 16. Anti-patterns

### 16.1 Role-by-interface reasoning

Avoid:

~~~text
Manager = Codex app
Implementer = VS Code
~~~

as a universal rule.

Correct formulation:

~~~text
role + execution mode + controls → surface
~~~

### 16.2 Surface-as-authority reasoning

Avoid:

~~~text
the interface can perform the action
therefore the agent may perform the action
~~~

Capability does not confer authority.

### 16.3 Surface-as-model reasoning

Avoid assuming that opening a task in:

- Codex app;
- VS Code;
- Chat;
- Work;

implicitly selects an approved model, reasoning effort, or worker profile.

### 16.4 Owner-dispatch-every-worker reasoning

Avoid interpreting owner-controlled initial dispatch as requiring the owner to manually launch every downstream worker.

Where Manager delegation is authorized:

~~~text
Owner → Manager → bounded downstream workers
~~~

is the intended architecture.

### 16.5 Manager as universal implementer

The Manager SHOULD NOT absorb implementation simply because it can edit the repository.

Role combination requires a deliberate topology decision.

### 16.6 Chat as prolonged mechanical coding environment

Large edit/test/debug loops SHOULD normally move to Codex.

### 16.7 IDE as durable project memory

Editor context is working context, not authoritative project memory.

Durable decisions and evidence MUST be recorded in the appropriate authoritative source.

### 16.8 Reviewer editing the reviewed candidate

A reviewer who modifies the candidate has ceased performing independent review of that candidate.

Any changed successor candidate requires its own applicable review.

### 16.9 “Independent” review without fresh context

A review performed with inherited implementation context MUST NOT be labelled independent review.

### 16.10 Verdict without candidate identity

A state-bound verdict MUST NOT be issued against an ambiguously defined or only partially bound candidate.

## 17. Policy precedence

This policy governs execution-surface allocation only.

It MUST NOT override higher-order CYAxiverse governance.

Unless another governing policy explicitly defines a different hierarchy, precedence is:

~~~text
1. Higher-order repository / project governance

2. Governing scientific, safety, review, and execution policies

3. Task-specific specification or handoff,
   insofar as it is consistent with 1–2

4. Agent-role and review procedures

5. This surface-allocation policy
~~~

A task packet MAY explicitly override this policy's **default surface choice**.

It MUST NOT use that override to bypass:

- repository governance;
- independent-review requirements;
- scientific controls;
- permission boundaries;
- authorization requirements.

An inconsistent lower-order instruction is not made authoritative merely because it is task-specific.

## 18. Lifecycle state and evidence

Surface choice SHOULD remain visible as part of execution provenance where doing so materially assists reproducibility or auditability.

For substantial tasks, the execution record SHOULD permit reconstruction of:

- role;
- selected surface;
- model/configuration where governed;
- permission profile;
- worker topology where relevant;
- baseline repository state;
- resulting candidate identity;
- validation performed;
- reviewer identity/configuration where governed;
- final review state.

The objective is not to record interface trivia, but to preserve execution facts that may explain differences in behavior or evidential quality.

## 19. Non-normative summary

The following is a convenience summary only. Section 2 remains authoritative.

~~~text
                        CYAxiverse
                             │
          ┌──────────────────┼───────────────────┐
          │                  │                   │
       Chat / Work        Codex app          Codex IDE
          │                  │                   │
 discussion/research      orchestration       interactive
 Control Desk             autonomous work     implementation
 Review Desk              worktrees           Julia REPL
 specifications           parallel agents     debugger
 durable synthesis        long tasks          rapid feedback
~~~

A single role may legitimately operate on more than one surface.

For example:

~~~text
Implementer
    ├── autonomous implementation → Codex app
    └── interactive debugging     → Codex IDE
~~~

The governing conceptual model is:

~~~text
responsibility → role

authority → packet

capability → model/configuration

allowed effects → permissions

execution environment → execution mode / surface

review independence → fresh isolated context
~~~

## 20. Default operational rule

For routine CYAxiverse work:

> **Discuss and refine in Chat.  
> Perform evidence-bearing research in Work.  
> Orchestrate autonomous repository work in the Codex app.  
> Use the IDE when implementation becomes editor-, Julia-, debugger-, or feedback-loop intensive.  
> Conduct required independent review in a fresh isolated context.**

These are defaults, not substitutes for governance.

## 21. Policy evolution

Amendments to this policy SHOULD be justified by observed workflow evidence.

A proposed change SHOULD identify:

- observed failure or inefficiency;
- affected role and execution mode;
- proposed allocation change;
- effect on correctness;
- effect on scientific risk;
- effect on independence;
- effect on permissions;
- effect on token/context efficiency;
- effect on reproducibility;
- new failure modes introduced.

Surface preferences SHOULD evolve when evidence supports doing so.

Role boundaries, authority boundaries, and independent-review requirements MUST NOT be weakened merely to accommodate a preferred interface.
