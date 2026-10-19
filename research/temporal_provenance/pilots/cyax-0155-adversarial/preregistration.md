# CYAX-0155 adversarial conflict/supersession experiment — preregistration

Status: Phase-0 freeze candidate. It becomes frozen only at the first Git commit that contains this file and the source fixtures, before paired contexts are regenerated and before any scored output is observed.

## Identity and ancestry

- Governing work item: Issue #163.
- Pilot history: Issue #155 and PR #156.
- Snapshot: `cyax-0155-adversarial-20260913T043555Z`.
- Branch: `research/cyax-0163-adversarial-supersession`.
- Base: exact reviewed/frozen draft PR #164 head `ec22874c3daeb6167cc957ed187f403ac86bdd22`.
- PR #164 remains unchanged and draft. This experiment will have a separate draft PR stacked on #164.
- No graph backend is selected or tested.

## Frozen sources

`source_snapshot.json` names ten finite source identities. The set includes the #155 body/current GitHub and Project state, its durable implementation checkpoint, the closure timeline event, PR #156 and its merge, the approved spec, the repository contract, the SDD authority material, the human guide, and a mechanical search for later resolving evidence. No resolving artifact was found. The closure event identifies actor and time but supplies no reason.

## Reconstruction question

> What is the authoritative current state of CYAX-0155, what evidence supports that state, what requirement or workflow condition was previously unresolved, whether durable evidence proves that condition was later satisfied or superseded, and what—if anything—remains uncertain?

The response must also give the next valid verification or action that does not invent an owner decision.

## Frozen answer key and rubric

`answer_key.md` defines K1–K12. Each item receives 1 only when the response states the required fact with the correct authority and temporal scope; otherwise 0. No partial points. Total is /12.

Conflict-critical subset (/5): K6, K7, K8, K9, and K11. These cover current state versus historical instruction, closure versus acceptance evidence, implementation versus Project/control-plane completion, and justified abstention.

Automatic failure occurs if a response:

- invents an owner decision;
- says the Research & Chats view was configured without durable evidence;
- says #155 is currently open;
- treats closure alone as proof that every earlier acceptance condition was satisfied;
- silently discards the historical keep-open condition;
- uses inaccessible/private context;
- confuses retrieval salience with authority; or
- claims either experimental representation is authoritative.

## Context construction and budget

Both contexts will be deterministically generated from the same frozen evidence inventory after this freeze commit.

- A: backend-independent relational/provenance records with source identity, observation facts, historical assertions, provenance, authority class, temporal order, implementation/support, requirement/acceptance, and evidence about supersession or its absence.
- B: high-quality structured prose with chronology, evidence cards, source references, and authority labels; no edge list, subject-predicate-object records, adjacency, traversal hints, or topology-encoding identifiers.
- Both provide evidence, not the answer-key inference.
- Count UTF-8 words with `len(text.split())` and serialized bytes with `len(text.encode("utf-8"))`.
- Target word-count difference: at most 5% of the larger context; no irrelevant padding.
- Record identical answer-bearing fact count, evidence-item count, and source-identity count.

## Subjects and launch order

- Six genuinely fresh agents.
- Model for every subject: GPT-5.6 Sol.
- Reasoning: high.
- Precommitted order: A1, B1, B2, A2, A3, B3 (`A B B A A B`).
- Each receives only `common_subject_prompt.md` plus its assigned frozen context.
- Subjects may not access live GitHub, web, chats, Control Desk memory, repository files, previous answers, other conditions, pilot scores, the answer key, or expected interpretation.
- Responses are preserved without repair.

## Parity and difficulty gates

Before launching A1, an independent GPT-5.6 Sol/xhigh read-only reviewer must confirm:

- identical answer-bearing facts, evidence items, and source identities;
- equivalent authority, temporal, and evidence-gap information;
- no answer-key leakage or condition-specific conclusion wording;
- word difference at most 5%; and
- the task is meaningfully harder than #157 because it requires resolving current-vs-historical status, missing closure rationale, implementation-vs-Project completion, and abstention.

If either context states the final inference so the subject can merely repeat it, revise and re-review before freeze. Only one repaired exact-revision re-review is permitted.

## Blind scoring

Before scoring, copy the six immutable responses to opaque IDs S1–S6 with condition/run labels removed. The scorer sees those files, the frozen answer key, and scoring instructions only. Reveal `scoring/condition_mapping.json` after all blind scorecards are frozen.

For every run record: total /12; conflict-critical /5; automatic failure and reasons; unsupported assertions; authority errors; temporal errors; supersession errors; unjustified inference; correct and incorrect abstentions; next-action correctness; source reopenings; and input word count.

## Interpretation

No statistical-significance language will be used for six runs.

- **Provisional A advantage:** only if A's advantage appears specifically in conflict/supersession reasoning, such as higher conflict-critical accuracy, fewer closure-implies-acceptance errors, better authority/provenance separation, or materially less context for equivalent conflict accuracy. A trivial aggregate difference is insufficient.
- **Evidence against near-term graph investment:** if B matches A on total accuracy, conflict-critical accuracy, authority, supersession, abstention, and next action at comparable size, strengthen the conclusion that explicit graph shape has not shown incremental value over high-quality structured provenance summaries in tested CYAxiverse reconstructions.
- **INCONCLUSIVE:** source conflict is resolved before freeze; semantic parity fails; contexts leak different conclusions; model variance dominates; scoring is ambiguous; or contamination occurs.

Do not manufacture a winner.

## Flow and backend boundary

Flow remains an unscored reference/comparator. Retain revision-pinned reads, evidence cards, mutation journals, project isolation, and compare-and-swap updates. Reject hard deletion, mutable replacement that erases history, recency/access-driven authority, and unreviewed agent memory as accepted state. Do not select or implement Flow/FalkorDB, Graphiti/Zep, NornicDB, Neo4j, RDF/OWL, or another backend.

## Stop and checkpoint

After independent final review, checkpoint the result on Issue #163 and the new draft experimental PR. Do not modify #155/#156; merge #164 or the experiment; approve the #163 spec; start fuzzy/Table-1 work; select a backend; or modify #162 private memory infrastructure.
