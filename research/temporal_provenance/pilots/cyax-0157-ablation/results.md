# CYAX-0157 held-out ablation results

## Frozen experiment

- Snapshot: `cyax-0157-ablation-20260913T0122Z`
- Preregistration freeze: `b1eea67cb3dd8dda6350a9372784f0174c8e2e02`
- Released context revision: `191cb52fcc0788b863d03f403f6710bcb600e544`
- Response freeze: `ddb2b7c0f6b4896a3257830d0320a30e43112d6a`
- Independent pre-run semantic-parity decision: PASS
- Model and reasoning for all subjects: GPT-5.6 Sol / high
- Launch order: A1, B1, B2, A2
- Source reopenings: zero for every run

## Context accounting

| Condition | Representation | Words | Bytes | SHA-256 | Sources | Evidence items | Facts |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| A | Temporal relational/provenance | 2,406 | 22,150 | `751159a373048d8d7159c07b8e4d3d55297298a025922403c14561b920132123` | 9 | 35 | 12 |
| B | Non-graph structured summary | 2,350 | 19,483 | `d9ebe46375f4e25b58087ac90f8860baa253b2f4fce1a810a17ebbf956f4a5b3` | 9 | 35 | 12 |

The context word-count difference is 56 words, or 2.328% of the larger context. Both contexts are inside the preregistered 2,300–2,500-word range.

## Blind scores after mapping reveal

| Run | Condition | Score | Automatic failures | Source reopenings | Stored output words |
| --- | --- | ---: | --- | ---: | ---: |
| A1 | A | 12/12 | None | 0 | 927 |
| A2 | A | 12/12 | None | 0 | 1,004 |
| B1 | B | 12/12 | None | 0 | 859 |
| B2 | B | 12/12 | None | 0 | 963 |

Condition A mean: 12.0/12. Condition B mean: 12.0/12. Difference: 0.0 points. There were no unsupported material assertions, incorrect abstentions, authority mistakes, supersession mistakes, next-action mistakes, or material provenance errors.

## Primary interpretation

Condition B matched Condition A on every preregistered reliability item with similar context size and zero reopenings. This experiment therefore provides no evidence that explicit graph-shaped relational structure improves reconstruction for the held-out CYAX-0157 pilot beyond careful structured summarization with provenance and authority boundaries.

The supported conclusion is narrower: the earlier clean-pilot benefit is currently attributable to curated structured context and provenance, not specifically to graph-shaped representation. This does not show that relational structure has no value in more adversarial conflict or supersession cases.

No graph backend is selected. The result narrows CYAX-0163 toward structured provenance/context assembly and requires stronger evidence before investment in graph infrastructure.

## Independent review and limitations

The independent methodology/evidence review of score freeze
`c52c4e0de711b55f824bd16c1fbfc438b5007a79` returned PASS. It reproduced the
freeze chronology, parity accounting, response hashes, blind mapping, score
totals, and privacy boundary. See `methodology_review.md` for the durable audit
record.

The experiment covers one work item with two runs per condition and a ceiling
score in both groups. It supports “no observed graph-shape benefit here,” not
statistical equivalence or general ineffectiveness. Static manifests record the
fresh-agent, launch-order, and zero-reopening controls but cannot independently
prove runtime isolation. The scorecard and mapping also share one commit; the
independent scorer attested to blind scoring, but Git alone does not establish
the reveal order.
