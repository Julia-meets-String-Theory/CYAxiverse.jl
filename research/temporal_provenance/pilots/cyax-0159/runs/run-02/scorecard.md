# Run 02 scorecard

Scoring uses the pre-registered six-dimension rubric.

| Condition | Authority | Current | History | Evidence | Uncertainty | Action | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unstructured run-01 baseline | 2 | 1 | 2 | 2 | 2 | 1 | **10/12** |
| Derived context, strict source budget | 2 | 2 | 2 | 2 | 2 | 2 | **12/12** |

Automatic-failure classification: none in either response.

Reliability classification: **pass**. The run-02 response improves the frozen
baseline by two points, reaches full credit in every dimension, and preserves
the authority, lifecycle, branch, candidate, and evidence-quality boundaries.

Efficiency classification: **pass**. The run-02 agent opened no canonical
source after reading the derived context. Its measured input was 2,359 words,
compared with 6,215 measured words for the baseline before the baseline's
additional Git excerpts. This is a 62.0% reduction.

Pilot classification: **provisional pass for the first meaningful success
condition**. A fresh agent reconstructed the selected pilot more reliably and
with less measured context than the unstructured baseline. This result supports
the value of a provenance-aware derived context layer. It does not isolate a
graph-specific effect because manual curation and context structure remain
confounds. Run 02 followed a run-01-informed ledger repair and is not blind.
The baseline also lacked live GitHub access, while the derived input included
timestamped GitHub observations. Treat this as a provisional pilot result, not
a backend-selection result.
