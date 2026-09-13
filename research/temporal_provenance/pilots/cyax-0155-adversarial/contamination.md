# Dispatch contamination record

Detected after all six raw responses and blind scorecards were frozen, before
condition-level interpretation or GitHub checkpointing.

The preregistration required every subject to receive the exact common prompt
and exact assigned frozen context. Instead, the manager sent inline
transcriptions/condensations:

- A1 received an inline near-complete relational transcription;
- B1 and B2 received the same inline structured-summary transcription;
- A2 and A3 received the same shorter relational transcription; and
- B3 received a shorter structured-summary transcription.

The inline prompts preserve the answer-bearing evidence, source identities,
authority labels, chronology, and evidence gap, but they are not byte-identical
to the frozen files. Replicates within A and B also did not all receive identical
input text. Therefore the planned input word counts in `runs/manifest.json`
cannot be treated as the actual dispatched counts, context-size fairness is not
established, and the runs do not satisfy the frozen isolation/input contract.

No subject used tools or reopened a source. No response was repaired. The
nominal scorecards remain useful diagnostic records but are not admissible for
an A-versus-B representation claim. Under the frozen interpretation rules the
experiment result is **INCONCLUSIVE due to experimental contamination**.

Repair requires a new preregistered execution revision that dispatches the exact
frozen bytes to six new fresh subjects and records exact per-run input hashes.
This run does not authorize that additional six-subject execution.
