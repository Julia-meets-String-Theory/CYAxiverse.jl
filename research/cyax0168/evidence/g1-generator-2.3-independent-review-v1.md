# CYAX-0168 G1 Generator-2.3 Independent Review v1

Reviewed exact evidence head:
`655690ed7d81d34291c8924a24b6cccb5aa725bb`

Verdict: **PASS WITH REQUIRED REVISIONS**

The central outcome is correct: **CYAX-0168 G1 is NOT SATISFIED**. The
smallest executed failure is **inconclusive/invalid host-control evidence**,
not a backend failure or a CYAX-0168 G0 contract defect. The unavailable
`pmset -g therm` result prevents the required nominal thermal-state evidence
from being established. The approved contract was not weakened, no decision
fixture was generated or accessed, and CYAX-0168 G2--G4 were not entered.

## Checks reproduced by the reviewer

- The authorized start head and normative generator-2.3 head are ancestors of
  the reviewed head.
- The approved `spec.md`, `plan.md`, and `tasks.md` are unchanged.
- The historical generator-2.2 report and failure-artifact hashes are
  unchanged.
- The C0 exact comparison replayed with all seven advertised checks passing.
- The primary generator suite passed 7/7 tests.
- The independent generator suite passed 51/51 tests.
- `git diff --check` passed.
- No decision fixture was generated or accessed.

## Required revisions before a clean G1 rerun

1. Reconcile the report's ordinary available-capacity value with the older
   committed host manifest. Freeze and identify one current-run host manifest;
   ordinary available capacity is decision-bearing under the approved
   contract.
2. Extend the exact generator comparison beyond C0 to the required C1--C3
   calibration cells, compare canonical serialized bytes rather than only
   Python structures, and include the complete generator manifest.
3. Record the exact candidate evidence head and current-run host-manifest
   identity in the execution packet.
4. Durably record the independent reproducer's restricted input packet so its
   isolation is replayable rather than only manager-attested.
5. Remove the quadratic repeated filler-candidate-vector representation from
   the generator trace without dropping required trace information. A bounded
   C1 replay did not complete in the review window, so scale readiness is not
   established.

These findings do not authorize continued execution after the failed host
gate. They are preserved as prerequisites for any later clean CYAX-0168 G1
rerun. T9 and T10 remain incomplete.
