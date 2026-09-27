# CYAX-0173 cross-fixture validation — proposed S2 plan

Governing intent: [spec.md](spec.md), draft pending exact independent Spec and
Scientific reviews and separate owner final S2 approval. This plan schedules
future evidence; it reports no fixture selection or study execution.

## Gate sequence

| Gate | Depends on | Future output and stop condition | Requirements |
| --- | --- | --- | --- |
| S2 approval | Frozen `spec.md`/`plan.md`/`tasks.md` | Fresh independent Spec and Scientific PASS on the same candidate; separate owner `approval_ref`. No study work before this gate. | XFV-01–14 |
| P0 currentness | S2 approval and a new execution handoff | Recheck live `vmm`, Issue #173, route source, B6 pins, capability policy, canonical data and scope. Material drift returns `BLOCKED_FOR_REBIND`. | XFV-01, XFV-13–14 |
| P0 manifest | Passing currentness | Metadata-only eligibility, N=5q+r ordered blocks, hash ranks, 30 selections, 20/10 labels, consumed-input provenance and prior-exposure register with durable source citations for known route outcomes; freeze exact manifest/digest before outcomes. Insufficient inventory stops. | XFV-01–03 |
| B6 control | Frozen manifest and exact B6 pins | Replay `B6-XFV2` fixed checks and retain exact output; failure stops interpretation without retuning. | XFV-04 |
| Discovery | Passing B6 | Execute only the approved ladder/route diagnostics for 20 discovery fixtures; preserve total endpoint states, precision readback, residuals, predictors, assembly/solve legs and raw evidence. | XFV-05–10 |
| Discovery freeze | Complete discovery accounting | Freeze correlations, candidate adjacent-partition boundary or exact negative/inconclusive status, code SHA-256 and all parameters before opening holdout. | XFV-10–11 |
| Holdout | Frozen discovery artifact | Unseal/run ten selected fixtures, apply the frozen rule without retuning, retain exposure and denominator/inconclusive evidence. | XFV-12 |
| Synthesis | Complete fixture accounting | Check >=24/30 converged references, report all outcomes and limitations; no production inference. | XFV-07, XFV-09, XFV-13 |
| Study review | Exact frozen later execution candidate | Fresh independent Spec and Scientific reviews on one commit/tree; unresolved or blocking findings return without promotion. | XFV-14 |

## Planned verification and evidence

- Replay selection from an independently sorted metadata inventory and the
  exact hash string. Assert five unique contiguous blocks with first-`r`
  remainder allocation, six selected per block, and fixed 20/10 split.
- Check no selected fixture with a route outcome was replaced, and verify
  manifest identities against consumed bytes, not configured paths.
- Verify the B6 replay exit/status and each fixed numerical gate separately.
- Read back actual precision at every ladder point and test restoration on
  success and failure. Preserve unresolved/failure status as data.
- Independently test boundary selection with synthetic equal values, both
  infinities, adjacent binary64 values, equality, empty classes, missing
  predictors and deterministic ties. The `LE x<=vi`/`GE x>=vj` partitions must
  retain the finite-side cut next to `+Inf`.
- Verify discovery artifact chronology precedes holdout unsealing; rebuild
  generated summaries deterministically and compare identities.
- Run `git diff --check`, applicable repository document checks, exact-path
  comparison, and a public privacy/provenance scan on the future candidate.

Study implementation is limited to a separately approved execution handoff.
The current preparation writes only the three contract documents. It does not
create the manifest, run B6, inspect new fixture outcomes, change production
code/data, merge, or close Issue #173.
