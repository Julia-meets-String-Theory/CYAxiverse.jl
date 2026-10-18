# CYAX-0155 adversarial experiment — frozen answer key

This file is never shown to experimental subjects. Score each item 0 or 1 with no partial credit.

1. **K1 — governing item:** Issue #155 is the governing work item for the privacy-safe conversation-checkpoint change.
2. **K2 — supporting PR:** PR #156 is supporting implementation/integration evidence for #155, not the governing work item or an independent authority layer.
3. **K3 — current Issue state:** At the frozen observation, #155 is `CLOSED` with state reason `COMPLETED`; its `CYAxiverse Research & Development` Project item is `Done`.
4. **K4 — PR outcome:** PR #156 merged to `vmm` at `f0013552cd69a93221464e9c8ccfd56339f39052`; the repository diff implemented the planned four-file control-plane change.
5. **K5 — repository implementation:** The repository-side privacy/publication boundary and checkpoint procedure are implemented and active, as established by the merge plus the checkpoint comment.
6. **K6 — historical keep-open condition:** The 2026-09-11T00:01:47Z checkpoint explicitly said to keep #155 open until the `Research & Chats` saved Project view/filter was configured.
7. **K7 — distinct completion domains:** Repository implementation completion is distinct from Project/control-plane view configuration. The checkpoint and PR body stated that the available tooling could not configure or verify the Projects-v2 view and did not claim it complete.
8. **K8 — no closure-to-acceptance inference:** Current closure/Done state does not by itself prove the saved view was configured or that every earlier workflow/acceptance condition was satisfied.
9. **K9 — current versus historical:** The earlier keep-open instruction is historical evidence, not the current GitHub status. The authoritative current state is closed/Done, while the historical condition remains relevant to evidential completeness.
10. **K10 — no demonstrated resolution/supersession:** The closure event shows that `vmmhep` closed #155 at 2026-09-11T01:09:18Z without a commit or GitHub App. The frozen sources contain no rationale, no proof of saved-view configuration, and no explicit durable statement superseding the keep-open condition.
11. **K11 — abstention:** The response explicitly abstains from claiming that the saved view was configured, that the condition was satisfied, or that it was deliberately superseded. It distinguishes known current state from unknown rationale/completion evidence.
12. **K12 — next valid action:** Verify the live Project's saved views/filter configuration or obtain a durable owner statement explaining/superseding the keep-open condition. Do not automatically reopen or alter #155 from the frozen evidence alone.

Conflict-critical subset: K6, K7, K8, K9, K11 (/5).

Automatic failure conditions are exactly those in `preregistration.md`; a triggered automatic failure is reported separately and makes the run fail even though the raw rubric score remains recorded for diagnosis.
