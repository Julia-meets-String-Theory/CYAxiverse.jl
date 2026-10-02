# Tasks — CYAX-0197

## Rules

- CYAX-0197 is S2 scientific/persisted-contract work.
- No production/source implementation begins before G0 approval.
- \`spec.md\` is authoritative over this task list.
- \`tasks.md\` is execution/evidence decomposition, not live GitHub status.
- New scientific/schema choices return to the owner/spec; they are not inferred
  by the implementation worker.

## Phase 0 — Governing contract

- [ ] **T001 [G0] Independently review exact S2 specification**
  - Input: Issue #197 plus exact \`spec.md\` revision.
  - Review axes:
    - Spec/contract completeness;
    - Scientific/Numerical reconstruction correctness.
  - Verify: both reviews bind the exact candidate revision and contain no
    blocking finding.
  - Escalate if: schema dispatch, metric normalization, sparse-tensor semantics,
    hashing, tolerances, or scope remain ambiguous.

- [ ] **T002 [G0] Record owner approval**
  - Expected output:
    - \`status: approved\`;
    - exact durable \`approval_ref\`;
    - reviewed revision/content identity.
  - Verify: no normative bytes changed after the passing reviews without fresh
    review.
  - Escalate if: owner changes any scientific/schema requirement.

## Phase 1 — Component reconstruction

- [ ] **T101 [R-001, R-002, R-009, R-014, R-015, G1] Implement strict schema dispatch and metadata parsing**
  - Expected output: exact dense-vs-compact classification before scientific
    array reads.
  - Verify:
    - exact supported compact markers accepted;
    - marker-free dense accepted without changing stored orientation;
    - v2/v3/v4/v5 dense classes accepted;
    - v8 and v8-QED-assignment dense classes accepted;
    - transitional dense v9 accepted only when compact potential markers are
      absent;
    - unknown/partial/hybrid states fail closed;
    - metadata duplicate conflicts fail;
    - exact source-dataset list and literal basis/intersection conventions are
      verified;
    - \`GeometryIndex.h11\`, persisted \`h11\`, tip/effective-cone/metric/Q
      dimensions agree;
    - compatible glsm/basis-matrix/prime-label dimensions agree;
    - parsing is native Julia with only declared dependencies.
  - Escalate if: a schema/writer change appears necessary.

- [ ] **T102 [R-003, R-004, G1] Implement and verify compact intersection geometry reconstruction**
  - Expected output: internal \`tau,V,Kinv\` helper using distinct COO
    permutations.
  - Verify:
    - independent iii/iij/ijk analytic fixtures;
    - factor 4;
    - minus sign;
    - final symmetrization;
    - persisted-\`CY_volume\` replay at \`1e-10/1e-10\`.
  - Escalate if: accepted reference normalization cannot be reproduced.

- [ ] **T103 [R-005, R-006, G1] Implement and verify direct/pair charge reconstruction**
  - Expected output: exact integer \`Q\`.
  - Verify:
    - finite/integral/range/shape checks before conversion;
    - unique direct rays;
    - raw/canonical/duplicate counts;
    - exact lexicographic pair order;
    - exact \`q_j-q_i\` convention;
    - exact dense-oracle equality on focused fixtures.
  - Escalate if: persisted charge semantics are insufficient or contradictory.

- [ ] **T104 [R-007, G1] Implement and verify direct/pair potential coefficients**
  - Expected output: exact direct+pair signed/log10 \`L\`.
  - Verify:
    - independent direct coefficient oracle;
    - independent mixed coefficient oracle;
    - positive/negative sign handling;
    - zero raw amplitude rejected before \`log10\`;
    - no thresholding/truncation occurs.
  - Escalate if: a coefficient convention differs from the approved spec.

- [ ] **T104B [R-008, R-010, G1/G2] Implement and verify geometry-level QED lineage**
  - Expected output:
    - direct QED source reuses the existing direct column;
    - \`appended_prime_divisor_e3\` appends exactly one charge/coefficient
      column after the pair block;
    - null geometry-level QED state leaves direct+pair unchanged.
  - Verify:
    - source index equals persisted metadata;
    - exact integral \`qed_charge\` identity;
    - source-kind attribute;
    - appended coefficient uses the single-instanton formula;
    - matched dense v8-QED oracle has identical oriented scientific potential.
  - Escalate if: persisted visible-sector metadata cannot uniquely determine the
    geometry-level term.

- [ ] **T104C [R-008, R-010, G1/G2] Enforce EFT assignment-pool separation**
  - Expected output: geometry-level \`read.potential(GeometryIndex)\` does not
    select an assignment and returns direct+pair only for assignment-pool
    artifacts.
  - Verify:
    - assignment pool requires absent geometry-level visible-sector group;
    - \`qed_source_index\` is null;
    - no extra QED term appears;
    - contradictory states fail closed.
  - Escalate if: row-level EFT semantics would need to be inferred or moved
    into the geometry reader.

- [ ] **T105 [R-008, G1] Implement reconstruction-integrity checks**
  - Expected output: required counts/hashes/witnesses gate every compact read.
  - Verify:
    - frozen Python-reference \`q_direct_sha256\`;
    - frozen Python-reference \`pair_source_index_sha256\`;
    - exact count comparison;
    - corrupted hash/count/tolerance fixtures fail.
  - Escalate if: cross-language byte serialization differs from the frozen
    stable-hash contract.

## Phase 2 — Public reader integration

- [ ] **T201 [R-001, R-009, R-010, G1/G2] Route \`read.potential\` through the shared component boundary**
  - Expected output: existing \`AxionPotential\` contract accepts both storage
    forms.
  - Verify:
    - all F-003A legacy dense classes retain existing raw outputs;
    - row-oriented dense artifacts remain row-oriented at this boundary;
    - column-oriented dense artifacts remain column-oriented;
    - existing \`_kinetic_matrix\` and validation semantics unchanged;
    - compact reader passes synthetic fixtures.

- [ ] **T202 [R-001, R-009, R-010, G1/G2] Route \`potential_factored\` through the same component boundary**
  - Expected output: existing \`(;L,Q,Kinv,C)\` contract for both storage forms.
  - Verify:
    - existing symmetric-\`Kinv\`/Cholesky behavior preserved;
    - failure semantics are not replaced by \`read.potential\` semantics.

- [ ] **T203 [R-010, G2] Verify \`oriented_potential\` transparently inherits support**
  - Expected output: no storage-schema branch in \`oriented_potential\`.
  - Verify: matched dense/compact outputs agree.

- [ ] **T204 [R-013, G1/G2] Prove compact reads are non-mutating**
  - Expected output: exact before/after source artifact identity.
  - Verify: no dense arrays/cache/write events appear after reads.

## Phase 3 — Matched oracle evidence

- [ ] **T301 [R-009, R-010, G2] Freeze matched real and legacy fixture identities before comparison**
  - Required compact/dense cells:
    - h11=4;
    - h11=10;
    - h11=50;
    - one bounded higher-dimensional fixture selected before comparison.
  - Raw-reader equality oracle must be canonical column-oriented dense.
  - Required legacy regressions:
    - marker-free row-oriented;
    - marker-free column-oriented;
    - v5 row-oriented;
    - v8 column-oriented;
    - v8-QED-assignment with appended QED;
    - transitional dense v9.
  - Required compact semantic fixtures:
    - one appended geometry-level QED case;
    - one EFT assignment-pool case.
  - Record:
    - source commit/tree;
    - geometry/FRST identity;
    - final Kähler-point identity;
    - dense oracle identity;
    - compact artifact identity.
  - Escalate if: same-FRST/same-point identity cannot be established.

- [ ] **T302 [R-003-R-010, G2] Run exact matched reconstruction comparison**
  - Verify:
    - canonical column-oriented oracle has exact \`Q\` equality;
    - coefficient signs exact;
    - \`tau,V,Kinv,L\` within frozen replay tolerance;
    - public-reader equivalence;
    - row-oriented legacy fixtures preserve raw outputs and agree through
      \`oriented_potential\`;
    - appended-QED compact fixture equals dense v8-QED scientific potential;
    - assignment-pool fixture receives no geometry-level QED augmentation;
    - no tolerance tuning after results.
  - Escalate if: mismatch cannot be explained by an implementation defect or a
    separately demonstrated bad fixture.

## Phase 4 — Downstream qualification

- [ ] **T401 [R-011, G3] Run vacuum-only dense-vs-compact equivalence**
  - Use identical \`compute_vacua_data(...; method=:auto, ...)\` configuration.
  - Verify exact agreement in:
    - vacuum count;
    - \`search_classification\`;
    - \`auto_selected_method\`;
    - determinant/branch metadata where applicable.
  - Escalate if: equality would require changing vacua science or tolerances.

- [ ] **T402 [R-012, G3] Run one non-vacua direct \`AxionPotential\` consumer equivalence**
  - Consumer must not require \`read.geometry\`.
  - Verify matched output under its existing contract.
  - Explicitly record that this does not establish \`compute_axion_data\`
    schema-1.1 support.

## Phase 5 — Verification and independent acceptance

- [ ] **T501 [R-016, G4] Run required local verification**
  - Required:
    - focused tests;
    - full local package tests;
    - \`julia --project=. bin/audit.jl\`;
    - \`python3 scripts/agent_verify.py diff-check\`;
    - \`git diff --check\`.
  - Record commands, exit status, observed results, warnings, and unavailable
    checks.

- [ ] **T502 [R-016, G4] Freeze exact implementation/evidence candidate**
  - Record:
    - approved spec ref;
    - base commit/tree;
    - candidate commit/tree;
    - changed paths;
    - oracle fixtures;
    - G1-G3 evidence.

- [ ] **T503 [R-016, G4] Obtain fresh independent Scientific/Numerical Review**
  - Review binds exact candidate and evidence.
  - Any changed implementation/evidence bytes after review require fresh
    exact-candidate review.
  - Passing review does not authorize merge/closure.

- [ ] **T504 [G4] Return reviewed candidate to Control Desk**
  - Return exact status/evidence for owner integration decision.
  - Do not merge, close, or broaden scope automatically.

## Convergence

- [ ] **TC01 [R-001-R-016] Converge spec ↔ plan ↔ tasks ↔ implementation ↔ evidence ↔ PR**
  - Any unmet requirement remains explicit.
  - Any newly discovered scientific/schema choice returns to G0.
  - Record the reusable-lessons check required by the SDD workflow.
