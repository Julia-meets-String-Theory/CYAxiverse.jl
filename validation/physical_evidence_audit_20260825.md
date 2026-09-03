# Physical evidence audit — 2026-08-25

Status: complete for the authorized read-only Steps 1–3 audit; no inflation or scale point was evaluated.

- Fixed inputs audited: 18 (three each at h11=5,…,10).
- Selection manifest SHA-256: `a6df5dca258c11724d4162477cdee7cc34e5802f2f3f296a7ea64b55f23c3247`.
- Required source commit: `9f31d716eaab8d63d3f76826a40de5ae38c7015d`.
- Required source branch: `agents/physical-scale-inflation-20260825`.
- Repository guard: HEAD, branch, and tracked-file status passed; only the four expected untracked audit outputs were present before the audit.
- Manifest binding: strict JSON parse passed for all 18 paths, artifact hashes, h11 values, three distinct polytope identities per h11, triangulation identities, and orientifold.requested=false.
- Artifact SHA-256 verification: 18/18 passed.
- Complete certificate sidecars: 0; blocked by shared genuine gaps.
- Audit resource record: final run internal elapsed `2.1730830669403076` s, maximum resident set `554631168` bytes; no geometry-local arrays are retained between inputs.

## Evidence classifications

| Class | Count | Fields |
| --- | ---: | --- |
| `stored_alias` | 6 | `basis_identity`, `curve_volumes`, `cy_volume`, `divisor_volumes`, `kinv`, `prime_divisor_volumes` |
| `exactly_reconstructable` | 4 | `charge_orientation`, `effective_divisor_volumes`, `kahler_margin`, `precision_bits` |
| `source_fixed_convention` | 2 | `moduli_status`, `visible_sector_status` |
| `genuinely_unavailable` | 9 | `configuration_digest`, `instanton_control`, `normalization`, `perturbative_control`, `phase_convention`, `potent_curve_volumes`, `source_identity`, `spd_tolerance`, `units` |

## Genuine scientific gaps requiring user decisions

- `configuration_digest`: Provide or approve a canonical configuration identity
- `instanton_control`: User must decide whether volume-only control is an approved proxy
- `normalization`: User must approve the artifact-level normalization mapping to the pilot contract
- `perturbative_control`: User must provide potent-ray evidence or approve a proxy
- `phase_convention`: User must supply or approve an explicit phase convention
- `potent_curve_volumes`: User must provide a potent-ray classification or approve a proxy definition
- `source_identity`: Provide or approve a complete source/environment identity
- `spd_tolerance`: User must approve a declared SPD tolerance policy
- `units`: The owner-selected M_s=M_Pl contract does not prove the legacy artifact unit convention; approve a source-backed units declaration

## Exact reconstruction formulas

- Effective-divisor volumes: normalize stored `effective_cone` to one ray per row, then `E * divisor_volumes`.
- Kähler margin: normalize stored `kahler_hyperplanes` to one hyperplane per row, then `minimum(H * tip)`; the historical writer checked `>= 1 - 1e-6`.
- Charge orientation: `Q` is `h11 × N`; direct columns equal the transpose of normalized one-ray-per-row `E`, followed by lexicographically ordered pair differences.
- Stored coefficient audit: `L` is compared to the source V_CY^-2 direct/pair formula with explicit status. The existing cited diagnostic tolerance is `1e-10` in `validation/run_author_code_coefficient_bridge.py:238-253`; all 18 records are within that tolerance with zero sign mismatches.
- Raw kinetic-matrix diagnostic: `Kinv - Kinv'` is measured before symmetrization; SPD eigenspectra remain diagnostic only because no artifact SPD tolerance is declared.
- Basis identity evidence: each JSONL record contains exact basis/basis_matrix values, canonical shape/value SHA-256 hashes, basis convention, and stored kappa/QCD index bases. A separate basis index-base field is not stored in the legacy artifact.

## Reproduction and verification

- Recovery record compressed SHA-256: `e88c24e538a8e4ab62c8f3fba3852ef3d045ef1ae1e6cb9be4aa5ac188100bf4`.
- Recovery record JSONL SHA-256: `d7056751a8b207147cc56c6415ab3a5753053026137818b39795383cf5d752e5`.
- Audit script SHA-256: `0aca26bd733bce759d28b54f6533ba43e2ec91f65f85c5de186fc5bdb587265b`.
- Audit JSONL SHA-256: `7e1b1198adebbc4c93adc3b273f4328ef8235d1881d8b7478828f0159c69afad`.
- Audit compressed JSONL SHA-256: `2c4caa8daf4d04238509c4c793022489f13e2e07fa1a6e810e7c565c0c5554fe`.
- Historical generator source SHA-256 at `770b09b7e503ccf01202b1ec2212149c7bd50a5`: `52d227d15e7231ff7e83faf23bb56be9a40fd2dc9b1c421d3ea7d894c35e9686`.
- Committed continuation source SHA-256: `c43bf109750eca12eb78d5a5118b402db29a7adc6aa1eeb928dd26a6dc5e3e4e`.
- Derivation ledger SHA-256: `97bb1147d088c6d66407de9a1966b8872541ebdcb5dfd822c1b82722c9ef90c5`.
- Reproduction command: `julia --startup-file=no --project=/Users/vmehta/Documents/CYAxiverse/cyaxiverse/CYAxiverse.jl scripts/audit_physical_certificate.jl`.
- Verification passed: repository HEAD/branch/tracked-status guard; strict manifest parsing and 18-entry binding; 18/18 artifact hashes; HDF5 dataset/metadata checks; raw Kinv asymmetry measurement before symmetrization; effective-volume and Kähler-margin reconstructions; charge orientation and pair-order checks; Q/L diagnostics with explicit `1e-10` existing tolerance and statuses; canonical basis values/hashes; JSONL parsing; `gzip -t`; and `git diff --check`.
- The run read the neighboring cached Julia environment only; it did not install or fetch dependencies, write source/input files, evaluate inflation or scales, compute orientifolds, generate geometry, expand the population, or create a database.

## Per-input records

The machine-readable per-input evidence, observed values, source IDs, array shapes, formulas, and hashes are in [`physical_evidence_audit_20260825.jsonl`](physical_evidence_audit_20260825.jsonl).

## Separate-gate policy update

The 2026-08-25 user decision separates pre-calculation scaling evidence from
post/domain physical controls. The audit's nine genuinely unavailable fields
are therefore split by use: `configuration_digest`, `normalization`,
`phase_convention`, `source_identity`, `spd_tolerance`, and `units` remain
mandatory `physical_scaling_gate` evidence, while potent-ray evidence,
instanton control, perturbative control, moduli control, and visible-sector
applicability are recorded by the independent `physical_control_gate`.

For the fixed legacy artifacts, no complete scaling certificate sidecar is
created. Their scaling gate remains non-passed because the scaling metadata,
source/configuration provenance, SPD policy, and precision/conversion contract
are not complete. The control gate may be `not_established` in a diagnostic
calculation only after a scaling gate passes; it never supports a physically
viable, production-qualified, or validated-candidate label. A screen hit with
both gates passed remains `eligible_not_validated`. Historical homotopy and
fixed-volume outputs remain nonphysical and use `not_applicable` for both
physical gates. No inflation, scale, orientifold, geometry, or production
work was performed by this policy update.

Implementation hashes for this policy update:

- `scripts/inflation_scale_continuation.jl`: `fa2fd5833d972179ac89eaf55f3affca229767d33093c425d0bf863173182a72`
- `scripts/build_orientifold_vacua_inflation.jl`: `052d6f68fd4a5ddcd834f515df1f58fe8af9f221513d40887d29bf51aa8ce1e0`
- `test/runtests.jl`: `3162165807033f6df5596c47171ee7a11add492652b5e4adc552cbb187bab90e`
