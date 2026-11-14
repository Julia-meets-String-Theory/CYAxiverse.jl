# CYAX-0232 R-005 report

Command: `julia --startup-file=no --project=. scripts/r005_scaling_harness.jl report <out_dir>`,
run by the owner on macOS (Julia 1.12) at branch head `0a22a56`, whose code is the G1
candidate `90f6f43`. The table below is the harness output, unedited.

Candidate evaluated at c = 1 (stored Kinv) and c = 1/2 (2·Kinv).
Log10 fields: offset c1 − c½. Linear fields: ratio c1/c½ (max |deviation| over entries).

| Output | Kind | c = 1 vs c = 1/2 |
|---|---|---|
| `infl/F1/fK_log10` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `infl/F1/fpert_log10` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `infl/F1/lambda_22_indices` | count | equal: [1; 1; 0; 0;;] |
| `infl/F1/lambda_22_log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `infl/F1/lambda_22_sign` | count | equal: [1] |
| `infl/F1/lambda_31_indices` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `infl/F1/lambda_31_log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `infl/F1/lambda_31_sign` | count | equal: [-1, 1] |
| `infl/F1/lambda_self_log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `infl/F1/lambda_self_sign` | count | equal: [1, 1] |
| `infl/F1/m` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `infl/F1/mode_indices` | count | equal: [0, 1] |
| `infl/F2_hp/cubic/phase` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `infl/F2_hp/cubic/tensor` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `infl/F2_hp/decay/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `infl/F2_hp/decay/fpert` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `infl/F2_hp/masses/log10` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `infl/F2_hp/masses/sign` | count | equal: [1, 1] |
| `infl/F2_hp/metadata/cubic_basis` | string | equal |
| `infl/F2_hp/metadata/kinetic_normalization` | string | equal |
| `infl/F2_hp/metadata/schema_version` | string | equal |
| `infl/F2_hp/quart22/index` | count | equal: [1; 1; 0; 0;;] |
| `infl/F2_hp/quart22/log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `infl/F2_hp/quart22/sign` | count | equal: [1] |
| `infl/F2_hp/quart31/index` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `infl/F2_hp/quart31/log10` | log10 | offset -0.602059991328 (spread 2.2e-16) |
| `infl/F2_hp/quart31/sign` | count | equal: [1, -1] |
| `infl/F2_hp/quartdiag/log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `infl/F2_hp/quartdiag/sign` | count | equal: [1, 1] |
| `infl/F2_hp_memory/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `infl/F2_hp_memory/fpert` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `infl/F2_hp_memory/m` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `infl/F2_hp_memory/msign` | count | equal: [1, 1] |
| `infl/F2_hp_memory/λ22` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `infl/F2_hp_memory/λ22_i` | count | equal: [1; 1; 0; 0;;] |
| `infl/F2_hp_memory/λ22sign` | count | equal: [1] |
| `infl/F2_hp_memory/λ31` | log10 | offset -0.602059991328 (spread 2.2e-16) |
| `infl/F2_hp_memory/λ31_i` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `infl/F2_hp_memory/λ31sign` | count | equal: [1, -1] |
| `infl/F2_hp_memory/λself` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `infl/F2_hp_memory/λselfsign` | count | equal: [1, 1] |
| `infl/F2_pq/cubic/phase` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `infl/F2_pq/cubic/tensor` | linear | ratio 0.353553390593 (spread 2.8e-16) |
| `infl/F2_pq/decay/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `infl/F2_pq/decay/fpert` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `infl/F2_pq/masses/log10` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `infl/F2_pq/masses/sign` | count | equal: [1, 1] |
| `infl/F2_pq/metadata/cubic_basis` | string | equal |
| `infl/F2_pq/metadata/kinetic_normalization` | string | equal |
| `infl/F2_pq/metadata/schema_version` | string | equal |
| `infl/F2_pq_memory/f` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `infl/F2_pq_memory/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `infl/F2_pq_memory/m` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `infl/F2_pq_memory/msign` | count | equal: [1, 1] |
| `infl/F2_pq_memory/λ22` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `infl/F2_pq_memory/λ22_i` | count | equal: [1; 1; 0; 0;;] |
| `infl/F2_pq_memory/λ22sign` | count | equal: [1] |
| `infl/F2_pq_memory/λ31` | log10 | offset -0.602059991328 (spread 1.6e-15) |
| `infl/F2_pq_memory/λ31_i` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `infl/F2_pq_memory/λ31sign` | count | equal: [1, -1] |
| `infl/F2_pq_memory/λself` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `infl/F2_pq_memory/λselfsign` | count | equal: [1, 1] |
| `infl/F6/candidate_slowroll_saddles` | count | equal: 0 |
| `infl/F6/leading_branch_saddle_count` | count | equal: 3 |
| `infl/F6/leading_minima_count` | count | equal: 1 |
| `infl/F6/mass_max` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `infl/F6/mass_min` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `infl/inflation/point1/eigenvalues` | linear | ratio 0.500000000000 (spread 2.2e-16) |
| `infl/inflation/point1/epsilon` | linear | ratio 0.500000000000 (spread 0.0e+00) |
| `infl/inflation/point1/eta` | linear | ratio 0.500000000000 (spread 2.2e-16) |
| `infl/inflation/point1/gradient_norm` | linear | ratio 0.707106781187 (spread 0.0e+00) |
| `infl/inflation/point1/mode_counts` | count | equal: [0, 0, 2] |
| `infl/inflation/point1/value` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `infl/inflation/point2/eigenvalues` | linear | ratio 0.500000000000 (spread 0.0e+00) |
| `infl/inflation/point2/epsilon` | linear | ratio 0.500000000000 (spread 0.0e+00) |
| `infl/inflation/point2/eta` | linear | ratio 0.500000000000 (spread 0.0e+00) |
| `infl/inflation/point2/gradient_norm` | linear | ratio 0.707106781187 (spread 0.0e+00) |
| `infl/inflation/point2/mode_counts` | count | equal: [1, 0, 1] |
| `infl/inflation/point2/value` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `infl/inflation/point3/eigenvalues` | linear | ratio 0.500000000000 (spread 4.4e-16) |
| `infl/inflation/point3/epsilon` | linear | ratio 0.500000000000 (spread 0.0e+00) |
| `infl/inflation/point3/eta` | linear | ratio 0.500000000000 (spread 3.9e-16) |
| `infl/inflation/point3/gradient_norm` | linear | ratio 0.707106781187 (spread 0.0e+00) |
| `infl/inflation/point3/mode_counts` | count | equal: [1, 0, 1] |
| `infl/inflation/point3/value` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `infl/inflation/run_geometry/candidate_slowroll_saddles` | count | equal: 0 |
| `infl/inflation/run_geometry/leading_minima_count` | count | equal: 1 |
| `infl/inflation/run_geometry/mass_max` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `infl/inflation/run_geometry/mass_min` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `infl/inflation/run_geometry/negative_mass_count` | count | equal: 0 |
| `infl/inflation/run_geometry/saddle_count` | count | equal: 3 |
| `r003/F1/fK_log10` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r003/F1/fpert_log10` | log10 | offset 0.150514997832 (spread 7.1e-15) |
| `r003/F1/lambda_22_indices` | count | equal: [1; 1; 0; 0;;] |
| `r003/F1/lambda_22_log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r003/F1/lambda_22_sign` | count | equal: [1] |
| `r003/F1/lambda_31_indices` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `r003/F1/lambda_31_log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r003/F1/lambda_31_sign` | count | equal: [1, -1] |
| `r003/F1/lambda_self_log10` | log10 | offset -0.602059991328 (spread 1.4e-14) |
| `r003/F1/lambda_self_sign` | count | equal: [1, 1] |
| `r003/F1/m` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r003/F1/mode_indices` | count | equal: [0, 1] |
| `r003/F2_hp/cubic/phase` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `r003/F2_hp/cubic/tensor` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `r003/F2_hp/decay/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r003/F2_hp/decay/fpert` | log10 | offset 0.150514997832 (spread 7.1e-15) |
| `r003/F2_hp/masses/log10` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r003/F2_hp/masses/sign` | count | equal: [1, 1] |
| `r003/F2_hp/metadata/cubic_basis` | string | equal |
| `r003/F2_hp/metadata/kinetic_normalization` | string | equal |
| `r003/F2_hp/metadata/schema_version` | string | equal |
| `r003/F2_hp/quart22/index` | count | equal: [1; 1; 0; 0;;] |
| `r003/F2_hp/quart22/log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r003/F2_hp/quart22/sign` | count | equal: [1] |
| `r003/F2_hp/quart31/index` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `r003/F2_hp/quart31/log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r003/F2_hp/quart31/sign` | count | equal: [-1, -1] |
| `r003/F2_hp/quartdiag/log10` | log10 | offset -0.602059991328 (spread 1.4e-14) |
| `r003/F2_hp/quartdiag/sign` | count | equal: [1, 1] |
| `r003/F2_hp_memory/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r003/F2_hp_memory/fpert` | log10 | offset 0.150514997832 (spread 7.1e-15) |
| `r003/F2_hp_memory/m` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r003/F2_hp_memory/msign` | count | equal: [1, 1] |
| `r003/F2_hp_memory/λ22` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r003/F2_hp_memory/λ22_i` | count | equal: [1; 1; 0; 0;;] |
| `r003/F2_hp_memory/λ22sign` | count | equal: [1] |
| `r003/F2_hp_memory/λ31` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r003/F2_hp_memory/λ31_i` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `r003/F2_hp_memory/λ31sign` | count | equal: [-1, -1] |
| `r003/F2_hp_memory/λself` | log10 | offset -0.602059991328 (spread 1.4e-14) |
| `r003/F2_hp_memory/λselfsign` | count | equal: [1, 1] |
| `r003/F2_pq/cubic/phase` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `r003/F2_pq/cubic/tensor` | linear | ratio 0.353553390593 (spread 1.7e-16) |
| `r003/F2_pq/decay/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r003/F2_pq/decay/fpert` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r003/F2_pq/masses/log10` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r003/F2_pq/masses/sign` | count | equal: [1, 1] |
| `r003/F2_pq/metadata/cubic_basis` | string | equal |
| `r003/F2_pq/metadata/kinetic_normalization` | string | equal |
| `r003/F2_pq/metadata/schema_version` | string | equal |
| `r003/F2_pq_memory/f` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r003/F2_pq_memory/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r003/F2_pq_memory/m` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r003/F2_pq_memory/msign` | count | equal: [1, 1] |
| `r003/F2_pq_memory/λ22` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r003/F2_pq_memory/λ22_i` | count | equal: [1; 1; 0; 0;;] |
| `r003/F2_pq_memory/λ22sign` | count | equal: [1] |
| `r003/F2_pq_memory/λ31` | log10 | offset -0.602059991328 (spread 9.9e-14) |
| `r003/F2_pq_memory/λ31_i` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `r003/F2_pq_memory/λ31sign` | count | equal: [-1, 1] |
| `r003/F2_pq_memory/λself` | log10 | offset -0.602059991328 (spread 1.4e-14) |
| `r003/F2_pq_memory/λselfsign` | count | equal: [1, 1] |
| `r003/F3_electron_proxy/light_mode_count` | count | equal: 2 |
| `r003/F3_electron_proxy/log10_f_GeV` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r003/F3_electron_proxy/log10_g_GeVinv` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r003/F3_electron_proxy/log10_g_effective_GeVinv` | log10 | offset -0.451544993496 (spread 1.4e-14) |
| `r003/F3_electron_proxy/log10_light_threshold_eV` | log10 | offset 0.000000000000 (spread 0.0e+00) |
| `r003/F3_electron_proxy/log10_mass_eV` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r003/F3_electron_proxy/log10_photon_width_GeV` | log10 | offset -1.354634980488 (spread 0.0e+00) |
| `r003/F3_electron_proxy/log10_quartic_width_GeV` | log10 | offset -1.354634980488 (spread 0.0e+00) |
| `r003/F3_electron_proxy/mixing_theta` | linear | ratio 1.000000000000 (spread 1.1e-16) |
| `r003/F3_electron_proxy/persisted/log10_f_GeV` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r003/F3_electron_proxy/persisted/log10_mass_eV` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r003/F3_electron_proxy/persisted/q` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r003/F3_electron_proxy/persisted/theta_from_canonical` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r003/F3_electron_proxy/q` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r003/F3_electron_proxy/theta_from_canonical` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r003/F4/divisor_index` | count | equal: [2, 6] |
| `r003/F4/mass_log10_ev_reference` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r004/F1/fK_log10` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F1/fpert_log10` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F1/lambda_22_indices` | count | equal: [1; 1; 0; 0;;] |
| `r004/F1/lambda_22_log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r004/F1/lambda_22_sign` | count | equal: [1] |
| `r004/F1/lambda_31_indices` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `r004/F1/lambda_31_log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r004/F1/lambda_31_sign` | count | equal: [-1, -1] |
| `r004/F1/lambda_self_log10` | log10 | offset -0.602059991328 (spread 4.4e-16) |
| `r004/F1/lambda_self_sign` | count | equal: [1, 1] |
| `r004/F1/m` | log10 | offset -0.150514997832 (spread 1.1e-16) |
| `r004/F1/mode_indices` | count | equal: [0, 1] |
| `r004/F2_hp/cubic/phase` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `r004/F2_hp/cubic/tensor` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `r004/F2_hp/decay/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F2_hp/decay/fpert` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F2_hp/masses/log10` | log10 | offset -0.150514997832 (spread 1.1e-16) |
| `r004/F2_hp/masses/sign` | count | equal: [1, 1] |
| `r004/F2_hp/metadata/cubic_basis` | string | equal |
| `r004/F2_hp/metadata/kinetic_normalization` | string | equal |
| `r004/F2_hp/metadata/schema_version` | string | equal |
| `r004/F2_hp/quart22/index` | count | equal: [1; 1; 0; 0;;] |
| `r004/F2_hp/quart22/log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r004/F2_hp/quart22/sign` | count | equal: [1] |
| `r004/F2_hp/quart31/index` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `r004/F2_hp/quart31/log10` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r004/F2_hp/quart31/sign` | count | equal: [1, 1] |
| `r004/F2_hp/quartdiag/log10` | log10 | offset -0.602059991328 (spread 4.4e-16) |
| `r004/F2_hp/quartdiag/sign` | count | equal: [1, 1] |
| `r004/F2_hp_memory/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F2_hp_memory/fpert` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F2_hp_memory/m` | log10 | offset -0.150514997832 (spread 1.1e-16) |
| `r004/F2_hp_memory/msign` | count | equal: [1, 1] |
| `r004/F2_hp_memory/λ22` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r004/F2_hp_memory/λ22_i` | count | equal: [1; 1; 0; 0;;] |
| `r004/F2_hp_memory/λ22sign` | count | equal: [1] |
| `r004/F2_hp_memory/λ31` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r004/F2_hp_memory/λ31_i` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `r004/F2_hp_memory/λ31sign` | count | equal: [1, 1] |
| `r004/F2_hp_memory/λself` | log10 | offset -0.602059991328 (spread 4.4e-16) |
| `r004/F2_hp_memory/λselfsign` | count | equal: [1, 1] |
| `r004/F2_pq/cubic/phase` | linear | ratio 1.000000000000 (spread 0.0e+00) |
| `r004/F2_pq/cubic/tensor` | linear | ratio 0.353553390593 (spread 1.7e-16) |
| `r004/F2_pq/decay/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F2_pq/decay/fpert` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F2_pq/masses/log10` | log10 | offset -0.150514997832 (spread 1.1e-16) |
| `r004/F2_pq/masses/sign` | count | equal: [1, 1] |
| `r004/F2_pq/metadata/cubic_basis` | string | equal |
| `r004/F2_pq/metadata/kinetic_normalization` | string | equal |
| `r004/F2_pq/metadata/schema_version` | string | equal |
| `r004/F2_pq_memory/f` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F2_pq_memory/fK` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F2_pq_memory/m` | log10 | offset -0.150514997832 (spread 1.1e-16) |
| `r004/F2_pq_memory/msign` | count | equal: [1, 1] |
| `r004/F2_pq_memory/λ22` | log10 | offset -0.602059991328 (spread 0.0e+00) |
| `r004/F2_pq_memory/λ22_i` | count | equal: [1; 1; 0; 0;;] |
| `r004/F2_pq_memory/λ22sign` | count | equal: [1] |
| `r004/F2_pq_memory/λ31` | log10 | offset -0.602059991328 (spread 7.1e-15) |
| `r004/F2_pq_memory/λ31_i` | count | equal: [0 1; 0 1; 0 1; 1 0] |
| `r004/F2_pq_memory/λ31sign` | count | equal: [1, 1] |
| `r004/F2_pq_memory/λself` | log10 | offset -0.602059991328 (spread 4.0e-15) |
| `r004/F2_pq_memory/λselfsign` | count | equal: [1, 1] |
| `r004/F3_divisor_instanton/light_mode_count` | count | equal: 1 |
| `r004/F3_divisor_instanton/log10_f_GeV` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F3_divisor_instanton/log10_g_GeVinv` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r004/F3_divisor_instanton/log10_g_effective_GeVinv` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r004/F3_divisor_instanton/log10_light_threshold_eV` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r004/F3_divisor_instanton/log10_mass_eV` | log10 | offset -0.150514997832 (spread 1.1e-16) |
| `r004/F3_divisor_instanton/log10_photon_width_GeV` | log10 | offset -0.752574989160 (spread 1.4e-14) |
| `r004/F3_divisor_instanton/log10_quartic_width_GeV` | log10 | offset -1.354634980488 (spread 0.0e+00) |
| `r004/F3_divisor_instanton/mixing_theta` | linear | ratio 1.000000000000 (spread 2.2e-16) |
| `r004/F3_divisor_instanton/persisted/log10_f_GeV` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F3_divisor_instanton/persisted/log10_mass_eV` | log10 | offset -0.150514997832 (spread 1.1e-16) |
| `r004/F3_divisor_instanton/persisted/q` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r004/F3_divisor_instanton/persisted/theta_from_canonical` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r004/F3_divisor_instanton/q` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r004/F3_divisor_instanton/theta_from_canonical` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r004/F3_electron_proxy/light_mode_count` | count | equal: 1 |
| `r004/F3_electron_proxy/log10_f_GeV` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F3_electron_proxy/log10_g_GeVinv` | log10 | offset -0.150514997832 (spread 0.0e+00) |
| `r004/F3_electron_proxy/log10_g_effective_GeVinv` | log10 | offset -0.150514997832 (spread 3.0e-01) |
| `r004/F3_electron_proxy/log10_light_threshold_eV` | log10 | offset 0.000000000000 (spread 0.0e+00) |
| `r004/F3_electron_proxy/log10_mass_eV` | log10 | offset -0.150514997832 (spread 1.1e-16) |
| `r004/F3_electron_proxy/log10_photon_width_GeV` | log10 | offset -0.752574989160 (spread 6.0e-01) |
| `r004/F3_electron_proxy/log10_quartic_width_GeV` | log10 | offset -1.354634980488 (spread 0.0e+00) |
| `r004/F3_electron_proxy/mixing_theta` | linear | ratio 1.000000000000 (spread 2.2e-16) |
| `r004/F3_electron_proxy/persisted/log10_f_GeV` | log10 | offset 0.150514997832 (spread 0.0e+00) |
| `r004/F3_electron_proxy/persisted/log10_mass_eV` | log10 | offset -0.150514997832 (spread 1.1e-16) |
| `r004/F3_electron_proxy/persisted/q` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r004/F3_electron_proxy/persisted/theta_from_canonical` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r004/F3_electron_proxy/q` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r004/F3_electron_proxy/theta_from_canonical` | linear | ratio 0.707106781187 (spread 1.1e-16) |
| `r004/F4/divisor_index` | count | equal: [1, 2] |
| `r004/F4/mass_log10_ev_reference` | log10 | offset -0.150514997832 (spread 1.1e-16) |

## Reading against spec R-005 (added at G1, not harness output)

- Gated factors match the R-005 tables: f, fpert +½log₁₀2; masses
  −½log₁₀2; quartics −log₁₀4; pq cubic 2^-3/2, hp cubic 1; F3 q and
  θ-from-canonical 1/√2, mixing 1; `:divisor_instanton` g_eff and threshold
  1/√2, photon width 2^-5/2, quartic width 2^-9/2; `:electron_proxy`
  threshold fixed, light-mode g_eff 2^-3/2 and photon width 2^-9/2 (r003);
  ε, η, eig(K⁻¹H) ½; mass_min/max and F4 reference mass −½log₁₀2.
- r004 `:electron_proxy` rows with spread 0.30 and 0.60 hold one light and one
  heavy mode (light_mode_count 1); the per-mode factors are asserted to 1e-10
  in `test/kinetic_normalization.jl` (T015).
- Reported rows: every count, index, sign and classification in this report
  is equal between conventions; no mode crossed a threshold.
- Not in this report: e-fold counts, F6 `n_candidates`, `n_qualified`,
  `flow_efolds`, `flow_slow_roll_efolds`, F4 `model_lambda` and model counts,
  and the F5 optimizer outputs (all "may change; reported" in R-005).
