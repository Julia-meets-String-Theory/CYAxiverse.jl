#!/usr/bin/env julia

using CYAxiverse
using LinearAlgebra
using SHA
using TOML

const G = CYAxiverse.generate
const PB = CYAxiverse.paper_benchmarks
const Arb = G.ArbNumerics
const OUT = get(ENV, "CYAX173_REPLAY_OUTPUT", joinpath(@__DIR__, "p1_replay.toml"))
const EXECUTION_BASE = get(ENV, "CYAX173_EXECUTION_BASE",
    "74fea608684eae746f25ae45b18512f89b862fb0")

sha_repr(value) = bytes2hex(sha256(codeunits(repr(value))))
rows(matrix) = [collect(row) for row in eachrow(matrix)]

function main()
potential = PB.n5_potential(k=1.0)
K = PB.n5_kinetic_matrix(1.0)
Kmatrix = Matrix(K)
fixture_digest = bytes2hex(sha256(codeunits(join(string.(
    ("B6", potential.Q, potential.L, K, :float64, :high_precision,
     80, true, true, true), "|")))))
fixture_digest == "68261b5571df88e6afe75f985eafac27a144390779306d3e4c416a6fc1caf6ef" ||
    error("P0 B6 fixture digest mismatch: $fixture_digest")

original_arb_bits = precision(Arb.ArbFloat)
selection = G.LQtilde(potential.Q, potential.L)
C = G._canonical_factor_from_K(K)
metric_whitening_error = opnorm(C' * Kmatrix * C - I, Inf)

float_result = G.pq_spectrum(K, potential.L, potential.Q;
    mixing_correction=:float64, quartic_diagnostics=true,
    mass_basis_diagnostics=true, hierarchy_diagnostics=true)
float_masses, float_signs, float_basis =
    G.leading_hessian_mass_basis_float64(K, selection.Ltilde, selection.Qtilde)
float_matrix = G.leading_hessian_matrix_float64_scaled(C,
    selection.Ltilde, selection.Qtilde)

high_result = nothing
high_masses = nothing
high_signs = nothing
high_basis = nothing
high_matrix = nothing
actual_high_bits = 0
try
    high_result = G.pq_spectrum(K, potential.L, potential.Q;
        mixing_correction=:high_precision, prec=80,
        quartic_diagnostics=true, mass_basis_diagnostics=true,
        hierarchy_diagnostics=true)
    actual_high_bits = precision(Arb.ArbFloat)
    high_masses, high_signs, high_basis =
        G.leading_hessian_mass_basis(C, selection.Ltilde,
            selection.Qtilde; prec=80)
    high_matrix, _ = G.high_precision_leading_hessian(C,
        selection.Ltilde, selection.Qtilde; prec=80)
finally
    setprecision(Arb.ArbFloat; bits=original_arb_bits)
end
precision(Arb.ArbFloat) == original_arb_bits ||
    error("ArbFloat global precision was not restored")

expected_float = [13.860912539123262, 14.073585277123616,
    14.230434590975188, 21.948959703416346, 22.352482675621758]
expected_high = [-12.91855844342438, -4.5978009624626,
    -1.9630253110345888, 21.948959703416346, 22.352482675621758]
isapprox(float_result.m, expected_float; rtol=2e-13, atol=2e-13) ||
    error("Float64 replay did not reproduce P0 masses: $(float_result.m)")
isapprox(high_result.m, expected_high; rtol=2e-13, atol=2e-13) ||
    error("80-digit replay did not reproduce P0 masses: $(high_result.m)")
float_result.msign == [-1, 1, 1, 1, 1] || error("Float64 P0 signs changed")
high_result.msign == [1, 1, 1, 1, 1] || error("80-digit P0 signs changed")
length(float_result.λself) == 5 && length(float_result.λ31) == 20 &&
    length(float_result.λ22) == 10 || error("Float64 quartic cardinalities changed")
length(high_result.λself) == 5 && length(high_result.λ31) == 20 &&
    length(high_result.λ22) == 10 || error("80-digit quartic cardinalities changed")

report = Dict{String,Any}(
    "schema" => "cyax-0173-b6-replay-v1",
    "work_item" => "CYAX-0173 frozen N=5 B6 diagnostic",
    "reference_commit" => "7a40285bb5c313f7e8746b90644d5f45bb67be44",
    "execution_base" => EXECUTION_BASE,
    "reference_source_equivalence" => "B6 route sources generate.jl, structs.jl, paper_benchmarks/reduced_models.jl match byte-for-byte; the only poly102_inflation.jl delta is additional N=5 continuation code after the unchanged N5_Q/N5_QDOTTAU/N5_K_RAW data and existing kinetic-matrix method",
    "fixture_digest_p0" => fixture_digest,
    "inputs" => Dict(
        "k" => 1.0, "phases" => potential.phases,
        "Q" => rows(potential.Q), "L" => rows(potential.L),
        "K" => rows(Kmatrix), "C" => rows(C),
        "Qtilde" => rows(selection.Qtilde),
        "Ltilde" => rows(selection.Ltilde),
        "Qbar" => rows(selection.Qbar),
        "Lbar" => rows(selection.Lbar),
        "Q_sha256_repr" => sha_repr(potential.Q),
        "L_sha256_repr" => sha_repr(potential.L),
        "K_sha256_repr" => sha_repr(K),
        "C_sha256_repr" => sha_repr(C),
        "Qtilde_sha256_repr" => sha_repr(selection.Qtilde),
        "Ltilde_sha256_repr" => sha_repr(selection.Ltilde),
        "metric_whitening_error_inf" => metric_whitening_error),
    "routes" => Dict(
        "float64" => Dict(
            "mixing_correction" => "float64", "quartic_diagnostics" => true,
            "mass_basis_diagnostics" => true, "hierarchy_diagnostics" => true,
            "m" => float_result.m, "msign" => float_result.msign,
            "f" => float_result.f, "fK" => float_result.fK,
            "basis_m" => float_masses, "basis_signs" => float_signs,
            "basis_vectors" => rows(float_basis),
            "eigenmatrix" => rows(Matrix(float_matrix)),
            "eigenpair_residuals" => float_result.mass_basis_diagnostics.eigenpair_residuals,
            "nearest_relative_gaps" => float_result.mass_basis_diagnostics.nearest_relative_gaps,
            "orthogonality_error" => float_result.mass_basis_diagnostics.orthogonality_error,
            "lambda_self_sign" => float_result.λselfsign,
            "lambda_self_log10" => float_result.λself,
            "lambda31_indices" => rows(float_result.λ31_i),
            "lambda31_sign" => float_result.λ31sign,
            "lambda31_log10" => float_result.λ31,
            "lambda22_indices" => rows(float_result.λ22_i),
            "lambda22_sign" => float_result.λ22sign,
            "lambda22_log10" => float_result.λ22),
        "high_precision_80" => Dict(
            "mixing_correction" => "high_precision", "prec_decimal_digits" => 80,
            "arbfloat_precision_bits_readback" => actual_high_bits,
            "arbfloat_precision_bits_after_restore" => precision(Arb.ArbFloat),
            "quartic_diagnostics" => true, "mass_basis_diagnostics" => true,
            "hierarchy_diagnostics" => true,
            "m" => high_result.m, "msign" => high_result.msign,
            "f" => high_result.f, "fK" => high_result.fK,
            "basis_m" => high_masses, "basis_signs" => high_signs,
            "basis_vectors" => rows(high_basis),
            "eigenmatrix" => rows(Float64.(Matrix(high_matrix))),
            "eigenpair_residuals" => high_result.mass_basis_diagnostics.eigenpair_residuals,
            "nearest_relative_gaps" => high_result.mass_basis_diagnostics.nearest_relative_gaps,
            "orthogonality_error" => high_result.mass_basis_diagnostics.orthogonality_error,
            "lambda_self_sign" => high_result.λselfsign,
            "lambda_self_log10" => high_result.λself,
            "lambda31_indices" => rows(high_result.λ31_i),
            "lambda31_sign" => high_result.λ31sign,
            "lambda31_log10" => high_result.λ31,
            "lambda22_indices" => rows(high_result.λ22_i),
            "lambda22_sign" => high_result.λ22sign,
            "lambda22_log10" => high_result.λ22)),
    "precision_state" => Dict(
        "arbfloat_precision_bits_before" => original_arb_bits,
        "arbfloat_precision_bits_after" => precision(Arb.ArbFloat),
        "restored" => precision(Arb.ArbFloat) == original_arb_bits))

open(OUT, "w") do io
    TOML.print(io, report; sorted=true)
end
println("P1_REPLAY_PASS")
println("fixture_digest=$fixture_digest")
println("float64_m=$(repr(float_result.m))")
println("float64_signs=$(repr(float_result.msign))")
println("high_precision80_m=$(repr(high_result.m))")
println("high_precision80_signs=$(repr(high_result.msign))")
println("arbfloat_bits_readback=$actual_high_bits")
println("arbfloat_bits_restored=$(precision(Arb.ArbFloat))")
println("report=$OUT")
end

main()
