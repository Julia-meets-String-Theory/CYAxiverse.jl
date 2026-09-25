#!/usr/bin/env julia

using CYAxiverse
using LinearAlgebra
using SHA
using TOML

const G = CYAxiverse.generate
const PB = CYAxiverse.paper_benchmarks
const A = G.ArbNumerics
const OUT = joinpath(@__DIR__, "p2_ladder.toml")
const REQUESTED_DIGITS = (64, 80, 128, 256, 512)
const BASELINE_FLOAT_MASSES = [13.860912539123262, 14.073585277123616,
    14.230434590975188, 21.948959703416346, 22.352482675621758]
const BASELINE_FLOAT_SIGNS = [-1, 1, 1, 1, 1]
const BASELINE_HIGH_MASSES = [-12.91855844342438, -4.5978009624626,
    -1.9630253110345888, 21.948959703416346, 22.352482675621758]
const BASELINE_HIGH_SIGNS = [1, 1, 1, 1, 1]

sha_file(path) = bytes2hex(sha256(read(path)))
matrix_strings(matrix) = [string.(collect(row)) for row in eachrow(matrix)]
vector_strings(values) = string.(collect(values))

function max_abs_value(values)
    isempty(values) && return zero(eltype(values))
    maximum(abs, values)
end

function relative_inf_difference(left, right)
    denominator = opnorm(left, Inf)
    denominator == 0 ? opnorm(left - right, Inf) :
        opnorm(left - right, Inf) / denominator
end

function source_mass_order(eigenvalues, masses)
    sortperm(eachindex(masses);
        by=index -> (masses[index], abs(eigenvalues[index]), index),
        alg=MergeSort)
end

function float64_source_masses(eigenvalues, L)
    reference_log = G.instanton_scale_precision_diagnostics(L).reference_log10
    0.5 .* (log10.(abs.(eigenvalues)) .+ reference_log) .+
        9 .+ log10(2.435e18) .+ log10(2π)
end

function float64_theta_hessian_scaled(L, Q)
    reference_log = maximum(L[2, :])
    floor_log = log10(floatmin(Float64))
    scales = [L[2, a] - reference_log < floor_log ? 0.0 :
        L[1, a] * 10.0^(L[2, a] - reference_log) for a in axes(L, 2)]
    H = zeros(Float64, size(Q, 1), size(Q, 1))
    for a in eachindex(scales), i in axes(Q, 1), j in axes(Q, 1)
        H[i, j] += scales[a] * Q[i, a] * Q[j, a]
    end
    Hermitian(H)
end

function float64_scaled_instanton_terms(L)
    reference_log = maximum(L[2, :])
    floor_log = log10(floatmin(Float64))
    [L[2, a] - reference_log < floor_log ? 0.0 :
        L[1, a] * 10.0^(L[2, a] - reference_log) for a in axes(L, 2)]
end

function float64_theta_hessian_column_order(L, Q, column_order)
    scales = float64_scaled_instanton_terms(L)
    H = zeros(Float64, size(Q, 1), size(Q, 1))
    for a in column_order, i in axes(Q, 1), j in axes(Q, 1)
        H[i, j] += scales[a] * Q[i, a] * Q[j, a]
    end
    Hermitian(H)
end

function float64_theta_hessian_magnitude_sorted(L, Q)
    scales = float64_scaled_instanton_terms(L)
    H = zeros(Float64, size(Q, 1), size(Q, 1))
    for i in axes(Q, 1), j in axes(Q, 1)
        terms = [scales[a] * Q[i, a] * Q[j, a] for a in axes(Q, 2)]
        sort!(terms; by=abs, alg=MergeSort)
        total = 0.0
        for term in terms
            total += term
        end
        H[i, j] = total
    end
    Hermitian(H)
end

function whitened_normwise_residuals(W, basis, eigenvalues)
    Wmatrix = Matrix(W)
    matrix_norm = opnorm(Wmatrix, 2)
    [norm(Wmatrix * basis[:, i] - eigenvalues[i] * basis[:, i], 2) /
        (matrix_norm * norm(basis[:, i], 2) +
            abs(eigenvalues[i]) * norm(basis[:, i], 2))
        for i in eachindex(eigenvalues)]
end

function generalized_normwise_residuals(H, K, C, basis, eigenvalues)
    Hmatrix = Matrix(H)
    Kmatrix = Matrix(K)
    theta_basis = C * basis
    Hnorm = opnorm(Hmatrix, 2)
    Knorm = opnorm(Kmatrix, 2)
    [norm(Hmatrix * theta_basis[:, i] -
        eigenvalues[i] * Kmatrix * theta_basis[:, i], 2) /
        (Hnorm * norm(theta_basis[:, i], 2) +
            abs(eigenvalues[i]) * Knorm * norm(theta_basis[:, i], 2))
        for i in eachindex(eigenvalues)]
end

function clusters_by_eigenvalue(eigenvalues; threshold=1e-8)
    order = sortperm(eachindex(eigenvalues);
        by=index -> (eigenvalues[index], index), alg=MergeSort)
    scale = max_abs_value(eigenvalues)
    groups = Vector{Vector{Int}}()
    for index in order
        if isempty(groups)
            push!(groups, [index])
            continue
        end
        previous = last(groups[end])
        gap_scale = max(scale, abs(eigenvalues[index]), abs(eigenvalues[previous]))
        if abs(eigenvalues[index] - eigenvalues[previous]) <= threshold * gap_scale
            push!(groups[end], index)
        else
            push!(groups, [index])
        end
    end
    groups
end

function float64_reduction_order_sensitivity(C, L, Q, source_matrix,
        source_eigenvalues_by_mass, source_masses, source_signs)
    H_forward = Matrix(float64_theta_hessian_column_order(L, Q, axes(Q, 2)))
    H_reverse = Matrix(float64_theta_hessian_column_order(L, Q,
        reverse(collect(axes(Q, 2)))))
    H_magnitude_sorted = Matrix(float64_theta_hessian_magnitude_sorted(L, Q))
    matrix_variants = [
        ("source_forward_column_order", Matrix(source_matrix),
            "package scaled rank-one accumulation in ascending source-column order, then (C' * H) * C"),
        ("reverse_column_order", Matrix(transpose(C) * H_reverse * C),
            "same signed rank-one terms and fixed Qtilde/Ltilde/C; reverse source-column accumulation, then (C' * H) * C"),
        ("ascending_magnitude_per_entry", Matrix(transpose(C) * H_magnitude_sorted * C),
            "same signed rank-one terms and fixed Qtilde/Ltilde/C; sort each entry's terms by ascending absolute magnitude before serial summation, then (C' * H) * C"),
        ("right_associated_whitening_product", Matrix(transpose(C) *
            (H_forward * C)),
            "same forward-order H and fixed Qtilde/Ltilde/C; compute C' * (H * C) instead of (C' * H) * C")]
    maximum(abs, H_forward - Matrix(float64_theta_hessian_scaled(L, Q))) == 0.0 ||
        error("forward-order sensitivity control differs from the diagnostic source assembly")

    records = Dict{String,Any}[]
    for (name, matrix, method) in matrix_variants
        decomposition = eigen(Hermitian(matrix))
        eigenvalues = collect(decomposition.values)
        masses = float64_source_masses(eigenvalues, L)
        order = source_mass_order(eigenvalues, masses)
        ordered_eigenvalues = eigenvalues[order]
        ordered_masses = masses[order]
        signs = Int.(sign.(ordered_eigenvalues))
        push!(records, Dict(
            "name" => name,
            "accumulation_or_product_order" => method,
            "relative_inf_matrix_difference_from_source" =>
                relative_inf_difference(matrix, source_matrix),
            "signed_eigenvalues_by_mass" => vector_strings(ordered_eigenvalues),
            "signed_eigenvalue_deltas_vs_source" => vector_strings(
                ordered_eigenvalues .- source_eigenvalues_by_mass),
            "masses" => ordered_masses,
            "mass_deltas_vs_source" => ordered_masses .- source_masses,
            "signs" => signs,
            "sign_mismatch_count_vs_source" => count(
                i -> signs[i] != source_signs[i], eachindex(signs)),
            "light_modes" => [Dict(
                "mass_index" => i,
                "signed_eigenvalue" => string(ordered_eigenvalues[i]),
                "mass" => ordered_masses[i],
                "mass_delta_vs_source" => ordered_masses[i] - source_masses[i],
                "sign" => signs[i],
                "sign_changed_vs_source" => signs[i] != source_signs[i])
                for i in 1:min(3, length(ordered_masses))]))
    end
    Dict(
        "physical_inputs_held_fixed" => true,
        "fixture_digest" => "68261b5571df88e6afe75f985eafac27a144390779306d3e4c416a6fc1caf6ef",
        "fixed_inputs" => "the same frozen Float64 C, Qtilde, and Ltilde from the B6 fixture are used in every variant; only rank-one accumulation order or dense multiplication association changes",
        "common_scaling" => "all terms use the source Float64 common-log scaling, with the same underflow floor and mass restoration",
        "variants" => records)
end

function all_permutations(values::Vector{Int})
    isempty(values) && return [Int[]]
    output = Vector{Vector{Int}}()
    for (position, value) in pairs(values)
        remaining = [values[i] for i in eachindex(values) if i != position]
        for tail in all_permutations(remaining)
            push!(output, vcat(value, tail))
        end
    end
    output
end

function deterministic_assignment(left_basis, left_signs, left_clusters,
        right_basis, right_signs, right_clusters; tie_tolerance=1e-12)
    assignments = Dict{String,Any}()
    any_ambiguity = false
    for sign_value in (-1, 0, 1)
        left_indices = findall(==(sign_value), left_signs)
        right_indices = findall(==(sign_value), right_signs)
        key = string(sign_value)
        if length(left_indices) != length(right_indices)
            assignments[key] = Dict("status" => "cardinality_mismatch",
                "left_indices" => left_indices, "right_indices" => right_indices)
            any_ambiguity = true
            continue
        elseif isempty(left_indices)
            assignments[key] = Dict("status" => "empty")
            continue
        end
        candidates = NamedTuple[]
        for permutation in all_permutations(right_indices)
            score = 0.0
            for (position, left_index) in pairs(left_indices)
                overlap = abs(dot(@view(left_basis[:, left_index]),
                    @view(right_basis[:, permutation[position]])))
                score += overlap^2
            end
            push!(candidates, (; score, permutation))
        end
        maximum_score = maximum(candidate.score for candidate in candidates)
        near_optimal = filter(candidate ->
            maximum_score - candidate.score <= tie_tolerance, candidates)
        sort!(near_optimal; by=candidate -> Tuple(candidate.permutation))
        chosen = first(near_optimal)
        alternative_scores = [candidate.score for candidate in candidates
            if candidate.permutation != chosen.permutation]
        second_score = isempty(alternative_scores) ? -Inf :
            maximum(alternative_scores)
        assignment_tied = any(maximum_score - candidate.score <= tie_tolerance
            for candidate in candidates
            if candidate.permutation != chosen.permutation)
        pair_records = Dict{String,Any}[]
        for (position, left_index) in pairs(left_indices)
            right_index = chosen.permutation[position]
            overlap = abs(dot(@view(left_basis[:, left_index]),
                @view(right_basis[:, right_index])))
            left_clustered = any(length(cluster) > 1 && left_index in cluster
                for cluster in left_clusters)
            right_clustered = any(length(cluster) > 1 && right_index in cluster
                for cluster in right_clusters)
            identity_withheld = assignment_tied || left_clustered ||
                right_clustered || overlap < 0.5
            any_ambiguity |= identity_withheld
            push!(pair_records, Dict("left_index" => left_index,
                "right_index" => right_index,
                "absolute_canonical_overlap" => overlap,
                "identity_withheld" => identity_withheld))
        end
        assignments[key] = Dict("status" => "matched",
            "objective_sum_squared_overlaps" => maximum_score,
            "chosen_objective_sum_squared_overlaps" => chosen.score,
            "second_objective" => second_score,
            "tie_tolerance" => tie_tolerance,
            "assignment_tied" => assignment_tied,
            "pairs" => pair_records)
    end
    (; assignments, any_identity_withheld=any_ambiguity)
end

function principal_angles(left_basis, right_basis, indices)
    left = Matrix(qr(left_basis[:, indices]).Q)[:, 1:length(indices)]
    right = Matrix(qr(right_basis[:, indices]).Q)[:, 1:length(indices)]
    singular_values = svdvals(left' * right)
    [acos(clamp(value, 0.0, 1.0)) for value in singular_values]
end

function tensor_log_frobenius(log_entries)
    finite_logs = filter(isfinite, log_entries)
    isempty(finite_logs) && return -Inf
    maximum_log = maximum(finite_logs)
    maximum_log + 0.5 * log10(sum(10.0^(2 * (value - maximum_log))
        for value in finite_logs))
end

function light_quartic_tensor(Q, L, C, basis, light_indices=1:3)
    Qcanonical = Matrix(Q') * C
    Qpq = Qcanonical * basis[:, light_indices]
    charge_sign, charge_logabs = G._cache_charge_sign_logabs(Qpq)
    scale_sign = Int.(sign.(L[1, :]))
    scale_log = log(10) .* L[2, :]
    positive_logs = zeros(Float64, length(scale_sign))
    negative_logs = zeros(Float64, length(scale_sign))
    signs = Int[]
    logs = Float64[]
    absolute_sum_logs = Float64[]
    for i in 1:3, j in 1:3, k in 1:3, ell in 1:3
        value_sign, value_log, absolute_log = G.pq_contracted_log!(
            positive_logs, negative_logs, scale_sign, scale_log,
            charge_sign, charge_logabs, (i, j, k, ell))
        push!(signs, value_sign)
        push!(logs, value_sign == 0 ? -Inf :
            value_log * log10(exp(1)) + 4 * log10(2π))
        push!(absolute_sum_logs, absolute_log * log10(exp(1)) +
            4 * log10(2π))
    end
    (; signs, log10_components=logs,
        log10_frobenius=tensor_log_frobenius(logs),
        log10_absolute_component_sum=tensor_log_frobenius(absolute_sum_logs),
        basis_indices=collect(light_indices))
end

function source_and_independent_matrices(C, L, Q; prec::Int)
    source_matrix, Cprecision = G.high_precision_leading_hessian(C, L, Q; prec)
    actual_bits = precision(A.ArbFloat)
    T = eltype(Cprecision)
    scales = T.(L[1, :]) .* (T(10) .^ T.(L[2, :]))
    Htheta = zeros(T, size(Q, 1), size(Q, 1))
    for a in eachindex(scales)
        support = findall(value -> !iszero(value), @view Q[:, a])
        for j in support, i in support
            Htheta[i, j] += scales[a] * Q[i, a] * Q[j, a]
        end
    end
    independent_matrix = Hermitian(transpose(Cprecision) * Htheta * Cprecision)
    (; source_matrix, independent_matrix, Htheta, scales, Cprecision, actual_bits)
end

function one_precision(digits, K, Ltilde, Qtilde, Kmatrix, C, Q, L,
        float_matrix)
    prior_bits = precision(A.ArbFloat)
    result = nothing
    try
        result = G.pq_spectrum(K, L, Q; mixing_correction=:high_precision,
            prec=digits, quartic_diagnostics=true, mass_basis_diagnostics=true,
            hierarchy_diagnostics=true)
        route_bits = precision(A.ArbFloat)
        pieces = source_and_independent_matrices(C, Ltilde, Qtilde; prec=digits)
        T = eltype(Matrix(pieces.source_matrix))
        matrix = Matrix(pieces.source_matrix)
        decomposition = eigen(pieces.source_matrix)
        raw_eigenvalues = collect(decomposition.values)
        eigenvectors = Matrix(decomposition.vectors)
        offset = 9.0 + Float64(log10(G.constants()["MPlanck"])) +
            Float64(G.constants()["log2π"])
        raw_masses = Float64.(0.5 .* log10.(abs.(raw_eigenvalues))) .+ offset
        order = source_mass_order(raw_eigenvalues, raw_masses)
        ordered_eigenvalues = raw_eigenvalues[order]
        masses = raw_masses[order]
        signs = Int.(sign.(ordered_eigenvalues))
        basis = Float64.(eigenvectors[:, order])
        scale = max_abs_value(raw_eigenvalues)
        residuals = whitened_normwise_residuals(pieces.source_matrix,
            eigenvectors, raw_eigenvalues)
        backward_error = maximum(residuals)
        orthogonality = opnorm(transpose(eigenvectors) * eigenvectors -
            Matrix{T}(I, size(eigenvectors, 2), size(eigenvectors, 2)), Inf)
        minimum_abs = minimum(abs, raw_eigenvalues)
        condition = minimum_abs == 0 ? "Inf" : string(scale / minimum_abs)

        promoted_float_eigen = eigen(Hermitian(T.(float_matrix)))
        promoted_float_eigenvalues = collect(promoted_float_eigen.values)
        promoted_float_eigenvectors = Matrix(promoted_float_eigen.vectors)
        promoted_reference_log = T(maximum(Ltilde[2, :]))
        promoted_offset = 9.0 + Float64(log10(G.constants()["MPlanck"])) +
            Float64(log10(2π))
        promoted_raw_masses = Float64.(0.5 .* log10.(
            abs.(promoted_float_eigenvalues))) .+
            0.5 * Float64(promoted_reference_log) .+ promoted_offset
        promoted_order = source_mass_order(promoted_float_eigenvalues,
            promoted_raw_masses)
        promoted_float_masses = promoted_raw_masses[promoted_order]
        promoted_float_signs = Int.(sign.(
            promoted_float_eigenvalues[promoted_order]))
        promoted_scale = max_abs_value(promoted_float_eigenvalues)
        promoted_float_residuals = whitened_normwise_residuals(
            Hermitian(T.(float_matrix)), promoted_float_eigenvectors,
            promoted_float_eigenvalues)
        scaled_float_matrix_error = relative_inf_difference(
            T.(float_matrix) .* (T(10)^promoted_reference_log), matrix)

        K_hp = Hermitian(T.(Kmatrix))
        C_float_hp = T.(C)
        K_identity = Matrix{T}(I, size(Kmatrix, 1), size(Kmatrix, 2))
        float_whitening_error = opnorm(transpose(C_float_hp) *
            Matrix(K_hp) * C_float_hp - K_identity, Inf)
        C_hp = Matrix(inv(cholesky(K_hp).U))
        whitening_error_hp = opnorm(transpose(C_hp) * Matrix(K_hp) * C_hp -
            K_identity, Inf)
        hp_whitened = Hermitian(transpose(C_hp) * pieces.Htheta * C_hp)
        hp_whitened_decomposition = eigen(hp_whitened)
        hp_whitened_eigenvalues = collect(hp_whitened_decomposition.values)
        hp_whitened_eigenvectors = Matrix(hp_whitened_decomposition.vectors)
        hp_whitened_masses = Float64.(0.5 .* log10.(abs.(hp_whitened_eigenvalues))) .+ offset
        hp_whitened_order = source_mass_order(hp_whitened_eigenvalues,
            hp_whitened_masses)
        hp_generalized_residuals = generalized_normwise_residuals(
            pieces.Htheta, K_hp, C_hp, hp_whitened_eigenvectors,
            hp_whitened_eigenvalues)

        conversion_delta = zero(T)
        float_inputs = vcat(vec(Ltilde), vec(C))
        promotion_roundtrip_exact = true
        for value in float_inputs
            promotion_roundtrip_exact &= Float64(T(value)) == value
            conversion_delta = max(conversion_delta,
                abs(T(value) - T(string(value))))
        end

        light_tensor = light_quartic_tensor(Q, L, C, basis)
        light_clusters = clusters_by_eigenvalue(ordered_eigenvalues)
        public_mass_difference = maximum(abs.(masses .- result.m))
        (; result, route_bits, pieces, matrix, raw_eigenvalues, eigenvectors,
            ordered_eigenvalues, masses, signs, basis, order, scale,
            residuals, backward_error, orthogonality, condition,
            promoted_float_eigenvalues, promoted_float_masses,
            promoted_float_signs, promoted_float_order=promoted_order,
            promoted_float_residuals, scaled_float_matrix_error,
            whitening_error_hp, C_hp, hp_whitened_eigenvalues,
            hp_whitened_masses, hp_whitened_order, hp_generalized_residuals,
            conversion_delta, promotion_roundtrip_exact,
            float_whitening_error,
            light_tensor, light_clusters, public_mass_difference)
    finally
        setprecision(A.ArbFloat; bits=prior_bits)
    end
end

function precision_record(digits, record)
    source_independent_error = relative_inf_difference(
        record.matrix, Matrix(record.pieces.independent_matrix))
    values = (; digits_requested=digits,
        arbfloat_bits_readback=record.route_bits,
        arbfloat_bits_matrix=record.pieces.actual_bits,
        float64_to_arbfloat_roundtrip_exact=record.promotion_roundtrip_exact,
        float64_vs_shortest_decimal_expansion_gap_max_not_conversion_error=
            string(record.conversion_delta),
        float64_cholesky_whitening_relative_error=string(record.float_whitening_error),
        arbitrary_precision_cholesky_whitening_error=string(record.whitening_error_hp),
        arbitrary_precision_K_generalized_normwise_residuals=vector_strings(
            record.hp_generalized_residuals[record.hp_whitened_order]),
        arbitrary_precision_whitening_eigenvalues=vector_strings(
            record.hp_whitened_eigenvalues[record.hp_whitened_order]),
        arbitrary_precision_whitening_masses=record.hp_whitened_masses[
            record.hp_whitened_order],
        independent_hessian_relative_inf_error=string(source_independent_error),
        normwise_whitened_residual_definition=
            "||W*u-lambda*u||_2/(||W||_2*||u||_2+abs(lambda)*||u||_2); W and u are in the explicitly identified canonical whitened coordinates",
        generalized_K_residual_definition=
            "||H_theta*v-lambda*K*v||_2/(||H_theta||_2*||v||_2+abs(lambda)*||K||_2*||v||_2); theta coordinates, matrix norm is spectral 2-norm",
        float64_scaled_matrix_construction_relative_inf_error=
            string(record.scaled_float_matrix_error),
        high_precision_solver_on_float64_matrix_eigenvalues=vector_strings(
            record.promoted_float_eigenvalues[record.promoted_float_order]),
        high_precision_solver_on_float64_matrix_masses=
            record.promoted_float_masses,
        high_precision_solver_on_float64_matrix_signs=record.promoted_float_signs,
        high_precision_solver_on_float64_matrix_residuals=vector_strings(
            record.promoted_float_residuals[record.promoted_float_order]),
        condition_number_from_eigenvalues=record.condition,
        eigenvalues_in_solver_order=vector_strings(record.raw_eigenvalues),
        eigenvalues_in_mass_order=vector_strings(record.ordered_eigenvalues),
        masses=record.masses, signs=record.signs,
        eigenvector_order=record.order,
        eigenvectors_in_mass_order=[collect(row) for row in eachrow(record.basis)],
        dimensionless_eigenpair_residuals=vector_strings(record.residuals),
        max_dimensionless_eigenpair_residual=string(record.backward_error),
        dimensionless_orthogonality_inf=string(record.orthogonality),
        public_pq_spectrum_masses=record.result.m,
        public_vs_explicit_mass_max_abs_difference=record.public_mass_difference,
        public_pq_spectrum_signs=record.result.msign,
        quartic_self_sign=record.result.λselfsign,
        quartic_self_log10=record.result.λself,
        quartic31_indices=[collect(row) for row in eachrow(record.result.λ31_i)],
        quartic31_sign=record.result.λ31sign,
        quartic31_log10=record.result.λ31,
        quartic22_indices=[collect(row) for row in eachrow(record.result.λ22_i)],
        quartic22_sign=record.result.λ22sign,
        quartic22_log10=record.result.λ22,
        light_sector_tensor_sign=record.light_tensor.signs,
        light_sector_tensor_log10_components=record.light_tensor.log10_components,
        light_sector_quartic_tensor_log10_frobenius=
            record.light_tensor.log10_frobenius,
        light_sector_quartic_abs_component_sum_log10_frobenius=
            record.light_tensor.log10_absolute_component_sum,
        eigen_matrix_theta=matrix_strings(record.pieces.Htheta),
        canonical_pre_diagonalization_matrix=matrix_strings(record.matrix),
        canonical_independent_matrix=matrix_strings(record.pieces.independent_matrix),
        C_float64_converted_to_arb=matrix_strings(record.pieces.Cprecision),
        C_arbitrary_precision_from_K=matrix_strings(record.C_hp),
        light_eigenvalue_clusters=record.light_clusters)
    Dict{String,Any}(string(key) => value for (key, value) in pairs(values))
end

function main()
    potential = PB.n5_potential(k=1.0)
    K = PB.n5_kinetic_matrix(1.0)
    Kmatrix = Matrix(K)
    selection = G.LQtilde(potential.Q, potential.L)
    C = G._canonical_factor_from_K(K)
    fixture_digest = bytes2hex(sha256(codeunits(join(string.(
        ("B6", potential.Q, potential.L, K, :float64, :high_precision,
         80, true, true, true), "|")))))
    fixture_digest == "68261b5571df88e6afe75f985eafac27a144390779306d3e4c416a6fc1caf6ef" ||
        error("P0 B6 fixture digest mismatch: $fixture_digest")
    p1 = TOML.parsefile(joinpath(@__DIR__, "p1_replay_reference.toml"))
    p1["fixture_digest_p0"] == fixture_digest || error("reference P1 digest does not match")
    p1float = TOML.parsefile(joinpath(@__DIR__, "p1_replay.toml"))
    p1float["fixture_digest_p0"] == fixture_digest || error("comparison P1 digest does not match")

    original_bits = precision(A.ArbFloat)
    float_matrix = G.leading_hessian_matrix_float64_scaled(C,
        selection.Ltilde, selection.Qtilde)
    float_eigen = eigen(float_matrix)
    float_solver_eigenvalues = collect(float_eigen.values)
    float_solver_masses = float64_source_masses(float_solver_eigenvalues,
        selection.Ltilde)
    float_solver_mass_order = source_mass_order(float_solver_eigenvalues,
        float_solver_masses)
    float_ordered_eigenvalues = float_solver_eigenvalues[float_solver_mass_order]
    float_mass, float_sign, float_basis =
        G.leading_hessian_mass_basis_float64(K, selection.Ltilde,
            selection.Qtilde)
    isapprox(float_mass, BASELINE_FLOAT_MASSES; rtol=2e-13, atol=2e-13) ||
        error("Float64 source matrix no longer reproduces frozen masses")
    float_sign == BASELINE_FLOAT_SIGNS || error("Float64 source signs changed")
    isapprox(float_solver_masses[float_solver_mass_order], float_mass;
        rtol=2e-13, atol=2e-13) || error("paired Float64 solver values do not reproduce the mass basis")
    Int.(sign.(float_ordered_eigenvalues)) == float_sign ||
        error("paired Float64 solver eigenvalues do not reproduce source mass-basis signs")
    float_mass_basis = Matrix(float_basis)
    float_solver_vectors_by_mass = Matrix(
        float_eigen.vectors[:, float_solver_mass_order])
    float_solver_vector_overlaps = [abs(dot(@view(float_mass_basis[:, i]),
        @view(float_solver_vectors_by_mass[:, i]))) for i in axes(float_mass_basis, 2)]
    all(overlap >= 1.0 - 1e-12 for overlap in float_solver_vector_overlaps) ||
        error("Float64 reported basis columns do not pair with the solver eigenvectors")
    float_whitened_residuals = whitened_normwise_residuals(float_matrix,
        float_mass_basis, float_ordered_eigenvalues)
    float_theta_matrix = float64_theta_hessian_scaled(selection.Ltilde,
        selection.Qtilde)
    float_generalized_residuals = generalized_normwise_residuals(
        float_theta_matrix, Kmatrix, C, float_mass_basis,
        float_ordered_eigenvalues)
    reduction_order_sensitivity = float64_reduction_order_sensitivity(C,
        selection.Ltilde, selection.Qtilde, float_matrix,
        float_ordered_eigenvalues, float_mass, float_sign)

    precision_records = [one_precision(digits, K, selection.Ltilde,
        selection.Qtilde, Kmatrix, C, potential.Q, potential.L, float_matrix)
        for digits in REQUESTED_DIGITS]
    precision( A.ArbFloat) == original_bits || error("final ArbFloat state was not restored")
    reports = [precision_record(digits, record)
        for (digits, record) in zip(REQUESTED_DIGITS, precision_records)]

    highest = last(precision_records)
    isapprox(highest.masses, BASELINE_HIGH_MASSES; atol=2e-12, rtol=2e-12) ||
        error("512-digit ladder result does not reproduce frozen high-precision baseline")
    highest.signs == BASELINE_HIGH_SIGNS || error("512-digit signs differ from frozen baseline")
    isfinite(highest.light_tensor.log10_frobenius) ||
        error("light-sector quartic tensor norm is not finite")

    reference_high = precision_records[2]
    float_clusters = clusters_by_eigenvalue(float_ordered_eigenvalues)
    reference_clusters = reference_high.light_clusters
    comparison_float = deterministic_assignment(float_mass_basis,
        float_sign, float_clusters, reference_high.basis,
        reference_high.signs, reference_clusters)
    highest_clusters = highest.light_clusters
    ladder_comparisons = Dict{String,Any}[]
    for (digits, record) in zip(REQUESTED_DIGITS, precision_records)
        angles = principal_angles(record.basis, highest.basis, 1:3)
        assignment = deterministic_assignment(record.basis, record.signs,
            record.light_clusters, highest.basis, highest.signs,
            highest_clusters)
        push!(ladder_comparisons, Dict(
            "digits" => digits,
            "light_subspace_principal_angles_radians_vs_512" => angles,
            "light_subspace_projector_frobenius_distance_vs_512" =>
                norm(record.basis[:, 1:3] * record.basis[:, 1:3]' -
                    highest.basis[:, 1:3] * highest.basis[:, 1:3]'),
            "same_sign_assignment_vs_512" => assignment.assignments,
            "individual_identity_withheld" => assignment.any_identity_withheld))
    end
    float_high_angles = principal_angles(float_mass_basis, highest.basis, 1:3)
    light_tensor_float = light_quartic_tensor(potential.Q, potential.L, C,
        float_mass_basis)
    p1_high_basis = reduce(vcat,
        [permutedims(Float64.(row)) for row in
            p1["routes"]["high_precision_80"]["basis_vectors"]])
    p0_high_tensor = light_quartic_tensor(potential.Q, potential.L, C,
        p1_high_basis)

    env_root = dirname(dirname(@__DIR__))
    source_root = get(ENV, "CYAX173_REFERENCE_ROOT", env_root)
    benchmark_manifest = joinpath(env_root,
        "validation/p0_numerical_equivalence/environment_benchmarks/Manifest.toml")
    benchmark_project = joinpath(env_root,
        "validation/p0_numerical_equivalence/environment_benchmarks/Project.toml")
    source_paths = ["src/generate.jl", "src/structs.jl",
        "src/paper_benchmarks/reduced_models.jl",
        "src/paper_benchmarks/poly102_inflation.jl"]
    source_hashes = Dict(path => sha_file(joinpath(source_root, path))
        for path in source_paths)
    reference_commit = get(ENV, "CYAX173_REFERENCE_COMMIT",
        "7a40285bb5c313f7e8746b90644d5f45bb67be44")
    base_commit = get(ENV, "CYAX173_EXECUTION_BASE",
        "74fea608684eae746f25ae45b18512f89b862fb0")

    report = Dict{String,Any}(
        "schema" => "cyax-0173-spectrum-diagnostic-v1",
        "work_item" => "CYAX-0173 frozen N=5 B6 diagnostic",
        "reference_commit" => reference_commit,
        "reference_tree" => "ef79ad720ea0ed1ad2193b2616a97319d8282a61",
        "execution_base" => base_commit,
        "reference_checkout_id" => "temporary_archive_checkout_at_" * reference_commit,
        "source_paths_sha256" => source_hashes,
        "matrix_and_residual_methods" => Dict(
            "coordinate_transform" =>
                "C=inv(cholesky(K).U); theta=C*phi; C'*K*C=I; Q_phi=Q'*C",
            "canonical_leading_hessian" =>
                "H_theta=Qtilde*diag(sign(Ltilde[1,:])*10^Ltilde[2,:])*Qtilde'; W=C'*H_theta*C; W*u=lambda*u is equivalent to H_theta*v=lambda*K*v when v=C*u and C is the exact K-whitening factor",
            "float64_canonical_matrix" =>
                "source leading_hessian_matrix_float64_scaled(C,Ltilde,Qtilde), with its common scale recorded and restored in mass conversion",
            "precision_conversion" =>
                "frozen Float64 C and Ltilde entries are promoted directly to ArbFloat; round-trip equality verifies input preservation, while the separately recorded shortest-decimal expansion gap is not a conversion-error estimate",
            "direct_rank_one_check" =>
                "diagnostic accumulates each nonzero-supported Q column's signed scale times q*q' into H_theta, then forms C'*H_theta*C; this mirrors the package helper's accumulation order and checks formula/input-path agreement, not an independent physical oracle",
            "fixed_input_reduction_order_sensitivity" =>
                "hold the frozen Float64 C, Qtilde, Ltilde, common log scale, and underflow floor fixed; compare the source forward column order with reverse column order, ascending-magnitude per-entry term reduction, and right-associated dense whitening multiplication, then report each scaled matrix/eigenspectrum change",
            "normwise_whitened_residual" =>
                "||W*u-lambda*u||_2/(||W||_2*||u||_2+abs(lambda)*||u||_2); 2-norms and canonical whitened coordinates",
            "normwise_generalized_residual" =>
                "||H_theta*v-lambda*K*v||_2/(||H_theta||_2*||v||_2+abs(lambda)*||K||_2*||v||_2); 2-norms and theta coordinates",
            "independent_metric_cross_check" =>
                "recompute C_K=inv(cholesky(K_high_precision).U), solve C_K'*H_theta*C_K, and evaluate the generalized H_theta*v=lambda*K*v residuals; this checks the metric transform independently from C converted from Float64"),
        "environment" => Dict(
            "julia_version" => string(VERSION),
            "julia_build_commit" => string(Base.GIT_VERSION_INFO.commit),
            "machine" => Sys.MACHINE, "arch" => string(Sys.ARCH),
            "blas_vendor" => string(BLAS.vendor()),
            "blas_threads" => BLAS.get_num_threads(),
            "julia_threads" => Threads.nthreads(),
            "benchmark_project_sha256" => sha_file(benchmark_project),
            "benchmark_manifest_sha256" => sha_file(benchmark_manifest),
            "requested_precision_decimal_digits" => collect(REQUESTED_DIGITS),
            "arbfloat_precision_bits_before" => original_bits,
            "arbfloat_precision_bits_after_all_runs" => precision(A.ArbFloat),
            "arbfloat_precision_restored" => precision(A.ArbFloat) == original_bits,
            "matching_declaration_sha256" => sha_file(joinpath(@__DIR__,
                "matching_declaration.json")),
            "p1_reference_report_sha256" => sha_file(joinpath(@__DIR__,
                "p1_replay_reference.toml"))),
        "fixture" => Dict(
            "p0_digest" => fixture_digest,
            "Q" => [collect(row) for row in eachrow(potential.Q)],
            "L" => matrix_strings(potential.L),
            "K" => matrix_strings(Kmatrix),
            "C_float64" => matrix_strings(C),
            "Qtilde" => [collect(row) for row in eachrow(selection.Qtilde)],
            "Ltilde" => matrix_strings(selection.Ltilde),
            "Qbar" => [collect(row) for row in eachrow(selection.Qbar)],
            "Lbar" => matrix_strings(selection.Lbar),
            "metric_whitening_error_inf" =>
                opnorm(transpose(C) * Kmatrix * C - I, Inf),
            "K_condition_2" => cond(Kmatrix, 2),
            "transformation" => "C=inv(cholesky(K).U); theta=C*phi; C'*K*C=I; Q_phi=Q'*C; H_theta=Qtilde*diag(sign(Ltilde[1,:])*10^Ltilde[2,:])*Qtilde'; W=C'*H_theta*C; W*u=lambda*u is equivalent to H_theta*v=lambda*K*v for v=C*u",
            "float64_pre_diagonalization_matrix" => matrix_strings(float_matrix)),
        "float64_vs_reference_high_precision80" => Dict(
            "float64_masses" => float_mass, "float64_signs" => float_sign,
            "high_precision80_masses" => BASELINE_HIGH_MASSES,
            "high_precision80_signs" => BASELINE_HIGH_SIGNS,
            "metric_consistent_light_subspace_principal_angles_radians" =>
                principal_angles(float_mass_basis,
                    p1_high_basis,
                    1:3),
            "light_subspace_projector_frobenius_distance_float64_vs_80" =>
                norm(float_mass_basis[:, 1:3] * float_mass_basis[:, 1:3]' -
                    p1_high_basis[:, 1:3] * p1_high_basis[:, 1:3]'),
            "light_subspace_principal_angles_float64_vs_512" => float_high_angles,
            "principal_angle_roundoff_note" =>
                "Angles from 1.49e-8 to 2.58e-8 radians with projector Frobenius distances 0 to 3.6e-16 are acos(Float64 singular-value) resolution floor; do not interpret them as physical subspace rotation.",
            "float64_eigenpair_residuals_from_reference_replay" =>
                p1["routes"]["float64"]["eigenpair_residuals"],
            "high_precision80_eigenpair_residuals_from_reference_replay" =>
                p1["routes"]["high_precision_80"]["eigenpair_residuals"],
            "float64_normwise_whitened_residual_definition" =>
                "||W*u-lambda*u||_2/(||W||_2*||u||_2+abs(lambda)*||u||_2); W is the Float64-assembled scaled canonical matrix and u is in canonical coordinates.",
            "float64_normwise_whitened_residuals_float64_assembled_matrix" =>
                vector_strings(float_whitened_residuals),
            "float64_normwise_generalized_residual_definition" =>
                "||H_theta*v-lambda*K*v||_2/(||H_theta||_2*||v||_2+abs(lambda)*||K||_2*||v||_2); H_theta and K are Float64, v=C*u in theta coordinates.",
            "float64_normwise_generalized_residuals_float64_assembled_Htheta_K" =>
                vector_strings(float_generalized_residuals),
            "float64_solver_eigenvalues_paired_by_reported_mass_order" =>
                vector_strings(float_ordered_eigenvalues),
            "absolute_overlap_reported_mass_basis_vs_paired_solver_vectors" =>
                vector_strings(float_solver_vector_overlaps),
            "float64_light_eigenvalue_clusters_numeric_eigenvalue_order" =>
                float_clusters,
            "high_precision80_light_eigenvalue_clusters_numeric_eigenvalue_order" =>
                reference_clusters,
            "fixed_input_reduction_order_sensitivity" => reduction_order_sensitivity,
            "reference_replay_builtin_mass_basis_accuracy_residual_definition" =>
                "||W*u-lambda*u||_2/max(abs(lambda),eps(Float64)*||W||_infinity); retained as a separate source diagnostic, not the normwise formula above.",
            "same_sign_one_to_one_assignment" => comparison_float.assignments,
            "individual_identity_withheld" => comparison_float.any_identity_withheld,
            "float64_light_sector_quartic_tensor_log10_frobenius" =>
                light_tensor_float.log10_frobenius,
            "high_precision80_light_sector_quartic_tensor_log10_frobenius" =>
                p0_high_tensor.log10_frobenius),
        "precision_ladder" => reports,
        "matching_to_512" => ladder_comparisons)
    open(OUT, "w") do io
        TOML.print(io, report; sorted=true)
    end
    println("P2_LADDER_PASS")
    println("fixture_digest=$fixture_digest")
    for (digits, record) in zip(REQUESTED_DIGITS, precision_records)
        println("digits=$digits bits=$(record.route_bits) masses=$(repr(record.masses)) signs=$(repr(record.signs)) max_backward_error=$(record.backward_error) cond=$(record.condition)")
    end
    println("matching_declaration_sha256=$(sha_file(joinpath(@__DIR__, "matching_declaration.json")))")
    println("arbfloat_bits_restored=$(precision(A.ArbFloat))")
    println("report=$OUT")
end

main()
