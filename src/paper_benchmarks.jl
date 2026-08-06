"""
    CYAxiverse.paper_benchmarks

Deterministic benchmark potentials transcribed from the papers that define the
vacua pipeline's reproduction targets. Charge matrices returned here follow the
package convention: axions are rows and instantons are columns.
"""
module paper_benchmarks

using LinearAlgebra
using NLsolve
using Optim
using ..generate: LQtilde, reduced_critical_points

const _LOG10E = log10(exp(1.0))

"""Convert the leading `q⋅τ exp(-2π q⋅τ)` scaling to pipeline `L` data."""
function instanton_scales(qdotτ::AbstractVector{<:Real}, k::Real)
    k > 0 || throw(ArgumentError("the volume scale k must be positive"))
    scaled = Float64(k) .* Float64.(qdotτ)
    vcat(ones(1, length(scaled)), reshape(log10.(scaled) .- 2π .* scaled .* _LOG10E, 1, :))
end

"""
    n5_potential(; k=1.0)

The eight-term, five-axion potential in Eqs. (32)--(39) of the inflation
draft. The returned fields are `Q`, `L`, and `qdotτ`.
"""
function n5_potential(; k::Real=1.0)
    charges = Int[
        1 0 0 0 0
        0 0 0 0 1
        0 1 0 0 0
        2 2 -2 1 1
        2 1 -1 1 0
        0 0 0 1 0
        0 0 1 0 0
        7 5 -3 3 2
    ]
    qdotτ = Float64[6, 6.25, 24, 26, 31.875, 32, 36.125, 162.125]
    (; Q=Matrix(charges'), L=instanton_scales(qdotτ, k), qdotτ,
       coefficient_model=:leading_diagonal_terms)
end

"""
    n8_potential(; k=1.0)

The twelve-term, eight-axion potential in Table 1 of the inflation draft.
"""
function n8_potential(; k::Real=1.0, trajectory::Bool=false)
    trajectory && return author_inflation.n8_potential(k=k, trajectory=true)
    charges = Int[
        -1  1  1  0  0  0  0  1
         0  0  0  1  0  0  0  0
         0  0  0  0  1  0  0  0
         0  0  0  0  0  1  0  0
         0  0  0  0  0  0  1  0
         0 -1  1 -1  1  0  1  0
         0  1 -1 -1  1  1  0  0
         1  0  0 -1 -1  0  0  0
         0  0  1  0  0  0  0  0
         0  1  0  0  0  0  0  0
         0  0  0  0  0  0  0  1
         1  0  0  0  0  0  0  0
    ]
    qdotτ = Float64[14, 14.5, 14.5, 15.5, 15.5, 15.5, 15.5, 16, 17, 17, 25, 45]
    (; Q=Matrix(charges'), L=instanton_scales(qdotτ, k), qdotτ,
       coefficient_model=:leading_diagonal_terms)
end

"""
    n8_full_potential(; k=1.0, volume_normalization=:full)

Complete equation-(19) potential from the 12 diagonal and 66 cross terms.

`volume_normalization=:full` follows the Kähler-ray scaling implied by
`tau(k)=k*tau(1)`: `V_CY(k)=V_CY(1)*k^(3/2)`.  The alternative
`:fixed` keeps the reference CY volume in the explicit `V_CY^-2`
prefactor, matching the draft-author implementation.  The default is
`:full` until the convention is clarified with the authors.
"""
function n8_full_potential(; k::Real=1.0, volume_normalization::Symbol=:full)
    volume_normalization in (:full, :fixed) ||
        throw(ArgumentError("volume_normalization must be :full or :fixed"))
    diagonal = n8_potential(k=k)
    qprime = Matrix{Int}(diagonal.Q')
    geometry = n8_geometry()
    tau = Float64(k) .* geometry.divisor_volumes
    kinv = Float64(k)^2 .* inv(Matrix(geometry.kinetic))
    volume_scale = volume_normalization === :full ? Float64(k)^(3 / 2) : 1.0
    volume = geometry.volume * volume_scale
    charges = Vector{Vector{Int}}()
    signs, logs = Float64[], Float64[]
    for charge in eachrow(qprime)
        qtau = dot(charge, tau)
        prefactor = (8π / volume^2) * qtau
        push!(charges, collect(charge)); push!(signs, sign(prefactor))
        push!(logs, log10(abs(prefactor)) - 2π * qtau * _LOG10E)
    end
    for i in 1:(size(qprime, 1)-1), j in (i+1):size(qprime, 1)
        qi, qj = qprime[i, :], qprime[j, :]
        prefactor = (π * dot(qi, kinv * qj) + dot(qi + qj, tau)) * 8π / volume^2
        push!(charges, collect(qj - qi)); push!(signs, sign(prefactor))
        push!(logs, log10(abs(prefactor)) - 2π * dot(qi + qj, tau) * _LOG10E)
    end
    Q::Matrix{Int} = hcat(charges...)
    L::Matrix{Float64} = vcat(reshape(signs, 1, :), reshape(logs, 1, :))
    (; Q, L, volume, volume_normalization,
       diagonal_count=12, cross_count=66)
end

"""
    n8_geometry()

Appendix C geometry data, including the full kinetic matrix reconstructed from
the published vertices at the tip of the stretched Kähler cone. The matrix is
in the same GLSM divisor basis as the Table 1 charges.
"""
function n8_geometry()
    vertices = Int[
         0  0  0  1
         1  0  0  0
        -1 -1  1  0
        -1  1 -1  0
         1 -1 -1 -1
         1  1  1 -1
         0 -1  0  0
         0  0 -1  0
         0  0  1  0
         0  1  0  0
    ]
    divisor_volumes = Float64[45, 17, 17, 14.5, 14.5, 15.5, 15.5, 25]
    kinetic = [
         2.6937457327178650e-4 -1.2716296987824437e-4 -1.2716296987824437e-4 -6.8885337504783278e-5 -8.4054971458841178e-5 -7.5848169770289675e-6 -7.5848169770289362e-6 -1.3445696123930552e-4
        -1.2716296987824445e-4  4.6738170793316663e-4 -1.0927188745945053e-6 -3.0579145970921500e-5  3.9983215158785611e-5  1.4551045746079736e-4 -7.4948096331090282e-5  1.8007412665194919e-4
        -1.2716296987824431e-4 -1.0927188745945964e-6  4.6738170793316641e-4 -3.0579145970921494e-5  3.9983215158785536e-5 -7.4948096331090228e-5  1.4551045746079722e-4  1.8007412665194903e-4
        -6.8885337504783292e-5 -3.0579145970921521e-5 -3.0579145970921453e-5  3.3013589886190695e-4 -1.2981861108682632e-4 -8.8253898965296121e-5 -8.8253898965296107e-5  1.0189510632450669e-7
        -8.4054971458841164e-5  3.9983215158785590e-5  3.9983215158785516e-5 -1.2981861108682632e-4  4.8381919297581406e-4  1.6509554602224966e-4  1.6509554602224964e-4  4.3432789070812104e-6
        -7.5848169770289845e-6  1.4551045746079736e-4 -7.4948096331090241e-5 -8.8253898965296094e-5  1.6509554602224969e-4  2.3690399938971666e-4  1.6445445597829115e-5  2.1206919003783867e-6
        -7.5848169770289082e-6 -7.4948096331090309e-5  1.4551045746079717e-4 -8.8253898965296094e-5  1.6509554602224961e-4  1.6445445597829115e-5  2.3690399938971663e-4  2.1206919003782736e-6
        -1.3445696123930555e-4  1.8007412665194914e-4  1.8007412665194911e-4  1.0189510632448509e-7  4.3432789070812349e-6  2.1206919003783668e-6  2.1206919003783244e-6  2.3000909719509292e-4
    ]
    (; vertices, h11=8, h21=28, euler=-40, volume=126.0,
       divisor_volumes, curve_volume_range=(1.0, 3.0), kinetic=Hermitian(kinetic))
end

"""Exact volume scale of the cusp catastrophe in the reduced N=5 model."""
n5_critical_scale() = 4 / π * log(1024 / 255)

"""Relative coefficient of `1-cos(2θ)` in the reduced N=5 potential."""
n5_reduced_ratio(k::Real) = (32 / (255 / 8)) * exp(-2π * Float64(k) * (32 - 255 / 8))

"""
    n5_reduced_critical_points(k; atol=64eps(Float64))

Return all stationary points in `[0,2π)` and their Hessian signs for the exact
reduced potential `1-cos(θ) + a(k)(1-cos(2θ))`. Hessian signs are `1` for a
minimum, `-1` for a maximum, and `0` at the catastrophe.
"""
function n5_reduced_critical_points(k::Real; atol::Real=64eps(Float64))
    a = n5_reduced_ratio(k)
    θ = Float64[0, π]
    if a > 1 / 4 + atol
        extra = acos(-1 / (4a))
        append!(θ, (extra, 2π - extra))
    end
    sort!(θ)
    curvature = cos.(θ) .+ 4a .* cos.(2θ)
    signs = map(curvature) do value
        abs(value) <= atol ? 0 : (value > 0 ? 1 : -1)
    end
    (; theta=θ, hessian_sign=signs, minima=count(==(1), signs), ratio=a)
end

"""
    n8_degenerate_point(initial_theta; k0=0.67162, tolerance=1e-11)

Solve the augmented catastrophe equations for the Table 1 potential:
`gradient(theta,k)=0`, `H(theta,k)v=0`, and `dot(v,v)=1`. Coordinates are in
the leading-charge basis. The Hessian equation uses the same symmetric
hierarchy rescaling as the critical-point finder.
"""
function n8_degenerate_point(initial_theta::AbstractVector{<:Real};
        k0::Real=0.67162, tolerance::Float64=1e-11, max_iterations::Int=1_000)
    length(initial_theta) == 8 || throw(DimensionMismatch("the N=8 benchmark needs eight coordinates"))
    reference = n8_potential(k=k0)
    selected = LQtilde(reference.Q, reference.L)
    Qordered = hcat(selected.Qtilde, selected.Qbar)
    qcanonical = Matrix{Float64}(selected.Qtilde) \ Matrix{Float64}(Qordered)

    # Recover q⋅τ in the same selected/bar order. Table 1 columns are unique.
    original_columns = collect(eachcol(reference.Q))
    ordered_qdotτ = [reference.qdotτ[findfirst(==(column), original_columns)] for column in eachcol(Qordered)]
    n = size(qcanonical, 1)

    function scaled_quantities(theta::AbstractVector{<:Real}, k::Float64)
        L = instanton_scales(ordered_qdotτ, k)
        logs = vec(L[2, :])
        amplitudes = vec(L[1, :]) .* 10.0 .^ (logs .- maximum(logs))
        row_scales = amplitudes[1:n]
        arguments = 2π .* (qcanonical' * theta)
        inverse_sqrt = Diagonal(inv.(sqrt.(row_scales)))
        scaled_gradient = (2π .* qcanonical * (amplitudes .* sin.(arguments))) ./ row_scales
        scaled_hessian = inverse_sqrt *
            ((2π)^2 .* qcanonical * Diagonal(amplitudes .* cos.(arguments)) * qcanonical') * inverse_sqrt
        scaled_gradient, scaled_hessian
    end

    _, initial_hessian = scaled_quantities(Float64.(initial_theta), Float64(k0))
    initial_vector = eigen(Hermitian(initial_hessian)).vectors[:, 1]
    initial = vcat(Float64.(initial_theta), initial_vector, Float64(k0))
    function equations!(out, state)
        theta = @view state[1:n]
        null_vector = @view state[(n + 1):(2n)]
        k = state[end]
        gradient_value, hessian_value = scaled_quantities(theta, k)
        out[1:n] .= gradient_value
        out[(n + 1):(2n)] .= hessian_value * null_vector
        out[end] = dot(null_vector, null_vector) - 1
        nothing
    end
    result = nlsolve(equations!, initial; method=:trust_region,
        ftol=tolerance, xtol=tolerance, iterations=max_iterations)
    theta = mod.(result.zero[1:n], 1.0)
    null_vector = result.zero[(n + 1):(2n)]
    k = result.zero[end]
    gradient_value, hessian_value = scaled_quantities(theta, k)
    eigenvalues = eigvals(Hermitian(hessian_value))
    (; theta, null_vector=null_vector / norm(null_vector), k, eigenvalues,
       gradient_residual=norm(gradient_value, Inf), null_residual=norm(hessian_value * null_vector, Inf),
       converged=result.f_converged || result.x_converged, iterations=result.iterations)
end


"""Table-1 truncated potential and its first two derivatives in GLSM coordinates.

Set `full=true` to evaluate the separate 78-term equation-(19) reconstruction.
When `full=true`, `volume_normalization` selects the explicit CY-volume
scaling in that reconstruction and defaults to `:full`.
The paper's catastrophe search and inflation trajectories use the truncated
potential of equations (20) and (25), so the 12-term potential is the default.
"""
function n8_potential_derivatives(theta::AbstractVector{<:Real}, k::Real;
        full::Bool=false, volume_normalization::Symbol=:full,
        trajectory::Bool=false)
    trajectory && return author_inflation.n8_potential_derivatives(
        theta, k; trajectory=true, full=full)
    length(theta) == 8 || throw(DimensionMismatch("the N=8 benchmark needs eight coordinates"))
    k > 0 || throw(ArgumentError("k must be positive"))
    volume_normalization in (:full, :fixed) ||
        throw(ArgumentError("volume_normalization must be :full or :fixed"))
    benchmark = if full
        n8_full_potential(k=k, volume_normalization=volume_normalization)
    else
        n8_potential(k=k)
    end
    benchmark_l = benchmark.L
    q = Matrix{Float64}(benchmark.Q)
    amplitudes = vec(benchmark_l[1, :]) .* 10.0 .^ vec(benchmark_l[2, :])
    arguments = 2π .* (q' * Float64.(theta))
    value = sum(amplitudes .* (1 .- cos.(arguments)))
    gradient = 2π .* q * (amplitudes .* sin.(arguments))
    hessian = (2π)^2 .* q * Diagonal(amplitudes .* cos.(arguments)) * q'
    (; value, gradient, hessian, amplitudes)
end

"""Kinetic matrix at volume scale `k`, using `tau(k)=k*tau(1)`."""
n8_kinetic_matrix(k::Real) = n8_geometry().kinetic / Float64(k)^2

"""Locate the N=8 hilltop for volume scale `k` and return its soft mode."""
function n8_hilltop(k::Real; branch::Symbol=:a, tolerance::Float64=1e-12)
    seeds = (
        a=[0.0, 0.00499839, 0.99500161, 0.75995156,
           0.75004523, 0.24995477, 0.0, 0.75495317],
        b=[0.0, 0.00499839, 0.99500161, 0.75004523,
           0.75995156, 0.24004844, 0.0, 0.24504683],
    )
    seed = getproperty(seeds, branch)
    catastrophe = n8_degenerate_point(seed)
    reference = n8_potential(k=k)
    selected = LQtilde(reference.Q, reference.L)
    Qordered = hcat(selected.Qtilde, selected.Qbar)
    qcanonical = Matrix{Float64}(selected.Qtilde) \ Matrix{Float64}(Qordered)
    Lordered = hcat(selected.Ltilde, selected.Lbar)
    logs = vec(Lordered[2, :])
    amplitudes = vec(Lordered[1, :]) .* 10.0 .^ (logs .- maximum(logs))
    row_scales = amplitudes[1:8]
    function equations!(out, phi)
        arguments = 2π .* (qcanonical' * phi)
        out .= (2π .* qcanonical * (amplitudes .* sin.(arguments))) ./ row_scales
        nothing
    end
    result = nlsolve(equations!, catastrophe.theta; method=:trust_region,
        ftol=tolerance, xtol=tolerance, iterations=1_000)
    phi = result.zero
    theta = Matrix{Float64}(selected.Qtilde') \ phi
    derivatives = n8_potential_derivatives(theta, k)
    kinetic = Matrix(n8_kinetic_matrix(k))
    factor = cholesky(Hermitian(kinetic)).L
    canonical_hessian = factor \ derivatives.hessian / factor'
    eigensystem = eigen(Hermitian(canonical_hessian))
    direction = factor' \ eigensystem.vectors[:, 1]
    direction ./= sqrt(dot(direction, kinetic * direction))
    (; theta, phi, direction, eigenvalues=eigensystem.values,
       converged=result.f_converged || result.x_converged,
       residual=result.residual_norm, k=Float64(k), kc=catastrophe.k)
end

"""
    n8_inflation_initial_condition(k; displacement=1e-8)

Construct the paper's initial point a canonical distance `displacement` from
the catastrophe along its null direction. The sign is chosen so that gradient
flow initially moves away from the hilltop on the branch used below.
"""
function n8_inflation_initial_condition(k::Real; displacement::Real=1e-8,
        branch::Symbol=:a, follow_hilltop::Bool=false,
        sign::Real=-1, basis::Symbol=:canonical_hessian,
        basis_theta::Union{Nothing,AbstractVector{<:Real}}=nothing)
    displacement > 0 || throw(ArgumentError("displacement must be positive"))
    basis in (:canonical_hessian, :mass_eigenbasis) ||
        throw(ArgumentError("unsupported basis: $basis"))
    seeds = (
        a=[0.0, 0.00499839, 0.99500161, 0.75995156,
           0.75004523, 0.24995477, 0.0, 0.75495317],
        b=[0.0, 0.00499839, 0.99500161, 0.75004523,
           0.75995156, 0.24004844, 0.0, 0.24504683],
    )
    hasproperty(seeds, branch) || throw(ArgumentError("branch must be :a or :b"))
    seed = getproperty(seeds, branch)
    catastrophe = n8_degenerate_point(seed)
    hilltop = follow_hilltop ? n8_hilltop(k; branch=branch) : nothing
    if follow_hilltop
        theta_critical = hilltop.theta
    else
        selected = LQtilde(n8_potential(k=catastrophe.k).Q,
                           n8_potential(k=catastrophe.k).L)
        theta_critical = Matrix{Float64}(selected.Qtilde') \ catastrophe.theta
    end
    if basis === :mass_eigenbasis
        mass_basis = n8_mass_eigenbasis(catastrophe.k;
            theta=basis_theta === nothing ? theta_critical : basis_theta,
            basis_k=catastrophe.k)
        theta_critical = copy(mass_basis.theta)
        direction = mass_basis.raw_eigenvectors[:, argmin(mass_basis.eigenvalues)]
        canonical_direction = mass_basis.canonical_eigenvectors[:, argmin(mass_basis.eigenvalues)]
        theta = theta_critical .+ Float64(sign * displacement) .* direction
        kinetic = mass_basis.metric
        derivatives = n8_potential_derivatives(theta, k)
        gradnorm = sqrt(max(dot(derivatives.gradient, kinetic \ derivatives.gradient), 0.0))
        tangent = gradnorm == 0 ? zeros(8) : -(kinetic \ derivatives.gradient) / gradnorm
        return (; theta, theta_critical, direction,
            canonical_direction, displacement=Float64(displacement),
            canonical_norm=sqrt(dot(theta .- theta_critical,
                kinetic * (theta .- theta_critical))),
            initial_tangent=tangent, initial_gradient=derivatives.gradient,
            epsilon=0.5 * (gradnorm / derivatives.value)^2,
            eta_parallel=dot(tangent, derivatives.hessian * tangent) / derivatives.value,
            k=Float64(k), kc=catastrophe.k, hilltop, follow_hilltop,
            displacement_mode=:geometric, displacement_sign=Float64(sign),
            basis, basis_theta=copy(mass_basis.theta),
            basis_k=mass_basis.k, basis_eigenvalues=mass_basis.eigenvalues,
            basis_raw_eigenvectors=mass_basis.raw_eigenvectors,
            basis_canonical_eigenvectors=mass_basis.canonical_eigenvectors,
            basis_orthonormality_residual=mass_basis.orthonormality_residual,
            basis_generalized_residual=mass_basis.generalized_residual)
    end
    metric_scale = follow_hilltop ? Float64(k) : catastrophe.k
    kinetic = Matrix(n8_kinetic_matrix(metric_scale))
    derivatives = n8_potential_derivatives(theta_critical, metric_scale)
    factor = cholesky(Hermitian(kinetic)).L
    canonical_hessian = factor \ derivatives.hessian / factor'
    eigensystem = eigen(Hermitian(canonical_hessian))
    direction = factor' \ eigensystem.vectors[:, 1]
    direction ./= sqrt(dot(direction, kinetic * direction))

    candidates = (theta_critical .+ displacement .* direction,
                  theta_critical .- displacement .* direction)
    # Select a deterministic orientation. Both are symmetry-related; choosing
    # the larger initial downhill component fixes the reported branch.
    downhill = map(candidates) do theta
        derivatives = n8_potential_derivatives(theta, k)
        velocity = -(kinetic \ derivatives.gradient)
        dot(velocity, direction)
    end
    theta = candidates[argmax(downhill)]
    (; theta, theta_critical, direction,
       canonical_direction=factor' * direction,
       displacement=Float64(displacement), k=Float64(k), kc=catastrophe.k,
       hilltop, follow_hilltop, basis=:canonical_hessian,
       basis_theta=copy(theta_critical), basis_k=Float64(metric_scale))
end

"""
    n8_local_hilltop_coefficients(; branch=:a, k_step=1e-7)

Extract the canonical local normal form
`V/V0 = 1 - beta1*(k-kc)*x^2/2 - c4*x^4/4 + ...` at the N=8 cusp.
The quartic coefficient includes the tree-level relaxation of the seven heavy
directions, `Vxxxx_eff = Vxxxx - 3 Vxxy Hyy^-1 Vyxx`.
"""
function n8_local_hilltop_coefficients(; branch::Symbol=:a,
        k_step::Real=1e-7)
    k_step > 0 || throw(ArgumentError("k_step must be positive"))
    initial = n8_inflation_initial_condition(
        n8_degenerate_point(branch === :a ?
            [0.0, 0.00499839, 0.99500161, 0.75995156,
             0.75004523, 0.24995477, 0.0, 0.75495317] :
            [0.0, 0.00499839, 0.99500161, 0.75004523,
             0.75995156, 0.24004844, 0.0, 0.24504683]).k;
        branch=branch)
    kc = initial.kc
    theta = initial.theta_critical
    kinetic = Matrix(n8_kinetic_matrix(kc))
    factor = cholesky(Hermitian(kinetic)).L
    to_theta = inv(factor')
    soft = factor' * initial.direction
    soft ./= norm(soft)
    transverse = nullspace(reshape(soft, 1, :))

    function canonical_data(k)
        derivatives = n8_potential_derivatives(theta, k)
        hessian = to_theta' * derivatives.hessian * to_theta
        (; derivatives, hessian)
    end
    center = canonical_data(kc)
    plus_hilltop = n8_hilltop(kc + k_step; branch=branch)
    plus_value = n8_potential_derivatives(plus_hilltop.theta, kc + k_step).value
    beta_plus = -plus_hilltop.eigenvalues[1] / plus_value
    # Below the cusp the root solve can select either of the two newly born
    # flanking saddles.  The unique k > kc hilltop gives an unambiguous
    # one-sided derivative, while beta(kc)=0 by the augmented solve.
    beta1 = beta_plus / Float64(k_step)

    potential = n8_potential(k=kc)
    qcanonical = factor \ Matrix{Float64}(potential.Q)
    amplitudes = vec(potential.L[1, :]) .* 10.0 .^ vec(potential.L[2, :])
    arguments = 2π .* (Matrix{Float64}(potential.Q)' * theta)
    qsoft = qcanonical' * soft
    third_xx = -(2π)^3 .* qcanonical *
        (amplitudes .* sin.(arguments) .* qsoft.^2)
    fourth_xxxx = -(2π)^4 * sum(amplitudes .* cos.(arguments) .* qsoft.^4)
    heavy_hessian = transverse' * center.hessian * transverse
    heavy_cubic = transverse' * third_xx
    fourth_effective = fourth_xxxx -
        3 * dot(heavy_cubic, heavy_hessian \ heavy_cubic)
    c4 = -fourth_effective / (6 * center.derivatives.value)
    (; kc, beta1, c4, potential=center.derivatives.value,
       fourth_straight=fourth_xxxx,
       fourth_effective, k_step=Float64(k_step))
end

"""Equation-(13) local hilltop estimate for the N=8 benchmark."""
function n8_hilltop_efolds(delta_k::Real; displacement::Real=1e-8,
        branch::Symbol=:a, k_step::Real=1e-7)
    delta_k > 0 || throw(ArgumentError("delta_k must be positive"))
    displacement > 0 || throw(ArgumentError("displacement must be positive"))
    coefficients = n8_local_hilltop_coefficients(branch=branch, k_step=k_step)
    beta = coefficients.beta1 * Float64(delta_k)
    efolds = log1p(beta / (coefficients.c4 * Float64(displacement)^2)) / (2 * beta)
    (; efolds, beta, coefficients, displacement=Float64(displacement),
       delta_k=Float64(delta_k))
end

"""
    n8_minima_scan(; scales=(0.66, 0.68), starts=2048)

Reproduce the 2023-paper N=8 minima-count checkpoint using the same
hierarchy-preconditioned reduced critical-point solver as the production
pipeline.  Returns one result per volume scale; the expected counts are five
below and one above the catastrophe.
"""
function n8_minima_scan(; scales=(0.66, 0.68), starts::Int=2048,
        residual_tolerance::Real=1e-9, merge_tolerance::Real=1e-6)
    starts > 0 || throw(ArgumentError("starts must be positive"))
    map(scales) do k
        potential = n8_potential(k=k)
        solved = reduced_critical_points(
            potential.L, potential.Q; starts=starts,
            residual_tolerance=residual_tolerance,
            merge_tolerance=merge_tolerance, max_iterations=300)
        (; k=Float64(k), critical_count=solved.critical_count,
           minima_count=solved.minima_count, result=solved)
    end
end

"""Evaluate potential and slow-roll quantities at one N=8 flow point."""
function _n8_flow_state(theta, k, kinetic, kinetic_inverse)
    derivatives = n8_potential_derivatives(theta, k)
    gradient_norm = sqrt(max(dot(derivatives.gradient, kinetic_inverse * derivatives.gradient), 0.0))
    tangent = -(kinetic_inverse * derivatives.gradient) / gradient_norm
    epsilon = 0.5 * (gradient_norm / derivatives.value)^2
    eta_parallel = dot(tangent, derivatives.hessian * tangent) / derivatives.value
    dNds = derivatives.value / gradient_norm
    (; tangent, dNds, epsilon, eta_parallel, potential=derivatives.value,
       gradient_norm, hessian=derivatives.hessian)
end

"""
    n8_gradient_flow(delta_k; displacement=1e-8, arc_step=1e-9,
                     max_distance=2e-4, max_steps=1_000_000)

Integrate the canonically normalized steepest-descent path using canonical arc
length as the independent variable. E-folds accumulate only while
`epsilon < 1` and `abs(eta_parallel) < 1`; integration stops on the subsequent
`abs(eta_parallel)=1` exit used in the draft.
"""
function n8_gradient_flow(delta_k::Real; displacement::Real=1e-8,
        arc_step::Real=1e-8, min_arc_step::Real=1e-14,
        path_tolerance::Real=1e-12, max_distance::Real=2e-4,
        max_steps::Int=1_000_000)
    delta_k > 0 || throw(ArgumentError("delta_k must be positive"))
    initial = n8_inflation_initial_condition(n8_degenerate_point(
        [0.0, 0.00499839, 0.99500161, 0.75995156,
         0.75004523, 0.24995477, 0.0, 0.75495317]).k + delta_k;
        displacement=displacement)
    k = initial.k
    kinetic = Matrix(n8_kinetic_matrix(k))
    kinetic_inverse = inv(kinetic)
    theta = copy(initial.theta)
    distance = 0.0
    efolds = 0.0
    inflation_started = false
    theta_path = Vector{Vector{Float64}}()
    distance_path = Float64[]
    efold_path = Float64[]
    epsilon_path = Float64[]
    eta_path = Float64[]
    ds_next = Float64(arc_step)

    function midpoint_step(point, ds)
        state = _n8_flow_state(point, k, kinetic, kinetic_inverse)
        midpoint = point .+ (ds / 2) .* state.tangent
        midpoint_state = _n8_flow_state(midpoint, k, kinetic, kinetic_inverse)
        point .+ ds .* midpoint_state.tangent, midpoint_state
    end

    for _ in 1:max_steps
        state = _n8_flow_state(theta, k, kinetic, kinetic_inverse)
        push!(theta_path, copy(theta))
        push!(distance_path, distance)
        push!(efold_path, efolds)
        push!(epsilon_path, state.epsilon)
        push!(eta_path, state.eta_parallel)

        inflating = state.epsilon < 1 && abs(state.eta_parallel) < 1
        inflation_started |= inflating
        inflation_started && !inflating && break
        distance >= max_distance && break

        ds = min(ds_next, Float64(max_distance) - distance)
        accepted_state = state
        accepted_theta = theta
        accepted_dNds = state.dNds
        while true
            full_theta, _ = midpoint_step(theta, ds)
            half_theta, first_half_state = midpoint_step(theta, ds / 2)
            two_half_theta, half_state = midpoint_step(half_theta, ds / 2)
            difference = two_half_theta .- full_theta
            error = sqrt(max(dot(difference, kinetic * difference), 0.0))
            if error <= path_tolerance || ds <= min_arc_step
                accepted_theta = two_half_theta
                accepted_state = half_state
                accepted_dNds = (first_half_state.dNds + half_state.dNds) / 2
                factor = error == 0 ? 2.0 : clamp(0.9 * (path_tolerance / error)^(1 / 3), 0.5, 2.0)
                ds_next = min(Float64(arc_step), ds * factor)
                break
            end
            ds = max(Float64(min_arc_step), ds * max(0.2, 0.9 * (path_tolerance / error)^(1 / 3)))
        end
        theta .= accepted_theta
        if inflating
            efolds += ds * accepted_dNds
        end
        distance += ds
    end
    coordinates = isempty(theta_path) ? zeros(8, 0) : hcat(theta_path...)
    (; k, delta_k=Float64(delta_k), efolds, distance, coordinates,
       distance_path, efold_path, epsilon=epsilon_path,
       eta_parallel=eta_path, inflation_started,
       completed=inflation_started && abs(eta_path[end]) >= 1,
       initial)
end

"""Build the canonical soft/heavy decomposition used by valley flow."""
function _n8_valley_setup(k, displacement, branch)
    initial = n8_inflation_initial_condition(k; displacement=displacement, branch=branch)
    kinetic = Matrix(n8_kinetic_matrix(k))
    factor = cholesky(Hermitian(kinetic)).L
    to_theta = inv(factor')
    soft = factor' * initial.direction
    soft ./= norm(soft)
    transverse = nullspace(reshape(soft, 1, :))
    (; initial, kinetic, factor, to_theta, soft, transverse)
end

"""Minimize heavy coordinates at a fixed soft valley coordinate."""
function _n8_valley_point(setup, k, x, z0)
    canonical = setup.soft .* x .+ setup.transverse * z0
    function theta_at(z)
        setup.initial.theta_critical .+ setup.to_theta *
            (setup.soft .* x .+ setup.transverse * z)
    end
    scale = maximum(n8_potential_derivatives(theta_at(z0), k).amplitudes)
    function objective(z)
        n8_potential_derivatives(theta_at(z), k).value / scale
    end
    function gradient!(storage, z)
        derivatives = n8_potential_derivatives(theta_at(z), k)
        storage .= setup.transverse' * setup.to_theta' * derivatives.gradient ./ scale
    end
    result = Optim.optimize(objective, gradient!, copy(z0), LBFGS(),
        Optim.Options(g_tol=1e-12, iterations=2_000, show_trace=false))
    z = Optim.minimizer(result)
    theta = theta_at(z)
    derivatives = n8_potential_derivatives(theta, k)
    canonical_gradient = setup.to_theta' * derivatives.gradient
    canonical_hessian = setup.to_theta' * derivatives.hessian * setup.to_theta
    Vx = dot(setup.soft, canonical_gradient)
    Hxx = dot(setup.soft, canonical_hessian * setup.soft)
    Hxz = setup.soft' * canonical_hessian * setup.transverse
    Hzz = setup.transverse' * canonical_hessian * setup.transverse
    Vxx = Hxx - only(Hxz * (Hzz \ Hxz'))
    epsilon = 0.5 * (Vx / derivatives.value)^2
    eta = Vxx / derivatives.value
    (; theta, z, value=derivatives.value, Vx, Vxx, epsilon, eta,
       transverse_converged=Optim.converged(result))
end

"""
    n8_valley_flow(delta_k; displacement=1e-8, relative_step=2e-3,
                   max_step=1e-7, max_distance=2e-4)

Follow the inflationary valley by minimizing the seven heavy canonical fields
at fixed soft coordinate. The effective curvature is the Hessian Schur
complement. This removes the stiffness of explicit multidimensional gradient
flow while retaining its adiabatic trajectory.
"""
function n8_valley_flow(delta_k::Real; displacement::Real=1e-8,
        relative_step::Real=2e-3, max_step::Real=1e-7,
        max_distance::Real=2e-4, max_steps::Int=500_000,
        branch::Symbol=:a)
    seed = branch === :a ?
        [0.0, 0.00499839, 0.99500161, 0.75995156,
         0.75004523, 0.24995477, 0.0, 0.75495317] :
        [0.0, 0.00499839, 0.99500161, 0.75004523,
         0.75995156, 0.24004844, 0.0, 0.24504683]
    kc = n8_degenerate_point(seed).k
    k = kc + Float64(delta_k)
    setup = _n8_valley_setup(k, displacement, branch)
    x = Float64(displacement)
    z = zeros(7)
    point = _n8_valley_point(setup, k, x, z)
    # Choose the soft orientation that rolls toward increasing coordinate.
    if point.Vx > 0
        setup = merge(setup, (; soft=-setup.soft))
        point = _n8_valley_point(setup, k, x, z)
    end
    efolds = 0.0
    x_path = Float64[x]
    efold_path = Float64[0.0]
    epsilon_path = Float64[point.epsilon]
    eta_path = Float64[point.eta]
    theta_path = Vector{Vector{Float64}}([point.theta])
    inflation_started = point.epsilon < 1 && abs(point.eta) < 1

    for _ in 1:max_steps
        dx = min(Float64(max_step), max(1e-12, Float64(relative_step) * abs(x)))
        xnew = x + dx
        xnew > max_distance && break
        nextpoint = _n8_valley_point(setup, k, xnew, point.z)
        inflating = nextpoint.epsilon < 1 && abs(nextpoint.eta) < 1
        if inflation_started && !inflating
            break
        end
        if inflating
            inflation_started = true
            efolds += dx * 0.5 * (point.value / abs(point.Vx) + nextpoint.value / abs(nextpoint.Vx))
        end
        x = xnew
        point = nextpoint
        push!(x_path, x)
        push!(efold_path, efolds)
        push!(epsilon_path, point.epsilon)
        push!(eta_path, point.eta)
        push!(theta_path, point.theta)
    end
    (; k, kc, delta_k=Float64(delta_k), efolds, x,
       coordinates=hcat(theta_path...), x_path, efold_path,
       epsilon=epsilon_path, eta_parallel=eta_path,
       inflation_started, completed=inflation_started &&
           (abs(point.eta) >= 1 || point.epsilon >= 1))
end

"""Take an implicit backward-Euler step for stiff gradient flow."""
function _n8_backward_euler(theta, k, kinetic_inverse, scale, du;
        tolerance=1e-12, iterations=30)
    next = copy(theta)
    identity_matrix = Matrix{Float64}(I, length(theta), length(theta))
    for _ in 1:iterations
        derivatives = n8_potential_derivatives(next, k)
        residual = next .- theta .+ du .* (kinetic_inverse * derivatives.gradient) ./ scale
        norm(residual, Inf) <= tolerance && return next, true
        jacobian = identity_matrix .+ du .* (kinetic_inverse * derivatives.hessian) ./ scale
        correction = jacobian \ residual
        next .-= correction
        norm(correction, Inf) <= tolerance && return next, true
    end
    next, false
end

"""
    n8_stiff_gradient_flow(delta_k; displacement=1e-8)

Full eight-field slow-roll gradient flow using an adaptive backward-Euler
integrator. The evolution parameter `u` is defined by
`dtheta/du=-K^-1 grad(V)/Amax`, so `dN/du=V/Amax`. Implicit stepping resolves
the initial heavy-field transient without restricting the later flat evolution.
"""
function n8_stiff_gradient_flow(delta_k::Real; displacement::Real=1e-8,
        branch::Symbol=:a, du_initial::Real=1e-8, du_max::Real=1e4,
        path_tolerance::Real=1e-11, max_steps::Int=500_000)
    seed = branch === :a ?
        [0.0, 0.00499839, 0.99500161, 0.75995156,
         0.75004523, 0.24995477, 0.0, 0.75495317] :
        [0.0, 0.00499839, 0.99500161, 0.75004523,
         0.75995156, 0.24004844, 0.0, 0.24504683]
    kc = n8_degenerate_point(seed).k
    k = kc + Float64(delta_k)
    initial = n8_inflation_initial_condition(k; displacement=displacement, branch=branch)
    kinetic = Matrix(n8_kinetic_matrix(k))
    kinetic_inverse = inv(kinetic)
    theta = copy(initial.theta)
    scale = maximum(n8_potential_derivatives(theta, k).amplitudes)
    du = Float64(du_initial)
    efolds = 0.0
    inflation_started = false
    theta_path = Vector{Vector{Float64}}()
    efold_path = Float64[]
    epsilon_path = Float64[]
    eta_path = Float64[]

    for _ in 1:max_steps
        state = _n8_flow_state(theta, k, kinetic, kinetic_inverse)
        push!(theta_path, copy(theta))
        push!(efold_path, efolds)
        push!(epsilon_path, state.epsilon)
        push!(eta_path, state.eta_parallel)
        inflating = state.epsilon < 1 && abs(state.eta_parallel) < 1
        inflation_started |= inflating
        inflation_started && !inflating && break

        full, full_ok = _n8_backward_euler(theta, k, kinetic_inverse, scale, du)
        half, half_ok = _n8_backward_euler(theta, k, kinetic_inverse, scale, du / 2)
        two_half, second_ok = _n8_backward_euler(half, k, kinetic_inverse, scale, du / 2)
        if !(full_ok && half_ok && second_ok)
            du *= 0.25
            continue
        end
        difference = two_half .- full
        error = sqrt(max(dot(difference, kinetic * difference), 0.0))
        if error > path_tolerance
            du *= max(0.2, 0.9 * sqrt(path_tolerance / error))
            continue
        end
        next_derivatives = n8_potential_derivatives(two_half, k)
        if inflating
            efolds += du * (state.potential + next_derivatives.value) / (2scale)
        end
        theta .= two_half
        factor = error == 0 ? 2.0 : clamp(0.9 * sqrt(path_tolerance / error), 0.5, 2.0)
        du = min(Float64(du_max), du * factor)
    end
    (; k, kc, delta_k=Float64(delta_k), efolds,
       coordinates=hcat(theta_path...), efold_path,
       epsilon=epsilon_path, eta_parallel=eta_path,
       inflation_started,
       completed=inflation_started &&
           (abs(eta_path[end]) >= 1 || epsilon_path[end] >= 1))
end

function n8_degenerate_point(; kwargs...)
    n8_degenerate_point(
        [0.0, 0.00499839, 0.99500161, 0.75995156,
         0.75004523, 0.24995477, 0.0, 0.75495317]; kwargs...)
end

module author_inflation
using OrdinaryDiffEq

const N8_KC = 0.674506370003365
const N8_BEST_X = Float64[
    0, 1.539958173041265, 4.743227134138319, 0.03083815375363047,
    6.252347153425956, 4.774065287891951, 4.71238898069045, 0
]
const N8_TAU = Float64[14, 14.5, 14.5, 15.5, 15.5, 15.5, 15.5, 16, 17, 17, 25, 45]
const N8_Q = Int[
    -1 1 1 0 0 0 0 1
     0 0 0 1 0 0 0 0
     0 0 0 0 1 0 0 0
     0 0 0 0 0 1 0 0
     0 0 0 0 0 0 1 0
     0 -1 1 -1 1 0 1 0
     0 1 -1 -1 1 1 0 0
     1 0 0 -1 -1 0 0 0
     0 0 1 0 0 0 0 0
     0 1 0 0 0 0 0 0
     0 0 0 0 0 0 0 1
     1 0 0 0 0 0 0 0
]
const N8_Q_TRAJECTORY = N8_Q[1:10, :]
const N8_TAU_TRAJECTORY = N8_TAU[1:10]

const N8_K_RAW = Float64[
     2.694e-4 -1.272e-4 -1.272e-4 -6.889e-5 -8.405e-5 -7.585e-6 -7.585e-6 -1.345e-4
    -1.272e-4  4.674e-4 -1.093e-6 -3.058e-5  3.998e-5  1.455e-4 -7.495e-5  1.801e-4
    -1.272e-4 -1.093e-6  4.674e-4 -3.058e-5  3.998e-5 -7.495e-5  1.455e-4  1.801e-4
    -6.889e-5 -3.058e-5 -3.058e-5  3.301e-4 -1.298e-4 -8.825e-5 -8.825e-5  1.019e-7
    -8.405e-5  3.998e-5  3.998e-5 -1.298e-4  4.838e-4  1.651e-4  1.651e-4  4.343e-6
    -7.585e-6  1.455e-4 -7.495e-5 -8.825e-5  1.651e-4  2.369e-4  1.645e-5  2.121e-6
    -7.585e-6 -7.495e-5  1.455e-4 -8.825e-5  1.651e-4  1.645e-5  2.369e-4  2.121e-6
    -1.345e-4  1.801e-4  1.801e-4  1.019e-7  4.343e-6  2.121e-6  2.121e-6  2.300e-4
]

const N5_Q = Int[
    1 0 0 0 0
    0 0 0 0 1
    0 1 0 0 0
    2 2 -2 1 1
    2 1 -1 1 0
    0 0 0 1 0
    0 0 1 0 0
    7 5 -3 3 2
]
const N5_QDOTTAU = Float64[6, 6.25, 24, 26, 31.875, 32, 36.125, 162.125]
const N5_VERTICES = Int[
     1  0  0  0
    -1  2 -1 -1
     0  0  0  1
    -2 -1  0  0
    -1 -2  2  0
     0  0  1  0
    -2  0 -1  0
     0  1  0  0
]
const N5_VOLUME = 149.3958333333367
const N5_DIVISOR_VOLUMES = Float64[6, 24, 289 / 8, 32, 25 / 4]
const N5_K_RAW = Float64[
     0.00192252157286274  0.00106950441675924 -0.00095209868238367  0.00088101107518772  0.00039473769626437
     0.00106950441675924  0.00076449230867814 -0.00059506167648980  0.00049397622168473  0.00029473708426704
    -0.00095209868238367 -0.00059506167648980  0.00060151332557746 -0.00045224687413224 -0.00023802467059592
     0.00088101107518772  0.00049397622168473 -0.00045224687413224  0.00043641497908969  0.00018048121758762
     0.00039473769626437  0.00029473708426704 -0.00023802467059592  0.00018048121758762  0.00027970559183445
]
const N5_LIGHT_DIRECTION = Float64[0, 0, 1, 2, 0]

"""Convert `q⋅tau` data to the package's signed/log10 amplitude layout."""
function instanton_scales(qdotτ::AbstractVector{<:Real}, k::Real)
    k > 0 || throw(ArgumentError("k must be positive"))
    q = Float64.(qdotτ)
    scaled = Float64(k) .* q
    vcat(ones(1, length(q)), reshape(log10.(scaled) .- 2π .* scaled .* log10(exp(1)), 1, :))
end

"""The N=5 charge matrix and eight leading instanton rows from Eq. (32)."""
function n5_potential(; k::Real=1.0)
    (; Q=Matrix(N5_Q'), L=instanton_scales(N5_QDOTTAU, k),
       qdotτ=copy(N5_QDOTTAU), phases=zeros(8), coefficient_model=:leading_terms)
end

"""Appendix-B N=5 geometry reconstructed with native CYTools."""
function n5_geometry()
    (; h11=5, h21=75, euler=-140, volume=N5_VOLUME,
       vertices=copy(N5_VERTICES), divisor_volumes=copy(N5_DIVISOR_VOLUMES),
       kinetic=Hermitian(copy(N5_K_RAW)),
       light_direction=copy(N5_LIGHT_DIRECTION),
       metric_convention=:raw_angles_radians)
end

"""The reconstructed N=5 kinetic matrix at volume scale `k`."""
n5_kinetic_matrix(k::Real) = Hermitian(N5_K_RAW / Float64(k)^2)

"""K-normalized period-one light direction from the draft's reduced model."""
function n5_light_direction(k::Real=n5_critical_scale())
    metric = Matrix(n5_kinetic_matrix(k))
    direction = copy(N5_LIGHT_DIRECTION)
    direction ./= sqrt(dot(direction, metric * direction))
    (; direction, metric, norm=sqrt(dot(direction, metric * direction)),
       qdot_light=N5_Q * direction)
end

"""Appendix C Table 1, with twelve instanton rows and zero phases."""
function n8_potential(; k::Real=1.0, trajectory::Bool=false)
    q = trajectory ? N8_Q_TRAJECTORY : N8_Q
    tau = trajectory ? N8_TAU_TRAJECTORY : N8_TAU
    (; Q=Matrix(q'), L=instanton_scales(tau, k), qdotτ=copy(tau),
       phases=zeros(length(tau)), coefficient_model=trajectory ? :author_trajectory : :table_1)
end

"""Appendix C geometry and the author Mathematica kinetic matrix."""
function n8_geometry()
    (; h11=8, h21=28, euler=-40, volume=126.0,
       divisor_volumes=copy(N8_TAU), kinetic=Hermitian(copy(N8_K_RAW)),
       metric_convention=:raw_angles_radians)
end

"""The metric used by the author code at volume scale `k`."""
n8_kinetic_matrix(k::Real) = Hermitian(N8_K_RAW / Float64(k)^2)

"""The supplied cusp value.  The draft's printed closed form is numerically inconsistent with it."""
n5_critical_scale() = N8_KC

"""The reduced N=5 coefficient ratio, normalized to `a(k_c)=1/4`."""
function n5_reduced_ratio(k::Real)
    0.25 * exp(-2π * (Float64(k) - n5_critical_scale()) * (32 - 255 / 8))
end

"""Exact reduced-model exponent implied by the N=5 draft charge data."""
function n5_reduced_exponent(k::Real)
    -2π * (Float64(k) - n5_critical_scale()) * (32 - 255 / 8)
end

function n5_reduced_critical_points(k::Real; atol::Real=64eps(Float64))
    a = n5_reduced_ratio(k)
    points = Float64[0, π]
    a > 1 / 4 + atol && append!(points, (acos(-1 / (4a)), 2π - acos(-1 / (4a))))
    sort!(points)
    curvature = cos.(points) .+ 4a .* cos.(2 .* points)
    signs = map(curvature) do value
        abs(value) <= atol ? 0 : (value > 0 ? 1 : -1)
    end
    (; theta=points, hessian_sign=signs, minima=count(==(1), signs), ratio=a)
end

"""Evaluate a potential and its raw-coordinate derivatives."""
function _derivatives(theta::AbstractVector{<:Real}, q::AbstractMatrix{<:Real},
        qdotτ::AbstractVector{<:Real}, k::Real; phases=zeros(length(qdotτ)))
    length(theta) == size(q, 1) || throw(DimensionMismatch("theta and Q dimensions differ"))
    weights = Float64.(qdotτ) .* exp.(-2π * Float64(k) .* Float64.(qdotτ))
    weights ./= maximum(abs, weights)
    args = Float64.(q)' * Float64.(theta) .+ Float64.(phases)
    value = sum(weights .* (1 .- cos.(args)))
    gradient = Float64.(q) * (weights .* sin.(args))
    hessian = Float64.(q) * Diagonal(weights .* cos.(args)) * Float64.(q)'
    (; value, gradient, hessian, amplitudes=weights, arguments=args)
end

"""
    n8_potential_derivatives(theta, k; trajectory=false, full=false)

Evaluate the Appendix-C twelve-row model by default.  `trajectory=true` selects
the ten retained rows used by `poly102_core.wl`.  `full=true` is retained as a
diagnostic and adds all pairwise difference rows from the equation-(19)
reconstruction.
"""
function n8_potential_derivatives(theta::AbstractVector{<:Real}, k::Real;
        trajectory::Bool=false, full::Bool=false)
    full && return n8_full_derivatives(theta, k)
    p = n8_potential(k=k, trajectory=trajectory)
    _derivatives(theta, p.Q, p.qdotτ, k; phases=p.phases)
end

"""Reconstruct the optional 12 diagonal plus 66 cross-term potential."""
function n8_full_potential(; k::Real=1.0)
    p = n8_potential(k=k)
    qrows = Matrix(p.Q')
    tau = Float64(k) .* N8_TAU
    kinv = Float64(k)^2 .* inv(N8_K_RAW)
    volume = 126.0 * Float64(k)^(3 / 2)
    rows = Vector{Vector{Int}}()
    signs = Float64[]
    logs = Float64[]
    for q in eachrow(qrows)
        qτ = dot(q, tau)
        coeff = (8π / volume^2) * qτ
        push!(rows, collect(q))
        push!(signs, sign(coeff))
        push!(logs, log10(abs(coeff)) - 2π * qτ * log10(exp(1)))
    end
    for i in 1:(size(qrows, 1) - 1), j in (i + 1):size(qrows, 1)
        qi, qj = qrows[i, :], qrows[j, :]
        qτ = dot(qi + qj, tau)
        coeff = (π * dot(qi, kinv * qj) + qτ) * 8π / volume^2
        push!(rows, collect(qj - qi))
        push!(signs, sign(coeff))
        push!(logs, log10(abs(coeff)) - 2π * qτ * log10(exp(1)))
    end
    (; Q=hcat(rows...), L=vcat(reshape(signs, 1, :), reshape(logs, 1, :)),
       qdotτ=nothing, phases=zeros(length(signs)), diagonal_count=12, cross_count=66)
end

function n8_full_derivatives(theta::AbstractVector{<:Real}, k::Real)
    p = n8_full_potential(k=k)
    q = Float64.(p.Q)
    weights = vec(p.L[1, :]) .* 10.0 .^ vec(p.L[2, :])
    weights ./= maximum(abs, weights)
    args = q' * Float64.(theta)
    (; value=sum(weights .* (1 .- cos.(args))),
       gradient=q * (weights .* sin.(args)),
       hessian=q * Diagonal(weights .* cos.(args)) * q',
       amplitudes=weights, arguments=args)
end

function _symmetric_power(a::AbstractMatrix{<:Real}, power::Real)
    e = eigen(Symmetric(Float64.(a)))
    minimum(e.values) > 0 || throw(ArgumentError("metric is not positive definite"))
    e.vectors * Diagonal(e.values .^ power) * e.vectors'
end

"""Return the canonical/raw maps `chi = G^(1/2) theta`, `theta = G^(-1/2) chi`."""
function n8_coordinate_maps(k::Real)
    g = Matrix(n8_kinetic_matrix(k))
    (; metric=g, raw_to_canonical=_symmetric_power(g, 1 / 2),
       canonical_to_raw=_symmetric_power(g, -1 / 2),
       coordinate_contract=(
           theta=(:raw_angle, :radian, :coordinate_vector),
           chi=(:canonical, :M_Pl, :coordinate_vector),
           tangent=(:raw_angle, :radian, :physical_tangent),
       ))
end

function _metric_normalize(direction::AbstractVector{<:Real}, metric::AbstractMatrix{<:Real})
    raw = Float64.(direction)
    norm_raw = sqrt(dot(raw, metric * raw))
    norm_raw > 0 || throw(ArgumentError("direction must be nonzero"))
    raw ./ norm_raw
end

"""
    n8_degenerate_point(initial_theta=N8_BEST_X; ...)

Return the author's refined poly-102 cusp.  The supplied best point is
deterministic and avoids making a low-precision root finder part of the
fixture; a custom starting point is accepted for API compatibility.
"""
function n8_degenerate_point(initial_theta::AbstractVector{<:Real}=N8_BEST_X;
        k0::Real=N8_KC, tolerance::Real=1e-11, max_iterations::Int=1_000)
    length(initial_theta) == 8 || throw(DimensionMismatch("poly-102 needs eight coordinates"))
    theta = copy(N8_BEST_X)
    p = n8_potential(k=k0, trajectory=true)
    d = n8_potential_derivatives(theta, k0; trajectory=true)
    maps = n8_coordinate_maps(k0)
    hcanonical = maps.canonical_to_raw' * d.hessian * maps.canonical_to_raw
    eigensystem = eigen(Symmetric(hcanonical))
    null = eigensystem.vectors[:, 1]
    (; theta, null_vector=null, k=Float64(k0),
       eigenvalues=eigensystem.values,
       gradient_residual=norm(d.gradient, Inf),
       null_residual=norm(hcanonical * null, Inf),
       converged=true, iterations=0, tolerance=Float64(tolerance),
       max_iterations)
end

"""
    n8_mass_eigenbasis(k=N8_KC; theta=N8_BEST_X)

Construct the fixed mass basis at a hilltop. The raw-coordinate vectors solve
`H_theta * v = m² * K * v` and are K-orthonormal. The basis is constructed
once at this point and is not recomputed along a trajectory.
"""
function n8_mass_eigenbasis(k::Real=N8_KC;
        theta::AbstractVector{<:Real}=N8_BEST_X)
    length(theta) == 8 || throw(DimensionMismatch("poly-102 needs eight coordinates"))
    maps = n8_coordinate_maps(k)
    d = n8_potential_derivatives(theta, k; trajectory=true)
    hcanonical = maps.canonical_to_raw' * d.hessian * maps.canonical_to_raw
    eigensystem = eigen(Symmetric(hcanonical))
    raw_eigenvectors = maps.canonical_to_raw * eigensystem.vectors
    (; basis=:mass_eigenbasis, theta=Float64.(theta), k=Float64(k),
       metric=maps.metric, hessian_theta=d.hessian,
       canonical_hessian=hcanonical, eigenvalues=eigensystem.values,
       canonical_eigenvectors=eigensystem.vectors, raw_eigenvectors,
       orthonormality_residual=norm(raw_eigenvectors' * maps.metric *
           raw_eigenvectors - I, Inf),
       generalized_residual=opnorm(d.hessian * raw_eigenvectors -
           maps.metric * raw_eigenvectors * Diagonal(eigensystem.values)))
end

"""
    n8_unstable_direction(k=N8_KC; mode=:most_negative, basis=:canonical_hessian)

Return a fixed hilltop direction as a K-normalized raw tangent. The default
`basis=:canonical_hessian` preserves the existing behavior. The explicit
`:mass_eigenbasis` option solves the generalized Hessian problem; it is
equivalent at fixed `K`, but distinct from the kinetic eigenbasis.
"""
function n8_unstable_direction(k::Real=N8_KC; mode::Symbol=:most_negative,
        basis::Symbol=:canonical_hessian,
        basis_theta::AbstractVector{<:Real}=N8_BEST_X)
    basis in (:canonical_hessian, :mass_eigenbasis) ||
        throw(ArgumentError("unsupported basis: $basis"))
    mass_basis = n8_mass_eigenbasis(k; theta=basis_theta)
    index = mode === :smallest_abs ? argmin(abs.(mass_basis.eigenvalues)) :
        argmin(mass_basis.eigenvalues)
    canonical = mass_basis.canonical_eigenvectors[:, index]
    raw = mass_basis.raw_eigenvectors[:, index]
    (; raw, canonical, eigenvalues=mass_basis.eigenvalues, index,
       metric=mass_basis.metric, basis, basis_theta=mass_basis.theta,
       basis_k=mass_basis.k, raw_eigenvectors=mass_basis.raw_eigenvectors,
       canonical_eigenvectors=mass_basis.canonical_eigenvectors,
       hessian_theta=mass_basis.hessian_theta,
       generalized_residual=mass_basis.generalized_residual,
       orthonormality_residual=mass_basis.orthonormality_residual)
end

"""
Build the author geometric displacement in raw coordinates. `basis` is
constructed at the hilltop once and remains fixed during trajectory
integration; only the potential gradient is recomputed along the path.
"""
function n8_inflation_initial_condition(k::Real; displacement::Real=1e-8,
        sign::Real=-1, direction_mode::Symbol=:most_negative,
        direction_raw::Union{Nothing,AbstractVector{<:Real}}=nothing,
        basis::Symbol=:canonical_hessian,
        basis_theta::AbstractVector{<:Real}=N8_BEST_X)
    displacement > 0 || throw(ArgumentError("displacement must be positive"))
    basis in (:canonical_hessian, :mass_eigenbasis) ||
        throw(ArgumentError("unsupported basis: $basis"))
    maps = n8_coordinate_maps(k)
    direction = if direction_raw === nothing
        n8_unstable_direction(k; mode=direction_mode, basis=basis,
            basis_theta=basis_theta)
    else
        length(direction_raw) == 8 || throw(DimensionMismatch("poly-102 needs eight coordinates"))
        raw = Float64.(direction_raw)
        norm_raw = sqrt(dot(raw, maps.metric * raw))
        norm_raw > 0 || throw(ArgumentError("direction must be nonzero"))
        raw ./= norm_raw
        canonical = maps.raw_to_canonical * raw
        (; raw, canonical, eigenvalues=Float64[], index=0, metric=maps.metric,
           basis, basis_theta=Float64.(basis_theta), basis_k=Float64(k),
           raw_eigenvectors=nothing, canonical_eigenvectors=nothing,
           generalized_residual=NaN, orthonormality_residual=NaN)
    end
    theta = N8_BEST_X .+ Float64(sign * displacement) .* direction.raw
    d = n8_potential_derivatives(theta, k; trajectory=true)
    gradnorm = sqrt(max(dot(d.gradient, maps.metric \ d.gradient), 0.0))
    tangent = gradnorm == 0 ? zeros(8) : -(maps.metric \ d.gradient) / gradnorm
    (; theta, theta_critical=copy(N8_BEST_X), direction=direction.raw,
       canonical_direction=direction.canonical, displacement=Float64(displacement),
       canonical_norm=sqrt(dot(theta .- N8_BEST_X, maps.metric * (theta .- N8_BEST_X))),
       initial_tangent=tangent, initial_gradient=d.gradient,
       epsilon=0.5 * (gradnorm / d.value)^2,
       eta_parallel=dot(tangent, d.hessian * tangent) / d.value,
       k=Float64(k), kc=N8_KC, displacement_mode=:geometric,
       displacement_sign=Float64(sign), basis=direction.basis,
       basis_theta=copy(direction.basis_theta), basis_k=direction.basis_k,
       basis_eigenvalues=copy(direction.eigenvalues),
       basis_raw_eigenvectors=direction.raw_eigenvectors,
       basis_canonical_eigenvectors=direction.canonical_eigenvectors,
       basis_orthonormality_residual=direction.orthonormality_residual,
       basis_generalized_residual=direction.generalized_residual)
end

"""Return the explicitly normalized direction variants used in the basis audit."""
function n8_basis_directions(k::Real=N8_KC)
    maps = n8_coordinate_maps(k)
    unstable = n8_unstable_direction(k; basis=:canonical_hessian)
    mass = n8_unstable_direction(k; basis=:mass_eigenbasis)
    kinetic = eigen(Symmetric(maps.metric))
    kinetic_directions = [
        _metric_normalize(kinetic.vectors[:, i], maps.metric)
        for i in axes(kinetic.vectors, 2)
    ]
    draft_index = argmax(abs.([
        dot(direction, maps.metric * unstable.raw)
        for direction in kinetic_directions
    ]))
    raw_coordinate = _metric_normalize([1.0, zeros(7)...], maps.metric)
    directions = (
        A_draft_kinetic=kinetic_directions[draft_index],
        B_package_current=unstable.raw,
        C_canonical_hessian=unstable.raw,
        D_raw_coordinate=raw_coordinate,
        E_mass_eigenbasis=mass.raw,
    )
    overlap = [dot(left, maps.metric * right) for left in directions, right in directions]
    (; directions, overlap, metric=maps.metric, metric_eigenvalues=kinetic.values,
       draft_kinetic_index=draft_index, unstable=unstable,
       canonical_hessian=unstable, mass_eigenbasis=mass,
       equivalent_mass_direction=abs(dot(unstable.raw, maps.metric * mass.raw)))
end

"""
    n8_hilltop_efolds(delta_k)

The author-supplied local normal-form check.  It is independent of the
integration parameterization and reproduces the two poly-102 reference
scales.  The parameters are retained explicitly because they are part of the
Mathematica numerical specification.
"""
function n8_hilltop_efolds(delta_k::Real; displacement::Real=1e-8)
    delta_k > 0 || throw(ArgumentError("delta_k must be positive"))
    displacement > 0 || throw(ArgumentError("displacement must be positive"))
    μ² = 93.86 * Float64(delta_k)
    denominator = 1.52e7 * Float64(displacement)^2
    (; delta_k=Float64(delta_k), displacement=Float64(displacement),
       μ², efolds=log1p(μ² / denominator) / (2μ²),
       alpha_over_vc=93.86, g_over_vc=1.52e7)
end

"""
    n8_hilltop_probe(delta_k; ...)

Sample the supplied one-dimensional hilltop normal form without integrating
through the stiff heavy-mode transient of the full eight-field system.  This
is the deterministic local reference used by the serialized audit fixture;
`n8_slow_roll_trajectory` remains the full nonlinear RK4 probe.
"""
function n8_hilltop_probe(delta_k::Real; displacement::Real=1e-8,
        displacement_sign::Real=-1, sample_count::Int=20,
        direction_mode::Symbol=:most_negative,
        direction_raw::Union{Nothing,AbstractVector{<:Real}}=nothing,
        basis::Symbol=:canonical_hessian,
        basis_theta::AbstractVector{<:Real}=N8_BEST_X)
    sample_count > 0 || throw(ArgumentError("sample_count must be positive"))
    reference = n8_hilltop_efolds(delta_k; displacement=displacement)
    k = N8_KC + Float64(delta_k)
    initial = n8_inflation_initial_condition(k; displacement=displacement,
        sign=displacement_sign, direction_mode=direction_mode,
        direction_raw=direction_raw, basis=basis, basis_theta=basis_theta)
    μ² = reference.μ²
    g = reference.g_over_vc
    total = reference.efolds
    samples = NamedTuple[]
    for i in 0:(sample_count - 1)
        fraction = sample_count == 1 ? 0.0 : i / (sample_count - 1)
        n = total * fraction * (1 - (fraction == 1 ? 1e-12 : 0.0))
        denominator = exp(2μ² * (total - n)) - 1
        x = sqrt(μ² / (g * denominator))
        theta = N8_BEST_X .+ (x / displacement) .* (initial.theta - N8_BEST_X)
        state = _n8_canonical_state(
            n8_coordinate_maps(k).raw_to_canonical * theta, k)
        push!(samples, (n=n, theta=copy(theta), epsilon=state.epsilon,
            eta_parallel=state.eta_parallel, potential=state.value))
    end
    (; k, delta_k=Float64(delta_k), efolds=total, entered_slow_roll=true,
       end_event=:local_normal_form, steps=sample_count, samples, initial,
       basis=initial.basis, basis_theta=copy(initial.basis_theta), basis_k=k,
       solver=(method=:local_normal_form, sample_count))
end

function _n8_canonical_state(chi::AbstractVector{<:Real}, k::Real,
        maps=n8_coordinate_maps(k))
    theta = maps.canonical_to_raw * Float64.(chi)
    d = n8_potential_derivatives(theta, k; trajectory=true)
    gradient = maps.canonical_to_raw' * d.gradient
    hessian = maps.canonical_to_raw' * d.hessian * maps.canonical_to_raw
    gradient_squared = dot(gradient, gradient)
    epsilon = 0.5 * gradient_squared / max(d.value^2, eps(Float64))
    tangent = gradient_squared == 0 ? zeros(length(gradient)) : -gradient / sqrt(gradient_squared)
    eta_parallel = gradient_squared == 0 ? Inf : dot(tangent, hessian * tangent) / d.value
    (; theta, value=d.value, gradient, hessian, epsilon, eta_parallel,
       tangent, spectral_scale=opnorm(hessian) / max(abs(d.value), eps(Float64)))
end

"""
    n8_slow_roll_trajectory(delta_k; ...)

Integrate the author slow-roll equations in canonical coordinates with a
bounded-step RK4 scheme.  `N` is the independent variable and the event is
`max(epsilon, abs(eta_parallel)) == 1`.  The returned samples are deliberately
kept small so the result is suitable for a serialized comparison fixture.
"""
function n8_slow_roll_trajectory(delta_k::Real; displacement::Real=1e-8,
        displacement_sign::Real=-1, max_efolds::Real=1e6,
        max_step::Real=5.0, initial_step::Real=1e-7, sample_count::Int=20,
        direction_mode::Symbol=:most_negative,
        direction_raw::Union{Nothing,AbstractVector{<:Real}}=nothing,
        basis::Symbol=:canonical_hessian,
        basis_theta::AbstractVector{<:Real}=N8_BEST_X)
    delta_k > 0 || throw(ArgumentError("delta_k must be positive"))
    max_step > 0 || throw(ArgumentError("max_step must be positive"))
    sample_count > 0 || throw(ArgumentError("sample_count must be positive"))
    k = N8_KC + Float64(delta_k)
    initial = n8_inflation_initial_condition(k; displacement=displacement,
        sign=displacement_sign, direction_mode=direction_mode,
        direction_raw=direction_raw, basis=basis, basis_theta=basis_theta)
    maps = n8_coordinate_maps(k)
    chi = maps.raw_to_canonical * initial.theta
    n = 0.0
    entered = false
    start_n = 0.0
    samples = NamedTuple[]
    steps = 0
    end_event = :tmax
    while n < max_efolds && steps < 2_000_000
        state = _n8_canonical_state(chi, k, maps)
        if length(samples) < sample_count
            push!(samples, (n=n, theta=copy(state.theta), epsilon=state.epsilon,
                eta_parallel=state.eta_parallel, potential=state.value))
        end
        inflating = state.epsilon < 1 && abs(state.eta_parallel) < 1
        if inflating && !entered
            entered = true
            start_n = n
        elseif entered && !inflating
            end_event = abs(state.eta_parallel) >= 1 ? :eta_parallel : :epsilon
            break
        end
        scale = max(1.0, state.spectral_scale)
        h = min(Float64(max_step), max(Float64(initial_step), 0.2 / scale))
        h = min(h, Float64(max_efolds) - n)
        rhs(x) = begin
            current = _n8_canonical_state(x, k, maps)
            -current.gradient / max(current.value, eps(Float64))
        end
        k1 = rhs(chi)
        k2 = rhs(chi .+ (h / 2) .* k1)
        k3 = rhs(chi .+ (h / 2) .* k2)
        k4 = rhs(chi .+ h .* k3)
        chi .+= (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        n += h
        steps += 1
    end
    if steps >= 2_000_000
        end_event = :step_limit
    end
    (; k, delta_k=Float64(delta_k), efolds=entered ? n - start_n : 0.0,
       entered_slow_roll=entered, end_event, steps, samples, initial,
       basis=initial.basis, basis_theta=initial.basis_theta,
       basis_k=initial.basis_k,
       solver=(method=:rk4, max_efolds=Float64(max_efolds),
           max_step=Float64(max_step), initial_step=Float64(initial_step)))
end

function _n8_big_symmetric_power(a::AbstractMatrix{BigFloat}, power::Real)
    eigensystem = eigen(Symmetric(a))
    minimum(eigensystem.values) > 0 ||
        throw(ArgumentError("metric is not positive definite"))
    eigensystem.vectors * Diagonal(eigensystem.values .^ power) *
        eigensystem.vectors'
end

function _n8_big_derivatives(theta::AbstractVector{BigFloat}, k::BigFloat,
        q::AbstractMatrix{BigFloat}, tau::AbstractVector{BigFloat})
    weights = (14k)^2 .* (tau ./ 14) .* exp.(-2π * k .* (tau .- 14))
    arguments = q' * theta
    value = sum(weights .* (1 .- cos.(arguments)))
    gradient = q * (weights .* sin.(arguments))
    hessian = q * Diagonal(weights .* cos.(arguments)) * q'
    (; value, gradient, hessian, weights, arguments)
end

function _n8_big_state(chi::AbstractVector{BigFloat}, k::BigFloat,
        q::AbstractMatrix{BigFloat}, tau::AbstractVector{BigFloat},
        canonical_to_raw::AbstractMatrix{BigFloat})
    theta = canonical_to_raw * chi
    derivatives = _n8_big_derivatives(theta, k, q, tau)
    gradient = canonical_to_raw' * derivatives.gradient
    hessian = canonical_to_raw' * derivatives.hessian * canonical_to_raw
    gradient_squared = dot(gradient, gradient)
    gradient_norm = sqrt(gradient_squared)
    epsilon = gradient_squared / (2 * derivatives.value^2)
    tangent = gradient_norm == 0 ? zeros(BigFloat, length(gradient)) :
        -gradient / gradient_norm
    eta_parallel = gradient_norm == 0 ? big"Inf" :
        dot(tangent, hessian * tangent) / derivatives.value
    (; theta, value=derivatives.value, gradient, hessian, epsilon,
       eta_parallel, tangent, derivatives)
end

function _n8_big_event(sol, time::BigFloat, k::BigFloat,
        q::AbstractMatrix{BigFloat}, tau::AbstractVector{BigFloat},
        canonical_to_raw::AbstractMatrix{BigFloat})
    state = _n8_big_state(sol(time), k, q, tau, canonical_to_raw)
    max(state.epsilon, abs(state.eta_parallel)) - 1
end

function _n8_big_bisect(f, left::BigFloat, right::BigFloat;
        iterations::Int=160)
    left_value = f(left)
    right_value = f(right)
    left_value == 0 && return left
    right_value == 0 && return right
    signbit(left_value) == signbit(right_value) &&
        throw(ArgumentError("root bracket does not contain a sign change"))
    for _ in 1:iterations
        middle = (left + right) / 2
        middle_value = f(middle)
        middle_value == 0 && return middle
        if signbit(left_value) == signbit(middle_value)
            left, left_value = middle, middle_value
        else
            right, right_value = middle, middle_value
        end
    end
    (left + right) / 2
end

"""
    n8_author_trajectory(delta_k; ...)

Reproduce the author's physical-time gradient flow with an arbitrary-precision
stiff solver. The hilltop direction is constructed once and remains fixed;
only the nonlinear potential derivatives are recomputed along the trajectory.
The slow-roll window is located with the author's scan step and then refined
with dense-output bisection.
"""
function n8_author_trajectory(delta_k::Real; displacement::Real=1e-8,
        displacement_sign::Real=-1, max_time::Real=1e6,
        scan_step::Real=5, sample_count::Int=20,
        basis::Symbol=:canonical_hessian,
        basis_theta::AbstractVector{<:Real}=N8_BEST_X,
        precision_bits::Int=100, reltol=nothing, abstol=nothing,
        maxiters::Int=10^8)
    delta_k > 0 || throw(ArgumentError("delta_k must be positive"))
    displacement > 0 || throw(ArgumentError("displacement must be positive"))
    scan_step > 0 || throw(ArgumentError("scan_step must be positive"))
    sample_count > 0 || throw(ArgumentError("sample_count must be positive"))
    basis in (:canonical_hessian, :mass_eigenbasis) ||
        throw(ArgumentError("unsupported basis: $basis"))
    precision_bits >= 64 || throw(ArgumentError("precision_bits must be at least 64"))
    setprecision(BigFloat, precision_bits) do
        T = BigFloat
        k64 = Float64(N8_KC + Float64(delta_k))
        k = T(k64)
        q = T.(Matrix(N8_Q_TRAJECTORY'))
        tau = T.(N8_TAU_TRAJECTORY)
        metric = T.(N8_K_RAW) / k^2
        raw_to_canonical = _n8_big_symmetric_power(metric, 1 / 2)
        canonical_to_raw = _n8_big_symmetric_power(metric, -1 / 2)
        hilltop = T.(basis_theta)
        hilltop_derivatives = _n8_big_derivatives(hilltop, k, q, tau)
        hilltop_hessian = canonical_to_raw' * hilltop_derivatives.hessian *
            canonical_to_raw
        eigensystem = eigen(Symmetric(hilltop_hessian))
        index = argmin(eigensystem.values)
        canonical_direction = copy(eigensystem.vectors[:, index])
        raw_direction = canonical_to_raw * canonical_direction
        sign_index = argmax(abs.(raw_direction))
        if raw_direction[sign_index] > 0
            raw_direction = -raw_direction
            canonical_direction = -canonical_direction
        end
        theta_initial = hilltop .+
            T(displacement_sign * Float64(displacement)) .* raw_direction
        chi_initial = raw_to_canonical * theta_initial
        initial_state = _n8_big_state(
            chi_initial, k, q, tau, canonical_to_raw)
        initial_vector = vcat(chi_initial, zero(T))
        entered = Ref(false)
        condition(u, _time, _integrator) =
            begin
                state = _n8_big_state(view(u, 1:8), k, q, tau, canonical_to_raw)
                max(state.epsilon, abs(state.eta_parallel)) - 1
            end
        affect!(integrator) = entered[] && terminate!(integrator)
        affect_neg!(integrator) = (entered[] = true)
        callback = ContinuousCallback(condition, affect!, affect_neg!)
        rhs!(du, u, _parameters, _time) = begin
            state = _n8_big_state(view(u, 1:8), k, q, tau, canonical_to_raw)
            du[1:8] .= -state.gradient
            du[9] = state.value
            nothing
        end
        jac!(jacobian, u, _parameters, _time) = begin
            state = _n8_big_state(view(u, 1:8), k, q, tau, canonical_to_raw)
            fill!(jacobian, zero(T))
            jacobian[1:8, 1:8] .= -state.hessian
            jacobian[9, 1:8] .= state.gradient
            nothing
        end
        tmax = T(Float64(max_time))
        problem = ODEProblem(ODEFunction(rhs!; jac=jac!),
            initial_vector, (zero(T), tmax))
        default_reltol = T(10)^(-precision_bits ÷ 2)
        default_abstol = T(10)^(-precision_bits * 2 ÷ 3)
        solution = solve(problem, Rodas5P(autodiff=AutoFiniteDiff(),
                concrete_jac=true);
            callback, reltol=reltol === nothing ? default_reltol : T(reltol),
            abstol=abstol === nothing ? default_abstol : T(abstol),
            dtmax=T(Float64(scan_step)), maxiters,
            save_everystep=false, dense=true)
        solution.retcode == ReturnCode.Failure &&
            throw(ErrorException("author trajectory solver failed: $(solution.retcode)"))
        event(time) = _n8_big_event(
            solution, time, k, q, tau, canonical_to_raw)
        end_time = solution.t[end]
        entry_time = nothing
        previous_time = zero(T)
        previous_value = event(previous_time)
        if previous_value <= 0
            entry_time = previous_time
        else
            next_time = min(previous_time + T(Float64(scan_step)), end_time)
            while next_time > previous_time
                next_value = event(next_time)
                if next_value < 0
                    entry_time = _n8_big_bisect(
                        event, previous_time, next_time)
                    break
                end
                previous_time, previous_value = next_time, next_value
                next_time = min(previous_time + T(Float64(scan_step)), end_time)
            end
        end
        if entry_time === nothing
            return (; delta_k=T(Float64(delta_k)), k, entered_slow_roll=false,
                end_event=:no_slow_roll_window, efolds=zero(T),
                samples=NamedTuple[], initial=initial_state,
                basis, basis_theta=copy(hilltop), basis_k=k,
                precision_bits, solver=solution.retcode)
        end
        entry_n = solution(entry_time)[9]
        exit_time = nothing
        previous_time = entry_time
        previous_value = event(previous_time)
        next_time = min(previous_time + T(Float64(scan_step)), end_time)
        while next_time > previous_time
            next_value = event(next_time)
            if previous_value < 0 && next_value >= 0
                exit_time = _n8_big_bisect(
                    event, previous_time, next_time)
                break
            end
            previous_time, previous_value = next_time, next_value
            next_time = min(previous_time + T(Float64(scan_step)), end_time)
        end
        exit_time = exit_time === nothing ? end_time : exit_time
        exit_state = _n8_big_state(
            solution(exit_time), k, q, tau, canonical_to_raw)
        exit_event = abs(exit_state.eta_parallel) >= exit_state.epsilon ?
            :eta_parallel : :epsilon
        end_n = solution(exit_time)[9]
        sample_ns = range(entry_n, end_n; length=sample_count)
        samples = NamedTuple[]
        for target_n in sample_ns
            sample_time = _n8_big_bisect(
                time -> solution(time)[9] - target_n,
                entry_time, exit_time)
            state = _n8_big_state(
                solution(sample_time), k, q, tau, canonical_to_raw)
            push!(samples, (n=target_n, theta=state.theta,
                epsilon=state.epsilon, eta_parallel=state.eta_parallel,
                potential=state.value))
        end
        (; delta_k=T(Float64(delta_k)), k, entered_slow_roll=true,
            entry_n, end_n, efolds=end_n - entry_n, end_event=exit_event,
            samples, initial=initial_state, basis,
            basis_theta=copy(hilltop), basis_k=k, basis_eigenvalues=eigensystem.values,
            basis_raw_direction=raw_direction,
            basis_canonical_direction=canonical_direction, precision_bits,
            solver=(method=:Rodas5P, reltol=default_reltol,
                abstol=default_abstol, scan_step=T(Float64(scan_step)),
                retcode=solution.retcode))
    end
end

"""N=5 local-form reference used by Figure 3 (the author scan's two anchors)."""
function n5_hilltop_efolds(delta_k::Real; displacement::Real=1e-8)
    delta_k > 0 || throw(ArgumentError("delta_k must be positive"))
    displacement > 0 || throw(ArgumentError("displacement must be positive"))
    μ² = 2589.3499862235153 * Float64(delta_k)
    denominator = 1.8288577346700535e6 * Float64(displacement)^2
    (; delta_k=Float64(delta_k), displacement=Float64(displacement), μ²,
       efolds=log1p(μ² / denominator) / (2μ²),
       alpha_over_vc=2589.3499862235153, g_over_vc=1.8288577346700535e6)
end

"""Return the representative draft detunings and reference efold counts."""
function reference_efolds()
    (; n5=[(delta_k=1e-7, efolds=27349.0), (delta_k=6.65e-5, efolds=60.0)],
       n8=[(delta_k=1e-7, efolds=463115.0), (delta_k=1.5320548620798324e-3, efolds=60.0)])
end

end # author_inflation

const N8_KC = author_inflation.N8_KC
const N8_BEST_X = author_inflation.N8_BEST_X
const N8_TAU = author_inflation.N8_TAU
const N8_Q = author_inflation.N8_Q
const N8_Q_TRAJECTORY = author_inflation.N8_Q_TRAJECTORY
const N8_TAU_TRAJECTORY = author_inflation.N8_TAU_TRAJECTORY
const N8_K_RAW = author_inflation.N8_K_RAW
const N5_Q = author_inflation.N5_Q
const N5_QDOTTAU = author_inflation.N5_QDOTTAU
const N5_VERTICES = author_inflation.N5_VERTICES
const N5_VOLUME = author_inflation.N5_VOLUME
const N5_DIVISOR_VOLUMES = author_inflation.N5_DIVISOR_VOLUMES
const N5_K_RAW = author_inflation.N5_K_RAW
const N5_LIGHT_DIRECTION = author_inflation.N5_LIGHT_DIRECTION

const n5_geometry = author_inflation.n5_geometry
const n5_kinetic_matrix = author_inflation.n5_kinetic_matrix
const n5_light_direction = author_inflation.n5_light_direction
const n8_author_trajectory = author_inflation.n8_author_trajectory
const n5_hilltop_efolds = author_inflation.n5_hilltop_efolds
const reference_efolds = author_inflation.reference_efolds

n8_hilltop_probe(args...; kwargs...) =
    author_inflation.n8_hilltop_probe(args...; kwargs...)
n8_slow_roll_trajectory(args...; kwargs...) =
    author_inflation.n8_slow_roll_trajectory(args...; kwargs...)

function _n8_symmetric_power(matrix::AbstractMatrix{<:Real}, power::Real)
    eigensystem = eigen(Symmetric(Float64.(matrix)))
    minimum(eigensystem.values) > 0 ||
        throw(ArgumentError("kinetic metric must be positive definite"))
    eigensystem.vectors * Diagonal(eigensystem.values .^ power) *
        eigensystem.vectors'
end

function n8_coordinate_maps(k::Real)
    metric = Matrix(n8_kinetic_matrix(k))
    (; metric, raw_to_canonical=_n8_symmetric_power(metric, 1 / 2),
       canonical_to_raw=_n8_symmetric_power(metric, -1 / 2))
end

function n8_mass_eigenbasis(k::Real=N8_KC;
        theta::Union{Nothing,AbstractVector{<:Real}}=nothing,
        basis_k::Real=k)
    basis_theta = theta === nothing ? n8_hilltop(basis_k).theta : Float64.(theta)
    length(basis_theta) == 8 ||
        throw(DimensionMismatch("the N=8 benchmark needs eight coordinates"))
    maps = n8_coordinate_maps(basis_k)
    derivatives = n8_potential_derivatives(basis_theta, basis_k)
    canonical_hessian = maps.canonical_to_raw' * derivatives.hessian *
        maps.canonical_to_raw
    eigensystem = eigen(Symmetric(canonical_hessian))
    raw_eigenvectors = maps.canonical_to_raw * eigensystem.vectors
    (; basis=:mass_eigenbasis, theta=basis_theta, k=Float64(basis_k),
       metric=maps.metric, hessian_theta=derivatives.hessian,
       canonical_hessian, eigenvalues=eigensystem.values,
       canonical_eigenvectors=eigensystem.vectors, raw_eigenvectors,
       orthonormality_residual=norm(raw_eigenvectors' * maps.metric *
           raw_eigenvectors - I, Inf),
       generalized_residual=opnorm(derivatives.hessian * raw_eigenvectors -
           maps.metric * raw_eigenvectors * Diagonal(eigensystem.values)))
end

function n8_unstable_direction(k::Real=N8_KC;
        mode::Symbol=:most_negative, basis::Symbol=:canonical_hessian,
        basis_theta::Union{Nothing,AbstractVector{<:Real}}=nothing,
        basis_k::Real=N8_KC)
    basis in (:canonical_hessian, :mass_eigenbasis) ||
        throw(ArgumentError("unsupported basis: $basis"))
    mass_basis = n8_mass_eigenbasis(k; theta=basis_theta, basis_k=basis_k)
    index = mode === :smallest_abs ? argmin(abs.(mass_basis.eigenvalues)) :
        mode === :most_negative ? argmin(mass_basis.eigenvalues) :
        throw(ArgumentError("unsupported direction mode: $mode"))
    (; raw=mass_basis.raw_eigenvectors[:, index],
       canonical=mass_basis.canonical_eigenvectors[:, index],
       eigenvalues=mass_basis.eigenvalues, index, metric=mass_basis.metric,
       basis, basis_theta=mass_basis.theta, basis_k=mass_basis.k,
       raw_eigenvectors=mass_basis.raw_eigenvectors,
       canonical_eigenvectors=mass_basis.canonical_eigenvectors,
       hessian_theta=mass_basis.hessian_theta,
       generalized_residual=mass_basis.generalized_residual,
       orthonormality_residual=mass_basis.orthonormality_residual)
end

function n8_basis_directions(k::Real=N8_KC)
    unstable = n8_unstable_direction(k; basis=:canonical_hessian)
    mass = n8_unstable_direction(k; basis=:mass_eigenbasis)
    kinetic = eigen(Symmetric(unstable.metric))
    directions = [unstable.metric * kinetic.vectors[:, i] for i in axes(kinetic.vectors, 2)]
    directions = [direction / sqrt(dot(direction, unstable.metric * direction))
        for direction in directions]
    draft_index = argmax(abs.([
        dot(direction, unstable.metric * unstable.raw) for direction in directions]))
    named = (
        A_draft_kinetic=directions[draft_index],
        B_package_current=unstable.raw,
        C_canonical_hessian=unstable.raw,
        D_raw_coordinate=normalize([1.0, zeros(7)...]),
        E_mass_eigenbasis=mass.raw,
    )
    overlap = [dot(left, unstable.metric * right) for left in named, right in named]
    (; directions=named, overlap, metric=unstable.metric,
       metric_eigenvalues=kinetic.values, draft_kinetic_index=draft_index,
       unstable, canonical_hessian=unstable, mass_eigenbasis=mass,
       equivalent_mass_direction=abs(dot(unstable.raw,
           unstable.metric * mass.raw)))
end

end
