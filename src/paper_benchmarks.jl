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
function n8_potential(; k::Real=1.0)
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

"""Complete equation-(19) potential from the 12 diagonal and 66 cross terms."""
function n8_full_potential(; k::Real=1.0)
    diagonal = n8_potential(k=k)
    qprime = Matrix{Int}(diagonal.Q')
    geometry = n8_geometry()
    tau = Float64(k) .* geometry.divisor_volumes
    kinv = Float64(k)^2 .* inv(Matrix(geometry.kinetic))
    volume = geometry.volume * Float64(k)^(3 / 2)
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
    (; Q=hcat(charges...), L=vcat(reshape(signs, 1, :), reshape(logs, 1, :)),
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

    function scaled_quantities(theta, k)
        L = instanton_scales(ordered_qdotτ, k)
        logs = vec(L[2, :])
        amplitudes = vec(L[1, :]) .* 10.0 .^ (logs .- maximum(logs))
        row_scales = amplitudes[1:n]
        arguments = 2π .* (qcanonical' * theta)
        gradient = 2π .* qcanonical * (amplitudes .* sin.(arguments))
        hessian = (2π)^2 .* qcanonical * Diagonal(amplitudes .* cos.(arguments)) * qcanonical'
        inverse_sqrt = Diagonal(inv.(sqrt.(row_scales)))
        gradient ./ row_scales, inverse_sqrt * hessian * inverse_sqrt
    end

    _, initial_hessian = scaled_quantities(Float64.(initial_theta), Float64(k0))
    initial_vector = eigen(Hermitian(initial_hessian)).vectors[:, 1]
    initial = vcat(Float64.(initial_theta), initial_vector, Float64(k0))
    function equations!(out, state)
        theta = @view state[1:n]
        null_vector = @view state[(n + 1):(2n)]
        k = state[end]
        gradient, hessian = scaled_quantities(theta, k)
        out[1:n] .= gradient
        out[(n + 1):(2n)] .= hessian * null_vector
        out[end] = dot(null_vector, null_vector) - 1
        nothing
    end
    result = nlsolve(equations!, initial; method=:trust_region,
        ftol=tolerance, xtol=tolerance, iterations=max_iterations)
    theta = mod.(result.zero[1:n], 1.0)
    null_vector = result.zero[(n + 1):(2n)]
    k = result.zero[end]
    gradient, hessian = scaled_quantities(theta, k)
    eigenvalues = eigvals(Hermitian(hessian))
    (; theta, null_vector=null_vector / norm(null_vector), k, eigenvalues,
       gradient_residual=norm(gradient, Inf), null_residual=norm(hessian * null_vector, Inf),
       converged=result.f_converged || result.x_converged, iterations=result.iterations)
end


"""Table-1 truncated potential and its first two derivatives in GLSM coordinates.

Set `full=true` to evaluate the separate 78-term equation-(19) reconstruction.
The paper's catastrophe search and inflation trajectories use the truncated
potential of equations (20) and (25), so the 12-term potential is the default.
"""
function n8_potential_derivatives(theta::AbstractVector{<:Real}, k::Real;
        full::Bool=false)
    length(theta) == 8 || throw(DimensionMismatch("the N=8 benchmark needs eight coordinates"))
    k > 0 || throw(ArgumentError("k must be positive"))
    benchmark = full ? n8_full_potential(k=k) : n8_potential(k=k)
    q = Matrix{Float64}(benchmark.Q)
    amplitudes = vec(benchmark.L[1, :]) .* 10.0 .^ vec(benchmark.L[2, :])
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
        branch::Symbol=:a, follow_hilltop::Bool=false)
    displacement > 0 || throw(ArgumentError("displacement must be positive"))
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
    (; theta, theta_critical, direction, displacement=Float64(displacement),
       k=Float64(k), kc=catastrophe.k, hilltop, follow_hilltop)
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

end
