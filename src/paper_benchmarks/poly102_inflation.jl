module author_inflation
using LinearAlgebra
using NLsolve
using OrdinaryDiffEq
using LinearSolve

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
        coeff = (8π^2 / volume^2) *
            (dot(qi, kinv * qj) + qτ)
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

Return the author's refined poly-102 cusp by solving the augmented
gradient/Hessian-null equations from the supplied starting point.
"""
function n8_degenerate_point(initial_theta::AbstractVector{<:Real}=N8_BEST_X;
        k0::Real=N8_KC, tolerance::Real=1e-11, max_iterations::Int=1_000)
    length(initial_theta) == 8 || throw(DimensionMismatch("poly-102 needs eight coordinates"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    theta₀ = Float64.(initial_theta)
    k₀ = Float64(k0)
    d₀ = n8_potential_derivatives(theta₀, k₀; trajectory=true)
    maps₀ = n8_coordinate_maps(k₀)
    hcanonical₀ = maps₀.canonical_to_raw' * d₀.hessian * maps₀.canonical_to_raw
    null₀ = eigen(Symmetric(hcanonical₀)).vectors[:, 1]
    initial = vcat(theta₀, null₀, k₀)
    function equations!(out, state)
        theta = @view state[1:8]
        null = @view state[9:16]
        k = state[17]
        derivatives = n8_potential_derivatives(theta, k; trajectory=true)
        maps = n8_coordinate_maps(k)
        hcanonical = maps.canonical_to_raw' * derivatives.hessian * maps.canonical_to_raw
        out[1:8] .= derivatives.gradient
        out[9:16] .= hcanonical * null
        out[17] = dot(null, null) - 1
        nothing
    end
    result = nlsolve(equations!, initial; method=:trust_region,
        ftol=tolerance, xtol=tolerance, iterations=max_iterations)
    theta = mod.(result.zero[1:8], 2π)
    null = result.zero[9:16]
    k = result.zero[17]
    derivatives = n8_potential_derivatives(theta, k; trajectory=true)
    maps = n8_coordinate_maps(k)
    hcanonical = maps.canonical_to_raw' * derivatives.hessian * maps.canonical_to_raw
    eigensystem = eigen(Symmetric(hcanonical))
    gradient_residual = norm(derivatives.gradient, Inf)
    null_residual = norm(hcanonical * null, Inf)
    normalized_residual = abs(dot(null, null) - 1)
    converged = (result.f_converged || result.x_converged) &&
        gradient_residual <= tolerance && null_residual <= tolerance &&
        normalized_residual <= tolerance
    (; theta, null_vector=null / norm(null), k,
       eigenvalues=eigensystem.values, gradient_residual, null_residual,
       normalized_null_vector_residual=normalized_residual, converged,
       iterations=result.iterations, tolerance=Float64(tolerance), max_iterations)
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
    n8_hilltop_normal_form_efolds(delta_k)

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
    n8_hilltop_normal_form(delta_k; ...)

Sample the supplied one-dimensional hilltop normal form without integrating
through the stiff heavy-mode transient of the full eight-field system.  This
is the deterministic local reference used by the serialized audit fixture;
`n8_efold_gradient_flow` remains the full nonlinear RK4 probe.
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
    n8_efold_gradient_flow(delta_k; ...)

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
    state = _n8_big_state(view(sol(time), 1:8), k, q, tau, canonical_to_raw)
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
    n8_physical_gradient_flow(delta_k; ...)

Reproduce the author's physical-time gradient flow with an arbitrary-precision
stiff solver. The hilltop direction is constructed once and remains fixed;
only the nonlinear potential derivatives are recomputed along the trajectory.
All bracketed slow-roll windows are located from saved solver states and then
refined with dense-output bisection. The longest completed window is selected
to avoid counting a transient crossing. `efolds` is the total accumulated
e-fold coordinate at the selected exit; `slow_roll_efolds` retains the
duration of the selected window.
"""
function n8_author_trajectory(delta_k::Real; displacement::Real=1e-8,
        displacement_sign::Real=-1, max_time::Real=1e6,
        scan_step::Real=5, sample_count::Int=20,
        max_step::Real=100,
        initial_step::Real=1e-5,
        method::Symbol=:Rodas5P,
        basis::Symbol=:canonical_hessian,
        basis_theta::AbstractVector{<:Real}=N8_BEST_X,
        precision_bits::Int=100, reltol=nothing, abstol=nothing,
        maxiters::Int=10^8)
    delta_k > 0 || throw(ArgumentError("delta_k must be positive"))
    displacement > 0 || throw(ArgumentError("displacement must be positive"))
    scan_step > 0 || throw(ArgumentError("scan_step must be positive"))
    max_step > 0 || throw(ArgumentError("max_step must be positive"))
    initial_step > 0 || throw(ArgumentError("initial_step must be positive"))
    method === :Rodas5P ||
        throw(ArgumentError("unsupported solver method: $method; use :Rodas5P"))
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
        tgrad!(gradient, u, _parameters, _time) = fill!(gradient, zero(T))
        tmax = T(Float64(max_time))
        problem = ODEProblem(ODEFunction(rhs!; jac=jac!, tgrad=tgrad!,
                jac_prototype=zeros(T, 9, 9)),
            initial_vector, (zero(T), tmax))
        default_reltol = T(10)^(-precision_bits ÷ 2)
        default_abstol = T(10)^(-precision_bits * 2 ÷ 3)
        used_reltol = reltol === nothing ? default_reltol : T(reltol)
        used_abstol = abstol === nothing ? default_abstol : T(abstol)
        algorithm = Rodas5P(autodiff=AutoFiniteDiff(),
            linsolve=GenericLUFactorization())
        solution = solve(problem, algorithm;
            reltol=used_reltol, abstol=used_abstol,
            dt=min(tmax, T(Float64(initial_step))),
            dtmax=min(tmax, T(Float64(max_step))), maxiters,
            save_everystep=true, dense=true)
        solution.retcode == ReturnCode.Failure &&
            throw(ErrorException("author trajectory solver failed: $(solution.retcode)"))
        event(time) = _n8_big_event(
            solution, time, k, q, tau, canonical_to_raw)
        end_time = solution.t[end]
        # There can be short transient slow-roll windows before the final
        # inflationary interval. Retain every completed window and select the
        # longest one below, matching the reference's final-finite-exit policy.
        # An open interval at the solver horizon is retained only when no
        # completed interval exists, and is reported as :tmax rather than as a
        # physical end event.
        completed_windows = Tuple{T, T, Symbol}[]
        entry_time = nothing
        previous_time = solution.t[1]
        previous_value = begin
            state = _n8_big_state(
                view(solution.u[1], 1:8), k, q, tau, canonical_to_raw)
            max(state.epsilon, abs(state.eta_parallel)) - 1
        end
        if previous_value <= 0
            entry_time = previous_time
        end
        # Use saved accepted-step states for the sign scan.  The dense solution
        # is still used to refine each bracket, but fixed scan_step sampling
        # would recompute expensive BigFloat Hessians unnecessarily across a
        # long physical-time horizon.
        for index in 2:length(solution.t)
            next_time = solution.t[index]
            next_value = begin
                state = _n8_big_state(
                    view(solution.u[index], 1:8), k, q, tau, canonical_to_raw)
                max(state.epsilon, abs(state.eta_parallel)) - 1
            end
            if entry_time === nothing
                if previous_value >= 0 && next_value < 0
                    entry_time = _n8_big_bisect(
                        event, previous_time, next_time)
                end
            elseif previous_value < 0 && next_value >= 0
                exit_time = _n8_big_bisect(
                    event, previous_time, next_time)
                exit_state = _n8_big_state(
                    view(solution(exit_time), 1:8), k, q, tau, canonical_to_raw)
                exit_event = abs(exit_state.eta_parallel) >= exit_state.epsilon ?
                    :eta_parallel : :epsilon
                push!(completed_windows, (entry_time, exit_time, exit_event))
                entry_time = nothing
            end
            previous_time, previous_value = next_time, next_value
        end
        open_window = entry_time === nothing ? nothing :
            (entry_time, end_time, :tmax)
        windows = if isempty(completed_windows)
            open_window === nothing ? Tuple{T, T, Symbol}[] : [open_window]
        else
            completed_windows
        end
        if isempty(windows)
            return (; delta_k=T(Float64(delta_k)), k, entered_slow_roll=false,
                end_event=:no_slow_roll_window, efolds=zero(T),
                slow_roll_efolds=zero(T),
                terminated=false,
                samples=NamedTuple[], initial=initial_state,
                basis, basis_theta=copy(hilltop), basis_k=k,
                precision_bits,
                solver=(method, reltol=used_reltol, abstol=used_abstol,
                    retcode=solution.retcode,
                    accepted_steps=solution.stats.naccept,
                    rejected_steps=solution.stats.nreject,
                    rhs_evaluations=solution.stats.nf,
                    jacobian_evaluations=solution.stats.njacs))
        end
        window_index = argmax(window[2] - window[1] for window in windows)
        entry_time, exit_time, exit_event = windows[window_index]
        entry_n = solution(entry_time)[9]
        end_n = solution(exit_time)[9]
        sample_ns = sample_count == 1 ? (entry_n,) :
            range(entry_n, end_n; length=sample_count)
        samples = NamedTuple[]
        for target_n in sample_ns
            sample_time = _n8_big_bisect(
                time -> solution(time)[9] - target_n,
                entry_time, exit_time)
            state = _n8_big_state(
                view(solution(sample_time), 1:8), k, q, tau, canonical_to_raw)
            push!(samples, (n=target_n, theta=state.theta,
                epsilon=state.epsilon, eta_parallel=state.eta_parallel,
                potential=state.value))
        end
        (; delta_k=T(Float64(delta_k)), k, entered_slow_roll=true,
            entry_n, end_n, efolds=end_n,
            slow_roll_efolds=end_n - entry_n, end_event=exit_event,
            terminated=exit_event != :tmax,
            samples, initial=initial_state, basis,
            basis_theta=copy(hilltop), basis_k=k, basis_eigenvalues=eigensystem.values,
            basis_raw_direction=raw_direction,
            basis_canonical_direction=canonical_direction, precision_bits,
            solver=(method, reltol=used_reltol,
                abstol=used_abstol, scan_step=T(Float64(scan_step)),
                max_step=T(Float64(max_step)),
                initial_step=T(Float64(initial_step)),
                retcode=solution.retcode, accepted_steps=solution.stats.naccept,
                rejected_steps=solution.stats.nreject, rhs_evaluations=solution.stats.nf,
                jacobian_evaluations=solution.stats.njacs))
    end
end

"""N=5 hilltop normal-form reference for the two benchmark anchors."""
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

# Scientific names for the author-normal-form API. The historical names remain
# available as compatibility aliases while callers migrate.
const n8_physical_gradient_flow = n8_author_trajectory
const n8_hilltop_normal_form_efolds = n8_hilltop_efolds
const n8_hilltop_normal_form = n8_hilltop_probe
const n8_efold_gradient_flow = n8_slow_roll_trajectory
const n5_hilltop_normal_form_efolds = n5_hilltop_efolds
const benchmark_efold_targets = reference_efolds

end # author_inflation
