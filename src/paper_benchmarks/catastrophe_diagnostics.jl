const CATASTROPHE_BENCHMARK_SCHEMA = "catastrophic-inflation-benchmark-v1"

const PAPER_SOURCE_IDENTITY = (
    identifier="arXiv:2608.14780v1",
    title="Catastrophic Inflation in the Axiverse",
    url="https://arxiv.org/abs/2608.14780",
    source_sha256="b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6",
)

const VOLUME_SCALING_CONVENTION = (
    divisor_volumes=:linear,
    cy_volume=:three_halves,
    kinetic_metric=:inverse_square,
    charges=:fixed_integer_basis,
    phases=:fixed_additive_arguments,
    scale_status=:benchmark_convention,
    physical_status=:not_established,
)

function _benchmark_input_digest(values...)
    bytes2hex(sha256(join(repr.(values), "\n")))
end

"""
    benchmark_source_identity()

Return the immutable paper identity used by the benchmark fixtures.
"""
benchmark_source_identity() = PAPER_SOURCE_IDENTITY

"""
    benchmark_manifest()

Return a replayable manifest for the two named paper examples.  The manifest
keeps the instanton action list separate from the divisor-volume Kähler point
and records the fixed-saxion scientific boundary explicitly.
"""
function benchmark_manifest()
    n5 = author_inflation.n5_geometry()
    n8 = author_inflation.n8_geometry()
    n5_q = Matrix(author_inflation.N5_Q')
    n8_q = Matrix(author_inflation.N8_Q')
    n8_q_trajectory = Matrix(author_inflation.N8_Q_TRAJECTORY')
    (; schema_version=CATASTROPHE_BENCHMARK_SCHEMA,
        source=PAPER_SOURCE_IDENTITY,
        source_revision=PAPER_SOURCE_IDENTITY.identifier,
        selection_route=:paper_named_appendix_fixtures,
        counting_unit=:geometry,
        coordinate_convention=(
            raw_angles=(:radian, :coordinate_vector),
            canonical_fields=(:M_Pl, :coordinate_vector),
            charges=:axions_are_rows_in_package_Q,
        ),
        volume_scaling=VOLUME_SCALING_CONVENTION,
        diagnostic_tolerances=(
            gradient=1e-8, hessian=1e-8, derivative=1e-8,
            precision_bits=53,
        ),
        precision_rerun=(enabled=true, precision_bits=120),
        examples=(
            n5=(h11=n5.h11, h21=n5.h21, euler=n5.euler,
                volume=n5.volume, vertices=copy(n5.vertices),
                divisor_volumes=copy(n5.divisor_volumes),
                Q=n5_q, qdotτ=copy(author_inflation.N5_QDOTTAU),
                phases=zeros(length(author_inflation.N5_QDOTTAU)),
                kinetic=Matrix(n5.kinetic),
                triangulation=(
                    status=:source_reconstructed,
                    witness=:appendix_b_vertices,
                    witness_digest=_benchmark_input_digest(n5.vertices),
                )),
            n8=(h11=n8.h11, h21=n8.h21, euler=n8.euler,
                volume=n8.volume, vertices=copy(n8.vertices),
                divisor_volumes=copy(n8.divisor_volumes),
                instanton_actions=copy(author_inflation.N8_INSTANTON_ACTIONS),
                Q=n8_q, Q_trajectory=n8_q_trajectory,
                qdotτ=copy(author_inflation.N8_TAU),
                qdotτ_trajectory=copy(author_inflation.N8_TAU_TRAJECTORY),
                phases=zeros(length(author_inflation.N8_TAU)),
                phases_trajectory=zeros(length(author_inflation.N8_TAU_TRAJECTORY)),
                kinetic=Matrix(n8.kinetic),
                triangulation=(
                    status=:source_reconstructed,
                    witness=:appendix_c_vertices,
                    witness_digest=_benchmark_input_digest(n8.vertices),
                )),
        ),
        input_digest=_benchmark_input_digest(
            n5_q, author_inflation.N5_QDOTTAU, n5.vertices, n5.kinetic,
            n8_q, author_inflation.N8_TAU, n8.vertices, n8.kinetic,
        ),
        phase_examples=(
            n5=(status=:parameterized,
                source_note="The paper supplies a nonzero phase parameter; callers must provide the published phase vector."),
            n8=(status=:parameterized,
                delta=0.04,
                source_note="The paper reports δ=0.04; the complete per-instanton vector is an explicit fixture input."),
        ),
        claim_boundary=(
            fixed_saxions=true,
            complete_nonperturbative_superpotential=false,
            moduli_stabilization=:not_established,
        ))
end

function _catastrophe_impl(theta, charges, amplitudes, metric, phases;
        argument_scale, tolerance, gradient_tolerance, hessian_tolerance,
        derivative_tolerance, null_direction, precision_bits)
    T = promote_type(Float64, eltype(theta), eltype(charges), eltype(amplitudes),
        eltype(metric), eltype(phases), typeof(argument_scale))
    T <: AbstractFloat || throw(ArgumentError("catastrophe inputs must be real"))
    θ = T.(theta)
    q = T.(charges)
    a = T.(amplitudes)
    g = Matrix{T}(metric)
    phase = T.(phases)
    n, p = size(q)
    length(θ) == n || throw(DimensionMismatch("theta and charge dimensions differ"))
    length(a) == p || throw(DimensionMismatch("amplitudes and charge columns differ"))
    length(phase) == p || throw(DimensionMismatch("one phase is required per instanton"))
    size(g) == (n, n) || throw(DimensionMismatch("kinetic metric must be square"))
    issymmetric(g) || throw(ArgumentError("kinetic metric must be symmetric"))

    metric_eigen = eigen(Symmetric(g))
    all(>(zero(T)), metric_eigen.values) ||
        throw(ArgumentError("kinetic metric must be positive definite"))
    canonical_to_raw = metric_eigen.vectors *
        Diagonal(inv.(sqrt.(metric_eigen.values))) * metric_eigen.vectors'
    arguments = T(argument_scale) .* (q' * θ) .+ phase
    sine = sin.(arguments)
    cosine = cos.(arguments)
    value = sum(a .* (one(T) .- cosine))
    gradient_raw = T(argument_scale) .* q * (a .* sine)
    hessian_raw = T(argument_scale)^2 .* q * Diagonal(a .* cosine) * q'
    gradient = canonical_to_raw' * gradient_raw
    hessian = canonical_to_raw' * hessian_raw * canonical_to_raw
    hessian_eigen = eigen(Symmetric(hessian))
    hessian_scale = max(maximum(abs, a), maximum(abs, hessian_eigen.values))
    hessian_scale = hessian_scale == zero(T) ? eps(T) : hessian_scale
    null_cutoff = T(hessian_tolerance) * hessian_scale
    near_null_indices = findall(abs.(hessian_eigen.values) .<= null_cutoff)
    index = if null_direction === nothing
        argmin(abs.(hessian_eigen.values))
    else
        length(null_direction) == n ||
            throw(DimensionMismatch("null direction has the wrong dimension"))
        direction = T.(null_direction)
        norm(direction) > zero(T) ||
            throw(ArgumentError("null direction must be nonzero"))
        direction ./= norm(direction)
        argmax(abs.(hessian_eigen.vectors' * direction))
    end
    canonical_direction = if null_direction === nothing
        copy(hessian_eigen.vectors[:, index])
    else
        direction = T.(null_direction)
        direction ./ norm(direction)
    end
    canonical_to_raw_direction = canonical_to_raw * canonical_direction
    canonical_charge = T(argument_scale) .* (canonical_to_raw' * q)
    projected_charge = canonical_charge' * canonical_direction
    projected_second = dot(canonical_direction, hessian * canonical_direction)
    projected_third = -sum(a .* sine .* projected_charge.^3)
    projected_fourth = -sum(a .* cosine .* projected_charge.^4)
    canonical_gradient_residual = norm(gradient, Inf)
    derivative_scale = max(maximum(abs, a), abs(value),
        maximum(abs, hessian_eigen.values), abs(projected_third),
        abs(projected_fourth))
    derivative_scale = derivative_scale == zero(T) ? eps(T) : derivative_scale
    derivative_cutoff = T(derivative_tolerance) * derivative_scale
    stationary = canonical_gradient_residual <=
        T(gradient_tolerance) * derivative_scale
    unique_null = length(near_null_indices) == 1
    classification = if !stationary || !unique_null
        :unresolved
    elseif abs(projected_second) > derivative_cutoff
        :unresolved
    elseif abs(projected_third) > derivative_cutoff
        :fold
    elseif abs(projected_fourth) > derivative_cutoff
        :cusp
    else
        :unresolved
    end
    normal_form = classification === :cusp ?
        (projected_fourth < zero(T) ? :quartic_hilltop : :quartic) :
        classification === :fold ? :cubic_shoulder : :unresolved
    transverse_indices = [i for i in eachindex(hessian_eigen.values)
        if !(i in near_null_indices)]
    transverse_eigenvalues = hessian_eigen.values[transverse_indices]
    (; classification, normal_form,
        is_stationary=stationary, higher_dimensional=length(near_null_indices) > 1,
        potential=value, gradient=gradient, hessian=hessian,
        canonical_direction, raw_direction=canonical_to_raw_direction,
        near_null_index=index, near_null_eigenvalue=hessian_eigen.values[index],
        near_null_eigenvalues=hessian_eigen.values[near_null_indices],
        hessian_eigenvalues=hessian_eigen.values,
        transverse_hessian_eigenvalues=transverse_eigenvalues,
        projected_derivatives=(
            second=projected_second, third=projected_third, fourth=projected_fourth,
        ),
        projected_gradient=dot(canonical_direction, gradient),
        gradient_residual=canonical_gradient_residual,
        null_cutoff, derivative_cutoff,
        precision_bits, tolerance=T(tolerance),
        gradient_tolerance=T(gradient_tolerance),
        hessian_tolerance=T(hessian_tolerance),
        derivative_tolerance=T(derivative_tolerance),
        argument_scale=T(argument_scale), phases=phase)
end

"""
    local_catastrophe_diagnostic(theta, Q, amplitudes, metric; kwargs...)

Classify a stationary point using the canonical near-null direction and its
projected second, third, and fourth derivatives.  `Q` has axions in rows and
instantons in columns; `amplitudes` are the signed coefficients in
`sum(amplitude * (1 - cos(argument_scale * Q' * theta + phase)))`.
"""
function local_catastrophe_diagnostic(theta::AbstractVector, Q::AbstractMatrix,
        amplitudes::AbstractVector, metric::AbstractMatrix;
        phases=nothing, argument_scale::Real=1, precision_bits::Int=53,
        tolerance::Real=1e-8, gradient_tolerance::Real=tolerance,
        hessian_tolerance::Real=tolerance,
        derivative_tolerance::Real=tolerance, null_direction=nothing)
    precision_bits >= 53 || throw(ArgumentError("precision_bits must be at least 53"))
    phase = phases === nothing ? zeros(length(amplitudes)) : phases
    if precision_bits > 53
        return setprecision(BigFloat, precision_bits) do
            _catastrophe_impl(BigFloat.(theta), BigFloat.(Q), BigFloat.(amplitudes),
                BigFloat.(metric), BigFloat.(phase);
                argument_scale=BigFloat(argument_scale),
                tolerance=BigFloat(tolerance),
                gradient_tolerance=BigFloat(gradient_tolerance),
                hessian_tolerance=BigFloat(hessian_tolerance),
                derivative_tolerance=BigFloat(derivative_tolerance),
                null_direction=null_direction === nothing ? nothing :
                    BigFloat.(null_direction),
                precision_bits=precision_bits)
        end
    end
    _catastrophe_impl(theta, Q, amplitudes, metric, phase;
        argument_scale, tolerance, gradient_tolerance, hessian_tolerance,
        derivative_tolerance, null_direction, precision_bits)
end

function local_catastrophe_diagnostic(theta::AbstractVector, Q::AbstractMatrix,
        L::AbstractMatrix, metric::AbstractMatrix; kwargs...)
    size(L, 1) == 2 || throw(DimensionMismatch("L must have two rows"))
    size(Q, 2) == size(L, 2) ||
        throw(DimensionMismatch("Q columns and L columns differ"))
    T = promote_type(Float64, eltype(L))
    amplitudes = T.(L[1, :]) .* T(10) .^ T.(L[2, :])
    local_catastrophe_diagnostic(theta, Q, amplitudes, metric; kwargs...)
end

const catastrophe_diagnostic = local_catastrophe_diagnostic
const classify_catastrophe = local_catastrophe_diagnostic

function n5_catastrophe_diagnostic(; k::Real=author_inflation.n5_critical_scale(),
        theta::AbstractVector{<:Real}=[π], phases=nothing,
        precision_bits::Int=53, tolerance::Real=1e-8,
        derivative_tolerance::Real=tolerance)
    phase = phases === nothing ? zeros(2) : phases
    ratio = author_inflation.n5_reduced_ratio(k)
    result = local_catastrophe_diagnostic(theta, reshape([1, 2], 1, 2),
        [1.0, ratio], reshape([1.0], 1, 1);
        phases=phase, argument_scale=1, precision_bits, tolerance,
        derivative_tolerance)
    merge(result, (; example=:n5, k=Float64(k),
        kc=author_inflation.n5_critical_scale(),
        source_identity=PAPER_SOURCE_IDENTITY.identifier))
end

function n8_catastrophe_diagnostic(; theta=nothing, k::Real=N8_KC,
        trajectory::Bool=true, phases=nothing, precision_bits::Int=53,
        tolerance::Real=1e-8, derivative_tolerance::Real=tolerance)
    if theta === nothing
        phases !== nothing && any(!iszero, phases) &&
            throw(ArgumentError("a nonzero-phase point must provide theta"))
        critical = author_inflation.n8_degenerate_point()
        theta = critical.theta
        k = critical.k
    end
    potential = author_inflation.n8_potential(
        k=k, trajectory=trajectory, phases=phases)
    metric = author_inflation.n8_kinetic_matrix(k)
    result = local_catastrophe_diagnostic(theta, potential.Q, potential.L, metric;
        phases=potential.phases, argument_scale=1, precision_bits, tolerance,
        derivative_tolerance)
    merge(result, (; example=:n8_poly102, k=Float64(k), kc=N8_KC,
        source_identity=PAPER_SOURCE_IDENTITY.identifier,
        trajectory_truncation=trajectory ? :ten_rows : :twelve_rows))
end

"""
    phase_fixture(example; delta=0.04, phases=nothing, instanton=1)

Build a deterministic nonzero-phase input while keeping the per-instanton
assignment explicit.  The paper reports the N=8 scalar `δ=0.04`, but a scalar
alone does not identify a complete phase vector; callers may therefore provide
the published vector or use the documented single-instanton probe.
"""
function phase_fixture(example::Symbol; delta::Real=0.04, phases=nothing,
        instanton::Int=1, trajectory::Bool=false)
    count = example === :n5 ? length(N5_QDOTTAU) :
        example === :n8 ? (trajectory ? length(N8_TAU_TRAJECTORY) : length(N8_TAU)) :
        throw(ArgumentError("example must be :n5 or :n8"))
    values = if phases === nothing
        1 <= instanton <= count ||
            throw(ArgumentError("instanton index is outside the benchmark"))
        result = zeros(Float64, count)
        result[instanton] = Float64(delta)
        result
    else
        Float64.(phases)
    end
    length(values) == count ||
        throw(DimensionMismatch("one phase is required per instanton"))
    (; example, trajectory, delta=Float64(delta), phases=values,
        assignment=phases === nothing ? :single_instanton_probe : :published_vector,
        phase_convention=:additive_argument_radians,
        source_identity=PAPER_SOURCE_IDENTITY.identifier)
end

function cumulative_turning(samples)
    length(samples) < 2 && return zero(Float64)
    all(hasproperty(sample, :tangent) for sample in samples) ||
        return NaN
    total = zero(promote_type(Float64, eltype(first(samples).tangent)))
    for (left, right) in zip(samples[1:end-1], samples[2:end])
        cosine = clamp(dot(left.tangent, right.tangent), -one(total), one(total))
        total += acos(cosine)
    end
    total
end

function trajectory_observables(trajectory; sample_index::Int=length(trajectory.samples))
    isempty(trajectory.samples) && throw(ArgumentError("trajectory has no samples"))
    1 <= sample_index <= length(trajectory.samples) ||
        throw(BoundsError(trajectory.samples, sample_index))
    sample = trajectory.samples[sample_index]
    epsilon = sample.epsilon
    eta_parallel = sample.eta_parallel
    potential = sample.potential
    delta_H = epsilon > 0 && potential > 0 ?
        sqrt(potential) / (5sqrt(6π * epsilon)) : oftype(potential, NaN)
    (; sample_index, epsilon, eta_parallel,
        n_s=1 - 6epsilon + 2eta_parallel,
        delta_H, scalar_amplitude=delta_H,
        scalar_amplitude_convention=:paper_delta_H,
        cumulative_turning=cumulative_turning(trajectory.samples),
        units=:M_Pl)
end
