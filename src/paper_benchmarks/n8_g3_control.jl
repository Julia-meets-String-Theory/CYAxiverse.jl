"""Bounded G3 local control continuation for the source-twelve N=8 model.

The control is the audited positive Kähler direction
`t(k, alpha)=sqrt(k)*(T_REF + alpha*U)`, with period-one GLSM axions and
the approved P96/CYTools metric.  This module is local-control evidence only;
it does not change the radial G2 classifier or make an off-ray population
claim.
"""

const _G3_T_REF = Rational{BigInt}[1, 4, 4, -2, 4, 3, 3, 3]
const _G3_U = Rational{BigInt}[0, 1, 2, -1, 1, 1, 1, 1]
const _G3_ALPHA_MAX = Rational{BigInt}(1, 20)

const _G3_KAPPA_ENTRIES = (
    (0,0,0,-2), (0,0,3,-1), (0,0,4,-1), (0,0,5,1), (0,0,6,1),
    (0,1,1,-2), (0,1,5,2), (0,1,7,2), (0,2,2,-2), (0,2,6,2),
    (0,2,7,2), (0,3,3,-1), (0,4,4,-1), (0,4,5,1), (0,4,6,1),
    (0,5,5,-3), (0,5,6,1), (0,6,6,-3), (0,7,7,-4),
    (1,1,1,-1), (1,1,3,-1), (1,1,4,-1), (1,1,5,1), (1,1,7,-1),
    (1,3,3,-1), (1,3,7,1), (1,4,4,-1), (1,4,5,2), (1,4,7,1),
    (1,5,5,-3), (1,5,7,1), (1,7,7,3),
    (2,2,2,-1), (2,2,3,-1), (2,2,4,-1), (2,2,6,1), (2,2,7,-1),
    (2,3,3,-1), (2,3,7,1), (2,4,4,-1), (2,4,6,2), (2,4,7,1),
    (2,6,6,-3), (2,6,7,1), (2,7,7,3),
    (3,3,3,-1), (3,7,7,-2),
    (4,4,4,-1), (4,4,5,1), (4,4,6,1), (4,5,5,-3), (4,5,6,1),
    (4,6,6,-3), (4,7,7,-2),
    (5,5,5,6), (5,5,6,-1), (5,5,7,-1), (5,6,6,-1), (5,6,7,1),
    (5,7,7,-1), (6,6,6,6), (6,6,7,-1), (6,7,7,-1), (7,7,7,-10),
)

const _G3_MORI = Int[
     0  0 -1  0 -1  0  2  1; 0  0  0  0  1  0 -1  0;
     0  0 -1 -1  0  0  0  1; 1  0  1  0  0  0 -1  0;
     1  1  0  1  0  0  0  0; 0 -1  0  0  0  1  0  1;
     1  1  0  0  1 -2  0  0; 0  0  0  0  1 -1  0  0;
     1  0  0  0  1 -1 -1  1; 0  0 -1  0  0  0  1  1;
     0  0  1  0  0  0  0 -1; 0  0  0  1  0  0  1  0;
     0  1  0  0  0 -1  1 -1; -1  0  0 -1  0  0  0  0;
     0  0  0  1  1  0  0  0; 0  0  0  1  0  1  0  0;
     0  1  1  0  0  0  0 -2; 0  1  0  0  1 -2  0  0;
     1  0  1  1  0  0  0  0; 1  0  0  0  0  0  0  0;
    -1  0  0  0 -1  1  1  0; 0 -1  0 -1  0  0  0  1;
     1  0  0  0  1 -1  0  0; 1  0  0  0  1  0 -1  0;
     0  0  1  0  0  0 -1  0; 0  0  1  0  0  1 -1 -1;
     0  0  0 -1  0  0  0  0; 1  0  0  1  0  0  1  0;
     1  0  0  1  0  0  0  1; 0  0  0  0 -1  1  1  0;
     0 -1  0  0 -1  2  0  1; 0  1  0  1  0  0  0  0;
     0  1  0  0  0  0  0 -1; 1  1  0  0  0 -1  0  0;
     1  0  1  0  1  0 -2  0; 0  0  1  1  0  0  0  0;
     0  0  1  0  1  0 -2  0; 1  0  0  1  0  1  0  0;
     0  1  0  0  0 -1  0  0;
]

function _g3_kappa(::Type{T}) where {T<:AbstractFloat}
    kappa = zeros(T, 8, 8, 8)
    for (i0, j0, k0, value) in _G3_KAPPA_ENTRIES
        i, j, k = i0 + 1, j0 + 1, k0 + 1
        for p in Set(((i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)))
            kappa[p...] = T(value)
        end
    end
    kappa
end

function _g3_geometry(k::T, alpha::T) where {T<:AbstractFloat}
    t_shape = T.(_G3_T_REF) .+ alpha .* T.(_G3_U)
    kappa = _g3_kappa(T)
    A = [sum(kappa[i,j,l] * t_shape[l] for l in 1:8) for i in 1:8, j in 1:8]
    tau_shape = [sum(kappa[i,j,l] * t_shape[j] * t_shape[l]
        for j in 1:8, l in 1:8) / T(2) for i in 1:8]
    volume_shape = sum(kappa[i,j,l] * t_shape[i] * t_shape[j] * t_shape[l]
        for i in 1:8, j in 1:8, l in 1:8) / T(6)
    tau = k .* tau_shape
    volume = k^(T(3)/T(2)) * volume_shape
    kinv_shape = T(4) .* (tau_shape * tau_shape' .- volume_shape .* A)
    metric = inv(Symmetric(kinv_shape)) ./ k^2
    Q = Matrix{T}(_N8_SOURCE_CHARGES')
    actions = k .* (Q' * tau_shape)
    curves = sqrt(k) .* (T.(_G3_MORI) * t_shape)
    prime_divisors = Q' * tau
    full_factor_scale = T(8) * T(π) / volume^2
    full_amplitudes = full_factor_scale .* actions .*
        exp.(-T(2) * T(π) .* actions)
    (; k, alpha, t_shape, tau_shape, tau, volume_shape, volume, A,
       kinv_shape, metric, Q, actions, curves, prime_divisors,
       full_amplitudes, full_factor_scale,
       normalized_amplitudes=full_amplitudes ./ maximum(full_amplitudes),
       source_coefficients=full_amplitudes)
end

function _g3_geometry_derivatives(k::T, alpha::T, geometry) where {T<:AbstractFloat}
    u = T.(_G3_U)
    tau_shape_alpha = geometry.A * u
    volume_shape_alpha = dot(geometry.tau_shape, u)
    dS_alpha = k .* (geometry.Q' * tau_shape_alpha)
    dlogV_alpha = volume_shape_alpha / geometry.volume_shape
    dlogamp_alpha = -T(2) * dlogV_alpha .+
        (one(T) ./ geometry.actions .- T(2) * T(π)) .* dS_alpha
    dlogamp_k = ((one(T) .- T(2) * T(π) .* geometry.actions) .- T(3)) ./ k
    max_index = argmax(geometry.full_amplitudes)
    dlogamp_alpha_norm = dlogamp_alpha .- dlogamp_alpha[max_index]
    dlogamp_k_norm = dlogamp_k .- dlogamp_k[max_index]
    (; dS_alpha, dlogV_alpha, dlogamp_alpha,
       dlogamp_k, amp_alpha=geometry.normalized_amplitudes .* dlogamp_alpha_norm,
       amp_k=geometry.normalized_amplitudes .* dlogamp_k_norm)
end

struct G3ContinuationStep{T<:AbstractFloat}
    theta::Vector{T}
    null_vector::Vector{T}
    k::T
    alpha::T
    gradient_residual::T
    null_residual::T
    augmented_residual::T
    full_gradient_residual::T
    full_null_residual::T
    normalization_residual::T
    metric_null_norm::T
    metric_min_eigenvalue::T
    volume::T
    min_curve_volume::T
    min_prime_divisor::T
    full_factor_scale::T
    min_action::T
    max_full_amplitude::T
    projected_d3::T
    projected_d4::T
    canonical_hessian_eigenvalues::Vector{T}
    canonical_null_count::Int
    transverse_min_eigenvalue::T
    converged::Bool
    iterations::Int
    ds::T
    branch_id::Int
    step_index::Int
    corrector_condition::T
    rejected_steps::Int
    corrector_method::Symbol
end

struct G3ContinuationResult{T<:AbstractFloat}
    steps::Vector{G3ContinuationStep{T}}
    branch_id::Int
    status::Symbol
    termination_reason::Symbol
    attempted_steps::Int
    accepted_steps::Int
    rejected_steps::Int
    source_data::Symbol
    metric_contract::Symbol
    control::Symbol
end

"""Find a local positive-alpha augmented seed from the radial event.

The radial event is singular when alpha is held fixed.  A signed null-mode
displacement at a small positive alpha supplies the two local unfolding
branches without a grid search; failure of either signed seed is retained as
a diagnostic status.
"""
function n8_g3_seed_from_radial(theta0::AbstractVector{<:Real},
        null0::AbstractVector{<:Real}, k0::Real; alpha_seed::Real=1e-8,
        displacement::Real=1e-4, side::Int=-1, tolerance::Real=1e-12,
        max_iterations::Int=120, k_bounds::Tuple{Real,Real}=(0.5, 0.9))
    T = Float64
    alpha = T(alpha_seed)
    theta = T.(theta0) .+ T(side * displacement) .* T.(null0)
    null_vector = T.(null0)
    null_vector ./= norm(null_vector)
    k = T(k0)
    for iter in 1:max_iterations
        geometry = _g3_geometry(k, alpha)
        derivatives = _g3_geometry_derivatives(k, alpha, geometry)
        system = _g3_augmented_system(theta, null_vector, k, alpha, geometry, derivatives)
        residual = norm(system.residual, Inf)
        if residual <= T(tolerance)
            return (; theta=mod.(copy(theta), one(T)), null_vector=copy(null_vector),
                k, alpha, converged=true, iterations=iter, residual,
                side, displacement, status=:seed_converged,
                source=:radial_event_null_displacement)
        end
        jacobian = system.jacobian[:, 1:17]
        correction = try jacobian \ (-system.residual)
        catch; qr(jacobian) \ (-system.residual) end
        old = residual
        accepted = false
        step = one(T)
        while step >= T(2)^(-14)
            candidate = vcat(theta, null_vector, k) .+ step .* correction
            if candidate[end] < T(k_bounds[1]) || candidate[end] > T(k_bounds[2])
                step *= T(0.5); continue
            end
            cg = _g3_geometry(candidate[end], alpha)
            cd = _g3_geometry_derivatives(candidate[end], alpha, cg)
            cs = _g3_augmented_system(candidate[1:8], candidate[9:16],
                candidate[end], alpha, cg, cd)
            if norm(cs.residual, Inf) < old
                theta .= candidate[1:8]
                null_vector .= candidate[9:16]
                k = candidate[end]
                accepted = true
                break
            end
            step *= T(0.5)
        end
        accepted || return (; theta=mod.(copy(theta), one(T)), null_vector=copy(null_vector),
            k, alpha, converged=false, iterations=iter, residual,
            side, displacement, status=:seed_failed,
            source=:radial_event_null_displacement)
    end
    geometry = _g3_geometry(k, alpha)
    derivatives = _g3_geometry_derivatives(k, alpha, geometry)
    system = _g3_augmented_system(theta, null_vector, k, alpha, geometry, derivatives)
    (; theta=mod.(copy(theta), one(T)), null_vector=copy(null_vector), k, alpha,
       converged=false, iterations=max_iterations, residual=norm(system.residual, Inf),
       side, displacement, status=:seed_max_iterations,
       source=:radial_event_null_displacement)
end

function _g3_augmented_system(theta::AbstractVector{T}, null_vector::AbstractVector{T},
        k::T, alpha::T, geometry, derivatives) where {T<:AbstractFloat}
    two_pi = T(2) * T(π)
    args = two_pi .* (geometry.Q' * theta)
    sine, cosine = sin.(args), cos.(args)
    # A common positive amplitude factor leaves the stationary and null
    # equations unchanged.  Normalize it here so the augmented Jacobian is
    # well conditioned; full Eq. (19) amplitudes remain in every diagnostic.
    amp = geometry.normalized_amplitudes
    gradient = two_pi .* geometry.Q * (amp .* sine)
    hessian = two_pi^2 .* geometry.Q * Diagonal(amp .* cosine) * geometry.Q'
    gradient_k = two_pi .* geometry.Q * (derivatives.amp_k .* sine)
    gradient_alpha = two_pi .* geometry.Q * (derivatives.amp_alpha .* sine)
    hessian_k = two_pi^2 .* geometry.Q * Diagonal(derivatives.amp_k .* cosine) * geometry.Q'
    hessian_alpha = two_pi^2 .* geometry.Q * Diagonal(derivatives.amp_alpha .* cosine) * geometry.Q'
    qv = geometry.Q' * null_vector
    tensor_theta = zeros(T, 8, 8)
    for i in 1:8, l in 1:8
        tensor_theta[i,l] = -two_pi^3 * sum(amp .* sine .* geometry.Q[i,:] .*
            geometry.Q[l,:] .* qv)
    end
    residual = vcat(gradient, hessian * null_vector,
        dot(null_vector, null_vector) - one(T))
    jacobian = zeros(T, 17, 18)
    jacobian[1:8, 1:8] .= hessian
    jacobian[1:8, 17] .= gradient_k
    jacobian[1:8, 18] .= gradient_alpha
    jacobian[9:16, 1:8] .= tensor_theta
    jacobian[9:16, 9:16] .= hessian
    jacobian[9:16, 17] .= hessian_k * null_vector
    jacobian[9:16, 18] .= hessian_alpha * null_vector
    jacobian[17, 9:16] .= T(2) .* null_vector
    (; residual, jacobian, gradient, hessian, args, sine, cosine,
       gradient_k, gradient_alpha, hessian_k, hessian_alpha)
end

function _g3_step_diagnostics(theta, null_vector, k, alpha, geometry, system)
    metric_eigenvalues = eigvals(Symmetric(geometry.metric))
    metric_null_norm = sqrt(dot(null_vector, geometry.metric * null_vector))
    vmetric = null_vector ./ metric_null_norm
    qv = geometry.Q' * vmetric
    two_pi = eltype(theta)(2) * eltype(theta)(π)
    projected_d3 = -(two_pi)^3 * sum(geometry.full_amplitudes .* system.sine .* qv.^3)
    projected_d4 = -(two_pi)^4 * sum(geometry.full_amplitudes .* system.cosine .* qv.^4)
    normalized_gradient_residual = norm(system.gradient, Inf)
    normalized_null_residual = norm(system.hessian * null_vector, Inf)
    normalized_augmented_residual = norm(system.residual, Inf)
    full_gradient = two_pi .* geometry.Q * (geometry.full_amplitudes .* system.sine)
    full_hessian = two_pi^2 .* geometry.Q *
        Diagonal(geometry.full_amplitudes .* system.cosine) * geometry.Q'
    full_gradient_residual = norm(full_gradient, Inf)
    full_null_residual = norm(full_hessian * null_vector, Inf)
    metric_factor = cholesky(Symmetric(geometry.metric))
    canonical_hessian = metric_factor.L \ system.hessian / metric_factor.U
    canonical_hessian = (canonical_hessian + canonical_hessian') / eltype(theta)(2)
    canonical_hessian_eigenvalues = eigvals(Symmetric(canonical_hessian))
    # Use a precision-aware reporting floor.  The Float64 continuation has
    # residual-scale near-null eigenvalues around 1e-9, while high-precision
    # refinements drive the same eigenvalue below the working epsilon.  The
    # fourth-root floor separates that mode from the observed O(1e-2)
    # transverse spectrum at both precisions.
    canonical_null_cutoff = max(eltype(theta)(4) * sqrt(eps(eltype(theta))),
        sqrt(sqrt(eps(eltype(theta)))))
    canonical_null_count = count(abs.(canonical_hessian_eigenvalues) .<= canonical_null_cutoff)
    transverse = canonical_hessian_eigenvalues[
        abs.(canonical_hessian_eigenvalues) .> canonical_null_cutoff]
    transverse_min_eigenvalue = isempty(transverse) ? zero(eltype(theta)) : minimum(transverse)
    (; metric_min_eigenvalue=minimum(metric_eigenvalues), metric_null_norm,
       volume=geometry.volume, min_curve_volume=minimum(geometry.curves),
       min_prime_divisor=minimum(geometry.prime_divisors),
       full_factor_scale=geometry.full_factor_scale,
       min_action=minimum(geometry.actions),
       max_full_amplitude=maximum(geometry.full_amplitudes), projected_d3,
       projected_d4, normalized_gradient_residual, normalized_null_residual,
       normalized_augmented_residual, full_gradient_residual,
       full_null_residual, canonical_hessian_eigenvalues, canonical_null_count,
       canonical_null_cutoff, transverse_min_eigenvalue)
end

"""Continue the local augmented degeneracy equations by alpha predictor/corrector.

Alpha is the bounded continuation coordinate.  At each target alpha, the
17-equation augmented system is solved in `(theta, null_vector, k)` using a
predictor from the analytic alpha column and a damped Newton corrector.
"""
function n8_g3_predictor_corrector(theta0::AbstractVector{<:Real},
        null0::AbstractVector{<:Real}, k0::Real; alpha0::Real=0,
        ds::Real=1e-3, n_steps::Int=50, alpha_bounds::Tuple{Real,Real}=(0, 1/20),
        k_bounds::Tuple{Real,Real}=(0.5, 0.9), tolerance::Real=1e-10,
        max_corrector_iterations::Int=40, min_ds::Real=1e-7,
        branch_id::Int=1, initial_direction::Real=1)
    T = Float64
    k, alpha = T(k0), T(alpha0)
    theta, null_vector = T.(theta0), T.(null0)
    lower_k, upper_k = T(k_bounds[1]), T(k_bounds[2])
    lower_alpha, upper_alpha = T(alpha_bounds[1]), T(alpha_bounds[2])
    if !all(isfinite, theta) || !all(isfinite, null_vector) ||
            !isfinite(k) || !isfinite(alpha) || norm(null_vector) == 0 ||
            k <= 0 || alpha < lower_alpha || alpha > upper_alpha ||
            k < lower_k || k > upper_k
        return G3ContinuationResult{T}(G3ContinuationStep{T}[], branch_id,
            :invalid_initial_state, :invalid_initial_state, 0, 0, 0,
            :exact_integer_rational_table1, :P96_CYTools, :source12_positive_alpha)
    end
    null_vector ./= norm(null_vector)
    geometry = _g3_geometry(k, alpha)
    derivatives = _g3_geometry_derivatives(k, alpha, geometry)
    initial = _g3_augmented_system(theta, null_vector, k, alpha, geometry, derivatives)
    diagnostics = _g3_step_diagnostics(theta, null_vector, k, alpha, geometry, initial)
    initial_condition = try cond(initial.jacobian[:, 1:17]) catch; T(Inf) end
    initial_converged = diagnostics.normalized_augmented_residual <= T(tolerance)
    steps = G3ContinuationStep{T}[]
    push!(steps, G3ContinuationStep{T}(copy(theta), copy(null_vector), k, alpha,
        norm(initial.gradient, Inf), norm(initial.hessian * null_vector, Inf),
        diagnostics.normalized_augmented_residual,
        diagnostics.full_gradient_residual, diagnostics.full_null_residual,
        abs(initial.residual[end]), diagnostics.metric_null_norm,
        diagnostics.metric_min_eigenvalue, diagnostics.volume,
        diagnostics.min_curve_volume, diagnostics.min_prime_divisor,
        diagnostics.full_factor_scale, diagnostics.min_action,
        diagnostics.max_full_amplitude,
        diagnostics.projected_d3, diagnostics.projected_d4,
        copy(diagnostics.canonical_hessian_eigenvalues),
        diagnostics.canonical_null_count, diagnostics.transverse_min_eigenvalue,
        initial_converged, 0, zero(T),
        branch_id, 0, initial_condition, 0, :initial))

    if !initial_converged
        return G3ContinuationResult{T}(steps, branch_id, :invalid_initial_state,
            :invalid_initial_state, 0, 0, 0,
            :exact_integer_rational_table1, :P96_CYTools, :source12_positive_alpha)
    end

    tangent_z = -(initial.jacobian[:, 1:17] \ initial.jacobian[:, 18])
    direction = initial_direction == 0 ? one(T) : sign(T(initial_direction))
    tangent_z .*= direction

    current_ds = direction * abs(T(ds))
    termination_reason = :running
    attempted_steps = 0
    total_rejected = 0
    for step_index in 1:n_steps
        attempted_steps = step_index
        target_alpha = alpha + current_ds
        if target_alpha < lower_alpha || target_alpha > upper_alpha
            termination_reason = :bounds_reached
            break
        end
        if k <= lower_k + T(1e-3) || k >= upper_k - T(1e-3)
            termination_reason = :k_bounds_reached
            break
        end
        # Predictor: first order response of (theta,v,k) to alpha.
        geom_now = _g3_geometry(k, alpha)
        deriv_now = _g3_geometry_derivatives(k, alpha, geom_now)
        sys_now = _g3_augmented_system(theta, null_vector, k, alpha, geom_now, deriv_now)
        dz_dalpha = -(sys_now.jacobian[:, 1:17] \ sys_now.jacobian[:, 18])
        state = vcat(theta, null_vector, k) .+ current_ds .* dz_dalpha
        converged, iterations, rejected = false, 0, 0
        condition = T(Inf)
        for iter in 1:max_corrector_iterations
            iterations = iter
            th, vv, kk = state[1:8], state[9:16], state[17]
            geom = _g3_geometry(kk, target_alpha)
            deriv = _g3_geometry_derivatives(kk, target_alpha, geom)
            sys = _g3_augmented_system(th, vv, kk, target_alpha, geom, deriv)
            condition = cond(sys.jacobian[:, 1:17])
            residual = sys.residual
            if norm(residual, Inf) <= T(tolerance)
                converged = true
                break
            end
            correction = try
                sys.jacobian[:, 1:17] \ (-residual)
            catch
                qr(sys.jacobian[:, 1:17]) \ (-residual)
            end
            old = norm(residual, Inf)
            accepted = false
            step = one(T)
            while step >= T(2)^(-12)
                candidate = state .+ step .* correction
                if candidate[17] < lower_k || candidate[17] > upper_k
                    step *= T(0.5); rejected += 1; continue
                end
                cgeom = _g3_geometry(candidate[17], target_alpha)
                cderiv = _g3_geometry_derivatives(candidate[17], target_alpha, cgeom)
                csys = _g3_augmented_system(candidate[1:8], candidate[9:16],
                    candidate[17], target_alpha, cgeom, cderiv)
                new = norm(csys.residual, Inf)
                if new < old
                    state .= candidate; accepted = true; break
                end
                step *= T(0.5); rejected += 1
            end
            accepted || break
        end
        if !converged
            total_rejected += rejected
            current_ds *= T(0.5)
            if abs(current_ds) < T(min_ds) || step_index == n_steps
                termination_reason = :step_failed
                break
            end
            continue
        end
        total_rejected += rejected
        theta .= state[1:8]; null_vector .= state[9:16]; null_vector ./= norm(null_vector)
        k, alpha = state[17], target_alpha
        geom = _g3_geometry(k, alpha)
        deriv = _g3_geometry_derivatives(k, alpha, geom)
        sys = _g3_augmented_system(theta, null_vector, k, alpha, geom, deriv)
        diagnostics = _g3_step_diagnostics(theta, null_vector, k, alpha, geom, sys)
        push!(steps, G3ContinuationStep{T}(copy(theta), copy(null_vector), k, alpha,
            norm(sys.gradient, Inf), norm(sys.hessian * null_vector, Inf),
            norm(sys.residual, Inf), diagnostics.full_gradient_residual,
            diagnostics.full_null_residual,
            abs(sys.residual[end]), diagnostics.metric_null_norm,
            diagnostics.metric_min_eigenvalue, diagnostics.volume,
            diagnostics.min_curve_volume, diagnostics.min_prime_divisor,
            diagnostics.full_factor_scale, diagnostics.min_action,
            diagnostics.max_full_amplitude,
            diagnostics.projected_d3, diagnostics.projected_d4,
            copy(diagnostics.canonical_hessian_eigenvalues),
            diagnostics.canonical_null_count, diagnostics.transverse_min_eigenvalue,
            true, iterations,
            current_ds, branch_id, step_index, condition, rejected, :predictor_corrector))
        iterations <= 5 && (current_ds = direction * min(abs(current_ds) * 1.2, T(0.01)))
    end
    termination_reason == :running && (termination_reason = :max_steps)
    status = termination_reason in (:bounds_reached, :k_bounds_reached,
        :step_failed, :invalid_initial_state, :max_steps) ? termination_reason :
        :max_steps
    G3ContinuationResult{T}(steps, branch_id, status, termination_reason,
        attempted_steps, length(steps) - 1, total_rejected,
        :exact_integer_rational_table1, :P96_CYTools, :source12_positive_alpha)
end

"""Refine one fixed-alpha augmented solution with exact source geometry."""
function n8_g3_bigfloat_augmented_solve(theta_seed::AbstractVector{<:Real},
        null_seed::AbstractVector{<:Real}, k_seed::Real, alpha::Real;
        precision_bits::Int=256, tolerance::Real=BigFloat("1e-60"),
        max_iterations::Int=300)
    precision_bits >= 128 || throw(ArgumentError("precision_bits must be >= 128"))
    setprecision(BigFloat, precision_bits) do
        T = BigFloat
        theta = T.(theta_seed)
        null_vector = T.(null_seed)
        null_vector ./= norm(null_vector)
        k = T(k_seed)
        alpha_b = T(alpha)
        for iter in 1:max_iterations
            geometry = _g3_geometry(k, alpha_b)
            derivatives = _g3_geometry_derivatives(k, alpha_b, geometry)
            system = _g3_augmented_system(theta, null_vector, k, alpha_b,
                geometry, derivatives)
            residual = norm(system.residual, Inf)
            if residual <= T(tolerance)
                diagnostics = _g3_step_diagnostics(theta, null_vector, k, alpha_b,
                    geometry, system)
                return merge(diagnostics, (; theta=mod.(copy(theta), one(T)),
                    null_vector=copy(null_vector), k, alpha=alpha_b,
                    gradient_residual=diagnostics.normalized_gradient_residual,
                    null_residual=diagnostics.normalized_null_residual,
                    augmented_residual=diagnostics.normalized_augmented_residual,
                    normalization_residual=abs(system.residual[end]),
                    converged=true, iterations=iter, precision_bits,
                    source_precision_bits=precision_bits,
                    source_data=:exact_integer_rational_table1,
                    metric_contract=:P96_CYTools, full_factor=true,
                    action_vector=copy(geometry.actions),
                    amplitudes=copy(geometry.full_amplitudes),
                    tau=copy(geometry.tau), volume=geometry.volume,
                    metric=copy(geometry.metric)))
            end
            jacobian = system.jacobian[:, 1:17]
            correction = try jacobian \ (-system.residual)
            catch; qr(jacobian) \ (-system.residual) end
            old = residual
            accepted = false
            step = one(T)
            while step >= T(2)^(-14)
                candidate = vcat(theta, null_vector, k) .+ step .* correction
                if candidate[end] <= zero(T)
                    step *= T(0.5); continue
                end
                cg = _g3_geometry(candidate[end], alpha_b)
                cd = _g3_geometry_derivatives(candidate[end], alpha_b, cg)
                cs = _g3_augmented_system(candidate[1:8], candidate[9:16],
                    candidate[end], alpha_b, cg, cd)
                if norm(cs.residual, Inf) < old
                    theta .= candidate[1:8]
                    null_vector .= candidate[9:16]
                    k = candidate[end]
                    accepted = true
                    break
                end
                step *= T(0.5)
            end
            accepted || break
        end
        geometry = _g3_geometry(k, alpha_b)
        derivatives = _g3_geometry_derivatives(k, alpha_b, geometry)
        system = _g3_augmented_system(theta, null_vector, k, alpha_b,
            geometry, derivatives)
        diagnostics = _g3_step_diagnostics(theta, null_vector, k, alpha_b,
            geometry, system)
        merge(diagnostics, (; theta=mod.(copy(theta), one(T)), null_vector=copy(null_vector),
            k, alpha=alpha_b, gradient_residual=diagnostics.normalized_gradient_residual,
            null_residual=diagnostics.normalized_null_residual,
            augmented_residual=diagnostics.normalized_augmented_residual,
            normalization_residual=abs(system.residual[end]), converged=false,
            iterations=max_iterations, precision_bits,
            source_precision_bits=precision_bits,
            source_data=:exact_integer_rational_table1,
            metric_contract=:P96_CYTools, full_factor=true))
    end
end

function n8_g3_local_diagnostics(theta::AbstractVector{<:Real}, null_vector::AbstractVector{<:Real},
        k::Real, alpha::Real; precision_bits::Int=256)
    precision_bits >= 128 || throw(ArgumentError("precision_bits must be >= 128"))
    setprecision(BigFloat, precision_bits) do
        T = BigFloat
        geometry = _g3_geometry(T(k), T(alpha))
        derivatives = _g3_geometry_derivatives(T(k), T(alpha), geometry)
        theta_b = T.(theta); null_b = T.(null_vector); null_b ./= norm(null_b)
        system = _g3_augmented_system(theta_b, null_b, T(k), T(alpha), geometry, derivatives)
        diagnostics = _g3_step_diagnostics(theta_b, null_b, T(k), T(alpha), geometry, system)
        merge(diagnostics, (; theta=mod.(theta_b, one(T)), null_vector=copy(null_b),
            k=T(k), alpha=T(alpha), gradient_residual=diagnostics.normalized_gradient_residual,
            null_residual=diagnostics.normalized_null_residual,
            augmented_residual=diagnostics.normalized_augmented_residual,
            normalization_residual=abs(system.residual[end]),
            source_precision_bits=precision_bits, source_data=:exact_integer_rational_table1,
            metric_contract=:P96_CYTools, full_factor=true,
            action_vector=copy(geometry.actions), amplitudes=copy(geometry.full_amplitudes),
            tau=copy(geometry.tau), volume=geometry.volume, metric=copy(geometry.metric),
            trial_geometry=geometry))
    end
end
