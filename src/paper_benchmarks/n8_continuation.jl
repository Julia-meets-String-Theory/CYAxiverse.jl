"""N8 radial multifield catastrophe continuation (G2, P96 contract).

Pseudo-arclength continuation of stationary branches ∇V(θ,k)=0 for the
twelve-term Table 1 potential through the radial degeneracy. Uses hierarchy
preconditioning (same as the existing augmented solver) for numerical
stability. Eigenvalue monitoring uses P96 canonical metric K_θ=M96/k².

Convention: period-one GLSM θ, argument 2πQθ, metric K_θ=M96/k².
"""

struct N8ContinuationStep{T<:AbstractFloat}
    theta::Vector{T}
    k::T
    gradient_residual::T
    canonical_hessian_min::T
    canonical_hessian_eigenvalues::Vector{T}
    converged::Bool
    iterations::Int
    tangent_theta::Vector{T}
    tangent_k::T
    ds::T
    branch_id::Int
    step_index::Int
    conditioned_hessian_min::T
    corrector_method::Symbol
    bordered_rank::Int
    bordered_condition::T
    rejected_steps::Int
end

struct N8ContinuationResult{T<:AbstractFloat}
    steps::Vector{N8ContinuationStep{T}}
    branch_id::Int
    catastrophe_bracket::Union{Nothing,Tuple{Int,Int}}
    catastrophe_k::T
    catastrophe_theta::Union{Nothing,Vector{T}}
    status::Symbol
end

# Immutable source data used by the high-precision validation path.  Keeping
# the charges as integers and the actions as rationals prevents a Float64
# literal from being widened and mislabeled as recovered precision.
const _N8_SOURCE_CHARGES = Int[
    -1  1  1  0  0  0  0  1;  0  0  0  1  0  0  0  0
     0  0  0  0  1  0  0  0;  0  0  0  0  0  1  0  0
     0  0  0  0  0  0  1  0;  0 -1  1 -1  1  0  1  0
     0  1 -1 -1  1  1  0  0;  1  0  0 -1 -1  0  0  0
     0  0  1  0  0  0  0  0;  0  1  0  0  0  0  0  0
     0  0  0  0  0  0  0  1;  1  0  0  0  0  0  0  0
]
const _N8_SOURCE_ACTIONS = (
    14//1, 29//2, 29//2, 31//2, 31//2, 31//2,
    31//2, 16//1, 17//1, 17//1, 25//1, 45//1)

function _n8_exact_source_data(::Type{T}) where {T<:AbstractFloat}
    Matrix{T}(_N8_SOURCE_CHARGES'), T[T(a) for a in _N8_SOURCE_ACTIONS]
end

function _n8_amplitudes_and_dk(qdottau_ref::AbstractVector{T}, k::T) where {T<:AbstractFloat}
    scaled = k .* qdottau_ref
    amplitudes = scaled .* exp.(T(-2) * T(π) .* scaled)
    dk = amplitudes .* (one(T) .- T(2) * T(π) .* scaled) ./ k
    amplitudes, dk
end

function _n8_preconditioned_system(theta::AbstractVector{T}, k::T,
        qcanonical::AbstractMatrix{T}, qdottau_ref::AbstractVector{T},
        phases::AbstractVector{T}) where {T<:AbstractFloat}
    two_pi = T(2) * T(π)
    n = length(theta)
    amplitudes, amp_dk = _n8_amplitudes_and_dk(qdottau_ref, k)
    # Use one scalar normalization for the residual and its *true* Jacobian.
    # The earlier prototype divided each residual row by a different leading
    # amplitude but paired it with a symmetric, differently transformed
    # Hessian.  That matrix was not the derivative of the continuation
    # residual, so its null vector did not define a valid bordered system.
    # A scalar scale keeps the stationarity equations equivalent while
    # retaining the physical Hessian signature and a genuine dF/dk.
    log_amplitudes = log.(abs.(amplitudes))
    max_index = argmax(log_amplitudes)
    max_log = log_amplitudes[max_index]
    normalization_scale = exp(max_log)
    normalized_amp = amplitudes ./ normalization_scale
    dmax_log = amp_dk[max_index] / amplitudes[max_index]
    normalized_dk = (amp_dk .- amplitudes .* dmax_log) ./ normalization_scale
    row_scales = ones(T, n)
    arguments = two_pi .* (qcanonical' * theta) .+ phases
    sine = sin.(arguments)
    cosine = cos.(arguments)
    gradient_raw = two_pi .* qcanonical * (normalized_amp .* sine)
    gradient = gradient_raw
    hessian = two_pi^2 .* qcanonical * Diagonal(normalized_amp .* cosine) * qcanonical'
    gradient_dk = two_pi .* qcanonical * (normalized_dk .* sine)
    (; gradient, hessian, gradient_dk,
       amplitudes=normalized_amp, row_scales,
       arguments, sine, cosine, amplitude_log_scale=max_log,
       normalization_scale, max_amplitude_index=max_index)
end

function _n8_setup_leading_charge(k::T) where {T<:AbstractFloat}
    potential = _n8_potential(k=k)
    Q = potential.Q
    qdottau_ref = T.(potential.qdotτ)
    selected = LQtilde(Q, potential.L)
    Qordered = hcat(selected.Qtilde, selected.Qbar)
    qcanonical = Matrix{T}(selected.Qtilde) \ Matrix{T}(Qordered)
    # Charge vectors alone are not a term identity.  Preserve the source
    # action attached to each reordered (Q,L) pair returned by LQtilde.
    Lord = hcat(selected.Ltilde, selected.Lbar)
    ordered_qdotτ = T.(map(axes(Qordered, 2)) do i
        match = findfirst(j -> Q[:, j] == Qordered[:, i] &&
            isapprox(potential.L[:, j], Lord[:, i]; atol=T(1e-12), rtol=T(1e-12)),
            axes(Q, 2))
        match === nothing && throw(ArgumentError("LQtilde lost N8 term identity"))
        qdottau_ref[match]
    end)
    phases_ordered = zeros(T, size(Qordered, 2))
    Qtilde = Matrix{T}(selected.Qtilde)
    (; qcanonical, ordered_qdotτ, phases=phases_ordered,
       Qtilde, Q, Qordered, potential, selected)
end

function _n8_theta_to_leading(theta::AbstractVector{T},
        Qtilde::AbstractMatrix{T}) where {T<:AbstractFloat}
    Qtilde' * theta
end

function _n8_leading_to_theta(phi::AbstractVector{T},
        Qtilde::AbstractMatrix{T}) where {T<:AbstractFloat}
    Matrix{T}(Qtilde') \ phi
end

function _n8_canonical_hessian_eigenvalues(raw_hessian::AbstractMatrix{T},
        k::T) where {T<:AbstractFloat}
    geometry = n8_geometry()
    metric = T.(Matrix(geometry.kinetic)) ./ k^2
    factor = cholesky(Symmetric(metric))
    canonical = factor.L \ raw_hessian / factor.U
    eigvals(Symmetric(canonical))
end

function _n8_raw_hessian(theta::AbstractVector{T}, k::T,
        Q::AbstractMatrix{<:Integer}, qdottau_ref::AbstractVector{T},
        phases::AbstractVector{T}) where {T<:AbstractFloat}
    two_pi = T(2) * T(π)
    qfloat = T.(Q)
    amplitudes, _ = _n8_amplitudes_and_dk(qdottau_ref, k)
    arguments = two_pi .* (qfloat' * theta) .+ phases
    cosine = cos.(arguments)
    gradient = two_pi .* qfloat * (amplitudes .* sin.(arguments))
    hessian = two_pi^2 .* qfloat * Diagonal(amplitudes .* cosine) * qfloat'
    (; gradient, hessian, amplitudes)
end

"""
    n8_pseudo_arclength_continuation(theta0, k0; ds=5e-4, ...)

Pseudo-arclength continuation of ∇V(θ,k)=0 for the twelve-term N8 potential.
Works in hierarchy-preconditioned leading-charge coordinates for numerical
stability. Input and output use period-one GLSM coordinates.

Eigenvalues are monitored using the P96 canonical Hessian (K_θ^{-1/2} H K_θ^{-1/2})
and the hierarchy-conditioned Hessian (for catastrophe bracket detection).

P96 convention: period-one θ, argument 2πQθ, metric M96/k².
"""
function n8_pseudo_arclength_continuation(
        theta0::AbstractVector{<:Real}, k0::Real;
        ds::Real=5e-4, n_steps::Int=300,
        tolerance::Real=1e-10, max_corrector_iterations::Int=50,
        min_ds::Real=1e-8, max_ds::Real=5e-2,
        k_bounds::Tuple{Real,Real}=(0.5, 0.9),
        phases=nothing, branch_id::Int=1,
        ds_growth::Real=1.2, ds_shrink::Real=0.5,
        target_iterations::Int=5,
        initial_k_direction::Real=0)
    T = Float64
    # Scale the radial coordinate in the arclength metric.  The angular
    # coordinates are O(1), while the published event is localized in a
    # narrow k interval.  This keeps an initial ds from proposing a large
    # angular displacement with an effectively frozen k.
    k_arc_scale = T(1e-2)
    setup = _n8_setup_leading_charge(T(k0))
    n = 8

    phi = _n8_theta_to_leading(T.(theta0), setup.Qtilde)
    k = T(k0)

    sys = _n8_preconditioned_system(phi, k, setup.qcanonical,
        setup.ordered_qdotτ, setup.phases)
    grad_res = norm(sys.gradient, Inf)
    if grad_res > T(1e-4)
        @warn "initial point scaled gradient = $grad_res"
    end

    J = hcat(sys.hessian, sys.gradient_dk .* k_arc_scale)
    F = svd(J)
    tangent_full = F.Vt[end, :]
    tangent_phi = tangent_full[1:n]
    tangent_k = tangent_full[end] * k_arc_scale
    # `ds` chooses the direction along the oriented curve.  Keep the default
    # tangent oriented toward increasing k so a negative ds from above and a
    # positive ds from below both move toward the event.
    requested_direction = initial_k_direction == 0 ? one(T) :
        sign(T(initial_k_direction))
    if requested_direction != 0 && sign(tangent_k) != requested_direction
        tangent_phi .*= -one(T)
        tangent_k = -tangent_k
    end

    theta_glsm = _n8_leading_to_theta(phi, setup.Qtilde)
    raw = _n8_raw_hessian(theta_glsm, k, setup.Q, T.(setup.potential.qdotτ),
        zeros(T, size(setup.Q, 2)))
    can_eig = _n8_canonical_hessian_eigenvalues(raw.hessian, k)
    cond_eig = eigvals(Symmetric(sys.hessian))

    steps = N8ContinuationStep{T}[]
    tangent_theta_glsm = _n8_leading_to_theta(tangent_phi, setup.Qtilde)
    tangent_norm = sqrt(dot(tangent_theta_glsm, tangent_theta_glsm) + tangent_k^2)
    tangent_phi ./= tangent_norm
    tangent_theta_glsm ./= tangent_norm
    tangent_k /= tangent_norm
    push!(steps, N8ContinuationStep{T}(copy(theta_glsm), k, grad_res,
        minimum(can_eig), copy(can_eig), true, 0,
        copy(tangent_theta_glsm), tangent_k, zero(T), branch_id, 0,
        minimum(cond_eig), :initial, rank(J), T(NaN), 0))

    current_ds = T(ds)
    failure_encountered = false
    termination_reason = :running
    catastrophe_bracket = nothing
    catastrophe_k = T(NaN)
    catastrophe_theta = nothing

    for step_idx in 1:n_steps
        phi_pred = phi .+ current_ds .* tangent_phi
        k_pred = k + current_ds * tangent_k

        if k_pred < T(k_bounds[1]) || k_pred > T(k_bounds[2])
            termination_reason = :bounds_reached
            break
        end

        phi_try = copy(phi_pred)
        k_try = k_pred
        converged = false
        iters = 0
        residual = T(Inf)
        bordered = Matrix{T}(undef, n + 1, n + 1)
        rhs = Vector{T}(undef, n + 1)
        bordered_rank = 0
        bordered_condition = T(Inf)
        rejected_steps = 0
        corrector_method = :bordered

        for iter in 1:max_corrector_iterations
            iters = iter
            sys_try = _n8_preconditioned_system(phi_try, k_try,
                setup.qcanonical, setup.ordered_qdotτ, setup.phases)
            bordered[1:n, 1:n] .= sys_try.hessian
            bordered[1:n, n+1] .= sys_try.gradient_dk .* k_arc_scale
            bordered[n+1, 1:n] .= tangent_phi
            bordered[n+1, n+1] = tangent_k / k_arc_scale
            rhs[1:n] .= .-sys_try.gradient
            rhs[n+1] = -dot(tangent_phi, phi_try .- phi_pred) -
                (tangent_k / k_arc_scale) * ((k_try - k_pred) / k_arc_scale)
            bordered_rank = rank(bordered)
            bordered_condition = cond(bordered)
            correction = try
                bordered \ rhs
            catch
                qr(bordered) \ rhs
            end
            # Near a fold the bordered solve is itself poorly conditioned.
            # A full Newton correction can jump to a different periodic root;
            # accept only a residual-decreasing line-search step and let the
            # outer arclength controller reduce ds when no safe step exists.
            old_residual = max(norm(sys_try.gradient, Inf), abs(rhs[n+1]))
            accepted = false
            alpha = one(T)
            phi_candidate = similar(phi_try)
            k_candidate = k_try
            candidate_residual = old_residual
            while alpha >= T(2.0)^(-12)
                phi_candidate .= phi_try .+ alpha .* correction[1:n]
                k_candidate = k_try + alpha * correction[end] * k_arc_scale
                trust_radius = max(T(10) * abs(current_ds), T(1e-3))
                local_step_norm = sqrt(sum(abs2, phi_candidate .- phi_pred) +
                    ((k_candidate - k_pred) / k_arc_scale)^2)
                if local_step_norm > trust_radius
                    alpha *= T(0.5)
                    continue
                end
                sys_candidate = _n8_preconditioned_system(phi_candidate, k_candidate,
                    setup.qcanonical, setup.ordered_qdotτ, setup.phases)
                arc_candidate = dot(tangent_phi, phi_candidate .- phi_pred) +
                    (tangent_k / k_arc_scale) * ((k_candidate - k_pred) / k_arc_scale)
                candidate_residual = max(norm(sys_candidate.gradient, Inf),
                    abs(arc_candidate))
                if candidate_residual <= old_residual * (one(T) - T(1e-4) * alpha) ||
                        candidate_residual <= T(tolerance)
                    accepted = true
                    break
                end
                rejected_steps += 1
                alpha *= T(0.5)
            end
            if !accepted
                rejected_steps += 1
                converged = false
                residual = old_residual
                break
            end
            phi_try .= phi_candidate
            k_try = k_candidate
            residual = candidate_residual
            if residual <= T(tolerance)
                converged = true
                break
            end
        end

        # A fold has a nearly vertical projection onto k.  If the bordered
        # corrector cannot take the requested arclength step, continue the
        # same chain with a short fixed-k Newton corrector.  This is a
        # well-posed local fallback (the previous point remains the seed),
        # and it supplies the failure boundary that tells the caller when the
        # radial projection has reached the catastrophe.
        if !converged
            k_fallback = k + sign(T(ds)) * min(abs(T(ds)), T(5e-5))
            if T(k_bounds[1]) <= k_fallback <= T(k_bounds[2])
                phi_fallback = copy(phi)
                fallback_ok = false
                fallback_residual = T(Inf)
                fallback_iters = 0
                for fit in 1:100
                    fallback_iters = fit
                    sf = _n8_preconditioned_system(phi_fallback, k_fallback,
                        setup.qcanonical, setup.ordered_qdotτ, setup.phases)
                    fallback_residual = norm(sf.gradient, Inf)
                    if fallback_residual <= T(tolerance)
                        fallback_ok = true
                        break
                    end
                    delta = try
                        sf.hessian \ (-sf.gradient)
                    catch
                        qr(sf.hessian) \ (-sf.gradient)
                    end
                    alpha = one(T)
                    old = fallback_residual
                    while alpha >= T(2.0)^(-12)
                        candidate = phi_fallback .+ alpha .* delta
                        sc = _n8_preconditioned_system(candidate, k_fallback,
                            setup.qcanonical, setup.ordered_qdotτ, setup.phases)
                        nr = norm(sc.gradient, Inf)
                        if nr < old
                            phi_fallback .= candidate
                            fallback_residual = nr
                            break
                        end
                        alpha *= T(0.5)
                    end
                end
                if fallback_ok
                    corrector_method = :fixed_k_fallback
                    phi_try .= phi_fallback
                    k_try = k_fallback
                    residual = fallback_residual
                    iters = max(iters, fallback_iters)
                    converged = true
                end
            end
        end

        if !converged
            current_ds *= T(ds_shrink)
            if abs(current_ds) < T(min_ds)
                failure_encountered = true
                termination_reason = :step_failed
                theta_glsm_try = _n8_leading_to_theta(phi_try, setup.Qtilde)
                push!(steps, N8ContinuationStep{T}(
                    copy(theta_glsm_try), k_try, T(residual),
                    T(NaN), T[], false, iters,
                    _n8_leading_to_theta(tangent_phi, setup.Qtilde),
                    tangent_k, current_ds, branch_id, step_idx, T(NaN),
                    :failed, bordered_rank, bordered_condition, rejected_steps))
                break
            end
            continue
        end

        phi .= phi_try
        k = k_try
        sys = _n8_preconditioned_system(phi, k, setup.qcanonical,
            setup.ordered_qdotτ, setup.phases)
        J = hcat(sys.hessian, sys.gradient_dk .* k_arc_scale)
        F_svd = svd(J)
        new_tangent = F_svd.Vt[end, :]
        prev_full = vcat(tangent_phi, tangent_k / k_arc_scale)
        if dot(new_tangent, prev_full) < zero(T)
            new_tangent .*= -one(T)
        end
        tangent_phi = new_tangent[1:n]
        tangent_k = new_tangent[end] * k_arc_scale

        theta_glsm = _n8_leading_to_theta(phi, setup.Qtilde)
        raw = _n8_raw_hessian(theta_glsm, k, setup.Q, T.(setup.potential.qdotτ),
            zeros(T, size(setup.Q, 2)))
        can_eig = _n8_canonical_hessian_eigenvalues(raw.hessian, k)
        cond_eig = eigvals(Symmetric(sys.hessian))

        tangent_theta_glsm = _n8_leading_to_theta(tangent_phi, setup.Qtilde)
        tangent_norm = sqrt(dot(tangent_theta_glsm, tangent_theta_glsm) + tangent_k^2)
        tangent_phi ./= tangent_norm
        tangent_theta_glsm ./= tangent_norm
        tangent_k /= tangent_norm
        push!(steps, N8ContinuationStep{T}(copy(theta_glsm), k,
            norm(sys.gradient, Inf), minimum(can_eig), copy(can_eig),
            converged, iters, copy(tangent_theta_glsm), tangent_k,
            current_ds, branch_id, step_idx, minimum(cond_eig),
            corrector_method, bordered_rank, bordered_condition, rejected_steps))

        if catastrophe_bracket === nothing && length(steps) >= 2
            prev = steps[end-1]
            curr = steps[end]
            if prev.conditioned_hessian_min * curr.conditioned_hessian_min < zero(T)
                catastrophe_bracket = (length(steps) - 1, length(steps))
                catastrophe_k = (prev.k + curr.k) / 2
            end
        end

        if iters <= target_iterations
            current_ds = sign(current_ds) *
                min(T(max_ds), abs(current_ds) * T(ds_growth))
        elseif iters > 2 * target_iterations
            current_ds = sign(current_ds) *
                max(T(min_ds), abs(current_ds) * T(ds_shrink))
        end
    end

    termination_reason == :running && (termination_reason = :max_attempts)
    status = catastrophe_bracket !== nothing ? :catastrophe_detected :
        (failure_encountered ? :step_failed :
            (length(steps) >= n_steps ? :max_steps : termination_reason))

    N8ContinuationResult{T}(steps, branch_id, catastrophe_bracket,
        catastrophe_k, catastrophe_theta, status)
end

"""
    n8_find_regular_branches(k; n_starts=512, residual_tolerance=1e-10)

Find distinct stationary points of ∇V(θ,k)=0 at fixed k using the
hierarchy-preconditioned leading-charge solver. Returns period-one GLSM
coordinates with P96 canonical Hessian inertia.
"""
function n8_find_regular_branches(k::Real; n_starts::Int=512,
        residual_tolerance::Real=1e-10, merge_tolerance::Real=1e-6)
    T = Float64
    setup = _n8_setup_leading_charge(T(k))
    n = 8

    found = Vector{NamedTuple{(:theta, :phi, :gradient_residual,
        :canonical_eigenvalues, :conditioned_eigenvalues,
        :n_negative, :n_zero, :converged),
        Tuple{Vector{T}, Vector{T}, T, Vector{T}, Vector{T}, Int, Int, Bool}}}()

    for start_idx in 1:n_starts
        phi = start_idx == 1 ? zeros(T, n) : rand(T, n)
        converged = false
        for iter in 1:200
            sys = _n8_preconditioned_system(phi, T(k), setup.qcanonical,
                setup.ordered_qdotτ, setup.phases)
            if norm(sys.gradient, Inf) <= T(residual_tolerance)
                converged = true
                break
            end
            cond_val = cond(sys.hessian)
            if !isfinite(cond_val) || cond_val > T(1e14)
                break
            end
            phi .-= sys.hessian \ sys.gradient
        end
        if !converged; continue; end
        phi .= mod.(phi, one(T))
        sys = _n8_preconditioned_system(phi, T(k), setup.qcanonical,
            setup.ordered_qdotτ, setup.phases)
        if norm(sys.gradient, Inf) > T(residual_tolerance); continue; end

        theta_glsm = _n8_leading_to_theta(phi, setup.Qtilde)
        theta_glsm .= mod.(theta_glsm, one(T))
        raw = _n8_raw_hessian(theta_glsm, T(k), setup.Q, T.(setup.potential.qdotτ),
            zeros(T, size(setup.Q, 2)))
        can_eig = _n8_canonical_hessian_eigenvalues(raw.hessian, T(k))
        cond_eig = eigvals(Symmetric(sys.hessian))
        # The regular seeds are deliberately away from the event.  Use a
        # roundoff-scale inertia cutoff so a soft but nonzero mode is retained
        # as positive/negative and is not silently relabeled ``zero``.
        zero_tol = T(1e-10) * max(one(T), maximum(abs, cond_eig))
        n_negative = count(<(-zero_tol), cond_eig)
        n_zero = count(e -> abs(e) <= zero_tol, cond_eig)

        duplicate = any(f -> maximum(min.(abs.(f.theta .- theta_glsm),
            one(T) .- abs.(f.theta .- theta_glsm))) < T(merge_tolerance), found)
        if !duplicate
            push!(found, (; theta=copy(theta_glsm), phi=copy(phi),
                gradient_residual=norm(sys.gradient, Inf),
                canonical_eigenvalues=copy(can_eig),
                conditioned_eigenvalues=copy(cond_eig),
                n_negative, n_zero, converged))
        end
    end
    sort!(found, by=f -> f.n_negative)
    found
end

"""
    n8_continuation_catastrophe_localization(result; bisection_steps=50)

Refine the catastrophe bracket by bisection on the conditioned Hessian
smallest eigenvalue.
"""
function n8_continuation_catastrophe_localization(result::N8ContinuationResult{T};
        bisection_steps::Int=50, tolerance::T=T(1e-12)) where {T<:AbstractFloat}
    result.catastrophe_bracket === nothing &&
        throw(ArgumentError("no catastrophe bracket"))
    idx_lo, idx_hi = result.catastrophe_bracket
    step_lo = result.steps[idx_lo]
    step_hi = result.steps[idx_hi]

    setup = _n8_setup_leading_charge(step_lo.k)
    n = 8

    phi_lo = _n8_theta_to_leading(step_lo.theta, setup.Qtilde)
    k_lo = step_lo.k
    eig_lo = step_lo.conditioned_hessian_min
    phi_hi = _n8_theta_to_leading(step_hi.theta, setup.Qtilde)
    k_hi = step_hi.k
    eig_hi = step_hi.conditioned_hessian_min

    for _ in 1:bisection_steps
        k_mid = (k_lo + k_hi) / 2
        phi_mid = (phi_lo .+ phi_hi) ./ 2
        for iter in 1:100
            sys = _n8_preconditioned_system(phi_mid, k_mid, setup.qcanonical,
                setup.ordered_qdotτ, setup.phases)
            if norm(sys.gradient, Inf) <= tolerance; break; end
            c = cond(sys.hessian)
            if !isfinite(c) || c > T(1e14)
                reg = sys.hessian + T(1e-10) * I
                phi_mid .-= reg \ sys.gradient
            else
                phi_mid .-= sys.hessian \ sys.gradient
            end
        end
        sys = _n8_preconditioned_system(phi_mid, k_mid, setup.qcanonical,
            setup.ordered_qdotτ, setup.phases)
        eig_mid = minimum(eigvals(Symmetric(sys.hessian)))

        if eig_lo * eig_mid < zero(T)
            phi_hi .= phi_mid; k_hi = k_mid; eig_hi = eig_mid
        else
            phi_lo .= phi_mid; k_lo = k_mid; eig_lo = eig_mid
        end
        abs(k_hi - k_lo) < tolerance && break
    end

    k_cat = (k_lo + k_hi) / 2
    phi_cat = (phi_lo .+ phi_hi) ./ 2
    for iter in 1:100
        sys = _n8_preconditioned_system(phi_cat, k_cat, setup.qcanonical,
            setup.ordered_qdotτ, setup.phases)
        if norm(sys.gradient, Inf) <= tolerance; break; end
        c = cond(sys.hessian)
        if !isfinite(c) || c > T(1e14)
            reg = sys.hessian + T(1e-10) * I
            phi_cat .-= reg \ sys.gradient
        else
            phi_cat .-= sys.hessian \ sys.gradient
        end
    end

    theta_cat = _n8_leading_to_theta(phi_cat, setup.Qtilde)
    raw = _n8_raw_hessian(theta_cat, k_cat, setup.Q, T.(setup.potential.qdotτ),
        zeros(T, size(setup.Q, 2)))
    can_eig = _n8_canonical_hessian_eigenvalues(raw.hessian, k_cat)
    sys_final = _n8_preconditioned_system(phi_cat, k_cat, setup.qcanonical,
        setup.ordered_qdotτ, setup.phases)
    cond_eig = eigvals(Symmetric(sys_final.hessian))

    (; k=k_cat, theta=mod.(theta_cat, one(T)),
       gradient_residual=norm(sys_final.gradient, Inf),
       canonical_hessian_min=minimum(can_eig),
       canonical_hessian_eigenvalues=copy(can_eig),
       conditioned_hessian_min=minimum(cond_eig),
       conditioned_hessian_eigenvalues=copy(cond_eig),
       bracket_width=abs(k_hi - k_lo))
end

"""
    n8_bigfloat_continuation_refine(theta_seed, k_seed; precision_bits=256)

Refine a Float64 result at BigFloat precision using exact integer charges
and rational actions.
"""
function n8_bigfloat_continuation_refine(theta_seed::AbstractVector{<:Real},
        k_seed::Real; precision_bits::Int=256,
        tolerance::Real=BigFloat("1e-40"), max_iterations::Int=200)
    precision_bits >= 128 || throw(ArgumentError("precision_bits must be >= 128"))
    setprecision(BigFloat, precision_bits) do
        T = BigFloat
        two_pi = T(2) * T(π)
        Q, qdottau_exact = _n8_exact_source_data(T)
        phase = zeros(T, 12)
        theta = T.(theta_seed)
        k = T(k_seed)
        qfloat = T.(Q)

        for iter in 1:max_iterations
            amplitudes = (k .* qdottau_exact) .* exp.(T(-2) * T(π) .* k .* qdottau_exact)
            arguments = two_pi .* (qfloat' * theta) .+ phase
            gradient = two_pi .* qfloat * (amplitudes .* sin.(arguments))
            hessian = two_pi^2 .* qfloat * Diagonal(amplitudes .* cos.(arguments)) * qfloat'
            grad_norm = norm(gradient, Inf)
            # This residual is evaluated from amplitudes that are already
            # source-normalized by the target construction.  Apply the
            # declared absolute BigFloat tolerance; multiplying by the
            # physical amplitude would demand an extra ~25 decimal digits
            # and can label a numerically converged stationary point failed.
            if grad_norm <= T(tolerance)
                hess_eig = eigvals(Symmetric(hessian))
                return (; theta=mod.(copy(theta), one(T)), k,
                    gradient_residual=grad_norm,
                    hessian_eigenvalues=copy(hess_eig),
                    hessian_min_eigenvalue=minimum(hess_eig),
                    converged=true, iterations=iter, precision_bits)
            end
            theta .-= hessian \ gradient
        end
        amplitudes = (k .* qdottau_exact) .* exp.(T(-2) * T(π) .* k .* qdottau_exact)
        arguments = two_pi .* (T.(Q)' * theta) .+ phase
        gradient = two_pi .* T.(Q) * (amplitudes .* sin.(arguments))
        hessian = two_pi^2 .* T.(Q) * Diagonal(amplitudes .* cos.(arguments)) * T.(Q)'
        hess_eig = eigvals(Symmetric(hessian))
        (; theta=mod.(copy(theta), one(T)), k,
            gradient_residual=norm(gradient, Inf),
            hessian_eigenvalues=copy(hess_eig),
            hessian_min_eigenvalue=minimum(hess_eig),
            converged=false, iterations=max_iterations, precision_bits)
    end
end

"""
    n8_bigfloat_augmented_solve(theta_seed, k_seed; precision_bits=256)

BigFloat augmented solve: gradient=0, Hv=0, v·v=1. Uses exact integer charges
and rational actions. Independent validation target.
"""
function n8_bigfloat_augmented_solve(theta_seed::AbstractVector{<:Real},
        k_seed::Real; precision_bits::Int=256,
        tolerance::Real=BigFloat("1e-40"), max_iterations::Int=500)
    precision_bits >= 128 || throw(ArgumentError("precision_bits must be >= 128"))
    setprecision(BigFloat, precision_bits) do
        T = BigFloat
        two_pi = T(2) * T(π)
        Q, qdottau_exact = _n8_exact_source_data(T)
        phase = zeros(T, 12)
        n = 8
        qfloat = T.(Q)
        theta = T.(theta_seed)
        k = T(k_seed)

        function eval_system(th, kk)
            amp = (kk .* qdottau_exact) .* exp.(T(-2) * T(π) .* kk .* qdottau_exact)
            args = two_pi .* (qfloat' * th) .+ phase
            g = two_pi .* qfloat * (amp .* sin.(args))
            H = two_pi^2 .* qfloat * Diagonal(amp .* cos.(args)) * qfloat'
            amp_dk = amp .* (one(T) .- T(2) * T(π) .* kk .* qdottau_exact) ./ kk
            g_dk = two_pi .* qfloat * (amp_dk .* sin.(args))
            H_dk = two_pi^2 .* qfloat * Diagonal(amp_dk .* cos.(args)) * qfloat'
            (; g, H, amp, args, sin_args=sin.(args), cos_args=cos.(args),
               g_dk, H_dk, amp_dk)
        end

        sys = eval_system(theta, k)
        hess_eig = eigen(Symmetric(sys.H))
        v = hess_eig.vectors[:, argmin(abs.(hess_eig.values))]

        state = vcat(theta, v, k)
        for iter in 1:max_iterations
            th = state[1:n]; vv = state[n+1:2n]; kk = state[end]
            sys = eval_system(th, kk)
            F = vcat(sys.g, sys.H * vv, dot(vv, vv) - one(T))
            if norm(F, Inf) <= T(tolerance)
                hess_eig_f = eigvals(Symmetric(sys.H))
                vn = vv ./ norm(vv)
                return (; theta=mod.(copy(th), one(T)), null_vector=copy(vn),
                    k=kk, gradient_residual=norm(sys.g, Inf),
                    null_residual=norm(sys.H * vn, Inf),
                    hessian_eigenvalues=copy(hess_eig_f),
                    hessian_min_eigenvalue=minimum(abs.(hess_eig_f)),
                    converged=true, iterations=iter, precision_bits)
            end
            J = zeros(T, 2n+1, 2n+1)
            J[1:n, 1:n] .= sys.H
            J[1:n, end] .= sys.g_dk
            for a in 1:12, i in 1:n, j in 1:n, l in 1:n
                J[n+i, l] += -two_pi^3 * sys.amp[a] * sys.sin_args[a] *
                    Q[i,a] * Q[j,a] * Q[l,a] * vv[j]
            end
            J[n+1:2n, n+1:2n] .= sys.H
            J[n+1:2n, end] .= sys.H_dk * vv
            J[2n+1, n+1:2n] .= T(2) .* vv
            state .-= J \ F
        end
        th = state[1:n]; vv = state[n+1:2n]; kk = state[end]
        sys = eval_system(th, kk)
        hess_eig_f = eigvals(Symmetric(sys.H))
        vn = vv ./ norm(vv)
        (; theta=mod.(copy(th), one(T)), null_vector=copy(vn),
            k=kk, gradient_residual=norm(sys.g, Inf),
            null_residual=norm(sys.H * vn, Inf),
            hessian_eigenvalues=copy(hess_eig_f),
            hessian_min_eigenvalue=minimum(abs.(hess_eig_f)),
            converged=false, iterations=max_iterations, precision_bits)
    end
end

"""
    n8_bigfloat_p96_diagnostic(theta, k; precision_bits=256)

Evaluate the P96 projected diagnostic with exact twelve-term source data.
The reconstructed P96 metric is currently a Float64 witness, so the result
records that 53-bit metric boundary instead of implying a high-precision
canonical metric.
"""
function n8_bigfloat_p96_diagnostic(theta_seed::AbstractVector{<:Real},
        k_seed::Real; precision_bits::Int=256,
        tolerance::Real=BigFloat("1e-8"))
    precision_bits >= 128 || throw(ArgumentError("precision_bits must be >= 128"))
    setprecision(BigFloat, precision_bits) do
        T = BigFloat
        Q, qdottau_exact = _n8_exact_source_data(T)
        k = T(k_seed)
        amplitudes = (k .* qdottau_exact) .*
            exp.(T(-2) * T(π) .* k .* qdottau_exact)
        # n8_geometry() is the exact-intersection reconstruction rounded at
        # the repository's Float64 boundary. Preserve that provenance in the
        # returned certificate while evaluating the potential at target bits.
        metric = T.(Matrix(n8_geometry().kinetic)) ./ k^2
        result = local_catastrophe_diagnostic(T.(theta_seed), Q, amplitudes, metric;
            phases=zeros(T, 12), argument_scale=T(2) * T(π), precision_bits,
            tolerance=T(tolerance), gradient_tolerance=T(tolerance),
            hessian_tolerance=T(tolerance), derivative_tolerance=T(tolerance))
        merge(result, (; metric_source_precision_bits=53,
            metric_precision_boundary=:float64_reconstructed,
            source_data=:exact_integer_rational_table1,
            coordinate_contract=:P96))
    end
end

"""
    n8_continuation_classify(theta, k; precision_bits=53)

Classify catastrophe at (theta, k) using P96 canonical metric and argument_scale=2π.
"""
function n8_continuation_classify(theta::AbstractVector{<:Real}, k::Real;
        precision_bits::Int=53, tolerance::Real=1e-8)
    potential = _n8_potential(k=k)
    metric = Matrix(n8_kinetic_matrix(k))
    amplitudes = vec(potential.L[1, :]) .* 10.0 .^ vec(potential.L[2, :])
    local_catastrophe_diagnostic(theta, potential.Q, amplitudes, metric;
        argument_scale=2π, precision_bits, tolerance,
        gradient_tolerance=tolerance, hessian_tolerance=tolerance,
        derivative_tolerance=tolerance)
end

"""
    n8_continuation_compare_matcher(continuation_steps, matcher_records; ...)

Compare continuation vs post-hoc matcher branch identities.
"""
function n8_continuation_compare_matcher(
        continuation_steps::Vector{N8ContinuationStep{T}},
        matcher_records;
        matching_tolerance::T=T(1e-4)) where {T<:AbstractFloat}
    disagreements = NamedTuple[]
    cont_points = [(; s.theta, s.k, s.branch_id, step=i)
        for (i, s) in enumerate(continuation_steps) if s.converged]
    for cp in cont_points, mr in matcher_records
        if maximum(min.(abs.(cp.theta .- mr.corrected_theta),
                one(T) .- abs.(cp.theta .- mr.corrected_theta))) < matching_tolerance
            if cp.branch_id != mr.seed_index
                push!(disagreements, (;
                    continuation_branch=cp.branch_id,
                    matcher_seed=mr.seed_index,
                    k=cp.k, distance=maximum(min.(
                        abs.(cp.theta .- mr.corrected_theta),
                        one(T) .- abs.(cp.theta .- mr.corrected_theta)))))
            end
        end
    end
    (; n_continuation=length(cont_points),
       n_matcher=length(matcher_records),
       n_disagreements=length(disagreements),
       disagreements)
end
