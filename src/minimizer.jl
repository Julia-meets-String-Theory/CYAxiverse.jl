"""
    CYAxiverse.minimizer
Some minimization / optimization routines to find vacua and other such explorations.

"""
module minimizer

using HDF5
using LinearAlgebra
using ArbNumerics, Tullio, LoopVectorization, NormalForms
using GenericLinearAlgebra
using Distributions
using Random
using Optim, LineSearches, Dates, HDF5, NLsolve

using ..filestructure: cyax_file, minfile, present_dir
using ..read: potential
using ..structs: GeometryIndex

"""Return the first `n` prime integers for deterministic Halton sampling."""
function _first_primes(n::Int)
    primes = Int[]
    candidate = 2
    while length(primes) < n
        isprime = all(p -> p * p > candidate || candidate % p != 0, primes)
        isprime && push!(primes, candidate)
        candidate += 1
    end
    primes
end

"""Evaluate one coordinate of a Halton sequence in the given prime base."""
function _radical_inverse(index::Int, base::Int)
    result = 0.0
    factor = inv(Float64(base))
    while index > 0
        index, digit = divrem(index, base)
        result += digit * factor
        factor /= base
    end
    result
end

function _legacy_qx_typed(QV::AbstractMatrix{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    qx = zeros(T, size(QV, 1))
    for c in axes(QV, 1), i in axes(QV, 2)
        qx[c] += QV[c, i] * x[i]
    end
    qx
end

function _legacy_qx(QV::AbstractMatrix, x::AbstractVector)
    T = typeof(ArbFloat(0))
    _legacy_qx_typed(T.(QV), T.(x))
end

function _legacy_gradient_typed(LV::AbstractVector{T}, QV::AbstractMatrix{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    qx = _legacy_qx_typed(QV, x)
    gradient = zeros(T, size(QV, 2))
    for i in axes(QV, 2), c in axes(QV, 1)
        gradient[i] += LV[c] * QV[c, i] * sin(qx[c])
    end
    gradient
end

function _legacy_gradient(LV::AbstractVector, QV::AbstractMatrix, x::AbstractVector)
    T = typeof(ArbFloat(0))
    _legacy_gradient_typed(T.(LV), T.(QV), T.(x))
end

function _legacy_hessian_typed(LV::AbstractVector{T}, QV::AbstractMatrix{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    qx = _legacy_qx_typed(QV, x)
    hessian = zeros(T, size(QV, 2), size(QV, 2))
    for i in axes(QV, 2), j in axes(QV, 2), c in axes(QV, 1)
        hessian[i, j] += LV[c] * QV[c, i] * QV[c, j] * cos(qx[c])
    end
    Hermitian(hessian)
end

function _legacy_hessian(LV::AbstractVector, QV::AbstractMatrix, x::AbstractVector)
    T = typeof(ArbFloat(0))
    _legacy_hessian_typed(T.(LV), T.(QV), T.(x))
end

function _phase_hessian!(hessian::AbstractMatrix, LV::AbstractVector,
        QV::AbstractMatrix, cosine_phases)
    for i in axes(QV, 1), j in axes(QV, 1)
        if i >= j
            hessian[i, j] = sum(LV' *
                (@view(QV[i, :]) .* @view(QV[j, :]) .* cosine_phases))
        end
    end
    hessian
end


"""
    critical_points(L, Q; phases=zeros(size(Q, 2)), starts=4096,
                    residual_tolerance=1e-10, merge_tolerance=1e-7,
                    initial_points=nothing)

Find critical points of an integer-charge cosine potential deterministically in
the unit torus. `Q` has axions in rows and instantons in columns; `L[1, :]`
stores coefficient signs and `L[2, :]` stores base-10 logarithmic magnitudes.

Initial points are a Halton sequence, so repeated calls are reproducible. Roots
are folded into `[0,1)^N`, deduplicated using periodic distance, and classified
by Hessian inertia. Equation residuals and Newton Jacobians are scaled locally
by the largest logarithmic amplitude touching each equation, while the
classification Hessian is assembled with a symmetric diagonal congruence
scaling. Optional `initial_points` are deterministic torus seeds; negative
physical-Hessian modes at those seeds are also sampled to reach displaced
minima in highly dimensional problems. The returned coordinates use the paper
convention in which the potential arguments are `2π Q'θ + phases`.
"""
function critical_points(L::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Real};
        phases::AbstractVector{<:Real}=zeros(size(Q, 2)), starts::Int=4096,
        residual_tolerance::Float64=1e-10, merge_tolerance::Float64=1e-7,
        max_iterations::Int=200,
        coordinate_basis::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
    equation_scales::Union{Nothing, AbstractVector{<:Real}}=nothing,
    initial_points::Union{Nothing, AbstractMatrix{<:Real}}=nothing)
    size(L, 1) == 2 || throw(DimensionMismatch("L must have two rows"))
    size(Q, 2) == size(L, 2) || throw(DimensionMismatch("Q columns must match L columns"))
    length(phases) == size(Q, 2) || throw(DimensionMismatch("one phase is required per instanton"))
    starts > 0 || throw(ArgumentError("starts must be positive"))

    n, p = size(Q)
    q = coordinate_basis === nothing ? Matrix{Float64}(Q) : Matrix{Float64}(coordinate_basis) \ Matrix{Float64}(Q)
    if initial_points !== nothing
        size(initial_points, 1) == n || throw(DimensionMismatch("initial points must have one row per axion"))
    end
    phase = Float64.(phases)
    signs = Float64.(L[1, :])
    logscale = Float64.(L[2, :])
    minimum_logscale = log10(floatmin(Float64))
    row_logscale = zeros(n)
    for i in 1:n
        support = findall(!iszero, @view q[i, :])
        row_logscale[i] = isempty(support) ? maximum(logscale) : maximum(logscale[support])
    end
    scaled_amplitudes = zeros(n, p)
    for i in 1:n, j in 1:p
        q[i, j] == 0 && continue
        delta = logscale[j] - row_logscale[i]
        delta >= minimum_logscale && (scaled_amplitudes[i, j] = signs[j] * 10.0^delta)
    end
    row_scales = equation_scales === nothing ? ones(n) : Float64.(equation_scales) ./ maximum(abs, equation_scales)
    length(row_scales) == n || throw(DimensionMismatch("one equation scale is required per axion"))
    all(>(0), row_scales) || throw(ArgumentError("equation scales must be positive"))
    twoπ = 2π

    function scaled_physical_hessian!(out, θ)
        fill!(out, 0.0)
        arguments = twoπ .* (q' * θ) .+ phase
        for row in 1:n, col in 1:n, j in 1:p
            (q[row, j] == 0 || q[col, j] == 0) && continue
            delta = logscale[j] - (row_logscale[row] + row_logscale[col]) / 2
            delta < minimum_logscale && continue
            out[row, col] += q[row, j] * q[col, j] * signs[j] *
                10.0^delta * cos(arguments[j])
        end
        out .*= twoπ^2
        nothing
    end

    function gradient!(out, θ)
        fill!(out, 0.0)
        arguments = twoπ .* (q' * θ) .+ phase
        for i in 1:n, j in 1:p
            out[i] += q[i, j] * scaled_amplitudes[i, j] * sin(arguments[j])
        end
        out .*= twoπ
        out ./= row_scales
        nothing
    end
    function hessian!(out, θ)
        fill!(out, 0.0)
        arguments = twoπ .* (q' * θ) .+ phase
        for i in 1:n, k in 1:n, j in 1:p
            out[i, k] += q[i, j] * q[k, j] * scaled_amplitudes[i, j] * cos(arguments[j])
        end
        out .*= twoπ^2
        out ./= row_scales
        nothing
    end

    roots = Vector{Vector{Float64}}()
    residuals = Float64[]
    bases = _first_primes(n)
    seed_points = initial_points === nothing ? zeros(n, 0) : Matrix{Float64}(initial_points)
    if !isempty(seed_points)
        seed_hessian = zeros(n, n)
        for seed in eachcol(seed_points)
            scaled_physical_hessian!(seed_hessian, seed)
            values, vectors = eigen(Hermitian(seed_hessian))
            for mode in findall(<(-100 * eps(Float64)), values)
                direction = vectors[:, mode]
                direction ./= maximum(abs, direction)
                seed_points = hcat(seed_points,
                    mod.(seed .+ 0.125 .* direction, 1.0),
                    mod.(seed .- 0.125 .* direction, 1.0))
            end
        end
    end
    seed_count = size(seed_points, 2)
    for sample in 0:(seed_count + starts - 1)
        θ0 = sample < seed_count ? @view(seed_points[:, sample + 1]) :
            (sample == seed_count ? zeros(n) : [_radical_inverse(sample - seed_count, base) for base in bases])
        result = nlsolve(gradient!, hessian!, θ0; method=:newton,
            ftol=residual_tolerance, xtol=residual_tolerance, iterations=max_iterations)
        (result.f_converged || result.x_converged) || continue
        θ = mod.(result.zero, 1.0)
        residual = result.residual_norm
        residual <= residual_tolerance || continue
        duplicate = any(root -> maximum(min.(abs.(θ .- root), 1 .- abs.(θ .- root))) <= merge_tolerance, roots)
        duplicate && continue
        push!(roots, θ)
        push!(residuals, residual)
    end

    coordinates = isempty(roots) ? zeros(n, 0) : hcat(roots...)
    inertia = Vector{NTuple{3, Int}}(undef, length(roots))
    hessian_eigenvalues = Vector{Vector{Float64}}(undef, length(roots))
    hessian = zeros(n, n)
    for (i, θ) in enumerate(roots)
        scaled_physical_hessian!(hessian, θ)
        values = eigvals(Hermitian(hessian))
        fill!(hessian, 0.0)
        scale = max(maximum(abs, values), 1.0)
        zero_tolerance = 100 * residual_tolerance * scale
        inertia[i] = (count(<(-zero_tolerance), values), count(x -> abs(x) <= zero_tolerance, values), count(>(zero_tolerance), values))
        hessian_eigenvalues[i] = values
    end
    minima_mask = [entry == (0, 0, n) for entry in inertia]
    minima = coordinates[:, minima_mask]
    (; coordinates, minima, inertia, hessian_eigenvalues, residuals,
       critical_count=length(roots), minima_count=count(minima_mask), starts)
end

function minimize(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix,x0::Vector,gradσ::Matrix,θparalleltest::Matrix,Qtilde::Matrix,algo,prec)
    setprecision(ArbFloat,digits=prec)
    T = typeof(ArbFloat(0))
    LV_t = T.(LV)
    QV_t = T.(QV)
    x0 = T.(x0)
    Arb0 = ArbFloat(0.)
    Arb1 = ArbFloat(1.)
    Arb2π = ArbFloat(2π)
    threshold = 0.01
    QX(x::Vector) = _legacy_qx_typed(QV_t, x)
    function fitness(x::Vector)
        V = dot(LV_t,(Arb1 .- cos.(QX(x))))
        return V
    end
    function grad!(gradient::Vector, x::Vector)
        gradient .= _legacy_gradient_typed(LV_t, QV_t, x)
    end
    hess(x::Vector) = _legacy_hessian_typed(LV_t, QV_t, x)
    hess!(hessian::Matrix, x::Vector) = (hessian .= _legacy_hessian_typed(LV_t, QV_t, x))
    grad(x) = _legacy_gradient_typed(LV_t, QV_t, x)
    res = optimize(fitness,grad!,hess!,
                x0, algo,
                Optim.Options(x_tol =minimum(abs.(LV)),g_tol =minimum(threshold .* abs.(gradσ))))
    Vmin = Optim.minimum(res)
    xmin = Optim.minimizer(res)
    GC.gc()
    if Float64(log10(abs(minimum(eigen(hess(xmin)).values)))) < -prec && sum(Float64.(log10.(abs.(grad(xmin)))) .< log10.(abs.(threshold .* gradσ))) == (h11 - size(gradσ[gradσ .== 0.],1))
        a = mod.(((T.(θparalleltest) * xmin)/Arb2π),Arb1)
        atilde = T.(Qtilde) * xmin/Arb2π
        a_sign = Int.(sign.(a))
        a_log = Float64.(log10.(abs.(a)))
        atilde_sign = Int.(sign.(atilde))
        atilde_log = Float64.(log10.(abs.(atilde)))
        Vmin_sign = Int(sign(Vmin))
        Vmin_log = Float64(log10(abs(Vmin)))
        xmin_log = Float64.(log10.(abs.(xmin)))
        xmin_sign = Int.(sign.(xmin))

        keys = ["±V", "logV","±x", "logx", "±a","loga", "±ã", "logã"]
        vals = [Vmin_sign, Vmin_log, xmin_sign, xmin_log, a_sign, a_log, atilde_sign, atilde_log]
        return Dict(zip(keys,vals))
        GC.gc()
    end
end

function minimize(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix,x0::Vector,gradσ::Matrix,algo,prec)
    setprecision(ArbFloat; digits=prec)
    T = typeof(ArbFloat(0))
    LV_t = T.(LV)
    QV_t = T.(QV)
    x0 = T.(x0)
    Arb0 = ArbFloat(0.)
    Arb1 = ArbFloat(1.)
    Arb2π = ArbFloat(2π)
    threshold = 0.01
    QX(x::Vector) = _legacy_qx_typed(QV_t, x)
    function fitness(x::Vector)
        V = dot(LV_t,(Arb1 .- cos.(QX(x))))
        return V
    end
    function grad!(gradient::Vector, x::Vector)
        gradient .= _legacy_gradient_typed(LV_t, QV_t, x)
    end
    function hess(x::Vector)
        hessfull = _legacy_hessian_typed(LV_t, QV_t, x)
    end
    function hess!(hessian::Matrix, x::Vector)
        hessian .= _legacy_hessian_typed(LV_t, QV_t, x)
    end
    grad(x) = _legacy_gradient_typed(LV_t, QV_t, x)
    res = optimize(fitness,grad!,hess!,
                x0, algo,
                Optim.Options(x_tol =minimum(abs.(LV)),g_tol =minimum(threshold .* abs.(gradσ))))
    Vmin = Optim.minimum(res)
    xmin = Optim.minimizer(res)
    GC.gc()
    # if Float64(log10(abs(minimum(eigen(hess(xmin)).values)))) < -prec && sum(Float64.(log10.(abs.(grad(xmin)))) .< log10.(abs.(threshold .* gradσ))) == (h11 - size(gradσ[gradσ .== 0.],1))
    hess_eigs = Float64(log10(abs(minimum(eigen(hess(xmin)).values)))) 
    hess_sign = sign((minimum(eigen(hess(xmin)).values)))
    sum_grad = sum(Float64.(log10.(abs.(grad(xmin)))))
    Vmin_sign = Int(sign(Vmin))
    Vmin_log = Float64(log10(abs(Vmin)))
    xmin_log = Float64.(log10.(abs.(xmin)))
    xmin_sign = Int.(sign.(xmin))

    keys = ["±V", "logV","±x", "logx", "Heigs", "Hsign", "gradsum"]
    vals = [Vmin_sign, Vmin_log, xmin_sign, xmin_log, hess_eigs, hess_sign, sum_grad]
    return Dict(zip(keys,vals))
    GC.gc()
    # end
end


function minimize(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix,x0::Vector,gradσ::Matrix,Qtilde::Matrix,algo,prec)
    setprecision(ArbFloat,digits=prec)
    T = typeof(ArbFloat(0))
    LV_t = T.(LV)
    QV_t = T.(QV)
    x0 = T.(x0)
    Arb0 = ArbFloat(0.)
    Arb1 = ArbFloat(1.)
    Arb2π = ArbFloat(2π)
    threshold = 0.01
    QX(x::Vector) = _legacy_qx_typed(QV_t, x)
    function fitness(x::Vector)
        V = dot(LV_t,(Arb1 .- cos.(QX(x))))
        return V
    end
    function grad!(gradient::Vector, x::Vector)
        gradient .= _legacy_gradient_typed(LV_t, QV_t, x)
    end
    hess(x::Vector) = _legacy_hessian_typed(LV_t, QV_t, x)
    hess!(hessian::Matrix, x::Vector) = (hessian .= _legacy_hessian_typed(LV_t, QV_t, x))
    grad(x) = _legacy_gradient_typed(LV_t, QV_t, x)
    res = optimize(fitness,grad!,hess!,
                x0, algo,
                Optim.Options(x_tol =minimum(abs.(LV)),g_tol =minimum(threshold .* abs.(gradσ))))
    Vmin = Optim.minimum(res)
    xmin = Optim.minimizer(res)
    GC.gc()
    if Float64(log10(abs(minimum(eigen(hess(xmin)).values)))) < -prec && sum(Float64.(log10.(abs.(grad(xmin)))) .< log10.(abs.(threshold .* gradσ))) == (h11 - size(gradσ[gradσ .== 0.],1))
        atilde = T.(Qtilde) * xmin/Arb2π
        atilde_sign = Int.(sign.(atilde))
        atilde_log = Float64.(log10.(abs.(atilde)))
        Vmin_sign = Int(sign(Vmin))
        Vmin_log = Float64(log10(abs(Vmin)))
        xmin_log = Float64.(log10.(abs.(xmin)))
        xmin_sign = Int.(sign.(xmin))

        keys = ["±V", "logV","±x", "logx", "±ã", "logã"]
        vals = [Vmin_sign, Vmin_log, xmin_sign, xmin_log, atilde_sign, atilde_log]
        return Dict(zip(keys,vals))
        GC.gc()
    end
end

function minimize_save(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix,x0::Vector,gradσ::Matrix,θparalleltest::Matrix,Qtilde::Matrix,algo; prec::Int=5_000, run_num::Int=1)
    min_data = minimize(h11,tri,cy,LV,QV,x0,gradσ,θparalleltest,Qtilde,algo, prec)
    if min_data === nothing
        return nothing
    else
        h5open(minfile(h11, tri, cy), isfile(minfile(h11, tri, cy)) ? "r+" : "w") do file
            if haskey(file, "runs")
            else
                f0 = create_group(file,"runs")
            end
            f0 = create_group(file, "runs/$run_num")
            f1 = create_group(f0, "V")
            f1["log10",deflate=9] = min_data["logV"]
            f1["sign",deflate=9] = min_data["±V"]
            f2 = create_group(f0, "x")
            f2["log10",deflate=9] = min_data["logx"]
            f2["sign",deflate=9] = min_data["±x"]
            f3 = create_group(f0, "a")
            f3["log10",deflate=9] = min_data["loga"]
            f3["sign",deflate=9] = min_data["±a"]
            f4 = create_group(f0, "atilde")
            f4["log10",deflate=9] = min_data["logã"]
            f4["sign",deflate=9] = min_data["±ã"]
        end
    end
GC.gc()
end

function minimize_save(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix,x0::Vector,gradσ::Matrix,Qtilde::Matrix,algo; prec::Int=5_000, run_num::Int=1)
    min_data = minimize(h11,tri,cy,LV,QV,x0,gradσ,Qtilde,algo, prec)
    if min_data === nothing
        return nothing
    else
        h5open(minfile(h11, tri, cy), isfile(minfile(h11, tri, cy)) ? "r+" : "w") do file
            if haskey(file, "runs")
            else
                f0 = create_group(file,"runs")
            end
            f0 = create_group(file, "runs/$run_num")
            f1 = create_group(f0, "V")
            f1["log10",deflate=9] = min_data["logV"]
            f1["sign",deflate=9] = min_data["±V"]
            f2 = create_group(f0, "x")
            f2["log10",deflate=9] = min_data["logx"]
            f2["sign",deflate=9] = min_data["±x"]
            f4 = create_group(f0, "atilde")
            f4["log10",deflate=9] = min_data["logã"]
            f4["sign",deflate=9] = min_data["±ã"]
        end
    end
GC.gc()
end

"""
    grad_std(h11::Int, tri::Int, cy::Int, LV::Vector, QV::Matrix)

Return the per-component standard deviation of the axion-potential gradient
``\\partial_i V = \\sum_a \\Lambda_a Q_{a,i} \\sin(Q_a \\cdot x)`` sampled over
field space for geometry ``(h^{1,1}, \\mathrm{tri}, \\mathrm{cy})``, evaluated at
`ArbFloat` precision. `LV` is the vector of linear instanton scales
``\\Lambda_a`` and `QV` the charge matrix. Use it to gauge the gradient scale
before minimization.
"""
function grad_std(h11::Int,tri::Int,cy::Int,LV::Vector,QV::Matrix)
    T = typeof(ArbFloat(0))
    LV_t = T.(LV)
    QV_t = T.(QV)
    Arb0 = T(0.)
    Arb1 = T(1.)
    Arb2π = T(2π)
    QX(x::Vector) = _legacy_qx_typed(QV_t, x)
    grad(x::Vector) = _legacy_gradient_typed(LV_t, QV_t, x)
    n=100
    grad_all = zeros(h11,n)
    for j=1:n
        x0 = T.(rand(Uniform(0,2π),h11)) .* rand(T,h11)
        grad_all[:,j] = grad(x0)
    end
    return T.(std(grad_all, dims=2))
end

"""
    grad_std(LV::Vector, QV::Matrix)

Return the per-component standard deviation of the axion-potential gradient for
the linear instanton scales `LV` (``\\Lambda_a``) and charge matrix `QV`,
sampled over field space. See [`grad_std(h11, tri, cy, LV, QV)`](@ref) for the
geometry-indexed form.
"""
function grad_std(LV::Vector,QV::Matrix)
    if @isdefined h11 
    else 
        h11 = size(QV, 1)
    end
    function grad(x::Vector)
        grad_temp = LV' .* (QV .* sin.(x' * QV))
        sum(grad_temp, dims = 2)
    end
    n=100
    grad_all = zeros(h11,n)
    for j=1:n
        x0 = rand(Uniform(0,2π),h11) .* rand(h11)
        grad_all[:,j] = grad(x0)
    end
    return mean(grad_all, dims=2) .- 2. .* std(grad_all, dims=2)
end

function grad_std(h11::Int, tri::Int, cy::Int)
    pot_data = potential(h11,tri,cy)
    T = typeof(ArbFloat(0))
    QV::Matrix{T} = T.(pot_data.Q)
    LV::Matrix{Float64} = pot_data.L
    Lfull::Vector{T} = T.(LV[:,1]) .* T(10.) .^ T.(LV[:,2])
    grad_std(h11,tri,cy,Lfull,QV)
end


"""
    minimize(LV::Vector, QV, x0::Vector)

Minimize the axion potential
``V(x) = \\sum_a \\Lambda_a (1 - \\cos(Q_a \\cdot x))`` from the initial point
`x0`, where `LV` holds the linear instanton scales ``\\Lambda_a`` and `QV` is the
charge matrix. Return the located stationary point in field coordinates.
"""
function minimize(LV::Vector, QV, x0::Vector)
	if @isdefined h11
	else
		h11 = size(QV, 1)
	end
    @assert size(QV, 2) == size(LV, 1)
    threshold = 1e-2
    function fitness(x::Vector)
        sum(LV .* (1. .- cos.(x' * QV)))
    end
    function grad!(gradient::Vector, x::Vector)
        grad_temp = LV' .* (QV .* sin.(x' * QV))
        gradient .= sum(grad_temp, dims = 2)
    end
    function hess!(hessian::Matrix, x::Vector)
        cosine_phases = cos.(x' * QV)
        _phase_hessian!(hessian, LV, QV, cosine_phases)
        hessian .= hessian + hessian' - Diagonal(hessian)
    end
    function hess(x::Vector)
        hessian = zeros(size(x, 1), size(x, 1))
		cosine_phases = cos.(x' * QV)
		_phase_hessian!(hessian, LV, QV, cosine_phases)
		hessian + hessian' - Diagonal(hessian)
    end
    function grad(x::Vector)
        grad_temp = LV' .* (QV .* sin.(x' * QV))
        sum(grad_temp, dims = 2)
    end
    gradσ = grad_std(LV,QV)
    x_tol = minimum(abs.(LV))
    g_tol = eps() / threshold
	algo_LBFGS = LBFGS(linesearch = LineSearches.BackTracking());
    res = Optim.optimize(fitness, grad!, hess!, x0, Optim.Options(x_tol = x_tol, g_tol = g_tol))
    Vmin = Optim.minimum(res)
    xmin = Optim.minimizer(res)
    # GC.gc()
    # if abs(minimum(eigen(hess(xmin)).values)) < eps() && maximum(abs.(grad(xmin))) < eps() / threshold
    hess_eigs = Float64(log10(abs(minimum(eigen(hess(xmin)).values)))) 
    hess_sign = sign((minimum(eigen(hess(xmin)).values)))
    grad_log = log10.(abs.(grad(xmin)))
    Vmin_sign = Int(sign(Vmin))
    Vmin_log = Float64(log10(abs(Vmin)))
    xmin = @.ifelse(abs(xmin) < eps() / threshold, zero(xmin), xmin)
    xmin = @.ifelse(one(xmin) - mod(xmin / 2π, 1) < eps() / threshold || mod(xmin / 2π, 1) < eps() / threshold, zero(xmin), mod(xmin / 2π, 1))
    keys = ["±V", "logV","xmin", "Heigs", "Hsign", "gradlog"]
    vals = [Vmin_sign, Vmin_log, xmin, hess_eigs, hess_sign, grad_log]
    return Dict(zip(keys,vals))
        # GC.gc()
    # end
end
"""
    id_minimize(LV::Vector, QV::Matrix; ftol = eps(), iterations = 1_000)

This function takes the instanton charge matrix and their corresponding scales and finds the corresponding minima using the `id_basis` method.
    !!! warning
    Currently cannot locate local minima -- eigs(hess).values returned as negative in `nlsolve`...

"""
function id_minimize(LV::Vector, QV::Matrix; ftol = eps(), iterations = 1_000)
    @assert size(QV, 2) == size(LV, 1)

    
    function grad(x::Vector)
		grad_temp = LV' .* (QV .* sin.(x' * QV))
        sum(grad_temp, dims = 2)
	end
	function hess(x::Vector)
		hessian = zeros(size(x,1), size(x,1))
		cosine_phases = cos.(x' * QV)
		_phase_hessian!(hessian, LV, QV, cosine_phases)
		hessian = hessian + hessian' - Diagonal(hessian)
	end
    if maximum(denominator.(QV)) == 1
    else
        QV = NormalForms.snf(Matrix{Rational}((maximum(denominator.(QV)) .* QV)')).S'
    end
    x0 = rand(Uniform(0,2π),size(QV,1)) .* rand(size(QV,1))
    res = nlsolve(grad, hess, x0; ftol = ftol, iterations = iterations)
    xmin = res.zero
    # res, eigen(hess(xmin)).values
    if res.f_converged || res.x_converged
        if (sign(minimum(eigen(hess(xmin)).values)) ≥ 0. || abs(minimum(eigen(hess(xmin)).values)) ≤ 1e-10) && minimum(abs.(grad(xmin))) ≤ ftol
            hess_eigs = Float64(log10(abs(minimum(eigen(hess(xmin)).values)))) 
            hess_sign = sign((minimum(eigen(hess(xmin)).values)))
            grad_log = log10.(abs.(grad(xmin)))
            xmin = @.ifelse(abs(xmin) < ftol, zero(xmin), xmin)
            # xmin = @.ifelse(one(xmin) - mod(xmin / 2π, 1) < ftol || mod(xmin / 2π, 1) < ftol, zero(xmin), mod(xmin / 2π, 1)) ### this line removes some vacua!! (for rationals)
            keys = ["xinit", "xmin", "Heigs", "Hsign", "gradlog"]
            vals = [x0, xmin, hess_eigs, hess_sign, grad_log]
            return Dict(zip(keys,vals))
        end
    end
end

"""
    id_minimize(LV::Vector, QV::Vector; ftol = eps(), iterations = 1_000)

This function takes the instanton charge matrix and their corresponding scales and finds the corresponding minima using the `id_basis` method.
    !!! warning
    Currently cannot locate local minima -- eigs(hess).values returned as negative in `nlsolve`...

"""
function id_minimize(LV::Vector, QV::Vector; ftol = eps(), iterations = 1_000)
    @assert size(QV, 2) == size(LV, 1) "The number of columns of Q should be equal to the number of rows of Λ"
    function grad(x::Vector)
		grad_temp = LV' .* (QV .* sin.(x' * QV))
        sum(grad_temp, dims = 2)
	end
	function hess(x::Vector)
		hessian = zeros(size(x,1), size(x,1))
		cosine_phases = cos.(x' * QV)
		_phase_hessian!(hessian, LV, QV, cosine_phases)
		hessian = hessian + hessian' - Diagonal(hessian)
	end
    if maximum(denominator.(QV)) == 1
    else
        QV = NormalForms.snf((maximum(denominator.(QV)) .* Matrix{Rational}(QV)')).S'
    end
    x0 = rand(Uniform(0,2π),size(QV,1)) .* rand(size(QV,1))
    res = nlsolve(grad, hess, x0; ftol = ftol, iterations = iterations)
    xmin = res.zero
    # res, eigen(hess(xmin)).values
    # if res.f_converged || res.x_converged
    if sign(minimum(eigen(hess(xmin)).values)) ≥ 0. #&& minimum(abs.(grad(xmin))) ≤ ftol
        hess_eigs = Float64(log10(abs(minimum(eigen(hess(xmin)).values)))) 
        hess_sign = sign((minimum(eigen(hess(xmin)).values)))
        grad_log = log10.(abs.(grad(xmin)))
        xmin = @.ifelse(abs(xmin) < ftol, zero(xmin), xmin)
        # xmin = @.ifelse(one(xmin) - mod(xmin / 2π, 1) < ftol || mod(xmin / 2π, 1) < ftol, zero(xmin), mod(xmin / 2π, 1)) ### this line removes some vacua!! (for rationals)
        keys = ["xinit", "xmin", "Heigs", "Hsign", "gradlog"]
        vals = [x0, xmin, hess_eigs, hess_sign, grad_log]
        return Dict(zip(keys,vals))
    end
    # end
end

"""
    id_minima(LV::Vector, QV; ftol = eps(), iterations = 1_000)

Identify the distinct minima of the axion potential defined by the linear
instanton scales `LV` (``\\Lambda_a``) and charge matrix `QV`. `ftol` sets the
convergence tolerance and `iterations` caps the optimizer iterations. Return the
distinct minimum coordinates found.
"""
function id_minima(LV::Vector, QV; ftol = eps(), iterations = 1_000)
    if @isdefined h11
	else
		h11 = size(QV, 1)
	end
    @assert size(QV, 2) == size(LV, 1) "The number of rows of Λ should be equal to the number of columns of Q -- perhaps you need to transpose?"
    if maximum(denominator.(QV)) == 1 && size(QV, 1) == 1

    end

end
"""
    subspace_minimize(L, Q; runs=10_000, phase=zeros(max(collect(size(Q))...)))
Minimizes the subspace with `runs` iterations
"""
function subspace_minimize(L, Q; runs=10_000, phase::Matrix=zeros(max(collect(size(Q))...),1))
    xmin = []
    Random.seed!(9876543210)
	for _ in 1:runs, col in eachcol(phase)
		x0 = rand(Uniform(0,2π),size(Q,1)) .* rand(size(Q,1))
        x0 = x0 + col
		test_min = minimize(L, Q, x0)
		if test_min === nothing
		else
			push!(xmin, test_min["xmin"])
		end
	end
    push!(xmin, zeros(size(Q,1)))
	unique(xmin)
end

# function subspace_minimize(L, Q; runs=10_000, phase::Number=0)
#     xmin = []
#     Random.seed!(9876543210)
# 	for _ in 1:runs
# 		x0 = rand(Uniform(0,2π),size(Q,1)) .* rand(size(Q,1))
#         x0 = x0 .+ phase
# 		test_min = minimize(L, Q, x0)
# 		if test_min === nothing
# 		else
# 			push!(xmin, test_min["xmin"])
# 		end
# 	end
#     push!(xmin, zeros(size(Q,1)))
# 	unique(xmin)
# end

"""
    minima_lattice(v::Matrix{Float64})

Extract a lattice basis from the columns of `v`, a set of minima coordinate
vectors. Keep a maximal linearly independent set (columns whose Gram matrix has
a positive smallest eigenvalue) and discard near-zero columns. Return the matrix
of lattice basis vectors.
"""
function minima_lattice(v::Matrix{Float64})
    lattice_vectors = zeros(size(v, 1), 1)
    for col in eachcol(v)
        if sum(abs.(col)) < 1e-10
        else
            latt_temp = hcat(lattice_vectors, col)
            if eigmin(latt_temp' * latt_temp) > eps()
                lattice_vectors = latt_temp
            end
        end
    end

    keys = ["lattice_vectors"]
    vals = [lattice_vectors[:, 2:end]]
    return Dict(zip(keys,vals))
    GC.gc()
end



end
