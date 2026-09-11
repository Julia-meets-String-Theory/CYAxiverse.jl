#!/usr/bin/env julia

"""Issue 148 G3 prerequisite: exact off-ray geometry and local-control audit.

This script does not trace an off-ray catastrophe locus.  It reconstructs the
Appendix-D N=8 geometry in the source GLSM divisor basis, chooses one primitive
Kahler-cone direction, and audits the local map from two-cycle coordinates to
the source-twelve actions and amplitudes at the accepted radial event.

Run with:

    julia --startup-file=no --project=. scripts/audit_issue_148_g3_controls.jl
"""

using CYAxiverse
using HDF5
using LinearAlgebra
using Printf
using SHA
using Test

const PB = CYAxiverse.paper_benchmarks
const SOURCE_PDF_SHA256 =
    "b0f5539bf0fb40e401d93b8cfcbe3e725ba8849efdde2519646103d5f004d2e6"
const HDF5_SHA256 =
    "8fc483bc71a7e3f356b512adcfd44555f86a86bb765566c3f650984bf0ef1f4a"
const HDF5_PATH = joinpath(@__DIR__, "..", "paper_benchmarks", "appendix_c",
    "h11_008", "np_0000001", "cy_0000001", "cyax.h5")

# Paper Appendix D, Eqs. (90), (92), and (93).  CYTools orders the origin
# first, followed by the ten displayed vertices and the two extra points.
const SOURCE_VERTICES = Int[
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
const SOURCE_HEIGHTS = Int[0, 11, 11, 13, 13, 14, 14, 11, 11, 11, 11, 11, 12]
const SOURCE_BASIS_ZERO_BASED = Int[1, 3, 4, 5, 6, 9, 10, 11]

# Exact source-triangulation reconstruction with CYTools 1.4.12.  Each entry
# stores one sorted triple of zero-based indices and its symmetric kappa value.
const KAPPA_ENTRIES = (
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

# Exact toric Mori generators in the same two-cycle basis.  These are also the
# hyperplanes defining the toric Kahler cone.  Curve volumes are MORI * t.
const MORI = Int[
     0  0 -1  0 -1  0  2  1
     0  0  0  0  1  0 -1  0
     0  0 -1 -1  0  0  0  1
     1  0  1  0  0  0 -1  0
     1  1  0  1  0  0  0  0
     0 -1  0  0  0  1  0  1
     1  1  0  0  1 -2  0  0
     0  0  0  0  1 -1  0  0
     1  0  0  0  1 -1 -1  1
     0  0 -1  0  0  0  1  1
     0  0  1  0  0  0  0 -1
     0  0  0  1  0  0  1  0
     0  1  0  0  0 -1  1 -1
    -1  0  0 -1  0  0  0  0
     0  0  0  1  1  0  0  0
     0  0  0  1  0  1  0  0
     0  1  1  0  0  0  0 -2
     0  1  0  0  1 -2  0  0
     1  0  1  1  0  0  0  0
     1  0  0  0  0  0  0  0
    -1  0  0  0 -1  1  1  0
     0 -1  0 -1  0  0  0  1
     1  0  0  0  1 -1  0  0
     1  0  0  0  1  0 -1  0
     0  0  1  0  0  0 -1  0
     0  0  1  0  0  1 -1 -1
     0  0  0 -1  0  0  0  0
     1  0  0  1  0  0  1  0
     1  0  0  1  0  0  0  1
     0  0  0  0 -1  1  1  0
     0 -1  0  0 -1  2  0  1
     0  1  0  1  0  0  0  0
     0  1  0  0  0  0  0 -1
     1  1  0  0  0 -1  0  0
     1  0  1  0  1  0 -2  0
     0  0  1  1  0  0  0  0
     0  0  1  0  1  0 -2  0
     1  0  0  1  0  1  0  0
     0  1  0  0  0 -1  0  0
]

# Paper Table 1, sorted as in the published source-twelve potential.
const SOURCE_Q = Int[
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
const SOURCE_ACTIONS = Rational{BigInt}[
    14, 29//2, 29//2, 31//2, 31//2, 31//2,
    31//2, 16, 17, 17, 25, 45,
]

# Exact stretched-cone tip in two-cycle coordinates.  U is primitive, lies in
# the closure of the same Kahler cone, and is not proportional to T_REF.
const T_REF = Rational{BigInt}[1, 4, 4, -2, 4, 3, 3, 3]
const U = Rational{BigInt}[0, 1, 2, -1, 1, 1, 1, 1]
const ALPHA_MAX = 1//20

function exact_kappa()
    kappa = zeros(Rational{BigInt}, 8, 8, 8)
    for (i0, j0, k0, value) in KAPPA_ENTRIES
        i, j, k = i0 + 1, j0 + 1, k0 + 1
        for p in Set(((i,j,k), (i,k,j), (j,i,k), (j,k,i), (k,i,j), (k,j,i)))
            kappa[p...] = value
        end
    end
    kappa
end

function exact_geometry(t::AbstractVector{<:Rational})
    length(t) == 8 || throw(DimensionMismatch("expected eight two-cycle coordinates"))
    kappa = exact_kappa()
    A = [sum(kappa[i,j,k] * t[k] for k in 1:8) for i in 1:8, j in 1:8]
    tau = [sum(kappa[i,j,k] * t[j] * t[k] for j in 1:8, k in 1:8) / 2
        for i in 1:8]
    volume = sum(kappa[i,j,k] * t[i] * t[j] * t[k]
        for i in 1:8, j in 1:8, k in 1:8) / 6
    kinv = 4 .* (tau * tau' .- volume .* A)
    curves = Rational{BigInt}.(MORI) * t
    prime_divisors = Rational{BigInt}.(SOURCE_Q) * tau
    (; t=collect(t), A, tau, volume, kinv, curves, prime_divisors)
end

metric(g, ::Type{T}=BigFloat) where {T<:AbstractFloat} =
    Matrix(inv(Symmetric(T.(g.kinv))))

function numerical_rank(matrix::AbstractMatrix; relative_tolerance=1e-11)
    values = svdvals(Float64.(matrix))
    tolerance = relative_tolerance * maximum(values)
    count(>(tolerance), values), values, tolerance
end

function centered_rows(matrix::AbstractMatrix)
    matrix .- sum(matrix; dims=1) ./ size(matrix, 1)
end

sha256_file(path) = bytes2hex(sha256(read(path)))

@info "Issue 148 G3 local-control prerequisite"
@info "Source: arXiv:2608.14780v1, Appendix D, PDF SHA-256 $SOURCE_PDF_SHA256"
@info "Contract: source-twelve, zero phase, fixed saxions, period-one P96"
@info "No off-ray continuation or G2 reclassification is performed"

@testset "source geometry identity and exact reconstruction" begin
    @test sha256_file(HDF5_PATH) == HDF5_SHA256
    @test SOURCE_HEIGHTS == [0, 11, 11, 13, 13, 14, 14, 11, 11, 11, 11, 11, 12]
    @test rank(hcat(T_REF, U)) == 2
    @test foldl(gcd, abs.(numerator.(U)); init=big(0)) == 1

    reference = exact_geometry(T_REF)
    @test reference.volume == 126
    @test reference.tau == Rational{BigInt}[45,17,17,29//2,29//2,31//2,31//2,25]
    @test reference.prime_divisors == SOURCE_ACTIONS
    @test minimum(reference.curves) == 1
    @test maximum(reference.curves) == 3
    @test minimum(MORI * U) == 0
    @test all(>=(0), MORI * U)

    h5open(HDF5_PATH, "r") do file
        @test read(file["cytools/geometric/vertices"])' == SOURCE_VERTICES
        @test read(file["cytools/geometric/basis"]) == SOURCE_BASIS_ZERO_BASED
        @test isapprox(read(file["cytools/geometric/tip"]), Float64.(T_REF); atol=2e-14)
        @test isapprox(read(file["cytools/geometric/divisor_volumes"]),
            Float64.(reference.tau); atol=5e-13)
        @test isapprox(read(file["cytools/geometric/curve_volumes"]),
            Float64.(reference.curves); atol=5e-13)
        @test isapprox(read(file["cytools/geometric/CY_volume"]),
            Float64(reference.volume); atol=5e-12)
        @test isapprox(read(file["cytools/geometric/Kinv"]),
            Float64.(reference.kinv); atol=5e-11)
    end

    metric_exact = metric(reference)
    metric_root = BigFloat.(Matrix(PB.n8_geometry().kinetic))
    @test maximum(abs.(metric_exact .- metric_root)) < big"1e-18"
    @test isposdef(Symmetric(metric_exact))

    @info "  t_ref=$(T_REF), V=$(reference.volume), curve range=$(extrema(reference.curves))"
    @info "  source actions=$(reference.prime_divisors)"
    @info @sprintf("  exact/root P96 metric max error %.3e",
        Float64(maximum(abs.(metric_exact .- metric_root))))
end

reference = exact_geometry(T_REF)
endpoint = exact_geometry(T_REF .+ ALPHA_MAX .* U)

@testset "one-sided controlled alpha segment" begin
    # Paper k is a four-cycle scaling.  If s scales two-cycle coordinates,
    # then k=s^2, tau=k*tau_shape, V=k^(3/2)*V_shape, and K=k^-2*K_shape.
    augmented = PB.n8_degenerate_point()
    setup = PB._n8_setup_leading_charge(augmented.k)
    theta_seed = mod.(PB._n8_leading_to_theta(augmented.theta, setup.Qtilde), 1.0)
    refined = PB.n8_bigfloat_augmented_solve(theta_seed, augmented.k;
        precision_bits=256, tolerance=big"1e-65", max_iterations=800)
    @test refined.converged
    global G3_REFINED = refined

    k_c = refined.k
    s_c = sqrt(k_c)
    minimum_sampled_metric_eigenvalue = BigFloat(Inf)
    for index in 0:10
        alpha = index//200
        shape = exact_geometry(T_REF .+ alpha .* U)
        curves = s_c .* BigFloat.(shape.curves)
        primes = k_c .* BigFloat.(shape.prime_divisors)
        K = metric(shape) ./ k_c^2
        @test minimum(curves) > 0
        @test minimum(primes) > 1
        @test isposdef(Symmetric(K))
        minimum_sampled_metric_eigenvalue = min(minimum_sampled_metric_eigenvalue,
            minimum(eigvals(Symmetric(K))))
    end
    for (label, alpha, shape) in ((:radial, 0//1, reference),
            (:offray_endpoint, ALPHA_MAX, endpoint))
        curves = s_c .* BigFloat.(shape.curves)
        primes = k_c .* BigFloat.(shape.prime_divisors)
        volume = k_c * s_c * BigFloat(shape.volume)
        K = metric(shape) ./ k_c^2
        @test minimum(curves) > 0
        @test minimum(primes) > 1
        @test volume > 0
        @test isposdef(Symmetric(K))
        @info @sprintf("  %-15s alpha=%7.4f min_curve=%10.7f min_prime=%10.7f V=%12.7f eigmin(K)=%.6e",
            String(label), Float64(alpha), Float64(minimum(curves)),
            Float64(minimum(primes)), Float64(volume),
            Float64(minimum(eigvals(Symmetric(K)))))
    end
    @test all(endpoint.curves .>= reference.curves)
    @test all(endpoint.prime_divisors .>= reference.prime_divisors)
    @test endpoint.volume > reference.volume
    direction_geometry = exact_geometry(U)
    prime_slope_constant = Rational{BigInt}.(SOURCE_Q) * reference.A * U
    prime_slope_linear = Rational{BigInt}.(SOURCE_Q) * direction_geometry.A * U
    @test all(>=(0), prime_slope_constant)
    @test all(>=(0), prime_slope_linear)
    volume_slope_coefficients = (dot(reference.tau, U),
        dot(U, reference.A * U), dot(direction_geometry.tau, U))
    @test all(>=(0), volume_slope_coefficients)
    @test first(volume_slope_coefficients) > 0
    @info "  exact dV/dalpha polynomial coefficients=$(volume_slope_coefficients)"
    @info @sprintf("  eleven-point interval metric eigmin floor %.6e",
        Float64(minimum_sampled_metric_eigenvalue))
end

@testset "action and amplitude sensitivity ranks" begin
    refined = G3_REFINED
    k_c = refined.k
    T = BigFloat
    q = T.(SOURCE_Q)
    tau = T.(reference.tau)
    A = T.(reference.A)
    u = T.(U)
    volume = T(reference.volume)
    actions = k_c .* (q * tau)

    # Columns are the eight independent shape-coordinate perturbations at
    # fixed paper k.  The radial control is separately parameterized by log k.
    action_shape = k_c .* q * A
    log_reduced_shape = Diagonal(inv.(actions) .- T(2) * T(π)) * action_shape
    log_full_shape = log_reduced_shape .-
        T(2) .* ones(T, 12) * reshape(tau ./ volume, 1, :)
    relative_reduced_shape = centered_rows(log_reduced_shape)
    relative_full_shape = centered_rows(log_full_shape)

    rank_action, sv_action, tol_action = numerical_rank(action_shape)
    rank_log, sv_log, tol_log = numerical_rank(log_reduced_shape)
    rank_relative, sv_relative, tol_relative = numerical_rank(relative_full_shape)
    @test rank_action == 8
    @test rank_log == 8
    @test rank_relative == 8
    @test maximum(abs.(relative_full_shape .- relative_reduced_shape)) < big"1e-65"

    action_radial = actions                 # dS/d(log k)
    action_offray = action_shape * u        # dS/dalpha at fixed k
    action_offray_coefficients = Rational{BigInt}.(SOURCE_Q) * reference.A * U
    @test action_offray_coefficients == Rational{BigInt}[
        4, 12, 12, 3, 11, 3, 11, 6, 4, 12, 18, 30]
    log_radial_reduced = one(T) .- T(2) * T(π) .* actions
    log_radial_full = log_radial_reduced .- T(3) # V^-2, V ~ k^(3/2)
    log_offray_reduced = log_reduced_shape * u
    log_offray_full = log_full_shape * u
    radial_global_log_slope = -T(3)
    offray_global_log_slope = -T(2) * dot(tau, u) / volume
    action_controls = hcat(action_radial, action_offray)
    relative_controls = centered_rows(hcat(log_radial_full, log_offray_full))
    rank_action_controls, sv_action_controls, tol_action_controls =
        numerical_rank(action_controls)
    rank_relative_controls, sv_relative_controls, tol_relative_controls =
        numerical_rank(relative_controls)
    @test rank_action_controls == 2
    @test rank_relative_controls == 2

    radial_fit = dot(action_radial, action_offray) / dot(action_radial, action_radial)
    radial_residual = norm(action_offray .- radial_fit .* action_radial) /
        norm(action_offray)
    @test radial_residual > T("0.1")

    @info "  rank dS/dt = $rank_action, singular values=$(sv_action), tolerance=$tol_action"
    @info "  rank dlog(a_reduced)/dt = $rank_log, singular values=$(sv_log), tolerance=$tol_log"
    @info "  rank centered dlog(a_full)/dt = $rank_relative, singular values=$(sv_relative), tolerance=$tol_relative"
    @info "  [log(k), alpha] action rank=$rank_action_controls, singular values=$(sv_action_controls), tolerance=$tol_action_controls"
    @info "  [log(k), alpha] relative-amplitude rank=$rank_relative_controls, singular values=$(sv_relative_controls), tolerance=$tol_relative_controls"
    @info "  dS/dalpha=$(action_offray)"
    @info @sprintf("  V^-2 global log-amplitude slopes: d/dlog(k)=%.6f, d/dalpha=%.6f",
        Float64(radial_global_log_slope), Float64(offray_global_log_slope))
    @info @sprintf("  best radial fit to off-ray action control leaves relative residual %.6f",
        Float64(radial_residual))

    # Zero phases do not by themselves protect the radial cubic cancellation
    # against this control: measure the partial alpha derivative at the accepted
    # radial event, holding the event point and its P96-normalized null direction.
    metric_event = metric(reference) ./ k_c^2
    v = copy(refined.null_vector)
    v ./= sqrt(dot(v, metric_event * v))
    args = T(2) * T(π) .* (q * refined.theta)
    amplitudes = actions .* exp.(-T(2) * T(π) .* actions)
    max_index = argmax(amplitudes)
    normalized = amplitudes ./ amplitudes[max_index]
    qv = q * v
    cubic0 = -(T(2) * T(π))^3 *
        sum(normalized .* sin.(args) .* qv.^3)
    dlog_offray = log_offray_reduced
    dnormalized = normalized .* (dlog_offray .- dlog_offray[max_index])
    cubic_alpha = -(T(2) * T(π))^3 *
        sum(dnormalized .* sin.(args) .* qv.^3)

    delta = T("1e-7")
    shifted_actions = k_c .* q * T.(exact_geometry(T_REF .+ (1//10_000_000) .* U).tau)
    shifted_amplitudes = shifted_actions .* exp.(-T(2) * T(π) .* shifted_actions)
    shifted_normalized = shifted_amplitudes ./ shifted_amplitudes[max_index]
    cubic_shifted = -(T(2) * T(π))^3 *
        sum(shifted_normalized .* sin.(args) .* qv.^3)
    cubic_fd = (cubic_shifted - cubic0) / delta
    cubic_fd_relative_error = abs(cubic_fd - cubic_alpha) / abs(cubic_alpha)
    @test abs(cubic_alpha) > T("1e-8")
    @test cubic_fd_relative_error < T("1e-5")

    @info @sprintf("  normalized radial cubic %.9e", Float64(cubic0))
    @info @sprintf("  partial d(cubic)/dalpha %.9e", Float64(cubic_alpha))
    @info @sprintf("  one-sided finite-difference slope %.9e (relative error %.3e)",
        Float64(cubic_fd), Float64(cubic_fd_relative_error))
end

println("Issue 148 G3 prerequisite audit: PASS")
println("selected_direction_t=", U)
println("certified_alpha_interval=[0,", ALPHA_MAX, "]")
println("paper_k_equals_s_squared=true")
println("offray_continuation_started=false")
println("g2_classification_preserved=unresolved")
