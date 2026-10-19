#!/usr/bin/env julia

"""Reproduce the Issue 148 N=8 coordinate/metric normalization audit.

This script does not select a scientific convention.  It compares the two
existing contracts at the same potential-coordinate point:

  * paper coordinates `theta` with period one and `cos(2pi Q' theta)`, and
  * author-code coordinates `x = 2pi theta` with `cos(Q' x)`.

The author code uses the paper's numerical metric in the second contract.
The tensor transformation instead assigns `K_theta/(2pi)^2` to `x`.
"""

using CYAxiverse
using HDF5
using LinearAlgebra
using Printf

const Bench = CYAxiverse.paper_benchmarks
const Author = Bench.author_inflation
const TWO_PI = 2π
const C2 = TWO_PI^2

function symmetric_power(matrix::AbstractMatrix{<:Real}, power::Real)
    decomposition = eigen(Symmetric(Matrix(matrix)))
    decomposition.vectors * Diagonal(decomposition.values .^ power) *
        decomposition.vectors'
end

function require_close(label, actual, expected; atol=0.0, rtol=2e-11)
    isapprox(actual, expected; atol, rtol) ||
        error("$label: expected $expected, observed $actual")
end

# The root reconstruction retains the CYTools matrix precision.  The author
# fixture is the matrix printed to four significant figures in poly102_core.wl.
metric_precise = Matrix(Bench.n8_geometry().kinetic)
metric_author = Matrix(Author.N8_K_RAW)
geometry_file = joinpath(@__DIR__, "..", "paper_benchmarks", "appendix_c",
    "h11_008", "np_0000001", "cy_0000001", "cyax.h5")
metric_hdf5 = h5open(geometry_file, "r") do file
    Matrix(inv(Symmetric(read(file["cytools/geometric/Kinv"]))))
end
require_close("root/HDF5 precise kinetic matrix", metric_precise, metric_hdf5;
    atol=2e-18, rtol=2e-13)

k = Author.N8_KC
critical = Author.n8_degenerate_point()
x = critical.theta
theta = x / TWO_PI
potential = Author.n8_potential(k=k, trajectory=true)
author_derivatives = Author.n8_potential_derivatives(
    x, k; trajectory=true)

# Replay the period-one representation with the same ten terms and normalized
# amplitudes used by the author fixture.  This isolates the coordinate/metric
# question from the separate ten-versus-twelve-row source question.
arguments_period = TWO_PI .* (Matrix(potential.Q)' * theta) .+ potential.phases
amplitudes = author_derivatives.amplitudes
value_period = sum(amplitudes .* (1 .- cos.(arguments_period)))
gradient_period = TWO_PI .* Matrix(potential.Q) *
    (amplitudes .* sin.(arguments_period))
hessian_period = C2 .* Matrix(potential.Q) *
    Diagonal(amplitudes .* cos.(arguments_period)) * Matrix(potential.Q)'

require_close("potential arguments", arguments_period,
    author_derivatives.arguments; atol=3e-14)
require_close("potential value", value_period,
    author_derivatives.value; atol=1e-15)
require_close("raw gradient Jacobian", gradient_period,
    TWO_PI .* author_derivatives.gradient; atol=3e-13)
require_close("raw Hessian Jacobian", hessian_period,
    C2 .* author_derivatives.hessian; atol=3e-11)

# M is the paper's numerical period-one matrix at k.  The current author path
# also assigns M to raw radians, while coordinate covariance requires M/C2.
metric_period = metric_author / k^2
metric_raw_author = metric_period
metric_raw_tensor = metric_period / C2
to_raw_author = symmetric_power(metric_raw_author, -1 / 2)
to_raw_period = symmetric_power(metric_period, -1 / 2)
to_raw_tensor = symmetric_power(metric_raw_tensor, -1 / 2)

hessian_canonical_author = to_raw_author' * author_derivatives.hessian *
    to_raw_author
hessian_canonical_period = to_raw_period' * hessian_period * to_raw_period
hessian_canonical_tensor = to_raw_tensor' * author_derivatives.hessian *
    to_raw_tensor
require_close("canonical Hessian paper/author factor", hessian_canonical_period,
    C2 .* hessian_canonical_author; atol=3e-7, rtol=2e-11)
require_close("canonical Hessian coordinate covariance",
    hessian_canonical_period, hessian_canonical_tensor;
    atol=3e-7, rtol=2e-11)

delta_x = 1e-3 .* collect(1.0:8.0)
distance_author = sqrt(dot(delta_x, metric_raw_author * delta_x))
distance_tensor = sqrt(dot(delta_x, metric_raw_tensor * delta_x))
distance_period = sqrt(dot(delta_x / TWO_PI,
    metric_period * (delta_x / TWO_PI)))
require_close("distance covariance", distance_tensor, distance_period)
require_close("distance paper/author factor", distance_author,
    TWO_PI * distance_period)

# Evaluate representative slow-roll quantities away from the exact stationary
# point.  The ratios are algebraic consequences of the constant metric scale;
# they do not use the published e-fold benchmark as a fitting target.
x_probe = x .+ 2e-4 .* collect(1.0:8.0)
probe = Author.n8_potential_derivatives(x_probe, k; trajectory=true)
gradient_canonical_author = to_raw_author' * probe.gradient
gradient_canonical_tensor = to_raw_tensor' * probe.gradient
hessian_author_probe = to_raw_author' * probe.hessian * to_raw_author
hessian_tensor_probe = to_raw_tensor' * probe.hessian * to_raw_tensor
epsilon_author = dot(gradient_canonical_author, gradient_canonical_author) /
    (2 * probe.value^2)
epsilon_period = dot(gradient_canonical_tensor, gradient_canonical_tensor) /
    (2 * probe.value^2)
tangent_author = -gradient_canonical_author / norm(gradient_canonical_author)
tangent_period = -gradient_canonical_tensor / norm(gradient_canonical_tensor)
eta_author = dot(tangent_author, hessian_author_probe * tangent_author) /
    probe.value
eta_period = dot(tangent_period, hessian_tensor_probe * tangent_period) /
    probe.value
scalar_amplitude_author = probe.value^(3 / 2) /
    (sqrt(12) * π * norm(gradient_canonical_author))
scalar_amplitude_period = probe.value^(3 / 2) /
    (sqrt(12) * π * norm(gradient_canonical_tensor))
require_close("epsilon factor", epsilon_period, C2 * epsilon_author)
require_close("eta factor", eta_period, C2 * eta_author)
require_close("scalar-amplitude factor", scalar_amplitude_author,
    TWO_PI * scalar_amplitude_period)

# A fixed canonical direction gives the exact factors for every directional
# derivative.  Odd derivatives may be almost zero at the cusp, so compare the
# projected charges before evaluating them.
soft = eigen(Symmetric(hessian_author_probe)).vectors[:, 1]
charge_canonical_author = to_raw_author' * Matrix(potential.Q)
charge_canonical_period = TWO_PI .* charge_canonical_author
projected_author = charge_canonical_author' * soft
projected_period = charge_canonical_period' * soft
require_close("canonical charge factor", projected_period,
    TWO_PI .* projected_author; atol=3e-11)
third_author = -sum(probe.amplitudes .* sin.(probe.arguments) .*
    projected_author.^3)
third_period = -sum(probe.amplitudes .* sin.(probe.arguments) .*
    projected_period.^3)
fourth_author = -sum(probe.amplitudes .* cos.(probe.arguments) .*
    projected_author.^4)
fourth_period = -sum(probe.amplitudes .* cos.(probe.arguments) .*
    projected_period.^4)
require_close("third-derivative factor", third_period,
    TWO_PI^3 * third_author; rtol=2e-10)
require_close("fourth-derivative factor", fourth_period,
    TWO_PI^4 * fourth_author; rtol=2e-10)

relative_metric_rounding = norm(metric_author - metric_precise) /
    norm(metric_precise)
source_eigenvalues = eigvals(Symmetric(metric_precise))
author_eigenvalues = eigvals(Symmetric(metric_author))

println("Issue 148 N=8 metric-boundary replay")
@printf("author augmented k                         %.16f\n", critical.k)
@printf("metric precise/rounded relative Frobenius  %.6e\n",
    relative_metric_rounding)
println("precise metric eigenvalues                 ", source_eigenvalues)
println("rounded author metric eigenvalues          ", author_eigenvalues)
@printf("2pi                                         %.16f\n", TWO_PI)
@printf("(2pi)^2                                     %.16f\n", C2)
@printf("sample canonical distance, author contract  %.16e\n", distance_author)
@printf("sample canonical distance, paper contract   %.16e\n", distance_period)
@printf("distance ratio                              %.16f\n",
    distance_author / distance_period)
@printf("canonical Hessian matrix relative residual  %.6e\n",
    norm(hessian_canonical_period - C2 .* hessian_canonical_author) /
        norm(hessian_canonical_period))
@printf("epsilon ratio, paper/author                 %.16f\n",
    epsilon_period / epsilon_author)
@printf("eta ratio, paper/author                     %.16f\n",
    eta_period / eta_author)
@printf("scalar-amplitude ratio, author/paper        %.16f\n",
    scalar_amplitude_author / scalar_amplitude_period)
@printf("third directional derivative ratio          %.16f\n",
    third_period / third_author)
@printf("fourth directional derivative ratio         %.16f\n",
    fourth_period / fourth_author)
println("source scalar type                          Float64 (53 bits)")
println("BigFloat reruns promote the stored Float64 values; they do not add source digits")
