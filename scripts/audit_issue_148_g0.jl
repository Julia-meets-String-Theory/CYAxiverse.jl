#!/usr/bin/env julia

"""Focused numeric replay for the Issue 148 G0 benchmark audit.

This script is intentionally audit-only. It derives the N=5 reduced ratio
from the stored action data, compares both public benchmark namespaces, and
replays the independent 12-term and author-code 10-term N=8 augmented solves.
"""

using CYAxiverse
using LinearAlgebra
using Printf

const Bench = CYAxiverse.paper_benchmarks
const Poly102 = Bench.poly102_inflation

function require_close(label, actual, expected; atol=0.0, rtol=1e-12)
    isapprox(actual, expected; atol, rtol) ||
        error("$label: expected $expected, observed $actual")
end

source_n5 = Bench.n5_potential()
source_n5.qdotτ == [6, 6.25, 24, 26, 31.875, 32, 36.125, 162.125] ||
    error("the stored N=5 actions changed")

# Rows five and six are the two terms retained in the source light-direction
# reduction. Their coefficient ratio includes both q·tau prefactors.
q1tau, q2tau = source_n5.qdotτ[5], source_n5.qdotτ[6]
source_ratio(k) = (q2tau / q1tau) * exp(-2π * k * (q2tau - q1tau))
source_kc = log(4 * q2tau / q1tau) / (2π * (q2tau - q1tau))
closed_form_kc = (4 / π) * log(1024 / 255)

root_kc = Bench.n5_critical_scale()
poly102_kc = Poly102.n5_critical_scale()
root_at_root = Bench.n5_reduced_ratio(root_kc)
root_at_poly102 = Bench.n5_reduced_ratio(poly102_kc)
poly102_at_root = Poly102.n5_reduced_ratio(root_kc)
source_curvature_at_poly102 = -1 + 4 * source_ratio(poly102_kc)
poly102_curvature_at_root = -1 + 4 * poly102_at_root

# Read the ratio from the raw L data as a separate implementation-level check.
function raw_n5_ratio(k)
    potential = Poly102.n5_potential(k=k)
    sign_ratio = potential.L[1, 6] / potential.L[1, 5]
    sign_ratio * 10.0^(potential.L[2, 6] - potential.L[2, 5])
end

require_close("closed-form N=5 scale", source_kc, closed_form_kc; atol=2e-15)
require_close("root N=5 scale", root_kc, source_kc; atol=2e-15)
require_close("source ratio at source scale", root_at_root, 0.25; atol=2e-15)
require_close("raw/source ratio at source scale", raw_n5_ratio(root_kc),
    source_ratio(root_kc); rtol=1e-12)
require_close("raw/source ratio at poly102 scale", raw_n5_ratio(poly102_kc),
    source_ratio(poly102_kc); rtol=1e-12)
require_close("poly102 N=5 scale is the N=8 constant", poly102_kc, Poly102.N8_KC;
    atol=1e-15)
abs(root_kc - poly102_kc) > 1 || error("the expected N=5 namespace discrepancy is absent")

root_low = Bench.n5_reduced_critical_points(root_kc - 1e-4)
root_high = Bench.n5_reduced_critical_points(root_kc + 1e-4)
root_at = Bench.n5_reduced_critical_points(root_kc)
root_low.minima == 2 || error("source N=5 low-side minima count changed")
root_high.minima == 1 || error("source N=5 high-side minima count changed")
root_at.hessian_sign[2] == 0 || error("source N=5 cusp curvature is not zero")

artificial_diagnostic = Bench.n5_catastrophe_diagnostic(precision_bits=120)
source_scale_diagnostic = Bench.n5_catastrophe_diagnostic(
    k=root_kc, precision_bits=120)
artificial_diagnostic.classification == :cusp ||
    error("the existing synthetic N=5 diagnostic no longer reports a cusp")
source_scale_diagnostic.classification == :unresolved ||
    error("the defective N=5 diagnostic unexpectedly validates the source scale")

n8_seed = [
    0.0, 0.00499839, 0.99500161, 0.75995156,
    0.75004523, 0.24995477, 0.0, 0.75495317,
]
n8_table1 = Bench.n8_degenerate_point(n8_seed)
n8_author = Poly102.n8_degenerate_point()
n8_table1.converged || error("the 12-term N=8 augmented solve did not converge")
n8_author.converged || error("the 10-term N=8 augmented solve did not converge")
require_close("12-term N=8 scale", n8_table1.k, Poly102.N8_KC; atol=2e-12)
require_close("10-term N=8 scale", n8_author.k, Poly102.N8_KC; atol=2e-12)
n8_table1.gradient_residual < 1e-11 || error("12-term N=8 gradient residual failed")
n8_table1.null_residual < 1e-11 || error("12-term N=8 null residual failed")
n8_author.gradient_residual < 1e-11 || error("10-term N=8 gradient residual failed")
n8_author.null_residual < 1e-11 || error("10-term N=8 null residual failed")

n5_metric_eigenvalues = eigvals(Hermitian(Poly102.N5_K_RAW))
n8_metric_eigenvalues = eigvals(Hermitian(Poly102.N8_K_RAW))
n5_paper_rounded = reverse([3.57e-3, 2.20e-4, 1.06e-4, 8.05e-5, 2.54e-5])
n8_paper_rounded = reverse([
    8.20e-4, 6.35e-4, 5.97e-4, 3.13e-4,
    1.24e-4, 9.15e-5, 8.30e-5, 5.84e-5,
])
all(isapprox.(n5_metric_eigenvalues, n5_paper_rounded; rtol=3e-3)) ||
    error("stored N=5 metric eigenvalues do not match the paper's rounded table")
all(isapprox.(n8_metric_eigenvalues, n8_paper_rounded; rtol=3e-3)) ||
    error("stored N=8 metric eigenvalues do not match the paper's rounded table")

@printf("root/source n5 critical scale       %.16f\n", root_kc)
@printf("poly102 n5 critical scale           %.15f\n", poly102_kc)
@printf("difference                          %.16f\n", root_kc - poly102_kc)
@printf("source ratio at source scale        %.16f\n", source_ratio(root_kc))
@printf("source ratio at poly102 scale       %.16f\n", source_ratio(poly102_kc))
@printf("source curvature at theta=pi there  %.16f\n", source_curvature_at_poly102)
@printf("poly102 ratio at source scale       %.15f\n", poly102_at_root)
@printf("poly102 curvature there             %.15f\n", poly102_curvature_at_root)
println("existing diagnostic at artificial scale  ", artificial_diagnostic.classification)
println("existing diagnostic at source scale      ", source_scale_diagnostic.classification)
@printf("12-term N8 k                       %.16f\n", n8_table1.k)
@printf("12-term N8 gradient residual       %.3e\n", n8_table1.gradient_residual)
@printf("12-term N8 null residual           %.3e\n", n8_table1.null_residual)
println("12-term hierarchy-scaled Hessian eigenvalues[1:2]  ",
    n8_table1.eigenvalues[1:2])
@printf("10-term N8 k                       %.15f\n", n8_author.k)
@printf("10-term N8 gradient residual       %.3e\n", n8_author.gradient_residual)
@printf("10-term N8 null residual           %.3e\n", n8_author.null_residual)
println("N5 metric eigenvalues  ", n5_metric_eigenvalues)
println("N8 metric eigenvalues  ", n8_metric_eigenvalues)
println("issue-148 G0 focused numeric audit: PASS")
