#!/usr/bin/env julia
"""Issue 148 G3: bounded positive-alpha local control evidence."""

using LinearAlgebra
using Printf
using Test
using CYAxiverse

const PB = CYAxiverse.paper_benchmarks
const ALPHA_MAX = 1 / 20

@info "=== Issue 148 G3 local-control evidence ==="
@info "Contract: source-twelve, zero phase, fixed saxions, period-one P96/CYTools"
@info "Julia $(VERSION), $(Sys.MACHINE)"

# Reconstruct the accepted G2 event in source coordinates before constructing
# either local unfolding seed.  The off-ray solver is not seeded from a grid
# population or an off-ray negative-alpha point.
g2 = PB.n8_degenerate_point()
g2_setup = PB._n8_setup_leading_charge(g2.k)
g2_theta = mod.(PB._n8_leading_to_theta(g2.theta, g2_setup.Qtilde), 1.0)
g2_refined = PB.n8_bigfloat_augmented_solve(g2_theta, g2.k;
    precision_bits=256, tolerance=BigFloat("1e-70"), max_iterations=500)
@test g2_refined.converged
g2_theta_f64 = Float64.(g2_refined.theta)
g2_null_f64 = Float64.(g2_refined.null_vector)
g2_k_f64 = Float64(g2_refined.k)

@info "── 1. Exact source geometry and radial event ──"
radial = PB.n8_g3_local_diagnostics(g2_theta_f64, g2_null_f64, g2_k_f64, 0.0;
    precision_bits=128)
endpoint = PB.n8_g3_local_diagnostics(g2_theta_f64, g2_null_f64, g2_k_f64, ALPHA_MAX;
    precision_bits=128)
@info @sprintf("  radial k=%.16e V=%.8f min_curve=%.8f min_prime=%.8f Kmin=%.6e",
    g2_k_f64, Float64(radial.volume), Float64(radial.min_curve_volume),
    Float64(radial.min_prime_divisor), Float64(radial.metric_min_eigenvalue))
@info @sprintf("  endpoint alpha=%.5f V=%.8f min_curve=%.8f min_prime=%.8f Kmin=%.6e",
    ALPHA_MAX, Float64(endpoint.volume), Float64(endpoint.min_curve_volume),
    Float64(endpoint.min_prime_divisor), Float64(endpoint.metric_min_eigenvalue))
@test radial.volume > 0 && endpoint.volume > radial.volume
@test radial.min_curve_volume > 0 && endpoint.min_curve_volume > 0
@test radial.min_prime_divisor > 1 && endpoint.min_prime_divisor > 1
@test radial.metric_min_eigenvalue > 0 && endpoint.metric_min_eigenvalue > 0
@test radial.full_factor
@test endpoint.full_factor
@test length(radial.action_vector) == 12 && length(radial.amplitudes) == 12
@test radial.source_precision_bits == 128

@info "── 2. Signed local unfolding seeds ──"
seed_minus = PB.n8_g3_seed_from_radial(g2_theta_f64, g2_null_f64, g2_k_f64;
    alpha_seed=1e-8, displacement=1e-4, side=-1, tolerance=1e-12)
seed_plus = PB.n8_g3_seed_from_radial(g2_theta_f64, g2_null_f64, g2_k_f64;
    alpha_seed=1e-8, displacement=1e-4, side=1, tolerance=1e-12)
@info "  side=-1: status=$(seed_minus.status), converged=$(seed_minus.converged), k=$(seed_minus.k), residual=$(seed_minus.residual)"
@info "  side=+1: status=$(seed_plus.status), converged=$(seed_plus.converged), k=$(seed_plus.k), residual=$(seed_plus.residual)"
@test seed_minus.converged
@test seed_minus.alpha > 0 && seed_minus.alpha < ALPHA_MAX
@test seed_minus.status == :seed_converged
@test seed_plus.status in (:seed_failed, :seed_max_iterations)

@info "── 3. Genuine bounded predictor/corrector continuation ──"
branch = PB.n8_g3_predictor_corrector(seed_minus.theta, seed_minus.null_vector,
    seed_minus.k; alpha0=seed_minus.alpha, ds=1e-8, n_steps=100,
    alpha_bounds=(0, ALPHA_MAX), k_bounds=(0.5, 0.9), tolerance=1e-10,
    max_corrector_iterations=40, min_ds=1e-12, branch_id=1)
@info "  positive branch: status=$(branch.status), termination=$(branch.termination_reason), steps=$(length(branch.steps))"
@test branch.status == :k_bounds_reached
@test length(branch.steps) > 10
@test branch.attempted_steps >= branch.accepted_steps
@test branch.rejected_steps >= 0
@test branch.source_data == :exact_integer_rational_table1
@test branch.metric_contract == :P96_CYTools
@test branch.control == :source12_positive_alpha
@test all(s -> s.converged && 0 <= s.alpha <= ALPHA_MAX, branch.steps)
@test all(s -> s.gradient_residual < 1e-10 && s.null_residual < 1e-10,
    branch.steps)
@test all(s -> s.normalization_residual < 1e-10, branch.steps)
@test all(s -> s.volume > 0 && s.min_curve_volume > 0 && s.min_prime_divisor > 1,
    branch.steps)
@test all(s -> s.metric_min_eigenvalue > 0 && s.full_factor_scale > 0,
    branch.steps)
@test all(s -> s.corrector_method == :predictor_corrector || s.corrector_method == :initial,
    branch.steps)
@test any(s -> s.rejected_steps >= 0, branch.steps)
@info @sprintf("  reached alpha=%.9e, k=%.9e, min metric=%.6e, min prime=%.6f, attempted=%d rejected=%d",
    branch.steps[end].alpha, branch.steps[end].k,
    branch.steps[end].metric_min_eigenvalue, branch.steps[end].min_prime_divisor,
    branch.attempted_steps, branch.rejected_steps)

# This is a maintained failure boundary: disabling the corrector cannot be
# mistaken for a completed branch.
failure = PB.n8_g3_predictor_corrector(seed_minus.theta, seed_minus.null_vector,
    seed_minus.k; alpha0=seed_minus.alpha, ds=1e-8, n_steps=1,
    max_corrector_iterations=0, min_ds=1e-9, branch_id=2)
@test failure.status == :step_failed
@test length(failure.steps) == 1
@info "  disabled-corrector boundary: status=$(failure.status), termination=$(failure.termination_reason)"

@info "── 4. Independent 128/256 exact-source refinement ──"
ref128 = PB.n8_g3_bigfloat_augmented_solve(seed_minus.theta, seed_minus.null_vector,
    seed_minus.k, seed_minus.alpha; precision_bits=128,
    tolerance=BigFloat("1e-30"), max_iterations=500)
ref256_independent = PB.n8_g3_bigfloat_augmented_solve(seed_minus.theta,
    seed_minus.null_vector, seed_minus.k, seed_minus.alpha; precision_bits=256,
    tolerance=BigFloat("1e-70"), max_iterations=500)
ref256_chained = PB.n8_g3_bigfloat_augmented_solve(ref128.theta, ref128.null_vector,
    ref128.k, ref128.alpha; precision_bits=256, tolerance=BigFloat("1e-70"),
    max_iterations=500)
@info "  128: converged=$(ref128.converged), iterations=$(ref128.iterations), |F|=$(ref128.gradient_residual)"
@info "  256 independent: converged=$(ref256_independent.converged), iterations=$(ref256_independent.iterations), |F|=$(ref256_independent.gradient_residual)"
@info "  256 chained: converged=$(ref256_chained.converged), iterations=$(ref256_chained.iterations), |F|=$(ref256_chained.gradient_residual)"
@info "  k refinement 128->256=$(ref256_chained.k-ref128.k), independent-vs-chained=$(ref256_independent.k-ref256_chained.k)"
@test ref128.converged && ref256_independent.converged && ref256_chained.converged
@test ref128.iterations > 1 && ref256_independent.iterations > 1 && ref256_chained.iterations > 1
@test ref128.gradient_residual < BigFloat("1e-29")
@test ref256_independent.gradient_residual < BigFloat("1e-60")
@test ref256_chained.gradient_residual < BigFloat("1e-60")
@test abs(ref256_chained.k - ref128.k) > BigFloat("1e-40")
@test abs(ref256_independent.k - ref256_chained.k) < BigFloat("1e-45")
@test ref256_chained.source_precision_bits == 256
@test ref256_chained.full_factor

@info "── 5. Independent equations, geometry, and local derivative evidence ──"
final_diag = PB.n8_g3_local_diagnostics(ref256_chained.theta,
    ref256_chained.null_vector, ref256_chained.k, ref256_chained.alpha;
    precision_bits=256)
@info @sprintf("  alpha=%.9e k=%.16e normalized |g|=%.3e |Hv|=%.3e |v|_K=%.16e",
    Float64(final_diag.alpha), Float64(final_diag.k),
    Float64(final_diag.gradient_residual), Float64(final_diag.null_residual),
    Float64(final_diag.metric_null_norm))
@info @sprintf("  full factor=%.6e min_action=%.6e max_amplitude=%.6e D3=%.6e D4=%.6e",
    Float64(final_diag.full_factor_scale), Float64(final_diag.min_action),
    Float64(final_diag.max_full_amplitude), Float64(final_diag.projected_d3),
    Float64(final_diag.projected_d4))
@test final_diag.normalization_residual < BigFloat("1e-60")
@test final_diag.min_curve_volume > 0 && final_diag.min_prime_divisor > 1
@test final_diag.metric_min_eigenvalue > 0
@test length(final_diag.action_vector) == 12 && length(final_diag.amplitudes) == 12
@test final_diag.source_data == :exact_integer_rational_table1
@test final_diag.metric_contract == :P96_CYTools
@test final_diag.alpha > 0 && final_diag.alpha < BigFloat(ALPHA_MAX)

# Re-evaluate the audited partial cubic control sensitivity at the accepted
# radial event using a one-sided exact shape perturbation.  This is the local
# unfolding witness; it is not a total derivative along a catastrophe locus.
radial_geom = PB._g3_geometry(g2_refined.k, BigFloat(0))
radial_deriv = PB._g3_geometry_derivatives(g2_refined.k, BigFloat(0), radial_geom)
radial_sys = PB._g3_augmented_system(g2_refined.theta, g2_refined.null_vector,
    g2_refined.k, BigFloat(0), radial_geom, radial_deriv)
vmetric = g2_refined.null_vector /
    sqrt(dot(g2_refined.null_vector, radial_geom.metric * g2_refined.null_vector))
qv = radial_geom.Q' * vmetric
args = radial_sys.args
amp = radial_geom.normalized_amplitudes
damp = radial_deriv.amp_alpha
cubic_alpha = -(BigFloat(2) * BigFloat(π))^3 *
    sum(damp .* radial_sys.sine .* qv.^3)
@info "  partial d(D3)/dalpha at radial event=$(cubic_alpha)"
@test abs(cubic_alpha) > BigFloat("1e-8")

# Check the same sensitivity by an independent one-sided finite difference of
# the full-factor diagnostic.  The state is held at the audited radial event,
# so this checks the local control derivative rather than a continuation fit.
radial_diag = PB._g3_step_diagnostics(g2_refined.theta, g2_refined.null_vector,
    g2_refined.k, BigFloat(0), radial_geom, radial_sys)
radial_d3_normalized = -(BigFloat(2) * BigFloat(π))^3 *
    sum(radial_geom.normalized_amplitudes .* radial_sys.sine .* qv.^3)
delta_alpha = BigFloat("1e-10")
shift_geom = PB._g3_geometry(g2_refined.k, delta_alpha)
shift_deriv = PB._g3_geometry_derivatives(g2_refined.k, delta_alpha, shift_geom)
shift_sys = PB._g3_augmented_system(g2_refined.theta, g2_refined.null_vector,
    g2_refined.k, delta_alpha, shift_geom, shift_deriv)
shift_diag = PB._g3_step_diagnostics(g2_refined.theta, g2_refined.null_vector,
    g2_refined.k, delta_alpha, shift_geom, shift_sys)
shift_qv = shift_geom.Q' * vmetric
shift_d3_normalized = -(BigFloat(2) * BigFloat(π))^3 *
    sum(shift_geom.normalized_amplitudes .* shift_sys.sine .* shift_qv.^3)
fd_cubic_alpha = (shift_d3_normalized - radial_d3_normalized) / delta_alpha
@info "  finite-difference normalized D3 sensitivity=$(fd_cubic_alpha)"
@info "  full-factor D3 sensitivity=$( (shift_diag.projected_d3 - radial_diag.projected_d3) / delta_alpha )"
@test abs(fd_cubic_alpha - cubic_alpha) < abs(cubic_alpha) * BigFloat("1e-5")

@info "=== G3 evidence summary ==="
@info "  fate=positive-alpha branch terminates at k lower bound after genuine predictor/corrector steps"
@info "  opposing signed seed at positive alpha fails conservatively; no negative-alpha claim"
@info "  G2 radial classification preserved as unresolved"
println("Issue 148 G3 local-control evidence: PASS")
println("positive_branch_status=", branch.status)
println("positive_branch_termination=", branch.termination_reason)
println("positive_branch_steps=", length(branch.steps))
println("opposing_seed_status=", seed_plus.status)
println("g2_classification_preserved=unresolved")
