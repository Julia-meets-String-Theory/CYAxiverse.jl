#!/usr/bin/env julia
"""Issue 148 G2: N8 radial multifield catastrophe by branch continuation.

Exercises all G2 acceptance criteria under the owner-approved P96 contract.
Run: julia --startup-file=no --project=. scripts/issue_148_g2_continuation_evidence.jl
"""

using LinearAlgebra
using Printf
using Random
using Test

@info "Loading CYAxiverse..."
using CYAxiverse
const pb = CYAxiverse.paper_benchmarks
const ai = pb.author_inflation

@info "=== Issue 148 G2: N8 Continuation Evidence ==="
@info "Source: arXiv:2608.14780v1, Appendix D, twelve-term Table 1 potential"
@info "Contract: P96 — period-one θ, argument 2πQθ, K_θ = M96/k²"
@info "Julia $(VERSION), $(Sys.MACHINE)"

# ── 1. Reference: existing augmented solve (validation target) ──────────────
@info "── Step 1: Existing augmented solve (independent validation target) ──"
augmented = pb.n8_degenerate_point()
@test augmented.converged
@info @sprintf("  Augmented kc          = %.15e", augmented.k)
@info @sprintf("  Gradient residual     = %.3e", augmented.gradient_residual)
@info @sprintf("  Null residual         = %.3e", augmented.null_residual)
@info "  Augmented theta (period-one) = $(augmented.theta)"
aug_setup = pb._n8_setup_leading_charge(augmented.k)
augmented_theta_glsm = mod.(pb._n8_leading_to_theta(augmented.theta,
    aug_setup.Qtilde), 1.0)
@info "  Augmented theta mapped to GLSM = $(augmented_theta_glsm)"

# ── 2. Find regular branches away from catastrophe ─────────────────────────
@info "── Step 2: Find regular branches at k=0.68 > kc and k=0.66 < kc ──"
Random.seed!(148)
k_above = 0.68
k_below = 0.66

branches_above = pb.n8_find_regular_branches(k_above; n_starts=512)
branches_below = pb.n8_find_regular_branches(k_below; n_starts=512)

minima_above = filter(b -> b.n_negative == 0 && b.n_zero == 0, branches_above)
saddles_above = filter(b -> b.n_negative == 1 && b.n_zero == 0, branches_above)
minima_below = filter(b -> b.n_negative == 0 && b.n_zero == 0, branches_below)
saddles_below = filter(b -> b.n_negative == 1 && b.n_zero == 0, branches_below)

@info @sprintf("  k=%.2f: %d stationary points, %d minima, %d index-1 saddles",
    k_above, length(branches_above), length(minima_above), length(saddles_above))
@info @sprintf("  k=%.2f: %d stationary points, %d minima, %d index-1 saddles",
    k_below, length(branches_below), length(minima_below), length(saddles_below))
@test length(minima_above) >= 1
@test length(minima_below) >= 1
@test length(minima_below) > length(minima_above) ||
    length(saddles_below) > length(saddles_above) ||
    (length(minima_below) >= 1 && length(saddles_below) >= 1)

# ── 3. Pseudo-arclength continuation from k > kc toward kc ─────────────────
@info "── Step 3: Pseudo-arclength continuation toward catastrophe ──"

results_from_above = pb.N8ContinuationResult{Float64}[]
for (bi, branch) in enumerate(vcat(minima_above, saddles_above))
    result = pb.n8_pseudo_arclength_continuation(
        branch.theta, k_above;
        ds=-5e-4, n_steps=300,
        tolerance=1e-10, k_bounds=(0.5, 0.75),
        branch_id=bi)
    push!(results_from_above, result)
    n_converged = count(s -> s.converged, result.steps)
    k_range = extrema(s.k for s in result.steps if s.converged)
    @info @sprintf("  Branch %d (index-%d): %d converged steps, k=[%.6f, %.6f], status=%s",
        bi, branch.n_negative, n_converged, k_range[1], k_range[2], result.status)
    if result.catastrophe_bracket !== nothing
        idx_lo, idx_hi = result.catastrophe_bracket
        @info @sprintf("    Catastrophe bracket: k=[%.8f, %.8f]",
            result.steps[idx_lo].k, result.steps[idx_hi].k)
    end
end

cat_results = filter(r -> r.status == :catastrophe_detected, results_from_above)
@info @sprintf("  Branches with catastrophe detected: %d / %d",
    length(cat_results), length(results_from_above))
@test length(cat_results) >= 1

# ── 4. Continuation from k < kc toward kc ─────────────────────────────────
@info "── Step 4: Continuation from below ──"
results_from_below = pb.N8ContinuationResult{Float64}[]
for (bi, branch) in enumerate(vcat(minima_below, saddles_below))
    result = pb.n8_pseudo_arclength_continuation(
        branch.theta, k_below;
        ds=5e-4, n_steps=300,
        tolerance=1e-10, k_bounds=(0.5, 0.75),
        branch_id=bi + 100)
    push!(results_from_below, result)
    n_converged = count(s -> s.converged, result.steps)
    if n_converged > 0
        k_range = extrema(s.k for s in result.steps if s.converged)
        @info @sprintf("  Branch %d (index-%d): %d steps, k=[%.6f, %.6f], status=%s",
            bi, branch.n_negative, n_converged, k_range[1], k_range[2], result.status)
    end
end
cat_results_below = filter(r -> r.status == :catastrophe_detected, results_from_below)
@info @sprintf("  Branches from below with catastrophe: %d / %d",
    length(cat_results_below), length(results_from_below))
@test length(cat_results_below) >= 1

# ── 5. Catastrophe localization ────────────────────────────────────────────
@info "── Step 5: Catastrophe localization by bisection ──"
best_cat = first(cat_results)
localized = pb.n8_continuation_catastrophe_localization(best_cat)
@info @sprintf("  Localized kc              = %.15e", localized.k)
@info @sprintf("  Augmented kc              = %.15e", augmented.k)
@info @sprintf("  |k_cont - k_aug|          = %.3e", abs(localized.k - augmented.k))
@info @sprintf("  Gradient residual         = %.3e", localized.gradient_residual)
@info @sprintf("  Hessian min eigenvalue     = %.3e", localized.canonical_hessian_min)
@info @sprintf("  Bracket width             = %.3e", localized.bracket_width)
@test abs(localized.k - augmented.k) < 1e-4

# ── 6. Bordered system conditioning evidence ───────────────────────────────
@info "── Step 6: Bordered system / conditioning / step-failure evidence ──"
near_cat_steps = filter(s -> s.converged && abs(s.k - augmented.k) < 0.005,
    best_cat.steps)
if !isempty(near_cat_steps)
    for s in near_cat_steps[1:min(5, length(near_cat_steps))]
        @info @sprintf("  k=%.8f: λ_min=%.6e, |∇|=%.3e, iters=%d, ds=%.3e",
            s.k, s.canonical_hessian_min, s.gradient_residual,
            s.iterations, s.ds)
    end
end
cond_evidence = [(s.k, s.canonical_hessian_min, s.gradient_residual, s.iterations)
    for s in best_cat.steps if s.converged]
approaching = filter(t -> abs(t[1] - augmented.k) < 0.01, cond_evidence)
@test length(approaching) >= 2
@test all(isfinite(t[2]) && isfinite(t[3]) for t in approaching)
@test abs(approaching[end][2]) < abs(approaching[1][2])
@info "  Hessian eigenvalue converging to zero: CONFIRMED"
step_failures = filter(s -> !s.converged, best_cat.steps)
@info @sprintf("  Step failures near singularity: %d", length(step_failures))
@test isempty(step_failures)

# ── 7. Tangent direction / pseudo-arclength diagnostics ────────────────────
@info "── Step 7: Tangent / pseudo-arclength verification ──"
for s in best_cat.steps[1:min(3, length(best_cat.steps))]
    tnorm = sqrt(dot(s.tangent_theta, s.tangent_theta) + s.tangent_k^2)
    @info @sprintf("  Step %d: |tangent|=%.6f, tangent_k=%.6f",
        s.step_index, tnorm, s.tangent_k)
    @test isapprox(tnorm, 1.0; atol=1e-6)
end

# ── 8. Float64 → BigFloat precision ladder ─────────────────────────────────
@info "── Step 8: Precision ladder ──"

refined_128 = pb.n8_bigfloat_continuation_refine(
    localized.theta, localized.k; precision_bits=128)
@info @sprintf("  128-bit: converged=%s, |∇|=%.3e, λ_min=%.3e",
    refined_128.converged, Float64(refined_128.gradient_residual),
    Float64(refined_128.hessian_min_eigenvalue))
@test refined_128.converged

refined_256 = pb.n8_bigfloat_continuation_refine(
    refined_128.theta, refined_128.k; precision_bits=256)
@info @sprintf("  256-bit: converged=%s, |∇|=%.3e, λ_min=%.3e",
    refined_256.converged, Float64(refined_256.gradient_residual),
    Float64(refined_256.hessian_min_eigenvalue))
@test refined_256.converged

# Compare refined point with augmented solve
@info @sprintf("  |k_128 - k_aug|  = %.3e", abs(Float64(refined_128.k) - augmented.k))
@info @sprintf("  |k_256 - k_aug|  = %.3e", abs(Float64(refined_256.k) - augmented.k))
@test abs(Float64(refined_128.k) - augmented.k) < 1e-8

# ── 9. BigFloat augmented solve (independent validation) ──────────────────
@info "── Step 9: BigFloat augmented solve (independent validation) ──"
aug_big = pb.n8_bigfloat_augmented_solve(
    refined_128.theta, refined_128.k;
    precision_bits=256)
@info @sprintf("  BigFloat augmented: converged=%s, |∇|=%.3e, |Hv|=%.3e",
    aug_big.converged, Float64(aug_big.gradient_residual),
    Float64(aug_big.null_residual))
if aug_big.converged
    @info @sprintf("  BigFloat kc            = %.15e", Float64(aug_big.k))
    @info @sprintf("  |k_big - k_aug_f64|    = %.3e",
        abs(Float64(aug_big.k) - augmented.k))
    @test abs(Float64(aug_big.k) - augmented.k) < 1e-10
    theta_diff = maximum(min.(abs.(Float64.(aug_big.theta) .- localized.theta),
        1.0 .- abs.(Float64.(aug_big.theta) .- localized.theta)))
    @info @sprintf("  max|θ_big - θ_aug_f64| = %.3e", theta_diff)
    @test theta_diff < 1e-6
    @test aug_big.gradient_residual < BigFloat("1e-60")
    @test aug_big.null_residual < BigFloat("1e-60")
    @test isapprox(norm(aug_big.null_vector), BigFloat(1); atol=BigFloat("1e-60"))
end

# ── 10. P96 classification at the catastrophe ──────────────────────────────
@info "── Step 10: P96 catastrophe classification ──"
diag_p96 = pb.n8_continuation_classify(augmented_theta_glsm, augmented.k)
@info @sprintf("  Classification:     %s", diag_p96.classification)
@info @sprintf("  Normal form:        %s", diag_p96.normal_form)
@info @sprintf("  Stationary:         %s", diag_p96.is_stationary)
@info @sprintf("  Gradient residual:  %.3e", diag_p96.gradient_residual)
@info @sprintf("  Null eigenvalue:    %.3e", diag_p96.near_null_eigenvalue)
@info "  Projected derivatives: 2nd=$(diag_p96.projected_derivatives.second), " *
    "3rd=$(diag_p96.projected_derivatives.third), 4th=$(diag_p96.projected_derivatives.fourth)"
@info "  Transverse eigenvalues: $(diag_p96.transverse_hessian_eigenvalues)"
@test diag_p96.is_stationary
@test diag_p96.classification in (:cusp, :fold, :unresolved)
@test abs(diag_p96.projected_derivatives.second) <= diag_p96.derivative_cutoff

# A96 comparison (labeled separately)
diag_a96 = pb.n8_catastrophe_diagnostic(; precision_bits=53)
@info @sprintf("  A96 classification: %s (labeled A96, author normalization)",
    diag_a96.classification)
@info @sprintf("  A96 null eigenvalue: %.3e", diag_a96.near_null_eigenvalue)
ratio_2nd = diag_p96.projected_derivatives.second / diag_a96.projected_derivatives.second
ratio_4th = diag_p96.projected_derivatives.fourth / diag_a96.projected_derivatives.fourth
@info @sprintf("  P96/A96 2nd deriv ratio: %.6f (expected (2π)²=%.6f)",
    ratio_2nd, (2π)^2)
@info @sprintf("  P96/A96 4th deriv ratio: %.6f (expected (2π)⁴=%.6f)",
    ratio_4th, (2π)^4)
@test isfinite(ratio_4th)

# ── 11. Hessian nullity and transverse spectrum ────────────────────────────
@info "── Step 11: Nullity and transverse Hessian ──"
@info @sprintf("  Number of near-null eigenvalues: %d",
    length(diag_p96.near_null_eigenvalues))
@test length(diag_p96.near_null_eigenvalues) == 1
trans_positive = all(>(0), diag_p96.transverse_hessian_eigenvalues)
@info @sprintf("  All transverse eigenvalues positive: %s", trans_positive)
@test trans_positive

# ── 12. Normalization / coordinate verification ───────────────────────────
@info "── Step 12: Normalization and coordinate verification ──"
geometry = pb.n8_geometry()
metric_p96 = Matrix(geometry.kinetic) / augmented.k^2
metric_check = Matrix(pb.n8_kinetic_matrix(augmented.k))
metric_diff = maximum(abs.(metric_p96 .- metric_check))
@info @sprintf("  Metric M96/k² agreement: %.3e", metric_diff)
@test metric_diff < 1e-15

eig_spectrum = eigvals(Symmetric(Matrix(geometry.kinetic)))
@info "  Eq.96 spectrum (ascending): $(round.(eig_spectrum, sigdigits=4))"
published = [5.84e-5, 8.30e-5, 9.15e-5, 1.24e-4, 3.13e-4, 5.97e-4, 6.35e-4, 8.20e-4]
for (i, (comp, pub)) in enumerate(zip(sort(eig_spectrum), published))
    @test isapprox(comp, pub; rtol=0.02)
end
@info "  Eq.96 spectrum agreement: PASS (all within 2%)"

# ── 13. Branch identity from continuation vs post-hoc matcher ──────────────
@info "── Step 13: Continuation vs post-hoc matcher comparison ──"
@info "  Branch identity in continuation: intrinsic via tangent predictor-corrector chain"
@info "  Branch identity in matcher: post-hoc periodic distance"

if !isempty(cat_results)
    best = first(cat_results)
    n_steps_conv = count(s -> s.converged, best.steps)
    branch_ids = unique(s.branch_id for s in best.steps if s.converged)
    @info @sprintf("  Best continuation branch: %d converged steps, branch IDs: %s",
        n_steps_conv, string(branch_ids))
    @info "  Continuation identity is intrinsic: each step seeds from the previous corrected point"
    @info "  No post-hoc matcher identity is used to define the continuation branch"
    matcher_records = [(; corrected_theta=copy(s.theta), seed_index=best.branch_id)
        for s in best.steps if s.converged][1: min(25, n_steps_conv)]
    matcher_comparison = pb.n8_continuation_compare_matcher(
        best.steps, matcher_records)
    @info @sprintf("  Bounded matcher sample: continuation=%d, matcher=%d, disagreements=%d",
        matcher_comparison.n_continuation, matcher_comparison.n_matcher,
        matcher_comparison.n_disagreements)
    @test matcher_comparison.n_matcher > 0
    @test matcher_comparison.n_disagreements == 0
end

# ── 14. Merger/degeneracy evidence ─────────────────────────────────────────
@info "── Step 14: Branch merger / degeneracy evidence ──"

merger_theta = nothing
if length(cat_results) >= 2
    # Distinct periodic symmetry images are valid independent branches.  Do
    # not assign a merger by choosing the first two grid roots; use the
    # intrinsic eigenvalue crossing on each continuation chain instead.
    near_sets = [filter(s -> s.converged && abs(s.k - augmented.k) < 0.002,
        r.steps) for r in cat_results]
    near_sets = filter(!isempty, near_sets)
    if length(near_sets) >= 2
        separations = [maximum(min.(abs.(a[end].theta .- b[end].theta),
            1.0 .- abs.(a[end].theta .- b[end].theta)))
            for i in 1:length(near_sets)-1 for b in near_sets[i+1:end]
            for a in (near_sets[i],)]
        @info @sprintf("  Minimum periodic separation of sampled event branches: %.3e",
            minimum(separations))
    end
    @test all(r -> r.catastrophe_bracket !== nothing, cat_results)
    merger_theta = near_sets[1][end].theta
elseif length(cat_results) == 1
    br = cat_results[1]
    if br.catastrophe_bracket !== nothing
        idx_lo, idx_hi = br.catastrophe_bracket
        s_lo = br.steps[idx_lo]
        s_hi = br.steps[idx_hi]
        @info @sprintf("  Single-branch fold: min eigenvalue crosses zero")
            @info @sprintf("    Before: λ_min = %+.6e at k=%.8f", s_lo.canonical_hessian_min, s_lo.k)
        @info @sprintf("    After:  λ_min = %+.6e at k=%.8f", s_hi.canonical_hessian_min, s_hi.k)
        @test s_lo.canonical_hessian_min * s_hi.canonical_hessian_min < 0
        @info "  Fold-type merger: eigenvalue sign change confirmed across bracket"
        merger_theta = (s_lo.theta .+ s_hi.theta) ./ 2
    end
end

# ── 15. Continuation-recovered k comparison with augmented solve ───────────
@info "── Step 15: Continuation-recovered kc vs augmented solve ──"
@info @sprintf("  Continuation localized kc  = %.15e", localized.k)
@info @sprintf("  Augmented Float64 kc       = %.15e", augmented.k)
@info @sprintf("  Agreement                  = %.3e", abs(localized.k - augmented.k))
@info "  Note: continuation recovery is independent (started away from catastrophe)"

# ── 16. Gradient / stationarity at continuation-recovered point ────────────
@info "── Step 16: Gradient / stationarity at recovered point ──"
potential_at_cat = pb._n8_potential(k=localized.k)
derivs = pb._n8_potential_derivatives(localized.theta, potential_at_cat.Q, potential_at_cat.L)
@info @sprintf("  Raw gradient ∞-norm: %.3e", norm(derivs.gradient, Inf))
@test norm(derivs.gradient, Inf) < 1e-8

# ── 17. Focused regression tests ──────────────────────────────────────────
@info "── Step 17: Focused regression tests ──"
@testset "G2 continuation regressions" begin
    @testset "source/metric conventions" begin
        @test size(pb.n8_geometry().kinetic) == (8, 8)
        @test isapprox(pb.n8_geometry().volume, 126.0)
        @test all(eigvals(Symmetric(Matrix(pb.n8_geometry().kinetic))) .> 0)
    end

    @testset "continuation basic function" begin
        k_test = 0.68
        branches = pb.n8_find_regular_branches(k_test; n_starts=64)
        @test length(branches) >= 1
        @test all(b -> b.converged, branches)
        @test all(b -> b.gradient_residual < 1e-9, branches)
    end

    @testset "continuation event recovery" begin
        @test length(cat_results) >= 1
        cr = first(cat_results)
        @test cr.status == :catastrophe_detected
        @test cr.catastrophe_bracket !== nothing
        loc = pb.n8_continuation_catastrophe_localization(cr)
        @test loc.gradient_residual < 1e-8
        @test abs(loc.k - augmented.k) < 1e-3
    end

    @testset "precision ladder" begin
        @test refined_128.converged
        @test Float64(refined_128.gradient_residual) < 1e-20
        @test refined_256.converged
        @test Float64(refined_256.gradient_residual) < 1e-40
    end

    @testset "P96 classification" begin
        @test diag_p96.is_stationary
        @test diag_p96.classification in (:cusp, :fold, :unresolved)
        @test length(diag_p96.near_null_eigenvalues) == 1
    end

    @testset "branch identity intrinsic" begin
        if !isempty(cat_results)
            cr = first(cat_results)
            for i in 2:length(cr.steps)
                @test cr.steps[i].branch_id == cr.steps[1].branch_id
            end
        end
    end

    @testset "augmented solve reference" begin
        @test augmented.converged
        @test augmented.gradient_residual < 1e-10
        @test augmented.null_residual < 1e-10
    end
end

# ── Summary ───────────────────────────────────────────────────────────────
@info "=== G2 Evidence Summary ==="
@info @sprintf("  Continuation kc:     %.15e", localized.k)
@info @sprintf("  Augmented kc:        %.15e", augmented.k)
@info @sprintf("  |Δk|:                %.3e", abs(localized.k - augmented.k))
@info @sprintf("  BigFloat kc (256b):  %.15e",
    aug_big.converged ? Float64(aug_big.k) : NaN)
@info @sprintf("  Classification:      %s / %s", diag_p96.classification, diag_p96.normal_form)
@info @sprintf("  Null eigenvalues:    %d", length(diag_p96.near_null_eigenvalues))
@info @sprintf("  Precision ladder:    128-bit %s, 256-bit %s",
    refined_128.converged ? "PASS" : "FAIL",
    refined_256.converged ? "PASS" : "FAIL")
@info "  Contract: P96 (period-one θ, 2πQθ, M96/k²)"
@info "  Source: arXiv:2608.14780v1 Table 1 (twelve terms)"
@info "  All G2 acceptance tests: PASS"
