#!/usr/bin/env julia

using Pkg

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const AUDIT_ENV = mktempdir(prefix="cyaxiverse-audit-")

Pkg.activate(AUDIT_ENV)
Pkg.develop(Pkg.PackageSpec(path=PROJECT_ROOT))
Pkg.add([
    Pkg.PackageSpec(name="Aqua", version="0.8"),
    Pkg.PackageSpec(name="JET", version="0.12"),
])
Pkg.instantiate()

using Aqua
using JET
using LinearAlgebra
using CYAxiverse

function audit_type_stability()
    println("=== 1. TYPE STABILITY AUDIT (JET.jl) ===")

    report = JET.report_package(CYAxiverse; target_modules=(CYAxiverse,))
    show(stdout, MIME"text/plain"(), report)
    println()

    reports = JET.get_reports(report)
    isempty(reports) || error("JET found $(length(reports)) report(s) in CYAxiverse.")
end

function audit_package_health()
    println("\n=== 2. PACKAGE HEALTH (Aqua.jl) ===")
    Aqua.test_all(CYAxiverse; ambiguities=false)
end

function check_potential_derivatives(x, L, Q)
    scales = L[:, 1] .* 10.0 .^ L[:, 2]
    phases = vec(transpose(Q) * x)

    potential = CYAxiverse.generate.V(x, L, Q)
    expected_potential = sum(scales .* (1.0 .- cos.(phases)))
    isapprox(potential, expected_potential; rtol=1e-12, atol=1e-14) ||
        error("Potential value is inconsistent with its definition.")

    gradient = vec(collect(CYAxiverse.generate.jacobian(x, L, Q)))
    expected_gradient = Q * (scales .* sin.(phases))
    isapprox(gradient, expected_gradient; rtol=1e-12, atol=1e-14) ||
        error("Potential gradient is inconsistent with its definition.")

    hessian = Matrix(CYAxiverse.generate.hessian(x, L, Q))
    expected_hessian = Q * Diagonal(scales .* cos.(phases)) * transpose(Q)
    isapprox(hessian, expected_hessian; rtol=1e-12, atol=1e-14) ||
        error("Potential Hessian is inconsistent with its definition.")
    isapprox(hessian, transpose(hessian); rtol=0.0, atol=1e-14) ||
        error("Potential Hessian is not symmetric.")
end

function audit_physics()
    println("\n=== 3. PHYSICAL POTENTIAL SANITY CHECKS ===")

    # Exercise both the single-field and multi-field branches.
    check_potential_derivatives(
        [0.17],
        Float64[1.0 0.0; 1.0 -0.5; 1.0 -1.0],
        reshape(Int[1, 2, 3], 1, :),
    )
    check_potential_derivatives(
        [0.37, -0.21],
        Float64[1.0 0.0; 1.0 -0.5; 1.0 -1.0],
        Int[1 0 1; 0 1 1],
    )

    L = Float64[1.0 0.0; 1.0 -0.5; 1.0 -1.0]
    Q = Int[1 0 1; 0 1 1]
    hessian = Matrix(CYAxiverse.generate.hessian([0.37, -0.21], L, Q))
    all(isfinite, hessian) || error("Potential Hessian contains non-finite values.")
    all(>(0.0), eigvals(Symmetric(hessian))) ||
        error("Reference potential Hessian is not positive definite.")

    println("Potential, derivative, and positive-definiteness checks passed.")
end

audit_type_stability()
audit_package_health()
audit_physics()

println("\nALL AUDITS PASSED CLEANLY!")
