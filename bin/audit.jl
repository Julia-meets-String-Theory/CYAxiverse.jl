#!/usr/bin/env julia

using Pkg
using SHA

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const PROJECT_FILE = joinpath(PROJECT_ROOT, "Project.toml")
const MANIFEST_FILE = joinpath(PROJECT_ROOT, "Manifest.toml")
const AUDIT_CACHE_VERSION = 1
const AUDIT_DEPENDENCIES = (("Aqua", "0.8"), ("JET", "0.12"))

const AUDIT_CACHE_ROOT = let
    configured_root = get(ENV, "CYAXIVERSE_AUDIT_CACHE", "")
    isempty(configured_root) ?
        joinpath(first(DEPOT_PATH), "environments", "cyaxiverse-audit") :
        normpath(abspath(configured_root))
end

function file_digest(path)
    isfile(path) || return "missing"
    bytes2hex(SHA.sha256(read(path)))
end

function audit_dependency_spec()
    join(("$name=$version" for (name, version) in AUDIT_DEPENDENCIES), ",")
end

function cache_key_material()
    join([
        "cache-version=$(AUDIT_CACHE_VERSION)",
        "julia-version=$(VERSION)",
        "platform=$(Sys.MACHINE)",
        "project-root=$(PROJECT_ROOT)",
        "project-sha256=$(file_digest(PROJECT_FILE))",
        "manifest-sha256=$(file_digest(MANIFEST_FILE))",
        "audit-dependencies=$(audit_dependency_spec())",
    ], '\n')
end

const AUDIT_CACHE_KEY = bytes2hex(SHA.sha256(cache_key_material()))
const AUDIT_ENV = joinpath(
    AUDIT_CACHE_ROOT,
    "v$(AUDIT_CACHE_VERSION)-$(AUDIT_CACHE_KEY)",
)
const AUDIT_CACHE_MARKER_NAME = ".cyaxiverse-audit-cache"

function cache_marker(env)
    join([
        cache_key_material(),
        "environment-project-sha256=$(file_digest(joinpath(env, "Project.toml")))",
        "environment-manifest-sha256=$(file_digest(joinpath(env, "Manifest.toml")))",
    ], '\n') * '\n'
end

function cache_is_valid(env)
    isdir(env) || return false
    marker = joinpath(env, AUDIT_CACHE_MARKER_NAME)
    isfile(marker) || return false
    isfile(joinpath(env, "Project.toml")) || return false
    isfile(joinpath(env, "Manifest.toml")) || return false
    read(marker, String) == cache_marker(env)
end

function audit_dependency_specs()
    [Pkg.PackageSpec(name=name, version=version) for (name, version) in AUDIT_DEPENDENCIES]
end

function provision_audit_environment(env)
    Pkg.activate(env)
    Pkg.develop(Pkg.PackageSpec(path=PROJECT_ROOT))
    Pkg.add(audit_dependency_specs())
    Pkg.instantiate()
end

function ensure_audit_environment()
    if cache_is_valid(AUDIT_ENV)
        println("Using cached audit environment: $(AUDIT_ENV)")
        Pkg.activate(AUDIT_ENV)
        return
    end

    mkpath(AUDIT_CACHE_ROOT)
    staging = mktempdir(AUDIT_CACHE_ROOT; prefix=".cyaxiverse-audit-")
    try
        println("Creating cached audit environment: $(AUDIT_ENV)")
        provision_audit_environment(staging)
        open(joinpath(staging, AUDIT_CACHE_MARKER_NAME), "w") do io
            write(io, cache_marker(staging))
        end

        if ispath(AUDIT_ENV)
            cache_is_valid(AUDIT_ENV) || rm(AUDIT_ENV; recursive=true, force=true)
        end
        mv(staging, AUDIT_ENV)
        staging = nothing
    finally
        staging === nothing || rm(staging; recursive=true, force=true)
    end

    cache_is_valid(AUDIT_ENV) || error("Failed to create a valid audit environment cache.")
    Pkg.activate(AUDIT_ENV)
end

ensure_audit_environment()

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
