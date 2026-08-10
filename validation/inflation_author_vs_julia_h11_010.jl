#!/usr/bin/env julia

"""Compare the author coefficient map with the generic Julia homotopy.

This is a bounded diagnostic, not a physical inflation scan. It selects one
deterministic pseudo-random h11=10 geometry, reconstructs the author's
coefficient formula from the stored `(tau, Kinv, CY_volume, Q)` data, and
compares it with the persisted reference potential and with the current
generic pilot operation `L[2, :] <- k * L[2, :]`.
"""

using HDF5
using LinearAlgebra
using Printf
using Random

const DATA_ROOT = normpath(joinpath(@__DIR__, "..", "..", "data"))
const H11 = 10
const SEED = 20260809
const SCALE_GRID = (0.9, 1.1)

function geometry_files(data_root::AbstractString)
    root = joinpath(data_root, "h11_010")
    files = String[]
    for polytope in sort(readdir(root))
        path = joinpath(root, polytope, "cy_0000001", "cyax.h5")
        isfile(path) && push!(files, path)
    end
    isempty(files) && error("no h11=10 geometry files under $root")
    files
end

function read_geometry(path::AbstractString)
    h5open(path, "r") do file
        tau = Float64.(read(file["cytools/geometric/divisor_volumes"]))
        kinv = Float64.(read(file["cytools/geometric/Kinv"]))
        volume = Float64(read(file["cytools/geometric/CY_volume"]))
        q_raw = Int.(read(file["cytools/potential/Q"]))
        l_raw = Float64.(read(file["cytools/potential/L"]))
        h11 = length(tau)
        q_rows = size(q_raw, 2) == h11 ? q_raw : Matrix(q_raw')
        l_rows = size(l_raw, 2) == 2 ? l_raw : Matrix(l_raw')
        size(q_rows, 2) == h11 || error("Q has incompatible shape $(size(q_raw))")
        size(l_rows, 2) == 2 || error("L has incompatible shape $(size(l_raw))")
        size(q_rows, 1) == size(l_rows, 1) || error("Q/L term counts disagree")
        (; tau, kinv, volume, Q=q_rows, L=l_rows)
    end
end

function leading_count(term_count::Int)
    discriminant = 8 * term_count + 1
    nq = round(Int, (sqrt(discriminant) - 1) / 2)
    nq + nq * (nq - 1) ÷ 2 == term_count ||
        error("cannot infer leading effective-cone count from $term_count terms")
    nq
end

"""Author's stored-coefficient formula, with L rows `(sign, log10 amplitude)`."""
function author_coefficients(qprime::AbstractMatrix{<:Real}, tau, kinv, volume)
    nq, _ = size(qprime)
    terms = Vector{Vector{Int}}()
    prefactors = Float64[]
    exponents = Float64[]
    for i in 1:nq
        qi = Float64.(qprime[i, :])
        qtau = dot(qi, tau)
        push!(terms, Int.(qprime[i, :]))
        push!(prefactors, (8π / volume^2) * qtau)
        push!(exponents, -2π * log10(exp(1.0)) * qtau)
    end
    for i in 1:(nq - 1), j in (i + 1):nq
        qi = Float64.(qprime[i, :])
        qj = Float64.(qprime[j, :])
        push!(terms, Int.(qj - qi))
        prefactor = (π * dot(qi, kinv * qj) + dot(qi + qj, tau)) * 8π / volume^2
        push!(prefactors, prefactor)
        push!(exponents, -2π * log10(exp(1.0)) * dot(qi + qj, tau))
    end
    L = Matrix{Float64}(undef, length(prefactors), 2)
    for i in eachindex(prefactors)
        L[i, 1] = sign(prefactors[i])
        L[i, 2] = prefactors[i] == 0 ? -Inf : log10(abs(prefactors[i])) + exponents[i]
    end
    (; Q=vcat((permutedims(term) for term in terms)...), L)
end

function compare_coefficients(label, expected, observed)
    expected.Q == observed.Q || error("$label: charge rows differ")
    finite = isfinite.(expected.L[:, 2]) .& isfinite.(observed.L[:, 2])
    log_error = isempty(findall(finite)) ? NaN : maximum(abs.(expected.L[finite, 2] .- observed.L[finite, 2]))
    sign_mismatches = count(expected.L[:, 1] .!= observed.L[:, 1])
    @printf("%-34s finite-log-max-error=% .6e sign-mismatches=%d\n",
        label, log_error, sign_mismatches)
    (; log_error, sign_mismatches)
end

files = geometry_files(DATA_ROOT)
rng = MersenneTwister(SEED)
path = files[rand(rng, 1:length(files))]
data = read_geometry(path)
nq = leading_count(size(data.Q, 1))
qprime = data.Q[1:nq, :]

@printf("geometry=%s h11=%d term_count=%d leading_count=%d seed=%d\n",
    relpath(path, DATA_ROOT), length(data.tau), size(data.Q, 1), nq, SEED)

reference = author_coefficients(qprime, data.tau, data.kinv, data.volume)
base_check = compare_coefficients("author formula vs stored L", reference, data)
base_check.log_error < 1e-10 || error("base author/storage coefficient check failed")
base_check.sign_mismatches == 0 || error("base author/storage sign check failed")

for k in SCALE_GRID
    tau = k .* data.tau
    kinv = k^2 .* data.kinv
    fixed = author_coefficients(qprime, tau, kinv, data.volume)
    full = author_coefficients(qprime, tau, kinv, data.volume * k^(3 / 2))
    homotopy = copy(data.L)
    homotopy[:, 2] .*= k
    compare_coefficients("k=$(k) author fixed vs homotopy", fixed, (; Q=data.Q, L=homotopy))
    compare_coefficients("k=$(k) author full vs homotopy", full, (; Q=data.Q, L=homotopy))
end
