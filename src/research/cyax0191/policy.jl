"""Numerical choices frozen before any named Gate C benchmark execution."""
struct FrozenNumericalPolicy{T<:AbstractFloat}
    benchmark_id::Symbol
    precision_bits::Int
    coordinate_order::Tuple{Vararg{String}}
    field_scales::Tuple{Vararg{T}}
    potential_scale::T
    potential_scale_rule::String
    stationarity_tolerance::T
    root_tolerance::T
    optimizer_gradient_tolerance::T
    solver_method::String
    maximum_iterations::Int
    backtracking_factor::T
    minimum_step::T
    search_seed::Int
    search_budget::Int
    duplicate_rule::String
    absolute_spectral_threshold::T
    relative_spectral_threshold::T
    spectral_normalization::String
    frozen_before_gate_c::Bool
end

function _frozen_policy(benchmark_id::Symbol, chi::Int, gs_text::String,
        kcs_text::String, w0_text::String, bbhl_enabled::Bool,
        coordinate_order::Tuple)
    setprecision(BigFloat, 256) do
        T = BigFloat
        gs, kcs, w0 = parse(T, gs_text), parse(T, kcs_text), parse(T, w0_text)
        s = inv(gs)
        xi = -_zeta3(T) * T(chi) / (T(2) * (T(2) * T(π))^3)
        xihat = xi * s * sqrt(s)
        yref = one(T) + (bbhl_enabled ? xihat / T(2) : zero(T))
        yref > zero(T) || throw(DomainError(yref, "canonical reference point must have positive Y"))
        vscale = exp(kcs) * gs * w0^2 / (T(2) * yref^2)
        names = Tuple(String.(coordinate_order))
        policy = FrozenNumericalPolicy{T}(benchmark_id, 256, names,
            Tuple(fill(one(T), length(names))), vscale,
            "e^(K_cs)*g_s*|W0|^2/(2*Y_ref^2), evaluated at V_ref=1; independent of any stationary point",
            parse(T, "1e-12"), parse(T, "1e-12"), parse(T, "1e-12"),
            "damped Newton on the maximum componentwise scaled stationarity residual",
            10_000, parse(T, "0.5"), T(2)^(-20), 191019, 10_000,
            "canonical axion representatives; duplicates require scaled coordinate distance <= 1e-10",
            parse(T, "1e-12"), parse(T, "1e-10"),
            "generalized eigenvalues of S*H*S/V_scale and S*G*S, with frozen field scales S; equivalently m^2/V_scale",
            true)
        all(>(zero(T)), policy.field_scales) || error("frozen field scales must be positive")
        policy.potential_scale > zero(T) || error("frozen potential scale must be positive")
        policy
    end
end

const FROZEN_GATE_C_POLICIES = (
    _frozen_policy(:P0_B1, -40, "0.1", "0.1", "1e-4", false,
        ("t_1", "rho_1")),
    _frozen_policy(:P0_B2, -200, "0.2", "1.0", "0.68", true,
        ("t_1", "rho_1")),
    _frozen_policy(:P0_B3, -126, "0.10", "1.0", "1.0", true,
        ("t_1", "t_2", "t_3", "rho_1", "rho_2", "rho_3")),
)

function frozen_policy(benchmark_id::Symbol)
    index = findfirst(policy -> policy.benchmark_id === benchmark_id,
        FROZEN_GATE_C_POLICIES)
    index === nothing && throw(KeyError(benchmark_id))
    FROZEN_GATE_C_POLICIES[index]
end

function policy_manifest(policy::FrozenNumericalPolicy)
    (; policy_version="cyax0191-gatec-numerics-v1",
       benchmark_id=String(policy.benchmark_id),
       precision_bits=policy.precision_bits,
       numeric_type="BigFloat",
       coordinate_order=policy.coordinate_order,
       field_scales=Tuple(string.(policy.field_scales)),
       potential_scale=string(policy.potential_scale),
       potential_scale_rule=policy.potential_scale_rule,
       stationarity_tolerance=string(policy.stationarity_tolerance),
       root_tolerance=string(policy.root_tolerance),
       optimizer_gradient_tolerance=string(policy.optimizer_gradient_tolerance),
       solver_method=policy.solver_method,
       maximum_iterations=policy.maximum_iterations,
       backtracking_factor=string(policy.backtracking_factor),
       minimum_step=string(policy.minimum_step),
       search_seed=policy.search_seed,
       search_budget=policy.search_budget,
       duplicate_rule=policy.duplicate_rule,
       absolute_spectral_threshold=string(policy.absolute_spectral_threshold),
       relative_spectral_threshold=string(policy.relative_spectral_threshold),
       spectral_normalization=policy.spectral_normalization,
       frozen_before_gate_c=policy.frozen_before_gate_c)
end
