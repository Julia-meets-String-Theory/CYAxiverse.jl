"""Formula-level reconstruction of the scalar analytic Regge curves in Fig. 3.

All numerical inputs are evaluated with BigFloat. Masses are in solar masses,
scalar masses in eV, and returned rates in s^-1. This implements the source's
analytic approximation; it does not claim a digitized Fig. 3 comparison.
"""

using SHA

const BHSR_VALIDATION_DIR = normpath(joinpath(@__DIR__, "..", "validation",
                                               "cyax_0121_bhsr_reconstruction"))
const BHSR_SOURCE_MODE_MANIFEST = joinpath(BHSR_VALIDATION_DIR, "source_mode_manifest.json")
const BHSR_NUMERICAL_METHOD_MANIFEST = joinpath(BHSR_VALIDATION_DIR, "numerical_method_manifest.json")
const BHSR_SOURCE_MODE_MANIFEST_SHA256 =
    "a891735d45dd80e57689917416577203e43a68ec0c7e96b23897cb7c306725ed"
const BHSR_NUMERICAL_METHOD_MANIFEST_SHA256 =
    "7b9c1cf9e4787372801125d7c195288d36f3cb9e094b0a2ec179493c8e8392d5"
const BHSR_TOPOLOGY_METHOD_ADDENDUM = joinpath(BHSR_VALIDATION_DIR,
    "topology_method_addendum.md")
const BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256 =
    "909d283bd19af956ee2881537ee90dce07c0e1c7976717ac27cc7ca6402f7d39"

function _bhsr_verified_manifest(path::AbstractString, expected_sha256::AbstractString)
    isfile(path) || error("required frozen BHSR manifest is absent: $path")
    bytes = read(path)
    actual = bytes2hex(sha256(bytes))
    actual == expected_sha256 || error("frozen BHSR manifest hash mismatch for $(basename(path))")
    return String(bytes)
end

const _BHSR_SOURCE_MODE_MANIFEST_TEXT =
    _bhsr_verified_manifest(BHSR_SOURCE_MODE_MANIFEST, BHSR_SOURCE_MODE_MANIFEST_SHA256)
const _BHSR_METHOD_MANIFEST_TEXT =
    _bhsr_verified_manifest(BHSR_NUMERICAL_METHOD_MANIFEST,
                            BHSR_NUMERICAL_METHOD_MANIFEST_SHA256)
const _BHSR_TOPOLOGY_METHOD_ADDENDUM_TEXT =
    _bhsr_verified_manifest(BHSR_TOPOLOGY_METHOD_ADDENDUM,
                            BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256)

function _bhsr_json_string_field(object_text::AbstractString, key::AbstractString)
    pattern = Regex("\"" * key * "\"\\s*:\\s*\"([^\"]*)\"")
    found = match(pattern, object_text)
    found === nothing && error("frozen BHSR manifest is missing string field '$key'")
    return String(found.captures[1])
end

function _bhsr_json_int_field(object_text::AbstractString, key::AbstractString)
    pattern = Regex("\"" * key * "\"\\s*:\\s*(-?[0-9]+)")
    found = match(pattern, object_text)
    found === nothing && error("frozen BHSR manifest is missing integer field '$key'")
    return parse(Int, found.captures[1])
end

function _bhsr_json_object_blocks(array_text::AbstractString)
    return [String(found.captures[1]) for found in eachmatch(r"\{([^{}]*)\}"s, array_text)]
end

function _bhsr_json_array_text(manifest_text::AbstractString, key::AbstractString;
                               following_key::Union{Nothing,String} = nothing)
    ending = following_key === nothing ? "\\]" :
        "\\]\\s*,\\s*\"" * following_key * "\""
    pattern = Regex("\"" * key * "\"\\s*:\\s*\\[(.*?)" * ending, "s")
    found = match(pattern, manifest_text)
    found === nothing && error("frozen BHSR manifest is missing array '$key'")
    return String(found.captures[1])
end

const BHSR_SI = (
    G = "6.67430e-11",
    c = "299792458",
    hbar = "1.054571817e-34",
    eV_J = "1.602176634e-19",
    M_sun = "1.98847e30",
    year_s = "31557600",
)

_bhsr_big(x::BigFloat) = x
_bhsr_big(x::AbstractFloat) = parse(BigFloat, string(x))
_bhsr_big(x::Real) = BigFloat(x)

function _bhsr_validate_reference_topology_contract(precision_bits::Integer,
        max_iterations::Integer, absolute_tolerance::Real)
    precision_bits == 256 || throw(ArgumentError(
        "reference topology requires canonical precision_bits=256"))
    max_iterations == 140 || throw(ArgumentError(
        "reference topology requires canonical max_iterations=140"))
    _bhsr_big(absolute_tolerance) == big"1e-40" || throw(ArgumentError(
        "reference topology requires canonical absolute_tolerance=1e-40"))
    return nothing
end

struct BHSRMode
    n_r::Int
    l::Int
    m::Int
    label::String
    function BHSRMode(n_r::Integer, l::Integer, m::Integer, label::AbstractString)
        n_r >= 0 || throw(ArgumentError("radial overtone n_r must be nonnegative"))
        l >= 1 || throw(ArgumentError("scalar angular number l must be positive"))
        1 <= m <= l || throw(ArgumentError("this reconstruction requires 1 <= m <= l"))
        new(Int(n_r), Int(l), Int(m), String(label))
    end
end

function _bhsr_modes_from_manifest()
    array_text = _bhsr_json_array_text(_BHSR_SOURCE_MODE_MANIFEST_TEXT, "modes")
    modes = BHSRMode[]
    for block in _bhsr_json_object_blocks(array_text)
        label = _bhsr_json_string_field(block, "label")
        N = _bhsr_json_int_field(block, "N")
        l = _bhsr_json_int_field(block, "l")
        m = _bhsr_json_int_field(block, "m")
        n_r = _bhsr_json_int_field(block, "n_r")
        mode = BHSRMode(n_r, l, m, label)
        n_r + l + 1 == N ||
            error("frozen source mode has inconsistent principal number: $label")
        push!(modes, mode)
    end
    isempty(modes) && error("frozen source mode manifest contains no modes")
    length(unique(mode.label for mode in modes)) == length(modes) ||
        error("frozen source mode manifest contains duplicate labels")
    return modes
end

const _BHSR_SOURCE_MODES = _bhsr_modes_from_manifest()

"The nodeless family loaded from the hash-verified frozen source-mode manifest."
bhsr_nodeless_modes() = copy(_BHSR_SOURCE_MODES)

bhsr_principal_number(mode::BHSRMode) = mode.n_r + mode.l + 1

function _bhsr_validate_modes(modes::AbstractVector{BHSRMode})
    isempty(modes) && throw(ArgumentError("at least one source-manifest mode is required"))
    labels = getfield.(modes, :label)
    length(unique(labels)) == length(labels) ||
        throw(ArgumentError("mode selection contains duplicate source-mode identities"))
    allowed = Dict(mode.label => mode for mode in _BHSR_SOURCE_MODES)
    all(label -> haskey(allowed, label), labels) ||
        throw(ArgumentError("mode selection contains a mode absent from the frozen source manifest"))
    all(i -> modes[i].n_r == allowed[labels[i]].n_r &&
             modes[i].l == allowed[labels[i]].l &&
             modes[i].m == allowed[labels[i]].m,
        eachindex(modes)) ||
        throw(ArgumentError("mode data do not match the frozen source-mode identity"))
    source_order = [mode.label for mode in _BHSR_SOURCE_MODES if mode.label in labels]
    labels == source_order ||
        throw(ArgumentError("mode selection must retain frozen source-manifest order"))
    return modes
end

function _bhsr_model_id(model_id::AbstractString)
    id = String(strip(String(model_id)))
    isempty(id) && throw(ArgumentError("model_id must be explicit and nonempty"))
    return id
end

struct BHSRContourGrid
    model_id::String
    route_identity::String
    method_manifest_sha256::String
    topology_method_addendum_sha256::String
    source_mode_manifest_sha256::String
    mass_solar::Vector{BigFloat}
    union_spin::Vector{Union{Missing,BigFloat}}
    rows::Vector{Any}
    modes::Vector{BHSRMode}
    mu_eV::BigFloat
    tau_years::BigFloat
    delta_a::BigFloat
    precision_bits::Int
    root_max_iterations::Int
    root_absolute_tolerance::BigFloat
    mass_log10_min::BigFloat
    mass_log10_max::BigFloat
    source_grid::Bool
    contour_evaluator::Any
end

Base.length(grid::BHSRContourGrid) = length(grid.rows)
Base.getindex(grid::BHSRContourGrid, i::Int) = grid.rows[i]
Base.iterate(grid::BHSRContourGrid, state...) = iterate(grid.rows, state...)

function _bhsr_constants()
    (; (k => parse(BigFloat, v) for (k, v) in pairs(BHSR_SI))...)
end

"Dimensionless gravitational coupling alpha = G M mu/(hbar c^3), with mu in eV."
function bhsr_alpha(mass_solar::Real, mu_eV::Real; precision_bits::Integer = 256)
    mass_solar > 0 || throw(ArgumentError("black-hole mass must be positive"))
    mu_eV > 0 || throw(ArgumentError("scalar mass must be positive"))
    return setprecision(BigFloat, precision_bits) do
        k = _bhsr_constants()
        mass_kg = _bhsr_big(mass_solar) * k.M_sun
        mu_joule = _bhsr_big(mu_eV) * k.eV_J
        k.G * mass_kg * mu_joule / (k.hbar * k.c^3)
    end
end

"Occupation needed to extract the specified spin change, using source Eq. (9)."
function bhsr_nmax(mass_solar::Real, mode::BHSRMode; delta_a::Real = big"0.1",
                   precision_bits::Integer = 256)
    mass_solar > 0 || throw(ArgumentError("black-hole mass must be positive"))
    delta_a > 0 || throw(ArgumentError("delta_a must be positive"))
    return setprecision(BigFloat, precision_bits) do
        k = _bhsr_constants()
        mass_kg = _bhsr_big(mass_solar) * k.M_sun
        k.G * mass_kg^2 * _bhsr_big(delta_a) /
            (BigFloat(mode.m) * k.hbar * k.c)
    end
end

function _bhsr_A(mode::BHSRMode)
    n = mode.n_r
    l = mode.l
    top = BigFloat(2)^(4l + 2) * BigFloat(factorial(big(2l + n + 1)))
    top /= BigFloat(l + n + 1)^(2l + 4) * BigFloat(factorial(big(n)))
    ratio = BigFloat(factorial(big(l))) /
        (BigFloat(factorial(big(2l))) * BigFloat(factorial(big(2l + 1))))
    return top * ratio^2
end

"Signed analytic scalar rate in s^-1 from source Eqs. (12)-(14)."
function bhsr_rate_s(mass_solar::Real, mu_eV::Real, spin::Real, mode::BHSRMode;
                     precision_bits::Integer = 256)
    mass_solar > 0 || throw(ArgumentError("black-hole mass must be positive"))
    mu_eV > 0 || throw(ArgumentError("scalar mass must be positive"))
    0 <= spin <= 1 || throw(ArgumentError("dimensionless spin must lie in [0,1]"))
    return setprecision(BigFloat, precision_bits) do
        k = _bhsr_constants()
        a = _bhsr_big(spin)
        mass_kg = _bhsr_big(mass_solar) * k.M_sun
        alpha = k.G * mass_kg * _bhsr_big(mu_eV) * k.eV_J /
            (k.hbar * k.c^3)
        N = bhsr_principal_number(mode)
        rplus = 1 + sqrt(1 - a^2)
        omega_bar = alpha * (1 - alpha^2 / (2 * BigFloat(N)^2))
        omega_h_bar = a / (2rplus)
        detuning_bar = BigFloat(mode.m) * omega_h_bar - omega_bar
        xlm = prod(
            BigFloat(j^2) * (1 - a^2) +
                4rplus^2 * (BigFloat(mode.m) * omega_bar - alpha)^2
            for j in 1:mode.l
        )
        gamma_m = 2alpha * rplus * detuning_bar * alpha^(4mode.l + 4) *
            _bhsr_A(mode) * xlm
        t_m_seconds = k.G * mass_kg / k.c^3
        gamma_m / t_m_seconds
    end
end

"Residual of Gamma_SR*tau >= log(Nmax); nonnegative means the mode grows enough."
function bhsr_free_field_residual(mass_solar::Real, mu_eV::Real, spin::Real,
                                  mode::BHSRMode, tau_years::Real;
                                  delta_a::Real = 0.1,
                                  precision_bits::Integer = 256)
    tau_years > 0 || throw(ArgumentError("black-hole age must be positive"))
    return setprecision(BigFloat, precision_bits) do
        k = _bhsr_constants()
        nmax = bhsr_nmax(mass_solar, mode; delta_a, precision_bits)
        nmax > 1 || throw(DomainError(nmax, "source logarithmic threshold requires Nmax > 1"))
        gamma = bhsr_rate_s(mass_solar, mu_eV, spin, mode; precision_bits)
        gamma * _bhsr_big(tau_years) * k.year_s - log(nmax)
    end
end

const BHSR_SPIN_TOPOLOGY_SCAN_POINTS = 257
const BHSR_SPIN_TOPOLOGY_EXTREMUM_MAX_ITERATIONS = 240
const BHSR_SPIN_TOPOLOGY_NEAR_TANGENT_TOLERANCE = big"1e-40"

function _bhsr_refine_spin_extremum(f, lower::BigFloat, upper::BigFloat,
                                    maximize::Bool, x_tolerance::BigFloat,
                                    max_iterations::Integer)
    golden = (sqrt(BigFloat(5)) - 1) / 2
    left, right = lower, upper
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1, f2 = f(x1), f(x2)
    for _ in 1:max_iterations
        right - left <= x_tolerance && break
        better = maximize ? f1 > f2 : f1 < f2
        if better
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = f(x1)
        else
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = f(x2)
        end
    end
    right - left <= x_tolerance ||
        error("spin-topology extremum refinement did not meet tolerance")
    point = (left + right) / 2
    return point, f(point)
end

function _bhsr_refine_spin_root(f, lower::BigFloat, upper::BigFloat,
                                f_lower::BigFloat, f_upper::BigFloat,
                                x_tolerance::BigFloat,
                                max_iterations::Integer)
    iszero(f_lower) && return lower
    iszero(f_upper) && return upper
    sign(f_lower) != sign(f_upper) || return nothing
    left, right = lower, upper
    fleft = f_lower
    for _ in 1:max_iterations
        right - left <= x_tolerance && break
        midpoint = (left + right) / 2
        fmid = f(midpoint)
        iszero(fmid) && return midpoint
        if sign(fmid) == sign(fleft)
            left, fleft = midpoint, fmid
        else
            right = midpoint
        end
    end
    right - left <= x_tolerance ||
        error("spin-topology root did not meet absolute tolerance")
    return (left + right) / 2
end

function _bhsr_spin_topology_status(intervals, roots)
    isempty(intervals) && return isempty(roots) ? :no_positive_interval_resolved : :isolated_threshold_points
    length(intervals) > 1 && return :multiple_efficiency_intervals
    lower, upper = only(intervals)
    lower == 0 && upper == 1 && return :all_spins_efficient
    lower == 0 && return :efficiency_from_zero_bounded_above
    upper == 1 && return :single_onset_to_extremal_spin
    return :bounded_efficiency_interval
end

function _bhsr_isolate_positive_spin_intervals_at_resolution(f, scan_points::Integer;
        max_iterations::Integer = 140,
        extremum_max_iterations::Integer = BHSR_SPIN_TOPOLOGY_EXTREMUM_MAX_ITERATIONS,
        absolute_tolerance::Real = big"1e-40")
    scan_points >= 3 || throw(ArgumentError("spin-topology scan needs at least three points"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    x_tolerance = _bhsr_big(absolute_tolerance)
    x_tolerance > 0 || throw(ArgumentError("absolute_tolerance must be positive"))
    x_tolerance < 1 || throw(ArgumentError("absolute_tolerance must be less than the spin range"))
    extremum_max_iterations > 0 || throw(ArgumentError("extremum iteration limit must be positive"))
    points = [BigFloat(index - 1) / BigFloat(scan_points - 1) for index in 1:scan_points]
    values = BigFloat[f(point) for point in points]
    all(isfinite, values) || throw(DomainError(values, "spin residual must be finite on [0,1]"))
    nodes = Tuple{BigFloat,BigFloat}[(points[index], values[index])
                                    for index in eachindex(points)]
    extrema = Tuple{BigFloat,BigFloat}[]
    extremum_tolerance = x_tolerance
    for index in 2:(scan_points - 1)
        previous, current, following = values[index - 1], values[index], values[index + 1]
        is_maximum = current >= previous && current >= following &&
                     (current > previous || current > following)
        is_minimum = current <= previous && current <= following &&
                     (current < previous || current < following)
        if is_maximum || is_minimum
            point, value = _bhsr_refine_spin_extremum(f, points[index - 1],
                points[index + 1], is_maximum, extremum_tolerance,
                extremum_max_iterations)
            push!(nodes, (point, value))
            push!(extrema, (point, value))
        end
    end
    sort!(nodes; by = first)

    roots = BigFloat[]
    for (point, value) in nodes
        iszero(value) && push!(roots, point)
    end
    for index in 1:(length(nodes) - 1)
        lower, f_lower = nodes[index]
        upper, f_upper = nodes[index + 1]
        if !iszero(f_lower) && !iszero(f_upper) && sign(f_lower) != sign(f_upper)
            root = _bhsr_refine_spin_root(f, lower, upper, f_lower, f_upper,
                                           x_tolerance, max_iterations)
            !isnothing(root) && push!(roots, root)
        end
    end
    sort!(roots)
    unique_roots = BigFloat[]
    for root in roots
        if isempty(unique_roots) || root - last(unique_roots) > 2x_tolerance
            push!(unique_roots, root)
        end
    end

    boundaries = sort!(unique!(vcat(BigFloat[0], unique_roots, BigFloat[1])))
    intervals = Tuple{BigFloat,BigFloat}[]
    for index in 1:(length(boundaries) - 1)
        lower, upper = boundaries[index], boundaries[index + 1]
        upper > lower || continue
        f((lower + upper) / 2) > 0 && push!(intervals, (lower, upper))
    end
    near_tangent_extrema = [item for item in extrema
        if abs(item[2]) <= BHSR_SPIN_TOPOLOGY_NEAR_TANGENT_TOLERANCE]
    status = isempty(near_tangent_extrema) ?
        _bhsr_spin_topology_status(intervals, unique_roots) : :unavailable_near_tangent
    return (; roots = unique_roots,
            intervals,
            status,
            near_tangent_extrema,
            refined_extrema = extrema,
            scan_points = Int(scan_points),
            isolation_method = "uniform_scan_with_local_extremum_refinement")
end

function _bhsr_topology_endpoints_stable(previous, current, tolerance::BigFloat)
    previous.status == current.status || return false
    length(previous.roots) == length(current.roots) || return false
    length(previous.intervals) == length(current.intervals) || return false
    all(abs(previous.roots[index] - current.roots[index]) <= tolerance
        for index in eachindex(previous.roots)) || return false
    all(abs(previous.intervals[index][endpoint] - current.intervals[index][endpoint]) <= tolerance
        for index in eachindex(previous.intervals) for endpoint in 1:2) || return false
    return true
end

"Refine spin-topology isolation until crossings and intervals stabilize."
function _bhsr_isolate_positive_spin_intervals(f;
        scan_points::Integer = BHSR_SPIN_TOPOLOGY_SCAN_POINTS,
        max_iterations::Integer = 140,
        extremum_max_iterations::Integer = BHSR_SPIN_TOPOLOGY_EXTREMUM_MAX_ITERATIONS,
        absolute_tolerance::Real = big"1e-40")
    scan_points >= 3 || throw(ArgumentError("spin-topology scan needs at least three points"))
    tolerance = _bhsr_big(absolute_tolerance)
    stability_tolerance = big"1e-30"
    resolutions = [Int(scan_points), 2Int(scan_points) - 1, 4Int(scan_points) - 3]
    results = [_bhsr_isolate_positive_spin_intervals_at_resolution(f, points;
        max_iterations, extremum_max_iterations, absolute_tolerance) for points in resolutions]
    near_tangent_detected = any(result -> !isempty(result.near_tangent_extrema), results)
    stable = _bhsr_topology_endpoints_stable(results[1], results[2], stability_tolerance) &&
             _bhsr_topology_endpoints_stable(results[2], results[3], stability_tolerance)
    final = last(results)
    if !stable
        return (; roots = BigFloat[], candidate_roots = final.roots,
                intervals = Tuple{BigFloat,BigFloat}[],
                candidate_intervals = final.intervals,
                status = near_tangent_detected ? :unavailable_near_tangent :
                    :unavailable_topology_resolution,
                scan_points = final.scan_points,
                resolution_ladder = resolutions,
                resolution_status = near_tangent_detected ? :near_tangent : :unstable,
                near_tangent_extrema = final.near_tangent_extrema,
                refined_extrema = final.refined_extrema,
                isolation_method = final.isolation_method)
    end
    if near_tangent_detected
        return (; roots = BigFloat[], candidate_roots = final.roots,
                intervals = Tuple{BigFloat,BigFloat}[],
                candidate_intervals = final.intervals,
                status = :unavailable_near_tangent,
                scan_points = final.scan_points,
                resolution_ladder = resolutions,
                resolution_status = :near_tangent,
                near_tangent_extrema = final.near_tangent_extrema,
                refined_extrema = final.refined_extrema,
                isolation_method = final.isolation_method)
    end
    return (; roots = final.roots, candidate_roots = final.roots,
            intervals = final.intervals,
            candidate_intervals = final.intervals,
            status = final.status,
            scan_points = final.scan_points,
            resolution_ladder = resolutions,
            resolution_status = :stable,
            near_tangent_extrema = final.near_tangent_extrema,
            refined_extrema = final.refined_extrema,
            isolation_method = final.isolation_method)
end

_bhsr_topology_unavailable(status::Symbol) =
    status in (:unavailable_topology_resolution, :unavailable_near_tangent)

function _bhsr_union_spin_intervals(per_mode_topologies)
    intervals = Tuple{BigFloat,BigFloat}[]
    for topology in per_mode_topologies
        append!(intervals, topology.intervals)
    end
    isempty(intervals) && return intervals
    sort!(intervals; by = first)
    merged = Tuple{BigFloat,BigFloat}[]
    for (lower, upper) in intervals
        if isempty(merged) || lower >= last(merged)[2]
            push!(merged, (lower, upper))
        else
            previous_lower, previous_upper = pop!(merged)
            push!(merged, (previous_lower, max(previous_upper, upper)))
        end
    end
    return merged
end

function _bhsr_contour_topology_status(intervals)
    isempty(intervals) && return :no_positive_interval_resolved
    length(intervals) > 1 && return :multiple_boundaries_required
    lower, upper = only(intervals)
    lower == 0 && upper == 1 && return :all_spins_efficient
    lower == 0 && return :bounded_efficiency_from_zero
    upper == 1 && return :single_onset_to_extremal_spin
    return :bounded_efficiency_interval
end

"""Resolve spin crossings and efficient intervals for the source rate.

The frozen scalar `critical_spin` API returns only the first onset. This
result retains the crossings and positive intervals found by the recorded
resolution-stability audit on `[0,1]`.
"""
function bhsr_critical_spin_topology(mass_solar::Real, mu_eV::Real,
                                     mode::BHSRMode, tau_years::Real;
                                     model_id::AbstractString = "BHSR-CRITICAL-SPIN-TOPOLOGY",
                                     delta_a::Real = 0.1,
                                     precision_bits::Integer = 256,
                                     max_iterations::Integer = 140,
                                     absolute_tolerance::Real = big"1e-40",
                                     scan_points::Integer = BHSR_SPIN_TOPOLOGY_SCAN_POINTS)
    _bhsr_validate_reference_topology_contract(precision_bits, max_iterations,
                                               absolute_tolerance)
    id = _bhsr_model_id(model_id)
    return setprecision(BigFloat, precision_bits) do
        f(a) = bhsr_free_field_residual(mass_solar, mu_eV, a, mode, tau_years;
                                        delta_a, precision_bits)
        isolated = _bhsr_isolate_positive_spin_intervals(f; scan_points,
            max_iterations, absolute_tolerance)
        onset = _bhsr_topology_unavailable(isolated.status) ||
                isempty(isolated.intervals) ? missing : first(isolated.intervals)[1]
        return (; model_id = id, route_identity = "REFERENCE_2021_ANALYTIC",
                method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
                topology_method_addendum_sha256 = BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256,
                source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
                mass_solar = _bhsr_big(mass_solar), mu_eV = _bhsr_big(mu_eV),
                tau_years = _bhsr_big(tau_years), mode_label = mode.label,
                onset_spin = onset, roots = isolated.roots,
                candidate_roots = isolated.candidate_roots,
                intervals = isolated.intervals,
                candidate_intervals = isolated.candidate_intervals,
                refined_extrema = isolated.refined_extrema,
                topology_status = isolated.status,
                contour_topology_status = _bhsr_topology_unavailable(isolated.status) ?
                    isolated.status : _bhsr_contour_topology_status(isolated.intervals),
                scan_points = isolated.scan_points,
                resolution_ladder = isolated.resolution_ladder,
                resolution_status = isolated.resolution_status,
                near_tangent_extrema = isolated.near_tangent_extrema,
                near_tangent_residual_tolerance = BHSR_SPIN_TOPOLOGY_NEAR_TANGENT_TOLERANCE,
                extremum_max_iterations = BHSR_SPIN_TOPOLOGY_EXTREMUM_MAX_ITERATIONS,
                isolation_method = isolated.isolation_method)
    end
end

"""Return only the first efficiency onset spin.

This scalar does not describe the full efficient region. Call
`bhsr_critical_spin_topology` to inspect all crossings and intervals. A bounded
interval or multiple intervals cannot be represented as a single exclusion
contour and must be treated as unavailable by contour consumers.
"""
function bhsr_critical_spin(mass_solar::Real, mu_eV::Real, mode::BHSRMode,
                            tau_years::Real; kwargs...)
    return bhsr_critical_spin_topology(mass_solar, mu_eV, mode, tau_years;
                                       kwargs...).onset_spin
end

"Per-mode onsets and full interval union, with contour representability status."
function bhsr_regge_row(mass_solar::Real, mu_eV::Real, tau_years::Real,
                        modes::AbstractVector{BHSRMode} = bhsr_nodeless_modes();
                        model_id::AbstractString, delta_a::Real = big"0.1",
                        precision_bits::Integer = 256,
                        max_iterations::Integer = 140,
                        absolute_tolerance::Real = big"1e-40")
    _bhsr_validate_reference_topology_contract(precision_bits, max_iterations,
                                               absolute_tolerance)
    _bhsr_validate_modes(modes)
    id = _bhsr_model_id(model_id)
    topologies = [bhsr_critical_spin_topology(mass_solar, mu_eV, mode,
        tau_years; model_id = id, delta_a, precision_bits, max_iterations,
        absolute_tolerance)
        for mode in modes]
    roots = Pair{String,Union{Missing,BigFloat}}[]
    for (mode, topology) in zip(modes, topologies)
        push!(roots, mode.label => topology.onset_spin)
    end
    topology_resolved = all(topology -> !_bhsr_topology_unavailable(topology.topology_status), topologies)
    union_intervals = topology_resolved ? _bhsr_union_spin_intervals(topologies) :
        Tuple{BigFloat,BigFloat}[]
    union_item = isempty(union_intervals) ? nothing : begin
        onset = first(union_intervals)[1]
        matching = findfirst(topology -> any(interval -> interval[1] == onset,
                                              topology.intervals), topologies)
        (; mode = matching === nothing ? missing : modes[matching].label, spin = onset)
    end
    union_spin = union_item === nothing ? missing : union_item.spin
    failure_index = findfirst(topology -> _bhsr_topology_unavailable(topology.topology_status), topologies)
    contour_status = failure_index === nothing ? _bhsr_contour_topology_status(union_intervals) :
        topologies[failure_index].topology_status
    return (; model_id = id,
            route_identity = "REFERENCE_2021_ANALYTIC",
            method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
            topology_method_addendum_sha256 = BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256,
            source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
            mode_labels = getfield.(modes, :label),
            mass_solar = _bhsr_big(mass_solar),
            per_mode = roots,
            per_mode_topology = topologies,
            union_mode = union_item === nothing ? missing : union_item.mode,
            union_spin,
            union_intervals,
            topology_status = topology_resolved ? :resolved : contour_status,
            contour_topology_status = contour_status,
            onset_only = true,
            delta_a = _bhsr_big(delta_a),
            precision_bits)
end

"Direct logarithmic mass grid specified by the frozen Fig. 3 method manifest."
function bhsr_regge_grid(mu_eV::Real, tau_years::Real;
                         mass_log10_min::Real = -1,
                         mass_log10_max::Real = 2,
                         points::Integer = 1201,
                         modes::AbstractVector{BHSRMode} = bhsr_nodeless_modes(),
                         model_id::AbstractString,
                         delta_a::Real = big"0.1",
                         precision_bits::Integer = 256,
                         max_iterations::Integer = 140,
                         absolute_tolerance::Real = big"1e-40")
    _bhsr_validate_reference_topology_contract(precision_bits, max_iterations,
                                               absolute_tolerance)
    _bhsr_validate_modes(modes)
    points >= 2 || throw(ArgumentError("mass grid needs at least two points"))
    mass_log10_max > mass_log10_min || throw(ArgumentError("mass grid bounds are reversed"))
    id = _bhsr_model_id(model_id)
    return setprecision(BigFloat, precision_bits) do
        lower = BigFloat(mass_log10_min)
        width = BigFloat(mass_log10_max) - lower
        masses = [BigFloat(10)^(lower + width * BigFloat(i - 1) / (points - 1))
                  for i in 1:points]
        rows = Any[bhsr_regge_row(mass, mu_eV, tau_years, modes;
                                  model_id = id, delta_a, precision_bits,
                                  max_iterations, absolute_tolerance)
                   for mass in masses]
        union_spins = Union{Missing,BigFloat}[
            row.contour_topology_status != :single_onset_to_extremal_spin ||
            ismissing(row.union_spin) ? missing : row.union_spin for row in rows]
        source_grid = points == 1201 && lower == -1 && width == 3 &&
            getfield.(modes, :label) == getfield.(_BHSR_SOURCE_MODES, :label) &&
            precision_bits == 256 && max_iterations == 140 &&
            _bhsr_big(absolute_tolerance) == big"1e-40" &&
            _bhsr_big(delta_a) == big"0.1" &&
            _bhsr_big(mu_eV) == big"4.3e-12" &&
            _bhsr_big(tau_years) in (big"1e10", big"4.5e6")
        return BHSRContourGrid(id, "REFERENCE_2021_ANALYTIC",
            BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
            BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256,
            BHSR_SOURCE_MODE_MANIFEST_SHA256, masses, union_spins, rows,
            collect(modes), _bhsr_big(mu_eV), _bhsr_big(tau_years),
            _bhsr_big(delta_a), Int(precision_bits), Int(max_iterations),
            _bhsr_big(absolute_tolerance), lower, lower + width, source_grid, nothing)
    end
end

function _bhsr_union_spin_at_mass(grid::BHSRContourGrid, mass_solar::Real)
    if grid.contour_evaluator !== nothing
        value = grid.contour_evaluator(_bhsr_big(mass_solar))
        ismissing(value) && return missing
        value isa Real && isfinite(value) ||
            throw(ArgumentError("contour evaluator must return a finite spin or missing"))
        return _bhsr_big(value)
    end
    row = bhsr_regge_row(mass_solar, grid.mu_eV, grid.tau_years, grid.modes;
        model_id = grid.model_id, delta_a = grid.delta_a,
        precision_bits = grid.precision_bits,
        max_iterations = grid.root_max_iterations,
        absolute_tolerance = grid.root_absolute_tolerance)
    row.contour_topology_status == :single_onset_to_extremal_spin || return missing
    return row.union_spin
end

function (grid::BHSRContourGrid)(mass_solar::Real)
    mass = _bhsr_big(mass_solar)
    (first(grid.mass_solar) <= mass <= last(grid.mass_solar)) || return missing
    return _bhsr_union_spin_at_mass(grid, mass)
end

const BHSR_BOSENOVA_ROUTE = "REFERENCE_2021_BOSENOVA"
const BHSR_BOSENOVA_TIMESCALES_YEARS = ("1e10", "4.5e7", "4.5e6")

"Source Eq. (13) quantities, retaining the signed diagonal quartic as provenance."
function bhsr_bosenova_details(mass_solar::Real, mu_eV::Real, mode::BHSRMode,
                                lambda_iiii::Real; c_bose::Real = big"5",
                                reduced_planck_GeV::Real = big"2.435e18",
                                delta_a::Real = big"0.1",
                                precision_bits::Integer = 256)
    lambda = _bhsr_big(lambda_iiii)
    !iszero(lambda) || throw(DomainError(lambda, "lambda_iiii must be nonzero for finite f_pert"))
    c_bose > 0 || throw(ArgumentError("c_bose must be positive"))
    reduced_planck_GeV > 0 || throw(ArgumentError("reduced Planck mass must be positive"))
    return setprecision(BigFloat, precision_bits) do
        alpha = bhsr_alpha(mass_solar, mu_eV; precision_bits)
        alpha > 0 || throw(DomainError(alpha, "alpha must be positive"))
        N = bhsr_principal_number(mode)
        f_pert_GeV = (_bhsr_big(mu_eV) / BigFloat("1e9")) / sqrt(abs(lambda))
        n_bose = BigFloat("1e78") * _bhsr_big(c_bose) * BigFloat(N)^4 /
            alpha^3 * (_bhsr_big(mass_solar) / 10)^2 *
            (f_pert_GeV / _bhsr_big(reduced_planck_GeV))^2
        n_max = bhsr_nmax(mass_solar, mode; delta_a, precision_bits)
        return (; route_identity = BHSR_BOSENOVA_ROUTE,
                lambda_iiii_signed = lambda,
                lambda_iiii_magnitude = abs(lambda),
                principal_number = N,
                f_pert_GeV,
                n_bose,
                n_max,
                c_bose = _bhsr_big(c_bose),
                delta_a = _bhsr_big(delta_a),
                omitted_interactions = ("off_diagonal_quartics", "cubic_interactions"))
    end
end

"Residual for the source Eq. (26) self-interaction-modified spin-down condition."
function bhsr_bosenova_residual(mass_solar::Real, mu_eV::Real, spin::Real,
                                 mode::BHSRMode, lambda_iiii::Real,
                                 tau_years::Real; kwargs...)
    tau_years > 0 || throw(ArgumentError("black-hole age must be positive"))
    return setprecision(BigFloat, get(kwargs, :precision_bits, 256)) do
        details = bhsr_bosenova_details(mass_solar, mu_eV, mode, lambda_iiii; kwargs...)
        details.n_bose > 1 || throw(DomainError(details.n_bose, "source log threshold requires N_Bose > 1"))
        gamma = bhsr_rate_s(mass_solar, mu_eV, spin, mode;
                            precision_bits = get(kwargs, :precision_bits, 256))
        k = _bhsr_constants()
        gamma * _bhsr_big(tau_years) * k.year_s * (details.n_bose / details.n_max) -
            log(details.n_bose)
    end
end

"Strict Eq. (26) decision: equality is the transition and is not efficient."
bhsr_bosenova_efficient(residual::Real) = residual > 0

"""Resolve spin crossings and efficient intervals for source Eq. (26)."""
function bhsr_bosenova_critical_spin_topology(mass_solar::Real, mu_eV::Real,
        mode::BHSRMode, lambda_iiii::Real, tau_years::Real;
        model_id::AbstractString = "BHSR-BOSENOVA-SPIN-TOPOLOGY",
        precision_bits::Integer = 256,
        max_iterations::Integer = 140,
        absolute_tolerance::Real = big"1e-40",
        scan_points::Integer = BHSR_SPIN_TOPOLOGY_SCAN_POINTS,
        kwargs...)
    _bhsr_validate_reference_topology_contract(precision_bits, max_iterations,
                                               absolute_tolerance)
    id = _bhsr_model_id(model_id)
    return setprecision(BigFloat, precision_bits) do
        residual(a) = bhsr_bosenova_residual(mass_solar, mu_eV, a, mode,
                                             lambda_iiii, tau_years;
                                             precision_bits, kwargs...)
        isolated = _bhsr_isolate_positive_spin_intervals(residual; scan_points,
            max_iterations, absolute_tolerance)
        onset = _bhsr_topology_unavailable(isolated.status) ||
                isempty(isolated.intervals) ? missing : first(isolated.intervals)[1]
        return (; model_id = id, route_identity = BHSR_BOSENOVA_ROUTE,
                method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
                topology_method_addendum_sha256 = BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256,
                source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
                mass_solar = _bhsr_big(mass_solar), mu_eV = _bhsr_big(mu_eV),
                lambda_iiii = _bhsr_big(lambda_iiii), tau_years = _bhsr_big(tau_years),
                mode_label = mode.label, onset_spin = onset, roots = isolated.roots,
                candidate_roots = isolated.candidate_roots,
                intervals = isolated.intervals,
                candidate_intervals = isolated.candidate_intervals,
                refined_extrema = isolated.refined_extrema,
                topology_status = isolated.status,
                contour_topology_status = _bhsr_topology_unavailable(isolated.status) ?
                    isolated.status : _bhsr_contour_topology_status(isolated.intervals),
                scan_points = isolated.scan_points,
                resolution_ladder = isolated.resolution_ladder,
                resolution_status = isolated.resolution_status,
                near_tangent_extrema = isolated.near_tangent_extrema,
                near_tangent_residual_tolerance = BHSR_SPIN_TOPOLOGY_NEAR_TANGENT_TOLERANCE,
                extremum_max_iterations = BHSR_SPIN_TOPOLOGY_EXTREMUM_MAX_ITERATIONS,
                isolation_method = isolated.isolation_method)
    end
end

"Return only the first Eq. (26) efficiency onset spin; inspect full topology separately."
function bhsr_bosenova_critical_spin(mass_solar::Real, mu_eV::Real,
                                     mode::BHSRMode, lambda_iiii::Real,
                                     tau_years::Real; kwargs...)
    return bhsr_bosenova_critical_spin_topology(mass_solar, mu_eV, mode,
        lambda_iiii, tau_years; kwargs...).onset_spin
end

"Per-mode self-interaction transitions and their source-mode union boundary."
function bhsr_bosenova_regge_row(mass_solar::Real, mu_eV::Real,
                                  lambda_iiii::Real, tau_years::Real,
                                  modes::AbstractVector{BHSRMode} = bhsr_nodeless_modes();
                                  model_id::AbstractString, delta_a::Real = big"0.1",
                                  c_bose::Real = big"5",
                                  reduced_planck_GeV::Real = big"2.435e18",
                                  precision_bits::Integer = 256,
                                  max_iterations::Integer = 140,
                                  absolute_tolerance::Real = big"1e-40")
    _bhsr_validate_reference_topology_contract(precision_bits, max_iterations,
                                               absolute_tolerance)
    _bhsr_validate_modes(modes)
    id = _bhsr_model_id(model_id)
    topologies = [bhsr_bosenova_critical_spin_topology(mass_solar, mu_eV, mode,
        lambda_iiii, tau_years; model_id = id, delta_a, c_bose,
        reduced_planck_GeV, precision_bits, max_iterations, absolute_tolerance)
        for mode in modes]
    roots = Pair{String,Union{Missing,BigFloat}}[]
    for (mode, topology) in zip(modes, topologies)
        push!(roots, mode.label => topology.onset_spin)
    end
    topology_resolved = all(topology -> !_bhsr_topology_unavailable(topology.topology_status), topologies)
    union_intervals = topology_resolved ? _bhsr_union_spin_intervals(topologies) :
        Tuple{BigFloat,BigFloat}[]
    union_spin = isempty(union_intervals) ? missing : first(union_intervals)[1]
    failure_index = findfirst(topology -> _bhsr_topology_unavailable(topology.topology_status), topologies)
    contour_status = failure_index === nothing ? _bhsr_contour_topology_status(union_intervals) :
        topologies[failure_index].topology_status
    return (; model_id = id,
            route_identity = BHSR_BOSENOVA_ROUTE,
            method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
            topology_method_addendum_sha256 = BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256,
            source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
            mode_labels = getfield.(modes, :label),
            mass_solar = _bhsr_big(mass_solar),
            per_mode = roots,
            per_mode_topology = topologies,
            union_spin,
            union_intervals,
            topology_status = topology_resolved ? :resolved : contour_status,
            contour_topology_status = contour_status,
            onset_only = true)
end
"""Continued-fraction scalar bound-state validation for CYAX-0121.

The radial recurrence follows Dolan, arXiv:0705.2880, Eqs. (33)-(48), which
matches arXiv:1805.02016v2 Appendix A.4 Eqs. (77)-(94) after the source-typo
correction to chi authorized for this reconstruction. Frequencies are in M=1
units. The angular eigenvalue is computed by truncating the normalized
associated-Legendre basis to the requested number of modes.
"""


const BHSR_CF_ROUTE = "CF_2018_VALIDATION"
const BHSR_DOLAN_SOURCE = "arXiv:0705.2880v2"

function _bhsr_json_object_text(json_text::AbstractString, key::AbstractString)
    marker = match(Regex("\\\"" * key * "\\\"\\s*:\\s*\\{"), json_text)
    marker === nothing && error("frozen BHSR method manifest lacks object '$key'")
    start = marker.offset + ncodeunits(marker.match)
    depth = 1
    for index in start:lastindex(json_text)
        character = json_text[index]
        character == '{' && (depth += 1)
        character == '}' && (depth -= 1)
        depth == 0 && return String(SubString(json_text, start, prevind(json_text, index)))
    end
    error("unterminated object '$key' in frozen BHSR method manifest")
end

function _bhsr_json_int_array(object_text::AbstractString, key::AbstractString)
    found = match(Regex("\\\"" * key * "\\\"\\s*:\\s*\\[([^]]*)\\]"), object_text)
    found === nothing && error("frozen BHSR method manifest lacks integer array '$key'")
    values = [parse(Int, strip(token)) for token in split(found.captures[1], ',')
              if !isempty(strip(token))]
    isempty(values) && error("frozen BHSR method manifest has empty integer array '$key'")
    return values
end

"Read the CF target and refinement ladders from the hash-verified frozen manifest."
function bhsr_cf_method_spec()
    section = _bhsr_json_object_text(_BHSR_METHOD_MANIFEST_TEXT,
                                     "continued_fraction_method")
    target = _bhsr_json_object_text(section, "target")
    mode_label = _bhsr_json_string_field(target, "mode")
    mode_index = findfirst(mode -> mode.label == mode_label, _BHSR_SOURCE_MODES)
    mode_index === nothing && error("CF target mode is absent from frozen source-mode manifest")
    mode = _BHSR_SOURCE_MODES[mode_index]
    _bhsr_json_int_field(target, "N") == bhsr_principal_number(mode) ||
        error("CF target principal number does not match frozen mode identity")
    _bhsr_json_int_field(target, "l") == mode.l || error("CF target l does not match frozen mode")
    _bhsr_json_int_field(target, "m") == mode.m || error("CF target m does not match frozen mode")
    residual_text = _bhsr_json_string_field(section, "root_residual_tolerance")
    acceptance_text = _bhsr_json_string_field(section, "convergence_acceptance")
    return (; route_identity = BHSR_CF_ROUTE,
            alpha = parse(BigFloat, _bhsr_json_string_field(target, "alpha")),
            spin = parse(BigFloat, _bhsr_json_string_field(target, "spin")),
            mode,
            precision_ladder_bits = _bhsr_json_int_array(section, "precision_ladder_bits"),
            continued_fraction_orders = _bhsr_json_int_array(section, "continued_fraction_orders"),
            angular_spheroidal_truncations =
                _bhsr_json_int_array(section, "angular_spheroidal_truncations"),
            root_residual_tolerance = parse(BigFloat, residual_text),
            convergence_acceptance = acceptance_text,
            method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
            source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256)
end

_bhsr_cf_complex(x::Complex) = Complex{BigFloat}(_bhsr_big(real(x)), _bhsr_big(imag(x)))
_bhsr_cf_complex(x::Real) = Complex{BigFloat}(_bhsr_big(x), BigFloat(0))

function _bhsr_cf_cosine_coefficient(j::Int, m::Int)
    j < m && return BigFloat(0)
    return sqrt(BigFloat(j^2 - m^2) / BigFloat(4j^2 - 1))
end

"Angular separation eigenvalue Lambda near l(l+1), by a real-basis truncation."
function bhsr_cf_angular_eigenvalue(l::Integer, m::Integer, c_squared::Number;
                                    truncation::Integer, precision_bits::Integer = 256)
    0 <= m <= l || throw(ArgumentError("angular mode must satisfy 0 <= m <= l"))
    truncation >= 2 || throw(ArgumentError("angular truncation must include at least two basis modes"))
    return setprecision(BigFloat, precision_bits) do
        c2 = _bhsr_cf_complex(c_squared)
    angular_l = collect(Int(l):2:(Int(l) + 2 * (Int(truncation) - 1)))
        diagonal = Complex{BigFloat}[]
        offdiagonal = Complex{BigFloat}[]
        for j in angular_l
            lower = _bhsr_cf_cosine_coefficient(j, Int(m))
            upper = _bhsr_cf_cosine_coefficient(j + 1, Int(m))
            push!(diagonal, Complex{BigFloat}(BigFloat(j * (j + 1))) -
                             c2 * (lower^2 + upper^2))
        end
        for index in 1:(length(angular_l) - 1)
            j = angular_l[index]
            push!(offdiagonal, -c2 * _bhsr_cf_cosine_coefficient(j + 1, Int(m)) *
                               _bhsr_cf_cosine_coefficient(j + 2, Int(m)))
        end

        eigenvalue = Complex{BigFloat}(BigFloat(l * (l + 1)))
        step_tolerance = BigFloat(2)^(-precision_bits + 16)
        for _ in 1:100
            p_previous, dp_previous = Complex{BigFloat}(1), Complex{BigFloat}(0)
            p_current, dp_current = diagonal[1] - eigenvalue, Complex{BigFloat}(-1)
            for index in 2:length(diagonal)
                p_next = (diagonal[index] - eigenvalue) * p_current -
                         offdiagonal[index - 1]^2 * p_previous
                dp_next = -p_current + (diagonal[index] - eigenvalue) * dp_current -
                          offdiagonal[index - 1]^2 * dp_previous
                p_previous, dp_previous, p_current, dp_current =
                    p_current, dp_current, p_next, dp_next
            end
            derivative = dp_current
            iszero(derivative) && error("angular eigenvalue Newton derivative vanished")
            step = p_current / derivative
            eigenvalue -= step
            abs(step) <= step_tolerance && return eigenvalue
        end
        error("angular spheroidal eigenvalue did not converge")
    end
end

function _bhsr_cf_recurrence(omega::Complex{BigFloat}, alpha::BigFloat,
                             spin::BigFloat, mode::BHSRMode,
                             angular_truncation::Int, precision_bits::Int)
    b = sqrt(1 - spin^2)
    q = -sqrt(Complex{BigFloat}(alpha^2) - omega^2)
    real(q) < 0 || throw(DomainError(q, "bound-state branch requires Re(q) < 0"))
    r_plus = 1 + b
    omega_h = spin * mode.m / (2r_plus)
    sigma = 2r_plus * (omega - omega_h) / (2b)
    c_squared = spin^2 * (omega^2 - alpha^2)
    lambda = bhsr_cf_angular_eigenvalue(mode.l, mode.m, c_squared;
        truncation = angular_truncation, precision_bits)
    c0 = 1 - 2im * omega - (2im / b) * (omega - spin * mode.m / 2)
    c1 = -4 + 4im * (omega - im * q * (1 + b)) +
         (4im / b) * (omega - spin * mode.m / 2) - 2 * (omega^2 + q^2) / q
    c2 = 3 - 2im * omega - 2 * (q^2 - omega^2) / q -
         (2im / b) * (omega - spin * mode.m / 2)
    c3 = 2im * (omega - im * q)^3 / q + 2 * (omega - im * q)^2 * b +
         q^2 * spin^2 + 2im * q * spin * mode.m - lambda - 1 -
         (omega - im * q)^2 / q + 2q * b +
         (2im / b) * ((omega - im * q)^2 / q + 1) * (omega - spin * mode.m / 2)
    c4 = (omega - im * q)^4 / q^2 +
         2im * omega * (omega - im * q)^2 / q -
         (2im / (b * q)) * (omega - im * q)^2 * (omega - spin * mode.m / 2)
    alpha_n(n) = n^2 + (c0 + 1) * n + c0
    beta_n(n) = -2n^2 + (c1 + 2) * n + c3
    gamma_n(n) = n^2 + (c2 - 3) * n + c4
    return alpha_n, beta_n, gamma_n, q, lambda, sigma
end

"Evaluate the truncated Leaver continued-fraction eigenvalue residual."
function bhsr_cf_residual(omega::Complex, alpha::Real, spin::Real, mode::BHSRMode;
                          order::Integer, angular_truncation::Integer,
                          precision_bits::Integer = 256)
    order >= 2 || throw(ArgumentError("continued-fraction order must be at least two"))
    0 < alpha < 1 || throw(ArgumentError("bound-state target requires 0 < M*mu < 1"))
    0 <= spin < 1 || throw(ArgumentError("Kerr spin must lie in [0,1)"))
    _bhsr_validate_modes([mode])
    return setprecision(BigFloat, precision_bits) do
        w, a = _bhsr_cf_complex(omega), _bhsr_big(alpha)
        an, bn, gn, _, _, _ = _bhsr_cf_recurrence(
            w, a, _bhsr_big(spin), mode, Int(angular_truncation), Int(precision_bits))
        denominator = bn(Int(order))
        for n in (Int(order) - 1):-1:1
            iszero(denominator) && throw(DomainError(denominator, "CF denominator is zero"))
            denominator = bn(n) - an(n) * gn(n + 1) / denominator
        end
        iszero(denominator) && throw(DomainError(denominator, "CF denominator is zero"))
        bn(0) - an(0) * gn(1) / denominator
    end
end

function _bhsr_cf_analytic_seed(alpha::BigFloat, spin::BigFloat, mode::BHSRMode)
    principal = bhsr_principal_number(mode)
    omega_r = alpha * (1 - alpha^2 / (2BigFloat(principal)^2))
    r_plus = 1 + sqrt(1 - spin^2)
    omega_h = spin / (2r_plus)
    product = prod(BigFloat(j^2) * (1 - spin^2) +
                   4r_plus^2 * (BigFloat(mode.m) * omega_r - alpha)^2
                   for j in 1:mode.l)
    gamma_seed = 2alpha * r_plus * (BigFloat(mode.m) * omega_h - omega_r) *
                 alpha^(4mode.l + 4) * _bhsr_A(mode) * product
    return Complex{BigFloat}(omega_r, gamma_seed / 2)
end

"Solve one M=1 bound-state frequency by damped complex Newton iteration."
function bhsr_cf_solve(alpha::Real, spin::Real, mode::BHSRMode;
                       order::Integer, angular_truncation::Integer,
                       precision_bits::Integer = 256,
                       root_residual_tolerance::Real = big"1e-24",
                       max_iterations::Integer = 80,
                       initial_frequency::Union{Nothing,Complex} = nothing,
                       model_id::AbstractString = "CF-2018-DOLAN-VALIDATION")
    id = _bhsr_model_id(model_id)
    alpha > 0 || throw(ArgumentError("M*mu must be positive"))
    0 <= spin < 1 || throw(ArgumentError("Kerr spin must lie in [0,1)"))
    tolerance = _bhsr_big(root_residual_tolerance)
    tolerance > 0 || throw(ArgumentError("root residual tolerance must be positive"))
    max_iterations > 0 || throw(ArgumentError("Newton iteration limit must be positive"))
    return setprecision(BigFloat, precision_bits) do
        a = _bhsr_big(alpha)
        astar = _bhsr_big(spin)
        omega = isnothing(initial_frequency) ? _bhsr_cf_analytic_seed(a, astar, mode) :
                _bhsr_cf_complex(initial_frequency)
        finite_difference_step = max(BigFloat("1e-25"), abs(omega) * BigFloat("1e-20"))
        residual = bhsr_cf_residual(omega, a, astar, mode; order,
            angular_truncation, precision_bits)
        for iteration in 1:max_iterations
            if abs(residual) <= tolerance
                recurrence = _bhsr_cf_recurrence(omega, a, astar, mode,
                    Int(angular_truncation), Int(precision_bits))
                q = recurrence[4]
                return (; model_id = id,
                        route_identity = BHSR_CF_ROUTE,
                        method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
                        source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
                        source_id = BHSR_DOLAN_SOURCE,
                        alpha = a,
                        spin = astar,
                        mode_label = mode.label,
                        principal_number = bhsr_principal_number(mode),
                        omega_M = omega,
                        real_omega_M = real(omega),
                        imaginary_omega_M = imag(omega),
                        gamma_amplitude_M = imag(omega),
                        gamma_occupation_M = 2imag(omega),
                        q_bound_branch = q,
                        continued_fraction_order = Int(order),
                        angular_spheroidal_truncation = Int(angular_truncation),
                        precision_bits = Int(precision_bits),
                        root_residual = abs(residual),
                        root_residual_tolerance = tolerance,
                        iterations = iteration - 1,
                        status = :evaluated)
            end
            derivative = (bhsr_cf_residual(omega + finite_difference_step, a, astar,
                mode; order, angular_truncation, precision_bits) -
                bhsr_cf_residual(omega - finite_difference_step, a, astar,
                mode; order, angular_truncation, precision_bits)) /
                (2finite_difference_step)
            iszero(derivative) && error("CF Newton derivative vanished")
            correction = residual / derivative
            # A short backtracking search keeps the iterate on the bound-state
            # branch and requires the eigenvalue residual to decrease.
            accepted = false
            scale = BigFloat(1)
            for _ in 1:12
                candidate = omega - scale * correction
                candidate_residual = bhsr_cf_residual(candidate, a, astar, mode;
                    order, angular_truncation, precision_bits)
                if abs(candidate_residual) < abs(residual)
                    omega, residual, accepted = candidate, candidate_residual, true
                    break
                end
                scale /= 2
            end
            accepted || error("CF Newton iteration could not reduce the residual")
        end
        error("CF eigenfrequency did not reach residual tolerance $tolerance")
    end
end

function _bhsr_cf_relative_change(a::Complex, b::Complex)
    return abs(a - b) / max(abs(a), abs(b), eps(BigFloat))
end

function _bhsr_cf_relative_change(a::Real, b::Real)
    return abs(a - b) / max(abs(a), abs(b), eps(BigFloat))
end

"Run the frozen precision, CF-order, and angular-truncation ladders."
function bhsr_cf_manifest_refinement(; model_id::AbstractString = "CF-2018-FROZEN-TARGET")
    spec = bhsr_cf_method_spec()
    results = NamedTuple[]
    for precision in spec.precision_ladder_bits,
        order in spec.continued_fraction_orders,
        angular in spec.angular_spheroidal_truncations
        push!(results, bhsr_cf_solve(spec.alpha, spec.spin, spec.mode;
            order, angular_truncation = angular, precision_bits = precision,
            root_residual_tolerance = spec.root_residual_tolerance, model_id))
    end
    at(precision, order, angular) = only(filter(result ->
        result.precision_bits == precision &&
        result.continued_fraction_order == order &&
        result.angular_spheroidal_truncation == angular, results))
    p0, p1 = last(spec.precision_ladder_bits),
             spec.precision_ladder_bits[end - 1]
    n0, n1 = last(spec.continued_fraction_orders),
             spec.continued_fraction_orders[end - 1]
    a0, a1 = last(spec.angular_spheroidal_truncations),
             spec.angular_spheroidal_truncations[end - 1]
    precision_pair = (at(p1, n0, a0), at(p0, n0, a0))
    order_pair = (at(p0, n1, a0), at(p0, n0, a0))
    angular_pair = (at(p0, n0, a1), at(p0, n0, a0))
    compare(pair) = (; omega_relative_change =
            _bhsr_cf_relative_change(pair[1].omega_M, pair[2].omega_M),
        gamma_relative_change =
            _bhsr_cf_relative_change(pair[1].gamma_amplitude_M, pair[2].gamma_amplitude_M))
    changes = (; precision = compare(precision_pair),
                continued_fraction_order = compare(order_pair),
                angular_truncation = compare(angular_pair))
    threshold = big"1e-8"
    all(result -> result.root_residual <= spec.root_residual_tolerance, results) ||
        error("at least one frozen CF root missed its residual tolerance")
    converged = all(pair -> pair.omega_relative_change <= threshold &&
                            pair.gamma_relative_change <= threshold, values(changes))
    return (; model_id = _bhsr_model_id(model_id), route_identity = spec.route_identity,
            method_manifest_sha256 = spec.method_manifest_sha256,
            source_mode_manifest_sha256 = spec.source_mode_manifest_sha256,
            target = (; alpha = spec.alpha, spin = spec.spin, mode = spec.mode.label),
            results, refinement_changes = changes,
            relative_change_tolerance = threshold,
            status = converged ? :converged : :unavailable_refinement_convergence)
end

"Diagnostic CF solve at a published Dolan growth-rate table point."
function bhsr_cf_dolan_table_benchmark(alpha::Real, spin::Real;
                                       orders::AbstractVector{<:Integer} = [512, 1024, 2048],
                                       angular_truncations::AbstractVector{<:Integer} = [9, 13, 17],
                                       precision_bits::Integer = 256,
                                       published_growth_M::Real,
                                       model_id::AbstractString)
    mode = only(filter(item -> item.label == "|211>", _BHSR_SOURCE_MODES))
    results = NamedTuple[]
    for order in orders, angular in angular_truncations
        push!(results, bhsr_cf_solve(alpha, spin, mode; order,
            angular_truncation = angular, precision_bits,
            root_residual_tolerance = big"1e-24", model_id))
    end
    reference = _bhsr_big(published_growth_M)
    final_order = last(orders)
    final_angular = last(angular_truncations)
    final = only(filter(result -> result.continued_fraction_order == final_order &&
        result.angular_spheroidal_truncation == final_angular, results))
    penultimate_order = orders[end - 1]
    penultimate_angular = angular_truncations[end]
    penultimate = only(filter(result ->
        result.continued_fraction_order == penultimate_order &&
        result.angular_spheroidal_truncation == penultimate_angular, results))
    return (; model_id = _bhsr_model_id(model_id),
            route_identity = BHSR_CF_ROUTE,
            source_id = BHSR_DOLAN_SOURCE,
            benchmark_locator = "Dolan, Phys. Rev. D 76, 084001 (2007), Table III, PDF p. 11; arXiv:0705.2880v2 HTML Table 1, lines 278-282: maximum M*Im(omega)",
            method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
            source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
            alpha = _bhsr_big(alpha), spin = _bhsr_big(spin),
            mode_label = mode.label, published_growth_M = reference,
            results,
            final_growth_M = final.gamma_amplitude_M,
            relative_table_difference = abs(final.gamma_amplitude_M - reference) / reference,
            final_order_relative_change =
                _bhsr_cf_relative_change(penultimate.gamma_amplitude_M,
                                         final.gamma_amplitude_M),
            status = :diagnostic_benchmark)
end
