"""Source-approximation helpers for the Appendix-B BHSR likelihood.

These functions implement the projected one-dimensional Gaussian method and
probability products. Caller-supplied curves and sigmas produce diagnostics
only. The source Table I does not define a complete Gaussian-sigma set, so no
authoritative probability or source-wide threshold is returned.
"""

if !isdefined(@__MODULE__, :BHSR_SI)
    include(joinpath(@__DIR__, "bhsr_regge_reconstruction.jl"))
end

const BHSR_APPENDIX_B_ROUTE = "REFERENCE_2018_APPENDIX_B"
const BHSR_APPENDIX_B_EXPECTED_ROWS = 24
const BHSR_APPENDIX_B_UNRESOLVED_ROWS = (
    "Cygnus X-1", "GRS 1915+105", "NGC 3783", "MCG-6-30-15", "Mrk 110", "NGC 4051",
)
const BHSR_OBSERVATIONAL_MANIFEST = joinpath(BHSR_VALIDATION_DIR,
                                              "observational_data_manifest.json")
const BHSR_OBSERVATIONAL_MANIFEST_SHA256 =
    "9da9f70efc6e654e3c38d963bf6a9e5ad9bec791a56ff2b43025bf019a7a87db"
const _BHSR_OBSERVATIONAL_MANIFEST_TEXT =
    _bhsr_verified_manifest(BHSR_OBSERVATIONAL_MANIFEST,
                            BHSR_OBSERVATIONAL_MANIFEST_SHA256)
const _BHSR_APPENDIX_B_MANIFEST_SECTION = let found = match(
    r"\"reference_2018_appendix_b\"\s*:\s*\{(.*?)\n\s*\},\s*\"reference_2021_analytic_dataset\""s,
    _BHSR_OBSERVATIONAL_MANIFEST_TEXT)
    found === nothing && error("frozen observational manifest lacks the Appendix-B section")
    String(found.captures[1])
end
const BHSR_APPENDIX_B_SOURCE_SIGMA_STATUS =
    _bhsr_json_string_field(_BHSR_APPENDIX_B_MANIFEST_SECTION, "status")

struct BHSRBHIdentityEnsemble
    route_identity::String
    manifest_id::String
    observational_manifest_sha256::String
    bh_ids::Vector{String}
end

function bhsr_bh_identity_ensemble(route_identity::AbstractString = BHSR_APPENDIX_B_ROUTE)
    route = String(route_identity)
    route == BHSR_APPENDIX_B_ROUTE ||
        throw(ArgumentError("the frozen observational manifest does not enumerate BH identities for route '$route'"))
    section = _BHSR_APPENDIX_B_MANIFEST_SECTION
    rows = _bhsr_json_object_blocks(_bhsr_json_array_text(section, "rows"))
    ids = [_bhsr_json_string_field(row, "name") for row in rows]
    expected_count = _bhsr_json_int_field(section, "row_count")
    length(ids) == expected_count || error("frozen Appendix-B BH row count is inconsistent")
    length(unique(ids)) == length(ids) || error("frozen Appendix-B BH identities are duplicated")
    manifest_id = _bhsr_json_string_field(_BHSR_OBSERVATIONAL_MANIFEST_TEXT,
                                          "manifest_id")
    return BHSRBHIdentityEnsemble(route, manifest_id,
                                  BHSR_OBSERVATIONAL_MANIFEST_SHA256, ids)
end

function _bhsr_validate_bh_ensemble(ensemble::BHSRBHIdentityEnsemble)
    expected = bhsr_bh_identity_ensemble(ensemble.route_identity)
    ensemble.manifest_id == expected.manifest_id ||
        throw(ArgumentError("BH ensemble manifest identity is stale or unsupported"))
    ensemble.observational_manifest_sha256 == expected.observational_manifest_sha256 ||
        throw(ArgumentError("BH ensemble observational manifest hash is stale"))
    ensemble.bh_ids == expected.bh_ids ||
        throw(ArgumentError("BH ensemble identity set does not match its source manifest"))
    return ensemble
end

struct BHSRUnionContour{C}
    contours::C
    model_id::String
    route_identity::String
    method_manifest_sha256::String
    source_mode_manifest_sha256::String
    mode_labels::Vector{String}
    source_backed::Bool
end

function (contour::BHSRUnionContour)(x)
    return bhsr_union_boundary([curve(x) for curve in contour.contours])
end

"Union contour wrapper with explicit model and frozen-manifest identity."
function bhsr_union_boundary_function(contours::AbstractVector;
                                      model_id::AbstractString,
                                      mode_labels::AbstractVector{<:AbstractString} = String[])
    isempty(contours) && throw(ArgumentError("at least one mode contour is required"))
    id = _bhsr_model_id(model_id)
    labels = String.(mode_labels)
    if !isempty(labels)
        length(labels) == length(contours) ||
            throw(DimensionMismatch("each source contour needs one mode identity"))
        canonical = getfield.(_BHSR_SOURCE_MODES, :label)
        length(unique(labels)) == length(labels) ||
            throw(ArgumentError("contour mode identities must be unique"))
        all(label -> label in canonical, labels) ||
            throw(ArgumentError("contour mode identity is absent from the frozen source manifest"))
        source_order = [label for label in canonical if label in labels]
        labels == source_order ||
            throw(ArgumentError("contour mode identities must retain frozen manifest order"))
    end
    # A caller-supplied closure and mode labels do not prove source provenance.
    return BHSRUnionContour(contours, id, BHSR_APPENDIX_B_ROUTE,
        BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
        BHSR_SOURCE_MODE_MANIFEST_SHA256, labels, false)
end

function _bhsr_contour_identity(contour, model_id::AbstractString)
    id = _bhsr_model_id(model_id)
    if contour isa BHSRContourGrid
        contour.model_id == id || throw(ArgumentError("contour and likelihood model_id differ"))
        contour.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256 ||
            throw(ArgumentError("contour method-manifest identity is stale"))
        contour.source_mode_manifest_sha256 == BHSR_SOURCE_MODE_MANIFEST_SHA256 ||
            throw(ArgumentError("contour source-mode manifest identity is stale"))
        # The exact numerical recipe can be reproduced, but the frozen
        # manifest has no digitized Fig. 3 targets or authorized sigma set.
        return (; source_backed = false,
                contour_provenance_status = contour.source_grid ?
                    :matching_recipe_without_source_target_data : :custom_grid,
                mode_labels = getfield.(contour.modes, :label),
                contour_route_identity = contour.route_identity,
                contour_mass_support_solar =
                    (first(contour.mass_solar), last(contour.mass_solar)))
    elseif contour isa BHSRUnionContour
        contour.model_id == id || throw(ArgumentError("contour and likelihood model_id differ"))
        contour.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256 ||
            throw(ArgumentError("contour method-manifest identity is stale"))
        contour.source_mode_manifest_sha256 == BHSR_SOURCE_MODE_MANIFEST_SHA256 ||
            throw(ArgumentError("contour source-mode manifest identity is stale"))
        # `source_backed` on the public wrapper is never accepted as proof.
        return (; source_backed = false,
                contour_provenance_status = :caller_supplied_contour_functions,
                mode_labels = contour.mode_labels,
                contour_route_identity = contour.route_identity,
                contour_mass_support_solar = missing)
    end
    return (; source_backed = false,
            contour_provenance_status = :unsupported_contour_type,
            mode_labels = String[],
            contour_route_identity = "UNSPECIFIED_CONTOUR",
            contour_mass_support_solar = missing)
end

function _bhsr_likelihood_metadata(contour, model_id::AbstractString,
                                   axion_id::AbstractString, bh_id::AbstractString,
                                   bh_ensemble::BHSRBHIdentityEnsemble)
    id = _bhsr_model_id(model_id)
    axion = strip(String(axion_id))
    bh = strip(String(bh_id))
    isempty(axion) && throw(ArgumentError("axion_id must be explicit and nonempty"))
    isempty(bh) && throw(ArgumentError("bh_id must be explicit and nonempty"))
    ensemble = _bhsr_validate_bh_ensemble(bh_ensemble)
    bh in ensemble.bh_ids || throw(ArgumentError("bh_id is absent from the declared source ensemble"))
    contour_identity = _bhsr_contour_identity(contour, id)
    return (; model_id = id, axion_id = axion, bh_id = bh,
            route_identity = BHSR_APPENDIX_B_ROUTE,
            likelihood_route_identity = BHSR_APPENDIX_B_ROUTE,
            contour_route_identity = contour_identity.contour_route_identity,
            contour_mass_support_solar = contour_identity.contour_mass_support_solar,
            contour_provenance_status = contour_identity.contour_provenance_status,
            method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
            source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
            bh_ensemble_id = ensemble.manifest_id,
            observational_manifest_sha256 = ensemble.observational_manifest_sha256,
            mode_labels = contour_identity.mode_labels,
            source_backed = false,
            authority_status = :unavailable_source_sigma_set,
            source_sigma_manifest_status = BHSR_APPENDIX_B_SOURCE_SIGMA_STATUS,
            diagnostic_only = true,
            diagnostic_status = :not_evaluated,
            diagnostic_probability_allowed = missing,
            diagnostic_probability_disallowed_between_branches = missing,
            probability_disallowed_between_branches = missing,
            probability_allowed = missing)
end

function _bhsr_contour_value(f, x::BigFloat)
    value = f(x)
    ismissing(value) && return missing
    value isa Real || throw(ArgumentError("contour function must return a real value or missing"))
    isfinite(value) || return missing
    return _bhsr_big(value)
end

function _bhsr_erfc(x::BigFloat)
    y = BigFloat(0)
    ccall((:mpfr_erfc, Base.MPFR.libmpfr), Int32,
          (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode),
          y, x, Base.MPFR.ROUNDING_MODE[])
    return y
end

"Standard normal CDF evaluated by MPFR at the active BigFloat precision."
function bhsr_standard_normal_cdf(z::Real; precision_bits::Integer = 256)
    return setprecision(BigFloat, precision_bits) do
        x = _bhsr_big(z) / sqrt(BigFloat(2))
        x < 0 ? _bhsr_erfc(-x) / 2 : 1 - _bhsr_erfc(x) / 2
    end
end

"Five-point centered derivative, with a second-order one-sided boundary rule."
function bhsr_contour_derivative(f, x::Real; lower::Real = -Inf,
                                  upper::Real = Inf,
                                  step_scale::Real = big"1e-5")
    xb, lo, hi = _bhsr_big(x), _bhsr_big(lower), _bhsr_big(upper)
    h = _bhsr_big(step_scale) * max(BigFloat(1), abs(xb))
    h > 0 || throw(ArgumentError("derivative step must be positive"))
    if xb - 2h >= lo && xb + 2h <= hi
        values = map(x -> _bhsr_contour_value(f, x), (xb - 2h, xb - h, xb + h, xb + 2h))
        any(ismissing, values) && return missing
        return (values[1] - 8values[2] + 8values[3] - values[4]) / (12h)
    elseif xb - 2h < lo
        xb + 2h <= hi || throw(ArgumentError("support is too narrow for a one-sided derivative"))
        values = map(x -> _bhsr_contour_value(f, x), (xb, xb + h, xb + 2h))
        any(ismissing, values) && return missing
        return (-3values[1] + 4values[2] - values[3]) / (2h)
    else
        xb - 2h >= lo || throw(ArgumentError("support is too narrow for a one-sided derivative"))
        values = map(x -> _bhsr_contour_value(f, x), (xb, xb - h, xb - 2h))
        any(ismissing, values) && return missing
        return (3values[1] - 4values[2] + values[3]) / (2h)
    end
end

function bhsr_projected_sigma_y(sigma_x::Real, sigma_y::Real, derivative::Real)
    sx, sy, fp = _bhsr_big(sigma_x), _bhsr_big(sigma_y), _bhsr_big(derivative)
    sx > 0 && sy > 0 || throw(ArgumentError("measurement sigmas must be positive"))
    return sqrt(sy^2 + fp^2 * sx^2)
end

function bhsr_projected_sigma_x(sigma_x::Real, sigma_y::Real, inverse_derivative::Real)
    sx, sy, gp = _bhsr_big(sigma_x), _bhsr_big(sigma_y), _bhsr_big(inverse_derivative)
    sx > 0 && sy > 0 || throw(ArgumentError("measurement sigmas must be positive"))
    return sqrt(sx^2 + gp^2 * sy^2)
end

"Diagnostic calculation of the Eq. (97) projection for y=f(x) at xbar."
function bhsr_allowed_probability_direct(xbar::Real, ybar::Real,
                                          sigma_x::Real, sigma_y::Real,
                                          f; lower::Real = -Inf,
                                          upper::Real = Inf,
                                          model_id::AbstractString,
                                          axion_id::AbstractString,
                                          bh_id::AbstractString,
                                          bh_ensemble::BHSRBHIdentityEnsemble)
    identity = _bhsr_likelihood_metadata(f, model_id, axion_id, bh_id, bh_ensemble)
    xb, yb = _bhsr_big(xbar), _bhsr_big(ybar)
    lo, hi = _bhsr_big(lower), _bhsr_big(upper)
    if f isa BHSRContourGrid &&
       !(first(f.mass_solar) <= xb <= last(f.mass_solar))
        return merge(identity, (; projection = :y_of_x,
                status = :outside_frozen_mass_support,
                derivative = missing,
                sigma_effective = missing,
                probability_allowed = missing))
    end
    if xb < lo || xb > hi
        return merge(identity, (;
                projection = :y_of_x,
                status = :outside_contour_support,
                derivative = missing,
                sigma_effective = missing,
                probability_allowed = missing))
    end
    fbar = _bhsr_contour_value(f, xb)
    if ismissing(fbar)
        return merge(identity, (;
                projection = :y_of_x,
                status = :missing_contour_value,
                derivative = missing,
                sigma_effective = missing,
                probability_allowed = missing))
    end
    derivative_lower, derivative_upper = lo, hi
    if f isa BHSRContourGrid
        derivative_lower = max(lo, first(f.mass_solar))
        derivative_upper = min(hi, last(f.mass_solar))
    end
    derivative = bhsr_contour_derivative(f, xb;
        lower = derivative_lower, upper = derivative_upper)
    if ismissing(derivative)
        return merge(identity, (;
                projection = :y_of_x,
                status = :incomplete_derivative_support,
                derivative = missing,
                sigma_effective = missing,
                probability_allowed = missing))
    end
    sigma_eff = bhsr_projected_sigma_y(sigma_x, sigma_y, derivative)
    probability = bhsr_standard_normal_cdf((fbar - yb) / sigma_eff)
    return merge(identity, (;
            projection = :y_of_x,
            status = :nonauthoritative_diagnostic,
            diagnostic_status = :evaluated,
            derivative,
            sigma_effective = sigma_eff,
            diagnostic_probability_allowed = probability,
            probability_allowed = missing))
end

function _bhsr_monotone_inverse_branches(grid::BHSRContourGrid)
    y = grid.union_spin
    finite_indices = findall(value -> !ismissing(value), y)
    isempty(finite_indices) && return nothing, :no_finite_contour_support
    first_finite, last_finite = first(finite_indices), last(finite_indices)
    any(ismissing, y[first_finite:last_finite]) &&
        return nothing, :internal_contour_support_gap
    first_finite < last_finite || return nothing, :insufficient_contour_support

    branches = NamedTuple[]
    start_index = first_finite
    direction = sign(y[start_index + 1] - y[start_index])
    iszero(direction) && return nothing, :plateau_or_ambiguous_branch
    for i in (start_index + 1):(last_finite - 1)
        next_direction = sign(y[i + 1] - y[i])
        iszero(next_direction) && return nothing, :plateau_or_ambiguous_branch
        if next_direction != direction
            push!(branches, (; first_index = start_index, last_index = i))
            start_index = i
            direction = next_direction
        end
    end
    push!(branches, (; first_index = start_index, last_index = last_finite))
    return branches, :supported
end

function _bhsr_inverse_root_on_branch(grid::BHSRContourGrid, branch,
                                      target_spin::BigFloat;
                                      absolute_tolerance::BigFloat = big"1e-12",
                                      max_iterations::Integer = 180)
    first_index, last_index = branch.first_index, branch.last_index
    left, right = grid.mass_solar[first_index], grid.mass_solar[last_index]
    yleft, yright = grid.union_spin[first_index], grid.union_spin[last_index]
    min(yleft, yright) <= target_spin <= max(yleft, yright) || return nothing
    fleft, fright = yleft - target_spin, yright - target_spin
    iszero(fleft) && return (; mass_solar = left, branch)
    iszero(fright) && return (; mass_solar = right, branch)
    sign(fleft) != sign(fright) || return nothing

    for _ in 1:max_iterations
        right - left <= absolute_tolerance &&
            return (; mass_solar = (left + right) / 2, branch)
        midpoint = (left + right) / 2
        ymid = _bhsr_union_spin_at_mass(grid, midpoint)
        ismissing(ymid) && return :unsupported_support
        fmid = ymid - target_spin
        iszero(fmid) && return (; mass_solar = midpoint, branch)
        if sign(fmid) == sign(fleft)
            left, fleft = midpoint, fmid
        else
            right = midpoint
        end
    end
    return :root_nonconvergence
end

"""Diagnostic calculation of Eq. (98) for a grid with exactly two inverse roots.

The root locations are bracketed on the frozen mass grid and refined against
the analytic union contour. One-root, more-than-two-root, support-gap, and
mode-switch cases fail closed because Appendix B specifies the interval only
for its two-valued inverse.
"""
function bhsr_allowed_probability_inverse(xbar::Real, ybar::Real,
                                           sigma_x::Real, sigma_y::Real,
                                           grid::BHSRContourGrid;
                                           model_id::AbstractString,
                                           axion_id::AbstractString,
                                           bh_id::AbstractString,
                                           bh_ensemble::BHSRBHIdentityEnsemble,
                                           absolute_tolerance::Real = big"1e-12",
                                           max_iterations::Integer = 180)
    identity = _bhsr_likelihood_metadata(grid, model_id, axion_id, bh_id, bh_ensemble)
    xb, yb = _bhsr_big(xbar), _bhsr_big(ybar)
    atol = _bhsr_big(absolute_tolerance)
    atol > 0 || throw(ArgumentError("inverse root tolerance must be positive"))
    max_iterations > 0 || throw(ArgumentError("inverse root iteration limit must be positive"))
    unavailable(status; roots = BigFloat[], detail = "") =
        merge(identity, (; projection = :inverse_x, status,
                          inverse_roots = roots,
                          probability_disallowed_between_branches = missing,
                          nearest_branch = missing,
                          inverse_derivative = missing,
                          sigma_effective = missing,
                          probability_allowed = missing,
                          detail))

    (first(grid.mass_solar) <= xb <= last(grid.mass_solar)) ||
        return unavailable(:outside_frozen_mass_support)
    !ismissing(grid(xb)) && return unavailable(:direct_contour_is_defined)
    branches, branch_status = _bhsr_monotone_inverse_branches(grid)
    branches === nothing && return unavailable(branch_status)

    # A union-envelope switch between source modes is a cusp. Do not bracket
    # across that unresolved branch switch or assign it a one-sided derivative.
    for i in 1:(length(grid.mass_solar) - 1)
        y1, y2 = grid.union_spin[i], grid.union_spin[i + 1]
        if !ismissing(y1) && !ismissing(y2) &&
           grid.rows[i].union_mode != grid.rows[i + 1].union_mode &&
           min(y1, y2) <= yb <= max(y1, y2)
            return unavailable(:ambiguous_mode_switch_bracket)
        end
    end

    roots_with_branches = NamedTuple[]
    for branch in branches
        root = _bhsr_inverse_root_on_branch(grid, branch, yb;
            absolute_tolerance = atol, max_iterations)
        root === nothing && continue
        root isa Symbol && return unavailable(root)
        push!(roots_with_branches, root)
    end
    sort!(roots_with_branches; by = item -> item.mass_solar)
    roots = BigFloat[]
    unique_roots = NamedTuple[]
    for root in roots_with_branches
        if isempty(roots) || abs(root.mass_solar - last(roots)) > atol
            push!(roots, root.mass_solar)
            push!(unique_roots, root)
        end
    end
    length(roots) == 2 || return unavailable(
        length(roots) == 1 ? :unsupported_single_inverse_branch :
        length(roots) > 2 ? :unsupported_inverse_topology : :no_inverse_roots;
        roots)

    slopes = BigFloat[]
    for root in unique_roots
        branch = root.branch
        lower_spin = min(grid.union_spin[branch.first_index],
                         grid.union_spin[branch.last_index])
        upper_spin = max(grid.union_spin[branch.first_index],
                         grid.union_spin[branch.last_index])
        inverse_function(y) = begin
            refined = _bhsr_inverse_root_on_branch(grid, branch, _bhsr_big(y);
                absolute_tolerance = atol, max_iterations)
            refined isa NamedTuple ? refined.mass_solar : missing
        end
        slope = try
            bhsr_contour_derivative(inverse_function, yb;
                                    lower = lower_spin, upper = upper_spin)
        catch error
            error isa ArgumentError || rethrow()
            missing
        end
        ismissing(slope) && return unavailable(:incomplete_inverse_derivative_support; roots)
        push!(slopes, slope)
    end
    distances = abs.(roots .- xb)
    nearest_distance = minimum(distances)
    count(==(nearest_distance), distances) == 1 ||
        return unavailable(:ambiguous_nearest_inverse_branch; roots)
    nearest = findfirst(==(nearest_distance), distances)
    sigma_eff = bhsr_projected_sigma_x(sigma_x, sigma_y, slopes[nearest])
    low, high = extrema(roots)
    p_inside = bhsr_standard_normal_cdf((high - xb) / sigma_eff) -
        bhsr_standard_normal_cdf((low - xb) / sigma_eff)
    p_allowed = clamp(BigFloat(1) - p_inside, BigFloat(0), BigFloat(1))
    return merge(identity, (; projection = :inverse_x,
            status = :nonauthoritative_diagnostic,
            diagnostic_status = :evaluated,
            inverse_roots = roots,
            nearest_branch = nearest,
            inverse_derivative = slopes[nearest],
            sigma_effective = sigma_eff,
            diagnostic_probability_disallowed_between_branches = p_inside,
            probability_disallowed_between_branches = missing,
            diagnostic_probability_allowed = p_allowed,
            probability_allowed = missing,
            detail = "two source-grid-bracketed inverse roots"))
end

"Union/envelope boundary from the per-mode exclusion thresholds at one mass."
function bhsr_union_boundary(values::AbstractVector)
    present = [_bhsr_big(value) for value in values if !ismissing(value)]
    return isempty(present) ? missing : minimum(present)
end

function _bhsr_probability_product(values)
    isempty(values) && throw(ArgumentError("probability products require nonempty inputs"))
    any(ismissing, values) && return missing
    probabilities = _bhsr_big.(values)
    all(p -> 0 <= p <= 1, probabilities) ||
        throw(DomainError(probabilities, "allowed probabilities must lie in [0,1]"))
    return prod(probabilities)
end

function _bhsr_tree_incomplete(model_id, axion_ids, ensemble, received;
                               duplicate_terms = Tuple{String,String}[],
                               missing_terms = Tuple{String,String}[],
                               unexpected_axions = String[],
                               unexpected_bh_ids = String[],
                               unresolved_terms = Tuple{String,String}[],
                               source_sigma_gate_status::Symbol = :not_checked,
                               authority_status::Symbol = :unavailable_incomplete_coverage)
    return (; model_id,
            route_identity = BHSR_APPENDIX_B_ROUTE,
            method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
            source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
            bh_ensemble_id = ensemble.manifest_id,
            observational_manifest_sha256 = ensemble.observational_manifest_sha256,
            source_relevant_axion_ids = axion_ids,
            expected_bh_ids = ensemble.bh_ids,
            received_terms = received,
            duplicate_terms,
            missing_terms,
            unexpected_axions,
            unexpected_bh_ids,
            unresolved_terms,
            source_sigma_gate_status,
            authority_status,
            status = :unavailable,
            single_axion_allowed = fill(missing, length(axion_ids)),
            geometry_allowed = missing,
            geometry_excluded = missing,
            threshold_exceeded = missing)
end

"""Eq. (96) product with explicit source-ensemble coverage.

Each element must be a named likelihood result with `axion_id`, `bh_id`, the
frozen manifest hashes, `model_id`, and a resolved probability. The caller must
state every source-relevant axion identity. The BH identity set comes from an
explicit, hash-verified observational ensemble; this release enumerates the
2018 Table-I ensemble only. Caller-supplied contour functions and sigma values
remain formula diagnostics. The frozen observational manifest declares no
complete source-defined Gaussian sigma set, so this version never returns an
authoritative geometry probability or threshold result.
"""
function bhsr_probability_tree(likelihood_results::AbstractVector;
                               model_id::AbstractString,
                               source_relevant_axion_ids::AbstractVector{<:AbstractString},
                               bh_ensemble::BHSRBHIdentityEnsemble)
    id = _bhsr_model_id(model_id)
    axion_ids = strip.(String.(source_relevant_axion_ids))
    isempty(axion_ids) && throw(ArgumentError("source-relevant axion identities must be explicit"))
    any(isempty, axion_ids) && throw(ArgumentError("source-relevant axion identities cannot be empty"))
    length(unique(axion_ids)) == length(axion_ids) ||
        throw(ArgumentError("source-relevant axion identities must be unique"))
    ensemble = _bhsr_validate_bh_ensemble(bh_ensemble)
    expected_axions = Set(axion_ids)
    expected_bhs = Set(ensemble.bh_ids)
    received = Tuple{String,String}[]
    duplicates = Tuple{String,String}[]
    unresolved = Tuple{String,String}[]
    unexpected_axions = Set{String}()
    unexpected_bhs = Set{String}()
    seen = Dict{Tuple{String,String},Any}()

    for result in likelihood_results
        if result === missing || !(result isa NamedTuple)
            push!(unresolved, ("<missing-identity>", "<missing-identity>"))
            continue
        end
        required = (:model_id, :axion_id, :bh_id, :method_manifest_sha256,
                    :source_mode_manifest_sha256, :bh_ensemble_id,
                    :observational_manifest_sha256, :source_backed, :status,
                    :probability_allowed)
        if !all(field -> hasproperty(result, field), required)
            axion = hasproperty(result, :axion_id) ? String(result.axion_id) : "<missing-axion>"
            bh = hasproperty(result, :bh_id) ? String(result.bh_id) : "<missing-bh>"
            push!(unresolved, (axion, bh))
            continue
        end
        axion, bh = strip(String(result.axion_id)), strip(String(result.bh_id))
        pair = (axion, bh)
        push!(received, pair)
        axion in expected_axions || push!(unexpected_axions, axion)
        bh in expected_bhs || push!(unexpected_bhs, bh)
        if haskey(seen, pair)
            push!(duplicates, pair)
            continue
        end
        seen[pair] = result
        valid_identity = result.model_id == id &&
            result.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256 &&
            result.source_mode_manifest_sha256 == BHSR_SOURCE_MODE_MANIFEST_SHA256 &&
            result.bh_ensemble_id == ensemble.manifest_id &&
            result.observational_manifest_sha256 == ensemble.observational_manifest_sha256 &&
            result.source_backed === true
        probability_valid = result.status == :evaluated &&
            !ismissing(result.probability_allowed) &&
            result.probability_allowed isa Real &&
            isfinite(result.probability_allowed) && 0 <= result.probability_allowed <= 1
        (valid_identity && probability_valid && axion in expected_axions && bh in expected_bhs) ||
            push!(unresolved, pair)
    end

    missing_terms = Tuple{String,String}[(axion, bh)
        for axion in axion_ids for bh in ensemble.bh_ids
        if !haskey(seen, (axion, bh))]
    if !isempty(duplicates) || !isempty(missing_terms) || !isempty(unexpected_axions) ||
       !isempty(unexpected_bhs) || !isempty(unresolved)
        return _bhsr_tree_incomplete(id, axion_ids, ensemble, received;
            duplicate_terms = unique(duplicates), missing_terms,
            unexpected_axions = sort!(collect(unexpected_axions)),
            unexpected_bh_ids = sort!(collect(unexpected_bhs)), unresolved_terms = unique(unresolved))
    end

    source_sigma_gate = bhsr_sigma_gate(Any[]; bh_ensemble = ensemble)
    if !source_sigma_gate.source_authoritative
        sigma_unresolved = [("<source-defined-Gaussian-sigma-set>",
                             source_sigma_gate.source_manifest_status)]
        return _bhsr_tree_incomplete(id, axion_ids, ensemble, received;
            unresolved_terms = sigma_unresolved,
            source_sigma_gate_status = source_sigma_gate.status,
            authority_status = :unavailable_source_sigma_set)
    end

    single_axion_allowed = BigFloat[]
    for axion in axion_ids
        terms = [seen[(axion, bh)].probability_allowed for bh in ensemble.bh_ids]
        push!(single_axion_allowed, _bhsr_probability_product(terms))
    end
    geometry_allowed = _bhsr_probability_product(single_axion_allowed)
    geometry_excluded = BigFloat(1) - geometry_allowed
    return (; model_id = id,
            route_identity = BHSR_APPENDIX_B_ROUTE,
            method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
            source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
            bh_ensemble_id = ensemble.manifest_id,
            observational_manifest_sha256 = ensemble.observational_manifest_sha256,
            source_relevant_axion_ids = axion_ids,
            expected_bh_ids = ensemble.bh_ids,
            received_terms = received,
            duplicate_terms = Tuple{String,String}[],
            missing_terms = Tuple{String,String}[],
            unexpected_axions = String[],
            unexpected_bh_ids = String[],
            unresolved_terms = Tuple{String,String}[],
            source_sigma_gate_status = source_sigma_gate.status,
            authority_status = :authoritative,
            status = :evaluated,
            single_axion_allowed,
            geometry_allowed,
            geometry_excluded,
            threshold_exceeded = geometry_excluded > big"0.9545")
end

function _bhsr_positive_finite_sigma(value)
    return !ismissing(value) && value isa Real && isfinite(value) && value > 0
end

"""Audit row coverage and provenance shape; caller rows cannot establish source authority."""
function bhsr_sigma_gate(rows; bh_ensemble::BHSRBHIdentityEnsemble)
    ensemble = _bhsr_validate_bh_ensemble(bh_ensemble)
    counts = Dict{String,Int}()
    row_by_name = Dict{String,Any}()
    unresolved = String[]
    invalid_rows = String[]
    for (index, row) in enumerate(rows)
        if !(row isa NamedTuple) || !hasproperty(row, :name) ||
           !(row.name isa AbstractString) || isempty(strip(row.name))
            push!(invalid_rows, "row[$index]:missing_identity")
            continue
        end
        name = strip(String(row.name))
        counts[name] = get(counts, name, 0) + 1
        row_by_name[name] = row
        name in ensemble.bh_ids || push!(invalid_rows, "$name:unexpected_identity")
    end
    duplicate_ids = sort!([name for (name, count) in counts if count > 1])
    missing_ids = [name for name in ensemble.bh_ids if !haskey(counts, name)]
    for name in ensemble.bh_ids
        count = get(counts, name, 0)
        count == 1 || continue
        row = row_by_name[name]
        fields = (:sigma_mass, :sigma_spin, :sigma_mass_source, :sigma_spin_source,
                  :sigma_mass_confidence, :sigma_spin_confidence,
                  :sigma_mass_provenance, :sigma_spin_provenance)
        if !all(field -> hasproperty(row, field), fields)
            push!(unresolved, "$name:missing_sigma_provenance_fields")
            continue
        end
        complete = _bhsr_positive_finite_sigma(row.sigma_mass) &&
            _bhsr_positive_finite_sigma(row.sigma_spin) &&
            row.sigma_mass_source == "source_defined" &&
            row.sigma_spin_source == "source_defined" &&
            row.sigma_mass_confidence == "1sigma" &&
            row.sigma_spin_confidence == "1sigma" &&
            row.sigma_mass_provenance isa AbstractString &&
            !isempty(strip(row.sigma_mass_provenance)) &&
            row.sigma_spin_provenance isa AbstractString &&
            !isempty(strip(row.sigma_spin_provenance))
        complete || push!(unresolved, name)
    end
    diagnostic_input_status = isempty(duplicate_ids) && isempty(missing_ids) &&
        isempty(invalid_rows) && isempty(unresolved) ? :complete : :unavailable
    source_authoritative = diagnostic_input_status == :complete &&
        BHSR_APPENDIX_B_SOURCE_SIGMA_STATUS == "COMPLETE_SOURCE_DEFINED_GAUSSIAN_SIGMA_SET"
    status = source_authoritative ? :complete : :unavailable
    return (; model_id = "APPENDIX-B-SIGMA-GATE",
            route_identity = ensemble.route_identity,
            method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
            source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
            bh_ensemble_id = ensemble.manifest_id,
            observational_manifest_sha256 = ensemble.observational_manifest_sha256,
            expected_bh_ids = ensemble.bh_ids,
            expected_rows = length(ensemble.bh_ids),
            received_rows = length(rows),
            duplicate_ids,
            missing_ids,
            invalid_rows,
            unresolved,
            source_manifest_status = BHSR_APPENDIX_B_SOURCE_SIGMA_STATUS,
            source_authoritative,
            diagnostic_input_status,
            status)
end

"The source audit found no complete source-defined Table-I sigma set."
function bhsr_sourcewide_likelihood_status(; model_id::AbstractString = "APPENDIX-B-SOURCEWIDE-UNAVAILABLE")
    ensemble = bhsr_bh_identity_ensemble()
    unsupported_smbh_rows = [
        "Mrk 335", "Fairall 9", "Mrk 79", "NGC 3783", "MCG-6-30-15",
        "NGC 7469", "Ark 120", "Mrk 110", "NGC 4051",
    ]
    return (; model_id = _bhsr_model_id(model_id),
            route_identity = BHSR_APPENDIX_B_ROUTE,
            likelihood_route_identity = BHSR_APPENDIX_B_ROUTE,
            contour_model_id = "REFERENCE_2021_FIG3_STELLAR",
            contour_route_identity = "REFERENCE_2021_ANALYTIC",
            contour_mass_support_solar = (BigFloat("0.1"), BigFloat("100")),
            unsupported_contour_support_bh_ids = unsupported_smbh_rows,
            method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
            source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
            bh_ensemble_id = ensemble.manifest_id,
            observational_manifest_sha256 = ensemble.observational_manifest_sha256,
            status = :unavailable,
            source_rows = length(ensemble.bh_ids),
            unresolved_censored_spin_rows = collect(BHSR_APPENDIX_B_UNRESOLVED_ROWS),
            reason = "no source-defined conversion for censored or mixed-confidence errors; the frozen 2021 Fig. 3 stellar contour has support only on 0.1-100 M_sun and cannot evaluate the nine 2018 Table-I SMBH rows outside that support")
end
