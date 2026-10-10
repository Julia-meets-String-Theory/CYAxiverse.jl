struct EndpointCount
    trial::Rational{BigInt}
    negative::Int
    pivots::Vector{IA.Interval{BigFloat}}
end

struct SpectralCluster
    first::Int
    last::Int
    population::Int
    lower::Rational{BigInt}
    upper::Rational{BigInt}
    sign::Symbol
    relative_width_met::Bool
end

struct SpectrumResult
    status::Symbol
    reason::String
    route::Symbol
    goal::Symbol
    precision::Int
    metric_pivots::Vector{IA.Interval{BigFloat}}
    bounds::Vector{Tuple{Rational{BigInt},Rational{BigInt}}}
    clusters::Vector{SpectralCluster}
    endpoint_counts::Vector{EndpointCount}
    diagonal_ratios::Vector{IA.Interval{BigFloat}}
    negative_count::Int
    positive_count::Int
    unresolved_count::Int
    resolution_met::Bool
    attempts::Vector{String}
end

struct CertificationResult
    status::Symbol
    root::RootResult
    spectrum::SpectrumResult
    attempts::Vector{String}
end

function _empty_spectrum(status::Symbol, reason::String, goal::Symbol=:clusters, bits::Int=0)
    SpectrumResult(status, reason, :none, goal, bits, IA.Interval{BigFloat}[],
        Tuple{Rational{BigInt},Rational{BigInt}}[], SpectralCluster[], EndpointCount[],
        IA.Interval{BigFloat}[], 0, 0, 0, false, String[])
end

function _relative_met(lo::Rational{BigInt}, hi::Rational{BigInt}, target::Rational{BigInt})
    lo < hi || return false
    (lo > 0 || hi < 0) || return false
    distance = hi - lo
    nearest = min(abs(lo), abs(hi))
    nearest > 0 && distance / nearest <= target
end

function _classify_bounds(bounds::Vector{Tuple{Rational{BigInt},Rational{BigInt}}})
    negative = count(b -> b[2] < 0, bounds)
    positive = count(b -> b[1] > 0, bounds)
    return negative, positive, length(bounds) - negative - positive
end

function _cluster_from_interval(first::Int, last::Int,
        lo::Rational{BigInt}, hi::Rational{BigInt}, target::Rational{BigInt})
    sign = lo > 0 ? :positive : hi < 0 ? :negative : :unresolved
    SpectralCluster(first, last, last-first+1, lo, hi, sign,
                    sign !== :unresolved && _relative_met(lo, hi, target))
end

function _diagonal_clusters(ratios::Vector{IA.Interval{BigFloat}},
                            target::Rational{BigInt})
    n = length(ratios)
    lower = sort(Rational{BigInt}[Rational{BigInt}(_lo(x)) for x in ratios])
    upper = sort(Rational{BigInt}[Rational{BigInt}(_hi(x)) for x in ratios])
    bounds = Tuple{Rational{BigInt},Rational{BigInt}}[(lower[i], upper[i]) for i in 1:n]
    clusters = SpectralCluster[]
    first = 1
    current_upper = upper[1]
    for i in 2:n
        if lower[i] <= current_upper
            current_upper = max(current_upper, upper[i])
        else
            push!(clusters, _cluster_from_interval(first, i-1, lower[first], current_upper, target))
            first = i; current_upper = upper[i]
        end
    end
    push!(clusters, _cluster_from_interval(first, n, lower[first], current_upper, target))
    # Preserve each outward ratio while making the ordering claim explicit.
    return bounds, clusters
end

function _exact_zero(x::IA.Interval{BigFloat})
    iszero(_lo(x)) && iszero(_hi(x))
end

function _is_diagonal(A::Matrix{IA.Interval{BigFloat}})
    all(_exact_zero(A[i, j]) for j in axes(A, 2) for i in axes(A, 1) if i != j)
end

function _spectral_result(status, reason, route, goal, bits, pivots, bounds,
                          clusters, counts, ratios, attempts)
    negative, positive, unresolved = _classify_bounds(bounds)
    met = !isempty(clusters) && all(c -> c.relative_width_met, clusters) &&
        (goal === :clusters || all(c -> c.population == 1, clusters))
    SpectrumResult(status, reason, route, goal, bits, pivots, bounds, clusters,
                   counts, ratios, negative, positive, unresolved, met, attempts)
end

function _metric_ldl(model::CertificationModel, bits::Int)
    K = _interval_matrix(model.K, bits)
    result = _interval_ldl(K, bits)
    result, K
end

function _midmatrix(A::Matrix{IA.Interval{BigFloat}}, bits::Int)
    BigFloat[_interval_mid(A[i, j], bits) for i in axes(A, 1), j in axes(A, 2)]
end

function _initial_radius(H, K, bits::Int)
    value = BigFloat(1; precision=bits)
    try
        hm = _midmatrix(H, bits); km = _midmatrix(K, bits)
        estimate = opnorm(hm, Inf) * opnorm(_big_inverse(km), Inf)
        isfinite(estimate) && (value = max(value, estimate))
    catch
    end
    radius = BigFloat(1; precision=bits)
    while radius < value
        radius *= 2
    end
    radius
end

mutable struct _InertiaBudget
    limit::Int
    used::Int
    attempts::Vector{String}
end

function _inertia_at(H::Matrix{IA.Interval{BigFloat}}, K::Matrix{IA.Interval{BigFloat}},
                     t::Rational{BigInt}, bits::Int, budget::_InertiaBudget)
    budget.used >= budget.limit && return nothing
    budget.used += 1
    n = size(H, 1)
    ti = _iv(t, bits)
    shifted = Matrix{IA.Interval{BigFloat}}(undef, n, n)
    for j in 1:n, i in 1:n
        shifted[i, j] = _isub(H[i, j], _imul(ti, K[i, j], bits), bits)
    end
    ldl = _interval_ldl(shifted, bits)
    push!(budget.attempts, "trial=$t status=$(ldl.ok ? "counted" : "inconclusive")")
    ldl.ok || return nothing
    return EndpointCount(t, count(==(:negative), ldl.signs), ldl.pivots)
end

function _endpoint_rational(x::BigFloat)
    Rational{BigInt}(x)
end

function _make_spectrum_cluster(ca::EndpointCount, cb::EndpointCount,
        n::Int, goal::Symbol, target::Rational{BigInt})
    first_mode = ca.negative + 1
    last_mode = cb.negative
    first_mode <= last_mode || return nothing
    lo, hi = ca.trial, cb.trial
    cluster = _cluster_from_interval(first_mode, last_mode, lo, hi, target)
    done = cluster.relative_width_met && (goal === :clusters || cluster.population == 1)
    return cluster, done
end

function _inertia_spectrum(H, K, bits, options::ProofOptions, metric_pivots; used=1)
    n = size(H, 1)
    budget = _InertiaBudget(options.max_inertia_evaluations, used, String[])
    radius = _initial_radius(H, K, bits)
    lower_count = nothing; upper_count = nothing
    radius_exact = Rational{BigInt}(radius)
    for expansion in 0:options.max_expansions
        lower_count = _inertia_at(H, K, -radius_exact, bits, budget)
        upper_count = _inertia_at(H, K, radius_exact, bits, budget)
        lower_count === nothing || upper_count === nothing ||
            (lower_count.negative == 0 && upper_count.negative == n && break)
        budget.used >= budget.limit && break
        expansion == options.max_expansions && break
        radius *= 2
        radius_exact = Rational{BigInt}(radius)
    end
    (lower_count !== nothing && upper_count !== nothing &&
     lower_count.negative == 0 && upper_count.negative == n) ||
        return _spectral_result(:INCONCLUSIVE, budget.used >= budget.limit ? "budget_exhausted" : "global_bracket_unproved",
            :inertia_counts, options.spectral_goal, bits, metric_pivots,
            Tuple{Rational{BigInt},Rational{BigInt}}[], SpectralCluster[],
            EndpointCount[], IA.Interval{BigFloat}[], budget.attempts)

    # The successful global-bracket check above proves both endpoints exist.
    # Keep that invariant in the queue's element type so downstream refinement
    # does not carry the failed-evaluation `nothing` case into certified pairs.
    lower_endpoint = lower_count::EndpointCount
    upper_endpoint = upper_count::EndpointCount
    queue = Tuple{EndpointCount,EndpointCount}[(lower_endpoint, upper_endpoint)]
    final = Tuple{EndpointCount,EndpointCount}[]
    while !isempty(queue)
        ca, cb = popfirst!(queue)
        diff = cb.negative - ca.negative
        diff == 0 && continue
        cluster, done = _make_spectrum_cluster(ca, cb, n, options.spectral_goal,
                                               options.relative_width)
        if done
            push!(final, (ca, cb)); continue
        end
        a, b = ca.trial, cb.trial
        candidates = Rational{BigInt}[(a+b)/2]
        push!(candidates, (3a+b)/4, (a+3b)/4)
        chosen = nothing
        seen = Set{Rational{BigInt}}([a,b])
        for t in candidates
            t in seen && continue
            push!(seen, t)
            count_result = _inertia_at(H, K, t, bits, budget)
            count_result === nothing || (chosen = count_result; break)
        end
        if chosen === nothing
            push!(final, (ca, cb))
            continue
        end
        chosen_endpoint = chosen::EndpointCount
        ca.negative < chosen_endpoint.negative && push!(queue, (ca, chosen_endpoint))
        chosen_endpoint.negative < cb.negative && push!(queue, (chosen_endpoint, cb))
    end
    # A budget stop leaves every queued certified frontier item intact.
    append!(final, queue)
    sort!(final; by=x -> x[1].negative)
    bounds = Tuple{Rational{BigInt},Rational{BigInt}}[]
    clusters = SpectralCluster[]
    endpoint_map = Dict{Rational{BigInt},EndpointCount}()
    for (ca, cb) in final
        item = _make_spectrum_cluster(ca, cb, n, options.spectral_goal, options.relative_width)
        item === nothing && continue
        cluster, _ = item
        push!(clusters, cluster)
        for _ in cluster.first:cluster.last
            push!(bounds, (cluster.lower, cluster.upper))
        end
        endpoint_map[ca.trial] = ca
        endpoint_map[cb.trial] = cb
    end
    endpoint_counts = EndpointCount[endpoint_map[t] for t in sort(collect(keys(endpoint_map)))]
    covered = sum(c.population for c in clusters; init=0) == n
    all_met = covered && !isempty(clusters) && all(c -> c.relative_width_met, clusters) &&
        (options.spectral_goal === :clusters || all(c -> c.population == 1, clusters))
    reason = all_met ? "goal_met" : budget.used >= budget.limit ? "budget_exhausted" :
        any(c -> c.sign === :unresolved, clusters) ? "sign_unresolved" :
        options.spectral_goal === :individuals && any(c -> c.population > 1, clusters) ?
            "individuals_unresolved" : "refinement_inconclusive"
    status = covered ? :CERTIFIED_BOUNDS : :INCONCLUSIVE
    return _spectral_result(status, reason, :inertia_counts, options.spectral_goal,
        bits, metric_pivots, bounds, clusters, endpoint_counts,
        IA.Interval{BigFloat}[], budget.attempts)
end

function _diagonal_spectrum(H, K, bits, options::ProofOptions, metric_pivots)
    n = size(H, 1)
    ratios = IA.Interval{BigFloat}[_idiv(H[i, i], K[i, i], bits) for i in 1:n]
    bounds, clusters = _diagonal_clusters(ratios, options.relative_width)
    met = all(c -> c.relative_width_met, clusters) &&
        (options.spectral_goal === :clusters || all(c -> c.population == 1, clusters))
    reason = met ? "goal_met" : any(c -> c.sign === :unresolved, clusters) ? "sign_unresolved" :
        options.spectral_goal === :individuals && any(c -> c.population > 1, clusters) ?
            "individuals_unresolved" : "refinement_inconclusive"
    result = _spectral_result(:CERTIFIED_BOUNDS, reason, :diagonal,
        options.spectral_goal, bits, metric_pivots, bounds, clusters,
        EndpointCount[], ratios, String[])
    return result
end

function certify_generalized_spectrum(model::CertificationModel, root::RootResult;
        options=ProofOptions())
    options isa ProofOptions || throw(ArgumentError("options must be ProofOptions"))
    root.precision in options.precisions || throw(ArgumentError("root precision must be in options.precisions"))
    root.status === :CERTIFIED_ROOT || return _empty_spectrum(:INCONCLUSIVE, "root is not certified", options.spectral_goal, root.precision)
    (length(root.center) <= options.max_dimension && size(model.Q, 2) <= options.max_terms) ||
        return _empty_spectrum(:UNSUPPORTED, "RESOURCE_LIMIT", options.spectral_goal, root.precision)
    _backend_supported() || return _empty_spectrum(:UNSUPPORTED, "INTERVAL_BACKEND_VERSION", options.spectral_goal, root.precision)
    bits = root.precision
    return _with_precision(bits) do
        _revalidate_root(model, root, bits) ||
            return _empty_spectrum(:INVALID_INPUT, "root witness does not match model", options.spectral_goal, bits)
        options.max_inertia_evaluations == 0 &&
            return _empty_spectrum(:INCONCLUSIVE, "budget_exhausted", options.spectral_goal, bits)
        H = root.H_box
        metric, K = _metric_ldl(model, bits)
        if !metric.ok
            # An exact nonpositive diagonal itself proves K is not SPD.
            any(model.K[i, i] <= 0 for i in 1:size(model.K, 1)) &&
                return _empty_spectrum(:INVALID_INPUT, "METRIC_NOT_SPD", options.spectral_goal, bits)
            return _empty_spectrum(:INCONCLUSIVE, "METRIC_PD_UNPROVED", options.spectral_goal, bits)
        end
        all(==(:positive), metric.signs) ||
            return _empty_spectrum(:INVALID_INPUT, "METRIC_NOT_SPD", options.spectral_goal, bits)
        length(metric.pivots) == size(H, 1) ||
            return _empty_spectrum(:INCONCLUSIVE, "METRIC_PD_UNPROVED", options.spectral_goal, bits)
        if _is_diagonal(H) && _is_diagonal(K)
            _diagonal_spectrum(H, K, bits, options, metric.pivots)
        else
            _inertia_spectrum(H, K, bits, options, metric.pivots; used=1)
        end
    end
end

function certify(model::CertificationModel, theta0; options=ProofOptions(), candidate_provenance=nothing)
    options isa ProofOptions || throw(ArgumentError("options must be ProofOptions"))
    seed = _exact_seed(theta0, size(model.Q, 1))
    attempts = String[]
    best_root = _empty_root(:INCONCLUSIVE, "no root attempt")
    best_spectrum = _empty_spectrum(:INCONCLUSIVE, "spectrum not attempted", options.spectral_goal)
    best_level = 0
    for bits in options.precisions
        for radius in options.box_radii
            onebox = ProofOptions(; precisions=(bits,), max_newton_steps=options.max_newton_steps,
                max_backtracks=options.max_backtracks, box_radii=(radius,),
                max_expansions=options.max_expansions,
                max_inertia_evaluations=options.max_inertia_evaluations,
                relative_width=options.relative_width, spectral_goal=options.spectral_goal,
                max_dimension=options.max_dimension, max_terms=options.max_terms)
            root = certify_stationary_point(model, seed; options=onebox,
                                            candidate_provenance)
            push!(attempts, "root p=$bits radius=$radius status=$(root.status)")
            if root.status === :CERTIFIED_ROOT
                spectrum = certify_generalized_spectrum(model, root; options=onebox)
                push!(attempts, "spectrum p=$bits radius=$radius status=$(spectrum.status) reason=$(spectrum.reason)")
                level = spectrum.status === :CERTIFIED_BOUNDS ? 2 : 1
                if level > best_level
                    best_level = level; best_root = root; best_spectrum = spectrum
                end
                if spectrum.status === :CERTIFIED_BOUNDS && spectrum.resolution_met
                    return CertificationResult(:CERTIFIED, root, spectrum, attempts)
                end
            elseif best_level == 0 && best_root.status !== :CERTIFIED_ROOT
                best_root = root
            end
        end
    end
    status = best_level == 2 ? :INCONCLUSIVE_SPECTRUM :
             best_level == 1 ? :CERTIFIED_ROOT : best_root.status
    CertificationResult(status, best_root, best_spectrum, attempts)
end
