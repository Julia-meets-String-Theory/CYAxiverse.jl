struct VerificationReport
    status::Symbol
    reason::String
    claims::Vector{NamedTuple}
    provenance::Symbol
    precision::Union{Nothing,Int}
end

const _SHA256_RE = r"^[0-9a-f]{64}$"
const _GIT_SHA_RE = r"^[0-9a-f]{40}$"
const _RATIONAL_RE = r"^(0|-[1-9][0-9]*|[1-9][0-9]*)/([1-9][0-9]*)$"
const _MAX_TOML_BYTES = 16*1024*1024

function _preflight_add_bytes(total::Int, additional::Int)
    additional >= 0 && additional <= _MAX_TOML_BYTES - total ||
        throw(ResourceLimitError("RESOURCE_LIMIT: TOML byte limit exceeded before serialization"))
    total + additional
end

function _decimal_digits_lower_bound(value::BigInt)
    # GMP's base-10 estimate is exact or one too large; subtracting one gives
    # a nonallocating lower bound suitable for fail-fast cumulative refusal.
    max(Base.GMP.MPZ.sizeinbase(value, 10) - 1, 1)
end

function _rational_text_bytes(value::Rational)
    n, d = numerator(value), denominator(value)
    numerator_estimate = Base.GMP.MPZ.sizeinbase(n, 10)
    denominator_estimate = Base.GMP.MPZ.sizeinbase(d, 10)
    max(numerator_estimate, denominator_estimate) > 4097 &&
        throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    max(numerator_estimate - 1, 1) + (n < 0 ? 1 : 0) + 1 +
        max(denominator_estimate - 1, 1)
end

function _rational_text_bytes(value::Integer)
    n = BigInt(value)
    estimate = Base.GMP.MPZ.sizeinbase(n, 10)
    estimate > 4097 && throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    _decimal_digits_lower_bound(n) + (n < 0 ? 1 : 0) + 2 # canonical integer spelling is n/1
end

function _preflight_rational_array(total::Int, values::AbstractArray)
    total = _preflight_add_bytes(total, 2 + max(length(values) - 1, 0)) # brackets and separators
    for value in values
        total = _preflight_add_bytes(total, _rational_text_bytes(value) + 2) # quoted value
    end
    total
end

function _preflight_interval_array(total::Int, values::AbstractArray{<:IA.Interval{BigFloat}})
    count = length(values)
    total = _preflight_add_bytes(total, 2 + 3count + max(count - 1, 0))
    for value in values
        total = _preflight_add_bytes(total, _rational_text_bytes(_rational_bound(_lo(value))) + 2)
        total = _preflight_add_bytes(total, _rational_text_bytes(_rational_bound(_hi(value))) + 2)
    end
    total
end

function _preflight_input_rational_bytes(model::CertificationModel)
    total = 0
    for values in (model.Q, model.L, model.K, model.phases)
        total = _preflight_rational_array(total, values)
    end
    model.s === nothing && throw(ResourceLimitError("RESOURCE_LIMIT: exact input interpretation unavailable"))
    _preflight_add_bytes(total, _rational_text_bytes(model.s) + 2)
end

function _preflight_certificate_rational_bytes(root::RootResult, spectrum)
    total = 0
    for values in (root.original_seed, root.translation, root.normalized_seed,
            root.center, root.preconditioner)
        total = _preflight_rational_array(total, values)
    end
    for values in (root.box, root.F_center, root.H_box, root.W)
        total = _preflight_interval_array(total, values)
    end
    total = _preflight_add_bytes(total, _rational_text_bytes(root.q_upper) + 2)
    for value in root.options.box_radii
        total = _preflight_add_bytes(total, _rational_text_bytes(value) + 2)
    end
    total = _preflight_add_bytes(total, _rational_text_bytes(root.options.relative_width) + 2)
    if root.hessian_method === :direct_ldl
        total = _preflight_interval_array(total, root.hessian_pivots)
    end
    if spectrum isa SpectrumResult
        for values in (spectrum.metric_pivots, spectrum.diagonal_ratios)
            total = _preflight_interval_array(total, values)
        end
        for (lower, upper) in spectrum.bounds
            total = _preflight_add_bytes(total, _rational_text_bytes(lower) + 2)
            total = _preflight_add_bytes(total, _rational_text_bytes(upper) + 2)
        end
        for cluster in spectrum.clusters
            total = _preflight_add_bytes(total, _rational_text_bytes(cluster.lower) + 2)
            total = _preflight_add_bytes(total, _rational_text_bytes(cluster.upper) + 2)
        end
        for endpoint in spectrum.endpoint_counts
            total = _preflight_add_bytes(total, _rational_text_bytes(endpoint.trial) + 2)
            total = _preflight_interval_array(total, endpoint.pivots)
        end
    end
    total
end

function _preflight_toml_min_bytes(value, total::Int=0)
    if value isa AbstractString
        total = _preflight_add_bytes(total, ncodeunits(value) + 2)
    elseif value isa AbstractDict
        for (key, item) in value
            total = _preflight_add_bytes(total, ncodeunits(string(key)) + 1)
            total = _preflight_toml_min_bytes(item, total)
        end
    elseif value isa AbstractVector
        if all(item -> item isa AbstractDict, value)
            for item in value
                total = _preflight_toml_min_bytes(item, total)
            end
        else
            total = _preflight_add_bytes(total, 2 + max(length(value) - 1, 0))
            for item in value
                total = _preflight_toml_min_bytes(item, total)
            end
        end
    end
    total
end

function _ratstr(x::Rational)
    n, d = numerator(x), denominator(x)
    max(ndigits(abs(n)), ndigits(d)) <= 4096 || throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
    string(n, "/", d)
end
_ratstr(x::Integer) = _ratstr(BigInt(x)//BigInt(1))

function _parse_rational(x)
    x isa AbstractString || throw(ArgumentError("rational values must be quoted strings"))
    occursin(_RATIONAL_RE, x) || throw(ArgumentError("noncanonical rational"))
    left, right = split(x, '/'; limit=2)
    numerator_digits = length(left) - (startswith(left, "-") ? 1 : 0)
    (numerator_digits <= 4096 && length(right) <= 4096) || throw(ResourceLimitError("RESOURCE_LIMIT: rational digit limit exceeded"))
    n, d = parse(BigInt, left), parse(BigInt, right)
    d > 0 || throw(ArgumentError("rational denominator must be positive"))
    gcd(abs(n), d) == 1 || throw(ArgumentError("rational must be reduced"))
    (n == 0 && d != 1) && throw(ArgumentError("zero must be 0/1"))
    n // d
end

function _is_dyadic(q::Rational)
    d = denominator(q)
    d > 0 && ispow2(d)
end

function _array_record(A::AbstractArray)
    Dict("dims"=>Int[size(A)...], "data"=>String[_ratstr(x) for x in vec(A)])
end

function _interval_array_record(A::AbstractArray{<:IA.Interval{BigFloat}})
    Dict("dims"=>Int[size(A)...], "data"=>Any[Any[_ratstr(_rational_bound(_lo(x))),
        _ratstr(_rational_bound(_hi(x)))] for x in vec(A)])
end

function _identity_interpretation(model::CertificationModel)::_CertificationInterpretation
    (basis="periodic_string", orientation="axions_by_terms",
        theta_period="1/1", phase_units="radian",
        coefficient_convention="signed_base10_log",
        numeric_semantics="exact_supplied_values",
        spectrum_convention="scaled_hessian_pencil",
        physical_normalization="not_established", s=model.s)
end

function _interpretation(model::CertificationModel)
    model.s === nothing && throw(ResourceLimitError("RESOURCE_LIMIT: exact input interpretation unavailable"))
    values = _identity_interpretation(model)
    Dict{String,String}(String(key) => (key === :s ? _ratstr(value) : value)
        for (key, value) in pairs(values))
end

function _settings_record(o::ProofOptions)
    Dict("precisions"=>collect(o.precisions), "max_newton_steps"=>o.max_newton_steps,
        "max_backtracks"=>o.max_backtracks, "box_radii"=>String[_ratstr(x) for x in o.box_radii],
        "max_expansions"=>o.max_expansions,
        "max_inertia_evaluations"=>o.max_inertia_evaluations,
        "relative_width"=>_ratstr(o.relative_width), "spectral_goal"=>String(o.spectral_goal),
        "max_dimension"=>o.max_dimension, "max_terms"=>o.max_terms)
end

function _input_record(model::CertificationModel)
    model.resource_reason === nothing || throw(ResourceLimitError("RESOURCE_LIMIT: input model unavailable"))
    n, M = model.n, model.M
    n <= 32 && M <= 2048 || throw(ResourceLimitError("RESOURCE_LIMIT: input dimensions exceed schema"))
    _preflight_input_rational_bytes(model)
    Dict("schema"=>"cyax-scientific-input-v1", "n"=>n, "M"=>M,
        "Q"=>_array_record(model.Q), "L"=>_array_record(model.L),
        "K"=>_array_record(model.K), "phi"=>_array_record(model.phases),
        "interpretation"=>_interpretation(model))
end

function _result_identity(model::CertificationModel)
    interpretation = _identity_interpretation(model)
    model.resource_reason === nothing || return CertificationIdentity(nothing, :UNAVAILABLE,
        model.resource_reason, interpretation)
    try
        input_bytes = _toml_bytes(_input_record(model))
        return CertificationIdentity(bytes2hex(sha256(input_bytes)), :AVAILABLE,
            nothing, interpretation)
    catch error
        error isa ResourceLimitError || rethrow()
        return CertificationIdentity(nothing, :UNAVAILABLE, "RESOURCE_LIMIT",
            interpretation)
    end
end

function _toml_bytes(record::Dict)
    _preflight_toml_min_bytes(record)
    _check_tree_limits(record)
    # IOBuffer truncates at maxsize instead of throwing, so retain one byte
    # beyond the contract cap and reject the resulting size before use.
    io = IOBuffer(; maxsize=_MAX_TOML_BYTES + 1)
    TOML.print(io, record; sorted=true)
    bytes = take!(io)
    length(bytes) <= _MAX_TOML_BYTES || throw(ResourceLimitError("RESOURCE_LIMIT: TOML byte limit exceeded"))
    if isempty(bytes) || last(bytes) != 0x0a
        length(bytes) < _MAX_TOML_BYTES || throw(ResourceLimitError("RESOURCE_LIMIT: TOML byte limit exceeded"))
        push!(bytes, 0x0a)
    end
    length(bytes) <= _MAX_TOML_BYTES || throw(ResourceLimitError("RESOURCE_LIMIT: TOML byte limit exceeded"))
    return bytes
end

function _normalized_record(value, label::String, allowed::Tuple)
    value isa NamedTuple && (value = Dict{Any,Any}(pairs(value)))
    value isa AbstractDict || throw(ArgumentError("$label must be a record"))
    allowed_keys = Set(String.(allowed))
    out = Dict{String,Any}()
    for (raw_key, field_value) in pairs(value)
        raw_key isa AbstractString || raw_key isa Symbol ||
            throw(ArgumentError("$label keys must be strings or symbols"))
        key = String(raw_key)
        key in allowed_keys || throw(ArgumentError("$label has unknown key $key"))
        haskey(out, key) && throw(ArgumentError("$label has duplicate string/symbol key $key"))
        out[key] = field_value
    end
    out
end

function _metadata_string(value, label::String; nonempty::Bool=false)
    value isa AbstractString || throw(ArgumentError("$label must be a string"))
    text = String(value)
    isvalid(text) || throw(ArgumentError("$label must be valid UTF-8"))
    ncodeunits(text) <= 4096 || throw(ResourceLimitError("RESOURCE_LIMIT: metadata string limit exceeded"))
    (!nonempty || !isempty(text)) || throw(ArgumentError("$label must be nonempty"))
    text
end

function _windows_absolute_path(value::String)
    startswith(value, "\\\\") || occursin(r"^[A-Za-z]:", value)
end

function _normalized_policy_host(host::AbstractString)
    name = lowercase(replace(host, "["=>"", "]"=>""))
    # DNS permits one terminal root dot in an absolute name. Remove terminal
    # dots only for policy comparisons; keep the caller's original text intact.
    while !isempty(name) && last(name) == '.'
        name = chop(name; tail=1)
    end
    name
end

function _private_or_local_host(host::AbstractString)
    name = _normalized_policy_host(host)
    name in ("localhost", "localhost.localdomain") && return true
    any(endswith(name, suffix) for suffix in (".local", ".internal", ".private")) && return true
    octets = tryparse.(Int, split(name, '.'))
    if length(octets) == 4 && all(x -> x !== nothing && 0 <= x <= 255, octets)
        a, b = octets
        return a == 0 || a == 10 || a == 127 || (a == 169 && b == 254) ||
            (a == 172 && 16 <= b <= 31) || (a == 192 && b == 168)
    end
    if occursin(':', name)
        return name in ("::1", "0:0:0:0:0:0:0:1") || startswith(name, "fc") ||
            startswith(name, "fd") || startswith(name, "fe80:")
    end
    false
end

function _contains_embedded_absolute_path(value::AbstractString)
    # Repository URLs are valid source references; inspect prose and opaque
    # descriptors separately so their slashes are not mistaken for host paths.
    without_urls = replace(String(value),
        r"(?i)\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"']+" => " ")
    occursin(r"(?i)(?:^|[^A-Za-z0-9])(?:[A-Z]:[\\/]|\\\\[^\\/\s]+[\\/])",
        without_urls) && return true
    occursin(r"(?<![A-Za-z0-9._-])~[\\/]", without_urls) && return true
    occursin(r"(?<![A-Za-z0-9.])/(?:[^\s/]+/)*[^\s/]+", without_urls)
end

function _required_url_capture(match_result::RegexMatch, index::Int, label::String)
    capture = match_result.captures[index]
    capture === nothing && throw(ArgumentError("$label contains a malformed URL"))
    String(capture)
end

function _public_safe_metadata(value, label::String)
    text = _metadata_string(value, label; nonempty=true)
    lower = lowercase(text)
    (startswith(text, "/") || text == "~" || startswith(text, "~/") || startswith(text, "~\\") ||
        _windows_absolute_path(text) || _contains_embedded_absolute_path(text) ||
        startswith(lower, "file:")) && throw(ArgumentError("$label must not expose a local path"))
    occursin(r"(?i)(?:password|pass|token|access[_-]?token|secret|client[_-]?secret|api[_-]?key|authorization|auth)\s*=", text) &&
        throw(ArgumentError("$label must not contain credentials"))
    occursin(r"(?i)(?:^|[^A-Za-z0-9])(?:session(?:[_-]?id)?|machine(?:[_-]?id)?|hostname|device(?:[_-]?id)?|workspace(?:[_-]?id)?)\s*[=:]", text) &&
        throw(ArgumentError("$label must not contain machine or session identifiers"))
    occursin(r"(?i)(?:^|[^A-Za-z0-9])(?:timestamp|date[_-]?time|created[_-]?at|started[_-]?at|finished[_-]?at|elapsed(?:[_-]?(?:time|seconds?|ms))?|duration(?:[_-]?(?:seconds?|ms))?)\s*[=:]", text) &&
        throw(ArgumentError("$label must not contain timestamps or elapsed-time details"))
    occursin(r"\b\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b", text) &&
        throw(ArgumentError("$label must not contain timestamps"))

    for url_match in eachmatch(r"(?i)\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"']+", text)
        url = String(url_match.match)
        authority = match(r"^[A-Za-z][A-Za-z0-9+.-]*://([^/?#]+)", url)
        authority === nothing && throw(ArgumentError("$label contains a malformed URL"))
        hostport = _required_url_capture(authority, 1, label)
        occursin('@', hostport) && throw(ArgumentError("$label URL must not contain credentials"))
        host_match = startswith(hostport, "[") ? match(r"^\[([^\]]+)\]", hostport) : nothing
        host = if host_match === nothing
            first(split(hostport, ':'; limit=2))
        else
            _required_url_capture(host_match, 1, label)
        end
        _private_or_local_host(host) && throw(ArgumentError("$label URL must not name a local/private host"))
        normalized_host = _normalized_policy_host(host)
        any(normalized_host == domain || endswith(normalized_host, "." * domain)
            for domain in ("chatgpt.com", "chat.openai.com", "claude.ai")) &&
            throw(ArgumentError("$label must not contain a chat/share URL"))
        path_and_query = replace(url, r"^[A-Za-z][A-Za-z0-9+.-]*://[^/?#]+"=>"")
        occursin(r"(?i)/(?:share|sharing)(?:/|\?|#|$)", path_and_query) &&
            throw(ArgumentError("$label must not contain a public-share URL"))
        occursin(r"(?i)[?&](?:share|share_id|conversation_id|thread_id)=", path_and_query) &&
            throw(ArgumentError("$label must not contain a private share reference"))
    end
    if !occursin("://", text)
        bare_host = first(split(text, ['/','?','#']; limit=2))
        _private_or_local_host(first(split(bare_host, ':'; limit=2))) &&
            throw(ArgumentError("$label must not name a local/private host"))
    end
    text
end

function _public_safe_source_ref(value)
    text = _public_safe_metadata(value, "candidate source_ref")
    lower = lowercase(text)
    (startswith(text, "/") || text == "~" || startswith(text, "~/") || startswith(text, "~\\") ||
        _windows_absolute_path(text) ||
        _contains_embedded_absolute_path(text) ||
        startswith(lower, "file:")) && throw(ArgumentError("candidate source_ref must not expose a local path"))
    occursin(r"(?i)(?:password|pass|token|access[_-]?token|secret|client[_-]?secret|api[_-]?key|authorization|auth)\s*=", text) &&
        throw(ArgumentError("candidate source_ref must not contain credentials"))
    occursin(r"(?i)(?:^|[?&#;\s])(?:session(?:[_-]?id)?|machine(?:[_-]?id)?|hostname|device(?:[_-]?id)?|workspace(?:[_-]?id)?)\s*[=:]", text) &&
        throw(ArgumentError("candidate source_ref must not contain machine or session identifiers"))
    if !occursin("://", text)
        bare_host = first(split(text, ['/','?','#']; limit=2))
        _private_or_local_host(first(split(bare_host, ':'; limit=2))) &&
            throw(ArgumentError("candidate source_ref must not name a local/private host"))
    end
    if occursin("://", text)
        authority = match(r"^[A-Za-z][A-Za-z0-9+.-]*://([^/?#]+)", text)
        authority === nothing && throw(ArgumentError("candidate source_ref URL is malformed"))
        captured_hostport = authority.captures[1]
        captured_hostport isa AbstractString || throw(ArgumentError("candidate source_ref URL is malformed"))
        hostport = String(captured_hostport)
        occursin('@', hostport) && throw(ArgumentError("candidate source_ref URL must not contain credentials"))
        host_match = startswith(hostport, "[") ? match(r"^\[([^\]]+)\]", hostport) : nothing
        host = host_match === nothing ? first(split(hostport, ':'; limit=2)) : host_match.captures[1]
        _private_or_local_host(host) && throw(ArgumentError("candidate source_ref URL must not name a local/private host"))
        host = _normalized_policy_host(host)
        if any(host == domain || endswith(host, "." * domain)
                for domain in ("chatgpt.com", "chat.openai.com", "claude.ai"))
            throw(ArgumentError("candidate source_ref must not contain a chat/share URL"))
        end
        path_and_query = replace(text, r"^[A-Za-z][A-Za-z0-9+.-]*://[^/?#]+"=>"")
        occursin(r"(?i)/(?:share|sharing)(?:/|\?|#|$)", path_and_query) &&
            throw(ArgumentError("candidate source_ref must not contain a public-share URL"))
        occursin(r"(?i)[?&](?:share|share_id|conversation_id|thread_id)=", path_and_query) &&
            throw(ArgumentError("candidate source_ref must not contain a private share reference"))
    end
    text
end

function _safe_artifact_name(value)
    name = _metadata_string(value, "artifact_name"; nonempty=true)
    normalized = replace(name, '\\'=>'/')
    (startswith(normalized, "/") || name == "~" || startswith(name, "~/") || startswith(name, "~\\") ||
        _windows_absolute_path(name) ||
        startswith(lowercase(name), "file:") || occursin(r"^[A-Za-z][A-Za-z0-9+.-]*:", name)) &&
        throw(ArgumentError("artifact_name must be a relative name"))
    any(==(".."), split(normalized, '/')) &&
        throw(ArgumentError("artifact_name cannot traverse outside its evidence root"))
    _public_safe_metadata(name, "artifact_name")
end

function _source_record(value)
    value = _normalized_record(value, "source_ref", ("repository","commit","tree"))
    repository, commit, tree = get(value,"repository",nothing), get(value,"commit",nothing), get(value,"tree",nothing)
    repository = _metadata_string(repository, "repository"; nonempty=true)
    repository = _public_safe_source_ref(repository)
    commit = _metadata_string(commit, "commit"; nonempty=true)
    tree = _metadata_string(tree, "tree"; nonempty=true)
    commit isa AbstractString && occursin(_GIT_SHA_RE, commit) || throw(ArgumentError("commit must be lower-case Git SHA-1"))
    tree isa AbstractString && occursin(_GIT_SHA_RE, tree) || throw(ArgumentError("tree must be lower-case Git SHA-1"))
    Dict("repository"=>repository, "commit"=>commit, "tree"=>tree)
end

function _environment_record(value)
    value = _normalized_record(value, "environment_ref", ("receipt_sha256","artifact_name"))
    digest, name = get(value,"receipt_sha256",nothing), get(value,"artifact_name",nothing)
    digest = _metadata_string(digest, "receipt_sha256"; nonempty=true)
    digest isa AbstractString && occursin(_SHA256_RE, digest) || throw(ArgumentError("receipt_sha256 must be lower-case SHA-256"))
    name = _safe_artifact_name(name)
    Dict("receipt_sha256"=>digest, "artifact_name"=>name)
end

function _root_record(model::CertificationModel, root::RootResult)
    n = size(model.Q, 1)
    pivots = root.hessian_method === :direct_ldl ? root.hessian_pivots : IA.Interval{BigFloat}[]
    Dict("status"=>"certified_root", "lift"=>"unwrapped_centered_seed",
        "original_seed"=>_array_record(root.original_seed),
        "translation"=>_array_record(root.translation),
        "normalized_seed"=>_array_record(root.normalized_seed),
        "center"=>_array_record(root.center), "box"=>_interval_array_record(root.box),
        "preconditioner"=>_array_record(root.preconditioner),
        "F_center"=>_interval_array_record(root.F_center),
        "H_box"=>_interval_array_record(root.H_box), "W"=>_interval_array_record(root.W),
        "q_upper"=>_ratstr(root.q_upper))
end

function _hessian_record(root::RootResult, spectrum=nothing)
    n = length(root.original_seed)
    direct = root.hessian_method === :direct_ldl
    if root.hessian_method === :spectral_counts
        spectrum isa SpectrumResult && spectrum.status === :CERTIFIED_BOUNDS ||
            throw(ArgumentError("spectral Hessian counts require an accompanying certified spectrum"))
        negative, positive, unresolved = spectrum.negative_count, spectrum.positive_count, spectrum.unresolved_count
        _count_classification(negative, positive, unresolved, n) === root.hessian_classification ||
            throw(ArgumentError("spectral Hessian classification differs from spectrum sign counts"))
        isempty(root.hessian_pivots) || throw(ArgumentError("spectral Hessian evidence has no direct pivots"))
    else
        negative = direct ? count(x -> _hi(x) < 0, root.hessian_pivots) : 0
        positive = direct ? count(x -> _lo(x) > 0, root.hessian_pivots) : 0
        unresolved = n - negative - positive
    end
    direct && unresolved != 0 && throw(ArgumentError("direct Hessian pivots must have strict signs"))
    Dict("classification"=>String(root.hessian_classification),
        "method"=>String(root.hessian_method), "negative_count"=>negative,
        "positive_count"=>positive, "unresolved_count"=>unresolved,
        "pivots"=>_interval_array_record(root.hessian_method === :direct_ldl ?
            root.hessian_pivots : IA.Interval{BigFloat}[]))
end

function _spectrum_record(s::SpectrumResult)
    s.status === :CERTIFIED_BOUNDS || throw(ArgumentError("spectrum does not have complete certified coverage"))
    n = length(s.bounds)
    length(s.clusters) > 0 && sum(c.population for c in s.clusters) == n ||
        throw(ArgumentError("spectrum does not cover all modes"))
    cluster_records = [Dict("first"=>c.first, "last"=>c.last, "population"=>c.population,
        "lower"=>_ratstr(c.lower), "upper"=>_ratstr(c.upper), "sign"=>String(c.sign),
        "relative_width_met"=>c.relative_width_met) for c in s.clusters]
    endpoint_records = [Dict("trial"=>_ratstr(c.trial), "negative"=>c.negative,
        "pivots"=>_interval_array_record(c.pivots)) for c in s.endpoint_counts]
    Dict("status"=>"certified_bounds", "route"=>String(s.route), "goal"=>String(s.goal),
        "metric_pivots"=>_interval_array_record(s.metric_pivots),
        "bounds"=>Dict("dims"=>[n], "data"=>Any[Any[_ratstr(a),_ratstr(b)] for (a,b) in s.bounds]),
        "clusters"=>cluster_records, "negative_count"=>s.negative_count,
        "positive_count"=>s.positive_count, "unresolved_count"=>s.unresolved_count,
        "resolution_met"=>s.resolution_met,
        "reason"=>s.reason,
        "endpoint_counts"=>endpoint_records,
        "diagonal_ratios"=>_interval_array_record(s.diagonal_ratios))
end

function _claim_contains(claim::Tuple{Rational{BigInt},Rational{BigInt}},
                         actual::IA.Interval{BigFloat})
    claim[1] <= _rational_bound(_lo(actual)) && _rational_bound(_hi(actual)) <= claim[2]
end

function _check_root_claims(model::CertificationModel, root::RootResult,
                            fclaims, hclaims, wclaims, qclaim, pclaims, bits::Int)
    _revalidate_root(model, root, bits) || return false
    all(_claim_contains(fclaims[i], root.F_center[i]) for i in eachindex(fclaims)) || return false
    all(_claim_contains(hclaims[i], root.H_box[i]) for i in eachindex(hclaims)) || return false
    all(_claim_contains(wclaims[i], root.W[i]) for i in eachindex(wclaims)) || return false
    root.q_upper <= qclaim || return false
    if root.hessian_method === :direct_ldl
        length(pclaims) == length(root.hessian_pivots) || return false
        all(_claim_contains(pclaims[i], root.hessian_pivots[i]) for i in eachindex(pclaims)) || return false
    end
    true
end

function _interval_claims(value, dims::Vector{Int})
    _require_keys(value, ("dims","data"), "interval array")
    _validate_claim_dims(value["dims"], dims, "interval array")
    data = value["data"]
    data isa Vector || throw(ArgumentError("interval array data must be an array"))
    length(data) == prod(dims; init=1) || throw(ArgumentError("interval array length mismatch"))
    out = Tuple{Rational{BigInt},Rational{BigInt}}[]
    for pair in data
        pair isa Vector && length(pair) == 2 || throw(ArgumentError("interval datum must have two endpoints"))
        lo, hi = _parse_rational(pair[1]), _parse_rational(pair[2])
        lo <= hi || throw(ArgumentError("interval endpoints are reversed"))
        push!(out, (lo, hi))
    end
    reshape(out, dims...)
end

function _exact_claims(value, dims::Vector{Int})
    _require_keys(value, ("dims","data"), "exact array")
    _validate_claim_dims(value["dims"], dims, "exact array")
    data = value["data"]
    data isa Vector || throw(ArgumentError("exact array data must be an array"))
    length(data) == prod(dims; init=1) || throw(ArgumentError("exact array length mismatch"))
    out = Rational{BigInt}[_parse_rational(x) for x in data]
    reshape(out, dims...)
end

function _validate_claim_dims(observed, expected::Vector{Int}, label::String)
    observed isa Vector || throw(ArgumentError("$label dimensions must be an array"))
    length(observed) == length(expected) || throw(ArgumentError("$label dimensions have the wrong rank"))
    dims = Int[]
    for dimension in observed
        dimension isa Integer && !(dimension isa Bool) ||
            throw(ArgumentError("$label dimensions must contain strict integers"))
        0 <= dimension <= typemax(Int) || throw(ArgumentError("$label dimensions are outside the supported range"))
        push!(dims, Int(dimension))
    end
    dims == expected || throw(ArgumentError("$label dimensions do not match"))
    nothing
end

function _require_keys(value, expected, label)
    value isa AbstractDict || throw(ArgumentError("$label must be a table"))
    Set(String.(keys(value))) == Set(expected) || throw(ArgumentError("$label has missing or unknown keys"))
    nothing
end

const _RATIONAL_VALUE_PATHS = Set((
    ("Q","data"), ("L","data"), ("K","data"), ("phi","data"),
    ("interpretation","s"), ("settings","box_radii"), ("settings","relative_width"),
    ("root","original_seed","data"), ("root","translation","data"),
    ("root","normalized_seed","data"), ("root","center","data"),
    ("root","box","data"), ("root","preconditioner","data"),
    ("root","F_center","data"), ("root","H_box","data"),
    ("root","W","data"), ("root","q_upper"), ("hessian","pivots","data"),
    ("spectrum","metric_pivots","data"), ("spectrum","bounds","data"),
    ("spectrum","diagonal_ratios","data"),
    ("spectrum","clusters","lower"), ("spectrum","clusters","upper"),
    ("spectrum","endpoint_counts","trial"), ("spectrum","endpoint_counts","pivots","data")))

function _check_tree_limits(value, depth=0, path=())
    depth <= 12 || throw(ResourceLimitError("RESOURCE_LIMIT: nesting depth limit exceeded"))
    if value isa AbstractString
        isvalid(value) || throw(ArgumentError("serialized strings must be valid UTF-8"))
        if path in _RATIONAL_VALUE_PATHS && occursin(_RATIONAL_RE, value)
            parts = split(value, '/'; limit=2)
            numerator_digits = length(parts[1]) - (startswith(parts[1], "-") ? 1 : 0)
            (numerator_digits <= 4096 && length(parts[2]) <= 4096) ||
                throw(ResourceLimitError("RESOURCE_LIMIT: rational component limit exceeded"))
        else
            ncodeunits(value) <= 4096 || throw(ResourceLimitError("RESOURCE_LIMIT: metadata string limit exceeded"))
        end
    elseif value isa AbstractDict
        for (k,v) in value
            key = string(k)
            _check_tree_limits(key, depth+1, (path..., "<key>"))
            _check_tree_limits(v, depth+1, (path..., key))
        end
    elseif value isa AbstractVector
        for x in value; _check_tree_limits(x, depth+1, path); end
    end
end

function _read_toml_bytes(path::AbstractString)
    isfile(path) && !islink(path) || throw(ArgumentError("evidence path must be a regular file"))
    bytes = open(path, "r") do io
        read(io, 16*1024*1024 + 1)
    end
    length(bytes) <= 16*1024*1024 ||
        throw(ResourceLimitError("RESOURCE_LIMIT: TOML byte limit exceeded"))
    bytes
end

function _toml_key_depth(bytes, first::Int, last::Int)
    dots = 0
    quote_byte = 0x00
    escaped = false
    for index in first:last
        byte = bytes[index]
        if quote_byte != 0x00
            if quote_byte == 0x22 && escaped
                escaped = false
            elseif quote_byte == 0x22 && byte == 0x5c
                escaped = true
            elseif byte == quote_byte
                quote_byte = 0x00
            end
        elseif byte == 0x22 || byte == 0x27
            quote_byte = byte
        elseif byte == 0x2e
            dots += 1
        end
    end
    dots + 1
end

function _toml_key_component(bytes, first::Int, last::Int)
    while first <= last && (bytes[first] == 0x20 || bytes[first] == 0x09)
        first += 1
    end
    while last >= first && (bytes[last] == 0x20 || bytes[last] == 0x09 || bytes[last] == 0x0d)
        last -= 1
    end
    first <= last || return ""
    if bytes[first] == 0x27 && bytes[last] == 0x27 && last > first
        return String(copy(bytes[first+1:last-1]))
    elseif bytes[first] != 0x22 || bytes[last] != 0x22 || last <= first
        return String(copy(bytes[first:last]))
    end

    decoded = IOBuffer()
    index = first + 1
    stop = last - 1
    while index <= stop
        byte = bytes[index]
        if byte != 0x5c
            write(decoded, byte)
            index += 1
            continue
        end
        index += 1
        index <= stop || return String(copy(bytes[first:last]))
        escaped = bytes[index]
        if escaped == 0x62
            write(decoded, 0x08)
            index += 1
        elseif escaped == 0x74
            write(decoded, 0x09)
            index += 1
        elseif escaped == 0x6e
            write(decoded, 0x0a)
            index += 1
        elseif escaped == 0x66
            write(decoded, 0x0c)
            index += 1
        elseif escaped == 0x72
            write(decoded, 0x0d)
            index += 1
        elseif escaped == 0x22 || escaped == 0x5c
            write(decoded, escaped)
            index += 1
        elseif escaped == 0x75 || escaped == 0x55
            digits = escaped == 0x75 ? 4 : 8
            index += 1
            index + digits - 1 <= stop || return String(copy(bytes[first:last]))
            codepoint = tryparse(UInt32, String(copy(bytes[index:index+digits-1])); base=16)
            codepoint === nothing && return String(copy(bytes[first:last]))
            isvalid(Char, codepoint) || return String(copy(bytes[first:last]))
            print(decoded, Char(codepoint))
            index += digits
        else
            # Invalid escapes are left for TOML.parse to classify as malformed.
            return String(copy(bytes[first:last]))
        end
    end
    String(take!(decoded))
end

function _toml_header_info(bytes, first::Int, last::Int)
    opening = bytes[first] == 0x5b && first < last && bytes[first+1] == 0x5b ? 2 : 1
    index = first + opening
    quote_byte = 0x00
    escaped = false
    component_start = index
    path = String[]
    too_deep = false
    close_at = 0
    while index <= last
        byte = bytes[index]
        if quote_byte != 0x00
            if quote_byte == 0x22 && escaped
                escaped = false
            elseif quote_byte == 0x22 && byte == 0x5c
                escaped = true
            elseif byte == quote_byte
                quote_byte = 0x00
            end
        elseif byte == 0x22 || byte == 0x27
            quote_byte = byte
        elseif byte == 0x2e
            if length(path) < 13
                push!(path, _toml_key_component(bytes, component_start, index - 1))
            else
                too_deep = true
            end
            component_start = index + 1
        elseif byte == 0x23
            break
        elseif opening == 1 && byte == 0x5d
            close_at = index
            break
        elseif opening == 2 && index < last && byte == 0x5d && bytes[index+1] == 0x5d
            close_at = index + 1
            break
        end
        index += 1
    end
    close_at == 0 && return nothing
    for index in close_at+1:last
        (bytes[index] == 0x20 || bytes[index] == 0x09 || bytes[index] == 0x0d) && continue
        bytes[index] == 0x23 && break
        return nothing
    end
    if length(path) < 13
        push!(path, _toml_key_component(bytes, component_start, close_at - opening))
    else
        too_deep = true
    end
    all(component -> !isempty(component), path) || return nothing
    (path=path, is_array_table=opening == 2, too_deep=too_deep || length(path) > 12)
end

"""Reject excessive TOML container depth before calling TOML.parse."""
function _preflight_toml_depth(text::String)
    bytes = codeunits(text)
    isempty(bytes) && return nothing
    table_depth = 0
    array_table_paths = Set{Tuple}()
    value_base = 0
    stack = UInt8[]
    inline_key_start = 0
    in_value = false
    quote_state = :none
    escaped = false
    overflow = false
    malformed = false
    line_start = 1
    index = 1
    while index <= length(bytes)
        byte = bytes[index]
        if quote_state === :multibasic
            if byte == 0x5c
                index += 2
                continue
            elseif byte == 0x22 && index + 2 <= length(bytes) &&
                    bytes[index+1] == 0x22 && bytes[index+2] == 0x22
                run_end = index + 2
                while run_end < length(bytes) && bytes[run_end+1] == 0x22
                    run_end += 1
                end
                quote_state = :none
                escaped = false
                index = run_end + 1
                continue
            end
        elseif quote_state === :multiliteral
            if byte == 0x27 && index + 2 <= length(bytes) &&
                    bytes[index+1] == 0x27 && bytes[index+2] == 0x27
                run_end = index + 2
                while run_end < length(bytes) && bytes[run_end+1] == 0x27
                    run_end += 1
                end
                quote_state = :none
                index = run_end + 1
                continue
            end
        elseif quote_state === :basic
            if escaped
                escaped = false
            elseif byte == 0x5c
                escaped = true
            elseif byte == 0x22
                quote_state = :none
            end
        elseif quote_state === :literal
            byte == 0x27 && (quote_state = :none)
        elseif byte == 0x23
            while index <= length(bytes) && bytes[index] != 0x0a
                index += 1
            end
            continue
        elseif byte == 0x22
            if index + 2 <= length(bytes) && bytes[index+1] == 0x22 && bytes[index+2] == 0x22
                quote_state = :multibasic
                index += 3
                continue
            end
            quote_state = :basic
        elseif byte == 0x27
            if index + 2 <= length(bytes) && bytes[index+1] == 0x27 && bytes[index+2] == 0x27
                quote_state = :multiliteral
                index += 3
                continue
            end
            quote_state = :literal
        elseif byte == 0x0a
            isempty(stack) && (in_value = false)
            line_start = index + 1
        elseif !in_value && isempty(stack) && (byte == 0x20 || byte == 0x09 || index == line_start)
            probe = index
            while probe <= length(bytes) && (bytes[probe] == 0x20 || bytes[probe] == 0x09)
                probe += 1
            end
            if probe == index && byte != 0x5b
                index += 1
                continue
            elseif probe <= length(bytes) && bytes[probe] == 0x5b
                line_end = probe
                while line_end <= length(bytes) && bytes[line_end] != 0x0a
                    line_end += 1
                end
                header = _toml_header_info(bytes, probe, line_end - 1)
                if header === nothing
                    malformed = true
                else
                    path = header.path
                    ancestors = 0
                    if !header.too_deep
                        for prefix_length in 1:(length(path)-1)
                            ancestors += Tuple(path[1:prefix_length]) in array_table_paths
                        end
                    end
                    depth = header.too_deep ? 13 :
                        length(path) + ancestors + (header.is_array_table ? 1 : 0)
                    table_depth = depth
                    overflow |= header.too_deep || depth > 12
                    if header.is_array_table && !header.too_deep
                        push!(array_table_paths, Tuple(path))
                    end
                end
                index = line_end + 1
                line_start = index
                continue
            end
        end

        if quote_state === :none
            if !in_value && byte == 0x3d
                value_base = table_depth + _toml_key_depth(bytes, line_start, index - 1)
                # Count the assigned value after all dotted-key table levels;
                # otherwise a 12-component key reaches TOML.parse at the bound.
                overflow |= value_base >= 12
                in_value = true
                inline_key_start = index + 1
            elseif in_value && (byte == 0x5b || byte == 0x7b)
                push!(stack, byte)
                current_depth = value_base + length(stack) - 1
                overflow |= current_depth > 12
                byte == 0x7b && (inline_key_start = index + 1)
            elseif in_value && (byte == 0x5d || byte == 0x7d)
                expected = byte == 0x5d ? 0x5b : 0x7b
                if isempty(stack) || last(stack) != expected
                    malformed = true
                else
                    pop!(stack)
                    inline_key_start = index + 1
                end
            elseif in_value && !isempty(stack) && last(stack) == 0x7b && byte == 0x3d
                key_depth = _toml_key_depth(bytes, inline_key_start, index - 1)
                current_depth = value_base + length(stack) - 1 + key_depth
                overflow |= current_depth > 12
            elseif in_value && !isempty(stack) && last(stack) == 0x7b && byte == 0x2c
                inline_key_start = index + 1
            end
        end
        index += 1
    end
    # Unclosed strings/containers belong to TOML's malformed-input path.
    balanced = !malformed && quote_state === :none && isempty(stack)
    balanced && overflow && throw(ResourceLimitError("RESOURCE_LIMIT: nesting depth limit exceeded"))
    nothing
end

function _parse_toml(bytes::Vector{UInt8})
    text = String(copy(bytes))
    isvalid(text) || throw(ArgumentError("input is not valid UTF-8"))
    _preflight_toml_depth(text)
    record = try
        TOML.parse(text)
    catch e
        throw(ArgumentError("invalid TOML: $(sprint(showerror,e))"))
    end
    _check_tree_limits(record)
    return record
end

function _read_toml(path::AbstractString)
    bytes = _read_toml_bytes(path)
    return bytes, _parse_toml(bytes)
end

function _parse_input(record)
    _require_keys(record, ("schema","n","M","Q","L","K","phi","interpretation"), "input")
    record["schema"] == "cyax-scientific-input-v1" || throw(ArgumentError("unknown input schema"))
    n, M = _strict_int(record["n"], "n"), _strict_int(record["M"], "M")
    n >= 1 && M >= 1 || throw(ArgumentError("input dimensions must be positive"))
    n <= 32 && M <= 2048 || throw(ResourceLimitError("RESOURCE_LIMIT: input dimensions exceed schema"))
    q = _exact_claims(record["Q"], [n,M]); l = _exact_claims(record["L"], [2,M])
    k = _exact_claims(record["K"], [n,n]); phi = _exact_claims(record["phi"], [M])
    all(denominator(x)==1 for x in q) || throw(ArgumentError("Q entries must be integers"))
    for j in 1:n, i in 1:n
        k[i,j] == k[j,i] || throw(ArgumentError("K is not symmetric"))
    end
    interpretation = record["interpretation"]
    _require_keys(interpretation, ("basis","orientation","theta_period","phase_units",
        "coefficient_convention","numeric_semantics","spectrum_convention",
        "physical_normalization","s"), "interpretation")
    expected = ("periodic_string","axions_by_terms","1/1","radian",
        "signed_base10_log","exact_supplied_values","scaled_hessian_pencil","not_established")
    actual = (interpretation["basis"], interpretation["orientation"], interpretation["theta_period"],
        interpretation["phase_units"], interpretation["coefficient_convention"],
        interpretation["numeric_semantics"], interpretation["spectrum_convention"],
        interpretation["physical_normalization"])
    actual == expected || throw(ArgumentError("unsupported input interpretation"))
    s = _parse_rational(interpretation["s"])
    maximum(view(l,2,:)) == s || throw(ArgumentError("interpretation.s does not match L"))
    model = CertificationModel(q,l,k,vec(phi),s)
    return model, interpretation
end

function _strict_int(x, label; lower=typemin(Int), upper=typemax(Int))
    x isa Integer && !(x isa Bool) || throw(ArgumentError("$label must be an integer"))
    lower <= x <= upper || throw(ArgumentError("$label outside supported range"))
    Int(x)
end

function _parse_settings(record)
    _require_keys(record, ("precisions","max_newton_steps","max_backtracks","box_radii",
        "max_expansions","max_inertia_evaluations","relative_width","spectral_goal",
        "max_dimension","max_terms"), "settings")
    ps = record["precisions"]
    rs = record["box_radii"]
    ps isa Vector || throw(ArgumentError("precisions must be an array"))
    rs isa Vector || throw(ArgumentError("box_radii must be an array"))
    ProofOptions(precisions=ps,
        max_newton_steps=_strict_int(record["max_newton_steps"], "max_newton_steps"),
        max_backtracks=_strict_int(record["max_backtracks"], "max_backtracks"),
        box_radii=[_parse_rational(x) for x in rs],
        max_expansions=_strict_int(record["max_expansions"], "max_expansions"),
        max_inertia_evaluations=_strict_int(record["max_inertia_evaluations"], "max_inertia_evaluations"),
        relative_width=_parse_rational(record["relative_width"]),
        spectral_goal=Symbol(record["spectral_goal"]),
        max_dimension=_strict_int(record["max_dimension"], "max_dimension"),
        max_terms=_strict_int(record["max_terms"], "max_terms"))
end

function _read_source_ref(record)
    _require_keys(record, ("repository","commit","tree"), "source_ref")
    _source_record(record)
end

function _read_environment_ref(record)
    _require_keys(record, ("receipt_sha256","artifact_name"), "environment_ref")
    _environment_record(record)
end

function _claim_vector(A, dims, bits)
    [ _iv(pair[1], pair[2], bits) for pair in vec(A) ]
end

function _claim_matrix(A, dims, bits)
    out = Matrix{IA.Interval{BigFloat}}(undef, dims)
    claims = vec(A)
    for j in 1:dims[2], i in 1:dims[1]
        pair = claims[(j-1)*dims[1]+i]
        out[i,j] = _iv(pair[1],pair[2],bits)
    end
    out
end

function _valid_ref_hash(value)
    value isa AbstractString && occursin(_SHA256_RE, value)
end

function _verify_spectrum_claim(model, root, spectrum, bits, options; schema_only::Bool=false)
    n = size(model.Q, 1)
    _require_keys(spectrum, ("status","route","goal","metric_pivots","bounds","clusters",
        "negative_count","positive_count","unresolved_count","resolution_met","reason",
        "endpoint_counts","diagonal_ratios"), "spectrum")
    spectrum["status"] == "certified_bounds" || throw(ArgumentError("spectrum is not certified_bounds"))
    route = spectrum["route"]; route in ("diagonal","inertia_counts") || throw(ArgumentError("unknown spectrum route"))
    goal = spectrum["goal"]; goal == String(options.spectral_goal) || throw(ArgumentError("spectrum goal differs from settings"))
    mp = _interval_claims(spectrum["metric_pivots"], [n])
    all(pair -> pair[1] > 0, vec(mp)) || throw(ArgumentError("metric pivots must be strictly positive"))
    bclaims = _interval_claims(spectrum["bounds"], [n])
    clusters = spectrum["clusters"]
    clusters isa Vector && !isempty(clusters) || throw(ArgumentError("clusters must be a nonempty array"))
    parsed_clusters = SpectralCluster[]
    next = 1
    for item in clusters
        _require_keys(item, ("first","last","population","lower","upper","sign","relative_width_met"), "cluster")
        first = _strict_int(item["first"], "cluster.first", lower=1, upper=n)
        last = _strict_int(item["last"], "cluster.last", lower=first, upper=n)
        pop = _strict_int(item["population"], "cluster.population", lower=1, upper=n)
        pop == last-first+1 && first == next || throw(ArgumentError("clusters do not form exact ordered coverage"))
        lo, hi = _parse_rational(item["lower"]), _parse_rational(item["upper"])
        lo < hi || throw(ArgumentError("cluster endpoints must increase"))
        sign_value = item["sign"]
        sign_value isa AbstractString || throw(ArgumentError("cluster sign must be a string"))
        sign = Symbol(sign_value); sign in (:negative,:positive,:unresolved) || throw(ArgumentError("unknown cluster sign"))
        item["relative_width_met"] isa Bool || throw(ArgumentError("relative_width_met must be boolean"))
        c = _cluster_from_interval(first,last,lo,hi,options.relative_width)
        c.sign === sign && c.relative_width_met == item["relative_width_met"] ||
            throw(ArgumentError("cluster flags do not follow exact endpoints"))
        push!(parsed_clusters,c); next=last+1
        for i in first:last
            if route == "inertia_counts"
                bclaims[i] == (lo,hi) || throw(ArgumentError("mode bound differs from cluster"))
            else
                lo <= bclaims[i][1] <= bclaims[i][2] <= hi ||
                    throw(ArgumentError("diagonal mode bound lies outside its overlap cluster"))
            end
        end
    end
    next == n+1 || throw(ArgumentError("clusters do not cover all modes"))
    neg, pos, unresolved = _classify_bounds(vec(bclaims))
    _strict_int(spectrum["negative_count"], "negative_count", lower=0, upper=n) == neg || throw(ArgumentError("negative count mismatch"))
    _strict_int(spectrum["positive_count"], "positive_count", lower=0, upper=n) == pos || throw(ArgumentError("positive count mismatch"))
    _strict_int(spectrum["unresolved_count"], "unresolved_count", lower=0, upper=n) == unresolved || throw(ArgumentError("unresolved count mismatch"))
    spectrum["resolution_met"] isa Bool || throw(ArgumentError("resolution_met must be boolean"))
    resolution = all(c -> c.relative_width_met, parsed_clusters) &&
        (options.spectral_goal === :clusters || all(c -> c.population==1, parsed_clusters))
    spectrum["resolution_met"] == resolution || throw(ArgumentError("resolution flag mismatch"))
    reason = spectrum["reason"]
    reason in ("goal_met","budget_exhausted","refinement_inconclusive","sign_unresolved","individuals_unresolved") ||
        throw(ArgumentError("unknown spectrum reason"))
    resolution == (reason == "goal_met") || throw(ArgumentError("goal_met reason/flag mismatch"))
    ec = spectrum["endpoint_counts"]
    ec isa Vector || throw(ArgumentError("endpoint_counts must be an array"))
    dc = _interval_claims(spectrum["diagonal_ratios"], route == "diagonal" ? [n] : [0])
    if route == "diagonal"
        isempty(ec) || throw(ArgumentError("diagonal route has no endpoint counts"))
        expected_bounds, expected_clusters = _diagonal_clusters_claim(vec(dc), options.relative_width)
        expected_bounds == vec(bclaims) && _same_clusters(expected_clusters, parsed_clusters) ||
            throw(ArgumentError("diagonal bounds and clusters do not match submitted ratios"))
        schema_only && return true
        options.max_inertia_evaluations >= 1 || return false
        ratios = IA.Interval{BigFloat}[_iv(x[1],x[2],bits) for x in vec(dc)]
        H = root.H_box; K = _interval_matrix(model.K,bits)
        (_is_diagonal(H) && _is_diagonal(K)) || return false
        metric, _ = _metric_ldl(model,bits)
        metric.ok && all(==(:positive),metric.signs) || return false
        all(_claim_contains((dc[i][1],dc[i][2]), _idiv(H[i,i],K[i,i],bits)) for i in 1:n) || return false
        expected_bounds, expected_clusters = _diagonal_clusters_claim(vec(dc), options.relative_width)
        expected_bounds == vec(bclaims) && _same_clusters(expected_clusters, parsed_clusters) || return false
        all(_claim_contains((mp[i][1],mp[i][2]),metric.pivots[i]) for i in 1:n) || return false
        return true
    end
    length(ec) <= 2n || throw(ArgumentError("too many endpoint witnesses"))
    isempty(vec(dc)) || throw(ArgumentError("inertia route has no diagonal ratios"))
    endpoint_claims = Dict{Rational{BigInt},Tuple{Int,Vector{Tuple{Rational{BigInt},Rational{BigInt}}}}}()
    submitted_endpoints = Rational{BigInt}[]
    for item in ec
        _require_keys(item, ("trial","negative","pivots"), "endpoint_count")
        t = _parse_rational(item["trial"]); _is_dyadic(t) || throw(ArgumentError("endpoint trial must be dyadic"))
        negative = _strict_int(item["negative"], "endpoint negative", lower=0, upper=n)
        piv = _interval_claims(item["pivots"], [n])
        haskey(endpoint_claims,t) && throw(ArgumentError("duplicate endpoint trial"))
        all(pair -> pair[1] > 0 || pair[2] < 0, vec(piv)) ||
            throw(ArgumentError("endpoint pivots must exclude zero"))
        push!(submitted_endpoints,t)
        endpoint_claims[t] = (negative, vec(piv))
    end
    expected_endpoints = sort(unique(vcat([c.lower for c in parsed_clusters],[c.upper for c in parsed_clusters])))
    submitted_endpoints == expected_endpoints ||
        throw(ArgumentError("endpoint trials must equal sorted unique final cluster endpoints in canonical order"))
    length(expected_endpoints) <= 2n || throw(ArgumentError("endpoint set exceeds 2n"))
    issorted([endpoint_claims[t][1] for t in expected_endpoints]) ||
        throw(ArgumentError("endpoint counts are not nondecreasing"))
    schema_only && return true
    options.max_inertia_evaluations >= 1+length(expected_endpoints) || return false

    # Complete bounded endpoint structure before any numerical factorization.
    metric, K = _metric_ldl(model,bits)
    metric.ok && all(==(:positive),metric.signs) || return false
    all(_claim_contains((mp[i][1],mp[i][2]),metric.pivots[i]) for i in 1:n) || return false
    reproduced = Dict{Rational{BigInt},Int}()
    for t in expected_endpoints
        neg, pivclaims = endpoint_claims[t]
        shifted = Matrix{IA.Interval{BigFloat}}(undef,n,n); ti=_iv(t,bits)
        for j in 1:n, i in 1:n
            shifted[i,j]=_isub(root.H_box[i,j],_imul(ti,K[i,j],bits),bits)
        end
        ldl=_interval_ldl(shifted,bits)
        ldl.ok && length(ldl.pivots)==n || return false
        count(==(:negative),ldl.signs)==neg || return false
        all(_claim_contains(pivclaims[i],ldl.pivots[i]) for i in 1:n) || return false
        reproduced[t]=neg
    end
    for c in parsed_clusters
        get(reproduced,c.lower,-999)==c.first-1 || return false
        get(reproduced,c.upper,-999)==c.last || return false
    end
    return true
end

function _diagonal_clusters_claim(ratios::Vector{Tuple{Rational{BigInt},Rational{BigInt}}}, target)
    n=length(ratios)
    lower=sort(first.(ratios)); upper=sort(last.(ratios))
    bounds=Tuple{Rational{BigInt},Rational{BigInt}}[(lower[i],upper[i]) for i in 1:n]
    clusters=SpectralCluster[]; firstidx=1; current=upper[1]
    for i in 2:n
        if lower[i]<=current
            current=max(current,upper[i])
        else
            push!(clusters,_cluster_from_interval(firstidx,i-1,lower[firstidx],current,target))
            firstidx=i;current=upper[i]
        end
    end
    push!(clusters,_cluster_from_interval(firstidx,n,lower[firstidx],current,target))
    bounds,clusters
end

function _same_clusters(a::Vector{SpectralCluster}, b::Vector{SpectralCluster})
    length(a)==length(b) || return false
    all((x.first,x.last,x.population,x.lower,x.upper,x.sign,x.relative_width_met) ==
        (y.first,y.last,y.population,y.lower,y.upper,y.sign,y.relative_width_met)
        for (x,y) in zip(a,b))
end

function _candidate_provenance_record(value)
    value === nothing && return Dict("status"=>"unknown")
    record = _normalized_record(value, "candidate_provenance", ("status","method","source_ref"))
    status = get(record, "status", nothing)
    shorthand = status === nothing
    if shorthand
        Set(keys(record)) == Set(("method","source_ref")) ||
            throw(ArgumentError("candidate provenance shorthand requires method and source_ref only"))
        status = "supplied"
    end
    if status == "unknown"
        Set(keys(record)) == Set(("status",)) ||
            throw(ArgumentError("unknown candidate provenance cannot include method or source_ref"))
        return Dict("status"=>"unknown")
    elseif status == "supplied"
        expected_keys = shorthand ? Set(("method","source_ref")) : Set(("status","method","source_ref"))
        Set(keys(record)) == expected_keys ||
            throw(ArgumentError("supplied candidate provenance requires method and source_ref"))
        method = _public_safe_metadata(record["method"], "candidate method")
        source = _public_safe_source_ref(record["source_ref"])
        return Dict("status"=>"supplied","method"=>method,"source_ref"=>source)
    end
    throw(ArgumentError("unknown candidate provenance status"))
end

function _read_candidate_provenance(value)
    value isa AbstractDict || throw(ArgumentError("candidate_provenance must be a table"))
    status = get(value, "status", nothing)
    if status == "unknown"
        _require_keys(value, ("status",), "candidate_provenance")
    elseif status == "supplied"
        _require_keys(value, ("status","method","source_ref"), "candidate_provenance")
    else
        throw(ArgumentError("candidate_provenance requires an explicit valid status"))
    end
    _candidate_provenance_record(value)
end

function _read_certificate(record, model, input_digest)
    _require_keys(record,("schema","kind","input_sha256","source_ref","environment_ref",
        "settings","precision","candidate_provenance","interpretation","root","hessian","spectrum"),"certificate")
    record["schema"] == "cyax-scientific-certificate-v1" || throw(ArgumentError("unknown certificate schema"))
    kind=record["kind"]; kind in ("root","root_and_spectrum") || throw(ArgumentError("unknown certificate kind"))
    _valid_ref_hash(record["input_sha256"]) && record["input_sha256"]==input_digest ||
        throw(ArgumentError("certificate input digest mismatch"))
    source=_read_source_ref(record["source_ref"]); env=_read_environment_ref(record["environment_ref"])
    settings=_parse_settings(record["settings"])
    precision_bits=_strict_int(record["precision"],"precision",lower=256,upper=512)
    precision_bits in settings.precisions || throw(ArgumentError("certificate precision is not scheduled"))
    provenance=_read_candidate_provenance(record["candidate_provenance"])
    interpretation=record["interpretation"]
    _require_keys(interpretation,("basis","orientation","theta_period","phase_units",
        "coefficient_convention","numeric_semantics","spectrum_convention","physical_normalization","s"),"certificate interpretation")
    interpretation == _interpretation(model) || throw(ArgumentError("certificate interpretation differs from input"))
    root=record["root"]
    _require_keys(root,("status","lift","original_seed","translation","normalized_seed","center",
        "box","preconditioner","F_center","H_box","W","q_upper"),"root evidence")
    root["status"]=="certified_root" && root["lift"]=="unwrapped_centered_seed" ||
        throw(ArgumentError("unsupported root claim"))
    n=size(model.Q,1)
    original=vec(_exact_claims(root["original_seed"],[n]))
    translation_q=vec(_exact_claims(root["translation"],[n]))
    all(denominator(x)==1 for x in translation_q) || throw(ArgumentError("translation must be integral"))
    translation=BigInt[numerator(x) for x in translation_q]
    normalized=vec(_exact_claims(root["normalized_seed"],[n]))
    center=vec(_exact_claims(root["center"],[n]))
    all(_is_dyadic(x) for x in center) || throw(ArgumentError("center must be dyadic"))
    normalized == [original[i]-translation[i] for i in 1:n] || throw(ArgumentError("seed translation mismatch"))
    all(-1//2 <= normalized[i] < 1//2 for i in 1:n) || throw(ArgumentError("seed is not normalized"))
    boxclaims=_interval_claims(root["box"],[n]); boxvec=vec(boxclaims)
    for i in 1:n
        0 < boxvec[i][2]-boxvec[i][1] < 1 || throw(ArgumentError("root box width outside contract"))
        boxvec[i][1] < center[i] < boxvec[i][2] || throw(ArgumentError("center must be interior to original box"))
    end
    Cclaims=vec(_exact_claims(root["preconditioner"],[n,n]))
    all(_is_dyadic(x) for x in Cclaims) || throw(ArgumentError("preconditioner must be dyadic"))
    C=reshape(Cclaims,n,n)
    fclaims=vec(_interval_claims(root["F_center"],[n]))
    hclaims=vec(_interval_claims(root["H_box"],[n,n]))
    wclaims=vec(_interval_claims(root["W"],[n]))
    for j in 1:n, i in 1:n
        hclaims[(j-1)*n+i] == hclaims[(i-1)*n+j] || throw(ArgumentError("Hessian claim must be symmetric"))
    end
    all(boxvec[i][1] < wclaims[i][1] <= wclaims[i][2] < boxvec[i][2] for i in 1:n) ||
        throw(ArgumentError("stored Krawczyk image is not strictly inside original box"))
    qclaim=_parse_rational(root["q_upper"])
    qclaim >= 0 && qclaim < 1 || throw(ArgumentError("q_upper must be in [0,1)"))
    hessian=record["hessian"]
    _require_keys(hessian,("classification","method","negative_count","positive_count","unresolved_count","pivots"),"Hessian evidence")
    class=hessian["classification"]; class in ("MINIMUM","MAXIMUM","SADDLE","UNRESOLVED") || throw(ArgumentError("unknown Hessian class"))
    method=hessian["method"]; method in ("direct_ldl","spectral_counts","none") || throw(ArgumentError("unknown Hessian method"))
    neg=_strict_int(hessian["negative_count"],"negative_count",lower=0,upper=n)
    pos=_strict_int(hessian["positive_count"],"positive_count",lower=0,upper=n)
    unr=_strict_int(hessian["unresolved_count"],"unresolved_count",lower=0,upper=n)
    neg+pos+unr==n || throw(ArgumentError("Hessian counts do not sum to n"))
    pivdims=method=="direct_ldl" ? [n] : [0]
    pivclaims=vec(_interval_claims(hessian["pivots"],pivdims))
    if method=="direct_ldl"
        all(pair -> pair[1] > 0 || pair[2] < 0, pivclaims) ||
            throw(ArgumentError("direct Hessian pivots must exclude zero"))
    end
    if method=="none"
        (class=="UNRESOLVED" && neg==0 && pos==0 && unr==n) || throw(ArgumentError("none Hessian evidence must be unresolved"))
    end
    spectrum=record["spectrum"]
    if kind=="root"
        _require_keys(spectrum,("status","reason"),"absent spectrum")
        spectrum["status"]=="not_requested" && spectrum["reason"] in ("not_requested","coverage_unavailable") ||
            throw(ArgumentError("invalid absent spectrum"))
        method=="spectral_counts" && throw(ArgumentError("root-only certificate cannot assert spectral Hessian counts"))
    else
        spectrum isa AbstractDict || throw(ArgumentError("spectrum must be a table"))
    end
    return (;kind,source,env,settings,precision_bits,provenance,original,translation,
        normalized,center,boxclaims,C,fclaims,hclaims,wclaims,qclaim,hessian,class,method,
        neg,pos,unr,pivclaims,spectrum)
end

function _proof_interval_claims_fit(claims, actual)
    length(claims)==length(actual) || return false
    all(_claim_contains(claims[i],actual[i]) for i in eachindex(claims))
end

function _count_classification(negative::Int, positive::Int, unresolved::Int, n::Int)
    negative + positive + unresolved == n || throw(ArgumentError("Hessian sign counts do not sum to n"))
    negative > 0 && positive > 0 && return :SADDLE
    unresolved == 0 && positive == n && return :MINIMUM
    unresolved == 0 && negative == n && return :MAXIMUM
    :UNRESOLVED
end

function _verify_root_hessian(parsed, evaluated, spectrum_verified)
    n=length(parsed.original)
    h=parsed.hessian
    if parsed.method=="direct_ldl"
        evaluated.hessian_method==:direct_ldl && length(evaluated.hessian_pivots)==n || return false
        _proof_interval_claims_fit(parsed.pivclaims,evaluated.hessian_pivots) || return false
        signs=Symbol[_lo(p)>0 ? :positive : _hi(p)<0 ? :negative : :unresolved for p in evaluated.hessian_pivots]
        neg=count(==(:negative),signs);pos=count(==(:positive),signs);unr=n-neg-pos
        neg==parsed.neg && pos==parsed.pos && unr==parsed.unr || return false
        _classify(signs,n)==Symbol(parsed.class) || return false
    elseif parsed.method=="spectral_counts"
        spectrum_verified || return false
        spectrum=parsed.spectrum
        parsed.neg==spectrum["negative_count"] &&
            parsed.pos==spectrum["positive_count"] &&
            parsed.unr==spectrum["unresolved_count"] || return false
        _count_classification(parsed.neg,parsed.pos,parsed.unr,n)==Symbol(parsed.class) || return false
    elseif parsed.method=="none"
        parsed.class=="UNRESOLVED" && parsed.neg==0 && parsed.pos==0 && parsed.unr==n || return false
    end
    true
end

function _invalid(reason::String, precision=nothing)
    VerificationReport(:INVALID,reason,[(name="schema",outcome=:invalid,reason=reason)],:NOT_CHECKED,precision)
end
function _unproven(reason::String, precision=nothing)
    VerificationReport(:UNPROVEN,reason,[(name="submitted_claims",outcome=:unproved,reason=reason)],:NOT_CHECKED,precision)
end
function _unsupported_resource(precision=nothing)
    VerificationReport(:UNSUPPORTED,"RESOURCE_LIMIT",NamedTuple[],:NOT_CHECKED,precision)
end

"""Write exact, closed TOML inputs and a revalidated certificate in a fresh directory."""
function write_certificate(directory::AbstractString, model::CertificationModel, result;
                           source_ref, environment_ref)
    root = result isa RootResult ? result : result isa CertificationResult ? result.root :
        throw(ArgumentError("result must be RootResult or CertificationResult"))
    root.status === :CERTIFIED_ROOT || throw(ArgumentError("only a certified root can be written"))
    _backend_supported() || throw(ArgumentError("unsupported interval backend"))
    sr, er = _source_record(source_ref), _environment_record(environment_ref)
    provenance = _candidate_provenance_record(root.candidate_provenance)
    _preflight_input_rational_bytes(model)
    combined_spectrum = result isa CertificationResult &&
        result.spectrum.status === :CERTIFIED_BOUNDS ? result.spectrum : nothing
    root.hessian_method === :spectral_counts && combined_spectrum === nothing &&
        throw(ArgumentError("spectral Hessian counts require an accompanying certified spectrum"))
    lock(_PROOF_LOCK) do
        setprecision(BigFloat,root.precision) do
            setrounding(BigFloat,RoundNearest) do
                _revalidate_root(model,root,root.precision;
                    reuse_hessian=root.hessian_method === :spectral_counts) ||
                    throw(ArgumentError("root witness did not revalidate"))
                _preflight_certificate_rational_bytes(root,combined_spectrum)
                kind="root"; spectrum_record=Dict("status"=>"not_requested",
                    "reason"=>result isa CertificationResult ? "coverage_unavailable" : "not_requested")
                if combined_spectrum !== nothing
                    spectrum=combined_spectrum
                    _verify_spectrum_claim(model,root,_spectrum_record(spectrum),root.precision,
                                           root.options) || throw(ArgumentError("spectrum witness did not revalidate"))
                    kind="root_and_spectrum"; spectrum_record=_spectrum_record(spectrum)
                end
                input_bytes=_toml_bytes(_input_record(model))
                input_digest=bytes2hex(sha256(input_bytes))
                root.identity.input_sha256 == input_digest ||
                    throw(ArgumentError("result input identity does not match serialized model"))
                result isa CertificationResult && result.identity.input_sha256 != input_digest &&
                    throw(ArgumentError("certification input identity does not match serialized model"))
                certificate=Dict("schema"=>"cyax-scientific-certificate-v1","kind"=>kind,
                    "input_sha256"=>input_digest,"source_ref"=>sr,"environment_ref"=>er,
                    "settings"=>_settings_record(root.options),"precision"=>root.precision,
                    "candidate_provenance"=>provenance,
                    "interpretation"=>_interpretation(model),"root"=>_root_record(model,root),
                    "hessian"=>_hessian_record(root,combined_spectrum),"spectrum"=>spectrum_record)
                cert_bytes=_toml_bytes(certificate)
                out=abspath(directory)
                ispath(out) && throw(ArgumentError("certificate destination already exists"))
                mkpath(dirname(out)); mkdir(out)
                input_tmp=nothing; cert_tmp=nothing
                try
                    input_tmp,io=mktemp(out); write(io,input_bytes); close(io)
                    cert_tmp,io2=mktemp(out); write(io2,cert_bytes); close(io2)
                    report=verify_certificate(cert_tmp,input_tmp;
                        expected_input_sha256=input_digest,
                        expected_certificate_sha256=bytes2hex(sha256(cert_bytes)),
                        require=kind=="root" ? :root : :root_and_spectrum,
                        precision_bits=root.precision)
                    report.status===:VERIFIED || throw(ArgumentError("staged certificate failed replay: $(report.reason)"))
                    input_path=joinpath(out,"inputs.toml"); certificate_path=joinpath(out,"certificate.toml")
                    mv(input_tmp,input_path); input_tmp=nothing
                    mv(cert_tmp,certificate_path); cert_tmp=nothing
                    return (input_path=input_path,certificate_path=certificate_path,
                        input_sha256=input_digest,certificate_sha256=bytes2hex(sha256(cert_bytes)),kind=kind)
                catch
                    input_tmp!==nothing && isfile(input_tmp) && rm(input_tmp;force=true)
                    cert_tmp!==nothing && isfile(cert_tmp) && rm(cert_tmp;force=true)
                    isfile(joinpath(out,"certificate.toml")) && rm(joinpath(out,"certificate.toml");force=true)
                    isfile(joinpath(out,"inputs.toml")) && rm(joinpath(out,"inputs.toml");force=true)
                    rethrow()
                end
            end
        end
    end
end

"""Replay only the submitted exact witness; it never searches for another root."""
function verify_certificate(certificate_path::AbstractString, input_path::AbstractString;
        expected_input_sha256, expected_certificate_sha256=nothing,
        require=:root_and_spectrum, precision_bits=nothing)
    requested=require isa Symbol ? require : require isa AbstractString ? Symbol(require) :
        throw(ArgumentError("require must be :root or :root_and_spectrum"))
    requested in (:root,:root_and_spectrum) || throw(ArgumentError("unknown require selector"))
    _valid_ref_hash(expected_input_sha256) || return _invalid("expected input SHA-256 is malformed")
    expected_certificate_sha256===nothing || _valid_ref_hash(expected_certificate_sha256) ||
        return _invalid("expected certificate SHA-256 is malformed")
    input_bytes = try _read_toml_bytes(input_path) catch e
        e isa ResourceLimitError && return _unsupported_resource()
        return _invalid("input read failed: $(sprint(showerror,e))")
    end
    cert_bytes = try _read_toml_bytes(certificate_path) catch e
        e isa ResourceLimitError && return _unsupported_resource()
        return _invalid("certificate read failed: $(sprint(showerror,e))")
    end
    input_digest=bytes2hex(sha256(input_bytes)); cert_digest=bytes2hex(sha256(cert_bytes))
    input_digest==expected_input_sha256 || return _invalid("input digest mismatch")
    expected_certificate_sha256===nothing || cert_digest==expected_certificate_sha256 || return _invalid("certificate digest mismatch")
    input_record = try _parse_toml(input_bytes) catch e
        e isa ResourceLimitError && return _unsupported_resource()
        return _invalid("input parse failed: $(sprint(showerror,e))")
    end
    cert_record = try _parse_toml(cert_bytes) catch e
        e isa ResourceLimitError && return _unsupported_resource()
        return _invalid("certificate parse failed: $(sprint(showerror,e))")
    end
    model, input_interpretation = try _parse_input(input_record) catch e
        e isa ResourceLimitError && return _unsupported_resource()
        return _invalid("input schema invalid: $(sprint(showerror,e))")
    end
    input_size = size(model.Q)
    if input_size[1]>32 || input_size[2]>2048
        return VerificationReport(:UNSUPPORTED,"RESOURCE_LIMIT",NamedTuple[],:NOT_CHECKED,nothing)
    end
    parsed = try _read_certificate(cert_record,model,input_digest) catch e
        e isa ResourceLimitError && return _unsupported_resource()
        return _invalid("certificate schema invalid: $(sprint(showerror,e))")
    end
    if parsed.kind == "root_and_spectrum"
        try
            _verify_spectrum_claim(model, nothing, parsed.spectrum,
                parsed.precision_bits, parsed.settings; schema_only=true)
        catch e
            e isa ResourceLimitError && return _unsupported_resource(parsed.precision_bits)
            e isa ArgumentError && return _invalid("spectrum schema invalid: $(sprint(showerror,e))", parsed.precision_bits)
            rethrow()
        end
    end
    _backend_supported() || return VerificationReport(:UNSUPPORTED,"INTERVAL_BACKEND_VERSION",NamedTuple[],:NOT_CHECKED,nothing)
    if requested===:root_and_spectrum && parsed.kind=="root"
        return _unproven("root_and_spectrum was required but certificate is root-only",parsed.precision_bits)
    end
    replay_bits=precision_bits===nothing ? parsed.precision_bits :
        try _strict_int(precision_bits,"precision_bits",lower=2,upper=4096) catch e
            return _invalid("invalid replay precision: $(sprint(showerror,e))")
        end
    try
        result=_with_precision(replay_bits) do
            # Preserve every original rational claim; interval conversion is only
            # an operand conversion for the fixed proof adapter.
            bounds=vec(parsed.boxclaims)
            root=try
                _evaluate_root_claims(model,parsed.original,parsed.translation,parsed.normalized,
                    parsed.center,bounds,parsed.C,replay_bits,parsed.settings,parsed.provenance,
                    String[],_result_identity(model))
            catch e
                (e isa DivideError || e isa ProofArithmeticError) && return _unproven("root proof arithmetic was inconclusive",replay_bits)
                rethrow()
            end
            root.status===:CERTIFIED_ROOT || return _unproven("submitted root inclusion or q bound did not revalidate",replay_bits)
            _proof_interval_claims_fit(parsed.fclaims,root.F_center) || return _unproven("F(center) exceeds submitted claim",replay_bits)
            _proof_interval_claims_fit(parsed.hclaims,root.H_box) || return _unproven("H(X) exceeds submitted claim",replay_bits)
            _proof_interval_claims_fit(parsed.wclaims,root.W) || return _unproven("Krawczyk image exceeds submitted claim",replay_bits)
            root.q_upper<=parsed.qclaim || return _unproven("recomputed q exceeds submitted upper bound",replay_bits)
            hessian_pivots=parsed.pivclaims
            hessian_ok = if parsed.method=="direct_ldl"
                _proof_interval_claims_fit(hessian_pivots,root.hessian_pivots)
            elseif parsed.method=="none"
                true
            else
                true
            end
            hessian_ok || return _unproven("Hessian pivot claim did not revalidate",replay_bits)
            spectrum_ok=false
            if parsed.kind=="root_and_spectrum"
                spectrum_ok=try _verify_spectrum_claim(model,root,parsed.spectrum,replay_bits,parsed.settings) catch e
                    if e isa ArgumentError
                        return _invalid("spectrum schema invalid: $(sprint(showerror,e))",replay_bits)
                    elseif e isa DivideError || e isa ProofArithmeticError
                        false
                    else
                        rethrow()
                    end
                end
                spectrum_ok isa Bool || return spectrum_ok
                spectrum_ok || return _unproven("submitted generalized spectrum did not revalidate",replay_bits)
            end
            _verify_root_hessian(parsed,root,spectrum_ok) || return _unproven("Hessian classification/count claim did not revalidate",replay_bits)
            return VerificationReport(:VERIFIED,"submitted claims reproduced",
                [(name="root",outcome=:proved,reason="strict inclusion and contraction"),
                 (name="hessian",outcome=:proved,reason="submitted Hessian claim reproduced"),
                 (name="spectrum",outcome=parsed.kind=="root_and_spectrum" ? :proved : :not_requested,
                  reason=parsed.kind=="root_and_spectrum" ? "all mode bounds reproduced" : parsed.spectrum["reason"])],
                :NOT_CHECKED,replay_bits)
        end
        return result
    catch e
        e isa ResourceLimitError && return _unsupported_resource(replay_bits)
        (e isa DivideError || e isa ProofArithmeticError) && return _unproven("proof arithmetic was inconclusive",replay_bits)
        rethrow()
    end
end

"""Strict network-free replay CLI; argument paths and hashes are explicit."""
function _verify_cli(args=ARGS)
    allowed=("--inputs","--certificate","--expect-input-sha256","--expect-certificate-sha256","--require","--precision")
    values=Dict{String,String}(); i=1
    while i<=length(args)
        key=args[i]
        key in allowed || (println(stderr,"unknown argument: $key"); return 2)
        haskey(values,key) && (println(stderr,"repeated argument: $key"); return 2)
        i<length(args) || (println(stderr,"missing value for $key"); return 2)
        startswith(args[i+1],"--") && (println(stderr,"missing value for $key"); return 2)
        values[key]=args[i+1]; i+=2
    end
    required=("--inputs","--certificate","--expect-input-sha256","--expect-certificate-sha256")
    all(haskey(values,k) for k in required) || (println(stderr,"missing required argument"); return 2)
    req=get(values,"--require","root_and_spectrum")
    req in ("root","root_and_spectrum") || (println(stderr,"unknown --require selector");return 2)
    precision_value=get(values,"--precision",nothing)
    p=precision_value===nothing ? nothing : tryparse(Int,precision_value)
    (precision_value===nothing || p!==nothing) || (println(stderr,"invalid --precision");return 2)
    report=try
        verify_certificate(values["--certificate"],values["--inputs"];
            expected_input_sha256=values["--expect-input-sha256"],
            expected_certificate_sha256=values["--expect-certificate-sha256"],
            require=Symbol(req),precision_bits=p)
    catch e
        println(stderr,"unexpected verifier error: ",sprint(showerror,e)); return 4
    end
    println("status=",report.status," reason=",report.reason," provenance=",report.provenance)
    report.status===:VERIFIED && return 0
    report.status===:UNPROVEN && return 1
    report.status===:INVALID && return 2
    report.status===:UNSUPPORTED && return 3
    return 4
end
