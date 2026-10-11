module ScientificCertificationFixtures

using CYAxiverse
using SHA
using TOML
using LinearAlgebra
using Dates

const SC = CYAxiverse.scientific_certification
const ROOT = normpath(joinpath(@__DIR__, "../../.."))
const FIXTURE_PATH = joinpath(@__DIR__, "public.toml")
const VERIFIER_PATH = joinpath(ROOT, "scripts", "verify_scientific_certificate.jl")
include("oracles.jl")

_rat(x::Integer) = BigInt(x) // BigInt(1)
_rat(x::Rational) = BigInt(numerator(x)) // BigInt(denominator(x))
_rat(x::AbstractFloat) = Rational{BigInt}(x)
function _rat(x::AbstractString)
    parts = split(x, '/'; limit=2)
    length(parts) == 2 || throw(ArgumentError("rational fixture value must be numerator/denominator"))
    parse(BigInt, parts[1]) // parse(BigInt, parts[2])
end
_rat(x) = throw(ArgumentError("unsupported fixture scalar $(typeof(x))"))
_ratstr(x::Rational) = string(numerator(x), "/", denominator(x))
_ratstr(x::Integer) = string(x, "/1")

function _matrix(rows)
    isempty(rows) && throw(ArgumentError("empty fixture matrix"))
    width = length(first(rows))
    all(length(row) == width for row in rows) || throw(DimensionMismatch("ragged fixture matrix"))
    out = Matrix{Rational{BigInt}}(undef, length(rows), width)
    for i in eachindex(rows), j in eachindex(rows[i])
        out[i, j] = _rat(rows[i][j])
    end
    out
end

function _fixture_records()
    data = TOML.parsefile(FIXTURE_PATH)
    data["schema"] == "cyax-scientific-public-fixtures-v1" || error("unknown public fixture schema")
    fixtures = data["fixtures"]
    ids = [x["id"] for x in fixtures]
    length(ids) == length(unique(ids)) || error("duplicate public fixture id")
    return data["settings"], fixtures
end

function _record(id::String)
    _, fixtures = _fixture_records()
    matches = filter(x -> x["id"] == id, fixtures)
    length(matches) == 1 || throw(ArgumentError("unknown or duplicate public case $id"))
    only(matches)
end

function _model_seed(item)
    Q = _matrix(item["Q"])
    L = _matrix(item["L"])
    K = _matrix(item["K"])
    phi = [_rat(x) for x in item["phi"]]
    seed = [_rat(x) for x in item["seed"]]
    size(L, 2) == size(Q, 2) || throw(DimensionMismatch("fixture L/Q mismatch"))
    model = SC.prepare_model(Q, L, K; phases=phi)
    return model, seed, Q, L, K
end

function _benchmark_model(item)
    Q = Int.(reduce(vcat, [reshape(row, 1, :) for row in item["Q"]]))
    L = Float64.(reduce(vcat, [reshape(row, 1, :) for row in item["L"]]))
    K = Float64.(reduce(vcat, [reshape(row, 1, :) for row in item["K"]]))
    points = CYAxiverse.inflation_points
    context = points.prepare_context(Q, L, K)
    correction = points.correct_stationary_point(context, Float64.([1.5, -0.5]);
        residual_tolerance=1e-12)
    correction.status == :converged || error("T14 existing correction did not converge")
    length(correction.theta) == 2 || error("T14 correction returned wrong dimension")
    maximum(abs.(correction.theta .- [0.5, 0.5])) <= 1e-10 ||
        error("T14 existing correction is not near its declared periodic root")
    diagnostic = points.diagnose(context, correction.theta)
    diagnostic.negative_modes == 1 && diagnostic.positive_modes == 1 ||
        error("T14 existing diagnostic did not report a saddle")
    model = SC.prepare_model(Q, L, K; phases=[_rat(x) for x in item["phi"]])
    return model, [_rat(x) for x in correction.theta], Q, L, K, correction, diagnostic
end

function _t14_diagnostic_compatibility(diagnostic; relative_tolerance=BigInt(1)//BigInt(10)^10)
    values = diagnostic.hessian_eigenvalues
    length(values) == 2 || error("T14 numerical diagnostic must contain two generalized eigenvalues")
    issorted(values) || error("T14 numerical diagnostic eigenvalues are not ordered")
    values[1] < 0 < values[2] || error("T14 numerical diagnostic does not preserve signed mode order")
    references = oracle_t14_intervals(2048)
    observed = Rational{BigInt}[_rat(value) for value in values]
    relative_errors = Rational{BigInt}[]
    for i in eachindex(observed)
        lo, hi = references[i]
        reference = (lo + hi) / 2
        relative_error = abs(observed[i] - reference) / abs(reference)
        relative_error <= relative_tolerance ||
            error("T14 numerical diagnostic exceeds the declared relative tolerance")
        push!(relative_errors, relative_error)
    end
    Dict{String,Any}(
        "comparison_outcome" => "PASS",
        "comparison" => "signed ordered generalized Hessian eigenvalues versus exact-binary Float64 quadratic roots scaled by c=(2*pi)^2",
        "relative_tolerance" => _ratstr(relative_tolerance),
        "diagnostic_eigenvalues" => repr.(collect(values)),
        "reference_intervals" => [[_ratstr(lo), _ratstr(hi)] for (lo, hi) in references],
        "relative_errors" => string.(Float64.(relative_errors)))
end

function _process_alive(pid::Integer)
    ccall(:kill, Cint, (Cint, Cint), Cint(pid), Cint(0)) == 0
end

function _watchdog_self_test()
    (Sys.isapple() || Sys.islinux()) || error("process-group watchdog test is qualified only on Darwin and Linux")
    mktempdir() do parent
        pid_path = joinpath(parent, "child.pid")
        marker_path = joinpath(parent, "late-marker.txt")
        wrapper_path = joinpath(parent, "watchdog-wrapper.sh")
        write(wrapper_path, "#!/bin/sh\nsh -c 'sleep 30 & p=\$!; printf \"%s %s\" \"\$\$\" \"\$p\" > \"\$1\"; wait \"\$p\"; printf late > \"\$2\"' child \"\$1\" \"\$2\" &\nwait \$!\n")
        chmod(wrapper_path, 0o700)
        code, timed_out, elapsed = _run_process(String[wrapper_path, pid_path],
            joinpath(parent, "watchdog.log"), 2; cwd=ROOT)
        isfile(pid_path) || error("watchdog regression child did not start before timeout")
        child_pids = parse.(Int, split(read(pid_path, String)))
        length(child_pids) == 2 || error("watchdog regression did not record its owned process tree")
        reap_deadline = time() + 2
        while any(_process_alive, child_pids) && time() < reap_deadline
            sleep(0.05)
        end
        child_alive = any(_process_alive, child_pids)
        # Clean up a failure so the regression itself never leaks a child.
        for child_pid in reverse(child_pids)
            _process_alive(child_pid) && ccall(:kill, Cint, (Cint, Cint), Cint(child_pid), Cint(Base.SIGKILL))
        end
        late_marker = isfile(marker_path)
        (exit_code=code, timed_out=timed_out, elapsed_seconds=elapsed,
            child_alive_after_timeout=child_alive, late_marker=late_marker,
            passed=code == 124 && timed_out && !child_alive && !late_marker)
    end
end

function _options(item, precision::Int)
    settings, _ = _fixture_records()
    ratfield(k) = _rat(settings[k])
    SC.ProofOptions(precisions=(precision,),
        max_newton_steps=settings["max_newton_steps"],
        max_backtracks=settings["max_backtracks"],
        box_radii=[_rat(x) for x in settings["box_radii"]],
        max_expansions=settings["max_expansions"],
        max_inertia_evaluations=settings["max_inertia_evaluations"],
        relative_width=ratfield("relative_width"),
        spectral_goal=Symbol(item["spectral_goal"]))
end

function _expected_class(id)
    id == "T02" && return :MAXIMUM
    id in ("T04", "T14") && return :SADDLE
    :MINIMUM
end

function _run_math(id::String, precision::Int)
    item = _record(id)
    if get(item, "benchmark", false)
        model, seed, Q, L, K, correction, diagnostic = _benchmark_model(item)
        result = SC.certify(model, seed; options=_options(item, precision))
        return item, model, seed, Q, L, K, result, (; correction, diagnostic)
    end
    model, seed, Q, L, K = _model_seed(item)
    result = SC.certify(model, seed; options=_options(item, precision))
    item, model, seed, Q, L, K, result, NamedTuple()
end

function _check_basic(id, item, result)
    result.status == :CERTIFIED || error("$id p=$(result.root.precision): expected CERTIFIED, got $(result.status): $(result.root.reason) / $(result.spectrum.reason)")
    result.root.status == :CERTIFIED_ROOT || error("$id did not certify its root")
    result.spectrum.status == :CERTIFIED_BOUNDS || error("$id did not certify its spectrum")
    result.spectrum.goal == Symbol(item["spectral_goal"]) || error("$id spectral goal changed")
    result.spectrum.resolution_met || error("$id did not meet the frozen 2^-20 resolution")
    result.root.hessian_classification == _expected_class(id) ||
        error("$id expected $(_expected_class(id)), got $(result.root.hessian_classification)")
    result.spectrum.unresolved_count == 0 || error("$id has unresolved spectral modes")
    all(b -> b[1] < b[2] && (b[1] > 0 || b[2] < 0), result.spectrum.bounds) ||
        error("$id has a zero-crossing or empty spectral bound")
    all(c -> c.relative_width_met, result.spectrum.clusters) || error("$id has a wide spectral cluster")
    true
end

function _assert_contains(label, claim, reference)
    oracle_contains_interval(claim, reference) ||
        error("$label claim does not contain its independent exact/analytic reference")
end

function _exact_interval_endpoint(interval, bits, side)
    setprecision(BigFloat,bits) do
        side === :lower ? Rational{BigInt}(SC._lo(interval)) : Rational{BigInt}(SC._hi(interval))
    end
end

function _check_oracle(id, result, model)
    bounds = result.spectrum.bounds
    clo, chi = oracle_c_interval()
    c = (clo, chi)
    if id == "T01" || id == "T09_seed0" || id == "T09_seed1"
        _assert_contains("$id mode", only(bounds), c)
    elseif id == "T02"
        _assert_contains("T02 mode", only(bounds), (-chi, -clo))
    elseif id == "T03" || id == "T16_common_shift"
        _assert_contains("$id low mode", bounds[1], (clo//4, chi//4))
        _assert_contains("$id high mode", bounds[2], c)
    elseif id == "T04"
        refs = oracle_t04_intervals()
        _assert_contains("T04 negative mode", bounds[1], refs[1])
        _assert_contains("T04 positive mode", bounds[2], refs[2])
    elseif id == "T05"
        polynomial=oracle_t05_pencil_polynomial(model.Q,model.L,model.K)
        for i in 1:5
            _assert_contains("T05 mode $i", bounds[i],
                oracle_t05_scaled_reference(i, bounds[i], polynomial))
        end
    elseif id == "T06_hierarchy"
        _assert_contains("T06 hierarchy soft mode", bounds[1], oracle_scalar_curvature(-400//1))
        _assert_contains("T06 hierarchy refined reference", oracle_scalar_curvature(-400//1),
            oracle_scalar_curvature(-400//1; bits=2048))
        _assert_contains("T06 hierarchy hard mode", bounds[2], c)
        bounds[1][1] > 0 || error("T06 hierarchy lower bound is not strictly positive")
    elseif id == "T06_fractional_log"
        _assert_contains("T06 fractional soft mode", bounds[1], oracle_scalar_curvature(-1//2))
        _assert_contains("T06 fractional refined reference", oracle_scalar_curvature(-1//2),
            oracle_scalar_curvature(-1//2; bits=2048))
        _assert_contains("T06 fractional hard mode", bounds[2], c)
    elseif id == "T14"
        refs = oracle_t14_intervals()
        refined = oracle_t14_intervals(2048)
        _assert_contains("T14 negative generalized mode", bounds[1], refs[1])
        _assert_contains("T14 positive generalized mode", bounds[2], refs[2])
        _assert_contains("T14 negative refined oracle", refs[1], refined[1])
        _assert_contains("T14 positive refined oracle", refs[2], refined[2])
    elseif id == "T16_phase_unit"
        root_box = (_exact_interval_endpoint(result.root.box[1], result.root.precision, :lower),
            _exact_interval_endpoint(result.root.box[1], result.root.precision, :upper))
        _assert_contains("T16 phase root box", root_box, oracle_scalar_root(1//8))
        _assert_contains("T16 phase mode", only(bounds), c)
    elseif id == "T16_signed_charge"
        # The signed non-unit charge has the independent root +1/(32*pi).
        root_box = (_exact_interval_endpoint(result.root.box[1], result.root.precision, :lower),
            _exact_interval_endpoint(result.root.box[1], result.root.precision, :upper))
        _assert_contains("T16 signed-charge root box", root_box, oracle_scalar_root(-1//16))
        _assert_contains("T16 signed-charge mode", only(bounds), (4clo, 4chi))
    elseif id == "T17"
        length(result.spectrum.clusters) == 1 || error("T17 must retain one equal-mode cluster")
        only(result.spectrum.clusters).population == 2 || error("T17 cluster population must be two")
        _assert_contains("T17 cluster", (only(result.spectrum.clusters).lower,
            only(result.spectrum.clusters).upper), c)
    end
    true
end

function _result_record(path, id, precision, item, result, input_sha, certificate_sha;
        diagnostic_compatibility=nothing)
    root = result.root
    spectrum = result.spectrum
    record = Dict{String,Any}(
        "schema" => "cyax-scientific-case-result-v1", "case" => id,
        "family" => item["family"], "precision" => precision,
        "status" => String(result.status), "root_status" => String(root.status),
        "classification" => String(root.hessian_classification),
        "spectrum_status" => String(spectrum.status), "spectral_goal" => String(spectrum.goal),
        "resolution_met" => spectrum.resolution_met,
        "center" => [_ratstr(x) for x in root.center],
        "normalized_seed" => [_ratstr(x) for x in root.normalized_seed],
        "translation" => [string(x) for x in root.translation],
        "model_signature" => root.model_signature,
        "bounds" => [[_ratstr(b[1]), _ratstr(b[2])] for b in spectrum.bounds],
        "clusters" => [Dict("first"=>c.first,"last"=>c.last,"population"=>c.population,
            "lower"=>_ratstr(c.lower),"upper"=>_ratstr(c.upper),"sign"=>String(c.sign),
            "relative_width_met"=>c.relative_width_met) for c in spectrum.clusters],
        "input_sha256" => input_sha, "certificate_sha256" => certificate_sha)
    diagnostic_compatibility === nothing ||
        (record["benchmark_diagnostic_compatibility"] = diagnostic_compatibility)
    ispath(path) && error("refusing to overwrite evidence file $path")
    open(path, "w") do io
        TOML.print(io, record; sorted=true)
    end
    record
end

function _git(args...)
    read(Cmd(Cmd(["git", args...]); dir=ROOT), String) |> strip
end

function _source_identity()
    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    return (repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",
        commit, tree, clean=isempty(status))
end

function _hash_file(path)
    bytes2hex(sha256(read(path)))
end

function _environment_record(candidate, path)
    project = Base.active_project()
    project === nothing && error("qualification requires an active Julia project")
    project = abspath(project)
    manifest = joinpath(dirname(project), "Manifest.toml")
    isfile(manifest) || error("qualification requires the actual active Manifest.toml")
    record = Dict("schema"=>"cyax-scientific-environment-v1",
        "candidate_commit"=>candidate.commit,"candidate_tree"=>candidate.tree,
        "worktree_clean"=>candidate.clean,"julia_version"=>string(VERSION),
        "julia_executable"=>first(Base.julia_cmd().exec),
        "package_version"=>string(Base.pkgversion(CYAxiverse)),
        "interval_arithmetic_version"=>string(Base.pkgversion(SC.IA)),
        "active_project_sha256"=>_hash_file(project),
        "active_manifest_sha256"=>_hash_file(manifest),
        "active_project_path"=>project,"active_manifest_path"=>manifest,
        "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
        "openblas_num_threads"=>get(ENV,"OPENBLAS_NUM_THREADS","unset"),
        "julia_num_threads"=>get(ENV,"JULIA_NUM_THREADS","unset"))
    ispath(path) && error("refusing to overwrite evidence file $path")
    open(path, "w") do io
        TOML.print(io, record; sorted=true)
    end
    return (record=record, path=path, sha256=_hash_file(path),
        artifact_name=basename(path), project_dir=dirname(project))
end

function _warmup(precision)
    model = SC.prepare_model(reshape([1],1,1), reshape([1,0],2,1), reshape([1],1,1))
    result = SC.certify(model, [0]; options=SC.ProofOptions(precisions=(precision,), spectral_goal=:individuals))
    result.root.status == :CERTIFIED_ROOT || error("bounded qualification warm-up failed")
end

function _produce_case(id, precision, evidence_dir, environment_path, expected_environment_sha)
    ispath(evidence_dir) && error("producer destination collision: $evidence_dir")
    item = _record(id)
    envdata = read(environment_path)
    bytes2hex(sha256(envdata)) == expected_environment_sha || error("environment receipt digest changed")
    env_record = TOML.parse(String(envdata))
    candidate = _source_identity()
    env_record["candidate_commit"] == candidate.commit && env_record["candidate_tree"] == candidate.tree ||
        error("candidate identity changed after environment receipt")
    warmup_start = time_ns()
    _warmup(precision)
    warmup_proof_seconds = (time_ns() - warmup_start) / 1e9
    case_timing = @timed _run_math(id, precision)
    item, model, seed, _Q, _L, _K, result, _aux = case_timing.value
    case_compile_seconds = case_timing.compile_time
    warm_case_seconds = max(0.0, case_timing.time - case_compile_seconds)
    _check_basic(id, item, result)
    _check_oracle(id, result, model)
    source_ref = (repository=candidate.repository, commit=candidate.commit, tree=candidate.tree)
    environment_ref = (receipt_sha256=expected_environment_sha, artifact_name=basename(environment_path))
    written = SC.write_certificate(evidence_dir, model, result; source_ref, environment_ref)
    result_path = joinpath(evidence_dir, "case_result.toml")
    diagnostic_compatibility = id == "T14" ? _t14_diagnostic_compatibility(_aux.diagnostic) : nothing
    _result_record(result_path,id,precision,item,result,written.input_sha256,written.certificate_sha256;
        diagnostic_compatibility)
    println("case=$id precision=$precision status=CERTIFIED input_sha256=$(written.input_sha256) certificate_sha256=$(written.certificate_sha256) warmup_proof_seconds=$warmup_proof_seconds case_compile_seconds=$case_compile_seconds warm_case_seconds=$warm_case_seconds")
    return 0
end

function _set_report(path, data)
    ispath(path) && error("refusing to overwrite evidence file $path")
    open(path, "w") do io
        TOML.print(io, data; sorted=true)
    end
end

function _argv_record(path, argv, cwd, exit_code, timed_out; elapsed_seconds=nothing,
                      measured_child_argv=nothing, peak_rss_bytes=nothing,
                      peak_rss_raw=nothing, peak_rss_raw_unit=nothing,
                      peak_rss_method=nothing)
    record=Dict{String,Any}("argv"=>argv,"cwd"=>cwd,"exit_code"=>exit_code,
        "timed_out"=>timed_out,"julia_threads"=>get(ENV,"JULIA_NUM_THREADS","unset"),
        "openblas_num_threads"=>get(ENV,"OPENBLAS_NUM_THREADS","unset"))
    elapsed_seconds === nothing || (record["elapsed_seconds"]=elapsed_seconds)
    measured_child_argv === nothing || (record["measured_child_argv"]=measured_child_argv)
    if peak_rss_bytes !== nothing
        record["peak_rss_bytes"]=peak_rss_bytes
        record["peak_rss_raw"]=peak_rss_raw
        record["peak_rss_raw_unit"]=peak_rss_raw_unit
        record["peak_rss_method"]=peak_rss_method
    end
    _set_report(path,record)
end

const _PROCESS_GROUP_SUPERVISION = Sys.isapple() || Sys.islinux()

function _process_group_exists(pgid::Integer)
    status = ccall(:kill, Cint, (Cint, Cint), -Cint(pgid), Cint(0))
    status == 0 && return true
    error_code = Base.Libc.errno()
    error_code == Base.Libc.ESRCH && return false
    error_code == Base.Libc.EPERM && return true
    error("could not inspect owned process group (errno=$error_code)")
end

function _signal_process_group(pgid::Integer, signal::Integer)
    status = ccall(:kill, Cint, (Cint, Cint), -Cint(pgid), Cint(signal))
    status == 0 && return nothing
    error_code = Base.Libc.errno()
    error_code == Base.Libc.ESRCH && return nothing
    error("could not signal owned process group (errno=$error_code)")
end

function _wait_for_process_group_exit(pgid::Integer, timeout_seconds::Real)
    deadline = time() + timeout_seconds
    while _process_group_exists(pgid) && time() < deadline
        sleep(0.05)
    end
    !_process_group_exists(pgid)
end

function _terminate_process_group(pgid::Integer)
    _signal_process_group(pgid, Base.SIGTERM)
    _wait_for_process_group_exit(pgid, 0.25) && return nothing
    _signal_process_group(pgid, Base.SIGKILL)
    nothing
end

function _run_process(argv::Vector{String}, log_path::String, timeout_seconds::Int; cwd=ROOT)
    mkpath(dirname(log_path))
    open(log_path,"w") do log
        cmd = Cmd(Cmd(argv); dir=cwd, detach=_PROCESS_GROUP_SUPERVISION)
        proc = run(pipeline(addenv(cmd, "JULIA_NUM_THREADS"=>"1",
            "OPENBLAS_NUM_THREADS"=>"1"), stdout=log, stderr=log); wait=false)
        process_group = _PROCESS_GROUP_SUPERVISION ? getpid(proc) : nothing
        started = time()
        deadline = time() + timeout_seconds
        while !process_exited(proc) && time() < deadline
            sleep(0.1)
        end
        timed_out = !process_exited(proc)
        if timed_out
            if process_group === nothing
                kill(proc, Base.SIGKILL)
            else
                _terminate_process_group(process_group)
            end
        end
        wait(proc)
        if process_group !== nothing && !_wait_for_process_group_exit(process_group, 2.0)
            _signal_process_group(process_group, Base.SIGKILL)
            _wait_for_process_group_exit(process_group, 2.0) ||
                error("owned process group remained alive after watchdog cleanup")
            timed_out || error("process exited while owned subprocesses remained")
        end
        return (timed_out ? 124 : proc.exitcode, timed_out, time() - started)
    end
end

function _julia_argv(project_dir, args...)
    cmd = Base.julia_cmd()
    String[cmd.exec...; "--startup-file=no"; "--threads=1"; "--project=$project_dir"; string.(args)...]
end

function _peak_rss_bytes(raw::Integer, raw_unit::String)
    raw > 0 || error("peak RSS must be positive")
    raw_unit == "bytes" && return Int(raw)
    raw_unit == "KiB" && return Base.checked_mul(Int(raw),1024)
    error("unsupported peak RSS raw unit $raw_unit")
end

function _children_peak_rss_bytes()
    if Sys.isapple()
        storage=zeros(UInt8,176) # Darwin struct rusage; ru_maxrss follows two timeval fields.
        raw_unit="bytes"
        method="Darwin getrusage(RUSAGE_CHILDREN).ru_maxrss (bytes)"
    elseif Sys.islinux()
        Sys.WORD_SIZE == 64 || error("Linux peak RSS measurement is qualified only for the 64-bit rusage ABI")
        storage=zeros(UInt8,144) # Linux 64-bit struct rusage; ru_maxrss follows two timeval fields.
        raw_unit="KiB"
        method="Linux 64-bit getrusage(RUSAGE_CHILDREN).ru_maxrss (KiB converted to bytes)"
    else
        error("peak RSS measurement is qualified only for the Darwin and 64-bit Linux rusage ABIs")
    end
    status=ccall(:getrusage,Cint,(Cint,Ptr{UInt8}),-1,pointer(storage))
    status==0 || error("getrusage(RUSAGE_CHILDREN) failed with status $status")
    raw=unsafe_load(Ptr{Clong}(pointer(storage)+32))
    raw>0 || error("getrusage(RUSAGE_CHILDREN) returned no positive peak RSS")
    bytes=_peak_rss_bytes(raw,raw_unit)
    (bytes=bytes,raw=Int(raw),raw_unit=raw_unit,method=method)
end

function _measure_child(args)
    !isempty(args) || error("measured child command is empty")
    child=run(pipeline(Cmd(args);stdout=stdout,stderr=stderr);wait=false)
    wait(child)
    peak=_children_peak_rss_bytes()
    println("measurement_peak_rss_bytes=$(peak.bytes)")
    println("measurement_peak_rss_raw=$(peak.raw)")
    println("measurement_peak_rss_raw_unit=$(peak.raw_unit)")
    println("measurement_method=$(peak.method)")
    println("measurement_child_exit_code=$(child.exitcode)")
    child.exitcode
end

function _measurement_wrapper(project_dir, runner_path, argv)
    _julia_argv(project_dir,runner_path,"--measure-child","--",argv...)
end

function _producer_timing(log_path)
    body=read(log_path,String)
    timing=match(r"warmup_proof_seconds=([0-9.eE+-]+) case_compile_seconds=([0-9.eE+-]+) warm_case_seconds=([0-9.eE+-]+)",body)
    timing===nothing && error("producer timing markers are missing from $log_path")
    return (warmup_proof_seconds=parse(Float64,timing.captures[1]),
        case_compile_seconds=parse(Float64,timing.captures[2]),
        warm_case_seconds=parse(Float64,timing.captures[3]))
end

function _measured_peak_rss(log_path)
    body=read(log_path,String)
    bytes=match(r"measurement_peak_rss_bytes=([0-9]+)",body)
    raw=match(r"measurement_peak_rss_raw=([0-9]+)",body)
    unit=match(r"measurement_peak_rss_raw_unit=([^\r\n]+)",body)
    method=match(r"measurement_method=([^\r\n]+)",body)
    any(isnothing,(bytes,raw,unit,method)) && return nothing
    (bytes=parse(Int,bytes.captures[1]),raw=parse(Int,raw.captures[1]),
        raw_unit=unit.captures[1],method=method.captures[1])
end

function _replay_receipt(case_id, precision, result, environment, source, evidence_root, log_rel, replay_rel; require_provenance=true)
    cert = TOML.parsefile(joinpath(result["path"],"certificate.toml"))
    input_sha = result["input_sha256"]
    cert_sha = result["certificate_sha256"]
    producer_source = cert["source_ref"]
    producer_environment = cert["environment_ref"]
    observed_source = Dict("repository"=>source.repository,"commit"=>source.commit,"tree"=>source.tree)
    observed_environment = Dict("receipt_sha256"=>environment.sha256,"artifact_name"=>environment.artifact_name)
    provenance = producer_source == observed_source && producer_environment == observed_environment && source.clean &&
        environment.record["worktree_clean"]
    log_path = joinpath(evidence_root, log_rel)
    command_path = joinpath(evidence_root,"commands", "replay-$case_id-p$precision.toml")
    argv = _julia_argv(environment.project_dir, VERIFIER_PATH,
        "--inputs",joinpath(result["path"],"inputs.toml"),
        "--certificate",joinpath(result["path"],"certificate.toml"),
        "--expect-input-sha256",input_sha,"--expect-certificate-sha256",cert_sha,
        "--require","root_and_spectrum","--precision",string(precision))
    code, timedout, elapsed = _run_process(argv,log_path,300)
    _argv_record(command_path,argv,ROOT,code,timedout;elapsed_seconds=elapsed)
    text = read(log_path,String)
    verdict = occursin("status=VERIFIED",text) ? "VERIFIED" :
        occursin("status=UNPROVEN",text) ? "UNPROVEN" :
        occursin("status=INVALID",text) ? "INVALID" :
        occursin("status=UNSUPPORTED",text) ? "UNSUPPORTED" : "INTERNAL_ERROR"
    expected = Dict("VERIFIED"=>0,"UNPROVEN"=>1,"INVALID"=>2,"UNSUPPORTED"=>3)
    haskey(expected,verdict) || error("replay process did not return a declared result")
    code == expected[verdict] || error("replay exit code does not match mathematical verdict")
    ref_status = provenance ? "PASS" : "FAIL"
    replay = Dict("schema"=>"cyax-scientific-replay-v1","input_sha256"=>input_sha,
        "certificate_sha256"=>cert_sha,"producer_source"=>producer_source,
        "producer_environment"=>producer_environment,"observed_replay_source"=>observed_source,
        "observed_replay_environment"=>observed_environment,"replay_precision"=>precision,
        "required_kind"=>"root_and_spectrum","mathematical_verdict"=>verdict,
        "exit_code"=>code,"provenance_status"=>ref_status,
        "provenance_evidence_ref"=>"environment/producer-environment.toml",
        "claims"=>[Dict("name"=>"root","outcome"=>verdict=="VERIFIED" ? "proved" : "unproved",
            "reason"=>"fresh-process submitted root witness replay"),
            Dict("name"=>"spectrum","outcome"=>verdict=="VERIFIED" ? "proved" : "unproved",
            "reason"=>"fresh-process submitted generalized spectrum replay")],
        "command_evidence_ref"=>replace(log_rel,'\\'=>'/'))
    receipt_path = joinpath(evidence_root,replay_rel)
    _set_report(receipt_path,replay)
    verdict == "VERIFIED" || error("$case_id p=$precision replay was $verdict")
    require_provenance && !provenance && error("$case_id p=$precision execution provenance did not bind")
    return (receipt=receipt_path,log=log_path,command=command_path)
end

function _toml_rat(s)
    _rat(s)
end

function _read_result(path)
    record = TOML.parsefile(path)
    record["path"] = dirname(path)
    record
end

function _check_pairwise(records)
    by_key = Dict((r["case"],r["precision"])=>r for r in records)
    for family in ("T09_seed0","T09_seed1")
        for p in (256,512)
            a, b = by_key[("T09_seed0",p)],by_key[("T09_seed1",p)]
            a["normalized_seed"] == b["normalized_seed"] || error("T09 normalized seeds differ")
            a["center"] == b["center"] || error("T09 lifted proof centers differ")
            at = parse.(BigInt,a["translation"]); bt=parse.(BigInt,b["translation"])
            bt == at .+ 1 || error("T09 recorded translations do not differ by one")
        end
    end
    for p in (256,512)
        a,b=by_key[("T03",p)],by_key[("T16_common_shift",p)]
        a["bounds"] == b["bounds"] || error("T16 common scaling shift changed the scaled spectrum")
        a["model_signature"] != b["model_signature"] || error("T16 common scaling shift did not change input/model identity")
    end
    true
end

function _run_qualification_suite(evidence_dir::AbstractString, selected_ids,
                                  precisions::Tuple, profile::String; retain=true)
    out = abspath(evidence_dir)
    ispath(out) && error("public qualification evidence destination already exists: $out")
    mkpath(out)
    candidate = _source_identity()
    if retain && !candidate.clean
        error("retained public qualification requires a clean committed candidate")
    end
    mkpath(joinpath(out,"environment")); mkpath(joinpath(out,"commands"));
    mkpath(joinpath(out,"producer")); mkpath(joinpath(out,"replay")); mkpath(joinpath(out,"logs"))
    environment = _environment_record(candidate,joinpath(out,"environment","producer-environment.toml"))
    settings, all_fixtures = _fixture_records()
    fixtures = selected_ids === nothing ? all_fixtures :
        filter(item -> item["id"] in selected_ids, all_fixtures)
    selected_ids === nothing || length(fixtures) == length(selected_ids) ||
        error("fast qualification fixture selection is incomplete")
    producer_rows=Dict{String,Any}[]
    replay_rows=Dict{String,Any}[]
    records=Dict{String,Any}[]
    for item in fixtures, precision in precisions
        id=item["id"]
        destination=joinpath(out,"producer","$id-p$precision")
        argv=_julia_argv(environment.project_dir,@__FILE__,"--case",id,"--precision",string(precision),
            "--evidence-dir",destination,"--environment-receipt",environment.path,
            "--environment-sha256",environment.sha256)
        log=joinpath(out,"logs","producer-$id-p$precision.log")
        measure_peak = id in ("T03","T05")
        command_argv = measure_peak ? _measurement_wrapper(environment.project_dir,@__FILE__,argv) : argv
        code,timedout,elapsed=_run_process(command_argv,log,settings["watchdog_seconds"])
        peak_rss = measure_peak ? _measured_peak_rss(log) : nothing
        _argv_record(joinpath(out,"commands","producer-$id-p$precision.toml"),command_argv,ROOT,code,timedout;
            elapsed_seconds=elapsed,measured_child_argv=measure_peak ? argv : nothing,
            peak_rss_bytes=peak_rss===nothing ? nothing : peak_rss.bytes,
            peak_rss_raw=peak_rss===nothing ? nothing : peak_rss.raw,
            peak_rss_raw_unit=peak_rss===nothing ? nothing : peak_rss.raw_unit,
            peak_rss_method=peak_rss===nothing ? nothing : peak_rss.method)
        code==0 || error("$id p=$precision producer failed (exit=$code watchdog=$timedout): $(read(log,String))")
        measure_peak && peak_rss === nothing && error("peak RSS measurement missing for $id p=$precision")
        result=_read_result(joinpath(destination,"case_result.toml"))
        record_item=_record(id)
        _check_result_record(id,record_item,result)
        push!(records,result)
        timing=_producer_timing(log)
        row=Dict{String,Any}("case"=>id,"precision"=>precision,"status"=>result["status"],
            "classification"=>result["classification"],"n"=>length(item["Q"]),
            "M"=>length(item["Q"][1]),"interval_widths"=>[
                _ratstr(SC._parse_rational(b[2])-SC._parse_rational(b[1])) for b in result["bounds"]],
            "warmup_proof_seconds"=>timing.warmup_proof_seconds,
            "case_compile_seconds"=>timing.case_compile_seconds,
            "warm_case_seconds"=>timing.warm_case_seconds,
            "external_process_wall_seconds"=>elapsed,
            "input_sha256"=>result["input_sha256"],"certificate_sha256"=>result["certificate_sha256"],
            "result_ref"=>relpath(joinpath(destination,"case_result.toml"),out))
        if peak_rss !== nothing
            row["peak_rss_bytes"]=peak_rss.bytes
            row["peak_rss_raw"]=peak_rss.raw
            row["peak_rss_raw_unit"]=peak_rss.raw_unit
            row["peak_rss_method"]=peak_rss.method
        end
        push!(producer_rows,row)
        replay_log_rel=joinpath("logs","replay-$id-p$precision.log")
        replay_rel=joinpath("replay","$id-p$precision.toml")
        replay=_replay_receipt(id,precision,result,environment,candidate,out,replay_log_rel,replay_rel;
            require_provenance=retain)
        push!(replay_rows,Dict("case"=>id,"precision"=>precision,"verdict"=>"VERIFIED",
            "receipt_ref"=>relpath(replay.receipt,out),"log_ref"=>relpath(replay.log,out)))
    end
    profile == "public" && _check_pairwise(records)
    schema = profile == "public" ? "cyax-scientific-public-qualification-v1" :
        "cyax-scientific-fast-qualification-v1"
    suite=Dict("schema"=>schema,
        "candidate_commit"=>candidate.commit,"candidate_tree"=>candidate.tree,
        "environment_sha256"=>environment.sha256,"environment_ref"=>relpath(environment.path,out),
        "qualification_profile"=>"julia-1.12-ia-0.22.36-serial-$profile",
        "warm_timing_method"=>"Base.@timed measures _run_math; case_compile_seconds is Julia compiler time and warm_case_seconds is elapsed time minus compiler time; warmup_proof_seconds measures the bounded T01 warm-up separately",
        "peak_rss_method"=>"Platform-qualified getrusage(RUSAGE_CHILDREN) in a one-child Julia wrapper for T03/T05; row receipts record OS method, raw unit and bytes",
        "fixture_count"=>length(fixtures),"precision_count"=>length(precisions),
        "producer_count"=>length(producer_rows),"replay_count"=>length(replay_rows),
        "retained_final_evidence"=>retain,"producers"=>producer_rows,"replays"=>replay_rows)
    _set_report(joinpath(out,"qualification.toml"),suite)
    println("$(profile)_qualification=PASS producers=$(length(producer_rows)) replays=$(length(replay_rows)) candidate=$(candidate.commit) tree=$(candidate.tree)")
    0
end

run_public_suite(evidence_dir::AbstractString; retain=true) =
    _run_qualification_suite(evidence_dir, nothing, (256,512), "public"; retain)

run_fast_suite(evidence_dir::AbstractString; retain=true) =
    _run_qualification_suite(evidence_dir,
        ("T01","T02","T03","T04","T06_hierarchy","T06_fractional_log"),
        (256,), "fast"; retain)

function _check_result_record(id,item,result)
    result["case"]==id && result["family"]==item["family"] || error("case result identity mismatch")
    result["status"]=="CERTIFIED" && result["root_status"]=="CERTIFIED_ROOT" || error("$id did not certify")
    result["spectrum_status"]=="CERTIFIED_BOUNDS" && result["resolution_met"] || error("$id resolution gate failed")
    result["classification"]==String(_expected_class(id)) || error("$id classification mismatch")
    # The standalone child also checks independent references immediately before writing.
    if id=="T09_seed0" || id=="T09_seed1" || id=="T16_common_shift"
        return true
    end
    true
end

function qualify_from_test_profile!()
    full = get(ENV,"CYAXIVERSE_TEST_FULL","1") == "1"
    if full
        path = get(ENV,"CYAX_SCIENTIFIC_EVIDENCE_DIR","")
        if isempty(path)
            mktempdir(prefix="cyax-scientific-full-") do tmp
                run_public_suite(joinpath(tmp,"public-qualification"); retain=false)
            end
        else
            run_public_suite(path; retain=true)
        end
    else
        path = get(ENV,"CYAX_SCIENTIFIC_FAST_EVIDENCE_DIR","")
        if isempty(path)
            mktempdir(prefix="cyax-scientific-fast-") do tmp
                run_fast_suite(joinpath(tmp,"fast-qualification"); retain=false)
            end
        else
            run_fast_suite(path; retain=true)
        end
    end
    nothing
end

function _main(args=ARGS)
    if length(args)>=3 && args[1]=="--measure-child" && args[2]=="--"
        return _measure_child(args[3:end])
    elseif length(args)==1 && args[1]=="--watchdog-self-test"
        result = _watchdog_self_test()
        println("watchdog_self_test=", result.passed ? "PASS" : "FAIL",
            " exit_code=", result.exit_code, " timed_out=", result.timed_out,
            " child_alive_after_timeout=", result.child_alive_after_timeout,
            " late_marker=", result.late_marker,
            " elapsed_seconds=", result.elapsed_seconds)
        return result.passed ? 0 : 1
    elseif length(args)>=2 && args[1]=="--case"
        id=args[2]; precision=256; evidence=""; envpath=""; envsha=""
        i=3
        while i<=length(args)
            if args[i] in ("--precision","--evidence-dir","--environment-receipt","--environment-sha256") && i<length(args)
                key=args[i]; value=args[i+1]
                key=="--precision" ? (precision=parse(Int,value)) :
                key=="--evidence-dir" ? (evidence=value) :
                key=="--environment-receipt" ? (envpath=value) : (envsha=value)
                i+=2
            else
                error("unknown/incomplete case option $(args[i])")
            end
        end
        precision in (256,512) || error("only prescribed 256/512 precisions are accepted")
        !isempty(evidence) && !isempty(envpath) && !isempty(envsha) || error("case evidence/provenance arguments are required")
        return _produce_case(id,precision,evidence,envpath,envsha)
    elseif length(args)==4 && args[1]=="--suite" && args[2]=="public" && args[3]=="--evidence-dir"
        return run_public_suite(args[4]; retain=true)
    elseif length(args)==4 && args[1]=="--suite" && args[2]=="fast" && args[3]=="--evidence-dir"
        return run_fast_suite(args[4]; retain=true)
    elseif length(args)>=2 && args[1]=="--held-out-case"
        return _produce_held_out_case(args)
    elseif length(args)>=2 && args[1]=="--held-out"
        return _run_held_out(args)
    end
    error("usage: --suite public|fast --evidence-dir DIR | --case ID --precision 256|512 --evidence-dir DIR --environment-receipt FILE --environment-sha256 HASH | --held-out INPUT --oracle ORACLE --evidence-dir DIR")
end

function _held_out_data(input_path, oracle_path)
    input_bytes,input = SC._read_toml(input_path)
    oracle_bytes,oracle = SC._read_toml(oracle_path)
    Set(keys(input))==Set(("case","model","seed")) && input["case"]=="T15" ||
        error("T15 input envelope schema mismatch")
    Set(keys(oracle))==Set(("case","input_sha256","pencil_coefficients","mu_isolators",
        "expected_class","spectral_goal","relative_width","precisions")) && oracle["case"]=="T15" ||
        error("T15 oracle envelope schema mismatch")
    bytes2hex(sha256(input_bytes))==oracle["input_sha256"] || error("T15 oracle input digest mismatch")
    model, _ = SC._parse_input(input["model"])
    seed_record=input["seed"]
    seed_record isa AbstractDict || error("T15 seed must be an ExactArray")
    seed=vec(SC._exact_claims(seed_record,[3]))
    size(model.Q)==(3,5) || error("T15 model is outside the frozen n=3,M=5 family")
    expected_Q=BigInt[1 0 0 1 0; 0 1 0 1 1; 0 0 1 0 1]
    model.Q==Rational{BigInt}.(expected_Q) || error("T15 charge matrix is outside the frozen family")
    all(denominator(x)==1 && 1<=x<=3 for x in model.L[1,:]) || error("T15 coefficients must be positive integers in 1:3")
    all(iszero,model.L[2,:]) && all(iszero,model.phases) || error("T15 logs/phases must be exact zero")
    allowed_metrics=(BigInt[1 1 0; 1 3 0; 0 0 3],BigInt[1 2 0; 2 6 0; 0 0 3])
    any(model.K==Rational{BigInt}.(candidate) for candidate in allowed_metrics) ||
        error("T15 metric is outside the frozen triangular family")
    seed_abs=sort(abs.(seed))
    expected_seed=sort(Rational{BigInt}[1//2^8,1//2^9,1//2^10])
    seed_abs==expected_seed || error("T15 seed is outside the frozen signed seed family")
    A=Matrix{Rational{BigInt}}(model.Q*Diagonal(model.L[1,:])*transpose(model.Q))
    polynomial=oracle_exact_pencil_polynomial(A,model.K)
    coeff=oracle["pencil_coefficients"]
    coeff isa Vector && length(coeff)==4 && all(x->x isa AbstractString,coeff) ||
        error("T15 polynomial coefficient record is malformed")
    [SC._parse_rational(x) for x in coeff]==polynomial || error("T15 exact-pencil oracle disagrees with independent recomputation")
    isolator_record=oracle["mu_isolators"]
    isolators=vec(SC._interval_claims(isolator_record,[3]))
    for i in 1:3
        isolators[i][1]>0 || error("T15 isolators must be positive")
        i==1 || isolators[i-1][2] < isolators[i][1] || error("T15 isolators must be disjoint and increasing")
        _sturm_count(_sturm_sequence(polynomial),isolators[i][1],isolators[i][2])==1 ||
            error("T15 isolator does not contain exactly one root")
    end
    oracle["expected_class"]=="MINIMUM" || error("T15 expected class differs from contract")
    oracle["spectral_goal"]=="individuals" || error("T15 spectral goal differs from contract")
    _rat(oracle["relative_width"])==big(1)//big(1048576) || error("T15 resolution differs from contract")
    oracle["precisions"]==[256,512] || error("T15 precision schedule differs from contract")
    return (input_bytes=input_bytes,oracle_bytes=oracle_bytes,input=input,oracle=oracle,
        model=model,seed=seed,A=A,polynomial=polynomial,isolators=isolators,
        input_sha256=bytes2hex(sha256(input_bytes)))
end

function _produce_held_out_case(args)
    length(args)==7 || error("usage: --held-out-case INPUT ORACLE PRECISION DIR ENVIRONMENT_RECEIPT ENVIRONMENT_SHA256")
    input_path,oracle_path=args[2],args[3]
    precision=parse(Int,args[4]); precision in (256,512) || error("T15 precision must be 256 or 512")
    evidence_dir,environment_path,environment_sha=args[5],args[6],args[7]
    ispath(evidence_dir) && error("T15 producer destination collision")
    held=_held_out_data(input_path,oracle_path)
    environment_bytes=read(environment_path)
    bytes2hex(sha256(environment_bytes))==environment_sha || error("T15 environment receipt changed")
    env_record=TOML.parse(String(environment_bytes)); candidate=_source_identity()
    env_record["candidate_commit"]==candidate.commit && env_record["candidate_tree"]==candidate.tree && candidate.clean ||
        error("T15 requires the unchanged clean frozen candidate")
    _warmup(precision)
    settings,_=_fixture_records()
    options=SC.ProofOptions(precisions=(precision,),max_newton_steps=settings["max_newton_steps"],
        max_backtracks=settings["max_backtracks"],box_radii=[_rat(x) for x in settings["box_radii"]],
        max_expansions=settings["max_expansions"],max_inertia_evaluations=settings["max_inertia_evaluations"],
        relative_width=_rat(held.oracle["relative_width"]),spectral_goal=:individuals)
    result=SC.certify(held.model,held.seed;options)
    result.status==:CERTIFIED && result.root.hessian_classification==:MINIMUM ||
        error("T15 did not certify the held-out minimum")
    length(result.spectrum.bounds)==3 && result.spectrum.positive_count==3 && result.spectrum.resolution_met ||
        error("T15 did not resolve three positive individual modes")
    for i in 1:3
        reference=oracle_refine_scaled_root(held.polynomial,held.isolators[i],result.spectrum.bounds[i])
        _assert_contains("T15 eigenvalue $i",result.spectrum.bounds[i],reference)
    end
    source_ref=(repository=candidate.repository,commit=candidate.commit,tree=candidate.tree)
    environment_ref=(receipt_sha256=environment_sha,artifact_name=basename(environment_path))
    written=SC.write_certificate(evidence_dir,held.model,result;source_ref,environment_ref)
    item=Dict("id"=>"T15","family"=>"T15")
    _result_record(joinpath(evidence_dir,"case_result.toml"),"T15",precision,item,result,
        written.input_sha256,written.certificate_sha256)
    println("case=T15 precision=$precision status=CERTIFIED input_sha256=$(written.input_sha256) certificate_sha256=$(written.certificate_sha256)")
    0
end

function _run_held_out(args)
    values=Dict{String,String}(); i=1
    while i<=length(args)
        key=args[i]
        key in ("--held-out","--oracle","--evidence-dir") || error("unknown held-out option $key")
        i<length(args) || error("missing value for $key")
        haskey(values,key) && error("repeated held-out option $key")
        values[key]=args[i+1]; i+=2
    end
    all(haskey(values,k) for k in ("--held-out","--oracle","--evidence-dir")) ||
        error("held-out input, oracle and evidence directory are required")
    held=_held_out_data(values["--held-out"],values["--oracle"])
    out=abspath(values["--evidence-dir"])
    ispath(out) && error("T15 evidence destination collision")
    candidate=_source_identity(); candidate.clean || error("T15 requires a clean frozen candidate")
    mkpath(out); mkpath(joinpath(out,"environment")); mkpath(joinpath(out,"producer"));
    mkpath(joinpath(out,"replay")); mkpath(joinpath(out,"logs")); mkpath(joinpath(out,"commands"))
    environment=_environment_record(candidate,joinpath(out,"environment","producer-environment.toml"))
    producer_rows=Dict{String,Any}[]; replay_rows=Dict{String,Any}[]
    for precision in (256,512)
        id="T15"
        destination=joinpath(out,"producer","$id-p$precision")
        argv=_julia_argv(environment.project_dir,@__FILE__,"--held-out-case",values["--held-out"],
            values["--oracle"],string(precision),destination,environment.path,environment.sha256)
        log_rel=joinpath("logs","producer-$id-p$precision.log")
        code,timedout,elapsed=_run_process(argv,joinpath(out,log_rel),300)
        _argv_record(joinpath(out,"commands","producer-$id-p$precision.toml"),argv,ROOT,code,timedout;
            elapsed_seconds=elapsed)
        code==0 || error("T15 producer failed (exit=$code watchdog=$timedout); see $log_rel")
        record=_read_result(joinpath(destination,"case_result.toml"))
        record["status"]=="CERTIFIED" && record["classification"]=="MINIMUM" || error("T15 result record failed")
        push!(producer_rows,Dict("case"=>id,"precision"=>precision,"status"=>record["status"],
            "input_sha256"=>record["input_sha256"],"certificate_sha256"=>record["certificate_sha256"],
            "result_ref"=>relpath(joinpath(destination,"case_result.toml"),out)))
        replay_log_rel=joinpath("logs","replay-$id-p$precision.log")
        replay_rel=joinpath("replay","$id-p$precision.toml")
        replay=_replay_receipt(id,precision,record,environment,candidate,out,replay_log_rel,replay_rel)
        push!(replay_rows,Dict("case"=>id,"precision"=>precision,"verdict"=>"VERIFIED",
            "receipt_ref"=>relpath(replay.receipt,out),"log_ref"=>relpath(replay.log,out)))
    end
    suite=Dict("schema"=>"cyax-scientific-held-out-qualification-v1",
        "candidate_commit"=>candidate.commit,"candidate_tree"=>candidate.tree,
        "input_sha256"=>held.input_sha256,"oracle_sha256"=>bytes2hex(sha256(held.oracle_bytes)),
        "environment_sha256"=>environment.sha256,"producer_count"=>length(producer_rows),
        "replay_count"=>length(replay_rows),"producers"=>producer_rows,"replays"=>replay_rows)
    _set_report(joinpath(out,"qualification.toml"),suite)
    println("held_out_qualification=PASS producers=2 replays=2 candidate=$(candidate.commit) tree=$(candidate.tree)")
    0
end
end # module

if abspath(PROGRAM_FILE)==@__FILE__
    exit(ScientificCertificationFixtures._main(ARGS))
end
