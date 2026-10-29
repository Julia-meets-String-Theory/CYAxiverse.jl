using SHA
using TOML

const SC = CYAxiverse.scientific_certification

@test SC._IR[] === nothing

# CYAX_R2_PROOFOPTIONS_EXACT_RATIONAL_TESTS_BEGIN
@testset "ProofOptions require exact rational radius and width inputs" begin
    int_options=SC.ProofOptions(precisions=(256,),
        box_radii=(1//2^20,1//2^30),relative_width=1//2^20)
    @test int_options.box_radii == (BigInt(1)//BigInt(2)^20,BigInt(1)//BigInt(2)^30)
    @test int_options.relative_width == BigInt(1)//BigInt(2)^20

    big_radii=Rational{BigInt}[BigInt(1)//BigInt(2)^20,BigInt(1)//BigInt(2)^30]
    big_options=SC.ProofOptions(precisions=[256,512],box_radii=big_radii,
        relative_width=BigInt(1)//BigInt(2)^20)
    @test big_options.box_radii == Tuple(big_radii)
    @test big_options.relative_width == BigInt(1)//BigInt(2)^20

    for float_type in (Float16,Float32,Float64,BigFloat)
        radius=float_type(0.25)
        width=float_type(2.0^-20)
        @test_throws ArgumentError SC.ProofOptions(precisions=(256,),box_radii=(radius,))
        @test_throws ArgumentError SC.ProofOptions(precisions=(256,),relative_width=width)
    end
end
# CYAX_R2_PROOFOPTIONS_EXACT_RATIONAL_TESTS_END

# PR199_MODEL_INVARIANT_REGRESSIONS_BEGIN
@testset "Public proofs reject inconsistent mutable and directly constructed models" begin
    R=Rational{BigInt}
    options=SC.ProofOptions(precisions=(256,),spectral_goal=:individuals)
    valid=SC.prepare_model(R[1 0;0 1],R[1 1;0 0],R[2 1;1 2])
    result=SC.certify(valid,R[0,0];options)
    @test result.root.status === :CERTIFIED_ROOT
    @test result.spectrum.status === :CERTIFIED_BOUNDS

    asymmetric=deepcopy(valid)
    asymmetric.K[1,2]=100//1
    stale_scale=deepcopy(valid)
    stale_scale.L[2,1]=1//1
    fractional_charge=deepcopy(valid)
    fractional_charge.Q[1,1]=1//2
    short_phases=deepcopy(valid)
    pop!(short_phases.phases)
    nonfinite=deepcopy(valid)
    nonfinite.phases[1]=big(1)//big(0)
    direct_asymmetric=SC.CertificationModel(copy(valid.Q),copy(valid.L),
        R[2 100;1 2],copy(valid.phases),valid.s)
    direct_stale_scale=SC.CertificationModel(copy(valid.Q),copy(valid.L),
        copy(valid.K),copy(valid.phases),1//big(1))
    direct_dimensions=SC.CertificationModel(copy(valid.Q),copy(valid.L),
        copy(valid.K),copy(valid.phases),valid.s,nothing,3,2)
    direct_missing_scale=SC.CertificationModel(copy(valid.Q),copy(valid.L),
        copy(valid.K),copy(valid.phases),nothing,nothing,2,2)
    for (model,error_type) in ((asymmetric,ArgumentError),
            (stale_scale,ArgumentError),(fractional_charge,ArgumentError),
            (short_phases,DimensionMismatch),(direct_asymmetric,ArgumentError),
            (direct_stale_scale,ArgumentError),(direct_dimensions,DimensionMismatch),
            (direct_missing_scale,ArgumentError),(nonfinite,ArgumentError))
        @test_throws error_type SC.certify(model,R[0,0];options)
        @test_throws error_type SC.certify_stationary_point(model,R[0,0];options)
        @test_throws error_type SC.certify_generalized_spectrum(model,result.root;options)
    end
    mktempdir() do directory
        @test_throws "K must be exactly symmetric" SC.write_certificate(
            directory,asymmetric,result;source_ref=nothing,environment_ref=nothing)
        @test_throws "model amplitude scale" SC.write_certificate(
            directory,stale_scale,result;source_ref=nothing,environment_ref=nothing)
        @test isempty(readdir(directory))
    end

    # Valid edits are still usable when their cached interpretation agrees.
    updated=deepcopy(valid)
    updated.L[2,1]=1//1
    updated=SC.CertificationModel(updated.Q,updated.L,updated.K,updated.phases,1//big(1))
    @test SC.certify(updated,R[0,0];options).spectrum.status === :CERTIFIED_BOUNDS
end
# PR199_MODEL_INVARIANT_REGRESSIONS_END

function _run_scientific_cli(_repository_root, input_path, certificate_path,
        input_sha256, certificate_sha256; precision=256)
    cli_args = ["--inputs", input_path, "--certificate", certificate_path,
        "--expect-input-sha256", input_sha256,
        "--expect-certificate-sha256", certificate_sha256,
        "--require", "root_and_spectrum", "--precision", string(precision)]
    # Same limit as the qualification fixture's watchdog (fixtures TOML settings).
    # The verifier runs in-process, so the limit is enforced only at task yield
    # points; a non-yielding hang still needs the process-isolated CI step.
    timeout_seconds = TOML.parsefile(joinpath(@__DIR__, "fixtures",
        "scientific_certification", "public.toml"))["settings"]["watchdog_seconds"]
    mktemp() do _path, output
        task = @async redirect_stdout(output) do
            redirect_stderr(output) do
                SC._verify_cli(cli_args)
            end
        end
        timedwait(() -> istaskdone(task), Float64(timeout_seconds)) === :ok ||
            error("scientific verifier CLI exceeded the $(timeout_seconds) s watchdog")
        exit_code = fetch(task)
        flush(output); seekstart(output)
        (exit_code=exit_code, output=read(output,String), argv=cli_args)
    end
end

@requires_git @testset "coverage-enabled CLI child keeps candidate source clean" begin
    repository_root=normpath(joinpath(@__DIR__,".."))
    status_argv=["git","-C",repository_root,"status","--porcelain=v1","--untracked-files=all"]
    before_status=read(Cmd(status_argv),String)
    options=Base.JLOptions()
    detached_coverage=child_coverage_args()
    trace_path=nothing
    if options.code_coverage!=0 && options.output_code_coverage==C_NULL
        @test length(detached_coverage)==1
        trace_path=split(only(detached_coverage),'=';limit=2)[2]
        @test endswith(trace_path,".info")
        @test startswith(relpath(trace_path,repository_root),"..")
        @test !ispath(trace_path)
    else
        @test isempty(detached_coverage)
    end

    argv=vcat(String[Base.julia_cmd().exec...],detached_coverage,String["--startup-file=no",
        "--project=$repository_root",joinpath(repository_root,"scripts",
        "verify_scientific_certificate.jl")])
    mktemp() do _path,output_file
        process=run(pipeline(Cmd(Cmd(argv);dir=repository_root);
            stdout=output_file,stderr=output_file);wait=false)
        wait(process)
        flush(output_file); seekstart(output_file)
        output=read(output_file,String)
        @test process.exitcode==2
        @test occursin("missing required argument",output)
    end
    if trace_path!==nothing
        @test isfile(trace_path)
        if isfile(trace_path)
            @test filesize(trace_path)>0
            rm(trace_path;force=true)
        end
    end
    @test read(Cmd(status_argv),String)==before_status
end

@testset "Bounded scientific certification" begin
    @testset "exact input and phase conventions" begin
        @test_throws DimensionMismatch SC.prepare_model(reshape([1], 1, 1), [1 0 0], reshape([1], 1, 1))
        @test_throws DimensionMismatch SC.prepare_model(zeros(Int,0,1),reshape([1,0],2,1),zeros(Int,0,0))
        @test_throws DimensionMismatch SC.prepare_model(zeros(Int,1,0),zeros(Int,2,0),reshape([1],1,1))
        # The public scientific contract uses Q with shape n×M; unlike the
        # synthetic SK-02/SK-08 (term,axion) reader fixture, it is not M×n.
        q_transposed = [1 0; 0 1; 1 1]
        l_for_n2_m3 = [1 1 1; 0 0 0]
        @test_throws DimensionMismatch SC.prepare_model(q_transposed, l_for_n2_m3, Matrix{Int}(I, 2, 2))
        @test size(SC.prepare_model(transpose(q_transposed), l_for_n2_m3,
            Matrix{Int}(I, 2, 2)).Q) == (2, 3)
        @test_throws ArgumentError SC.prepare_model(reshape([1.5], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1))
        @test_throws ArgumentError SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([NaN], 1, 1))
        @test_throws ArgumentError SC.prepare_model(reshape([1], 1, 1),
            reshape([1, 0], 2, 1), reshape([Inf], 1, 1))
        @test_throws ArgumentError SC.prepare_model(reshape([1], 1, 1),
            reshape([1, 0], 2, 1), reshape([-Inf], 1, 1))
        @test_throws ArgumentError SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1); phases=[NaN])
        @test_throws ArgumentError SC.prepare_model(reshape([true], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1))
        @test_throws ArgumentError SC.prepare_model(reshape([SC.IA.interval(1)], 1, 1),
            reshape([1, 0], 2, 1), reshape([1], 1, 1))
        @test_throws ArgumentError SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([true], 1, 1))
        @test_throws ArgumentError SC.ProofOptions(max_newton_steps=true)
        @test_throws ArgumentError SC.ProofOptions(precisions=(128,))
        @test_throws ArgumentError SC.ProofOptions(box_radii=(true,))
        @test_throws ArgumentError SC.ProofOptions(relative_width=true)
        @test_throws ArgumentError SC.ProofOptions(spectral_goal=:physical)
        positional_options=SC.ProofOptions()
        @test SC.ProofOptions(positional_options.precisions,
            positional_options.max_newton_steps,positional_options.max_backtracks,
            positional_options.box_radii,positional_options.max_expansions,
            positional_options.max_inertia_evaluations,positional_options.relative_width,
            positional_options.spectral_goal,positional_options.max_dimension,
            positional_options.max_terms).precisions==positional_options.precisions
        @test_throws ArgumentError SC.ProofOptions((128,),
            positional_options.max_newton_steps,positional_options.max_backtracks,
            positional_options.box_radii,positional_options.max_expansions,
            positional_options.max_inertia_evaluations,positional_options.relative_width,
            positional_options.spectral_goal,positional_options.max_dimension,
            positional_options.max_terms)

    limited_model = SC.prepare_model(reshape([1, 1], 1, 2), [1 1; 0 0], reshape([1], 1, 1))
    limited = SC.certify_stationary_point(limited_model, [0];
        options=SC.ProofOptions(precisions=(256,), max_terms=1))
    @test limited.status == :UNSUPPORTED
    @test limited.reason == "RESOURCE_LIMIT"

        q_input = reshape([1], 1, 1)
        l_input = reshape([1, 0], 2, 1)
        k_input = reshape([2], 1, 1)
        phase_input = [1//8]
        copied = SC.prepare_model(q_input, l_input, k_input; phases=phase_input)
        q_input[1] = 7; l_input[1,1] = 9; k_input[1] = 11; phase_input[1] = 1//4
        @test copied.Q == reshape([1//1], 1, 1)
        @test copied.L[1,1] == 1 && copied.K[1,1] == 2 && copied.phases == [1//8]

        exact_float=SC._exact_rational(0.1)
        exact_bigfloat=setprecision(BigFloat,512) do
            BigFloat(1)+BigFloat(2)^(-500)
        end
        negative_bigfloat=setprecision(BigFloat,512) do
            -exact_bigfloat
        end
        exact_bigfloat_rational=(big(2)^500+1)//(big(2)^500)
        @test SC._exact_rational(exact_bigfloat)==exact_bigfloat_rational
        @test SC._exact_rational(negative_bigfloat)==-exact_bigfloat_rational
        @test SC._exact_rational(BigFloat(0;precision=512))==0//1
        @test SC._exact_rational(BigFloat(-0.0;precision=512))==0//1
        component_bits=SC._RATIONAL_COMPONENT_BITS
        largest_integer=BigInt(10)^4095
        largest_integer_float=BigFloat(largest_integer;precision=component_bits)
        @test SC._exact_rational(largest_integer_float)==largest_integer//BigInt(1)
        over_limit_integer_float=BigFloat(BigInt(10)^4096;precision=component_bits)
        @test_throws SC.ResourceLimitError SC._exact_rational(over_limit_integer_float)
        denominator_exponent=SC._MAX_DYADIC_DENOMINATOR_EXPONENT
        largest_denominator=BigInt(1)<<denominator_exponent
        @test ndigits(largest_denominator)==4096
        @test SC._exact_rational(ldexp(BigFloat(1;precision=256),-denominator_exponent))==
            BigInt(1)//largest_denominator
        @test_throws SC.ResourceLimitError SC._exact_rational(
            ldexp(BigFloat(1;precision=256),-(denominator_exponent+1)))
        for exponent in (14_000,-14_000)
            oversized=ldexp(BigFloat(1;precision=256),exponent)
            @test_throws SC.ResourceLimitError SC._exact_rational(oversized)
            @test_throws SC.ResourceLimitError SC._endpoint_rational(oversized)
            oversized_interval=SC._iv(oversized,oversized,256)
            @test_throws SC.ResourceLimitError SC._interval_array_record([oversized_interval])
            @test_throws SC.ResourceLimitError SC._diagonal_clusters([oversized_interval],
                BigInt(1)//BigInt(1))
            oversized_model=SC.prepare_model(reshape([1],1,1),
                reshape([1,0],2,1),reshape([oversized],1,1))
            @test oversized_model.resource_reason=="RESOURCE_LIMIT"
            @test SC.certify(oversized_model,[0]).status==:UNSUPPORTED
        end
        small_model=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
        oversized_seed=ldexp(BigFloat(1;precision=256),14_000)
        oversized_seed_result=SC.certify_stationary_point(small_model,[oversized_seed];
            options=SC.ProofOptions(precisions=(256,)))
        @test oversized_seed_result.status==:UNSUPPORTED
        @test oversized_seed_result.reason=="RESOURCE_LIMIT"
        @test isempty(oversized_seed_result.original_seed)
        @test SC.certify_generalized_spectrum(small_model,oversized_seed_result).status==:UNSUPPORTED
        @test SC.certify(small_model,[oversized_seed]).status==:UNSUPPORTED
        expansion_model=SC.prepare_model(Matrix{Int}(I,2,2),[1 1;0 -10_000],Matrix{Int}(I,2,2))
        expansion_result=SC.certify_stationary_point(expansion_model,[0,0];
            options=SC.ProofOptions(precisions=(256,)))
        @test expansion_result.status==:UNSUPPORTED
        @test expansion_result.reason=="RESOURCE_LIMIT"
        high_precision_sparse=setprecision(BigFloat,100_000) do
            BigFloat(1)+ldexp(BigFloat(1),-500)
        end
        @test precision(high_precision_sparse)==100_000
        @test SC._exact_rational(high_precision_sparse)==exact_bigfloat_rational
        setprecision(BigFloat,384) do
            setrounding(BigFloat,RoundUp) do
                ambient_precision=precision(BigFloat)
                ambient_rounding=rounding(BigFloat)
                @test SC._exact_rational(exact_bigfloat)==exact_bigfloat_rational
                @test precision(BigFloat)==ambient_precision
                @test rounding(BigFloat)==ambient_rounding
            end
        end
        high_precision_k=reshape([exact_bigfloat],1,1)
        high_precision_phase=[exact_bigfloat]
        high_precision_model=SC.prepare_model(reshape([1],1,1),
            reshape([1,0],2,1),high_precision_k;phases=high_precision_phase)
        @test high_precision_model.K[1,1]==exact_bigfloat_rational
        @test high_precision_model.phases[1]==exact_bigfloat_rational
        high_precision_k[1]=2
        high_precision_phase[1]=BigFloat(0.2)
        @test high_precision_model.K[1,1]==exact_bigfloat_rational
        @test high_precision_model.phases[1]==exact_bigfloat_rational
        @test exact_float!=1//10
        @test high_precision_model.K[1,1]!=1//1
        for value in (BigInt(1)//BigInt(10),0.1,exact_bigfloat,
                negative_bigfloat,BigFloat(0;precision=512))
            bits=value isa BigFloat ? 512 : 256
            exact=SC._exact_rational(value)
            interval=SC._with_precision(bits) do
                SC._iv(exact,bits)
            end
            encoded=SC._interval_array_record([interval])["data"][1]
            parsed=(SC._parse_rational(encoded[1]),SC._parse_rational(encoded[2]))
            actual=(SC._rational_bound(SC._lo(interval)),SC._rational_bound(SC._hi(interval)))
            @test parsed==actual
            @test parsed[1]<=exact<=parsed[2]
        end

        model = SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1);
            phases=[1//8])
        SC._with_precision(256) do
            phase = SC._phase_value(model, [SC._iv(0, 256)], 1, 256)
            @test SC._lo(phase) <= big(1)//big(8) <= SC._hi(phase)
            amplitude = only(SC._amplitudes(model, 256))
            @test SC._lo(amplitude) <= 1 <= SC._hi(amplitude)

            derivative_model = SC.prepare_model(reshape([1], 1, 1),
                reshape([1, 0], 2, 1), reshape([1], 1, 1))
            x = [SC._iv(1//8, 256)]
            gradient = only(SC._gradient(derivative_model, x, 256))
            hessian = SC._hessian(derivative_model, x, 256)[1, 1]
            setprecision(BigFloat, 512) do
                independent_gradient = sqrt(BigFloat(2)) * BigFloat(pi)
                independent_hessian = 2 * sqrt(BigFloat(2)) * BigFloat(pi)^2
                @test SC._lo(gradient) <= independent_gradient <= SC._hi(gradient)
                @test SC._lo(hessian) <= independent_hessian <= SC._hi(hessian)
            end
        end
    end

    @testset "T01 and T02 positive local proofs" begin
        options = SC.ProofOptions(precisions=(256,), spectral_goal=:individuals)
        model = SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1))
        minimum_result = SC.certify(model, [0]; options)
        @test minimum_result.status == :CERTIFIED
        @test minimum_result.root.status == :CERTIFIED_ROOT
        @test minimum_result.root.hessian_classification == :MINIMUM
        @test length(minimum_result.spectrum.bounds) == 1
        @test minimum_result.spectrum.bounds[1][1] > 0
        synthetic_unsupported_root=SC._empty_root(:UNSUPPORTED,
            "INTERVAL_BACKEND_VERSION",[0//1],options;
            identity=SC._result_identity(model))
        unsupported_spectrum=SC.certify_generalized_spectrum(model,
            synthetic_unsupported_root;options)
        @test unsupported_spectrum.status==:UNSUPPORTED
        @test unsupported_spectrum.reason=="INTERVAL_BACKEND_VERSION"
        @test_throws ArgumentError SC.certify_generalized_spectrum(model,
            minimum_result.root;options=SC.ProofOptions(precisions=(512,)))

        maximum_result = SC.certify(model, [1//2]; options)
        @test maximum_result.status == :CERTIFIED
        @test maximum_result.root.normalized_seed == [-1//2]
        @test maximum_result.root.hessian_classification == :MAXIMUM
        @test maximum_result.spectrum.bounds[1][2] < 0
    end

    @testset "exact certificate round-trip and semantic mutation" begin
        model = SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1))
        result = SC.certify(model, [0]; options=SC.ProofOptions(precisions=(256,), spectral_goal=:individuals))
        @test result.status == :CERTIFIED
        repository_root = normpath(joinpath(@__DIR__, ".."))
        (; commit, tree) = head_identity(repository_root)
        source = (repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",
            commit, tree)
        mktempdir() do parent
            project = Base.active_project()
            project === nothing && error("test qualification requires an active project")
            manifest = joinpath(dirname(project), "Manifest.toml")
            environment_bytes = Vector{UInt8}(codeunits(join((
                "schema=cyax-scientific-test-environment-v1",
                "julia_version=$(VERSION)",
                "interval_arithmetic_version=$(Base.pkgversion(SC.IA))",
                "project_sha256=$(bytes2hex(sha256(read(project))))",
                "manifest_sha256=$(bytes2hex(sha256(read(manifest))))"), "\n") * "\n"))
            environment_path = joinpath(parent, "producer-environment.toml")
            write(environment_path, environment_bytes)
            environment = (receipt_sha256=bytes2hex(sha256(environment_bytes)),
                artifact_name=basename(environment_path))
            first = SC.write_certificate(joinpath(parent, "first"), model, result;
                source_ref=source, environment_ref=environment)
            second = SC.write_certificate(joinpath(parent, "second"), model, result;
                source_ref=source, environment_ref=environment)
            @test read(first.input_path) == read(second.input_path)
            @test read(first.certificate_path) == read(second.certificate_path)
            report = SC.verify_certificate(first.certificate_path, first.input_path;
                expected_input_sha256=first.input_sha256,
                expected_certificate_sha256=first.certificate_sha256,
                require=:root_and_spectrum, precision_bits=256)
            @test report.status == :VERIFIED

            unsafe_writer = (repository="file:///fixture-private/source-location",
                commit=source.commit, tree=source.tree)
            @test_throws ArgumentError SC.write_certificate(joinpath(parent,"unsafe-writer"),
                model,result;source_ref=unsafe_writer,environment_ref=environment)
            @test !ispath(joinpath(parent,"unsafe-writer"))

            unsafe_document=TOML.parsefile(first.certificate_path)
            unsafe_document["source_ref"]["repository"]="session_id=fixture-only"
            unsafe_path=joinpath(parent,"unsafe-source-ref-mutant.toml")
            write(unsafe_path,SC._toml_bytes(unsafe_document))
            unsafe_digest=bytes2hex(sha256(read(unsafe_path)))
            unsafe_report=SC.verify_certificate(unsafe_path,first.input_path;
                expected_input_sha256=first.input_sha256,
                expected_certificate_sha256=unsafe_digest,
                require=:root_and_spectrum,precision_bits=256)
            @test unsafe_report.status==:INVALID
            @test occursin("certificate schema invalid",unsafe_report.reason)
        end
    end

    @testset "unsupported and degenerate claims do not certify" begin
        options = SC.ProofOptions(precisions=(256,))
        zero_model = SC.prepare_model(reshape([1, 0], 2, 1), reshape([1, 0], 2, 1), Matrix{Int}(I, 2, 2))
        result = SC.certify_stationary_point(zero_model, [0, 0]; options)
        @test result.status == :INCONCLUSIVE

        flat_model = SC.prepare_model(reshape([1 1], 1, 2), [1 -1; 0 0], reshape([1], 1, 1))
        flat = SC.certify_stationary_point(flat_model, [0]; options)
        @test flat.status != :CERTIFIED_ROOT

    simple_model = SC.prepare_model(reshape([1],1,1), reshape([1,0],2,1), reshape([1],1,1))
    simple_root = SC.certify_stationary_point(simple_model,[0];options)
    exhausted = SC.certify_generalized_spectrum(simple_model,simple_root;
        options=SC.ProofOptions(precisions=(256,),max_inertia_evaluations=0))
    @test exhausted.status == :INCONCLUSIVE
    @test exhausted.reason == "budget_exhausted"
    end

    @testset "T05, T06, T07, T08, T09, T12, T16, T17, and T18 controls" begin
        options = SC.ProofOptions(precisions=(256,), spectral_goal=:individuals)
        Q5 = hcat(Matrix{Int}(I, 5, 5),
            [1 0 0 0; 1 1 0 0; 0 1 1 0; 0 0 1 1; 0 0 0 1])
        model5 = SC.prepare_model(Q5, [ones(Int, 9)'; zeros(Int, 9)'], diagm(1:5))
        result5 = SC.certify(model5, zeros(5); options)
        @test result5.status == :CERTIFIED
        @test result5.root.hessian_classification == :MINIMUM
        @test length(result5.spectrum.bounds) == 5
        @test result5.spectrum.positive_count == 5
        @test any(!iszero(result5.root.H_box[i,j]) for i in 1:5 for j in 1:5 if i != j)

        hierarchy = SC.prepare_model(Matrix{Int}(I, 2, 2),
            [1 1; 0 -400], Matrix{Int}(I, 2, 2))
        tiny = SC.certify(hierarchy, [0,0]; options)
        @test tiny.status == :CERTIFIED
        @test tiny.spectrum.bounds[1][1] > 0
        @test tiny.spectrum.bounds[1][1] < tiny.spectrum.bounds[2][1]
        fractional = SC.prepare_model(Matrix{Int}(I, 2, 2),
            [1 1; 0 -1//2], Matrix{Int}(I, 2, 2))
        frac = SC.certify(fractional, [0,0]; options)
        @test frac.status == :CERTIFIED
        @test frac.spectrum.resolution_met

        deficient = SC.prepare_model(reshape([1,0],2,1), reshape([1,0],2,1), Matrix{Int}(I,2,2))
        @test SC.certify_stationary_point(deficient,[0,0];options).status == :INCONCLUSIVE

        zero_second_coefficient = SC.prepare_model(Matrix{Int}(I,2,2),
            [1 0; 0 0], Matrix{Int}(I,2,2))
        @test zero_second_coefficient.Q==Matrix{Rational{BigInt}}(I,2,2)
        @test zero_second_coefficient.L[1,2]==0
        zero_second_result = SC.certify_stationary_point(zero_second_coefficient,[0,0];options)
        @test zero_second_result.status == :INCONCLUSIVE
        @test zero_second_result.hessian_classification == :UNRESOLVED
        @test zero_second_result.status != :CERTIFIED_ROOT

        badmetric = SC.prepare_model(Matrix{Int}(I,2,2),[1 1;0 0],[1 0;0 -1])
        badroot = SC.certify_stationary_point(badmetric,[0,0];options)
        badspec = SC.certify_generalized_spectrum(badmetric,badroot;options)
        @test badspec.status == :INVALID_INPUT
        semidefinite = SC.prepare_model(Matrix{Int}(I,2,2),[1 1;0 0],[1 0;0 0])
        semiroot = SC.certify_stationary_point(semidefinite,[0,0];options)
        @test SC.certify_generalized_spectrum(semidefinite,semiroot;options).status == :INVALID_INPUT
        @test_throws ArgumentError SC.prepare_model(Matrix{Int}(I,2,2),[1 1;0 0],[1 1;0 1])

        seed0 = SC.certify_stationary_point(SC.prepare_model(reshape([1],1,1),
            reshape([1,0],2,1),reshape([1],1,1)),[0];options)
        seed1 = SC.certify_stationary_point(SC.prepare_model(reshape([1],1,1),
            reshape([1,0],2,1),reshape([1],1,1)),[1];options)
        @test seed0.normalized_seed == seed1.normalized_seed
        @test seed0.center == seed1.center
        @test seed1.translation == seed0.translation .+ 1

        flat_model = SC.prepare_model(reshape([1 1],1,2),[1 -1;0 0],reshape([1],1,1))
        @test SC.certify_stationary_point(flat_model,[0];options).status != :CERTIFIED_ROOT

        phase_model = SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1);
            phases=[1//8])
        phase_result = SC.certify(phase_model,[0];options)
        @test phase_result.status == :CERTIFIED
        @test phase_result.root.hessian_classification == :MINIMUM
        @test abs(BigFloat(phase_result.root.center[1]) + 1/(16*BigFloat(pi))) < big"1e-60"
        signed_model = SC.prepare_model(reshape([-2],1,1),reshape([1,0],2,1),reshape([1],1,1);
            phases=[1//8])
        signed_result = SC.certify(signed_model,[0];options)
        @test signed_result.status == :CERTIFIED
        @test signed_result.root.hessian_classification == :MINIMUM
        @test abs(BigFloat(signed_result.root.center[1]) - 1/(32*BigFloat(pi))) < big"1e-60"

        clustered = SC.certify(SC.prepare_model(Matrix{Int}(I,2,2),[1 1;0 0],Matrix{Int}(I,2,2)),[0,0];
            options=SC.ProofOptions(precisions=(256,),spectral_goal=:clusters))
        @test clustered.status == :CERTIFIED
        @test length(clustered.spectrum.clusters) == 1
        @test clustered.spectrum.clusters[1].population == 2

        SC._with_precision(256) do
            A = SC._interval_matrix(BigInt[0 1;1 0] .// BigInt(1),256)
            K = SC._interval_matrix(Matrix{Rational{BigInt}}(I,2,2),256)
            metric,_ = SC._metric_ldl(SC.prepare_model(Matrix{Int}(I,2,2),[1 1;0 0],Matrix{Int}(I,2,2)),256)
            budget = SC._InertiaBudget(4,0,String[])
            @test SC._inertia_at(A,K,BigInt(0)//BigInt(1),256,budget) === nothing
            @test SC._inertia_at(A,K,BigInt(-2)//BigInt(1),256,budget).negative == 0
            @test SC._inertia_at(A,K,BigInt(2)//BigInt(1),256,budget).negative == 2
            @test metric.ok
        end
    end
end

@testset "T10–T13 submitted-claim replay controls" begin
    options256 = SC.ProofOptions(precisions=(256,), spectral_goal=:individuals)
    model1 = SC.prepare_model(reshape([1],1,1), reshape([1,0],2,1), reshape([1],1,1))
    result1 = SC.certify(model1,[0]; options=options256)
    model4 = SC.prepare_model(Matrix{Int}(I,2,2),[1 -1; 0 0],[2 1;1 2])
    result4 = SC.certify(model4,[0,0]; options=options256)
    @test result1.status == :CERTIFIED
    @test result4.status == :CERTIFIED

    repository_root = normpath(joinpath(@__DIR__, ".."))
    (; commit, tree) = head_identity(repository_root)
    source = (repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",commit,tree)
    mktempdir() do parent
        project=Base.active_project()
        manifest=joinpath(dirname(project),"Manifest.toml")
        envpath=joinpath(parent,"producer-environment.toml")
        envbytes=Vector{UInt8}(codeunits(join((
            "schema=cyax-scientific-test-environment-v1",
            "julia_version=$(VERSION)",
            "interval_arithmetic_version=$(Base.pkgversion(SC.IA))",
            "project_sha256=$(bytes2hex(sha256(read(project))))",
            "manifest_sha256=$(bytes2hex(sha256(read(manifest))))"),"\n")*"\n"))
        write(envpath,envbytes)
        environment=(receipt_sha256=bytes2hex(sha256(envbytes)),artifact_name=basename(envpath))
        one=SC.write_certificate(joinpath(parent,"one"),model1,result1;
            source_ref=source,environment_ref=environment)
        four=SC.write_certificate(joinpath(parent,"four"),model4,result4;
            source_ref=source,environment_ref=environment)
        verified_cli=_run_scientific_cli(repository_root,one.input_path,
            one.certificate_path,one.input_sha256,one.certificate_sha256)
        @test verified_cli.exit_code==0
        @test occursin("status=VERIFIED",verified_cli.output)
        one_doc=TOML.parsefile(one.certificate_path)
        four_doc=TOML.parsefile(four.certificate_path)
        function replay_mutant(name,doc; expected, source_cert=one, precision=256)
            path=joinpath(parent,"$name.toml")
            write(path,SC._toml_bytes(doc))
            report=SC.verify_certificate(path,source_cert.input_path;
                expected_input_sha256=source_cert.input_sha256,
                expected_certificate_sha256=bytes2hex(sha256(read(path))),
                require=:root_and_spectrum,precision_bits=precision)
            @test report.status == expected
            report
        end
        function replay_cli_mutant(name,source_cert; expected_status,expected_exit,precision=256)
            path=joinpath(parent,"$name.toml")
            logpath=joinpath(parent,"$name-cli.log")
            receipt=_run_scientific_cli(repository_root,source_cert.input_path,path,
                source_cert.input_sha256,bytes2hex(sha256(read(path)));precision)
            write(logpath,receipt.output)
            @test receipt.exit_code==expected_exit
            @test occursin("status=$expected_status",receipt.output)
            receipt.exit_code
        end

        bad_integrity=copy(read(one.certificate_path)); bad_integrity[end-1] ⊻= 0x01
        integrity_path=joinpath(parent,"integrity-mismatch.toml"); write(integrity_path,bad_integrity)
        integrity=SC.verify_certificate(integrity_path,one.input_path;
            expected_input_sha256=one.input_sha256,
            expected_certificate_sha256=one.certificate_sha256,
            require=:root_and_spectrum,precision_bits=256)
        @test integrity.status == :INVALID

        q_mutation=deepcopy(one_doc); q_mutation["root"]["q_upper"]="0/1"
        @test replay_mutant("understated-q",q_mutation;expected=:UNPROVEN).status != :VERIFIED

        outside=deepcopy(one_doc); outside["root"]["center"]["data"][1]="1/1"
        @test replay_mutant("center-outside-box",outside;expected=:INVALID).status == :INVALID

        shifted=deepcopy(one_doc)
        shifted["root"]["center"]["data"][1]="1/8"
        shifted["root"]["box"]["data"][1]=["1/16","3/16"]
        shifted["root"]["W"]["data"][1]=["1/8","1/8"]
        @test replay_mutant("box-excludes-root",shifted;expected=:UNPROVEN).status == :UNPROVEN

        zero_c=deepcopy(one_doc)
        zero_c["root"]["preconditioner"]["data"]=["0/1"]
        @test replay_mutant("zero-preconditioner",zero_c;expected=:UNPROVEN).status == :UNPROVEN

        narrow_hessian=deepcopy(one_doc)
        pivot=narrow_hessian["hessian"]["pivots"]["data"][1]
        pivot_mid=(SC._parse_rational(pivot[1])+SC._parse_rational(pivot[2]))/2
        narrow_hessian["hessian"]["pivots"]["data"][1]=[SC._ratstr(pivot_mid),SC._ratstr(pivot_mid)]
        @test replay_mutant("narrow-hessian",narrow_hessian;expected=:UNPROVEN).status == :UNPROVEN

        # Exercise the submitted H(X) enclosure itself, separately from the
        # derived LDL-pivot claim above. The mutant keeps the schema valid and
        # the helper binds its updated certificate digest to the original input.
        narrow_root_hessian=deepcopy(one_doc)
        root_hessian_box=narrow_root_hessian["root"]["H_box"]["data"][1]
        root_hessian_lower,root_hessian_upper=SC._parse_rational.(root_hessian_box)
        root_hessian_mid=(root_hessian_lower+root_hessian_upper)/2
        narrow_root_hessian["root"]["H_box"]["data"][1]=
            SC._ratstr.([root_hessian_mid,root_hessian_mid])
        root_hessian_report=replay_mutant("narrow-root-hessian-box",narrow_root_hessian;
            expected=:UNPROVEN)
        @test root_hessian_report.reason == "H(X) exceeds submitted claim"

        false_spectrum=deepcopy(one_doc)
        oldlo,oldhi=(SC._parse_rational(x) for x in false_spectrum["spectrum"]["bounds"]["data"][1])
        newpair=[SC._ratstr(oldlo+1),SC._ratstr(oldhi+1)]
        false_spectrum["spectrum"]["bounds"]["data"][1]=newpair
        false_spectrum["spectrum"]["diagonal_ratios"]["data"][1]=copy(newpair)
        false_spectrum["spectrum"]["clusters"][1]["lower"]=newpair[1]
        false_spectrum["spectrum"]["clusters"][1]["upper"]=newpair[2]
        @test replay_mutant("false-eigenvalue",false_spectrum;expected=:UNPROVEN).status == :UNPROVEN

        weakening=deepcopy(one_doc)
        weakening["spectrum"]["bounds"]["data"][1]=["1/1","100/1"]
        weakening["spectrum"]["diagonal_ratios"]["data"][1]=["1/1","100/1"]
        weakening["spectrum"]["clusters"][1]["lower"]="1/1"
        weakening["spectrum"]["clusters"][1]["upper"]="100/1"
        weakening["spectrum"]["clusters"][1]["relative_width_met"]=false
        weakening["spectrum"]["resolution_met"]=false
        weakening["spectrum"]["reason"]="refinement_inconclusive"
        weak_report=replay_mutant("honest-widening",weakening;expected=:VERIFIED)
        @test weak_report.status == :VERIFIED
        @test !weakening["spectrum"]["resolution_met"]

        twoprecision=SC.certify(model1,[0];options=SC.ProofOptions(precisions=(512,),spectral_goal=:individuals))
        @test twoprecision.status == :CERTIFIED
        high=SC.write_certificate(joinpath(parent,"high"),model1,twoprecision;
            source_ref=source,environment_ref=environment)
        lower_precision=SC.verify_certificate(high.certificate_path,high.input_path;
            expected_input_sha256=high.input_sha256,
            expected_certificate_sha256=high.certificate_sha256,
            require=:root_and_spectrum,precision_bits=256)
        @test lower_precision.status == :UNPROVEN

        false_count=deepcopy(four_doc)
        false_count["spectrum"]["endpoint_counts"][1]["negative"] += 1
        @test replay_mutant("false-inertia-count",false_count;expected=:UNPROVEN,source_cert=four).status == :UNPROVEN

        relabeled=deepcopy(four_doc)
        relabeled["hessian"]["classification"]="MINIMUM"
        relabeled["hessian"]["method"]="spectral_counts"
        relabeled["hessian"]["negative_count"]=0
        relabeled["hessian"]["positive_count"]=2
        relabeled["hessian"]["unresolved_count"]=0
        relabeled["hessian"]["pivots"]=Dict("dims"=>[0],"data"=>Any[])
        @test replay_mutant("saddle-called-minimum",relabeled;expected=:UNPROVEN,source_cert=four).status == :UNPROVEN

        unused=deepcopy(four_doc)
        spare=deepcopy(unused["spectrum"]["endpoint_counts"][1])
        spare["trial"]="0/1"
        push!(unused["spectrum"]["endpoint_counts"],spare)
        @test replay_mutant("unused-endpoint",unused;expected=:INVALID,source_cert=four).status == :INVALID

        wrong_interpretation=deepcopy(one_doc)
        wrong_interpretation["interpretation"]["physical_normalization"]="established"
        @test replay_mutant("wrong-interpretation",wrong_interpretation;expected=:INVALID).status == :INVALID

        malformed=copy(read(one.input_path)); malformed[end-1] ⊻= 0x01
        bad_input=joinpath(parent,"bad-input.toml"); write(bad_input,malformed)
        bad_input_report=SC.verify_certificate(one.certificate_path,bad_input;
            expected_input_sha256=one.input_sha256,
            expected_certificate_sha256=one.certificate_sha256,
            require=:root_and_spectrum,precision_bits=256)
        @test bad_input_report.status == :INVALID

        malformed_dims=deepcopy(one_doc)
        malformed_dims["root"]["W"]["dims"]=[2]
        @test replay_mutant("malformed-array-dimensions",malformed_dims;expected=:INVALID).status == :INVALID

        for (name,record) in (("boolean-center-dimensions", "center"),
                              ("boolean-interval-dimensions", "W"))
            boolean_dims=deepcopy(one_doc)
            boolean_dims["root"][record]["dims"]=[true]
            @test replay_mutant(name,boolean_dims;expected=:INVALID).status==:INVALID
            @test replay_cli_mutant(name,one;expected_status="INVALID",expected_exit=2)==2
        end

        unknown_key=deepcopy(one_doc)
        unknown_key["unexpected_claim"]="ignored?"
        @test replay_mutant("unknown-certificate-key",unknown_key;expected=:INVALID).status == :INVALID

        direct_relabel=deepcopy(four_doc)
        direct_relabel["hessian"]["classification"]="MINIMUM"
        @test replay_mutant("saddle-called-minimum-direct",direct_relabel;
            expected=:UNPROVEN,source_cert=four).status == :UNPROVEN

        false_endpoint=deepcopy(four_doc)
        spectrum=false_endpoint["spectrum"]
        first_cluster=spectrum["clusters"][1]
        second_cluster=spectrum["clusters"][2]
        new_lower="100/1"
        new_upper_first="429496729601/4294967296"
        new_middle="214748364801/2147483648"
        new_upper="107374182401/1073741824"
        old_endpoints=(first_cluster["lower"],first_cluster["upper"],
            second_cluster["lower"],second_cluster["upper"])
        first_cluster["lower"]=new_lower
        first_cluster["upper"]=new_upper_first
        second_cluster["lower"]=new_middle
        second_cluster["upper"]=new_upper
        first_cluster["sign"]="positive"
        second_cluster["sign"]="positive"
        spectrum["negative_count"]=0
        spectrum["positive_count"]=2
        spectrum["unresolved_count"]=0
        for i in first_cluster["first"]:first_cluster["last"]
            spectrum["bounds"]["data"][i]=[new_lower,new_upper_first]
        end
        for i in second_cluster["first"]:second_cluster["last"]
            spectrum["bounds"]["data"][i]=[new_middle,new_upper]
        end
        for endpoint in spectrum["endpoint_counts"]
            endpoint["trial"]==old_endpoints[1] && (endpoint["trial"]=new_lower)
            endpoint["trial"]==old_endpoints[2] && (endpoint["trial"]=new_upper_first)
            endpoint["trial"]==old_endpoints[3] && (endpoint["trial"]=new_middle)
            endpoint["trial"]==old_endpoints[4] && (endpoint["trial"]=new_upper)
        end
        endpoint_report=replay_mutant("false-inertia-endpoint",false_endpoint;
            expected=:UNPROVEN,source_cert=four)
        @test endpoint_report.reason == "submitted generalized spectrum did not revalidate"

        outer_endpoints=four_doc["spectrum"]["endpoint_counts"]
        broad_lower=four_doc["spectrum"]["clusters"][1]["lower"]
        broad_upper=four_doc["spectrum"]["clusters"][end]["upper"]
        broad_cluster=deepcopy(four_doc)
        broad_cluster["spectrum"]["bounds"]["data"]=[
            [broad_lower,broad_upper], [broad_lower,broad_upper]]
        broad_cluster["spectrum"]["clusters"]=[Dict(
            "first"=>1,"last"=>2,"population"=>2,
            "lower"=>broad_lower,"upper"=>broad_upper,
            "sign"=>"unresolved","relative_width_met"=>false)]
        broad_cluster["spectrum"]["endpoint_counts"]=[first(outer_endpoints),last(outer_endpoints)]
        broad_cluster["spectrum"]["negative_count"]=0
        broad_cluster["spectrum"]["positive_count"]=0
        broad_cluster["spectrum"]["unresolved_count"]=2
        broad_cluster["spectrum"]["resolution_met"]=false
        broad_cluster["spectrum"]["reason"]="sign_unresolved"
        broad_path=joinpath(parent,"honest-broad-inertia-cluster.toml")
        write(broad_path,SC._toml_bytes(broad_cluster))
        broad_digest=bytes2hex(sha256(read(broad_path)))
        broad_report=SC.verify_certificate(broad_path,four.input_path;
            expected_input_sha256=four.input_sha256,
            expected_certificate_sha256=broad_digest,require=:root_and_spectrum,
            precision_bits=256)
        @test broad_report.status==:VERIFIED

        unsupported_narrow_mode=deepcopy(broad_cluster)
        unsupported_narrow_mode["spectrum"]["bounds"]["data"][1]=
            copy(four_doc["spectrum"]["bounds"]["data"][1])
        narrow_report=replay_mutant("narrow-mode-inside-broad-inertia-cluster",
            unsupported_narrow_mode;expected=:INVALID,source_cert=four)
        @test occursin("mode bound differs from cluster",narrow_report.reason)
    end
end

include(joinpath(@__DIR__, "fixtures", "scientific_certification", "run_fixture.jl"))

@testset "fixture qualification children retain detached coverage output" begin
    repository_root=normpath(joinpath(@__DIR__,".."))
    options=Base.JLOptions()
    argv=ScientificCertificationFixtures._julia_argv(repository_root,
        ScientificCertificationFixtures.VERIFIER_PATH)
    coverage_flags=filter(arg->startswith(arg,"--code-coverage"),argv)
    if options.code_coverage!=0 && options.output_code_coverage==C_NULL
        @test count(arg->startswith(arg,"--code-coverage=@"),coverage_flags)>=1
        trace_flag=last(coverage_flags)
        @test startswith(trace_flag,"--code-coverage=")
        trace_path=split(trace_flag,'=';limit=2)[2]
        @test endswith(trace_path,".info")
        @test startswith(relpath(trace_path,repository_root),"..")
        @test !ispath(trace_path)
    else
        @test isempty(child_coverage_args())
    end
end

@testset "T15 oracle envelope uses exact frozen types" begin
    model=SC.prepare_model(BigInt[1 0 0 1 0;0 1 0 1 1;0 0 1 0 1],
        BigInt[1 1 1 1 1;0 0 0 0 0],BigInt[1 1 0;1 3 0;0 0 3])
    A=Matrix{Rational{BigInt}}(model.Q*Diagonal(model.L[1,:])*transpose(model.Q))
    polynomial=ScientificCertificationFixtures.oracle_exact_pencil_polynomial(A,model.K)
    sequence=ScientificCertificationFixtures._sturm_sequence(polynomial)
    isolators=Tuple{Rational{BigInt},Rational{BigInt}}[]
    for k in 0:102400
        lo=BigInt(k)//BigInt(1024)
        hi=BigInt(k+1)//BigInt(1024)
        if ScientificCertificationFixtures._sturm_count(sequence,lo,hi)==1
            push!(isolators,(lo,hi))
            length(isolators)==3 && break
        end
    end
    @test length(isolators)==3
    seed=Rational{BigInt}[1//256,1//512,-1//1024]
    input_record=Dict("case"=>"T15","model"=>SC._input_record(model),
        "seed"=>SC._array_record(seed))
    input_bytes=SC._toml_bytes(input_record)
    oracle=Dict("case"=>"T15","input_sha256"=>bytes2hex(sha256(input_bytes)),
        "pencil_coefficients"=>String[SC._ratstr(x) for x in polynomial],
        "mu_isolators"=>Dict("dims"=>[3],"data"=>Any[Any[SC._ratstr(lo),SC._ratstr(hi)]
            for (lo,hi) in isolators]),"expected_class"=>"MINIMUM",
        "spectral_goal"=>"individuals","relative_width"=>"1/1048576",
        "precisions"=>[256,512])

    mktempdir() do parent
        input_path=joinpath(parent,"synthetic-input.toml")
        oracle_path=joinpath(parent,"synthetic-oracle.toml")
        write(input_path,input_bytes)
        write(oracle_path,SC._toml_bytes(oracle))
        valid=ScientificCertificationFixtures._held_out_data(input_path,oracle_path)
        @test valid.relative_width==big(1)//big(1048576)

        function envelope_rejected(name,mutate!; message=nothing)
            candidate=deepcopy(oracle)
            mutate!(candidate)
            path=joinpath(parent,name*".toml")
            write(path,SC._toml_bytes(candidate))
            err=try
                ScientificCertificationFixtures._held_out_data(input_path,path)
                nothing
            catch exception
                exception
            end
            @test err isa Exception
            message===nothing || @test occursin(message,sprint(showerror,err))
        end
        envelope_rejected("float-width",item->(item["relative_width"]=1.0/1048576);
            message="quoted rational string")
        envelope_rejected("unreduced-width",item->(item["relative_width"]="2/2097152");
            message="rational must be reduced")
        envelope_rejected("float-precisions",item->(item["precisions"]=[256.0,512.0]);
            message="integer array")
        envelope_rejected("bool-precision",item->(item["precisions"]=Any[256,true]);
            message="integer array")
        envelope_rejected("short-precisions",item->(item["precisions"]=[256]);
            message="integer array")
        envelope_rejected("reversed-precisions",item->(item["precisions"]=[512,256]);
            message="precision schedule")
    end
end

@testset "P7 correction: scheduling, replay, resources and independent references" begin
    @testset "one proposal and one shared spectral budget per precision" begin
        model=SC.prepare_model(Matrix{Int}(I,2,2),[1 1;0 0],[2 1;1 2])
        radii=(big(1)//big(2)^20,big(1)//big(2)^30,
            big(1)//big(2)^40,big(1)//big(2)^60)
        options=SC.ProofOptions(precisions=(256,),max_newton_steps=1,
            box_radii=radii,max_expansions=0,max_inertia_evaluations=3,
            spectral_goal=:individuals)
        combined=SC.certify(model,[0,0];options)
        @test count(x->startswith(x,"proposal p=256 "),combined.attempts)==1
        @test combined.root.options.precisions==options.precisions
        @test combined.root.options.box_radii==options.box_radii
        @test count(x->occursin("spectrum budget p=256 used=3 limit=3",x),
            combined.attempts)==length(radii)
        @test count(x->startswith(x,"metric LDL status=attempted"),
            combined.spectrum.attempts)==1
        @test count(x->startswith(x,"root H LDL evaluations=1"),combined.attempts)==
            count(x->startswith(x,"root p=256 radius=") && occursin("status=",x),combined.attempts)
        @test count(x->startswith(x,"trial="),combined.spectrum.attempts)==2
        @test combined.spectrum.status==:CERTIFIED_BOUNDS
        @test !combined.spectrum.resolution_met
    end

    @testset "bounded inertia midpoint and quarter fallback" begin
        function run_inertia(H,K=Matrix{Rational{BigInt}}(I,2,2),limit=20)
            SC._with_precision(256) do
                h=SC._interval_matrix(H .// BigInt(1),256)
                k=SC._interval_matrix(K,256)
                options=SC.ProofOptions(precisions=(256,),max_expansions=1,
                    max_inertia_evaluations=limit,spectral_goal=:individuals)
                budget=SC._InertiaBudget(options.max_inertia_evaluations,0,String[])
                spectrum=SC._inertia_spectrum(h,k,256,options,
                    fill(SC._iv(1,256),2),budget)
                (spectrum=spectrum,budget=budget)
            end
        end

        contracted=run_inertia(BigInt[0 1;1 0])
        trials=[match(r"trial=([^ ]+)",x).captures[1] for x in contracted.budget.attempts
            if startswith(x,"trial=")]
        @test trials[end-2:end]==["0//1","-1//1","1//1"]
        @test length(contracted.spectrum.clusters)==1
        @test only(contracted.spectrum.clusters).population==2
        @test [e.trial for e in contracted.spectrum.endpoint_counts]==[-2//1,2//1]

        quarter=run_inertia(BigInt[-1 0;0 0],Rational{BigInt}[1//1 0//1;0//1 1//2],5)
        quarter_trials=[match(r"trial=([^ ]+)",x).captures[1] for x in quarter.budget.attempts
            if startswith(x,"trial=")]
        queue_start=findfirst(==("0//1"),quarter_trials)
        @test quarter_trials[queue_start:queue_start+2]==["0//1","-1//1","1//1"]
        @test [e.trial for e in quarter.spectrum.endpoint_counts]==[-2//1,1//1]
        @test [e.negative for e in quarter.spectrum.endpoint_counts]==[0,2]
        @test !(-1//1 in [e.trial for e in quarter.spectrum.endpoint_counts])

        stalled=run_inertia(BigInt[0 1;1 0],Matrix{Rational{BigInt}}(I,2,2),20)
        @test stalled.spectrum.status==:CERTIFIED_BOUNDS
        @test !stalled.spectrum.resolution_met
        @test stalled.spectrum.reason=="refinement_inconclusive"
        @test stalled.budget.used==7 && stalled.budget.used<stalled.budget.limit
    end

    @testset "diagonal overlap bounds and honest weakening replay" begin
        close_model=SC.prepare_model(Matrix{Int}(I,2,2),
            [1 1+1//2^50;0 0],Matrix{Int}(I,2,2))
        close=SC.certify(close_model,[0,0];
            options=SC.ProofOptions(precisions=(256,),spectral_goal=:clusters))
        @test close.status==:CERTIFIED
        @test length(close.spectrum.clusters)==1
        @test close.spectrum.bounds[1]!=close.spectrum.bounds[2]
        @test close.spectrum.bounds[1][1]==close.spectrum.clusters[1].lower
        @test close.spectrum.bounds[2][2]==close.spectrum.clusters[1].upper

        equal_model=SC.prepare_model(Matrix{Int}(I,2,2),[1 1;0 0],Matrix{Int}(I,2,2))
        equal=SC.certify(equal_model,[0,0];
            options=SC.ProofOptions(precisions=(256,),spectral_goal=:clusters))
        repository_root=normpath(joinpath(@__DIR__,".."))
        (; commit, tree) = head_identity(repository_root)
        source=(repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",commit,tree)
        mktempdir() do parent
            project=Base.active_project()
            manifest=joinpath(dirname(project),"Manifest.toml")
            envpath=joinpath(parent,"producer-environment.toml")
            envbytes=Vector{UInt8}(codeunits(join(("schema=cyax-scientific-test-environment-v1",
                "julia_version=$(VERSION)","interval_arithmetic_version=$(Base.pkgversion(SC.IA))",
                "project_sha256=$(bytes2hex(sha256(read(project))))",
                "manifest_sha256=$(bytes2hex(sha256(read(manifest))))"),"\n")*"\n"))
            write(envpath,envbytes)
            environment=(receipt_sha256=bytes2hex(sha256(envbytes)),artifact_name=basename(envpath))
            written=SC.write_certificate(joinpath(parent,"equal"),equal_model,equal;
                source_ref=source,environment_ref=environment)
            document=TOML.parsefile(written.certificate_path)
            document["spectrum"]["bounds"]["data"]=[["1/1","100/1"],["2/1","101/1"]]
            document["spectrum"]["diagonal_ratios"]["data"]=[["1/1","100/1"],["2/1","101/1"]]
            document["spectrum"]["clusters"][1]["lower"]="1/1"
            document["spectrum"]["clusters"][1]["upper"]="101/1"
            document["spectrum"]["clusters"][1]["relative_width_met"]=false
            document["spectrum"]["resolution_met"]=false
            document["spectrum"]["reason"]="refinement_inconclusive"
            path=joinpath(parent,"honest-diagonal-widening.toml")
            write(path,SC._toml_bytes(document))
            digest=bytes2hex(sha256(read(path)))
            report=SC.verify_certificate(path,written.input_path;
                expected_input_sha256=written.input_sha256,
                expected_certificate_sha256=digest,require=:root_and_spectrum,
                precision_bits=256)
            @test report.status==:VERIFIED
            @test !document["spectrum"]["resolution_met"]
        end
    end

    @testset "typed resource limits preserve invalid schema outcomes" begin
        large_n_model=SC.prepare_model(ones(Int,33,1),
            reshape([1,0],2,1),Matrix{Int}(I,33,33))
        large_n_result=SC.certify(large_n_model,zeros(33))
        @test large_n_model.resource_reason=="RESOURCE_LIMIT"
        @test isempty(SC._amplitudes(large_n_model,256))
        @test large_n_result.status==:UNSUPPORTED
        @test large_n_result.root.reason=="RESOURCE_LIMIT"
        @test large_n_result.identity.input_identity_status==:UNAVAILABLE
        large_m_model=SC.prepare_model(ones(Int,1,2049),
            zeros(Int,2,2049),reshape([1],1,1))
        large_m_result=SC.certify(large_m_model,[0])
        @test large_m_model.resource_reason=="RESOURCE_LIMIT"
        @test large_m_result.status==:UNSUPPORTED
        @test large_m_result.root.reason=="RESOURCE_LIMIT"
        @test_throws DimensionMismatch SC.prepare_model(ones(Int,33,1),
            zeros(Int,2,2),Matrix{Int}(I,33,33))
        asymmetric_large_metric=Matrix{Int}(I,33,33)
        asymmetric_large_metric[1,2]=1
        @test_throws ArgumentError SC.prepare_model(ones(Int,33,1),
            reshape([1,0],2,1),asymmetric_large_metric)
        for (name,shape) in (("zero-n",(0,1)),("zero-M",(1,0)))
            record=SC._input_record(SC.prepare_model(reshape([1],1,1),
                reshape([1,0],2,1),reshape([1],1,1)))
            if name=="zero-n"
                record["n"]=0
                record["Q"]=Dict("dims"=>[0,1],"data"=>String[])
                record["K"]=Dict("dims"=>[0,0],"data"=>String[])
            else
                record["M"]=0
                record["Q"]=Dict("dims"=>[1,0],"data"=>String[])
                record["L"]=Dict("dims"=>[2,0],"data"=>String[])
                record["phi"]=Dict("dims"=>[0],"data"=>String[])
            end
            @test_throws ArgumentError SC._parse_input(record)
        end
        big_q=Matrix{Rational{BigInt}}(I,33,33)
        malformed_big_model=SC.CertificationModel(big_q,reshape(Rational{BigInt}[1//1,0//1],2,1),
            Matrix{Rational{BigInt}}(I,33,33),zeros(Rational{BigInt},1),BigInt(0)//BigInt(1))
        @test_throws DimensionMismatch SC.certify(malformed_big_model,zeros(33);
            options=SC.ProofOptions(precisions=(256,)))
        big_model=SC.CertificationModel(big_q,
            repeat(reshape(Rational{BigInt}[1//1,0//1],2,1),1,33),
            Matrix{Rational{BigInt}}(I,33,33),zeros(Rational{BigInt},33),BigInt(0)//BigInt(1))
        result=SC.certify(big_model,zeros(33);options=SC.ProofOptions(precisions=(256,)))
        @test result.status==:UNSUPPORTED
        @test result.root.reason=="RESOURCE_LIMIT"
        @test_throws SC.ResourceLimitError SC._parse_rational(repeat("1",4097)*"/1")
        negative_4096=-(big(10)^4095)//big(1)
        negative_4096_text=SC._ratstr(negative_4096)
        @test SC._parse_rational(negative_4096_text)==negative_4096
        @test SC._check_tree_limits(Dict("Q"=>Dict("dims"=>[1,1],"data"=>[negative_4096_text])))===nothing
        @test_throws SC.ResourceLimitError SC._parse_rational("-"*repeat("9",4097)*"/1")
        long_metadata=repeat("x",4097)
        @test_throws SC.ResourceLimitError SC._check_tree_limits(Dict("metadata"=>long_metadata))
        nested=Any[]
        for _ in 1:13
            nested=[nested]
        end
        @test_throws SC.ResourceLimitError SC._check_tree_limits(nested)
        serialized_deep="value = "*repeat("[",13)*"0"*repeat("]",13)
        @test_throws SC.ResourceLimitError SC._preflight_toml_depth(serialized_deep)
        @test_throws SC.ResourceLimitError SC._parse_toml(Vector{UInt8}(codeunits(serialized_deep)))
        incomplete_deep="value = "*repeat("[",13)*"0\n"
        @test_throws SC.ResourceLimitError SC._preflight_toml_depth(incomplete_deep)
        @test_throws SC.ResourceLimitError SC._parse_toml(Vector{UInt8}(codeunits(incomplete_deep)))
        incomplete_inline_deep="value = "*repeat("{k = ",13)*"0\n"
        @test_throws SC.ResourceLimitError SC._preflight_toml_depth(incomplete_inline_deep)
        @test_throws SC.ResourceLimitError SC._parse_toml(Vector{UInt8}(codeunits(incomplete_inline_deep)))
        @test SC._preflight_toml_depth("value = [[0]]\n")===nothing
        bracket_metadata="quoted = \"[[[[[[[[[[[[\" # [[[[[[[[[[[[\nmultiline = \"\"\"\n{ [[ ]]}\n\"\"\"\ninline = { child = [1] }\n"
        @test SC._parse_toml(Vector{UInt8}(codeunits(bracket_metadata)))["inline"]["child"]==[1]
        shallow_invalid="value = [0\n"
        @test_throws ArgumentError SC._parse_toml(Vector{UInt8}(codeunits(shallow_invalid)))
        deep_table="["*join(fill("level",13),".")*"]\nvalue = 0\n"
        @test_throws SC.ResourceLimitError SC._preflight_toml_depth(deep_table)
        array_table_subtable="[[fruit]]\n[fruit.physical]\ncolor = \"red\"\n"
        @test SC._preflight_toml_depth(array_table_subtable)===nothing
        array_table_depth_ok="[[root]]\n[root."*join(fill("level",10),".")*"]\n"
        @test SC._preflight_toml_depth(array_table_depth_ok)===nothing
        array_table_depth_over="[[root]]\n[root."*join(fill("level",11),".")*"]\n"
        @test_throws SC.ResourceLimitError SC._preflight_toml_depth(array_table_depth_over)
        quoted_array_table_depth_over="[[\"root\"]]\n[ \"root\" . "*
            join(fill("level",11),".")*"]\n"
        @test_throws SC.ResourceLimitError SC._preflight_toml_depth(quoted_array_table_depth_over)
        escaped_array_table_depth_over="[[\"r\\u006fot\"]]\n[root."*
            join(fill("level",11),".")*"]\n"
        @test_throws SC.ResourceLimitError SC._preflight_toml_depth(escaped_array_table_depth_over)
        dotted_assignment_depth_ok=join(fill("level",11),".")*" = 0\n"
        @test SC._preflight_toml_depth(dotted_assignment_depth_ok)===nothing
        @test SC._parse_toml(Vector{UInt8}(codeunits(dotted_assignment_depth_ok)))["level"] isa AbstractDict
        deep_dotted_assignment=join(fill("level",12),".")*" = 0\n"
        @test_throws SC.ResourceLimitError SC._preflight_toml_depth(deep_dotted_assignment)
        @test_throws SC.ResourceLimitError SC._parse_toml(Vector{UInt8}(codeunits(deep_dotted_assignment)))
        deep_inline="value = "*repeat("{k = ",11)*"0"*repeat("}",11)*"\n"
        @test SC._preflight_toml_depth(deep_inline)===nothing
        @test SC._parse_toml(Vector{UInt8}(codeunits(deep_inline)))["value"] isa AbstractDict
        mktempdir() do parser_root
            deep_input=joinpath(parser_root,"deep.toml")
            shallow_certificate=joinpath(parser_root,"certificate.toml")
            write(shallow_certificate,"schema = \"cyax-scientific-certificate-v1\"\n")
            certificate_digest=bytes2hex(sha256(read(shallow_certificate)))
            for text in (serialized_deep*"\n", incomplete_deep, incomplete_inline_deep)
                write(deep_input,text)
                input_digest=bytes2hex(sha256(read(deep_input)))
                report=SC.verify_certificate(shallow_certificate,deep_input;
                    expected_input_sha256=input_digest,
                    expected_certificate_sha256=certificate_digest)
                @test report.status==:UNSUPPORTED
                @test report.reason=="RESOURCE_LIMIT"
                cli=_run_scientific_cli(normpath(joinpath(@__DIR__,"..")),deep_input,
                    shallow_certificate,input_digest,certificate_digest)
                @test cli.exit_code==3
                @test occursin("status=UNSUPPORTED reason=RESOURCE_LIMIT",cli.output)
            end
            write(deep_input,shallow_invalid)
            shallow_digest=bytes2hex(sha256(read(deep_input)))
            shallow_report=SC.verify_certificate(shallow_certificate,deep_input;
                expected_input_sha256=shallow_digest,
                expected_certificate_sha256=certificate_digest)
            @test shallow_report.status==:INVALID
            shallow_cli=_run_scientific_cli(normpath(joinpath(@__DIR__,"..")),deep_input,
                shallow_certificate,shallow_digest,certificate_digest)
            @test shallow_cli.exit_code==2
            @test occursin("status=INVALID",shallow_cli.output)
        end
        @test_throws SC.ResourceLimitError SC._ratstr(BigInt(1)//BigInt(10)^4097)
        @test_throws SC.ResourceLimitError SC._toml_bytes(
            Dict("payload"=>repeat("x",16*1024*1024)))

        model=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
        cert=SC.certify(model,[0];options=SC.ProofOptions(precisions=(256,),spectral_goal=:individuals))
        repository_root=normpath(joinpath(@__DIR__,".."))
        (; commit, tree) = head_identity(repository_root)
        source=(repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",commit,tree)
        mktempdir() do parent
            project=Base.active_project(); manifest=joinpath(dirname(project),"Manifest.toml")
            envbytes=Vector{UInt8}(codeunits(join(("schema=cyax-scientific-test-environment-v1",
                "julia_version=$(VERSION)","interval_arithmetic_version=$(Base.pkgversion(SC.IA))",
                "project_sha256=$(bytes2hex(sha256(read(project))))",
                "manifest_sha256=$(bytes2hex(sha256(read(manifest))))"),"\n")*"\n"))
            envpath=joinpath(parent,"producer-environment.toml");write(envpath,envbytes)
            environment=(receipt_sha256=bytes2hex(sha256(envbytes)),artifact_name=basename(envpath))
            written=SC.write_certificate(joinpath(parent,"valid"),model,cert;
                source_ref=source,environment_ref=environment)
            for (name,dimension_key,array_records) in (
                    ("zero-n","n",Dict("Q"=>Dict("dims"=>[0,1],"data"=>String[]),
                        "K"=>Dict("dims"=>[0,0],"data"=>String[]))),
                    ("zero-M","M",Dict("Q"=>Dict("dims"=>[1,0],"data"=>String[]),
                        "L"=>Dict("dims"=>[2,0],"data"=>String[]),
                        "phi"=>Dict("dims"=>[0],"data"=>String[]))))
                input_doc=TOML.parsefile(written.input_path)
                input_doc[dimension_key]=0
                for (key,value) in array_records; input_doc[key]=value; end
                input_bytes=SC._toml_bytes(input_doc)
                input_path=joinpath(parent,"$name-input.toml");write(input_path,input_bytes)
                input_sha=bytes2hex(sha256(input_bytes))
                cert_doc=TOML.parsefile(written.certificate_path)
                cert_doc["input_sha256"]=input_sha
                cert_path=joinpath(parent,"$name-certificate.toml")
                write(cert_path,SC._toml_bytes(cert_doc))
                report=SC.verify_certificate(cert_path,input_path;
                    expected_input_sha256=input_sha,
                    expected_certificate_sha256=bytes2hex(sha256(read(cert_path))),
                    require=:root_and_spectrum,precision_bits=256)
                @test report.status==:INVALID
                @test occursin("input schema invalid",report.reason)
            end
            oversized_metadata=TOML.parsefile(written.certificate_path)
            oversized_metadata["candidate_provenance"]["source_ref"]=repeat("x",4097)
            metadata_path=joinpath(parent,"oversized-metadata.toml")
            open(metadata_path,"w") do io
                TOML.print(io,oversized_metadata; sorted=true)
            end
            metadata_digest=bytes2hex(sha256(read(metadata_path)))
            metadata_report=SC.verify_certificate(metadata_path,written.input_path;
                expected_input_sha256=written.input_sha256,
                expected_certificate_sha256=metadata_digest,
                require=:root_and_spectrum,precision_bits=256)
            @test metadata_report.status==:UNSUPPORTED
            @test metadata_report.reason=="RESOURCE_LIMIT"
        wrong_metadata_hash=SC.verify_certificate(metadata_path,written.input_path;
            expected_input_sha256=written.input_sha256,
            expected_certificate_sha256=written.certificate_sha256,
            require=:root_and_spectrum,precision_bits=256)
        @test wrong_metadata_hash.status==:INVALID
        @test occursin("certificate digest mismatch",wrong_metadata_hash.reason)
            maxbytes=16*1024*1024
            oversized_input=joinpath(parent,"oversized-input.toml")
            open(oversized_input,"w") do io; seek(io,maxbytes);write(io,UInt8(0));end
            oversized_certificate=joinpath(parent,"oversized-certificate.toml")
            open(oversized_certificate,"w") do io; seek(io,maxbytes);write(io,UInt8(0));end
            for (input_path,certificate_path,input_sha,certificate_sha,name) in (
                    (oversized_input,written.certificate_path,written.input_sha256,written.certificate_sha256,"oversized-input"),
                    (written.input_path,oversized_certificate,written.input_sha256,
                        bytes2hex(sha256(read(oversized_certificate))),"oversized-certificate"))
                report=SC.verify_certificate(certificate_path,input_path;
                    expected_input_sha256=input_sha,expected_certificate_sha256=certificate_sha,
                    require=:root_and_spectrum,precision_bits=256)
                @test report.status==:UNSUPPORTED
                @test report.reason=="RESOURCE_LIMIT"
                cli=_run_scientific_cli(repository_root,input_path,certificate_path,
                    input_sha,certificate_sha;precision=256)
                @test cli.exit_code==3
                @test occursin("status=UNSUPPORTED reason=RESOURCE_LIMIT",cli.output)
            end
        end
    end

    @testset "independent exact oracles and correction composition" begin
        fixtures=ScientificCertificationFixtures
        function rational_atan_bounds(q::Integer,terms::Int)
            # The alternating series partial and its first omitted term form
            # an exact rational enclosure for each arctangent in Machin's formula.
            q_big=BigInt(q)
            partial=BigInt(0)//BigInt(1)
            for index in 0:(terms-1)
                term=BigInt(1)//(BigInt(2index+1)*q_big^(2index+1))
                partial+=iseven(index) ? term : -term
            end
            next=BigInt(1)//(BigInt(2terms+1)*q_big^(2terms+1))
            iseven(terms) ? (partial,partial+next) : (partial-next,partial)
        end
        atan5=rational_atan_bounds(5,150)
        atan239=rational_atan_bounds(239,55)
        pi_lower=16*atan5[1]-4*atan239[2]
        pi_upper=16*atan5[2]-4*atan239[1]
        @test 3 < pi_lower < pi_upper < 4
        @test pi_upper-pi_lower < BigInt(1)//(BigInt(2)^600)
        pi_mpfr=fixtures.oracle_pi_interval(1024)
        @test pi_lower <= pi_mpfr[1] <= pi_mpfr[2] <= pi_upper
        for phase in (-5//9,5//9,-1//8,1//8)
            if phase < 0
                reference=(-phase/(2*pi_upper),-phase/(2*pi_lower))
            else
                reference=(-phase/(2*pi_lower),-phase/(2*pi_upper))
            end
            observed=fixtures.oracle_scalar_root(phase)
            @test observed[1] <= reference[1] <= reference[2] <= observed[2]
        end
        for exponent in (-400//1,-1//2)
            @test fixtures.oracle_contains_interval(
                fixtures.oracle_scalar_curvature(exponent),
                fixtures.oracle_scalar_curvature(exponent;bits=2048))
        end
        reference14=fixtures.oracle_t14_intervals()
        refined14=fixtures.oracle_t14_intervals(2048)
        @test fixtures.oracle_contains_interval(reference14[1],refined14[1])
        @test fixtures.oracle_contains_interval(reference14[2],refined14[2])
        t05=fixtures._record("T05")
        model5=first(fixtures._model_seed(t05))
        polynomial=fixtures.oracle_t05_pencil_polynomial(model5.Q,model5.L,model5.K)
        @test polynomial==Rational{BigInt}[55//1,-435//1,1161//1,-1341//1,678//1,-120//1]

        model1=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
        high=SC.certify(model1,[0];options=SC.ProofOptions(
            precisions=(512,),spectral_goal=:individuals))
        @test high.status==:CERTIFIED
        repository_root=normpath(joinpath(@__DIR__,".."))
        (; commit, tree) = head_identity(repository_root)
        source=(repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",commit,tree)
        mktempdir() do parent
            project=Base.active_project();manifest=joinpath(dirname(project),"Manifest.toml")
            envbytes=Vector{UInt8}(codeunits(join(("schema=cyax-scientific-test-environment-v1",
                "julia_version=$(VERSION)","interval_arithmetic_version=$(Base.pkgversion(SC.IA))",
                "project_sha256=$(bytes2hex(sha256(read(project))))",
                "manifest_sha256=$(bytes2hex(sha256(read(manifest))))"),"\n")*"\n"))
            envpath=joinpath(parent,"producer-environment.toml");write(envpath,envbytes)
            environment=(receipt_sha256=bytes2hex(sha256(envbytes)),artifact_name=basename(envpath))
            written=SC.write_certificate(joinpath(parent,"high"),model1,high;
                source_ref=source,environment_ref=environment)
            document=TOML.parsefile(written.certificate_path)
            spectrum=document["spectrum"]
            oldlo,oldhi=(SC._parse_rational(x) for x in spectrum["bounds"]["data"][1])
            narrowlo=oldlo+BigInt(1)//BigInt(2)^1024
            spectrum["bounds"]["data"][1][1]=SC._ratstr(narrowlo)
            spectrum["diagonal_ratios"]["data"][1][1]=SC._ratstr(narrowlo)
            spectrum["clusters"][1]["lower"]=SC._ratstr(narrowlo)
            width=SC._relative_met(narrowlo,oldhi,high.root.options.relative_width)
            spectrum["clusters"][1]["relative_width_met"]=width
            spectrum["resolution_met"]=width
            spectrum["reason"]=width ? "goal_met" : "refinement_inconclusive"
            rounded=SC._iv(narrowlo,oldhi,256)
            original=SC._iv(oldlo,oldhi,256)
            @test SC._lo(rounded)==SC._lo(original)
            path=joinpath(parent,"too-narrow-updated-digest.toml")
            write(path,SC._toml_bytes(document))
            digest=bytes2hex(sha256(read(path)))
            report=SC.verify_certificate(path,written.input_path;
                expected_input_sha256=written.input_sha256,
                expected_certificate_sha256=digest,require=:root_and_spectrum,
                precision_bits=256)
            @test report.status==:UNPROVEN
            cli=_run_scientific_cli(repository_root,written.input_path,path,
                written.input_sha256,digest;precision=256)
            @test cli.exit_code==1
            @test occursin("status=UNPROVEN",cli.output)
        end

        points=CYAxiverse.inflation_points
        Q=Int[1 0;0 1]
        L=Float64[1.0 -0.1;0.0 0.0]
        K=Float64[2.0 0.3;0.3 1.0]
        context=points.prepare_context(Q,L,K)
        correction=points.correct_stationary_point(context,[1.5,-0.5];residual_tolerance=1e-12)
        @test correction.status==:converged
        model=SC.prepare_model(Q,L,K)
        result=SC.certify(model,correction.theta;
            options=SC.ProofOptions(precisions=(256,),spectral_goal=:individuals),
            candidate_provenance=(method="inflation_points.correct_stationary_point",
                source_ref="T14 exact binary benchmark input"))
        @test result.status==:CERTIFIED
        @test result.root.candidate_provenance["status"]=="supplied"
    end
end

@testset "MX-04 mathematical replay does not waive provenance" begin
    model = SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1))
    result = SC.certify(model, [0]; options=SC.ProofOptions(precisions=(256,), spectral_goal=:individuals))
    @test result.status == :CERTIFIED

    repository_root = normpath(joinpath(@__DIR__, ".."))
    (; commit, tree) = head_identity(repository_root)
    source = (repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl", commit, tree)
    mktempdir() do parent
        project = Base.active_project()
        project === nothing && error("provenance control requires an active project")
        manifest = joinpath(dirname(project), "Manifest.toml")
        envpath = joinpath(parent, "producer-environment.toml")
        envbytes = Vector{UInt8}(codeunits(join((
            "schema=cyax-scientific-test-environment-v1",
            "julia_version=$(VERSION)",
            "interval_arithmetic_version=$(Base.pkgversion(SC.IA))",
            "project_sha256=$(bytes2hex(sha256(read(project))))",
            "manifest_sha256=$(bytes2hex(sha256(read(manifest))))"), "\n") * "\n"))
        write(envpath, envbytes)
        environment_ref = (receipt_sha256=bytes2hex(sha256(envbytes)), artifact_name=basename(envpath))
        certificate = SC.write_certificate(joinpath(parent, "certificate"), model, result;
            source_ref=source, environment_ref)
        observed_environment = (sha256=environment_ref.receipt_sha256,
            artifact_name=environment_ref.artifact_name, project_dir=dirname(project),
            record=Dict("worktree_clean"=>true))
        stale_source = (repository=source.repository, commit=repeat("0", 40),
            tree=repeat("0", 40), clean=true)
        evidence_root = joinpath(parent, "replay-evidence")
        mkpath(joinpath(evidence_root, "commands"))
        mkpath(joinpath(evidence_root, "replay"))
        @test_throws ErrorException ScientificCertificationFixtures._replay_receipt(
            "MX04_BAD_PROVENANCE", 256,
            Dict("path"=>dirname(certificate.certificate_path),
                "input_sha256"=>certificate.input_sha256,
                "certificate_sha256"=>certificate.certificate_sha256),
            observed_environment, stale_source, evidence_root,
            "logs/mx04-replay.log", "replay/mx04-replay.toml";
            require_provenance=true)
        receipt = TOML.parsefile(joinpath(evidence_root, "replay", "mx04-replay.toml"))
        @test receipt["mathematical_verdict"] == "VERIFIED"
        @test receipt["provenance_status"] == "FAIL"
        @test !(receipt["mathematical_verdict"] == "VERIFIED" && receipt["provenance_status"] == "PASS")
    end
end

@testset "MX-04 provenance decision isolates each identity" begin
    decide = ScientificCertificationFixtures._provenance_decision
    all_match = (producer_observation_available=true, producer_profile_matches=true,
        replay_profile_matches=true, producer_source_matches=true,
        producer_environment_matches=true)
    @test decide(; all_match...) == (true, "PASS")
    # Exactly one identity differs per case; each alone must prevent PASS.
    for field in keys(all_match)
        field === :producer_observation_available && continue
        flipped = merge(all_match, NamedTuple{(field,)}((false,)))
        @test decide(; flipped...) == (false, "FAIL")
    end
    # Stale source alone fails even without a producer observation.
    @test decide(; merge(all_match, (producer_observation_available=false,
        producer_source_matches=false))...) == (false, "FAIL")
    # Unavailable producer observation with matching replay and source is not PASS.
    @test decide(; merge(all_match, (producer_observation_available=false,
        producer_profile_matches=false))...) == (false, "UNAVAILABLE")
end

@testset "R2 source correction regressions" begin
    repository_root=normpath(joinpath(@__DIR__,".."))
    (; commit, tree) = head_identity(repository_root)
    source=(repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",commit,tree)

    @testset "closed and byte-bounded metadata" begin
        digest=repeat("a",64)
        @test SC._public_safe_source_ref("fixtures/candidate-source") == "fixtures/candidate-source"
        @test SC._public_safe_source_ref("opaque:fixture-reference") == "opaque:fixture-reference"
        @test SC._public_safe_source_ref("~candidate") == "~candidate"
        @test SC._public_safe_source_ref("https://github.com/Julia-meets-String-Theory/CYAxiverse.jl") ==
            "https://github.com/Julia-meets-String-Theory/CYAxiverse.jl"
        @test SC._source_record((repository="owner/private-project",commit,tree))["repository"] ==
            "owner/private-project"
        @test SC._source_record((repository="opaque:fixture-reference",commit,tree))["repository"] ==
            "opaque:fixture-reference"
        for repository in ("/fixture-private/source", "C:\\fixture-private\\source",
                "\\\\fixture-host\\share\\source", "file:///fixture-private/source",
                "session_id=fixture-only", "https://user:fixture@example.invalid/source",
                "https://localhost/source", "https://[::1]/source")
            @test_throws ArgumentError SC._source_record((repository=repository,commit,tree))
        end
        @test_throws ArgumentError SC._source_record(merge(source,(unexpected="discarded",)))
        @test_throws ArgumentError SC._source_record(Dict("repository"=>source.repository,
            :repository=>"conflict","commit"=>commit,"tree"=>tree))
        @test_throws ArgumentError SC._environment_record((receipt_sha256=digest,
            artifact_name="fixtures/../private.toml"))
        @test_throws ArgumentError SC._environment_record((receipt_sha256=digest,
            artifact_name="C:\\private\\receipt.toml"))
        @test_throws ArgumentError SC._environment_record((receipt_sha256=digest,
            artifact_name="\\\\server\\share\\receipt.toml"))
        @test_throws ArgumentError SC._environment_record((receipt_sha256=digest,
            artifact_name="file:///private/receipt.toml"))
        @test SC._environment_record((receipt_sha256=digest,
            artifact_name="fixtures/environment.toml"))["artifact_name"] == "fixtures/environment.toml"
        @test SC._environment_record((receipt_sha256=digest,
            artifact_name="~artifact.toml"))["artifact_name"] == "~artifact.toml"

        # The watchdog uses POSIX process groups and /bin/sh; the fixture itself
        # is qualified only on Darwin and 64-bit Linux.
        if Sys.isapple() || Sys.islinux()
            watchdog=ScientificCertificationFixtures._watchdog_self_test()
            @test watchdog.exit_code==124
            @test watchdog.timed_out
            @test watchdog.slow_start_passed
            @test watchdog.missing_ready_rejected
            @test watchdog.slow_preparation_seconds>=1.5
            @test watchdog.slow_computation_seconds<1
            @test watchdog.timeout_phase=="computation"
            @test watchdog.preparation_seconds>=0
            @test watchdog.computation_seconds>=2
            @test !watchdog.child_alive_after_timeout
            @test !watchdog.late_marker
        else
            @info "skipping watchdog self-test: unsupported platform"
        end
        mktempdir() do readiness_root
            marker=joinpath(readiness_root,"warmed-ready")
            withenv("CYAX_SCIENTIFIC_READY_MARKER"=>marker) do
                ScientificCertificationFixtures._announce_ready()
                @test isdir(marker)
                @test_throws ErrorException ScientificCertificationFixtures._announce_ready()
            end
        end

        t14_item=ScientificCertificationFixtures._record("T14")
        _,_,_,_,_,_,t14_diagnostic=ScientificCertificationFixtures._benchmark_model(t14_item)
        t14_comparison=ScientificCertificationFixtures._t14_diagnostic_compatibility(t14_diagnostic)
        @test t14_comparison["comparison_outcome"]=="PASS"
        @test t14_comparison["relative_tolerance"]=="1/10000000000"
        @test length(t14_comparison["diagnostic_eigenvalues"])==2
        @test SC._provenance((method="synthetic",source_ref="fixtures/candidate")) ==
            Dict("status"=>"supplied","method"=>"synthetic","source_ref"=>"fixtures/candidate")
        @test_throws ArgumentError SC._provenance((status="unknown",method="synthetic",
            source_ref="fixtures/candidate"))
        @test_throws ArgumentError SC._provenance((status="other",method="synthetic",
            source_ref="fixtures/candidate"))
        @test_throws ArgumentError SC._provenance(Dict("method"=>"one",:method=>"two",
            "source_ref"=>"fixtures/candidate"))
        @test_throws ArgumentError SC._provenance((method="synthetic",source_ref="/tmp/candidate"))
        @test_throws ArgumentError SC._provenance((method="synthetic",
            source_ref="https://chatgpt.com/share/fixture"))
        @test_throws ArgumentError SC._provenance((method="synthetic",
            source_ref="https://127.0.0.1:8080/fixture"))
        @test_throws ArgumentError SC._provenance((method="synthetic",
            source_ref="s3://localhost/private-object"))
        @test_throws ArgumentError SC._provenance((method="synthetic",
            source_ref="ftp://user@example.org/candidate"))
        @test_throws ArgumentError SC._provenance((method="synthetic",
            source_ref="https://example.org/share/fixture"))

        long_metadata=repeat("é",2049)
        @test_throws SC.ResourceLimitError SC._metadata_string(long_metadata,"method")
        @test SC._metadata_string(repeat("é",2048),"method")==repeat("é",2048)
        @test_throws SC.ResourceLimitError SC._source_record((repository=repeat("x",4097),
            commit,tree))
        @test_throws SC.ResourceLimitError SC._provenance((method=repeat("x",4097),
            source_ref="fixtures/candidate"))
        long_ratio=repeat("9",2048)*"/"*repeat("3",2048)
        @test_throws SC.ResourceLimitError SC._toml_bytes(Dict("method"=>long_ratio))
        raw_metadata=Vector{UInt8}(codeunits("method = \""*long_ratio*"\"\n"))
        @test_throws SC.ResourceLimitError SC._parse_toml(raw_metadata)
        encoded=SC._toml_bytes(Dict("Q"=>Dict("dims"=>[1,1],"data"=>[long_ratio])))
        @test SC._parse_toml(encoded)["Q"]["data"][1]==long_ratio
    end

    @requires_git @testset "execution profile requires a clean matching child" begin
        F=ScientificCertificationFixtures
        actual=F._actual_environment_observation()
        @test actual["interval_arithmetic_registered_tree_sha1"] ==
            actual["interval_arithmetic_actual_tree_sha1"]
        @test actual["loaded_package_root"] == realpath(repository_root)
        expected=copy(actual)
        expected["worktree_clean"]=true
        dirty=copy(expected)
        dirty["worktree_clean"]=false
        @test !F._approved_profile_matches(expected,dirty)
        changed_profile=copy(expected)
        changed_profile["active_manifest_sha256"]=repeat("0",64)
        @test !F._approved_profile_matches(expected,changed_profile)

        mktempdir() do root
            run(Cmd(Cmd(["git","init","--quiet"]);dir=root))
            write(joinpath(root,"source.txt"),"baseline\n")
            run(Cmd(["git","-C",root,"add","source.txt"]))
            run(Cmd(["git","-C",root,"-c","user.name=Fixture","-c",
                "user.email=fixture@example.invalid","commit","--quiet","-m","baseline"]))
            clean_source=F._source_identity(root)
            write(joinpath(root,"source.txt"),"baseline plus uncommitted byte\n")
            dirty_source=F._source_identity(root)
            @test clean_source.commit==dirty_source.commit
            @test clean_source.tree==dirty_source.tree
            @test clean_source.clean
            @test !dirty_source.clean
            unchanged_head_expected=copy(expected)
            unchanged_head_expected["candidate_commit"]=clean_source.commit
            unchanged_head_expected["candidate_tree"]=clean_source.tree
            unchanged_head_actual=copy(unchanged_head_expected)
            unchanged_head_actual["worktree_clean"]=dirty_source.clean
            @test !F._approved_profile_matches(unchanged_head_expected,unchanged_head_actual)
        end
    end

    @testset "attempted but incomplete spectrum stays unavailable" begin
        model=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
        options=SC.ProofOptions(precisions=(256,),max_inertia_evaluations=0,
            spectral_goal=:individuals)
        combined=SC.certify(model,[0];options)
        @test combined.status==:CERTIFIED_ROOT
        @test combined.root.status==:CERTIFIED_ROOT
        @test combined.spectrum.status==:INCONCLUSIVE
        environment_bytes=Vector{UInt8}(codeunits("synthetic environment receipt"))
        mktempdir() do parent
            envpath=joinpath(parent,"environment.toml")
            write(envpath,environment_bytes)
            environment=(receipt_sha256=bytes2hex(sha256(environment_bytes)),
                artifact_name=basename(envpath))
            unavailable=SC.write_certificate(joinpath(parent,"attempted-root"),model,combined;
                source_ref=source,environment_ref=environment)
            document=TOML.parsefile(unavailable.certificate_path)
            @test document["kind"]=="root"
            @test document["spectrum"]==Dict("status"=>"not_requested",
                "reason"=>"coverage_unavailable")
            root_report=SC.verify_certificate(unavailable.certificate_path,unavailable.input_path;
                expected_input_sha256=unavailable.input_sha256,
                expected_certificate_sha256=unavailable.certificate_sha256,
                require=:root,precision_bits=256)
            @test root_report.status==:VERIFIED
            spectrum_claim=only(filter(claim->claim.name=="spectrum",root_report.claims))
            @test spectrum_claim.outcome==:not_requested
            @test spectrum_claim.reason=="coverage_unavailable"
            spectrum_gate=SC.verify_certificate(unavailable.certificate_path,unavailable.input_path;
                expected_input_sha256=unavailable.input_sha256,
                expected_certificate_sha256=unavailable.certificate_sha256,
                require=:root_and_spectrum,precision_bits=256)
            @test spectrum_gate.status==:UNPROVEN

            root_only=SC.certify_stationary_point(model,[0];options)
            not_requested=SC.write_certificate(joinpath(parent,"unrequested-root"),model,root_only;
                source_ref=source,environment_ref=environment)
            not_requested_document=TOML.parsefile(not_requested.certificate_path)
            @test not_requested_document["spectrum"]==Dict("status"=>"not_requested",
                "reason"=>"not_requested")
        end
    end

    @testset "request fidelity on noncertifying exits" begin
        model=SC.prepare_model(reshape([1,1],1,2),[1 1;0 0],reshape([1],1,1))
        options=SC.ProofOptions(precisions=(256,),max_terms=1)
        provenance=(method="fixture-search",source_ref="fixtures/search-run")
        seed=[5//4]
        standalone=SC.certify_stationary_point(model,seed;options,candidate_provenance=provenance)
        combined=SC.certify(model,seed;options,candidate_provenance=provenance)
        for root in (standalone,combined.root)
            @test root.status==:UNSUPPORTED && root.reason=="RESOURCE_LIMIT"
            @test root.original_seed==[5//4]
            @test root.translation==[1]
            @test root.normalized_seed==[1//4]
            @test root.options==options
            @test root.candidate_provenance==SC._provenance(provenance)
            @test root.model_signature==SC._model_signature(model)
            @test isempty(root.attempts)
        end
        @test isempty(combined.attempts)

        flat=SC.prepare_model(reshape([1,0],2,1),reshape([1,0],2,1),Matrix{Int}(I,2,2))
        no_witness=SC.certify_stationary_point(flat,[5//4,1//3];
            options=SC.ProofOptions(precisions=(256,),box_radii=(1//big(2)^20,)),
            candidate_provenance=provenance)
        no_witness_combined=SC.certify(flat,[5//4,1//3];
            options=SC.ProofOptions(precisions=(256,),box_radii=(1//big(2)^20,)),
            candidate_provenance=provenance)
        @test no_witness.status==:INCONCLUSIVE && no_witness.reason=="no finite root witness"
        @test no_witness.translation==[1,0] && no_witness.normalized_seed==[1//4,1//3]
        @test no_witness.options.precisions==(256,)
        @test no_witness.candidate_provenance==SC._provenance(provenance)
        @test !isempty(no_witness.attempts)
        @test no_witness_combined.root.status==:INCONCLUSIVE
        @test no_witness_combined.root.translation==[1,0]
        @test no_witness_combined.root.normalized_seed==[1//4,1//3]
        @test no_witness_combined.root.options.precisions==(256,)
        @test no_witness_combined.root.candidate_provenance==SC._provenance(provenance)
        @test no_witness_combined.root.attempts==no_witness_combined.attempts
    end

    @testset "result input identity and interpretation" begin
        options=SC.ProofOptions(precisions=(256,),spectral_goal=:individuals)
        model=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
        root=SC.certify_stationary_point(model,[0];options)
        spectrum=SC.certify_generalized_spectrum(model,root;options)
        combined=SC.certify(model,[0];options)
        @test root.status==:CERTIFIED_ROOT
        @test spectrum.status==:CERTIFIED_BOUNDS
        @test combined.status==:CERTIFIED

        root_fields=fieldnames(typeof(root))
        root_values=Any[getfield(root,name) for name in root_fields]
        root_values[findfirst(==(:W),root_fields)]=SC._with_precision(256) do
            [SC._iv(-1//1,1//1,256)]
        end
        stored_W_outside=SC.RootResult(root_values...)
        @test SC._lo(stored_W_outside.W[1])<SC._lo(stored_W_outside.box[1])
        @test SC._hi(stored_W_outside.W[1])>SC._hi(stored_W_outside.box[1])
        rejected_spectrum=SC.certify_generalized_spectrum(model,stored_W_outside;options)
        @test rejected_spectrum.status==:INVALID_INPUT
        @test rejected_spectrum.reason=="root witness does not match model"
        @test !rejected_spectrum.resolution_met
        for identity in (root.identity,spectrum.identity,combined.identity,combined.root.identity,
                combined.spectrum.identity)
            @test identity.input_identity_status==:AVAILABLE
            @test identity.input_identity_reason===nothing
            @test occursin(r"^[0-9a-f]{64}$",identity.input_sha256)
            @test identity.interpretation.basis=="periodic_string"
            @test identity.interpretation.orientation=="axions_by_terms"
            @test identity.interpretation.theta_period=="1/1"
            @test identity.interpretation.phase_units=="radian"
            @test identity.interpretation.coefficient_convention=="signed_base10_log"
            @test identity.interpretation.numeric_semantics=="exact_supplied_values"
            @test identity.interpretation.spectrum_convention=="scaled_hessian_pencil"
            @test identity.interpretation.s==0//1
            @test identity.interpretation.physical_normalization=="not_established"
        end
        @test !hasproperty(root,:physical_mass)
        @test !hasproperty(spectrum,:physical_mass)
        @test root.identity.input_sha256==spectrum.identity.input_sha256==combined.identity.input_sha256

        source=(repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",
            commit=repeat("a",40),tree=repeat("b",40))
        mktempdir() do parent
            envpath=joinpath(parent,"environment.toml")
            write(envpath,"identity fixture environment")
            environment=(receipt_sha256=bytes2hex(sha256(read(envpath))),artifact_name=basename(envpath))
            written=SC.write_certificate(joinpath(parent,"certificate"),model,combined;
                source_ref=source,environment_ref=environment)
            emitted_digest=bytes2hex(sha256(read(written.input_path)))
            @test emitted_digest==written.input_sha256
            @test emitted_digest==root.identity.input_sha256
            @test emitted_digest==spectrum.identity.input_sha256
            @test emitted_digest==combined.identity.input_sha256
            @test SC._toml_bytes(SC._input_record(model))==read(written.input_path)
        end

        huge=big(10)^4096
        oversized_model=SC.prepare_model(reshape([1 0],1,2),
            reshape([huge,0,0,0],2,2),reshape([1],1,1))
        unsupported=SC.certify_stationary_point(oversized_model,[0];
            options=SC.ProofOptions(precisions=(256,),max_terms=1))
        unsupported_combined=SC.certify(oversized_model,[0];
            options=SC.ProofOptions(precisions=(256,),max_terms=1))
        @test unsupported.status==:UNSUPPORTED
        @test unsupported.identity.input_sha256===nothing
        @test unsupported.identity.input_identity_status==:UNAVAILABLE
        @test unsupported.identity.input_identity_reason=="RESOURCE_LIMIT"
        @test unsupported.identity.interpretation.s==0//1
        @test unsupported.identity.interpretation.physical_normalization=="not_established"
        @test unsupported_combined.status==:UNSUPPORTED
        @test unsupported_combined.identity.input_sha256===nothing
        @test unsupported_combined.root.identity.input_identity_status==:UNAVAILABLE
        @test unsupported_combined.spectrum.identity.input_identity_status==:UNAVAILABLE
    end

    @testset "pivot, endpoint order, and writer revalidation" begin
        options=SC.ProofOptions(precisions=(256,),spectral_goal=:individuals)
        model1=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
        model2=SC.prepare_model(Matrix{Int}(I,2,2),[1 -1;0 0],[2 1;1 2])
        result1=SC.certify(model1,[0];options)
        result2=SC.certify(model2,[0,0];options)
        @test result1.status==:CERTIFIED && result2.status==:CERTIFIED
        mktempdir() do parent
            envbytes=Vector{UInt8}(codeunits("synthetic environment receipt"))
            envpath=joinpath(parent,"environment.toml")
            write(envpath,envbytes)
            environment=(receipt_sha256=bytes2hex(sha256(envbytes)),artifact_name=basename(envpath))
            one=SC.write_certificate(joinpath(parent,"one"),model1,result1;
                source_ref=source,environment_ref=environment)
            two=SC.write_certificate(joinpath(parent,"two"),model2,result2;
                source_ref=source,environment_ref=environment)
            one_doc=TOML.parsefile(one.certificate_path)
            two_doc=TOML.parsefile(two.certificate_path)
            function replay(name,doc,written)
                path=joinpath(parent,"$name.toml")
                write(path,SC._toml_bytes(doc))
                digest=bytes2hex(sha256(read(path)))
                SC.verify_certificate(path,written.input_path;
                    expected_input_sha256=written.input_sha256,
                    expected_certificate_sha256=digest,
                    require=:root_and_spectrum,precision_bits=256)
            end
            bad_metric=deepcopy(one_doc)
            bad_metric["spectrum"]["metric_pivots"]["data"][1]=["-1/1","2/1"]
            @test replay("bad-metric-pivot",bad_metric,one).status==:INVALID
            bad_hessian=deepcopy(one_doc)
            bad_hessian["hessian"]["pivots"]["data"][1]=["-1000/1","1000/1"]
            @test replay("bad-hessian-pivot",bad_hessian,one).status==:INVALID
            @test two_doc["spectrum"]["route"]=="inertia_counts"
            bad_endpoint_pivot=deepcopy(two_doc)
            bad_endpoint_pivot["spectrum"]["endpoint_counts"][1]["pivots"]["data"][1]=["-1000/1","1000/1"]
            @test replay("bad-endpoint-pivot",bad_endpoint_pivot,two).status==:INVALID
            reversed=deepcopy(two_doc)
            reverse!(reversed["spectrum"]["endpoint_counts"])
            @test replay("reversed-endpoints",reversed,two).status==:INVALID
            permuted=deepcopy(two_doc)
            permuted["spectrum"]["endpoint_counts"][1:2]=permuted["spectrum"]["endpoint_counts"][2:-1:1]
            @test replay("permuted-endpoints",permuted,two).status==:INVALID

            unproved=deepcopy(two_doc)
            unproved["root"]["q_upper"]="0/1"
            @test replay("unproved-q",unproved,two).status==:UNPROVEN
            unknown_nested=deepcopy(unproved)
            unknown_nested["spectrum"]["endpoint_counts"][1]["unknown_nested"]=true
            @test replay("unproved-and-unknown-endpoint-key",unknown_nested,two).status==:INVALID
            wrong_count_type=deepcopy(unproved)
            wrong_count_type["spectrum"]["negative_count"]=true
            @test replay("unproved-and-boolean-spectrum-count",wrong_count_type,two).status==:INVALID
            wrong_sign_type=deepcopy(unproved)
            wrong_sign_type["spectrum"]["clusters"][1]["sign"]=1
            @test replay("unproved-and-integer-cluster-sign",wrong_sign_type,two).status==:INVALID
            bad_endpoint_order=deepcopy(unproved)
            reverse!(bad_endpoint_order["spectrum"]["endpoint_counts"])
            @test replay("unproved-and-reversed-endpoints",bad_endpoint_order,two).status==:INVALID

            q_cli_path=joinpath(parent,"unproved-q-cli.toml")
            write(q_cli_path,SC._toml_bytes(unproved))
            q_cli_digest=bytes2hex(sha256(read(q_cli_path)))
            q_cli=_run_scientific_cli(repository_root,two.input_path,q_cli_path,
                two.input_sha256,q_cli_digest;precision=256)
            println("schema-order q-only CLI argv=",repr(q_cli.argv)," exit=",q_cli.exit_code,
                " output=",repr(q_cli.output))
            @test q_cli.exit_code==1
            @test occursin("status=UNPROVEN",q_cli.output)

            malformed_cli=deepcopy(unproved)
            malformed_cli["spectrum"]["endpoint_counts"][1]["unknown_nested"]=true
            malformed_cli_path=joinpath(parent,"malformed-spectrum-cli.toml")
            write(malformed_cli_path,SC._toml_bytes(malformed_cli))
            malformed_cli_digest=bytes2hex(sha256(read(malformed_cli_path)))
            malformed_cli_result=_run_scientific_cli(repository_root,two.input_path,
                malformed_cli_path,two.input_sha256,malformed_cli_digest;precision=256)
            println("schema-order malformed CLI argv=",repr(malformed_cli_result.argv),
                " exit=",malformed_cli_result.exit_code,
                " output=",repr(malformed_cli_result.output))
            @test malformed_cli_result.exit_code==2
            @test occursin("status=INVALID reason=spectrum schema invalid",malformed_cli_result.output)

            root=result1.root
            bad_pivots=copy(root.hessian_pivots)
            bad_pivots[1]=SC._iv(BigInt(-1000)//BigInt(1),BigInt(1000)//BigInt(1),root.precision)
            bad_root=SC.RootResult(root.status,root.model_signature,root.reason,
                copy(root.original_seed),copy(root.translation),copy(root.normalized_seed),
                copy(root.center),copy(root.box),copy(root.preconditioner),copy(root.F_center),
                copy(root.H_box),copy(root.W),root.q_upper,root.precision,root.options,
                root.hessian_classification,root.hessian_method,bad_pivots,
                copy(root.candidate_provenance),copy(root.attempts),root.identity)
            @test_throws ArgumentError SC.write_certificate(joinpath(parent,"mutated-root"),
                model1,bad_root;source_ref=source,environment_ref=environment)
        end
    end

    @testset "public spectrum budget stays internal" begin
        model=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
        result=SC.certify_stationary_point(model,[0];options=SC.ProofOptions(precisions=(256,)))
        @test_throws MethodError SC.certify_generalized_spectrum(model,result;
            options=SC.ProofOptions(precisions=(256,)),budget=nothing)
    end

    @testset "partial spectral signs classify a saddle and replay" begin
        Q=[1 1 0;1 -1 0;0 0 1]
        L=reshape(Rational{BigInt}[1//1,0//1,-1//1,0//1,1//big(2)^20,0//1],2,3)
        model=SC.prepare_model(Q,L,Matrix{Int}(I,3,3))
        options=SC.ProofOptions(precisions=(256,),box_radii=(1//big(2)^20,),
            max_expansions=1,max_inertia_evaluations=7,spectral_goal=:individuals)
        result=SC.certify(model,zeros(3);options)
        @test result.status==:INCONCLUSIVE_SPECTRUM
        @test result.root.status==:CERTIFIED_ROOT
        @test result.root.hessian_method==:spectral_counts
        @test result.root.hessian_classification==:SADDLE
        @test (result.spectrum.negative_count,result.spectrum.positive_count,
            result.spectrum.unresolved_count)==(1,1,1)
        @test count(x->startswith(x,"root H LDL evaluations=1 method=none"),result.attempts)==1
        @test count(x->startswith(x,"root H LDL evaluations="),result.attempts)==1
        mktempdir() do parent
            envbytes=Vector{UInt8}(codeunits("synthetic environment receipt"))
            envpath=joinpath(parent,"environment.toml")
            write(envpath,envbytes)
            environment=(receipt_sha256=bytes2hex(sha256(envbytes)),artifact_name=basename(envpath))
            @test_throws ArgumentError SC.write_certificate(joinpath(parent,"root-only"),
                model,result.root;source_ref=source,environment_ref=environment)
            written=SC.write_certificate(joinpath(parent,"combined"),model,result;
                source_ref=source,environment_ref=environment)
            document=TOML.parsefile(written.certificate_path)
            @test document["hessian"]["method"]=="spectral_counts"
            @test document["hessian"]["classification"]=="SADDLE"
            report=SC.verify_certificate(written.certificate_path,written.input_path;
                expected_input_sha256=written.input_sha256,
                expected_certificate_sha256=written.certificate_sha256,
                require=:root_and_spectrum,precision_bits=256)
            @test report.status==:VERIFIED

            false_minimum=deepcopy(document)
            false_minimum["hessian"]["classification"]="MINIMUM"
            path=joinpath(parent,"unresolved-called-minimum.toml")
            write(path,SC._toml_bytes(false_minimum))
            invalid=SC.verify_certificate(path,written.input_path;
                expected_input_sha256=written.input_sha256,
                expected_certificate_sha256=bytes2hex(sha256(read(path))),
                require=:root_and_spectrum,precision_bits=256)
            @test invalid.status==:UNPROVEN
        end
    end
end

@testset "R2 serialized provenance and embedded private paths" begin
    repository_root=normpath(joinpath(@__DIR__,".."))
    (; commit, tree) = head_identity(repository_root)
    source=(repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",commit,tree)
    model=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
    options=SC.ProofOptions(precisions=(256,),spectral_goal=:individuals)
    shorthand=(method="synthetic",source_ref="fixtures/candidate-source")
    @test SC._provenance(shorthand)==Dict("status"=>"supplied",
        "method"=>"synthetic","source_ref"=>"fixtures/candidate-source")
    @test SC._read_candidate_provenance(Dict("status"=>"unknown"))==Dict("status"=>"unknown")
    @test SC._read_candidate_provenance(Dict("status"=>"supplied",
        "method"=>"synthetic","source_ref"=>"fixtures/candidate-source"))==
        Dict("status"=>"supplied","method"=>"synthetic","source_ref"=>"fixtures/candidate-source")
    @test_throws ArgumentError SC._read_candidate_provenance(Dict(
        "method"=>"synthetic","source_ref"=>"fixtures/candidate-source"))
    @test_throws ArgumentError SC._read_candidate_provenance(Dict(
        "status"=>"unknown","method"=>"synthetic","source_ref"=>"fixtures/candidate-source"))

    unsafe_paths=("fixture description /home/example-user/private/snapshot.toml",
        "fixture description C:\\Users\\example-user\\private\\snapshot.toml",
        "fixture description \\\\example-host\\private-share\\snapshot.toml")
    dotted_chat="https://chatgpt.com./c/synthetic-review-fixture"
    dotted_local="https://localhost.localdomain./synthetic-review-fixture"
    dotted_private="https://192.168.1.5./synthetic-review-fixture"
    for reference in unsafe_paths
        @test_throws ArgumentError SC._public_safe_source_ref(reference)
        @test_throws ArgumentError SC._provenance((method="synthetic",source_ref=reference))
        @test_throws ArgumentError SC._source_record((repository=reference,commit,tree))
    end
    for reference in (dotted_chat,dotted_local,dotted_private,
            "https://chat.openai.com./c/synthetic-review-fixture",
            "https://claude.ai./c/synthetic-review-fixture")
        @test_throws ArgumentError SC._public_safe_metadata(reference,"synthetic metadata")
        @test_throws ArgumentError SC._public_safe_source_ref(reference)
        @test_throws ArgumentError SC._provenance((method="synthetic",source_ref=reference))
        @test_throws ArgumentError SC._source_record((repository=reference,commit,tree))
    end
    dotted_public="https://example.org./public-fixture"
    @test SC._public_safe_metadata(dotted_public,"synthetic metadata")==dotted_public
    @test SC._public_safe_source_ref(dotted_public)==dotted_public
    @test SC._public_safe_source_ref("fixtures/candidate-source")=="fixtures/candidate-source"
    @test SC._public_safe_source_ref("opaque:fixture-reference")=="opaque:fixture-reference"
    @test SC._public_safe_source_ref(source.repository)==source.repository

    result=SC.certify(model,[0];options,candidate_provenance=shorthand)
    @test result.status==:CERTIFIED
    mktempdir() do parent
        environment_bytes=Vector{UInt8}(codeunits("synthetic public environment receipt"))
        environment_path=joinpath(parent,"environment.toml")
        write(environment_path,environment_bytes)
        environment=(receipt_sha256=bytes2hex(sha256(environment_bytes)),
            artifact_name=basename(environment_path))
        written=SC.write_certificate(joinpath(parent,"valid"),model,result;
            source_ref=source,environment_ref=environment)
        valid_report=SC.verify_certificate(written.certificate_path,written.input_path;
            expected_input_sha256=written.input_sha256,
            expected_certificate_sha256=written.certificate_sha256,
            require=:root_and_spectrum,precision_bits=256)
        @test valid_report.status==:VERIFIED

        valid_document=TOML.parsefile(written.certificate_path)
        function replay_mutation(name,mutate!)
            document=deepcopy(valid_document)
            mutate!(document)
            path=joinpath(parent,name*".toml")
            write(path,SC._toml_bytes(document))
            digest=bytes2hex(sha256(read(path)))
            report=SC.verify_certificate(path,written.input_path;
                expected_input_sha256=written.input_sha256,
                expected_certificate_sha256=digest,
                require=:root_and_spectrum,precision_bits=256)
            (path,digest,report)
        end

        missing_status_path,missing_status_digest,missing_status_report=replay_mutation(
            "missing-provenance-status",doc->delete!(doc["candidate_provenance"],"status"))
        @test missing_status_report.status==:INVALID

        unsafe_reference=first(unsafe_paths)
        unsafe_provenance_path,_,unsafe_provenance_report=replay_mutation(
            "unsafe-candidate-provenance",doc->(doc["candidate_provenance"]=Dict(
                "status"=>"supplied","method"=>"synthetic","source_ref"=>unsafe_reference)))
        @test unsafe_provenance_report.status==:INVALID
        unsafe_source_path,_,unsafe_source_report=replay_mutation(
            "unsafe-top-level-source",doc->(doc["source_ref"]["repository"]=unsafe_reference))
        @test unsafe_source_report.status==:INVALID

        dotted_provenance_path,_,dotted_provenance_report=replay_mutation(
            "dotted-chat-provenance",doc->(doc["candidate_provenance"]["source_ref"]=dotted_chat))
        @test dotted_provenance_report.status==:INVALID
        @test occursin("chat/share URL",dotted_provenance_report.reason)
        dotted_source_path,_,dotted_source_report=replay_mutation(
            "dotted-chat-top-level-source",doc->(doc["source_ref"]["repository"]=dotted_chat))
        @test dotted_source_report.status==:INVALID
        @test occursin("chat/share URL",dotted_source_report.reason)

        @test_throws ArgumentError SC.write_certificate(joinpath(parent,"unsafe-source-writer"),
            model,result;source_ref=merge(source,(repository=unsafe_reference,)),
            environment_ref=environment)
        @test !ispath(joinpath(parent,"unsafe-source-writer"))
        @test_throws ArgumentError SC.write_certificate(joinpath(parent,"dotted-chat-source-writer"),
            model,result;source_ref=merge(source,(repository=dotted_chat,)),
            environment_ref=environment)
        @test !ispath(joinpath(parent,"dotted-chat-source-writer"))

        root_fields=fieldnames(typeof(result.root))
        root_values=Any[getfield(result.root,name) for name in root_fields]
        provenance_index=findfirst(==(:candidate_provenance),root_fields)
        root_values[provenance_index]=Dict("status"=>"supplied","method"=>"synthetic",
            "source_ref"=>unsafe_reference)
        unsafe_root=SC.RootResult(root_values...)
        unsafe_provenance_directory=joinpath(parent,"unsafe-provenance-writer")
        @test_throws ArgumentError SC.write_certificate(unsafe_provenance_directory,
            model,unsafe_root;source_ref=source,environment_ref=environment)
        @test !isfile(joinpath(unsafe_provenance_directory,"inputs.toml"))
        @test !isfile(joinpath(unsafe_provenance_directory,"certificate.toml"))

        dotted_provenance_values=Any[getfield(result.root,name) for name in root_fields]
        dotted_provenance_values[provenance_index]=Dict("status"=>"supplied","method"=>"synthetic",
            "source_ref"=>dotted_chat)
        dotted_provenance_root=SC.RootResult(dotted_provenance_values...)
        dotted_provenance_directory=joinpath(parent,"dotted-chat-provenance-writer")
        @test_throws ArgumentError SC.write_certificate(dotted_provenance_directory,
            model,dotted_provenance_root;source_ref=source,environment_ref=environment)
        @test !ispath(dotted_provenance_directory)

        cli_argv=vcat(String.(Base.julia_cmd().exec),child_coverage_args(),String["--startup-file=no",
            "--project=$repository_root",joinpath(repository_root,"scripts",
            "verify_scientific_certificate.jl"),"--inputs",written.input_path,
            "--certificate",missing_status_path,"--expect-input-sha256",written.input_sha256,
            "--expect-certificate-sha256",missing_status_digest,"--require",
            "root_and_spectrum","--precision","256"])
        mktemp() do _log_path,output_file
            process=run(pipeline(Cmd(Cmd(cli_argv);dir=repository_root);
                stdout=output_file,stderr=output_file);wait=false)
            wait(process)
            flush(output_file);seekstart(output_file)
            output=read(output_file,String)
            @test process.exitcode==2
            @test occursin("status=INVALID",output)
        end
    end
end

# The skills-checker JSON and action controls live in scripts/test_skills_checker_controls.py.

@testset "Full public qualification profile" begin
    @test ScientificCertificationFixtures.qualify_from_test_profile!() === nothing
end

@testset "release pins are a pure predicate on the observed record" begin
    F=ScientificCertificationFixtures
    pinned=Dict("julia_version"=>"1.12.0","interval_arithmetic_version"=>"0.22.36",
        "interval_arithmetic_registered_tree_sha1"=>"79342df41c3c24664e5bf29395cfdf2f2a599412")
    @test F._release_pins_match(pinned)
    for (key,value) in (("julia_version","1.11.4"),("interval_arithmetic_version","0.22.37"),
            ("interval_arithmetic_registered_tree_sha1",repeat("0",40)))
        changed=copy(pinned)
        changed[key]=value
        @test !F._release_pins_match(changed)
    end
end

@testset "thread-pool predicate rejects nonserial profiles (pure, no children)" begin
    F=ScientificCertificationFixtures
    profile=F._actual_environment_observation()
    if Threads.nthreads(:default)==1 && Threads.nthreads(:interactive)==0
        profile["worktree_clean"]=true
        @test F._approved_profile_matches(profile,profile;pinned=false)
    end
    # Each field is set to a value the approved one-thread profile forbids, on an
    # otherwise approved record; only that field may cause the rejection.
    approved=copy(profile)
    approved["julia_threads"]=1;approved["julia_default_threads"]=1
    approved["julia_interactive_threads"]=0;approved["blas_threads"]=1
    approved["julia_num_threads"]="1";approved["openblas_num_threads"]="1"
    approved["loaded_package_root"]=realpath(F.ROOT)
    approved["loaded_package_entrypoint"]=realpath(joinpath(F.ROOT,"src","CYAxiverse.jl"))
    approved["certificate_source_sha256"]="present"
    approved["interval_arithmetic_registered_tree_sha1"]="x"
    approved["interval_arithmetic_actual_tree_sha1"]="x"
    @test F._approved_profile_matches(approved,approved;pinned=false)
    for (key,bad) in (("julia_threads",2),("julia_default_threads",2),
            ("julia_interactive_threads",1),("blas_threads",2),
            ("julia_num_threads","2"),("openblas_num_threads","2"))
        rejected=copy(approved);rejected[key]=bad
        @test !F._approved_profile_matches(rejected,rejected;pinned=false)
    end
end

# CYAX_R2_SERIAL_QUALIFICATION_PROFILE_TESTS_BEGIN
@requires_git @testset "CI qualification profile is scoped and checks the real thread pools" begin
    F=ScientificCertificationFixtures
    original_blas=BLAS.get_num_threads()
    try
      withenv("JULIA_NUM_THREADS"=>"caller-julia-setting","OPENBLAS_NUM_THREADS"=>"caller-openblas-setting",
              "CYAXIVERSE_TEST_FULL"=>nothing) do
        Threads.nthreads(:default)==1 && Threads.nthreads(:interactive)==0 ||
            error("profile regression requires the approved one-thread Julia test process")
        BLAS.set_num_threads(2)
        for (full_setting,expected_full) in ((nothing,true),("0",false),("1",true))
          withenv("CYAXIVERSE_TEST_FULL"=>full_setting) do
            @test F._test_profile_is_full()===expected_full
            observed=F._with_serial_qualification_profile() do
                @test get(ENV,"CYAXIVERSE_TEST_FULL","1")==
                    (full_setting===nothing ? "1" : full_setting)
                @test F._test_profile_is_full()===expected_full
                profile=F._actual_environment_observation()
                @test profile["julia_threads"]==1
                @test profile["julia_default_threads"]==1
                @test profile["julia_interactive_threads"]==0
                @test profile["blas_threads"]==1
                @test profile["julia_num_threads"]=="1"
                @test profile["openblas_num_threads"]=="1"
                profile["worktree_clean"]=true # Exercise profile matching independently of an edited checkout.
                # Release pins (Julia, IntervalArithmetic) are checked by the dedicated CI step.
                @test F._approved_profile_matches(profile,profile;pinned=false)
                for (key,value) in (("julia_default_threads",2),("julia_interactive_threads",1))
                    invalid_profile=copy(profile)
                    invalid_profile[key]=value
                    @test !F._approved_profile_matches(invalid_profile,invalid_profile;pinned=false)
                end
                missing_pool=copy(profile)
                delete!(missing_pool,"julia_interactive_threads")
                @test !F._approved_profile_matches(missing_pool,missing_pool;pinned=false)
                get(ENV,"CYAXIVERSE_TEST_FULL","1")=="1"
            end
            @test observed===expected_full
            @test BLAS.get_num_threads()==2
            @test ENV["JULIA_NUM_THREADS"]=="caller-julia-setting"
            @test ENV["OPENBLAS_NUM_THREADS"]=="caller-openblas-setting"
            @test get(ENV,"CYAXIVERSE_TEST_FULL","1")==
                (full_setting===nothing ? "1" : full_setting)
            @test (haskey(ENV,"CYAXIVERSE_TEST_FULL") == (full_setting!==nothing))
          end
        end

        restored_after_error=try
            F._with_serial_qualification_profile() do
                error("serial-profile restoration sentinel")
            end
            false
        catch err
            err isa ErrorException && occursin("serial-profile restoration sentinel",sprint(showerror,err))
        end
        @test restored_after_error
        @test BLAS.get_num_threads()==2
        @test ENV["JULIA_NUM_THREADS"]=="caller-julia-setting"
        @test ENV["OPENBLAS_NUM_THREADS"]=="caller-openblas-setting"

        child_code="""
            using LinearAlgebra, CYAxiverse
            include(ARGS[1])
            F=ScientificCertificationFixtures
            before_blas=BLAS.get_num_threads()
            ENV[\"JULIA_NUM_THREADS\"]=\"child-julia-setting\"
            ENV[\"OPENBLAS_NUM_THREADS\"]=\"child-openblas-setting\"
            rejected=try
                F._with_serial_qualification_profile() do
                    error(\"nonserial child entered the qualification callback\")
                end
                false
            catch err
                occursin(\"started with one thread\",sprint(showerror,err))
            end
            @assert rejected
            pools=parse.(Int,split(ARGS[2],','))
            @assert Threads.nthreads(:default)==pools[1]
            @assert Threads.nthreads(:interactive)==(length(pools)==2 ? pools[2] : 0)
            @assert BLAS.get_num_threads()==before_blas
            @assert ENV[\"JULIA_NUM_THREADS\"]==\"child-julia-setting\"
            @assert ENV[\"OPENBLAS_NUM_THREADS\"]==\"child-openblas-setting\"
            println(\"nonserial_parent_rejected=true default=\",Threads.nthreads(:default),
                \" interactive=\",Threads.nthreads(:interactive),\" blas_threads=\",BLAS.get_num_threads())
            println(\"cache_flags=\",Base.CacheFlags())
        """
        # Package images are keyed on these flags; a child that differs from this
        # process (e.g. without Pkg.test's --check-bounds=yes) recompiles every
        # dependency of CYAxiverse instead of reusing this process's images.
        parent_cache_flags="cache_flags=$(Base.CacheFlags())"
        repository_root=normpath(joinpath(@__DIR__,".."))
        for (thread_setting,expected_counts) in (("2,0","default=2 interactive=0"),
                ("1,1","default=1 interactive=1"))
            # Base.julia_cmd() carries this process's cache flags; the later --threads
            # wins over any inherited one, and child_coverage_args() keeps coverage
            # output out of the checkout.
            child_argv=vcat(String.(Base.julia_cmd().exec),child_coverage_args(),String["--startup-file=no",
                "--threads=$thread_setting","--project=$repository_root","-e",child_code,
                joinpath(@__DIR__,"fixtures","scientific_certification","run_fixture.jl"),thread_setting])
            child_env=addenv(Cmd(child_argv),"JULIA_NUM_THREADS"=>thread_setting,"OPENBLAS_NUM_THREADS"=>"2")
            child_stdout=""
            child_exit=mktemp() do _path,log
                process=run(pipeline(child_env,stdout=log,stderr=log);wait=false)
                wait(process)
                flush(log);seekstart(log)
                output=read(log,String)
                child_stdout=output
                process.exitcode
            end
            @test child_exit==0
            @test occursin("nonserial_parent_rejected=true $expected_counts blas_threads=2",child_stdout)
            @test occursin(parent_cache_flags,child_stdout)
        end
      end
    finally
        BLAS.set_num_threads(original_blas)
    end
end
# CYAX_R2_SERIAL_QUALIFICATION_PROFILE_TESTS_END

# R2 focused correction regressions. This terminal block is selected by the
# detached focused runner; the public package suite remains unchanged here.
@testset "R2 focused privacy and serialization bounds" begin
    repository_root=normpath(joinpath(@__DIR__,".."))
    (; commit, tree) = head_identity(repository_root)
    source=(repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",commit,tree)
    safe_source_ref="fixtures/candidate-source"
    safe_method="deterministic fixture search"
    safe_name="fixtures/environment.toml"
    digest=repeat("a",64)

    @test SC._provenance((method=safe_method,source_ref=safe_source_ref))["method"]==safe_method
    @test SC._environment_record((receipt_sha256=digest,artifact_name=safe_name))["artifact_name"]==safe_name

    unsafe_methods=("/home/example-user/private/search.jl",
        "fixture method /private/workspace/search.jl",
        "machine_id=synthetic-review-host session_id=synthetic-session",
        "hostname=synthetic-review-host",
        "elapsed_seconds=1.25",
        "started_at=2026-10-11T12:34:56Z")
    for method in unsafe_methods
        @test_throws ArgumentError SC._provenance((method=method,source_ref=safe_source_ref))
    end
    unsafe_names=("hostname=synthetic-review-host_session_id=synthetic-session.toml",
        "fixtures/session_id=synthetic-session.toml",
        "fixtures/search /home/example-user/private/search.jl.toml",
        "fixtures/elapsed_seconds=1.25.toml")
    for artifact_name in unsafe_names
        @test_throws ArgumentError SC._environment_record((receipt_sha256=digest,artifact_name=artifact_name))
    end

    model=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
    options=SC.ProofOptions(precisions=(256,),spectral_goal=:individuals)
    provenance=(method=safe_method,source_ref=safe_source_ref)
    result=SC.certify(model,[0];options,candidate_provenance=provenance)
    @test result.status==:CERTIFIED
    @test_throws ArgumentError SC.certify_stationary_point(model,[0];options,
        candidate_provenance=(method=first(unsafe_methods),source_ref=safe_source_ref))
    mktempdir() do parent
        envbytes=Vector{UInt8}(codeunits("synthetic public environment receipt"))
        environment=(receipt_sha256=bytes2hex(sha256(envbytes)),artifact_name=safe_name)
        written=SC.write_certificate(joinpath(parent,"valid"),model,result;
            source_ref=source,environment_ref=environment)
        report=SC.verify_certificate(written.certificate_path,written.input_path;
            expected_input_sha256=written.input_sha256,
            expected_certificate_sha256=written.certificate_sha256,
            require=:root_and_spectrum,precision_bits=256)
        @test report.status==:VERIFIED

        unsafe_environment=(receipt_sha256=environment.receipt_sha256,
            artifact_name=first(unsafe_names))
        unsafe_environment_path=joinpath(parent,"unsafe-environment")
        @test_throws ArgumentError SC.write_certificate(unsafe_environment_path,model,result;
            source_ref=source,environment_ref=unsafe_environment)
        @test !ispath(unsafe_environment_path)

        unsafe_provenance=deepcopy(result)
        unsafe_provenance.root.candidate_provenance["method"]=first(unsafe_methods)
        unsafe_provenance_path=joinpath(parent,"unsafe-provenance")
        @test_throws ArgumentError SC.write_certificate(unsafe_provenance_path,model,
            unsafe_provenance;source_ref=source,environment_ref=environment)
        @test !ispath(unsafe_provenance_path)

        original=TOML.parsefile(written.certificate_path)
        function verify_mutant(name,mutate!)
            document=deepcopy(original)
            mutate!(document)
            path=joinpath(parent,name*".toml")
            write(path,SC._toml_bytes(document))
            certificate_sha=bytes2hex(sha256(read(path)))
            SC.verify_certificate(path,written.input_path;
                expected_input_sha256=written.input_sha256,
                expected_certificate_sha256=certificate_sha,
                require=:root_and_spectrum,precision_bits=256)
        end
        unsafe_method_report=verify_mutant("unsafe-method",document->
            (document["candidate_provenance"]["method"]=first(unsafe_methods)))
        @test unsafe_method_report.status==:INVALID
        @test occursin("candidate method",unsafe_method_report.reason)
        unsafe_name_report=verify_mutant("unsafe-artifact-name",document->
            (document["environment_ref"]["artifact_name"]=first(unsafe_names)))
        @test unsafe_name_report.status==:INVALID
        @test occursin("machine or session",unsafe_name_report.reason)
    end

    small_model=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
    small_identity=SC._result_identity(small_model)
    @test small_identity.input_identity_status==:AVAILABLE
    large_charge=big(10)^256 # 257 decimal digits; bounded and below each-component limit.
    large_model=SC.prepare_model(fill(large_charge,32,2048),zeros(Int,2,2048),
        Matrix{Int}(I,32,32))
    SC._result_identity(small_model) # warm the measured path before allocation accounting.
    identity_allocation=@allocated begin
        large_identity=SC._result_identity(large_model)
    end
    @test large_identity.input_identity_status==:UNAVAILABLE
    @test large_identity.input_identity_reason=="RESOURCE_LIMIT"
    @test identity_allocation < 4*1024*1024
    println("input_preflight_allocated_bytes=",identity_allocation)

    small_result=SC.certify(small_model,[0];options)
    mktempdir() do parent
        bounded_output=joinpath(parent,"resource-limited-certificate")
        envbytes=Vector{UInt8}(codeunits("synthetic public environment receipt"))
        environment=(receipt_sha256=bytes2hex(sha256(envbytes)),artifact_name=safe_name)
        @test_throws SC.ResourceLimitError SC.write_certificate(bounded_output,large_model,
            small_result;source_ref=source,environment_ref=environment)
        @test !ispath(bounded_output)
    end

    large_rational=string(big(10)^4095,"/1")
    oversized_document=Dict("Q"=>Dict("dims"=>[1,4097],
        "data"=>fill(large_rational,4097)))
    @test SC._toml_bytes(Dict("Q"=>Dict("dims"=>[1,1],"data"=>["1/1"]))) isa Vector{UInt8}
    serialization_error=nothing
    serialization_allocation=@allocated begin
        try
            SC._toml_bytes(oversized_document)
        catch error
            serialization_error=error
        end
    end
    @test serialization_error isa SC.ResourceLimitError
    @test serialization_allocation < 4*1024*1024
    println("generic_toml_preflight_allocated_bytes=",serialization_allocation)

    certificate_root=small_result.root
    append!(certificate_root.original_seed,fill(big(10)^4095//1,4097))
    certificate_error=nothing
    certificate_allocation=@allocated begin
        try
            SC._preflight_certificate_rational_bytes(certificate_root,nothing)
        catch error
            certificate_error=error
        end
    end
    @test certificate_error isa SC.ResourceLimitError
    @test certificate_allocation < 4*1024*1024
    println("certificate_preflight_allocated_bytes=",certificate_allocation)
end
