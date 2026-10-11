using SHA
using TOML

const SC = CYAxiverse.scientific_certification

function _run_scientific_cli(repository_root, input_path, certificate_path,
        input_sha256, certificate_sha256; precision=256)
    argv=vcat(String[Base.julia_cmd().exec...],String["--startup-file=no","--project=$repository_root",
        joinpath(repository_root,"scripts","verify_scientific_certificate.jl"),
        "--inputs",input_path,"--certificate",certificate_path,
        "--expect-input-sha256",input_sha256,
        "--expect-certificate-sha256",certificate_sha256,
        "--require","root_and_spectrum","--precision",string(precision)])
    mktemp() do _path, output
        process=run(pipeline(Cmd(Cmd(argv);dir=repository_root);stdout=output,stderr=output);wait=false)
        wait(process)
        flush(output); seekstart(output)
        (exit_code=process.exitcode,output=read(output,String),argv=argv)
    end
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
        commit = strip(read(Cmd(["git", "-C", repository_root, "rev-parse", "HEAD"]), String))
        tree = strip(read(Cmd(["git", "-C", repository_root, "rev-parse", "HEAD^{tree}"]), String))
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

            forged = TOML.parsefile(first.certificate_path)
            forged["root"]["q_upper"] = "0/1"
            forged_path = joinpath(parent, "forged.toml")
            open(forged_path, "w") do io
                TOML.print(io, forged; sorted=true)
            end
            forged_hash = bytes2hex(sha256(read(forged_path)))
            rejected = SC.verify_certificate(forged_path, first.input_path;
                expected_input_sha256=first.input_sha256,
                expected_certificate_sha256=forged_hash,
                require=:root_and_spectrum, precision_bits=256)
            @test rejected.status in (:INVALID, :UNPROVEN)
            @test rejected.status != :VERIFIED
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
    commit = strip(read(Cmd(["git","-C",repository_root,"rev-parse","HEAD"]),String))
    tree = strip(read(Cmd(["git","-C",repository_root,"rev-parse","HEAD^{tree}"]),String))
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
        commit=strip(read(Cmd(["git","-C",repository_root,"rev-parse","HEAD"]),String))
        tree=strip(read(Cmd(["git","-C",repository_root,"rev-parse","HEAD^{tree}"]),String))
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
        @test_throws SC.ResourceLimitError SC.prepare_model(zeros(Int,33,1),
            reshape([1,0],2,1),Matrix{Int}(I,33,33))
        @test_throws SC.ResourceLimitError SC.prepare_model(zeros(Int,1,2049),
            zeros(Int,2,2049),reshape([1],1,1))
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
        big_model=SC.CertificationModel(big_q,reshape(Rational{BigInt}[1//1,0//1],2,1),
            Matrix{Rational{BigInt}}(I,33,33),zeros(Rational{BigInt},1),BigInt(0)//BigInt(1))
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
        @test_throws SC.ResourceLimitError SC._ratstr(BigInt(1)//BigInt(10)^4097)
        @test_throws SC.ResourceLimitError SC._toml_bytes(
            Dict("payload"=>repeat("x",16*1024*1024)))

        model=SC.prepare_model(reshape([1],1,1),reshape([1,0],2,1),reshape([1],1,1))
        cert=SC.certify(model,[0];options=SC.ProofOptions(precisions=(256,),spectral_goal=:individuals))
        repository_root=normpath(joinpath(@__DIR__,".."))
        commit=strip(read(Cmd(["git","-C",repository_root,"rev-parse","HEAD"]),String))
        tree=strip(read(Cmd(["git","-C",repository_root,"rev-parse","HEAD^{tree}"]),String))
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
        commit=strip(read(Cmd(["git","-C",repository_root,"rev-parse","HEAD"]),String))
        tree=strip(read(Cmd(["git","-C",repository_root,"rev-parse","HEAD^{tree}"]),String))
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
    commit = strip(read(Cmd(["git", "-C", repository_root, "rev-parse", "HEAD"]), String))
    tree = strip(read(Cmd(["git", "-C", repository_root, "rev-parse", "HEAD^{tree}"]), String))
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

@testset "R2 source correction regressions" begin
    repository_root=normpath(joinpath(@__DIR__,".."))
    commit=strip(read(Cmd(["git","-C",repository_root,"rev-parse","HEAD"]),String))
    tree=strip(read(Cmd(["git","-C",repository_root,"rev-parse","HEAD^{tree}"]),String))
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
                "https://localhost/source")
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

        watchdog=ScientificCertificationFixtures._watchdog_self_test()
        @test watchdog.exit_code==124
        @test watchdog.timed_out
        @test !watchdog.child_alive_after_timeout
        @test !watchdog.late_marker

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

    @testset "execution profile requires a clean matching child" begin
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

@testset "Full public qualification profile" begin
    @test ScientificCertificationFixtures.qualify_from_test_profile!() === nothing
end
