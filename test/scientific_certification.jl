using SHA
using TOML

const SC = CYAxiverse.scientific_certification

@testset "Bounded scientific certification" begin
    @testset "exact input and phase conventions" begin
        @test_throws DimensionMismatch SC.prepare_model(reshape([1], 1, 1), [1 0 0], reshape([1], 1, 1))
        # The public scientific contract uses Q with shape n×M; unlike the
        # synthetic SK-02/SK-08 (term,axion) reader fixture, it is not M×n.
        q_transposed = [1 0; 0 1; 1 1]
        l_for_n2_m3 = [1 1 1; 0 0 0]
        @test_throws DimensionMismatch SC.prepare_model(q_transposed, l_for_n2_m3, Matrix{Int}(I, 2, 2))
        @test size(SC.prepare_model(transpose(q_transposed), l_for_n2_m3,
            Matrix{Int}(I, 2, 2)).Q) == (2, 3)
        @test_throws ArgumentError SC.prepare_model(reshape([1.5], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1))
        @test_throws ArgumentError SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([NaN], 1, 1))
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
    end
end

include(joinpath(@__DIR__, "fixtures", "scientific_certification", "run_fixture.jl"))

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

@testset "Full public qualification profile" begin
    @test ScientificCertificationFixtures.qualify_from_test_profile!() === nothing
end
