@testset "Optional Python interpreter contract" begin
    interpreter = CYAxiverse.python_interpreter
    effective = joinpath("synthetic", "effective", "python")

    @test interpreter.desired_python(Dict{String,String}()) === nothing
    @test interpreter.check_configured_python(
        effective; environment=Dict{String,String}()) == effective

    mktempdir() do root
        requested = joinpath(root, "requested", "python")
        effective = joinpath(root, "effective", "python")
        mkpath(dirname(requested))
        mkpath(dirname(effective))
        write(requested, "synthetic requested interpreter")
        write(effective, "synthetic effective interpreter")

        @test interpreter.check_python_interpreter(requested, requested) == requested
        error_value = try
            interpreter.check_python_interpreter(requested, effective)
            nothing
        catch error
            error
        end
        @test error_value isa ArgumentError
        message = sprint(showerror, error_value)
        @test occursin("CYAXIVERSE_PYTHON", message)
        @test occursin("does not rebind", message)
        @test occursin("Pkg.build(\"PyCall\")", message)
        @test occursin("restart Julia", message)
        @test occursin("rebuild PyCall automatically", message)
    end
end

@testset "Changed notebook configuration smoke" begin
    notebook_names = [
        "cytools_wrapper_repro.jl",
        "optim_testing backup 1.jl",
        "optim_testing.jl",
        "stage_production_statistics.jl",
        "vacua_pipeline.jl",
    ]
    notebook_dir = joinpath(@__DIR__, "..", "notebooks")
    for name in notebook_names
        path = joinpath(notebook_dir, name)
        source = read(path, String)
        @test Meta.parseall(source; filename=path) isa Expr
        @test occursin("Pkg.activate(@__DIR__)", source)
        @test !occursin("ENV[\"PYTHON\"] =", source)
    end

    cytools_source = read(joinpath(notebook_dir, "cytools_wrapper_repro.jl"), String)
    @test occursin("using PyCall", cytools_source)
    @test occursin("CYTools.enable_cytools!()", cytools_source)
    vacua_source = read(joinpath(notebook_dir, "vacua_pipeline.jl"), String)
    @test !occursin("CYAXIVERSE_PYTHON", vacua_source)

    init_source = read(joinpath(@__DIR__, "..", "src", "init_python.jl"), String)
    @test Meta.parseall(init_source; filename="src/init_python.jl") isa Expr
    @test !occursin("Pkg.build", init_source)
end

@testset "Retired 2026-08-25 evidence producers fail closed" begin
    producer_names = [
        "audit_physical_certificate.jl",
        "generate_physical_scaling_sidecars_20260825.jl",
        "preflight_physical_scaling_20260825.jl",
        "run_physical_scale_inflation_pilot_20260825.jl",
    ]
    script_dir = joinpath(@__DIR__, "..", "scripts")
    mktempdir() do root
        output_root = joinpath(root, "outputs")
        for name in producer_names
            script = joinpath(script_dir, name)
            command = addenv(
                `$(Base.julia_cmd()) --startup-file=no $script --output-root $output_root`,
                "CYAXIVERSE_AUDIT_OUTPUT_DIR" => output_root,
                "CYAXIVERSE_AUDIT_REPOSITORY" => joinpath(root, "repository"),
                "CYAXIVERSE_DATA_DIR" => joinpath(root, "data"),
                "CYAXIVERSE_PILOT_REPOSITORY" => joinpath(root, "repository"),
                "CYAXIVERSE_PROJECT_ROOT" => joinpath(root, "project"),
            )
            output = IOBuffer()
            process = run(pipeline(ignorestatus(command), stdout=output, stderr=output))
            message = String(take!(output))
            @test !success(process)
            @test occursin("evidence workflow is retired and cannot be run", message)
            @test occursin(
                "validation/RETIRED_physical_scaling_evidence_20260825.md", message)
            @test !ispath(output_root)
        end
    end
end
