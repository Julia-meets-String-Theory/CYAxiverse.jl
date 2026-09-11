function _pluto_cell_bodies(source::AbstractString)
    lines = split(source, '\n'; keepempty=true)
    starts = findall(line -> startswith(line, "# ╔═╡ "), lines)
    cells = String[]
    for (position, start) in enumerate(starts)
        marker = strip(lines[start])
        marker == "# ╔═╡ Cell order:" && continue
        stop = position < length(starts) ? starts[position + 1] - 1 : length(lines)
        push!(cells, join(lines[(start + 1):stop], "\n"))
    end
    return cells
end

function _notebook_display_cell(body::AbstractString)
    stripped = strip(body)
    return startswith(stripped, "md\"") || startswith(stripped, "html\"")
end

function _notebook_initialization_source(path::AbstractString)
    source = read(path, String)
    cells = _pluto_cell_bodies(source)
    activation = findfirst(cell -> occursin("Pkg.activate(@__DIR__)", cell), cells)
    activation === nothing && error("notebook has no Pkg.activate(@__DIR__) cell: $path")

    last_initialization_cell = activation
    for index in (activation + 1):length(cells)
        _notebook_display_cell(cells[index]) && break
        last_initialization_cell = index
    end
    return join(cells[activation:last_initialization_cell], "\n")
end

function _run_notebook_initialization_smoke(path::AbstractString, source::AbstractString)
    notebook = abspath(path)
    notebook_dir = dirname(notebook)
    smoke_source = if basename(notebook) == "cytools_wrapper_repro.jl"
        enable_call = findfirst("CYTools.enable_cytools!()", source)
        enable_call === nothing && error("CYTools enable call is missing from $notebook")
        prefix_end = prevind(source, first(enable_call))
        string(source[begin:prefix_end], "\nend")
    else
        source
    end
    include_source = if basename(notebook) == "cytools_wrapper_repro.jl"
        synthetic_config = repr("""
        mosek_is_activated = True
        def set_mosek_path(path):
            return None
        def check_mosek_license():
            return None
        """)
        synthetic_package = repr("""
        from .config import *
        version = "synthetic"
        def fetch_polytopes(*args, **kwargs):
            return []
        class Polytope:
            pass
        class Cone:
            pass
        """)
        """
        mktempdir() do synthetic_python_root
            synthetic_cytools_dir = joinpath(synthetic_python_root, "cytools")
            mkpath(synthetic_cytools_dir)
            write(joinpath(synthetic_cytools_dir, "config.py"), $synthetic_config)
            write(joinpath(synthetic_cytools_dir, "__init__.py"), $synthetic_package)
            ENV["PYTHONPATH"] = string(
                synthetic_python_root,
                Sys.iswindows() ? ";" : ":",
                get(ENV, "PYTHONPATH", ""),
            )
            Base.include_string(Main, initialization, notebook)
        end
        """
    else
        "Base.include_string(Main, initialization, notebook)"
    end
    synthetic_contract = if basename(notebook) == "cytools_wrapper_repro.jl"
        """
        using CYAxiverse
        mktempdir() do root
            requested = joinpath(root, "requested", "python")
            effective = joinpath(root, "effective", "python")
            mkpath(dirname(requested))
            mkpath(dirname(effective))
            write(requested, "synthetic requested interpreter")
            write(effective, "synthetic effective interpreter")
            @assert CYAxiverse.python_interpreter.check_configured_python(
                effective; environment=Dict{String,String}()) == effective
            error_value = try
                CYAxiverse.python_interpreter.check_configured_python(
                    effective; environment=Dict("CYAXIVERSE_PYTHON" => requested))
                nothing
            catch error
                error
            end
            @assert error_value isa ArgumentError
            message = sprint(showerror, error_value)
            @assert occursin("CYAXIVERSE_PYTHON", message)
            @assert occursin("does not rebind", message)
        end
        """
    else
        ""
    end
    command_source = """
    notebook = $(repr(notebook))
    initialization = $(repr(smoke_source))
    $include_source
    @assert normpath(Base.active_project()) == normpath(joinpath(dirname(notebook), "Project.toml"))
    $synthetic_contract
    println("notebook initialization runtime smoke passed: ", basename(notebook))
    """
    command = addenv(
        `$(Base.julia_cmd()) --startup-file=no --project=$notebook_dir -e $command_source`,
        "JULIA_LOAD_PATH" => "@:@stdlib",
    )
    output = IOBuffer()
    process = run(pipeline(ignorestatus(command), stdout=output, stderr=output))
    return success(process), String(take!(output))
end

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

@testset "Changed notebook configuration static checks" begin
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

@testset "Changed notebook initialization runtime smoke" begin
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
        initialization = _notebook_initialization_source(path)
        passed, output = _run_notebook_initialization_smoke(path, initialization)
        @test passed
        @test occursin("notebook initialization runtime smoke passed", output)
    end
end

@testset "Retired 2026-08-25 evidence producers fail closed" begin
    julia_producer_names = [
        "audit_physical_certificate.jl",
        "generate_physical_scaling_sidecars_20260825.jl",
        "preflight_physical_scaling_20260825.jl",
        "run_physical_scale_inflation_pilot_20260825.jl",
    ]
    python_entrypoint_names = [
        "test_physical_scale_checkpoint_resume_20260825.py",
        "test_physical_scaling_sidecars_20260825.py",
        "validate_physical_scale_checkpoint_20260825.py",
        "validate_physical_scaling_sidecars_20260825.py",
    ]
    script_dir = joinpath(@__DIR__, "..", "scripts")
    mktempdir() do root
        output_root = joinpath(root, "outputs")
        for name in julia_producer_names
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
            @test occursin(
                "The 2026-08-25 physical-scaling v1 evidence workflow is retired and cannot be run.",
                message,
            )
            @test occursin(
                "See validation/RETIRED_physical_scaling_evidence_20260825.md.", message)
            @test occursin("new, path-safe evidence version", message)
            @test !ispath(output_root)
        end

        synthetic_repo = joinpath(root, "retired_repo")
        synthetic_script_dir = joinpath(synthetic_repo, "scripts")
        synthetic_validation_dir = joinpath(synthetic_repo, "validation")
        mkpath(synthetic_script_dir)
        mkpath(synthetic_validation_dir)
        for name in python_entrypoint_names
            cp(joinpath(script_dir, name), joinpath(synthetic_script_dir, name); force=true)
        end
        sentinel_paths = Dict(
            "test_physical_scaling_sidecars_20260825.py" =>
                joinpath(synthetic_validation_dir, "physical_scaling_sidecar_tests_20260825.json"),
            "test_physical_scale_checkpoint_resume_20260825.py" =>
                joinpath(synthetic_validation_dir, "physical_scale_checkpoint_tests_20260825.json"),
        )
        sentinel_bytes = Dict(
            name => Vector{UInt8}(codeunits("pre-existing sentinel for $name\n"))
            for name in keys(sentinel_paths)
        )
        for (name, path) in sentinel_paths
            write(path, sentinel_bytes[name])
        end

        python = something(Sys.which("python3"), "python3")
        for name in python_entrypoint_names
            script = joinpath(synthetic_script_dir, name)
            command = addenv(
                Cmd([python, script]),
                "CYAXIVERSE_AUDIT_OUTPUT_DIR" => output_root,
                "CYAXIVERSE_AUDIT_REPOSITORY" => joinpath(synthetic_repo, "repository"),
                "CYAXIVERSE_DATA_DIR" => joinpath(synthetic_repo, "data"),
                "CYAXIVERSE_PHYSICAL_SCALE_FIXTURE_ROOT" => joinpath(synthetic_repo, "fixture"),
                "CYAXIVERSE_PILOT_REPOSITORY" => joinpath(synthetic_repo, "repository"),
                "CYAXIVERSE_PROJECT_ROOT" => synthetic_repo,
            )
            output = IOBuffer()
            process = run(pipeline(ignorestatus(command), stdout=output, stderr=output))
            message = String(take!(output))
            @test !success(process)
            @test occursin(
                "The 2026-08-25 physical-scaling v1 evidence workflow is retired and cannot be run.",
                message,
            )
            @test occursin(
                "See validation/RETIRED_physical_scaling_evidence_20260825.md.",
                message,
            )
            @test occursin("new, path-safe evidence version", message)
            @test !ispath(output_root)
            if haskey(sentinel_paths, name)
                path = sentinel_paths[name]
                @test isfile(path)
                @test isfile(path) && read(path) == sentinel_bytes[name]
            end
        end
    end
end
