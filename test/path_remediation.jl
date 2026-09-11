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

function _pluto_cell_body(source::AbstractString, marker::AbstractString)
    lines = split(source, '\n'; keepempty=true)
    starts = findall(line -> startswith(line, "# ╔═╡ "), lines)
    marker_line = "# ╔═╡ $marker"
    for (position, start) in enumerate(starts)
        strip(lines[start]) == marker_line || continue
        stop = position < length(starts) ? starts[position + 1] - 1 : length(lines)
        return join(lines[(start + 1):stop], "\n")
    end
    error("notebook cell marker is missing: $marker")
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
        source
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
        synthetic_scipy = repr("""
        from . import optimize
        """)
        synthetic_integrate = repr("""
        def solve_ivp(*args, **kwargs):
            raise RuntimeError("synthetic scipy.solve_ivp must not run during initialization")
        """)
        synthetic_optimize = repr("""
        def root(*args, **kwargs):
            raise RuntimeError("synthetic scipy.optimize.root must not run during initialization")
        """)
        """
        mktempdir() do synthetic_python_root
            synthetic_cytools_dir = joinpath(synthetic_python_root, "cytools")
            synthetic_scipy_dir = joinpath(synthetic_python_root, "scipy")
            mkpath(synthetic_cytools_dir)
            mkpath(synthetic_scipy_dir)
            write(joinpath(synthetic_cytools_dir, "config.py"), $synthetic_config)
            write(joinpath(synthetic_cytools_dir, "__init__.py"), $synthetic_package)
            write(joinpath(synthetic_scipy_dir, "__init__.py"), $synthetic_scipy)
            write(joinpath(synthetic_scipy_dir, "integrate.py"), $synthetic_integrate)
            write(joinpath(synthetic_scipy_dir, "optimize.py"), $synthetic_optimize)
            ENV["PYTHONPATH"] = string(
                synthetic_python_root,
                Sys.iswindows() ? ";" : ":",
                get(ENV, "PYTHONPATH", ""),
            )
            Base.eval(Main, :(using PyCall))
            pycall = Base.invokelatest(getfield, Main, :PyCall)
            ENV["CYAXIVERSE_PYTHON"] = Base.invokelatest(getproperty, pycall, :python)
            Base.include_string(Main, initialization, notebook)
        end
        """
    else
        "Base.include_string(Main, initialization, notebook)"
    end
    synthetic_contract = if basename(notebook) == "cytools_wrapper_repro.jl"
        """
        mktempdir() do root
            mismatch = joinpath(root, "different", "python")
            mkpath(dirname(mismatch))
            write(mismatch, "synthetic mismatched interpreter")
            cytools = Base.invokelatest(getfield, Main, :CYTools)
            pycall = Base.invokelatest(getfield, Main, :PyCall)
            ENV["CYAXIVERSE_PYTHON"] =
                Base.invokelatest(getproperty, pycall, :python)
            ENV["CYAXIVERSE_PYTHON"] = mismatch
            error_value = try
                enable_cytools = Base.invokelatest(
                    getproperty, cytools, :enable_cytools!)
                Base.invokelatest(enable_cytools)
                nothing
            catch error
                error
            end
            @assert error_value isa ArgumentError
            message = sprint(showerror, error_value)
            @assert occursin("CYAXIVERSE_PYTHON", message)
            @assert occursin("does not rebind", message)
            @assert occursin("Pkg.build(\\\"PyCall\\\")", message)
        end
        """
    elseif basename(notebook) == "stage_production_statistics.jl"
        known_python = something(Sys.which("python3"), "python3")
        known_python_literal = repr(known_python)
        """
        old_configured = get(ENV, "CYAXIVERSE_PYTHON", nothing)
        old_path = get(ENV, "PATH", nothing)
        try
            ENV["CYAXIVERSE_PYTHON"] = $known_python_literal
            @assert parquet_python() == $known_python_literal
            mktempdir() do empty_path
                ENV["CYAXIVERSE_PYTHON"] = joinpath(empty_path, "missing-python")
                ENV["PATH"] = empty_path
                @assert parquet_python() === nothing
            end
        finally
            old_configured === nothing ? delete!(ENV, "CYAXIVERSE_PYTHON") :
                (ENV["CYAXIVERSE_PYTHON"] = old_configured)
            old_path === nothing ? delete!(ENV, "PATH") : (ENV["PATH"] = old_path)
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

function _retired_ledger_summary(path::AbstractString)
    python = something(Sys.which("python3"), "python3")
    parser = "import json,sys; data=json.load(open(sys.argv[1], encoding='utf-8')); print(data['artifact_count']); print(len(data['artifacts'])); print(*[item['path'] for item in data['artifacts']], sep='\\n')"
    output = read(Cmd([python, "-c", parser, path]), String)
    lines = split(chomp(output), '\n'; keepempty=false)
    artifact_count = parse(Int, lines[1])
    listed_count = parse(Int, lines[2])
    artifact_paths = String.(lines[3:end])
    return artifact_count, listed_count, artifact_paths
end

function _retirement_guard_precedes_dependency(path::AbstractString, kind::Symbol)
    lines = split(read(path, String), '\n'; keepempty=true)
    guard_line = if kind === :julia
        findfirst(line -> startswith(strip(line), "error(\"\"\""), lines)
    else
        findfirst(line -> startswith(strip(line), "raise SystemExit("), lines)
    end
    dependency_line = if kind === :julia
        findfirst(line -> startswith(strip(line), "using "), lines)
    else
        findfirst(line -> begin
            stripped = strip(line)
            (startswith(stripped, "import ") || startswith(stripped, "from ")) &&
                !startswith(stripped, "from __future__")
        end, lines)
    end
    return guard_line !== nothing && dependency_line !== nothing && guard_line < dependency_line
end

function _retirement_message_ok(message::AbstractString)
    return occursin(
        "The 2026-08-25 physical-scaling v1 evidence workflow is retired and cannot be run.",
        message,
    ) && occursin(
        "See validation/RETIRED_physical_scaling_evidence_20260825.md.", message) &&
        occursin("new, path-safe evidence version", message)
end

function _retirement_sentinels_unchanged(
    paths::AbstractVector, expected::AbstractVector)
    return length(paths) == length(expected) && all(
        isfile(path) && read(path) == expected[index]
        for (index, path) in enumerate(paths)
    )
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
        if name == "stage_production_statistics.jl"
            initialization = string(
                initialization,
                "\n",
                _pluto_cell_body(
                    read(path, String), "1dadfd63-8a43-44df-8d66-76ae3ec3dcf6"),
            )
        end
        passed, output = _run_notebook_initialization_smoke(path, initialization)
        @test passed
        @test occursin("notebook initialization runtime smoke passed", output)
    end
end

@testset "Retired 2026-08-25 evidence entrypoints fail closed" begin
    retired_entrypoints = [
        (:julia, "audit_physical_certificate.jl"),
        (:julia, "generate_physical_scaling_sidecars_20260825.jl"),
        (:julia, "preflight_physical_scaling_20260825.jl"),
        (:julia, "run_physical_scale_inflation_pilot_20260825.jl"),
        (:python, "test_physical_scale_checkpoint_resume_20260825.py"),
        (:python, "test_physical_scaling_sidecars_20260825.py"),
        (:python, "validate_physical_scale_checkpoint_20260825.py"),
        (:python, "validate_physical_scaling_sidecars_20260825.py"),
    ]
    script_dir = joinpath(@__DIR__, "..", "scripts")
    ledger_path = joinpath(
        @__DIR__, "..", "validation", "retired_physical_scaling_evidence_20260825.json")
    artifact_count, listed_count, artifact_paths = _retired_ledger_summary(ledger_path)
    @test artifact_count == 30
    @test listed_count == artifact_count
    @test length(artifact_paths) == artifact_count
    @test length(unique(artifact_paths)) == artifact_count
    @test all(path -> !isabspath(path) && startswith(path, "validation/"), artifact_paths)

    mktempdir() do root
        output_root = joinpath(root, "configured-output-root")
        synthetic_repo = joinpath(root, "retired_repo")
        synthetic_script_dir = joinpath(synthetic_repo, "scripts")
        mkpath(synthetic_script_dir)
        for (_, name) in retired_entrypoints
            cp(joinpath(script_dir, name), joinpath(synthetic_script_dir, name); force=true)
        end

        for (kind, name) in retired_entrypoints
            original = joinpath(script_dir, name)
            copied = joinpath(synthetic_script_dir, name)
            @test _retirement_guard_precedes_dependency(original, kind)
            @test _retirement_guard_precedes_dependency(copied, kind)
        end

        sentinel_paths = [joinpath(synthetic_repo, path) for path in artifact_paths]
        sentinel_bytes = [
            Vector{UInt8}(codeunits("retired-v1-sentinel-$index-$path\n"))
            for (index, path) in enumerate(artifact_paths)
        ]
        for (path, bytes) in zip(sentinel_paths, sentinel_bytes)
            mkpath(dirname(path))
            write(path, bytes)
        end
        @test _retirement_sentinels_unchanged(sentinel_paths, sentinel_bytes)

        python = something(Sys.which("python3"), "python3")
        for (kind, name) in retired_entrypoints
            script = joinpath(synthetic_script_dir, name)
            command = if kind === :julia
                addenv(
                    `$(Base.julia_cmd()) --startup-file=no $script --output-root $output_root`,
                    "JULIA_LOAD_PATH" => "@stdlib",
                    "CYAXIVERSE_AUDIT_OUTPUT_DIR" => output_root,
                    "CYAXIVERSE_AUDIT_REPOSITORY" => synthetic_repo,
                    "CYAXIVERSE_DATA_DIR" => joinpath(synthetic_repo, "data"),
                    "CYAXIVERSE_PILOT_REPOSITORY" => synthetic_repo,
                    "CYAXIVERSE_PROJECT_ROOT" => synthetic_repo,
                )
            else
                addenv(
                    Cmd([python, script]),
                    "PYTHONNOUSERSITE" => "1",
                    "PYTHONPATH" => "",
                    "CYAXIVERSE_AUDIT_OUTPUT_DIR" => output_root,
                    "CYAXIVERSE_AUDIT_REPOSITORY" => synthetic_repo,
                    "CYAXIVERSE_DATA_DIR" => joinpath(synthetic_repo, "data"),
                    "CYAXIVERSE_PHYSICAL_SCALE_FIXTURE_ROOT" => joinpath(synthetic_repo, "fixture"),
                    "CYAXIVERSE_PILOT_REPOSITORY" => synthetic_repo,
                    "CYAXIVERSE_PROJECT_ROOT" => synthetic_repo,
                )
            end
            output = IOBuffer()
            process = run(pipeline(ignorestatus(command), stdout=output, stderr=output))
            message = String(take!(output))
            @test !success(process)
            @test _retirement_message_ok(message)
            @test _retirement_sentinels_unchanged(sentinel_paths, sentinel_bytes)
            @test !ispath(output_root)
        end
    end
end
