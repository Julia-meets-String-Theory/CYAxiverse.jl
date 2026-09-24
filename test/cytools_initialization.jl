function _cytools_python_stub(root::AbstractString)
    cytools = joinpath(root, "cytools")
    numpy = joinpath(root, "numpy")
    scipy = joinpath(root, "scipy")
    mkpath(cytools)
    mkpath(numpy)
    mkpath(scipy)

    write(joinpath(cytools, "__init__.py"), """
    version = "cytools-fixture"
    class Polytope:
        def __init__(self, points, backend=None):
            self.points_value = points
    class Cone:
        def __init__(self, rays, hyperplanes=None, check=True):
            self.rays = rays
    def fetch_polytopes(*args, **kwargs):
        return []
    """)
    write(joinpath(cytools, "config.py"), """
    import os

    def _record(value):
        with open(os.environ["CYTOOLS_STUB_LOG"], "a", encoding="utf-8") as stream:
            stream.write(value + "\\n")

    def check_mosek_license():
        _record("check_mosek_license")
        if os.path.exists(os.environ["CYTOOLS_STUB_REFRESH_FAIL_FILE"]):
            raise RuntimeError("synthetic refresh failure")

    def set_mosek_path(path):
        _record("set_mosek_path:" + ("file" if os.path.isfile(path) else "not-file"))
        check_mosek_license()

    def mosek_is_activated():
        _record("mosek_is_activated")
        with open(os.environ["CYTOOLS_STUB_ACTIVATION_FILE"], encoding="utf-8") as stream:
            return stream.read().strip() == "1"
    """)
    write(joinpath(numpy, "__init__.py"), "from . import linalg\n")
    write(joinpath(numpy, "linalg.py"), "def matrix_rank(*args, **kwargs): return 0\n")
    write(joinpath(scipy, "__init__.py"), "from . import integrate, optimize\n")
    write(joinpath(scipy, "integrate.py"), "def solve_ivp(*args, **kwargs): return None\n")
    write(joinpath(scipy, "optimize.py"), "def root(*args, **kwargs): return None\n")
    return root
end

function _run_cytools_initialization_smoke()
    Base.find_package("PyCall") === nothing &&
        return false, "PyCall is unavailable in the active Julia test environment"

    mktempdir() do root
        python_root = _cytools_python_stub(joinpath(root, "python"))
        data_root = joinpath(root, "data")
        mkpath(data_root)
        log_path = joinpath(root, "mosek-calls.log")
        license_path = joinpath(root, "mosek.lic")
        activation_path = joinpath(root, "mosek-active")
        refresh_fail_path = joinpath(root, "mosek-refresh-fail")
        write(license_path, "synthetic license fixture")
        write(activation_path, "0")
        repo_root = normpath(joinpath(@__DIR__, ".."))
        active_project = something(Base.active_project(), repo_root)
        child_source = """
            using CYAxiverse
            @assert Base.get_extension(CYAxiverse, :CYAxiversePyCallExt) === nothing
            @assert !any(nameof(module_) == :PyCall for module_ in Base.loaded_modules_array())

            using PyCall
            extension = Base.get_extension(CYAxiverse, :CYAxiversePyCallExt)
            @assert extension !== nothing
            wrapper = extension.cytools_wrapper
            call_log = $(_repr_for_julia(log_path))
            data_root = $(_repr_for_julia(data_root))
            license_path = $(_repr_for_julia(license_path))
            activation_path = $(_repr_for_julia(activation_path))
            refresh_fail_path = $(_repr_for_julia(refresh_fail_path))
            ENV["CYAXIVERSE_DATA_DIR"] = data_root
            ENV["CYAXIVERSE_PYTHON"] = PyCall.python

            read_calls() = isfile(call_log) ? readlines(call_log) : String[]
            expect_disabled(f) = begin
                error_value = try
                    f()
                    nothing
                catch error
                    error
                end
                error_value isa ArgumentError || error("expected disabled-wrapper ArgumentError")
                occursin("CYTools integration is disabled", sprint(showerror, error_value)) ||
                    error("disabled-wrapper error did not identify the enable phase")
            end

            @assert isempty(read_calls())
            expect_disabled(() -> wrapper.cytools_version())
            geom_idx = CYAxiverse.structs.GeometryIndex(h11=1, polytope=1, frst=1)
            expect_disabled(() -> wrapper.hilbert_save(geom_idx, zeros(Int, 1, 1)))
            @assert isempty(read_calls())
            @assert !isfile(CYAxiverse.filestructure.cyax_file(geom_idx))

            extension.enable_cytools!()
            @assert extension.mosek_state() == :ENABLED_INACTIVE_LICENSE_FAILED
            @assert wrapper.cytools_version() == "cytools-fixture"
            @assert wrapper.fetch_polytopes(1, 1) !== nothing
            @assert wrapper.poly([[1, 2]]) !== nothing
            @assert wrapper.cone([[1, 2]]) !== nothing
            after_enable = read_calls()
            @assert after_enable == ["check_mosek_license", "mosek_is_activated"] string(after_enable)

            extension.enable_cytools!()
            @assert read_calls() == after_enable

            write(activation_path, "1")
            @assert extension.refresh_mosek_state!() == :ENABLED_ACTIVE
            @assert read_calls()[end-1:end] == ["check_mosek_license", "mosek_is_activated"]

            write(refresh_fail_path, "1")
            @assert extension.refresh_mosek_state!() == :RESTART_REQUIRED
            rm(refresh_fail_path)

            write(activation_path, "0")
            before_override = read_calls()
            extension.enable_cytools!(; mosek_license_path=license_path)
            @assert extension.mosek_state() == :ENABLED_INACTIVE_LICENSE_FAILED
            override_log = read_calls()
            @assert override_log[length(before_override)+1:end] == [
                "set_mosek_path:file", "check_mosek_license", "mosek_is_activated",
            ]
            extension.enable_cytools!(; mosek_license_path=license_path)
            @assert read_calls() == override_log

            write(activation_path, "1")
            @assert extension.refresh_mosek_state!() == :ENABLED_ACTIVE
            invalid_path_error = try
                extension.enable_cytools!(; mosek_license_path=joinpath(data_root, "missing.lic"))
                nothing
            catch error
                error
            end
            @assert invalid_path_error isa ArgumentError
            @assert !occursin(license_path, sprint(showerror, invalid_path_error))
            @assert extension.mosek_state() == :ENABLED_ACTIVE

            println("CYTools initialization smoke passed")
        """
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$active_project -e $child_source`,
            "PYTHONPATH" => string(
                python_root,
                Sys.iswindows() ? ";" : ":",
                get(ENV, "PYTHONPATH", ""),
            ),
            "CYTOOLS_STUB_LOG" => log_path,
            "CYTOOLS_STUB_ACTIVATION_FILE" => activation_path,
            "CYTOOLS_STUB_REFRESH_FAIL_FILE" => refresh_fail_path,
        )
        output = IOBuffer()
        process = run(pipeline(ignorestatus(command), stdout=output, stderr=output))
        return success(process), String(take!(output))
    end
end

function _repr_for_julia(value::AbstractString)
    return repr(String(value))
end

@testset "CYTools initialization boundary" begin
    extension_path = joinpath(@__DIR__, "..", "ext", "CYAxiversePyCallExt.jl")
    wrapper_path = joinpath(@__DIR__, "..", "add_functions", "cytools_wrapper.jl")
    extension_source = read(extension_path, String)
    wrapper_source = read(wrapper_path, String)

    @test Meta.parseall(extension_source; filename=extension_path) isa Expr
    @test Meta.parseall(wrapper_source; filename=wrapper_path) isa Expr
    @test !occursin("os.environ['HOME']", extension_source)
    @test !occursin("function __init__()", wrapper_source)
    @test occursin("mosek_is_activated()", extension_source)
    @test occursin("pycall(py\"_cyaxiverse_check_mosek\", Bool)", extension_source)
    @test occursin("_ensure_cytools_ready()", wrapper_source)

    if Base.find_package("PyCall") === nothing
        @test_skip false
    else
        passed, output = _run_cytools_initialization_smoke()
        @test passed
        @test occursin("CYTools initialization smoke passed", output)
    end
end
