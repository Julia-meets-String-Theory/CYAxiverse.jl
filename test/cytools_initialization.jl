function _cytools_python_stub(root::AbstractString)
    cytools = joinpath(root, "cytools")
    numpy = joinpath(root, "numpy")
    scipy = joinpath(root, "scipy")
    mkpath(cytools)
    mkpath(numpy)
    mkpath(scipy)

    write(joinpath(cytools, "__init__.py"), """
    import os
    version = "cytools-fixture"
    def _record_solver(value):
        with open(os.environ["CYTOOLS_STUB_SOLVER_LOG"], "a", encoding="utf-8") as stream:
            stream.write(value + "\\n")

    class Polytope:
        def __init__(self, points, backend=None):
            self.points_value = points
    class Cone:
        def __init__(self, rays, hyperplanes=None, check=True):
            _record_solver("cone_constructor")
            self.rays = rays
        def hilbert_basis(self):
            _record_solver("hilbert_basis")
            if os.path.exists(os.environ["CYTOOLS_STUB_OPERATION_FAIL_FILE"]):
                raise RuntimeError(os.environ["CYTOOLS_STUB_FAILURE_DETAIL"])
            return [[1, 0], [0, 1]]
    class _GeometryTipCone:
        def tip_of_stretched_cone(self, *args, **kwargs):
            if os.path.exists(os.environ["CYTOOLS_STUB_OPERATION_FAIL_FILE"]):
                raise RuntimeError(os.environ["CYTOOLS_STUB_FAILURE_DETAIL"])
    class GeometryFixture:
        def h21(self): return 1
        def glsm_charge_matrix(self, include_origin=False): return [[1]]
        def divisor_basis(self): return [0]
        def toric_kahler_cone(self): return _GeometryTipCone()
    def fetch_polytopes(*args, **kwargs):
        return []
    def select_optimizer_for_test():
        from cytools import config
        backend = "mosek" if config.mosek_is_activated() else "highs"
        _record_solver("selected_backend:" + backend)
        return backend
    """)
    write(joinpath(cytools, "config.py"), """
    import os

    def _record(value):
        with open(os.environ["CYTOOLS_STUB_LOG"], "a", encoding="utf-8") as stream:
            stream.write(value + "\\n")

    def check_mosek_license():
        _record("check_mosek_license")
        if os.path.exists(os.environ["CYTOOLS_STUB_REFRESH_FAIL_FILE"]):
            raise RuntimeError(os.environ["CYTOOLS_STUB_FAILURE_DETAIL"])

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
        solver_log_path = joinpath(root, "solver-calls.log")
        license_path = joinpath(root, "mosek.lic")
        activation_path = joinpath(root, "mosek-active")
        refresh_fail_path = joinpath(root, "mosek-refresh-fail")
        operation_fail_path = joinpath(root, "operation-fail")
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
            solver_call_log = $(_repr_for_julia(solver_log_path))
            data_root = $(_repr_for_julia(data_root))
            license_path = $(_repr_for_julia(license_path))
            activation_path = $(_repr_for_julia(activation_path))
            refresh_fail_path = $(_repr_for_julia(refresh_fail_path))
            operation_fail_path = $(_repr_for_julia(operation_fail_path))
            ENV["CYAXIVERSE_DATA_DIR"] = data_root
            ENV["CYAXIVERSE_PYTHON"] = PyCall.python

            read_calls() = isfile(call_log) ? readlines(call_log) : String[]
            read_solver_calls() = isfile(solver_call_log) ? readlines(solver_call_log) : String[]
            select_test_optimizer() = pycall(pyimport("cytools")[:select_optimizer_for_test], String)
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
            @assert isempty(read_solver_calls())
            mismatched_python_path = joinpath(data_root, "private-python")
            ENV["CYAXIVERSE_PYTHON"] = mismatched_python_path
            interpreter_error = try
                extension.enable_cytools!()
                nothing
            catch error
                error
            end
            @assert interpreter_error isa ArgumentError
            interpreter_diagnostic = extension.mosek_diagnostic()
            @assert occursin("interpreter validation", interpreter_diagnostic)
            @assert occursin("ArgumentError", interpreter_diagnostic)
            @assert occursin("does not rebind", interpreter_diagnostic)
            @assert occursin("restart Julia", interpreter_diagnostic)
            @assert occursin("enable_cytools!()", interpreter_diagnostic)
            @assert !occursin(mismatched_python_path, interpreter_diagnostic)
            @assert !occursin(mismatched_python_path, sprint(showerror, interpreter_error))
            @assert extension.mosek_state() == :CYTOOLS_DISABLED
            ENV["CYAXIVERSE_PYTHON"] = PyCall.python
            expect_disabled(() -> wrapper.cytools_version())
            geom_idx = CYAxiverse.structs.GeometryIndex(h11=1, polytope=1, frst=1)
            expected_geometry_dir = joinpath(data_root, "h11_001", "np_0000001", "cy_0000001")
            mkpath(expected_geometry_dir)
            h5_path = CYAxiverse.filestructure.cyax_file(geom_idx)
            @assert h5_path == joinpath(expected_geometry_dir, "cyax.h5") string(h5_path)
            using HDF5
            h5open(h5_path, "w") do file
                geometric = create_group(create_group(file, "cytools"), "geometric")
                geometric["sentinel"] = [7, 11]
            end
            before_disabled_save = read(h5_path)
            expect_disabled(() -> wrapper.hilbert_save(geom_idx, zeros(Int, 1, 1)))
            @assert isempty(read_calls())
            @assert isempty(read_solver_calls())
            @assert read(h5_path) == before_disabled_save
            h5open(h5_path, "r") do file
                @assert read(file["cytools/geometric/sentinel"]) == [7, 11]
            end

            extension.enable_cytools!()
            @assert extension.mosek_state() == :ENABLED_INACTIVE_LICENSE_FAILED
            @assert wrapper.cytools_version() == "cytools-fixture"
            @assert wrapper.fetch_polytopes(1, 1) !== nothing
            @assert wrapper.poly([[1, 2]]) !== nothing
            @assert wrapper.cone([[1, 2]]) !== nothing
            @assert wrapper.hilbert_basis([1 0; 0 1]) !== nothing
            after_hilbert_basis = read_solver_calls()
            @assert after_hilbert_basis == ["cone_constructor", "cone_constructor", "hilbert_basis"] string(after_hilbert_basis)

            # Saving a basis is HDF5-only: it must not consult a Python solver.
            basis = [1 0; 0 1]
            wrapper.hilbert_save(geom_idx, basis)
            @assert read_solver_calls() == after_hilbert_basis
            h5open(h5_path, "r") do file
                @assert read(file["cytools/geometric/sentinel"]) == [7, 11]
                dataset = file["cytools/geometric/hilbert_basis"]
                @assert read(dataset) == basis
                filters = collect(HDF5.get_create_properties(dataset).filters)
                @assert any(filter -> filter isa HDF5.Filters.Deflate && filter.level == 9, filters)
            end
            after_enable = read_calls()
            @assert after_enable == ["check_mosek_license", "mosek_is_activated"] string(after_enable)
            inactive_diagnostic = extension.mosek_diagnostic()
            @assert occursin("activation was false", inactive_diagnostic)
            @assert occursin("refresh_mosek_state!()", inactive_diagnostic)
            @assert !occursin(license_path, inactive_diagnostic)
            @assert extension._run_cytools_operation(select_test_optimizer, :fair_triangulation) == "highs"
            inactive_solver_calls = read_solver_calls()
            @assert last(inactive_solver_calls) == "selected_backend:highs"
            # The CYTools 1.4.12 matrix has no mandatory-MOSEK row; exercise
            # the operation boundary with a separately named synthetic case.
            inactive_mandatory_error = try
                extension._run_cytools_operation(select_test_optimizer, :fixture_mandatory_mosek;
                    requires_mosek=true)
                nothing
            catch error
                error
            end
            inactive_mandatory_message = sprint(showerror, inactive_mandatory_error)
            @assert inactive_mandatory_error isa ErrorException
            @assert occursin("operation `fixture_mandatory_mosek` requires active MOSEK", inactive_mandatory_message)
            @assert occursin("state: ENABLED_INACTIVE_LICENSE_FAILED", inactive_mandatory_message)
            @assert read_solver_calls() == inactive_solver_calls
            after_enable = read_calls()

            write(operation_fail_path, "fail")
            operation_error = try
                wrapper.hilbert_basis([1 0; 0 1])
                nothing
            catch error
                error
            end
            rm(operation_fail_path)
            operation_message = sprint(showerror, operation_error)
            @assert operation_error isa ErrorException
            @assert occursin("downstream wrapped operation `hilbert_basis` failed", operation_message)
            @assert occursin("cause type: PyError", operation_message)
            @assert occursin("Normaliz", operation_message)
            @assert occursin("MOSEK activation/configuration state: ENABLED_INACTIVE_LICENSE_FAILED", operation_message)
            @assert occursin("supported fallback", operation_message)
            @assert !occursin(license_path, operation_message)
            @assert !occursin(license_path, extension.mosek_diagnostic())
            @assert extension.mosek_state() == :ENABLED_INACTIVE_LICENSE_FAILED

            geometry_fixture = pycall(pyimport("cytools")[:GeometryFixture], PyObject)
            write(operation_fail_path, "fail")
            geometry_error = try
                wrapper.geometries_generate(1, geometry_fixture)
                nothing
            catch error
                error
            end
            rm(operation_fail_path)
            geometry_message = sprint(showerror, geometry_error)
            @assert geometry_error isa ErrorException
            @assert occursin("downstream wrapped operation `standard_geometry_tip` failed", geometry_message)
            @assert occursin("cause type: PyError", geometry_message)
            @assert occursin("tip_of_stretched_cone", geometry_message)
            @assert occursin("OSQP below dimension 25", geometry_message)
            @assert occursin("Highs otherwise", geometry_message)
            @assert occursin("MOSEK activation/configuration state: ENABLED_INACTIVE_LICENSE_FAILED", geometry_message)
            @assert !occursin(license_path, geometry_message)
            @assert extension.mosek_state() == :ENABLED_INACTIVE_LICENSE_FAILED

            extension.enable_cytools!()
            @assert read_calls() == after_enable

            write(activation_path, "1")
            @assert extension.refresh_mosek_state!() == :ENABLED_ACTIVE
            @assert read_calls()[end-1:end] == ["check_mosek_license", "mosek_is_activated"]
            @assert extension._run_cytools_operation(select_test_optimizer, :fair_triangulation) == "mosek"
            active_solver_calls = read_solver_calls()
            @assert last(active_solver_calls) == "selected_backend:mosek"

            write(refresh_fail_path, "1")
            @assert extension.refresh_mosek_state!() == :RESTART_REQUIRED
            refresh_diagnostic = extension.mosek_diagnostic()
            @assert occursin("license refresh", refresh_diagnostic)
            @assert occursin("PyError", refresh_diagnostic)
            @assert occursin("restart Julia", refresh_diagnostic)
            @assert occursin("enable_cytools!()", refresh_diagnostic)
            @assert !occursin(license_path, refresh_diagnostic)
            @assert extension.mosek_state() == :RESTART_REQUIRED
            @assert wrapper.cytools_version() == "cytools-fixture"
            @assert extension._run_cytools_operation(select_test_optimizer, :fair_triangulation) == "highs"
            fallback_solver_calls = read_solver_calls()
            @assert fallback_solver_calls == vcat(active_solver_calls, "selected_backend:highs")
            calls_before_mandatory_failure = copy(fallback_solver_calls)
            mandatory_error = try
                extension._run_cytools_operation(select_test_optimizer, :fixture_mandatory_mosek;
                    requires_mosek=true)
                nothing
            catch error
                error
            end
            mandatory_message = sprint(showerror, mandatory_error)
            @assert mandatory_error isa ErrorException
            @assert occursin("operation `fixture_mandatory_mosek` requires active MOSEK", mandatory_message)
            @assert occursin("state: RESTART_REQUIRED", mandatory_message)
            @assert occursin("No downstream operation was started", mandatory_message)
            @assert read_solver_calls() == calls_before_mandatory_failure
            @assert extension.mosek_state() == :RESTART_REQUIRED
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
            @assert extension._run_cytools_operation(select_test_optimizer, :fair_triangulation) == "mosek"
            invalid_path_error = try
                extension.enable_cytools!(; mosek_license_path=joinpath(data_root, "missing.lic"))
                nothing
            catch error
                error
            end
            @assert invalid_path_error isa ArgumentError
            @assert !occursin(license_path, sprint(showerror, invalid_path_error))
            @assert extension.mosek_state() == :ENABLED_ACTIVE

            alternate_license_path = joinpath(data_root, "alternate-mosek.lic")
            write(alternate_license_path, "synthetic alternate license fixture")
            write(refresh_fail_path, "1")
            extension.enable_cytools!(; mosek_license_path=alternate_license_path)
            @assert extension.mosek_state() == :RESTART_REQUIRED
            override_diagnostic = extension.mosek_diagnostic()
            @assert occursin("license path override", override_diagnostic)
            @assert occursin("PyError", override_diagnostic)
            @assert occursin("restart Julia", override_diagnostic)
            @assert occursin("enable_cytools!()", override_diagnostic)
            @assert !occursin(license_path, override_diagnostic)
            @assert !occursin(alternate_license_path, override_diagnostic)
            rm(refresh_fail_path)

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
            "CYTOOLS_STUB_SOLVER_LOG" => solver_log_path,
            "CYTOOLS_STUB_FAILURE_DETAIL" => license_path,
            "CYTOOLS_STUB_ACTIVATION_FILE" => activation_path,
            "CYTOOLS_STUB_REFRESH_FAIL_FILE" => refresh_fail_path,
            "CYTOOLS_STUB_OPERATION_FAIL_FILE" => operation_fail_path,
        )
        output = IOBuffer()
        process = run(pipeline(ignorestatus(command), stdout=output, stderr=output))
        return success(process), _sanitize_cytools_output(String(take!(output)); fixture_root=root)
    end
end

function _repr_for_julia(value::AbstractString)
    return repr(String(value))
end

function _sanitize_cytools_output(output::AbstractString; fixture_root=nothing)
    sanitized = String(output)
    home = get(ENV, "HOME", "")
    isempty(home) || (sanitized = replace(sanitized, home => "<HOME>"))
    fixture_root === nothing ||
        (sanitized = replace(sanitized, String(fixture_root) => "<fixture>"))
    return sanitized
end

function _julia_function_block(source::AbstractString, signature::AbstractString)
    lines = split(source, '\n'; keepempty=true)
    start = findfirst(line -> occursin(signature, strip(line)), lines)
    start === nothing && error("function signature not found: $signature")
    stop = findnext(index -> startswith(strip(lines[index]), "function "),
        eachindex(lines), start + 1)
    last = stop === nothing ? length(lines) : stop - 1
    return join(lines[start:last], "\n")
end

function _run_cytools_import_failure_smoke()
    Base.find_package("PyCall") === nothing && return false, "PyCall is unavailable"
    mktempdir() do root
        python_root = _cytools_python_stub(joinpath(root, "python"))
        secret_path = joinpath(root, "private-license.lic")
        write(secret_path, "synthetic license fixture")
        write(joinpath(python_root, "cytools", "__init__.py"),
            "raise RuntimeError(" * _repr_for_julia(secret_path) * ")\n")
        active_project = something(Base.active_project(), normpath(joinpath(@__DIR__, "..")))
        child_source = """
            using CYAxiverse
            using PyCall
            extension = Base.get_extension(CYAxiverse, :CYAxiversePyCallExt)
            secret_path = $(_repr_for_julia(secret_path))
            ENV["CYAXIVERSE_PYTHON"] = PyCall.python
            error_value = try
                extension.enable_cytools!()
                nothing
            catch error
                error
            end
            @assert error_value isa ErrorException
            @assert extension.mosek_state() == :CYTOOLS_DISABLED
            diagnostic = extension.mosek_diagnostic()
            @assert occursin("Python import and wrapper setup", diagnostic)
            @assert occursin("PyError", diagnostic)
            @assert occursin("restart Julia", diagnostic)
            @assert occursin("enable_cytools!()", diagnostic)
            @assert !occursin(secret_path, diagnostic)
            @assert !occursin(secret_path, sprint(showerror, error_value))
            println("CYTools import failure diagnostic smoke passed")
        """
        command = addenv(
            `$(Base.julia_cmd()) --startup-file=no --project=$active_project -e $child_source`,
            "PYTHONPATH" => string(
                python_root,
                Sys.iswindows() ? ";" : ":",
                get(ENV, "PYTHONPATH", ""),
            ),
        )
        output = IOBuffer()
        process = run(pipeline(ignorestatus(command), stdout=output, stderr=output))
        result = _sanitize_cytools_output(String(take!(output)); fixture_root=root)
        return success(process), result
    end
end

function _run_cytools_submodule_load_failure_smoke()
    Base.find_package("PyCall") === nothing && return false, "PyCall is unavailable"
    mktempdir() do root
        secret_path = joinpath(root, "private-module-detail")
        submodule_path = joinpath(root, "broken-submodule.jl")
        write(submodule_path, "error(" * _repr_for_julia(secret_path) * ")\n")
        active_project = something(Base.active_project(), normpath(joinpath(@__DIR__, "..")))
        child_source = """
            using CYAxiverse
            using PyCall
            extension = Base.get_extension(CYAxiverse, :CYAxiversePyCallExt)
            secret_path = $(_repr_for_julia(secret_path))
            submodule_path = $(_repr_for_julia(submodule_path))
            error_value = try
                extension._load_extension_submodule!(submodule_path, :cytools_wrapper)
                nothing
            catch error
                error
            end
            @assert error_value isa ErrorException
            diagnostic = extension.mosek_diagnostic()
            @assert occursin("extension/submodule load failed", diagnostic)
            @assert occursin("cytools_wrapper", diagnostic)
            @assert occursin("cause type: ErrorException", diagnostic)
            @assert occursin("enable_cytools!()", diagnostic)
            @assert occursin("MOSEK activation/configuration", diagnostic)
            @assert occursin("No downstream wrapped operation/backend selection was reached", diagnostic)
            @assert !occursin(secret_path, diagnostic)
            @assert !occursin(submodule_path, diagnostic)
            @assert !occursin(secret_path, sprint(showerror, error_value))
            @assert extension.mosek_state() == :CYTOOLS_DISABLED
            println("CYTools submodule-load diagnostic smoke passed")
        """
        command = `$(Base.julia_cmd()) --startup-file=no --project=$active_project -e $child_source`
        output = IOBuffer()
        process = run(pipeline(ignorestatus(command), stdout=output, stderr=output))
        result = _sanitize_cytools_output(String(take!(output)); fixture_root=root)
        return success(process), result
    end
end

# Candidate-bound source observations from installed CYTools 1.4.12. The
# optional source probe below checks the key Python call-chain anchors without
# constructing scientific objects or invoking an optimizer.
const _CYTOOLS_1_4_12_CAPABILITY_MATRIX = (
    (operation=:fast_triangulation, readiness_required=true, direct_optimizer=false,
        transitive_optimizer=false, backend="CGAL by default",
        mandatory_mosek=false,
        fallback="No optimizer fallback is needed on the fast heights path.",
        failure="Triangulation backend errors or the generator retry limit.",
        runtime_observation="No real geometry or optimizer execution; CYTools 1.4.12 source probe only."),
    (operation=:fair_triangulation, readiness_required=true, direct_optimizer=false,
        transitive_optimizer=true, backend="Cone.is_solid -> find_interior_point; Mosek only when active and dimension >= 25, otherwise Highs",
        mandatory_mosek=false,
        fallback="Highs when MOSEK is inactive, state is RESTART_REQUIRED, or dimension is below 25.",
        failure="Cone feasibility may fail or return a false negative; fair walks can stall or raise RuntimeError.",
        runtime_observation="No real geometry or optimizer execution; CYTools 1.4.12 source probe only."),
    (operation=:stored_simplices_reconstruction, readiness_required=true, direct_optimizer=false,
        transitive_optimizer=true, backend="Triangulation.is_valid -> Cone.is_solid -> find_interior_point; Mosek only when active and dimension >= 25, otherwise Highs",
        mandatory_mosek=false,
        fallback="Highs when MOSEK is inactive, state is RESTART_REQUIRED, or dimension is below 25.",
        failure="Invalid simplices raise ValueError; cone feasibility can affect validity checks.",
        runtime_observation="No real geometry or optimizer execution; CYTools 1.4.12 source probe only."),
    (operation=:standard_geometry_tip, readiness_required=true, direct_optimizer=true,
        transitive_optimizer=false, backend="tip_of_stretched_cone: OSQP below 25; Mosek when active at dimension >= 25; otherwise Highs",
        mandatory_mosek=false,
        fallback="OSQP below 25; Highs at dimension 25 or above when inactive or RESTART_REQUIRED.",
        failure="CYTools can return no tip or warn on an invalid optimizer result; later geometry work can then fail.",
        runtime_observation="No real geometry or optimizer execution; CYTools 1.4.12 source probe only."),
    (operation=:hilbert_basis, readiness_required=true, direct_optimizer=false,
        transitive_optimizer=false, backend="External Normaliz executable",
        mandatory_mosek=false,
        fallback="No solver fallback; Normaliz is a separate requirement.",
        failure="CYTools raises RuntimeError if Normaliz is unavailable or its output cannot be read.",
        runtime_observation="No real Normaliz execution; CYTools 1.4.12 source probe only."),
    (operation=:hilbert_save, readiness_required=true, direct_optimizer=false,
        transitive_optimizer=false, backend="HDF5 only",
        mandatory_mosek=false,
        fallback="No solver is consulted.",
        failure="Requires an existing HDF5 file; writes cytools/geometric/hilbert_basis with deflate level 9.",
        runtime_observation="Synthetic HDF5 fixture only; no real CYTools geometry was used."),
    (operation=:stored_tip_hilbert_generation, readiness_required=true, direct_optimizer=false,
        transitive_optimizer=true, backend="Stored-simplices validation can reach Cone.is_solid; no new tip solve",
        mandatory_mosek=false,
        fallback="Any transitive cone check uses Highs when MOSEK is inactive, state is RESTART_REQUIRED, or dimension is below 25.",
        failure="Reconstruction validity checks can fail; stored geometry and Hilbert data must exist.",
        runtime_observation="No real geometry or optimizer execution; wrapper/source paths inspected only."),
)

function _run_cytools_1_4_12_source_probe()
    Base.find_package("PyCall") === nothing && return :unavailable, "PyCall is unavailable"
    active_project = something(Base.active_project(), normpath(joinpath(@__DIR__, "..")))
    child_source = raw"""
        using PyCall
        cytools = try
            pyimport("cytools")
        catch
            println("CYTOOLS_UNAVAILABLE")
            exit(0)
        end
        version = String(cytools[:version])
        if version != "1.4.12"
            println("CYTOOLS_VERSION_OTHER")
            exit(0)
        end
        inspect = pyimport("inspect")
        tri = pyimport("cytools.triangulation")
        cone_module = pyimport("cytools.cone")
        polytope = cytools[:Polytope]
        triangulation = tri[:Triangulation]
        cone = cytools[:Cone]
        source(fn) = pycall(inspect[:getsource], String, fn)

        fast_api = source(polytope[:random_triangulations_fast])
        fast_generator = source(tri[:random_triangulations_fast_generator])
        fair_api = source(polytope[:random_triangulations_fair])
        fair_generator = source(tri[:random_triangulations_fair_generator])
        random_flips = source(triangulation[:random_flips])
        is_regular = source(triangulation[:is_regular])
        constructor = source(triangulation[:__init__])
        is_valid = source(triangulation[:is_valid])
        interior = source(cone[:find_interior_point])
        is_solid = source(cone[:is_solid])
        tip = source(cone[:tip_of_stretched_cone])
        hilbert = source(cone[:hilbert_basis])

        @assert occursin("random_triangulations_fast_generator", fast_api)
        @assert occursin("backend: str = " * string(Char(34), "cgal", Char(34)), lowercase(fast_generator))
        @assert !occursin("random_flips", fast_generator)
        @assert !occursin("mosek_is_activated", fast_generator)
        @assert occursin("random_triangulations_fair_generator", fair_api)
        @assert occursin("random_flips", fair_generator)
        @assert occursin("is_regular(backend=backend)", random_flips)
        @assert occursin("C.is_solid(backend=backend)", is_regular)
        @assert occursin("find_interior_point", is_solid)
        @assert occursin("is_solid", is_valid)
        @assert occursin("self.is_valid", constructor)
        @assert occursin("check_input_simplices", constructor)
        @assert occursin("config.mosek_is_activated()", interior)
        @assert occursin("ambient_dim() >= 25", interior)
        @assert occursin("backend = ", interior) && occursin("highs", interior)
        @assert occursin("ambient_dim() < 25", tip)
        @assert occursin("ambient_dim() >= 25", tip)
        @assert occursin("osqp", tip) && occursin("highs", tip)
        @assert occursin("mosek_is_activated", tip)
        @assert occursin("Normaliz", hilbert) && occursin("normaliz", hilbert)
        @assert !occursin("mosek_is_activated", hilbert)
        println("CYTOOLS_1_4_12_SOURCE_PROBE_PASSED")
    """
    command = `$(Base.julia_cmd()) --startup-file=no --project=$active_project -e $child_source`
    output = IOBuffer()
    process = run(pipeline(ignorestatus(command), stdout=output, stderr=output))
    result = _sanitize_cytools_output(String(take!(output)))
    if !success(process)
        return :failed, result
    elseif occursin("CYTOOLS_1_4_12_SOURCE_PROBE_PASSED", result)
        return :passed, result
    elseif occursin("CYTOOLS_UNAVAILABLE", result) || occursin("CYTOOLS_VERSION_OTHER", result)
        return :unavailable, result
    end
    return :failed, result
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
    @test occursin("_load_extension_submodule!", extension_source)
    @test occursin("_run_cytools_operation", extension_source)
    @test occursin("_ensure_cytools_ready()", wrapper_source)

    expected_operations = Set((:fast_triangulation, :fair_triangulation,
        :stored_simplices_reconstruction, :standard_geometry_tip, :hilbert_basis,
        :hilbert_save, :stored_tip_hilbert_generation))
    @test Set(row.operation for row in _CYTOOLS_1_4_12_CAPABILITY_MATRIX) == expected_operations
    @test all(row.readiness_required for row in _CYTOOLS_1_4_12_CAPABILITY_MATRIX)
    @test all(!row.mandatory_mosek for row in _CYTOOLS_1_4_12_CAPABILITY_MATRIX)
    @test all(!isempty(row.backend) && !isempty(row.fallback) && !isempty(row.failure)
        for row in _CYTOOLS_1_4_12_CAPABILITY_MATRIX)
    @test all(!isempty(row.runtime_observation) for row in _CYTOOLS_1_4_12_CAPABILITY_MATRIX)
    @test Dict(row.operation => (row.direct_optimizer, row.transitive_optimizer)
        for row in _CYTOOLS_1_4_12_CAPABILITY_MATRIX) == Dict(
            :fast_triangulation => (false, false),
            :fair_triangulation => (false, true),
            :stored_simplices_reconstruction => (false, true),
            :standard_geometry_tip => (true, false),
            :hilbert_basis => (false, false),
            :hilbert_save => (false, false),
            :stored_tip_hilbert_generation => (false, true),
        )
    fast_source = _julia_function_block(wrapper_source, "function topologies_generate_fast(")
    fair_source = _julia_function_block(wrapper_source, "function topologies_generate_fair(")
    reconstruction_source = _julia_function_block(wrapper_source,
        "function cy_from_poly(geom_idx::GeometryIndex)")
    geometry_source = _julia_function_block(wrapper_source,
        "function geometries_generate(h11,cy;")
    hilbert_source = _julia_function_block(wrapper_source, "function hilbert_basis(")
    save_source = _julia_function_block(wrapper_source, "function hilbert_save(")
    stored_tip_hilbert_source = _julia_function_block(wrapper_source,
        "function geometries_generate_hilbert(")
    @test occursin("random_triangulations_fast", fast_source)
    @test occursin("random_triangulations_fair", fair_source)
    @test occursin("_run_cytools_operation(:fast_triangulation) do", fast_source)
    @test occursin("_run_cytools_operation(:fair_triangulation) do", fair_source)
    @test occursin("p.triangulate(simplices=simplices)", reconstruction_source)
    @test occursin("_run_cytools_operation(:stored_simplices_reconstruction) do", reconstruction_source)
    @test occursin("tip_of_stretched_cone", geometry_source)
    @test occursin("_run_cytools_operation(:standard_geometry_tip) do", geometry_source)
    @test occursin(".hilbert_basis()", hilbert_source)
    @test occursin("_run_cytools_operation(:hilbert_basis) do", hilbert_source)
    @test occursin("cytools/geometric/hilbert_basis", save_source)
    @test occursin("_run_cytools_operation(:hilbert_save) do", save_source)
    @test occursin("deflate=9", save_source)
    @test occursin("cy_from_poly(geom_idx).cy", stored_tip_hilbert_source)
    @test occursin("tip = geom_data.tip", stored_tip_hilbert_source)
    @test !occursin("tip_of_stretched_cone", stored_tip_hilbert_source)
    @test occursin("_run_cytools_operation(:stored_tip_hilbert_generation) do", stored_tip_hilbert_source)

    probe_status, _ = _run_cytools_1_4_12_source_probe()
    if probe_status === :unavailable
        @test_skip probe_status === :passed
    else
        @test probe_status === :passed
    end

    if Base.find_package("PyCall") === nothing
        @test_skip false
    else
        passed, output = _run_cytools_initialization_smoke()
        @test passed
        @test occursin("CYTools initialization smoke passed", output)
        import_failure_passed, import_failure_output = _run_cytools_import_failure_smoke()
        @test import_failure_passed
        @test occursin("CYTools import failure diagnostic smoke passed", import_failure_output)
        submodule_failure_passed, submodule_failure_output = _run_cytools_submodule_load_failure_smoke()
        @test submodule_failure_passed
        @test occursin("CYTools submodule-load diagnostic smoke passed", submodule_failure_output)
    end
end
