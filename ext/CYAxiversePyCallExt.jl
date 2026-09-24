"""
    CYAxiversePyCallExt

Optional CYTools integration for CYAxiverse. Loading this extension loads its
Julia submodules only. Call `enable_cytools!()` to import CYTools and make its
wrapper functions available.
"""
module CYAxiversePyCallExt

using CYAxiverse
using PyCall

const _cytools_ready = Ref(false)
const _mosek_state = Ref(:CYTOOLS_DISABLED)
const _mosek_license_path = Ref{Union{Nothing, String}}(nothing)
const _mosek_diagnostic = Ref("CYTools is disabled. Call enable_cytools!() before using CYTools-backed wrappers.")
const _operation_context_key = gensym(:cytools_operation_context)

const _operation_backend_policies = Dict{Symbol, String}(
    :fetch_polytopes => "CYTools database lookup; no optimizer or MOSEK consultation.",
    :poly => "CYTools polytope construction; no optimizer or MOSEK consultation.",
    :cone => "CYTools cone construction; no optimizer or MOSEK consultation.",
    :cytools_version => "Python version query; no optimizer or MOSEK consultation.",
    :fast_triangulation => "random_triangulations_fast uses CGAL by default; no optimizer or MOSEK consultation.",
    :fair_triangulation => "random_triangulations_fair may consult MOSEK at dimension 25 or above when active; otherwise it can use Highs.",
    :stored_simplices_reconstruction => "Stored-simplex validity may consult MOSEK at dimension 25 or above when active; otherwise it can use Highs.",
    :standard_geometry_tip => "tip_of_stretched_cone uses OSQP below dimension 25, MOSEK at dimension 25 or above when active, and Highs otherwise.",
    :standard_geometry_generation => "Geometry generation uses tip_of_stretched_cone with OSQP below dimension 25, MOSEK at dimension 25 or above when active, and Highs otherwise.",
    :hilbert_basis => "CYTools Cone.hilbert_basis uses Normaliz; it does not consult MOSEK and has no MOSEK fallback.",
    :hilbert_save => "HDF5 persistence only; no optimizer or MOSEK backend is consulted.",
    :stored_tip_hilbert_generation => "Uses the stored tip; stored-simplex reconstruction may consult MOSEK at dimension 25 or above when active, otherwise Highs.",
)

function _exception_kind(error)
    cause = error isa LoadError ? getfield(error, :error) : error
    return string(nameof(typeof(cause)))
end

function _phase_label(phase::Symbol)
    phase === :interpreter_validation && return "interpreter validation"
    phase === :python_import && return "Python import and wrapper setup"
    phase === :initial_license_check && return "initial license check"
    phase === :license_path_override && return "license path override"
    phase === :license_refresh && return "license refresh"
    return "license check"
end

function _inactive_mosek_diagnostic(phase::Symbol)
    return "MOSEK activation was false after the $(_phase_label(phase)). CYTools remains enabled without MOSEK. Correct the license configuration, then call refresh_mosek_state!(); a process-local license file can be supplied with enable_cytools!(; mosek_license_path=...)."
end

function _failed_mosek_diagnostic(phase::Symbol, error, state::Symbol)
    cause = _exception_kind(error)
    if state === :RESTART_REQUIRED
        return "MOSEK setup failed during $(_phase_label(phase)) (cause type: $cause). Activation cannot be trusted in this Julia process. Correct the license configuration, restart Julia, then call enable_cytools!() again. CYTools remains enabled, but MOSEK use requires that restart."
    end
    return "MOSEK check failed during $(_phase_label(phase)) (cause type: $cause). CYTools remains enabled without MOSEK. Correct the license configuration, then call refresh_mosek_state!() or enable_cytools!(; mosek_license_path=...)."
end

function _run_cytools_operation(f::Function, operation::Symbol)
    ensure_cytools!()
    storage = task_local_storage()
    haskey(storage, _operation_context_key) && return f()

    storage[_operation_context_key] = operation
    try
        return f()
    catch error
        error isa InterruptException && rethrow()
        policy = get(_operation_backend_policies, operation,
            "Backend requirements depend on the CYTools operation; consult its supported upstream backends.")
        diagnostic = "CYTools downstream wrapped operation `$operation` failed (cause type: $(_exception_kind(error))). Backend selection policy: $policy CYTools was explicitly enabled and imported before this call. MOSEK activation/configuration state: $(_mosek_state[]). Check license configuration only if the selected backend requires it, and use a supported fallback when available. Exception text and local paths are omitted."
        _mosek_diagnostic[] = diagnostic
        throw(ErrorException(diagnostic))
    finally
        delete!(storage, _operation_context_key)
    end
end

function _load_extension_submodule!(path::AbstractString, name::Symbol)
    try
        return Base.include(@__MODULE__, path)
    catch error
        diagnostic = "CYAxiversePyCallExt extension/submodule load failed during `$name` (cause type: $(_exception_kind(error))). Check the optional Julia/PyCall and Python import environment, restart Julia, then use enable_cytools!() after the extension loads. MOSEK activation/configuration is checked separately during enable or refresh. No downstream wrapped operation/backend selection was reached. Exception text and local paths are omitted."
        _mosek_diagnostic[] = diagnostic
        throw(ErrorException(diagnostic))
    end
end

"""Reject wrapper calls until `enable_cytools!()` has established readiness."""
function ensure_cytools!()
    _cytools_ready[] && return nothing
    throw(ArgumentError("""
CYTools integration is disabled. Call:

    const CYTools = Base.get_extension(CYAxiverse, :CYAxiversePyCallExt)
    CYTools.enable_cytools!()

before calling CYTools-backed wrapper functions. This call does not rebuild
PyCall. If the configured interpreter is wrong, configure and rebuild PyCall
manually, restart Julia, and then enable CYTools.
"""))
end

"""
    mosek_state()

Return the most recently observed MOSEK activation state. The result is one of
`:CYTOOLS_DISABLED`, `:ENABLED_ACTIVE`, `:ENABLED_INACTIVE_LICENSE_FAILED`, or
`:RESTART_REQUIRED`.
"""
mosek_state() = _mosek_state[]

"""
    mosek_diagnostic()

Return path-free guidance for the most recent enable, MOSEK, or wrapped-operation
result. Exception text and local interpreter or license paths are omitted.
"""
mosek_diagnostic() = _mosek_diagnostic[]

function _define_mosek_helpers!()
    py"""
    from cytools import config
    import contextlib
    import io

    def _cyaxiverse_check_mosek():
        # License diagnostics may contain local paths. Keep them out of the
        # Julia console and return only the activation boolean.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            config.check_mosek_license()
        return bool(config.mosek_is_activated())

    def _cyaxiverse_set_mosek_path(path):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            config.set_mosek_path(path)
        return bool(config.mosek_is_activated())

    """
    return nothing
end

function _license_file_path(path)
    path isa AbstractString || throw(ArgumentError("mosek_license_path must name an existing license file"))
    isfile(path) || throw(ArgumentError("mosek_license_path must name an existing license file"))
    return String(path)
end

function _record_mosek_state!(; license_path=nothing, restart_on_failure=false, phase=:license_check)
    activated = try
        if license_path === nothing
            pycall(py"_cyaxiverse_check_mosek", Bool)
        else
            pycall(py"_cyaxiverse_set_mosek_path", Bool, license_path)
        end
    catch error
        _mosek_state[] = restart_on_failure ? :RESTART_REQUIRED : :ENABLED_INACTIVE_LICENSE_FAILED
        _mosek_diagnostic[] = _failed_mosek_diagnostic(phase, error, _mosek_state[])
        return _mosek_state[]
    end

    _mosek_state[] = activated ? :ENABLED_ACTIVE : :ENABLED_INACTIVE_LICENSE_FAILED
    _mosek_diagnostic[] = activated ?
        "MOSEK activation is active." : _inactive_mosek_diagnostic(phase)
    return _mosek_state[]
end

"""
    enable_cytools!(; mosek_license_path=nothing)

Check the configured PyCall interpreter, import CYTools, and enable the
CYTools-backed wrapper. MOSEK activation is recorded separately and does not
control CYTools readiness. If supplied, `mosek_license_path` must name an
existing MOSEK license file; it is used only for this Julia process.

This function never calls `Pkg.build`.
"""
function enable_cytools!(; mosek_license_path=nothing)
    try
        CYAxiverse.python_interpreter.check_configured_python(PyCall.python)
    catch error
        _mosek_diagnostic[] = "CYTools enable failed during interpreter validation (cause type: $(_exception_kind(error))). CYAXIVERSE_PYTHON does not rebind an already-built PyCall interpreter. Configure it to match PyCall, run Pkg.build(\"PyCall\") only as an explicit user action if needed, restart Julia, then call enable_cytools!() again. Interpreter paths are omitted."
        throw(ArgumentError(_mosek_diagnostic[]))
    end

    license_path = mosek_license_path === nothing ? nothing : _license_file_path(mosek_license_path)

    if _cytools_ready[]
        license_path === nothing && return nothing
        license_path == _mosek_license_path[] && return nothing
        _mosek_license_path[] = license_path
        _record_mosek_state!(; license_path, restart_on_failure=true, phase=:license_path_override)
        _mosek_state[] === :RESTART_REQUIRED && @warn _mosek_diagnostic[]
        return nothing
    end

    try
        cytools_wrapper._initialize_python_api!()
        _define_mosek_helpers!()
    catch error
        _mosek_diagnostic[] = "CYTools enable failed during Python import and wrapper setup (cause type: $(_exception_kind(error))). Confirm that PyCall's configured interpreter has CYTools, NumPy, and SciPy. Correct the environment, restart Julia, then call enable_cytools!() again. Exception details and local paths are omitted."
        throw(ErrorException(_mosek_diagnostic[]))
    end

    _cytools_ready[] = true
    if license_path === nothing
        _mosek_state[] = _record_mosek_state!(; phase=:initial_license_check)
    else
        _mosek_license_path[] = license_path
        _mosek_state[] = _record_mosek_state!(; license_path, phase=:license_path_override)
    end

    println("CYTools integration enabled; ", _mosek_diagnostic[])
    return nothing
end

"""
    refresh_mosek_state!()

Ask CYTools to recheck its configured MOSEK license. Returns the observed state,
or `:RESTART_REQUIRED` if CYTools cannot establish a fresh state in-process.
"""
function refresh_mosek_state!()
    ensure_cytools!()
    state = _record_mosek_state!(; restart_on_failure=true, phase=:license_refresh)
    state === :RESTART_REQUIRED && @warn _mosek_diagnostic[]
    return state
end

# The wrapper is defined after the readiness functions so its entry points can
# delegate all readiness checks to this single state owner.
_load_extension_submodule!(joinpath(@__DIR__, "..", "jlm_python", "jlm_python.jl"), :jlm_python)
_load_extension_submodule!(joinpath(@__DIR__, "..", "src", "jlm_minimizer.jl"), :jlm_minimizer)
_load_extension_submodule!(joinpath(@__DIR__, "..", "add_functions", "cytools_wrapper.jl"), :cytools_wrapper)

end # module CYAxiversePyCallExt
