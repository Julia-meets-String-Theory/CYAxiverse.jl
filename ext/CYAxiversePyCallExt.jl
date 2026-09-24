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

function _record_mosek_state!(; license_path=nothing, restart_on_failure=false)
    activated = try
        if license_path === nothing
            pycall(py"_cyaxiverse_check_mosek", Bool)
        else
            pycall(py"_cyaxiverse_set_mosek_path", Bool, license_path)
        end
    catch
        _mosek_state[] = restart_on_failure ? :RESTART_REQUIRED : :ENABLED_INACTIVE_LICENSE_FAILED
        return _mosek_state[]
    end

    _mosek_state[] = activated ? :ENABLED_ACTIVE : :ENABLED_INACTIVE_LICENSE_FAILED
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
    CYAxiverse.python_interpreter.check_configured_python(PyCall.python)

    license_path = mosek_license_path === nothing ? nothing : _license_file_path(mosek_license_path)

    if _cytools_ready[]
        license_path === nothing && return nothing
        license_path == _mosek_license_path[] && return nothing
        _mosek_license_path[] = license_path
        _record_mosek_state!(; license_path, restart_on_failure=true)
        return nothing
    end

    try
        cytools_wrapper._initialize_python_api!()
        _define_mosek_helpers!()
    catch error
        throw(ErrorException("""
CYTools enable failed during Python import and wrapper setup. Confirm that
PyCall's configured interpreter has CYTools, NumPy, and SciPy available.
PyCall was not rebuilt automatically. Underlying exception type: $(typeof(error))
"""))
    end

    _cytools_ready[] = true
    if license_path === nothing
        _mosek_state[] = _record_mosek_state!()
    else
        _mosek_license_path[] = license_path
        _mosek_state[] = _record_mosek_state!(; license_path)
    end

    println("CYTools integration enabled; MOSEK state: ", _mosek_state[])
    return nothing
end

"""
    refresh_mosek_state!()

Ask CYTools to recheck its configured MOSEK license. Returns the observed state,
or `:RESTART_REQUIRED` if CYTools cannot establish a fresh state in-process.
"""
function refresh_mosek_state!()
    ensure_cytools!()
    return _record_mosek_state!(; restart_on_failure=true)
end

# The wrapper is defined after the readiness functions so its entry points can
# delegate all readiness checks to this single state owner.
include(joinpath(@__DIR__, "..", "jlm_python", "jlm_python.jl"))
include(joinpath(@__DIR__, "..", "src", "jlm_minimizer.jl"))
include(joinpath(@__DIR__, "..", "add_functions", "cytools_wrapper.jl"))

end # module CYAxiversePyCallExt
