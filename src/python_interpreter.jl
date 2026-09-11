# Interpreter-selection checks for the optional PyCall integration.
module python_interpreter

"""Return the optional interpreter requested through `CYAXIVERSE_PYTHON`."""
function desired_python(environment=ENV)
    configured = strip(get(environment, "CYAXIVERSE_PYTHON", ""))
    isempty(configured) ? nothing : configured
end

function canonical_python(path::AbstractString)
    configured = strip(path)
    isempty(configured) && throw(ArgumentError("Python interpreter path must not be empty"))
    expanded = expanduser(configured)
    located = Sys.which(expanded)
    candidate = located === nothing ? expanded : located
    normalized = normpath(abspath(candidate))
    ispath(normalized) ? realpath(normalized) : normalized
end

"""
    check_python_interpreter(desired, effective)

Verify that an optional desired interpreter is the interpreter used by PyCall.
This function never rebuilds PyCall.
"""
function check_python_interpreter(
    desired::Union{Nothing,AbstractString}, effective::AbstractString)
    desired === nothing && return effective
    canonical_python(desired) == canonical_python(effective) && return effective
    throw(ArgumentError("""
CYAXIVERSE_PYTHON requests "$desired", but PyCall is using "$effective".
Setting CYAXIVERSE_PYTHON does not rebind an already-built PyCall. Rebuild
PyCall explicitly against the requested interpreter:

    ENV["PYTHON"] = ENV["CYAXIVERSE_PYTHON"]
    import Pkg
    Pkg.build("PyCall")

Then restart Julia and enable the CYTools integration again. CYAxiverse does
not rebuild PyCall automatically.
"""))
end

"""Check `CYAXIVERSE_PYTHON` against the interpreter used by PyCall."""
function check_configured_python(effective::AbstractString; environment=ENV)
    check_python_interpreter(desired_python(environment), effective)
end

end
