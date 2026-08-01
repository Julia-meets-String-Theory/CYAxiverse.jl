"""
    CYAxiversePyCallExt

Optional CYTools integration for CYAxiverse. Loaded automatically when both
`CYAxiverse` and `PyCall` are in scope.

## Usage

```julia
using CYAxiverse
using PyCall

const CYTools = Base.get_extension(CYAxiverse, :CYAxiversePyCallExt)
CYTools.enable_cytools!()

# geometry generation
CYTools.cytools_wrapper.topologies(h11, n)

# python-backed minimisation
CYTools.jlm_minimizer.minimize(geom_idx)
```
"""
module CYAxiversePyCallExt

using CYAxiverse
using PyCall

# ── initialisation state ──────────────────────────────────────────────────────

const _cytools_initialised = Ref(false)

"""
    ensure_cytools!()

Internal guard called at the entry point of every CYTools-backed function.
"""
function ensure_cytools!()
    _cytools_initialised[] && return
    error("""
CYTools integration is not yet enabled.  Call:

    const CYTools = Base.get_extension(CYAxiverse, :CYAxiversePyCallExt)
    CYTools.enable_cytools!()

before using CYTools-backed functions.  If that call errors, rebuild PyCall
against a Python environment that contains CYTools:

    ENV["PYTHON"] = "/path/to/cytools/bin/python"
    import Pkg; Pkg.build("PyCall")

then restart Julia and try again.
""")
end

"""
    enable_cytools!()

Explicitly initialise the CYTools integration:
1. Reports the Python executable PyCall is using.
2. Imports `cytools` and configures MOSEK.
3. Caches success so repeated calls are free.

Never calls `Pkg.build` automatically.
"""
function enable_cytools!()
    _cytools_initialised[] && return

    println("PyCall is using Python: ", PyCall.python)

    try
        py"""
        from cytools import config
        import os
        config.set_mosek_path(os.environ['HOME'])
        config.check_mosek_license()
        def _cyax_test_config():
            return config.mosek_is_activated
        """
    catch e
        error("""
CYTools integration is unavailable.  Ensure a Python environment containing
cytools is installed and configure PyCall to use it:

    ENV["PYTHON"] = "/path/to/cytools/bin/python"
    import Pkg; Pkg.build("PyCall")

then restart Julia and call enable_cytools!() again.

Underlying error: $e
""")
    end

    _cytools_initialised[] = true
    println("CYTools integration enabled.")
end

# ── Python-backed submodules ───────────────────────────────────────────────────
# Each file defines its own module (cytools_wrapper / jlm_python / jlm_minimizer).
# Include order matters: jlm_minimizer depends on jlm_python.

include(joinpath(@__DIR__, "..", "jlm_python", "jlm_python.jl"))
include(joinpath(@__DIR__, "..", "src", "jlm_minimizer.jl"))
include(joinpath(@__DIR__, "..", "add_functions", "cytools_wrapper.jl"))

end # module CYAxiversePyCallExt
