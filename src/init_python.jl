"""Validate the configured interpreter without rebuilding PyCall."""

include(joinpath(@__DIR__, "python_interpreter.jl"))
using PyCall

desired = python_interpreter.desired_python()
desired === nothing && error("""
Set CYAXIVERSE_PYTHON to the desired interpreter. If PyCall was built for a
different interpreter, rebuild it explicitly and restart Julia.
""")
python_interpreter.check_python_interpreter(desired, PyCall.python)
println("PyCall is using the requested Python interpreter: ", PyCall.python)
