# Installation

CYAxiverse.jl targets Julia 1.12. The core package is not registered in the
Julia General registry, and its core loading path does not require Python,
CYTools, Docker, or a graphical backend.

## Core Julia package

Install the development branch used by the current documentation in a Julia
project:

```julia
using Pkg
Pkg.add(url = "https://github.com/Julia-meets-String-Theory/CYAxiverse.jl.git",
        rev = "vmm")
```

Load the package and run a small check:

```julia
using CYAxiverse
CYAxiverse.greet_CYAxiverse()
```

The expected result is `"Hello CYAxiverse!"`. Importing `CYAxiverse` does not
load `PyCall`, `CairoMakie`, or `ColorSchemes`; those packages remain optional
extensions.

For a contributor checkout instead of a package installation, clone the
development branch and instantiate the project:

```sh
git clone --branch vmm https://github.com/Julia-meets-String-Theory/CYAxiverse.jl.git
cd CYAxiverse.jl
julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
julia --project=. --startup-file=no -e 'using CYAxiverse; println(CYAxiverse.greet_CYAxiverse())'
```

## Data-backed workflows

The package does not include a geometry database. Data-backed commands require
a database root containing entries such as
`h11_004/np_0000084/cy_0000001/cyax.h5`. Select that root with the
`CYAXIVERSE_DATA_DIR` environment variable or with a command's `--data-dir`
option:

```sh
export CYAXIVERSE_DATA_DIR=/path/to/data
```

The [User guide](@ref "Data directory selection") documents the complete
resolution order and the checkout-relative default. Geometry generation and
the data format are described in the [Pipelines](@ref) guide.

## Optional CYTools/PyCall integration

CYTools is needed only for workflows that generate or inspect geometry through
the Python bridge. Follow the current
[CYTools installation guide](https://github.com/LiamMcAllisterGroup/cytools/blob/main/INSTALL.md)
for platform support and dependency choices. Its current guide states that
CYTools runs on Linux and Apple Silicon (M-series) macOS; Intel-based Macs are
not supported. The upstream alternatives do not make MOSEK an unconditional
CYTools or CYAxiverse prerequisite: choose the installation and solver
features needed by your workflow.

Choose the upstream `pip` route when Normaliz is not needed:

```sh
python -m pip install cytools
```

The upstream guide notes that this route does not include Normaliz. For
Normaliz-backed functionality such as Hilbert bases, use its conda route:

```sh
git clone https://github.com/LiamMcAllisterGroup/cytools.git
cd cytools
conda env create -f environment.yml
conda activate cytools
```

Configure PyCall with the interpreter from the active CYTools environment.
Determine the path from Python itself, rather than copying a machine-specific
path into a project configuration:

```sh
python -c 'import sys; print(sys.executable)'
```

In a fresh Julia session, bind PyCall to the printed interpreter and rebuild
it:

```julia
using Pkg
Pkg.add("PyCall")
ENV["PYTHON"] = "/path/to/cytools/bin/python"
Pkg.build("PyCall")
```

Restart Julia after the rebuild. Then load and explicitly enable the optional
bridge before calling a CYTools-backed function:

```julia
using CYAxiverse
using PyCall

const CYTools = Base.get_extension(CYAxiverse, :CYAxiversePyCallExt)
CYTools.enable_cytools!()
CYTools.cytools_wrapper.cytools_version()
```

The current optional initialization path may perform a CYTools configuration
and optimizer/licence check. Its eager/check behavior is tracked separately in
[Issue #176](https://github.com/Julia-meets-String-Theory/CYAxiverse.jl/issues/176)
and is not part of the core loading contract. MOSEK activation is not required
to load `CYAxiverse`.

## Optional plotting

Plotting is an optional extension. Add its packages to the Julia project only
when you need rendered figures:

```julia
using Pkg
Pkg.add(["CairoMakie", "ColorSchemes"])

using CYAxiverse
using CairoMakie, ColorSchemes

const plotting = CYAxiverse.plotting
style = plotting.paper_style(resolution = (900, 650))
```

See the [User guide](userguide.md#optional-plotting) for plotting helpers and
publication-style examples.
