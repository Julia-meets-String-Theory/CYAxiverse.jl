# Pluto notebooks

The notebooks use a separate, opt-in environment so Pluto and PlutoUI are not
runtime dependencies of `CYAxiverse`.

From the repository root, initialize that environment once:

```sh
julia --project=notebooks -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then launch Pluto with the notebook environment active:

```sh
julia --project=notebooks scripts/testing/pluto.jl
```

The notebook files activate `notebooks/` relative to their own location, so
they do not depend on machine-specific checkout paths. The CYTools notebook
also requires a Python environment containing CYTools; that integration remains
explicitly opt-in through `PyCall`.
