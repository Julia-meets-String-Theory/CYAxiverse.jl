# CYAxiverse environment (Julia 1.12.7)

`Manifest.toml` is the resolved environment of CYAxiverse.jl at `vmm` commit
`a5eacd8cd4a5905161cab239a461ba252c64e8e0` used for the frozen dataset
(sha256 `bc566a3366115d37791186a3535e3072d0d913fd6271f949ed5fef6a703c8d5c`).
The upstream repository ignores `/Manifest.toml`, so copy this file into the root of the
CYAxiverse checkout before `julia --project=<checkout> -e 'using Pkg; Pkg.instantiate()'`.
