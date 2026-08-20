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

Keep `--project=notebooks`. The Pluto launcher imports `Revise`, which is a
notebook-environment dependency and is not a CYAxiverse runtime dependency.

## Production-run statistics

Open `stage_production_statistics.jl` to plot Stage-1 and Stage-2 run
statistics. The notebook reads `run_manifest.json`, the Stage-1 and Stage-2
terminal-status JSONL files, and `eft_models.parquet` when PyArrow is
available.

Set the data roots before launching Pluto, or enter them in the notebook. If
the variables are unset, the notebook starts with empty paths and reports that
no run artifacts were found:

```sh
export CYAXIVERSE_STAGE1_ROOT=/path/to/stage1_raw_frsts
export CYAXIVERSE_STAGE2_ROOT=/path/to/stage2_eft
julia --project=notebooks scripts/testing/pluto.jl
```

Set `CYAXIVERSE_PYTHON` to the Python executable from the `cytools` environment
when the notebook cannot find PyArrow automatically.

The notebook files activate `notebooks/` relative to their own location, so
they do not depend on machine-specific checkout paths. The CYTools notebook
also requires a Python environment containing CYTools; that integration remains
explicitly opt-in through `PyCall`.
