# Integration design: axion relic abundance in CYAxiverse.jl (`vmm`)

Branch `feature/axion-relic-abundance`, based on `vmm` at `f12fbb8` (the upstream head on 2026-10-31).

## 1. Base

| candidate | contains the revision `a5eacd8` that produced the reference dataset | vmm scripts used by the producers | chosen |
|---|---|---|---|
| `vmm` head `f12fbb8` | yes (ancestor) | present, unchanged in method since `a5eacd8` | **yes** |
| `main` `aa934d2` | no (diverged; the spectrum, G2 and vacua scripts are missing) | missing | no |
| `a5eacd8` itself | yes | present | no: a branch from an old commit would need an immediate merge of 69 upstream commits |

**Why `vmm` head:**
- `vmm` is the principal development line (AGENTS.md §5).
- At `f12fbb8` the producers reproduce the reference geometries, potentials, spectra and vacuum repairs dataset for dataset. This was measured with `references/shoroq/tests/regression_subset.sh`.
- The only exception is `spectrum/physical/metadata/configuration_digest`. That is a sha256 of the spectrum configuration, which contains the CYAxiverse commit, so it changes with the commit by construction.

**Changes between `a5eacd8` and `f12fbb8` that the producers touch:**
- `scripts/batch_vacua_pipeline.jl`: an optional worker-flag argument only;
- `src/filestructure.jl`: legacy hard-coded aliases became environment variables, and an explicit `CYAXIVERSE_DATA_DIR` behaves as before;
- `Project.toml`: adds the TOML standard library.

## 2. Placement: a self-contained research directory

Everything is in `research/axion_relic_abundance/`. The repository already uses `research/` for project-level research code that is outside the package API.

| layer | change |
|---|---|
| `src/`, `ext/`, `test/`, `scripts/`, `Project.toml`, docs, CI | **none**. The public Julia API, persisted schemas and package version are unchanged, so the version impact is none |
| `research/axion_relic_abundance/` | new |

**Not part of `src/`, deliberately.**
- The relic engine is an analysis layer on top of persisted spectrum files.
- It needs Python (pandas, mpmath, scipy). AGENTS.md §3 requires `using CYAxiverse` to stay free of Python.
- It uses a separately pinned Julia analysis environment, for byte-identical reproduction of the reference results.
- Moving the Julia relic driver into the package can be done later, as a reviewed S2 change, once the reference outputs are re-validated under Julia 1.12 (§6).

**CI.** `.github/workflows/CI.yml` runs on pull requests that touch `src/`, `test/`, `scripts/` and similar paths, but not `research/`. A PR containing only this directory therefore triggers no package CI. The checks of this contribution are run as documented in `INTEGRATION_TEST_REPORT` (outside the repository) and in §5.

## 3. Components and where they live

| class | components | in the branch |
|---|---|---|
| production code | `engine/input/`, `engine/relic/` (composer, A1 tables, relic driver with frozen physics block, validators, independent implementations C and B), `engine/theta/`, `engine/run_*.sh`, `engine/common.sh` | yes |
| supporting workflows | `producers/geometry/`, `producers/vacuum/`, `producers/run_*.sh` (they call the upstream vmm scripts and copy none of them) | yes |
| configuration and data | `configs/*.json`, `configs/paths.env.example`, `data/standardmodel2018.dat` (g\*(T), sha256-pinned) | yes |
| public example | `examples/minimal/` (two shipped non-Shoroq geometries, three T_reh, relic and θ in under a minute) | yes |
| tests and fixtures | `tests/test_static.py`, `tests/test_external_acceptance.py` (16 cases, self-contained), `tests/test_general.sh`, fixtures in `tests/external_acceptance/data/` (two genuine geometries, 0.3 MB) | yes |
| reference case | `references/shoroq/` (configs, expectations, sample manifest of 1.9 MB, sample reconstruction, regression tests) | yes |
| environments | `environment/julia_vmm/Manifest.toml` (a5eacd8), `environment/julia_analysis/` (Julia 1.8.4), Python requirements | yes |
| documentation | `README.md`, `METHODOLOGY.md`, `DATA_CONTRACTS.md`, `PROVENANCE.md`, `SCIENTIFIC_SCOPE_AND_CONVENTIONS.md`, this file | yes |
| frozen reference evidence | the accepted Shoroq dataset and results (about 3.6 GB of HDF5 and tables) | **no**: an external archive with its own SHA256SUMS, required only by `references/shoroq/tests/` |
| development-only material | audits, acceptance and physics-audit reports, logs, exploratory alternative-normalization and pair-sign spectra, legacy datasets, figures | **no** |

**Publication hygiene (AGENTS.md §1).**
- No absolute local path, username or host is tracked. `tests/test_static.py` scans text files for these.
- The two HDF5 fixtures had local paths in three provenance-only attributes (`construction_metadata_json`, `julia_materialization/driver_path`, `source_file`). The prefix was replaced by `<acceptance-root>/` and the files repacked with H5Ocopy.
  - Every dataset, every other attribute and every storage filter was verified identical.
  - `SHA256SUMS.txt` was updated (`PROVENANCE.md`).
- Note that the producers write absolute paths into newly generated provenance attributes. Users should keep generated data out of Git, which the package `.gitignore` does for `/work/`.

## 4. Paths and configuration

- Every location comes from `configs/paths.env` (git-ignored) or a file named by `SHOROQ_PATHS`.
- `WORK` can also be exported.
- Spectra lists and evidence paths are resolved relative to the list file.
- Within this checkout, an empty `CYAX_REPO` defaults to the enclosing CYAxiverse.jl repository. This is the only behavioural change made for integration.

**Revision pin.**
- The producers and `VERIFY` inputs check that `CYAX_REPO` is at `a5eacd8`, the revision of the reference dataset.
- Any other revision, including this branch, needs `ALLOW_OTHER_CYAX_COMMIT=1` and prints a warning.
- This is intentional: bitwise reproduction of the reference is demonstrated at `a5eacd8`, and at `f12fbb8` up to the configuration digest.
- The engine alone (existing spectra with `DOCUMENTED_EXTERNAL` evidence) needs no CYAxiverse checkout and no Julia 1.12.

## 5. Interfaces

**Inputs.** vmm HDF5 files (`cytools/…`, `spectrum/physical/…`) as written by `scripts/batch_physical_spectrum.jl`, listed in a `cyaxiverse_relic_spectra/1` JSON file. Each entry has an explicit vacuum-validation state; the contract is in `DATA_CONTRACTS.md`.

**Run file.** `cyaxiverse_relic_run/1`, strict. It contains:
- the reheating scenarios (label, T_reh in GeV, with hard limits and BBN notes);
- the approval flag for runs above 1000 modes;
- optional θ overrides.

**Outputs.** CSV and JSON under `$WORK`:
- per-mode and per-geometry tables with their abundance-validity class;
- validation and independent-verification reports;
- primary and sensitivity θ analyses, with cross-checks;
- run metadata with checksums.

**Entry points** (shell, the same for every user): `engine/run_prepare_from_spectra.sh`, `engine/run_relic.sh`, `engine/run_theta.sh`, `producers/run_geometry.sh`, `producers/run_vacuum_repair.sh`.

## 6. Open items for the maintainers

1. **Julia version of the analysis stages.**
   - They run with Julia 1.8.4 (`environment/julia_analysis`), the version of the frozen production run, and are byte-identical there.
   - The repository targets 1.12, and they have not been validated under 1.12.
   - Moving to 1.12 needs a re-resolved analysis environment and a regression at scientific (not bitwise) tolerance.
2. **Revision pin.**
   - Keep `a5eacd8` as the reproduction pin, or move it to `f12fbb8` after accepting the subset evidence, with new configuration digests.
3. **Conventions** (`SCIENTIFIC_SCOPE_AND_CONVENTIONS.md` §3):
   - the kinetic-normalization factor of two;
   - the pair-term prescription.
   - Both are upstream questions. This contribution inherits them unchanged.
4. **Frozen reference archive.**
   - It must be deposited (with its SHA256SUMS) before the Shoroq regression tests can be run by anyone but the author.
5. **Optional:**
   - an SDD spec (`specs/<id>-axion-relic-abundance/spec.md`; AGENTS.md classes this as S2 scientific work);
   - a pointer in the upstream documentation;
   - running `tests/test_static.py` in CI (needs Python with mpmath only).
