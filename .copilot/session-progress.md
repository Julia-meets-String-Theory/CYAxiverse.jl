CYAxiverse Julia/Pluto repro progress

Date: 2026-07-30

Completed:
- Verified the package loads in Julia 1.12.6 with the CYTools Python environment configured via PYTHON.
- Confirmed the CYTools wrapper can fetch polytopes and create a real CYTools object using the current API.
- Verified the wrapper-based geometry generation path can write geometry/potential data into the package’s HDF5 layout.
- Loaded generated geometry data back through the package read path and confirmed that potential data can be retrieved successfully.
- Extended the Pluto notebook repro to use the real CYTools wrapper flow instead of a synthetic toy example.
- Updated the notebook to use a more representative h11 = 4 example in place of the earlier h11 = 2 smoke-test case.
- Verified the notebook workflow structure: fetch polytope -> triangulate -> generate geometry -> read geometry -> read potential.

Current status:
- The end-to-end Pluto notebook path is now set up around the real CYTools wrapper and package read/minimizer modules.
- The next step is to run the notebook cells in Pluto and confirm the minimizer cell executes successfully with the generated data.
