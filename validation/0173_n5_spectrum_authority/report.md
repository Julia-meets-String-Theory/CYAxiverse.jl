# CYAX-0173 N=5 B6 diagnostic

## Result

**Classification:** `FLOAT64_CANCELLATION_OR_CONDITIONING`.

For this frozen N=5 B6 fixture, Float64 matrix roundoff is sufficient to lose the light eigenvalue information. The leading Hessian condition number is about `3.48e73`; the Float64 assembled canonical matrix differs from the high precision matrix built from the same frozen inputs by about `3.98e-17` in relative infinity norm. A high precision solve of the promoted Float64 matrix still gives light masses near `13.238`, `13.843`, and `13.999`, with signs `(+,+,-)`. The Float64 solver moves those results further to the reproduced P0 values near `13.861`, `14.074`, and `14.230`, with signs `(-,+,+)`.

The high precision ladder reproduces the frozen 80 digit masses from 80 through 512 decimal digits. At 64 digits, the smallest mass shifts by about `6.58e-5`. The Float64 and high precision light subspaces agree to a projector distance of `3.6e-16`; the reported `1.5e-8` to `2.6e-8` radian angles are the `acos(Float64)` resolution floor.

The same sorted-index Float64/high precision quartic diagnostics differ by up to `18.83` log10 units for self terms, `18.08` for 31 terms, and `17.34` for 22 terms; the 31 sign flags differ in 13 of 20 entries. Individual identities are withheld under the predeclared sign and cluster policy. The aggregate light-sector quartic tensor log10 Frobenius norms agree within about `7e-15`. Other downstream observables were not computed and remain authority-restricted.

This is bounded diagnostic evidence. Neither route is promoted to an oracle. It authorizes no production correction, spectrum migration, merge, or Issue closure.

## Evidence

- [P1 replay script](replay_b6.jl), [reference-checkout replay](p1_replay_reference.toml), and [secondary base replay](p1_replay.toml)
- [Pre-result matching declaration](matching_declaration.json) and [pre-ladder source/environment freeze](p2_source_environment_freeze.json)
- [P2 precision ladder and matrices](p2_ladder.toml)
- [P3 machine-readable classification and downstream summary](p3_diagnostic_synthesis.json)
- [Ladder diagnostic script](diagnose_ladder.jl)

The matching declaration was frozen at `2026-09-25 13:43:47 UTC` (SHA-256 `ef40308b2e1241bda488b7ba7e7c3fceb17d92b69155087d8782eb6fd9ea56ee`). The source/environment freeze was recorded at `2026-09-25 13:59:08 UTC`, before any ladder output was produced or inspected. The primary reference checkout is identified by commit `7a40285bb5c313f7e8746b90644d5f45bb67be44` and tree `ef79ad720ea0ed1ad2193b2616a97319d8282a61`; its P0 benchmark Project and Manifest hashes are recorded in the freeze.

The direct sparse rank-one matrix check mirrors the package helper’s accumulation order. Exact agreement checks the assembly formula and inputs; it is not an independent physical oracle. The stronger metric check recomputes the whitening factor from high precision K and measures the generalized `H_theta*v = lambda*K*v` residual.

## P6 specification repair

The Float64 normwise residuals use each eigensolver’s eigenvalue paired with its reported mass-basis vector. Absolute overlaps against the vectors from the same eigensolver range from `0.9999999999999999` to `1.0000000000000002`, confirming the pairing up to Float64 roundoff. The paired Float64 eigenvalues, in reported mass order, are `[-8.847741104041705e-14, 2.356014083237243e-13, 4.851505296555931e-13, 1327.1705458222668, 8510.84495988584]`. The whitened residuals are `[8.16e-18, 8.88e-17, 7.04e-17, 5.55e-16, 1.10e-16]`; the generalized H/K residuals are `[2.12e-17, 1.14e-16, 1.03e-16, 1.38e-16, 1.83e-17]`.

The original matching declaration remains unchanged. Applying its numeric-eigenvalue cluster rule gives `[[1,2,3],[4],[5]]` for both Float64 and high precision 80. Sign-restricted matching remains separate and reports cardinality mismatches: Float64 has one negative and four positive modes, while high precision 80 has five positive modes. Individual identity remains withheld.

The fixed-input sensitivity check holds C, Qtilde, Ltilde, scaling, and underflow floor constant. Reversing rank-one column order, summing each entry’s terms in ascending absolute magnitude, and using a right-associated whitening product each produced a bitwise-identical Float64 matrix. All had zero light-mass deltas and retained signs `(-,+,+)`. These tested reduction and product order changes do not explain the disputed modes.

## Commands and status

The reference checkout was a temporary archive at the commit and tree above. `WRITABLE_DEPOT`, `RETAINED_DEPOT`, and `REFERENCE_CHECKOUT` below name the runtime paths used for those locations; the durable artifacts contain no machine-local paths.

```sh
julia --startup-file=no -e 'Meta.parseall(read("validation/0173_n5_spectrum_authority/diagnose_ladder.jl", String)); println("PARSE_PASS")'
# exit 0

CYAX173_EXECUTION_BASE=74fea608684eae746f25ae45b18512f89b862fb0 \
CYAX173_REPLAY_OUTPUT=validation/0173_n5_spectrum_authority/p1_replay.toml \
JULIA_DEPOT_PATH="$WRITABLE_DEPOT:$RETAINED_DEPOT" JULIA_NUM_THREADS=1 \
julia --startup-file=no --project=validation/p0_numerical_equivalence/environment_benchmarks \
  validation/0173_n5_spectrum_authority/replay_b6.jl
# exit 0, P1_REPLAY_PASS

CYAX173_EXECUTION_BASE=7a40285bb5c313f7e8746b90644d5f45bb67be44 \
CYAX173_REPLAY_OUTPUT=validation/0173_n5_spectrum_authority/p1_replay_reference.toml \
JULIA_DEPOT_PATH="$WRITABLE_DEPOT:$RETAINED_DEPOT" JULIA_NUM_THREADS=1 \
julia --startup-file=no --project="$REFERENCE_CHECKOUT/validation/p0_numerical_equivalence/environment_benchmarks" \
  validation/0173_n5_spectrum_authority/replay_b6.jl
# exit 0, P1_REPLAY_PASS

CYAX173_REFERENCE_ROOT="$REFERENCE_CHECKOUT" \
CYAX173_REFERENCE_COMMIT=7a40285bb5c313f7e8746b90644d5f45bb67be44 \
CYAX173_EXECUTION_BASE=74fea608684eae746f25ae45b18512f89b862fb0 \
JULIA_DEPOT_PATH="$WRITABLE_DEPOT:$RETAINED_DEPOT" JULIA_NUM_THREADS=1 \
julia --startup-file=no --project="$REFERENCE_CHECKOUT/validation/p0_numerical_equivalence/environment_benchmarks" \
  validation/0173_n5_spectrum_authority/diagnose_ladder.jl
# exit 0, P2_LADDER_PASS; ArbFloat precision restored to 104 bits

python3 -m json.tool validation/0173_n5_spectrum_authority/p3_diagnostic_synthesis.json >/dev/null
# exit 0
```

P6 regeneration used the same command above for `diagnose_ladder.jl`, with exit 0 and `P2_LADDER_PASS`; it read back ArbFloat precision at 215, 268, 427, 853, and 1703 bits and restored the global setting to 104 bits. The updated diagnostic script SHA-256 is `ab697396771ef29e366860ac32818e99023b8a8774e938e0bf19f5cd147a01e4`; the regenerated ladder SHA-256 is `48cc2419d453b911b5decdf77299727c24330031796492a086c5f382fefa48e5`.
