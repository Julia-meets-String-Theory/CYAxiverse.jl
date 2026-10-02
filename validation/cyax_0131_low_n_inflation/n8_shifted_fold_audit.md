# CYAX-0131 N8 shifted-fold audit (provisional)

## Scope and fingerprint

This is a provisional, sanitized record of the shifted N8 continuation and the currently failed two-sided Hessian-sign gate. It does not claim acceptance. The calculation uses the approved 10-row author trajectory truncation only (`trajectory=true`), with a 10-element phase vector whose only nonzero entry is `phases[2] = 0.04` radians. No P96 rows, full Eq. 19 reconstruction, or N5 inference are used here.

- Candidate source commit: `a5eacd8cd4a5905161cab239a461ba252c64e8e0`
- Candidate source tree: `d419416514a44d943ddd20290885ac8ba4090999`
- `src/paper_benchmarks/poly102_inflation.jl` Git blob: `9ad74dc6adeeff6fc92c1e52f7b113df29e30b1d`
- Candidate worktree status before writing this evidence: clean
- Julia: `1.12.6`
- Active pre-existing environment: `validation/p0_numerical_equivalence/environment/Project.toml`
- Environment Project SHA-256: `ef6dabfa1ded67f26de09b9931c9b56969ef39a762af35da48efa78a4c7a67d9`
- Environment Manifest SHA-256: `a421591181011d15d08918f0f5e49a9f7537143a43de8bdec19506c702bccdcf`
- Loaded package path resolves to the candidate's `src/CYAxiverse.jl`; no candidate dependency files changed.

## Continuation command and result

The exact candidate source was loaded with `--project=validation/p0_numerical_equivalence/environment`. Starting from `N8_BEST_X` and `N8_KC`, a warm-started `n8_degenerate_point` solve was run for 400 increments of the same phase coordinate (`j * 1e-4`, `j=1:400`), with `tolerance=1e-11`; every intermediate solve reported convergence. The source call uses the author trajectory truncation internally.

```julia
using CYAxiverse
m = CYAxiverse.paper_benchmarks.author_inflation
let phases=zeros(10), theta=copy(m.N8_BEST_X), k=m.N8_KC, c=nothing
    for j in 1:400
        phases[2] = j * 1e-4
        c = m.n8_degenerate_point(theta; k0=k, tolerance=1e-11, phases=phases)
        c.converged || error("continuation failure at $j")
        theta = c.theta
        k = c.k
    end
    println((k=c.k, theta=c.theta, null=c.null_vector,
        eigs=c.eigenvalues, grad=c.gradient_residual,
        nullres=c.null_residual, iterations=c.iterations))
end
```

Result at phase `0.04`:

- `k = 0.5080234603057957`
- `theta = [6.243185307179586, 1.503238555579152, 4.779946751600315, 0.047557771218138846, 6.195627535961448, 4.847504522818514, 4.712389558274133, 6.243185307179705]`
- gradient residual `2.9136578132832325e-16`
- null residual `1.3705424574397003e-12`
- canonical Hessian eigenvalues `[-1.4919236281448062e-12, 0.007461229753338224, 1.2907076355836649, 2.3100910625698643, 2.857734750182269, 191.59970606757324, 272.50424788127606, 1513.9930888900021]`
- Float64 local refinement is provisional. A separate 128-bit BigFloat local refinement previously produced `k=0.5080234603138255175832528546556562960302`, gradient `1.93e-37`, null residual `6.08e-34`, and norm residual `9.87e-37`; this prior result is not treated as sign-change evidence.

## Fixed-k local root probes and gate status

At each `k = c.k ± 1e-5`, two raw-coordinate stationarity solves were seeded at `c.theta ± sqrt(1e-5) * G(c.k)^(-1/2) * c.null_vector`. They solve the same ten-row phased gradient with NLsolve trust-region (`ftol=xtol=1e-12`, `iterations=1000`). Hessian spectra are from `G(k)^(-1/2) * H_theta * G(k)^(-1/2)`. The wrapped torus distance is used only to compare the two returned points.

At `dk=-1e-5`, both solves converged (9 iterations), but both smallest Hessian eigenvalues were negative:

| seed sign | gradient infinity norm | minimum eigenvalue | iterations |
|---:|---:|---:|---:|
| -1 | `4.4520996318828804e-13` | `-1.8436149642736682e-4` | 9 |
| +1 | `4.568945997365445e-13` | `-1.8436210042420387e-4` | 9 |

The two returned torus points were separated by `0.0022015262959440562`.

At `dk=+1e-5`, neither solve converged within 1000 iterations. Their gradient infinity norms were `5.4954238821731775e-9` and `5.404834144259393e-9`; their minimum eigenvalues were `-1.1585956028292872e-7` and `-4.3235685891727117e-7`. Their returned points were separated by `6.521305660961105e-8`. These residual-limited points are not accepted as critical roots.

Therefore these probes establish neither a two-sided critical-point identity nor the required minimum-Hessian eigenvalue sign change. No tolerance, metric, normalization, phase, or row convention has been changed to force a pass. The exact shifted fold remains provisional, and this mandatory gate is currently unsatisfied.

## Central-branch continuation addendum

The displaced `±sqrt(|dk|)` seeds above reached different off-center stationary points and were not treated as an exhaustive branch search. Following the refined catastrophe point directly gives a distinct, source-consistent nearby stationary identity and resolves the sign gate.

Starting from the refined point `c.theta`, a fixed-k gradient solve at `k0 = c.k + 1e-4` converged. From that root, the same 10-row phased gradient was continued downward in `k` with increments `1e-5` through `c.k - 1e-5`. Every step from `c.k + 9e-5` through `c.k - 1e-5` converged; the endpoint solves each took one NLsolve trust-region iteration (`ftol=xtol=1e-13`, `iterations=2000`). At both endpoints, the gradient infinity norm is below `1e-13`; the seven noncritical eigenvalues remain positive and the smallest eigenvalue changes sign.

| location | `k` | gradient infinity norm | minimum canonical-Hessian eigenvalue |
|---|---:|---:|---:|
| lower endpoint | `0.5080134603057957` | `8.068673225218592e-14` | `+9.202600373622222e-5` |
| refined Float64 point | `0.5080234603057957` | `8.939112742124139e-14` | `-1.3707469558080399e-9` |
| upper endpoint | `0.5080334603057957` | `9.025195684114057e-14` | `-9.212145745994419e-5` |

The endpoint critical coordinates are:

- lower: `[6.243185307179586, 1.5032376603417517, 4.779947646837718, 0.04756324905562995, 6.195622058123956, 4.847510895893407, 4.712393369927696, 6.243185307179703]`
- upper: `[6.243185307179586, 1.5032394505581868, 4.779945856621284, 0.04755229371891238, 6.1956330134606725, 4.847498150340256, 4.712384397869959, 6.243185307179703]`

This establishes a two-sided stationary identity with the required minimum-Hessian sign change on the exact ten-row model. The separate displaced-seed probes remain recorded above as unsuccessful identities and are not used to claim this result. The independent 128-bit fold refinement lies inside the reported sign bracket.
