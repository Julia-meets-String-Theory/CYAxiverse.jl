# Scientific certification

The `CYAxiverse.scientific_certification` namespace provides bounded local
stationary-point proofs for an exact, finite axion potential and certified
generalized-Hessian spectral enclosures. Candidate generation remains the
caller's responsibility. A successful result proves the submitted local claim
for the exact supplied arrays; it does not prove that the candidate is globally
complete or physically normalized.

## Example

```julia
using CYAxiverse, SHA, TOML
const SC = CYAxiverse.scientific_certification

model = SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1))
result = SC.certify(model, [0];
    options=SC.ProofOptions(precisions=(256,), spectral_goal=:individuals),
    candidate_provenance=nothing) # No external candidate generator: provenance is unknown.
@assert result.status == :CERTIFIED
@assert result.root.hessian_classification == :MINIMUM

project_path = Base.active_project()
project_path === nothing && error("run this example inside the package environment")
manifest_path = joinpath(dirname(project_path), "Manifest.toml")
source_ref = (repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",
    commit=readchomp(`git rev-parse HEAD`), tree=readchomp(`git rev-parse HEAD^{tree}`))

mktempdir() do workdir
    environment_path = joinpath(workdir, "producer-environment.toml")
    environment = Dict("schema"=>"cyax-scientific-example-environment-v1",
        "julia_version"=>string(VERSION),
        "interval_arithmetic_version"=>string(Base.pkgversion(SC.IA)),
        "project_sha256"=>bytes2hex(sha256(read(project_path))),
        "manifest_sha256"=>bytes2hex(sha256(read(manifest_path))))
    open(environment_path, "w") do io
        TOML.print(io, environment; sorted=true)
    end
    environment_ref = (receipt_sha256=bytes2hex(sha256(read(environment_path))),
        artifact_name=basename(environment_path))
    written = SC.write_certificate(joinpath(workdir, "certificate"), model, result;
        source_ref, environment_ref)
    replay = SC.verify_certificate(written.certificate_path, written.input_path;
        expected_input_sha256=written.input_sha256,
        expected_certificate_sha256=written.certificate_sha256,
        require=:root_and_spectrum, precision_bits=256)
    @assert replay.status == :VERIFIED
end
```

`Q` is an `n × M` integer charge matrix (axions by terms). `L` has shape
`2 × M`: its first row contains signed coefficients, and its second row
contains base-10 logarithms. Radian phases are a separate `phases` keyword to
`prepare_model`; omitted phases are exactly zero. `K` is the exact symmetric
`n × n` metric matrix. The supplied `theta0` is exactly normalized to a
centered periodic lift, with its integer translation retained in the result.
Integer, rational, and finite binary floating-point values are captured as
exact rationals before proof arithmetic begins.

`prepare_model` copies the supplied arrays. The model's stored arrays remain
mutable, so every public proof operation and certificate writer rechecks their
dimensions, integer charges, finite entries, metric symmetry and amplitude
scale. An inconsistent model raises an argument or dimension error before
proof work. After changing logarithmic amplitudes, call `prepare_model` again
to recompute the scale. Do not mutate a model during a proof operation.

When an existing numerical correction or geometry reader supplies a candidate,
keep that operation as proposal work and certify its returned point against the
same oriented model. This example shows the repository reader and correction
API; the caller supplies an already authorized `GeometryIndex` and a public-safe
source reference for the correction method.

```julia
using CYAxiverse
const SC = CYAxiverse.scientific_certification

function certify_geometry_candidate(geom_idx, theta_seed; candidate_source_ref)
    loaded = CYAxiverse.read.oriented_potential(geom_idx)
    K = Matrix(loaded.K)
    correction_context = CYAxiverse.inflation_points.prepare_context(loaded.Q, loaded.L, K)
    correction = CYAxiverse.inflation_points.correct_stationary_point(
        correction_context, theta_seed; residual_tolerance=1e-12)
    correction.status == :converged || error("numerical correction did not converge")

    model = SC.prepare_model(loaded.Q, loaded.L, K;
        phases=fill(0//1, size(loaded.Q, 2)))
    result = SC.certify(model, correction.theta;
        options=SC.ProofOptions(precisions=(256,)),
        candidate_provenance=(method="inflation_points.correct_stationary_point",
            source_ref=candidate_source_ref))
    result.status == :CERTIFIED || error("the corrected point did not meet the proof goal")
    result
end
```

The exact certifier captures a returned floating-point seed as its exact binary
value. The numerical correction does not establish the proof, source history,
global completeness, or physical normalization. Supply candidate provenance only
when the method and source reference are known; otherwise leave it `unknown`.

Every root, spectrum, and combined result carries an `identity` value. Its
`input_sha256` is the digest of the exact canonical `inputs.toml` bytes that
`write_certificate` will emit. `input_identity_status` is `:AVAILABLE` when
those bytes fit the bounded representation and `:UNAVAILABLE` with an explicit
`input_identity_reason` when they do not. The `interpretation` retains the
exact rational `s` and records the basis, orientation, period, phase units,
coefficient and numeric conventions, scaled-Hessian pencil, and
`physical_normalization="not_established"`. These fields describe the supplied
finite model and make no physical-mass claim.

The API uses IntervalArithmetic 0.22.36 for its qualified certification path.
Other supported package configurations remain importable, but the certifier
returns an explicit unsupported-backend outcome. Proof intervals use explicit
correct-rounding primitives, strict Krawczyk inclusion and contraction, and
interval LDL pivots. Spectral bounds certify coverage of the scaled pencil
`H(theta*) v = lambda K v`; they do not convert values to physical masses.

`write_certificate` writes a fresh directory with exact TOML inputs and a
closed certificate. `verify_certificate` and
`scripts/verify_scientific_certificate.jl` replay the exact submitted claims
without network access. A verified certificate proves only its recorded local
claims. It does not establish global completeness, a vacuum population,
physical normalization, or correctness of an external candidate generator.

## API reference

```@docs
CYAxiverse.scientific_certification
CYAxiverse.scientific_certification.write_certificate
CYAxiverse.scientific_certification.verify_certificate
```
