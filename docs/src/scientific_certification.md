# Scientific certification

The `CYAxiverse.scientific_certification` namespace provides bounded local
stationary-point proofs for an exact, finite axion potential and certified
generalized-Hessian spectral enclosures. Candidate generation remains the
caller's responsibility. A successful result proves the submitted local claim
for the exact supplied arrays; it does not prove that the candidate is globally
complete or physically normalized.

## Example

```julia
using CYAxiverse
const SC = CYAxiverse.scientific_certification

model = SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1))
result = SC.certify(model, [0]; options=SC.ProofOptions(precisions=(256,)))
@assert result.status == :CERTIFIED
@assert result.root.hessian_classification == :MINIMUM
```

`Q` is an `n × M` integer charge matrix (axions by terms), `L` has shape
`2 × M` and stores signed base-10 logarithmic coefficients in its first row
and phases in radians in its second row, and `K` is the exact symmetric
`n × n` metric matrix. The supplied `theta0` is exactly normalized to a
centered periodic lift, with its integer translation retained in the result.
Integer, rational, and finite binary floating-point values are captured as
exact rationals before proof arithmetic begins.

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
