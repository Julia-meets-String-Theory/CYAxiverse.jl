"""Exact-input, interval-proof API for bounded stationary points and spectra."""
module scientific_certification

using LinearAlgebra
using SHA
using TOML
import IntervalArithmetic

const IA = IntervalArithmetic
const _BACKEND_VERSION = v"0.22.36"
const _PROOF_LOCK = ReentrantLock()

include("scientific_certification/intervals.jl")
include("scientific_certification/stationary.jl")
include("scientific_certification/spectra.jl")
include("scientific_certification/certificates.jl")

export CertificationModel, ProofOptions, RootResult, SpectrumResult,
       CertificationResult, VerificationReport, prepare_model,
       certify_stationary_point, certify_generalized_spectrum, certify,
       write_certificate, verify_certificate

end # module scientific_certification
