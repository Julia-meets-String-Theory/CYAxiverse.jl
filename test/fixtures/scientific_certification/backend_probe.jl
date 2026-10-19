using CYAxiverse
using IntervalArithmetic
using SHA
using TOML
using Test

const SC = CYAxiverse.scientific_certification
const IA = IntervalArithmetic
const D = SC._RD
const U = SC._RU

@assert Base.pkgversion(IA) == v"0.22.36"
@assert SC._backend_supported()

function require_encloses(x, lo::Rational, hi::Rational)
    @assert IA.isguaranteed(x) && IA.decoration(x) == IA.com
    @assert Rational{BigInt}(SC._lo(x)) <= lo
    @assert hi <= Rational{BigInt}(SC._hi(x))
end

# Qualify the fixed scalar adapter against exact rational endpoint references.
SC._with_precision(256) do
    a = SC._iv(-2//3, 1//3, 256)
    b = SC._iv(3//5, 7//5, 256)
    require_encloses(SC._iadd(a, b, 256), -1//15, 26//15)
    require_encloses(SC._isub(a, b, 256), -31//15, -4//15)
    require_encloses(SC._imul(a, b, 256), -14//15, 7//15)
    require_encloses(SC._idiv(a, b, 256), -10//9, 5//9)
    require_encloses(SC._isquare(a, 256), 0//1, 4//9)
    require_encloses(SC._iabs(a, 256), 0//1, 2//3)

    pi = SC._pi_interval(256)
    pi_digits = parse(BigInt, "314159265358979323846264338327950288419716939937510")
    pi_denominator = BigInt(10)^50
    pi_lo = Rational{BigInt}(SC._lo(pi)); pi_hi = Rational{BigInt}(SC._hi(pi))
    @assert pi_digits//pi_denominator <= pi_lo <= pi_hi <= (pi_digits+1)//pi_denominator
    x = SC._iv(10, 256)
    log10 = SC._ilog(x, 256)
    recovered = SC._iexp(log10, 256)
    require_encloses(recovered, 10//1, 10//1)

    # The input contains pi/2, so the upper sine bound must include +1.
    sine_box = SC._iv(3//2, 8//5, 256)
    sine = SC._isin(sine_box, 256)
    @assert SC._hi(sine) >= 1
    cosine = SC._icos(SC._iv(0, 256), 256)
    @assert SC._lo(cosine) <= 1 <= SC._hi(cosine)
    wide = SC._iv(0//1, 7//1, 256)
    require_encloses(SC._isin(wide, 256), -1//1, 1//1)
    require_encloses(SC._icos(wide, 256), -1//1, 1//1)

    for exponent in (1000, -1000)
        value = SC._iexp(SC._iv(exponent, 256), 256)
        @assert isfinite(SC._lo(value)) && isfinite(SC._hi(value))
        @assert SC._lo(value) > 0
        @assert IA.isguaranteed(value) && IA.decoration(value) == IA.com
    end
    @test_throws SC.ProofArithmeticError SC._iv(BigFloat(Inf), BigFloat(Inf), 256)
    @test_throws DomainError SC._ilog(SC._iv(0, 256), 256)
    @test_throws DivideError SC._idiv(SC._iv(1, 256), SC._iv(-1//1, 1//1, 256), 256)
end

# Explicit proof rounding must survive hostile ambient IntervalArithmetic
# defaults. This subprocess intentionally keeps the hostile settings active.
IA.configure(rounding=:none, numtype=Float64, power=:fast, matmul=:fast)
@assert IA.interval(1) isa IA.Interval{Float64}
ambient_sum() = setprecision(BigFloat, 128) do
    IA._add_round(BigFloat(1), BigFloat(2)^(-200), RoundUp)
end
@assert ambient_sum() == 1
fixed_sum = setprecision(BigFloat, 128) do
    SC._add(BigFloat(1), BigFloat(2)^(-200), RoundUp)
end
@assert fixed_sum > 1

prior_precision = precision(BigFloat)
prior_rounding = rounding(BigFloat)
setprecision(BigFloat, 384) do
    setrounding(BigFloat, RoundUp) do
        SC._with_precision(256) do
            @assert SC._hi(SC._iv(1//3, 256)) >= 1//3
        end
        @assert precision(BigFloat) == 384
        @assert rounding(BigFloat) == RoundUp
        try
            SC._with_precision(320) do
                error("expected adapter-state restoration probe")
            end
            error("restoration probe did not throw")
        catch e
            e isa ErrorException && occursin("expected adapter-state", sprint(showerror, e)) || rethrow()
        end
        @assert precision(BigFloat) == 384
        @assert rounding(BigFloat) == RoundUp
    end
end
@assert precision(BigFloat) == prior_precision
@assert rounding(BigFloat) == prior_rounding

# A complete certification and replay must not repair or mutate the ambient
# defaults. The false-claim control supplies its own updated digest.
model = SC.prepare_model(reshape([1], 1, 1), reshape([1, 0], 2, 1), reshape([1], 1, 1))
options = SC.ProofOptions(precisions=(256,), spectral_goal=:individuals)
result = SC.certify(model, [0]; options)
@assert result.status == :CERTIFIED
source = (repository="https://github.com/Julia-meets-String-Theory/CYAxiverse.jl",
    commit=strip(read(Cmd(["git", "rev-parse", "HEAD"]), String)),
    tree=strip(read(Cmd(["git", "rev-parse", "HEAD^{tree}"]), String)))
envbytes = Vector{UInt8}(codeunits("backend_probe=adverse_IA_configuration\n"))
environment = (receipt_sha256=bytes2hex(sha256(envbytes)), artifact_name="backend-probe-environment.toml")
mktempdir() do directory
    evidence = SC.write_certificate(joinpath(directory, "positive"), model, result;
        source_ref=source, environment_ref=environment)
    report = SC.verify_certificate(evidence.certificate_path, evidence.input_path;
        expected_input_sha256=evidence.input_sha256,
        expected_certificate_sha256=evidence.certificate_sha256,
        require=:root_and_spectrum, precision_bits=256)
    @assert report.status == :VERIFIED
    mutant = TOML.parsefile(evidence.certificate_path)
    mutant["root"]["q_upper"] = "0/1"
    badpath = joinpath(directory, "false-q.toml")
    write(badpath, SC._toml_bytes(mutant))
    rejected = SC.verify_certificate(badpath, evidence.input_path;
        expected_input_sha256=evidence.input_sha256,
        expected_certificate_sha256=bytes2hex(sha256(read(badpath))),
        require=:root_and_spectrum, precision_bits=256)
    @assert rejected.status == :UNPROVEN
end
@assert IA.interval(1) isa IA.Interval{Float64}
@assert ambient_sum() == 1
@assert fixed_sum > 1

println("backend_probe=PASS interval_arithmetic=0.22.36 precision=256 ambient_rounding=none")
