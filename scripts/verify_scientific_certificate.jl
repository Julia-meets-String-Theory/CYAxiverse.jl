#!/usr/bin/env julia

using CYAxiverse

const SC = CYAxiverse.scientific_certification
const OBSERVATION_DIR = get(ENV, "CYAX_SCIENTIFIC_REPLAY_OBSERVATION_DIR", "")

if isempty(OBSERVATION_DIR)
    exit(SC._verify_cli(ARGS))
end

include(joinpath(@__DIR__, "..", "test", "fixtures", "scientific_certification", "run_fixture.jl"))
const FIXTURES = ScientificCertificationFixtures
isdir(OBSERVATION_DIR) || error("replay observation directory is unavailable")
before = FIXTURES._actual_environment_observation()
precision_index = findfirst(==("--precision"), ARGS)
precision_index !== nothing && precision_index < length(ARGS) ||
    error("replay precision is unavailable for bounded warm-up")
precision = parse(Int, ARGS[precision_index + 1])
FIXTURES._warmup(precision)
FIXTURES._announce_ready()
exit_code = SC._verify_cli(ARGS)
after = FIXTURES._actual_environment_observation()
environment_path = joinpath(OBSERVATION_DIR, "replay-environment.toml")
FIXTURES._set_report(environment_path, after)
execution_path = joinpath(OBSERVATION_DIR, "replay-execution.toml")
FIXTURES._set_report(execution_path, Dict("schema"=>"cyax-scientific-replay-observation-v1",
    "verifier_path"=>realpath(@__FILE__), "verifier_sha256"=>FIXTURES._hash_file(@__FILE__),
    "verifier_argv"=>String.(ARGS),
    "exit_code"=>exit_code, "before"=>before, "after"=>after,
    "environment_observation_ref"=>basename(environment_path),
    "environment_observation_sha256"=>FIXTURES._hash_file(environment_path)))
exit(exit_code)
