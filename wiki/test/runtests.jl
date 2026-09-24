using JSON3
using Test
using CYAxiverseWikiRefresh

const W = CYAxiverseWikiRefresh
const SHA0 = "995163f0058488ea183ac645045ed8b1636bef4a"
const SHA1 = "f02621377c3c01f1c5ae85ef5ffd00a219aa8b8d"

function minimal_manifest()
    Dict{String,Any}(
        "schema_version" => 1,
        "initial_verified_commit" => SHA0,
        "repository" => Dict("owner" => "owner", "name" => "repo", "branch" => "vmm"),
        "pages" => Dict("example" => Dict(
            "title" => "Example",
            "authority" => "repository-backed",
            "mode" => "semantic",
            "sources" => ["src/file.jl"],
            "issues" => Any[],
            "pull_requests" => Any[],
        )),
    )
end

function minimal_state()
    Dict{String,Any}(
        "schema_version" => 1,
        "repository" => Dict("head" => SHA0, "branch" => "vmm"),
        "created_at" => "2026-09-23T00:00:00Z",
        "pages" => Dict("example" => Dict(
            "verified_commit" => SHA0,
            "reconciled_at" => "2026-09-23T00:00:00Z",
            "issues" => Dict{String,Any}(),
            "pull_requests" => Dict{String,Any}(),
        )),
    )
end

@testset "manifest validation and privacy" begin
    manifest = minimal_manifest()
    @test W._validate_manifest(manifest)

    private = deepcopy(manifest)
    private["pages"]["example"]["notion_page_id"] = "private-id"
    @test_throws W.StateValidationError W._validate_manifest(private)

    api_config = deepcopy(manifest)
    api_config["notion"] = Dict("api_version" => "2026-03-11")
    @test_throws W.StateValidationError W._validate_manifest(api_config)

    broken = deepcopy(manifest)
    broken["pages"]["example"]["mode"] = "autonomous-scientific-rewrite"
    @test_throws W.StateValidationError W._validate_manifest(broken)
end

@testset "state validation" begin
    manifest = minimal_manifest()
    state = minimal_state()
    @test W._validate_state(manifest, state)

    missing = deepcopy(state)
    delete!(missing["pages"], "example")
    @test_throws W.StateValidationError W._validate_state(manifest, missing)

    badsha = deepcopy(state)
    badsha["pages"]["example"]["verified_commit"] = "abc"
    @test_throws W.StateValidationError W._validate_state(manifest, badsha)
end

@testset "page filters fail closed" begin
    manifest = minimal_manifest()
    @test W._validate_page_key(manifest, "example") == "example"
    @test_throws W.StateValidationError W._validate_page_key(manifest, "typo")
end

@testset "drift materiality" begin
    empty = W.Drift("key", "Title", "repository-backed", "semantic",
        SHA0, SHA1, false, Dict{String,Any}[], Dict{String,Any}[], Dict{String,Any}[])
    @test !W.material(empty)

    changed = W.Drift("key", "Title", "repository-backed", "semantic",
        SHA0, SHA1, false,
        [Dict{String,Any}("path" => "src/a.jl",
            "before_blob" => "1", "after_blob" => "2")],
        Dict{String,Any}[], Dict{String,Any}[])
    @test W.material(changed)
end

@testset "packet digest determinism and tamper detection" begin
    snapshot = (
        schema_version=1,
        repository=(owner="owner", name="repo", branch="vmm"),
        page=(key="example", title="Example", authority="repository-backed"),
        current_head=SHA1,
        sources=[(path="src/file.jl", blob="0123456789012345678901234567890123456789")],
        issues=NamedTuple[],
        pull_requests=NamedTuple[],
    )
    @test W._snapshot_digest(snapshot) == W._snapshot_digest(snapshot)

    packet = Dict{String,Any}(
        "schema_version" => 1,
        "kind" => "cyaxiverse-wiki-reconciliation",
        "packet_id" => W._snapshot_digest(snapshot),
        "snapshot" => JSON3.read(JSON3.write(snapshot), Dict{String,Any}),
    )
    packet_id, parsed = W._validate_packet(packet)
    @test packet_id == W._snapshot_digest(snapshot)
    @test parsed.current_head == SHA1

    tampered = deepcopy(packet)
    tampered["snapshot"]["current_head"] = SHA0
    @test_throws W.StateValidationError W._validate_packet(tampered)
end

@testset "atomic state write and CAS" begin
    mktempdir() do dir
        path = joinpath(dir, "state.json")
        open(path, "w") do io
            write(io, "{\"a\":1}\n")
        end
        digest = W._file_digest(path)
        W._atomic_json_write(path, Dict("a" => 2); expected_digest=digest)
        @test occursin("\"a\":2", read(path, String))
        @test_throws W.StateValidationError W._atomic_json_write(
            path, Dict("a" => 3); expected_digest=digest)
    end
end

@testset "exit contract" begin
    @test W.EXIT_CLEAN == 0
    @test W.EXIT_DRIFT == 10
    @test W.EXIT_BROKEN_SOURCE == 11
    @test W.EXIT_OWNER_REVIEW == 12
end
