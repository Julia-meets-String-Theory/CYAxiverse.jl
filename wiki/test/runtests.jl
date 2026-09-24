using JSON3
using Test
using CYAxiverseWikiRefresh

const W = CYAxiverseWikiRefresh
const SHA0 = "995163f0058488ea183ac645045ed8b1636bef4a"
const SHA1 = "f02621377c3c01f1c5ae85ef5ffd00a219aa8b8d"
const SHA2 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
const BLOB0 = "1111111111111111111111111111111111111111"
const BLOB1 = "2222222222222222222222222222222222222222"

function issue_snapshot(state::String="open")
    Dict{String,Any}(
        "number" => 1,
        "title" => "Tracked issue",
        "state" => state,
        "state_reason" => state == "closed" ? "completed" : nothing,
        "closed_at" => state == "closed" ? "2026-09-23T00:00:00Z" : nothing,
    )
end

function pr_snapshot(state::String="open"; head_sha::String=SHA0)
    Dict{String,Any}(
        "number" => 2,
        "title" => "Tracked PR",
        "state" => state,
        "draft" => false,
        "merged_at" => state == "closed" ? "2026-09-23T00:00:00Z" : nothing,
        "head_sha" => head_sha,
        "base_ref" => "vmm",
    )
end

function minimal_manifest(; tracked=false)
    issues = tracked ? Any[1] : Any[]
    prs = tracked ? Any[2] : Any[]
    Dict{String,Any}(
        "schema_version" => 1,
        "initial_verified_commit" => SHA0,
        "repository" => Dict("owner" => "owner", "name" => "repo", "branch" => "vmm"),
        "pages" => Dict("example" => Dict(
            "title" => "Example",
            "authority" => "repository-backed",
            "mode" => "semantic",
            "sources" => ["src/file.jl"],
            "issues" => issues,
            "pull_requests" => prs,
        )),
    )
end

function minimal_state(; tracked=false)
    issues = tracked ? Dict("1" => issue_snapshot("open")) : Dict{String,Any}()
    prs = tracked ? Dict("2" => pr_snapshot("open"; head_sha=SHA0)) : Dict{String,Any}()
    Dict{String,Any}(
        "schema_version" => 1,
        "repository" => Dict("head" => SHA0, "branch" => "vmm"),
        "created_at" => "2026-09-23T00:00:00Z",
        "pages" => Dict("example" => Dict(
            "verified_commit" => SHA0,
            "reconciled_at" => "2026-09-23T00:00:00Z",
            "issues" => issues,
            "pull_requests" => prs,
        )),
    )
end

mutable struct FakeGitHub
    head::String
    tree::Dict{String,String}
    issues::Dict{Int,Dict{String,Any}}
    prs::Dict{Int,Dict{String,Any}}
end

W._tree_blobs(g::FakeGitHub, ref::AbstractString) = (g.head, copy(g.tree))
W._issue_snapshot(g::FakeGitHub, number::Integer) = deepcopy(g.issues[Int(number)])
W._pr_snapshot(g::FakeGitHub, number::Integer) = deepcopy(g.prs[Int(number)])

function current_fake()
    FakeGitHub(
        SHA1,
        Dict("src/file.jl" => BLOB1),
        Dict(1 => issue_snapshot("closed")),
        Dict(2 => pr_snapshot("closed"; head_sha=SHA1)),
    )
end

function tracked_snapshot(manifest, fake)
    W._snapshot_payload(
        manifest, "example", manifest["pages"]["example"],
        fake.head, fake.tree, fake,
        Dict{Int,Dict{String,Any}}(), Dict{Int,Dict{String,Any}}(),
    )
end

function tracked_drift()
    W.Drift(
        "example", "Example", "repository-backed", "semantic",
        SHA0, SHA1, true,
        [Dict{String,Any}(
            "path" => "src/file.jl",
            "before_blob" => BLOB0,
            "after_blob" => BLOB1,
        )],
        [Dict{String,Any}(
            "number" => 1,
            "before" => issue_snapshot("open"),
            "after" => issue_snapshot("closed"),
        )],
        [Dict{String,Any}(
            "number" => 2,
            "before" => pr_snapshot("open"; head_sha=SHA0),
            "after" => pr_snapshot("closed"; head_sha=SHA1),
        )],
    )
end

function write_packet(path, packet)
    open(path, "w") do io
        write(io, JSON3.write(packet))
        write(io, '\n')
    end
end

function write_state(path, state)
    open(path, "w") do io
        write(io, JSON3.write(state))
        write(io, '\n')
    end
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

    @test_throws W.StateValidationError W._validate_manifest(Any[1, 2, 3])
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

    @test_throws W.StateValidationError W._validate_state(manifest, Any["wrong-root"])
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
            "before_blob" => BLOB0, "after_blob" => BLOB1)],
        Dict{String,Any}[], Dict{String,Any}[])
    @test W.material(changed)
end

@testset "full packet identity and tamper detection" begin
    manifest = minimal_manifest(tracked=true)
    fake = current_fake()
    snapshot = tracked_snapshot(manifest, fake)
    drift = tracked_drift()
    packet_nt = W._packet_object(drift, snapshot)
    raw = W._plain(JSON3.read(JSON3.write(packet_nt)))

    packet_id, parsed, payload = W._validate_packet(raw)
    @test packet_id == String(raw["packet_id"])
    @test parsed.current_head == SHA1
    @test payload.baseline_verified_commit == SHA0

    mutators = [
        p -> (p["snapshot"]["current_head"] = SHA0),
        p -> (p["baseline_verified_commit"] = SHA2),
        p -> (p["changes"]["head_changed"] = false),
        p -> (p["changes"]["sources"][1]["after_blob"] = BLOB0),
        p -> (p["changes"]["issues"][1]["after"]["state"] = "open"),
        p -> (p["changes"]["pull_requests"][1]["after"]["state"] = "open"),
        p -> (p["execution_surface"] = "different surface"),
        p -> (p["instructions"][1] = "different instruction"),
    ]
    for mutate! in mutators
        tampered = deepcopy(raw)
        mutate!(tampered)
        @test_throws W.StateValidationError W._validate_packet(tampered)
    end

    malformed = Dict{String,Any}(
        "schema_version" => 1,
        "kind" => "cyaxiverse-wiki-reconciliation",
        "packet_id" => repeat("a", 64),
        "snapshot" => Dict("schema_version" => 1),
    )
    @test_throws W.StateValidationError W._validate_packet(malformed)
end

@testset "packet emission uses one frozen GitHub snapshot" begin
    manifest = minimal_manifest(tracked=true)
    fake = current_fake()
    snapshot = tracked_snapshot(manifest, fake)
    drift = tracked_drift()

    # GitHub moves after drift/snapshot collection. Packet emission must not
    # refetch and silently combine the old change summary with new live state.
    fake.issues[1] = issue_snapshot("open")
    fake.prs[2] = pr_snapshot("open"; head_sha=SHA2)

    mktempdir() do dir
        path, _ = W._write_packet(dir, drift, snapshot)
        packet = W._plain(JSON3.read(read(path, String)))
        _, parsed, payload = W._validate_packet(packet)
        @test parsed.issues[1].state == "closed"
        @test parsed.pull_requests[1].state == "closed"
        @test payload.changes.issues[1].after.state == "closed"
        @test payload.changes.pull_requests[1].after.state == "closed"
    end
end

@testset "locked state replacement serializes concurrent writers" begin
    mktempdir() do dir
        path = joinpath(dir, "state.json")
        write_state(path, Dict("a" => 1))
        digest = W._file_digest(path)

        ready = Channel{Bool}(1)
        release = Channel{Bool}(1)
        first = @async W._locked_atomic_json_write(
            path, Dict("a" => 2);
            expected_digest=digest,
            before_replace=() -> begin
                put!(ready, true)
                take!(release)
            end,
        )

        status = timedwait(() -> isready(ready) || istaskdone(first), 5.0)
        @test status == :ok
        if istaskdone(first)
            wait(first)  # surfaces the worker failure instead of hanging
        else
            take!(ready)
            @test_throws W.StateValidationError W._locked_atomic_json_write(
                path, Dict("a" => 3); expected_digest=digest)
            put!(release, true)
            wait(first)
            final = W._plain(JSON3.read(read(path, String)))
            @test final["a"] == 2
            @test !isdir(path * ".lock")
        end
    end
end

@testset "accept transaction binds packet and live GitHub state" begin
    manifest = minimal_manifest(tracked=true)
    drift = tracked_drift()

    mktempdir() do dir
        packet_path = joinpath(dir, "packet.json")
        state_path = joinpath(dir, "state.json")

        fake = current_fake()
        snapshot = tracked_snapshot(manifest, fake)
        packet = W._packet_object(drift, snapshot)
        write_packet(packet_path, packet)

        state = minimal_state(tracked=true)
        write_state(state_path, state)
        digest = W._file_digest(state_path)

        @test W._accept_page!(
            manifest, state, digest, fake, state_path,
            "example", packet_path, SHA1, true) == W.EXIT_CLEAN

        accepted = W._plain(JSON3.read(read(state_path, String)))
        @test accepted["pages"]["example"]["verified_commit"] == SHA1
        @test accepted["pages"]["example"]["issues"]["1"]["state"] == "closed"
        @test accepted["pages"]["example"]["pull_requests"]["2"]["state"] == "closed"
        @test accepted["pages"]["example"]["accepted_packet_id"] == packet.packet_id
    end

    # Issue movement after reconciliation invalidates the packet.
    mktempdir() do dir
        packet_path = joinpath(dir, "packet.json")
        state_path = joinpath(dir, "state.json")
        fake = current_fake()
        packet = W._packet_object(drift, tracked_snapshot(manifest, fake))
        write_packet(packet_path, packet)
        state = minimal_state(tracked=true)
        write_state(state_path, state)
        digest = W._file_digest(state_path)
        fake.issues[1] = issue_snapshot("open")
        @test_throws W.StateValidationError W._accept_page!(
            manifest, state, digest, fake, state_path,
            "example", packet_path, SHA1, true)
    end

    # PR movement after reconciliation invalidates the packet.
    mktempdir() do dir
        packet_path = joinpath(dir, "packet.json")
        state_path = joinpath(dir, "state.json")
        fake = current_fake()
        packet = W._packet_object(drift, tracked_snapshot(manifest, fake))
        write_packet(packet_path, packet)
        state = minimal_state(tracked=true)
        write_state(state_path, state)
        digest = W._file_digest(state_path)
        fake.prs[2] = pr_snapshot("open"; head_sha=SHA2)
        @test_throws W.StateValidationError W._accept_page!(
            manifest, state, digest, fake, state_path,
            "example", packet_path, SHA1, true)
    end

    # vmm movement invalidates the packet.
    mktempdir() do dir
        packet_path = joinpath(dir, "packet.json")
        state_path = joinpath(dir, "state.json")
        fake = current_fake()
        packet = W._packet_object(drift, tracked_snapshot(manifest, fake))
        write_packet(packet_path, packet)
        state = minimal_state(tracked=true)
        write_state(state_path, state)
        digest = W._file_digest(state_path)
        fake.head = SHA2
        @test_throws W.StateValidationError W._accept_page!(
            manifest, state, digest, fake, state_path,
            "example", packet_path, SHA1, true)
    end

    # Wrong page and repository identities fail closed even with valid packet IDs.
    mktempdir() do dir
        state_path = joinpath(dir, "state.json")
        packet_path = joinpath(dir, "packet.json")
        fake = current_fake()
        base_snapshot = tracked_snapshot(manifest, fake)

        wrong_page_snapshot = merge(base_snapshot, (
            page=(key="other", title="Other", authority="repository-backed"),))
        write_packet(packet_path, W._packet_object(drift, wrong_page_snapshot))
        state = minimal_state(tracked=true)
        write_state(state_path, state)
        @test_throws W.StateValidationError W._accept_page!(
            manifest, state, W._file_digest(state_path), fake, state_path,
            "example", packet_path, SHA1, true)

        wrong_repo_snapshot = merge(base_snapshot, (
            repository=(owner="other", name="repo", branch="vmm"),))
        write_packet(packet_path, W._packet_object(drift, wrong_repo_snapshot))
        state = minimal_state(tracked=true)
        write_state(state_path, state)
        @test_throws W.StateValidationError W._accept_page!(
            manifest, state, W._file_digest(state_path), fake, state_path,
            "example", packet_path, SHA1, true)
    end
end

@testset "exit contract" begin
    @test W.EXIT_CLEAN == 0
    @test W.EXIT_DRIFT == 10
    @test W.EXIT_BROKEN_SOURCE == 11
    @test W.EXIT_OWNER_REVIEW == 12
end
