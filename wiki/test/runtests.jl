using Test
using CYAxiverseWikiRefresh

const W = CYAxiverseWikiRefresh

@testset "verification marker" begin
    md = """
## Example

**Status:** Reviewed
**Verified against:** `vmm@995163f0058488ea183ac645045ed8b1636bef4a`
"""
    @test W._verified_marker(md) ==
        "**Verified against:** `vmm@995163f0058488ea183ac645045ed8b1636bef4a`"
    @test W._verified_marker("No verification metadata here") === nothing
end

@testset "manifest validation and privacy" begin
    manifest = Dict{String,Any}(
        "schema_version" => 1,
        "initial_verified_commit" => "995163f0058488ea183ac645045ed8b1636bef4a",
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
    @test W._validate_manifest(manifest)

    private = deepcopy(manifest)
    private["pages"]["example"]["notion_page_id"] = "private-id"
    @test_throws ArgumentError W._validate_manifest(private)

    broken = deepcopy(manifest)
    broken["pages"]["example"]["mode"] = "autonomous-scientific-rewrite"
    @test_throws ArgumentError W._validate_manifest(broken)
end

@testset "drift materiality" begin
    empty = W.Drift("key", "Title", "repository-backed", "semantic",
        "abc", "def", false, Dict{String,Any}[], Dict{String,Any}[], Dict{String,Any}[])
    @test !W.material(empty)

    changed = W.Drift("key", "Title", "repository-backed", "semantic",
        "abc", "def", false,
        [Dict{String,Any}("path" => "src/a.jl",
            "before_blob" => "1", "after_blob" => "2")],
        Dict{String,Any}[], Dict{String,Any}[])
    @test W.material(changed)
end

@testset "exit contract" begin
    @test W.EXIT_CLEAN == 0
    @test W.EXIT_DRIFT == 10
    @test W.EXIT_BROKEN_SOURCE == 11
    @test W.EXIT_OWNER_REVIEW == 12
end
