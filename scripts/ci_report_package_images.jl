# CYAX-0230 R-001: report the runner CPU and the package images restored from
# the CI depot cache, and how many of them this Julia would reject, and why.
#
# Diagnostic only. It reads the first depot and loads no package, so it writes
# nothing there; the workflow step counts depot entries changed while it runs.
# It calls internal Base functions of Julia 1.12 (`parse_cache_header`,
# `check_clone_targets`), in the order `Base.stale_cachefile` checks a cache
# file: header, CPU targets, object file. A Julia update may break it; the step
# is non-fatal, so a break shows as a missing table (spec, accepted limitations).
#
# Usage: julia --startup-file=no scripts/ci_report_package_images.jl

const CORE_MODULES = ("Core", "Base", "Main")

struct ImageFile
    package::String
    source::String       # version of the source tree it was built from (see source_label)
    target::String       # recorded CPU target names, "none" without native code
    flags::String        # cache flags that differ from Julia's defaults
    readable::Bool       # header this Julia can read
    target_rejected::Bool
    object_missing::Bool
    non_stdlib_deps::Bool # required modules beyond Core, Base, Main and Sys.STDLIB
    problem::String      # why the header is unreadable, "" when readable
end

passes(f::ImageFile) = f.readable && !f.target_rejected && !f.object_missing

function flags_label(flags::UInt8)
    opts = Base.translate_cache_flags(Base.CacheFlags(flags), Base.DefaultCacheFlags)
    return isempty(opts) ? "default" : join(opts, " ")
end

const STDLIB = Dict{Base.PkgId,Bool}()
function is_core_or_stdlib(id::Base.PkgId)
    id.name in CORE_MODULES && id.uuid === nothing && return true
    id.uuid === nothing && return false
    return get!(() -> Base.is_stdlib(id), STDLIB, id)
end

unreadable(package, problem) =
    ImageFile(package, "?", "(unreadable)", "(unreadable)", false, false, false, false, problem)

# The version of the package source a cache file was built from, so that files of
# an older version (which Julia rejects later, as stale) can be told apart from
# the version a job uses; the job log prints that version ("[uuid] Name vX.Y.Z").
# From the header's first include file: "vX.Y.Z" from the nearest project file
# above it ("no version" if that file has none); "no source" when the include
# file is not in the depot, with its tree when the path is packages/<Name>/<tree>/;
# "?" when there is no include or no readable project file.
function source_label(includes)
    isempty(includes) && return "?"
    file = first(includes).filename
    if !isfile(file)
        # Pkg names each tree directory with Base.version_slug: 5 characters of
        # Base.slug_chars ([A-Za-z0-9]).
        m = match(r"/packages/[^/]+/([A-Za-z0-9]{5})/", file)
        return m === nothing ? "no source" : "no source, tree $(m[1])"
    end
    dir = dirname(file)
    while true
        for name in ("JuliaProject.toml", "Project.toml")
            project = joinpath(dir, name)
            isfile(project) || continue
            version = try
                get(Base.parsed_toml(project), "version", nothing)
            catch err
                err isa InterruptException && rethrow()
                return "?"
            end
            return version === nothing ? "no version" : "v" * lstrip(string(version), 'v')
        end
        parent = dirname(dir)
        parent == dir && return "?"
        dir = parent
    end
end

function inspect(package::String, path::String)
    io = open(path, "r")
    try
        iszero(Base.isvalid_cache_header(io)) && return unreadable(package, "incompatible header")
        modules, (includes, _, _), required, _, _, _, clone_targets, flags =
            Base.parse_cache_header(io, path)
        isempty(modules) && return unreadable(package, "empty module list")
        source = source_label(includes)
        non_stdlib = any(!is_core_or_stdlib(first(req)) for req in required)
        if isempty(clone_targets)
            return ImageFile(package, source, "none", flags_label(flags), true, false, false,
                             non_stdlib, "")
        end
        target = join((t.name for t in Base.parse_image_targets(clone_targets)), ";")
        rejected = Base.check_clone_targets(clone_targets) !== nothing
        missing_object = !isfile(Base.ocachefile_from_cachefile(path))
        return ImageFile(package, source, target, flags_label(flags), true, rejected,
                         missing_object, non_stdlib, "")
    catch err
        # Any failure on one file (a damaged header can throw BoundsError,
        # InexactError and others) marks that file unreadable, not the report.
        err isa InterruptException && rethrow()
        return unreadable(package, string(nameof(typeof(err))))
    finally
        close(io)
    end
end

function image_files(compiled::String)
    files = ImageFile[]
    isdir(compiled) || return files
    for entry in sort(readdir(compiled))
        path = joinpath(compiled, entry)
        if isdir(path)
            for file in sort(readdir(path))
                endswith(file, ".ji") && push!(files, inspect(entry, joinpath(path, file)))
            end
        elseif endswith(entry, ".ji")
            push!(files, inspect(chopsuffix(entry, ".ji"), path))
        end
    end
    return files
end

function print_table(header::Vector{String}, rows::Vector{Vector{String}})
    widths = [maximum(length, (r[i] for r in vcat([header], rows))) for i in eachindex(header)]
    for row in vcat([header], rows)
        println(join((rpad(cell, w) for (cell, w) in zip(row, widths)), "  ") |> rstrip)
    end
end

yes_no_mixed(values) = all(values) ? "yes" : any(values) ? "mixed" : "no"

function main()
    info = Sys.cpu_info()
    println("CPU model: ", isempty(info) ? "unknown" : strip(info[1].model))
    println("Sys.CPU_NAME: ", Sys.CPU_NAME)
    println("JULIA_CPU_TARGET: ", get(ENV, "JULIA_CPU_TARGET", "not set"))

    version = "v$(VERSION.major).$(VERSION.minor)"
    files = image_files(joinpath(first(DEPOT_PATH), "compiled", version))
    packages = length(unique(f.package for f in files))
    println("Restored package images: ", length(files), " files of ", packages,
            " packages in compiled/", version)
    isempty(files) && return

    println()
    println("By recorded CPU target and cache flags (a file can count in more than one column):")
    groups = Dict{Tuple{String,String},Vector{ImageFile}}()
    for f in files
        push!(get!(Vector{ImageFile}, groups, (f.target, f.flags)), f)
    end
    rows = [[string(length(g)),
             string(count(!f.readable for f in g)),
             string(count(f.object_missing for f in g)),
             string(count(f.target_rejected for f in g)),
             flags, target]
            for ((target, flags), g) in sort(collect(groups); by=first)]
    print_table(["files", "unreadable_header", "no_object_file", "rejected_by_target",
                 "flags", "cpu_target"], rows)
    problems = Dict{String,Int}()
    for f in files
        f.readable || (problems[f.problem] = get(problems, f.problem, 0) + 1)
    end
    isempty(problems) || println("Unreadable headers by reason: ",
                                 join(("$k $v" for (k, v) in sort(collect(problems))), ", "))

    println()
    println("::group::Restored package images per package, source version and cache flags")
    bypkg = Dict{Tuple{String,String,String},Vector{ImageFile}}()
    for f in files
        push!(get!(Vector{ImageFile}, bypkg, (f.package, f.source, f.flags)), f)
    end
    rows = Vector{String}[]
    for ((package, source, flags), g) in sort(collect(bypkg); by=first)
        readable = filter(f -> f.readable, g)
        deps = isempty(readable) ? "?" : yes_no_mixed(f.non_stdlib_deps for f in readable)
        push!(rows, [package, source, flags, string(length(g)),
                     string(count(f.target_rejected for f in g)),
                     string(count(passes, g)), deps])
    end
    print_table(["package", "source", "flags", "files", "rejected_by_target", "passing",
                 "non_stdlib_deps"], rows)
    println("::endgroup::")
    return
end

main()
