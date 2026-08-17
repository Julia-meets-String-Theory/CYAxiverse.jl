#!/usr/bin/env julia

"""
    scripts/migrate_quartic_index_ordering.jl

Relabel the stored quartic component indices written by
`CYAxiverse.generate.hp_spectrum_save`.

## The defect

`hp_spectrum` used to build its quartic component lists with a
two-dimensional comprehension (`[(x, x, x, y) for x=1:h11, y=1:h11 if x!=y]`
and `[(x, x, y, y) for x=1:h11, y=1:h11 if x>y]`), which iterates
column-major (first index fastest). The fused instanton accumulation loop
that fills the corresponding value arrays runs the first index slowest.
Values and labels were therefore permuted relative to each other: every
`quart31` component is mislabelled for `h11 >= 2`, and `quart22` components
are mislabelled for `h11 >= 4` (`h11 <= 3` coincidentally agree, and
`h11 == 1` is always empty). The accumulated `log10` and `sign` values were
always correct; only the `index` labels attached to them were wrong.

`generate.hp_spectrum` now builds the same index lists with the flattened
loop order already used by `generate.pq_spectrum`
(`[(i, i, i, j) for i in 1:h11 for j in 1:h11 if i != j]` and
`[(i, i, j, j) for i in 1:h11 for j in 1:i-1]`), so the two spectrum paths
agree. Files already written with that flattened order do not need
migrating.

## What this script modifies

    spectrum/quart31/index
    spectrum/quart22/index

and only when the stored matrix is exactly the old two-dimensional order for
that dataset's own `h11` (recovered from the column count and cross-checked
against the geometry's path). The dataset is deleted and rewritten with the
current source's ordering, as an `Int` matrix compressed with `deflate=9`,
matching every other writer in this codebase.

## What this script never modifies

    spectrum/quart31/log10   spectrum/quart31/sign
    spectrum/quart22/log10   spectrum/quart22/sign

and every other dataset in the file. These datasets are never opened for
writing, and the values used to accumulate them are never read by this
script: correctness here is a question of column labelling only.

## Idempotency and unrecognised data

Every `index` dataset is classified before anything is written:

  - matches the current source's order  -> already correct; left alone.
  - matches the old two-dimensional order for its own `h11` -> migrated.
  - matches neither -> unrecognised; left alone and logged prominently.

Because `pq_spectrum` always used the flattened order, this classification
is also the only thing distinguishing `hp_spectrum`-written files from
`pq_spectrum`-written ones. Re-running this script after a successful
`--apply` pass reports only "already correct" and writes nothing.

Writing is opt-in: without `--apply` the script only reports what it would
do.
"""

using CYAxiverse
using HDF5
using Printf

const GeometryIndex = CYAxiverse.structs.GeometryIndex

const QUARTIC_KINDS = (:quart31, :quart22)
const QUARTIC_PROGRESS_INTERVAL = 2000
const QUARTIC_SUMMARY_HEADER =
    "h11,polytope,frst,dataset,columns,derived_h11,status,applied,detail,path"

"""Print command-line usage for the quartic index migration."""
function _quartic_usage()
    println("""
    Usage:
      julia --project=. scripts/migrate_quartic_index_ordering.jl [options]

    Rewrites spectrum/quart31/index and spectrum/quart22/index into the
    component order produced by the current source. spectrum/quart31/log10,
    spectrum/quart31/sign, spectrum/quart22/log10, and spectrum/quart22/sign
    are never read for the classification decision and never written.

    Reporting only unless --apply is given.

    Options:
      --data-dir PATH        Data root containing h11_*/np_*/cy_*/cyax.h5.
      --h11 N[,N...]         Restrict selection to these h11. May be repeated.
      --geometry H,P,F       Restrict to one explicit geometry. May be repeated.
      --limit N              Examine at most N selected geometries.
      --offset N             Skip the first N selected geometries.
      --apply                Opt in to rewriting index datasets that carry the
                             old ordering. Without this flag nothing is written.
      --summary PATH         CSV report path, one row per examined dataset.
      --append-summary       Append to an existing summary instead of replacing it.
      --verbose              Print a line for every geometry, including those
                             carrying no quartic datasets.
    """)
end

"""Parse migration command-line arguments into a typed options tuple."""
function _quartic_parse_args(args)
    options = (data_dir="", h11=Int[], geometries=GeometryIndex[], limit=nothing,
        offset=0, apply=false, summary="", append_summary=false, verbose=false)
    valued = ("--data-dir", "--h11", "--geometry", "--limit", "--offset", "--summary")
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            _quartic_usage(); exit(0)
        elseif arg == "--apply"
            options = merge(options, (; apply=true))
        elseif arg == "--append-summary"
            options = merge(options, (; append_summary=true))
        elseif arg == "--verbose"
            options = merge(options, (; verbose=true))
        elseif arg in valued
            i == length(args) && error("missing value for $arg")
            value = args[i + 1]
            if arg == "--data-dir"
                options = merge(options, (; data_dir=value))
            elseif arg == "--h11"
                append!(options.h11, parse.(Int, split(value, ",")))
            elseif arg == "--geometry"
                parts = parse.(Int, split(value, ","))
                length(parts) == 3 || error("--geometry must be H,P,F")
                push!(options.geometries, GeometryIndex(parts...))
            elseif arg == "--limit"
                options = merge(options, (; limit=parse(Int, value)))
            elseif arg == "--offset"
                options = merge(options, (; offset=parse(Int, value)))
            elseif arg == "--summary"
                options = merge(options, (; summary=value))
            end
            i += 1
        else
            error("unknown option: $arg")
        end
        i += 1
    end
    options[:offset] >= 0 || error("--offset must be nonnegative")
    options[:limit] === nothing || options[:limit] > 0 || error("--limit must be positive")
    all(>(0), options[:h11]) || error("--h11 must be positive")
    options
end

"""Parse the integer suffix of a directory name with the given prefix."""
function _quartic_prefixed_int(name::AbstractString, prefix::AbstractString)
    startswith(name, prefix) || return nothing
    try
        parse(Int, name[(lastindex(prefix) + 1):end])
    catch
        nothing
    end
end

"""Find geometries by scanning the selected database directory tree."""
function _quartic_scanned_geometries(h11_filter::AbstractVector{Int})
    root = CYAxiverse.filestructure.present_dir()
    isdir(root) || return GeometryIndex[]
    h11_dirs = isempty(h11_filter) ?
        sort(filter(name -> startswith(name, "h11_"), readdir(root))) :
        [string("h11_", lpad(h11, 3, "0")) for h11 in sort(unique(h11_filter))]
    geoms = GeometryIndex[]
    for h11_dir in h11_dirs
        h11 = _quartic_prefixed_int(h11_dir, "h11_")
        h11 === nothing && continue
        h11_path = joinpath(root, h11_dir)
        isdir(h11_path) || continue
        for np_dir in sort(filter(name -> startswith(name, "np_"), readdir(h11_path)))
            polytope = _quartic_prefixed_int(np_dir, "np_")
            polytope === nothing && continue
            np_path = joinpath(h11_path, np_dir)
            for cy_dir in sort(filter(name -> startswith(name, "cy_"), readdir(np_path)))
                frst = _quartic_prefixed_int(cy_dir, "cy_")
                frst === nothing && continue
                isfile(joinpath(np_path, cy_dir, "cyax.h5")) || continue
                push!(geoms, GeometryIndex(h11, polytope, frst))
            end
        end
    end
    geoms
end

"""
Find geometries from the package path index.

Only an index that already exists on disk is consulted:
`CYAxiverse.filestructure.paths_cy()` generates and writes `paths_cy.h5` as a
side effect when neither index file exists yet, and a reporting-only run of
this script must never write to the data root.
"""
function _quartic_indexed_geometries(h11_filter::AbstractVector{Int})
    root = CYAxiverse.filestructure.present_dir()
    (isfile(joinpath(root, "paths.h5")) || isfile(joinpath(root, "paths_cy.h5"))) ||
        return GeometryIndex[]
    try
        _, pathinds = CYAxiverse.filestructure.paths_cy()
        return [GeometryIndex(col...) for col in eachcol(pathinds)
                if isempty(h11_filter) || col[1] in h11_filter]
    catch
        GeometryIndex[]
    end
end

"""Combine explicit, indexed, and scanned selections with offset and limit."""
function _quartic_selected_geometries(options)
    geoms = if isempty(options[:geometries])
        indexed = _quartic_indexed_geometries(options[:h11])
        scanned = _quartic_scanned_geometries(options[:h11])
        length(scanned) > length(indexed) ? scanned : indexed
    else
        copy(options[:geometries])
    end
    first_index = min(options[:offset] + 1, length(geoms) + 1)
    geoms = geoms[first_index:end]
    options[:limit] === nothing ? geoms : geoms[1:min(options[:limit], length(geoms))]
end

"""
Return the stored (two-dimensional comprehension) component order.

This is the order the pre-fix `hp_spectrum` used to label its quartic
components: a filtered two-dimensional comprehension, which iterates
column-major and so runs the first index fastest against the value
accumulation loop, which runs it slowest.
"""
function _quartic_old_components(kind::Symbol, h11::Int)
    kind === :quart31 ? [(x, x, x, y) for x = 1:h11, y = 1:h11 if x != y] :
                        [(x, x, y, y) for x = 1:h11, y = 1:h11 if x > y]
end

"""
Return the current source's component order.

This is the flattened loop order the value accumulation always used, and the
order `generate.hp_spectrum` and `generate.pq_spectrum` now both emit.
"""
function _quartic_new_components(kind::Symbol, h11::Int)
    kind === :quart31 ? [(i, i, i, j) for i in 1:h11 for j in 1:h11 if i != j] :
                        [(i, i, j, j) for i in 1:h11 for j in 1:i-1]
end

"""Build the zero-based 4 x N index matrix that `hp_spectrum_save` stores."""
function _quartic_index_matrix(components)
    matrix = zeros(Int, 4, length(components))
    for column in eachindex(components)
        for row in 1:4
            matrix[row, column] = components[column][row] - 1
        end
    end
    matrix
end

"""
Recover `h11` from the column count of a quartic index dataset.

`quart31` has `h11 * (h11 - 1)` columns and `quart22` has
`h11 * (h11 - 1) / 2`. Return `nothing` when no positive integer `h11` fits
(including a negative or non-triangular column count).
"""
function _quartic_derived_h11(kind::Symbol, columns::Integer)
    columns >= 0 || return nothing
    discriminant = kind === :quart31 ? 1 + 4 * columns : 1 + 8 * columns
    root = isqrt(discriminant)
    root * root == discriminant || return nothing
    iseven(1 + root) || return nothing
    h11 = (1 + root) ÷ 2
    h11 >= 1 || return nothing
    expected = kind === :quart31 ? h11 * (h11 - 1) : (h11 * (h11 - 1)) ÷ 2
    expected == columns ? h11 : nothing
end

"""
Classify a stored index matrix against the two candidate orderings.

The new order is tested first so that the cases where the two coincide
(`quart22` at `h11 <= 3`, and every empty dataset at `h11 == 1`) are reported
as already correct and never rewritten.
"""
function _quartic_classify(stored, old_matrix, new_matrix)
    stored == new_matrix && return :correct
    stored == old_matrix && return :old
    :unrecognised
end

"""Format one dataset status for a progress line."""
function _quartic_status_label(status::Symbol, dry_run::Bool)
    status === :migrated && return dry_run ? "would-migrate" : "migrated"
    status === :correct && return "already-correct"
    status === :unrecognised && return "UNRECOGNISED"
    status === :h11_mismatch && return "h11-mismatch"
    string(status)
end

"""
Examine one geometry file without writing to it.

Return the per-dataset records and the rewrites they call for. The file is
opened read-only; `log10` and `sign` datasets are never read here, and
groups other than `spectrum/quart31` and `spectrum/quart22` are never
opened.
"""
function _quartic_examine(path, geom_idx)
    records = NamedTuple[]
    writes = NamedTuple[]
    h5open(path, "r") do file
        haskey(file, "spectrum") || return nothing
        spectrum = file["spectrum"]
        for kind in QUARTIC_KINDS
            name = String(kind)
            haskey(spectrum, name) || continue
            group = spectrum[name]
            if !haskey(group, "index")
                push!(records, (; geom=geom_idx, kind, columns=-1, derived_h11=nothing,
                    status=:unrecognised, detail="group has no index dataset", path))
                continue
            end
            stored = read(group, "index")
            if !(stored isa AbstractMatrix && eltype(stored) <: Integer && size(stored, 1) == 4)
                push!(records, (; geom=geom_idx, kind, columns=-1, derived_h11=nothing,
                    status=:unrecognised,
                    detail="expected a 4 x N integer matrix, found $(typeof(stored)) of size $(size(stored))",
                    path))
                continue
            end
            columns = size(stored, 2)
            derived_h11 = _quartic_derived_h11(kind, columns)
            if derived_h11 === nothing
                push!(records, (; geom=geom_idx, kind, columns, derived_h11,
                    status=:unrecognised,
                    detail="column count $columns is not a valid $name width for any h11", path))
                continue
            end
            if derived_h11 != geom_idx.h11
                push!(records, (; geom=geom_idx, kind, columns, derived_h11,
                    status=:h11_mismatch,
                    detail="dataset implies h11=$derived_h11, geometry path implies h11=$(geom_idx.h11)",
                    path))
                continue
            end
            old_matrix = _quartic_index_matrix(_quartic_old_components(kind, derived_h11))
            new_matrix = _quartic_index_matrix(_quartic_new_components(kind, derived_h11))
            classification = _quartic_classify(stored, old_matrix, new_matrix)
            if classification === :unrecognised
                push!(records, (; geom=geom_idx, kind, columns, derived_h11,
                    status=:unrecognised,
                    detail="stored order matches neither the old nor the current labelling",
                    path))
                continue
            end
            status = classification === :old ? :migrated : :correct
            status === :migrated && push!(writes, (; kind,
                matrix=convert(Matrix{eltype(stored)}, new_matrix)))
            detail = classification === :old ? "old two-dimensional comprehension order" :
                "already in the current source order"
            push!(records, (; geom=geom_idx, kind, columns, derived_h11, status, detail, path))
        end
        return nothing
    end
    records, writes
end

"""
Rewrite the selected `index` datasets in place.

HDF5.jl cannot reassign a dataset, so each one is deleted and recreated with
`deflate=9`, matching the compression every other writer in this codebase
uses for these datasets. Only `index` datasets are opened for writing;
`log10` and `sign` are never touched.
"""
function _quartic_rewrite!(path, writes)
    h5open(path, "r+") do file
        spectrum = file["spectrum"]
        for write_request in writes
            group = spectrum[String(write_request.kind)]
            HDF5.delete_object(group, "index")
            group["index", deflate=9] = write_request.matrix
        end
    end
    nothing
end

"""Quote a value when needed for the migration summary CSV."""
function _quartic_csv_escape(value)
    text = replace(string(value), '"' => "\"\"")
    occursin(r"[,\"\n]", text) ? string('"', text, '"') : text
end

"""Create the migration summary CSV and write its header when needed."""
function _quartic_write_summary_header(path; append=false)
    append && isfile(path) && return
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, QUARTIC_SUMMARY_HEADER)
    end
end

"""Append one examined dataset to the migration summary CSV."""
function _quartic_append_summary(path, record, applied::Bool)
    values = (record.geom.h11, record.geom.polytope, record.geom.frst, record.kind,
        record.columns < 0 ? "" : record.columns,
        record.derived_h11 === nothing ? "" : record.derived_h11,
        record.status, applied, record.detail, record.path)
    open(path, "a") do io
        println(io, join(_quartic_csv_escape.(values), ','))
        flush(io)
    end
end

"""Print the per-h11 breakdown of examined datasets."""
function _quartic_report_per_h11(per_h11, dry_run)
    isempty(per_h11) && return
    statuses = (:migrated, :correct, :unrecognised, :h11_mismatch)
    println("\nPer-h11 counts:")
    for h11 in sort(collect(keys(per_h11)))
        for kind in QUARTIC_KINDS
            counts = get(per_h11[h11], kind, Dict{Symbol,Int}())
            isempty(counts) && continue
            fields = ["$(_quartic_status_label(status, dry_run))=$(counts[status])"
                      for status in statuses if get(counts, status, 0) > 0]
            @printf("  h11=%-4d %-8s %s\n", h11, String(kind), join(fields, " "))
        end
    end
end

"""
    run_quartic_index_migration(options)

Examine the selected geometries and, when `--apply` is given, rewrite the
`spectrum/quart31/index` and `spectrum/quart22/index` datasets that still
carry the old two-dimensional labelling. `log10` and `sign` datasets are
never read or written. Return a summary named tuple; `success` is false only
when a file could not be examined or rewritten (missing files, malformed
groups, and unrecognised orderings are all reported but do not affect
`success`).
"""
function run_quartic_index_migration(options)
    data_dir = CYAxiverse.filestructure.resolve_data_dir(options[:data_dir])
    previous_data_dir = get(ENV, "CYAXIVERSE_DATA_DIR", nothing)
    ENV["CYAXIVERSE_DATA_DIR"] = data_dir
    try
        dry_run = !options[:apply]
        geoms = _quartic_selected_geometries(options)
        summary_path = options[:summary]
        isempty(summary_path) ||
            _quartic_write_summary_header(summary_path; append=options[:append_summary])
        @printf("Quartic index migration: %d geometries selected from %s (%s)\n",
            length(geoms), data_dir, dry_run ? "dry run, nothing will be written" : "applying")

        counts = Dict{Symbol,Int}(:migrated => 0, :correct => 0,
            :unrecognised => 0, :h11_mismatch => 0)
        per_h11 = Dict{Int,Dict{Symbol,Dict{Symbol,Int}}}()
        unrecognised_paths = String[]
        files_with_quartics = 0
        files_written = 0
        failed = 0

        for (index, geom_idx) in enumerate(geoms)
            path = CYAxiverse.filestructure.cyax_file(geom_idx)
            if !isfile(path)
                failed += 1
                println(stderr, "MISSING h11=$(geom_idx.h11) polytope=$(geom_idx.polytope) " *
                    "frst=$(geom_idx.frst): no cyax.h5 at $path")
                continue
            end
            local records, writes
            try
                records, writes = _quartic_examine(path, geom_idx)
            catch err
                failed += 1
                println(stderr, "FAILED $path: ", sprint(showerror, err))
                continue
            end
            if isempty(records)
                options[:verbose] && @printf("[%d/%d] h11=%d polytope=%d frst=%d no quartic datasets\n",
                    index, length(geoms), geom_idx.h11, geom_idx.polytope, geom_idx.frst)
                index % QUARTIC_PROGRESS_INTERVAL == 0 &&
                    @printf("  ... %d/%d geometries examined\n", index, length(geoms))
                continue
            end
            files_with_quartics += 1

            applied = false
            if !dry_run && !isempty(writes)
                try
                    _quartic_rewrite!(path, writes)
                    applied = true
                    files_written += 1
                catch err
                    failed += 1
                    println(stderr, "FAILED $path: ", sprint(showerror, err))
                end
            end

            for record in records
                counts[record.status] = get(counts, record.status, 0) + 1
                kinds = get!(per_h11, record.geom.h11, Dict{Symbol,Dict{Symbol,Int}}())
                statuses = get!(kinds, record.kind, Dict{Symbol,Int}())
                statuses[record.status] = get(statuses, record.status, 0) + 1
                if record.status === :unrecognised
                    push!(unrecognised_paths, string(path, " :: spectrum/", record.kind, "/index"))
                    println(stderr, "UNRECOGNISED spectrum/$(record.kind)/index in $path: ",
                        record.detail, " -- not written")
                elseif record.status === :h11_mismatch
                    println(stderr, "H11 MISMATCH spectrum/$(record.kind)/index in $path: ",
                        record.detail, " -- not written")
                end
                isempty(summary_path) || _quartic_append_summary(summary_path, record,
                    applied && record.status === :migrated)
            end

            labels = ["$(record.kind)=$(_quartic_status_label(record.status, dry_run))"
                      for record in records]
            @printf("[%d/%d] h11=%d polytope=%d frst=%d %s\n", index, length(geoms),
                geom_idx.h11, geom_idx.polytope, geom_idx.frst, join(labels, " "))
        end

        _quartic_report_per_h11(per_h11, dry_run)
        if !isempty(unrecognised_paths)
            println(stderr, "\nUnrecognised index datasets (left untouched):")
            for entry in unrecognised_paths
                println(stderr, "  ", entry)
            end
        end
        @printf("\n%s: geometries=%d with-quartics=%d files-written=%d %s=%d already-correct=%d unrecognised=%d h11-mismatch=%d failed=%d\n",
            dry_run ? "dry run" : "applied", length(geoms), files_with_quartics, files_written,
            dry_run ? "would-migrate" : "migrated", counts[:migrated],
            counts[:correct], counts[:unrecognised], counts[:h11_mismatch], failed)
        dry_run && counts[:migrated] > 0 &&
            println("Re-run with --apply to rewrite the index datasets listed above.")

        (; dry_run, geometries=length(geoms), files_with_quartics, files_written,
            migrated=counts[:migrated], already_correct=counts[:correct],
            unrecognised=counts[:unrecognised], h11_mismatch=counts[:h11_mismatch], failed,
            per_h11, unrecognised_paths, success=failed == 0)
    finally
        previous_data_dir === nothing ? delete!(ENV, "CYAXIVERSE_DATA_DIR") :
            (ENV["CYAXIVERSE_DATA_DIR"] = previous_data_dir)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_quartic_index_migration(_quartic_parse_args(ARGS)).success || exit(1)
end
