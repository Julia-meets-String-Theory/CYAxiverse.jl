#!/usr/bin/env julia

using HDF5
using CYAxiverse

"""Read a command-line option from `ARGS`, returning `default` if absent."""
function option_value(name, default)
    index = findfirst(==(name), ARGS)
    index === nothing && return default
    index == length(ARGS) && error("missing value for $name")
    ARGS[index + 1]
end

"""Copy one legacy physical spectrum into its corresponding `cyax.h5` file."""
function migrate_file!(legacy_path, target_path)
    h5open(target_path, "r+") do target
        haskey(target, "spectrum/physical/m") && return false
        h5open(legacy_path, "r") do legacy
            spectrum = haskey(target, "spectrum") ? target["spectrum"] : create_group(target, "spectrum")
            physical = haskey(spectrum, "physical") ? spectrum["physical"] : create_group(spectrum, "physical")
            legacy_physical = legacy["cytools/spectrum/physical"]
            for dataset_name in keys(legacy_physical)
                physical[dataset_name] = read(legacy_physical[dataset_name])
            end
            metadata = haskey(physical, "metadata") ? physical["metadata"] : create_group(physical, "metadata")
            legacy_metadata = legacy["metadata"]
            for metadata_name in keys(legacy_metadata)
                metadata[metadata_name] = read(legacy_metadata[metadata_name])
            end
        end
        return true
    end
end

"""Migrate the selected legacy spectrum files and return a process exit code."""
function main()
    data_dir = CYAxiverse.filestructure.resolve_data_dir(
        option_value("--data-dir", ""))
    offset = parse(Int, option_value("--offset", "0"))
    limit = parse(Int, option_value("--limit", "0"))
    offset >= 0 || error("--offset must be nonnegative")
    limit >= 0 || error("--limit must be nonnegative")

    legacy_root = joinpath(data_dir, "physical_spectrum")
    legacy_paths = String[]
    for entry in walkdir(legacy_root)
        directory = entry[1]
        for file_name in entry[3]
            endswith(file_name, ".h5") && push!(legacy_paths, joinpath(directory, file_name))
        end
    end
    sort!(legacy_paths)
    last_index = limit == 0 ? length(legacy_paths) : min(offset + limit, length(legacy_paths))
    selected_paths = offset < last_index ? legacy_paths[(offset + 1):last_index] : String[]

    copied = 0
    skipped = 0
    failed = 0
    for legacy_path in selected_paths
        relative = splitpath(relpath(legacy_path, legacy_root))
        length(relative) == 3 || error("unexpected legacy path: $legacy_path")
        cy_dir = splitext(relative[3])[1]
        target_path = joinpath(data_dir, relative[1], relative[2], cy_dir, "cyax.h5")
        try
            if migrate_file!(legacy_path, target_path)
                copied += 1
            else
                skipped += 1
            end
        catch error_value
            failed += 1
            println(stderr, "FAILED $legacy_path: ", sprint(showerror, error_value))
        end
    end

    println("offset=$offset selected=$(length(selected_paths)) copied=$copied skipped=$skipped failed=$failed")
    failed == 0 || return 1
    return 0
end

exit(main())
