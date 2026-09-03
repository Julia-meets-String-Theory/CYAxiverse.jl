### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 4f3cf4dd-0d6d-4f78-9648-739e6bd7ee15
begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", "notebooks"))
end

# ╔═╡ c6de49b0-685c-4a80-8ff4-5d8e8e7ce8ed
begin
    using CairoMakie
    using JSON
    using PlutoUI
    using Printf
    using Revise
    using Statistics
end

# ╔═╡ 4a50d8bb-22fb-4fa7-8aac-985526bf5bf4
html"""
<style>
main {
    margin: 0 auto;
    max-width: 95%;
    padding-left: max(80px, 6%);
    padding-right: max(80px, 6%);
}
</style>
"""

# ╔═╡ 0c363361-206c-45a4-aee0-55ea7d601d33
md"""
# Stage 1 and Stage 2 production statistics

Use this notebook to inspect the generated Stage-1 raw-FRST population and
the independent Stage-2 geometry/EFT run. The notebook reads only persisted
run artifacts. It does not recompute geometries or EFT rows.

The default paths come from `CYAXIVERSE_STAGE1_ROOT` and
`CYAXIVERSE_STAGE2_ROOT`. When those variables are unset, the notebook starts
with empty paths and reports that no run artifacts were found. Set the
variables or use the path fields below to inspect a run.
"""

# ╔═╡ 8d1e616b-0c6f-4d20-b4f4-9e614d1f7984
begin
    const DEFAULT_STAGE1_ROOT = get(ENV, "CYAXIVERSE_STAGE1_ROOT", "")
    const DEFAULT_STAGE2_ROOT = get(ENV, "CYAXIVERSE_STAGE2_ROOT", "")
end

# ╔═╡ 145bc6d0-58c2-4ea9-a39d-a7c76e7684d8
md"""
## Run selection

- Stage-1 root: $(@bind stage1_root PlutoUI.TextField(default=DEFAULT_STAGE1_ROOT))
- Stage-2 root: $(@bind stage2_root PlutoUI.TextField(default=DEFAULT_STAGE2_ROOT))

$(@bind reload_data PlutoUI.Button("Reload data"))
"""

# ╔═╡ c4dc28c8-a06a-4ff7-950d-0873ab60b7c0
begin
    stage1_root_value = ismissing(stage1_root) ? DEFAULT_STAGE1_ROOT : strip(stage1_root)
    stage2_root_value = ismissing(stage2_root) ? DEFAULT_STAGE2_ROOT : strip(stage2_root)
    reload_data_value = ismissing(reload_data) ? 0 : reload_data
end

# ╔═╡ f2f231cb-43d1-4d2f-956d-4dc52cdbcaef
begin
    """Return the first existing file in `root` from `names`."""
    function first_existing(root::AbstractString, names::AbstractString...)
        isempty(strip(root)) && return nothing
        for name in names
            path = joinpath(root, name)
            isfile(path) && return path
        end
        return nothing
    end

    """Read one JSON document, or return an empty dictionary when absent."""
    function read_json_or_empty(path)
        path === nothing && return Dict{String,Any}()
        isfile(path) || return Dict{String,Any}()
        return JSON.parse(read(path, String))
    end

    """Read a JSONL artifact into a vector of JSON dictionaries."""
    function read_jsonl(path)
        path === nothing && return Dict{String,Any}[]
        isfile(path) || return Dict{String,Any}[]
        records = Dict{String,Any}[]
        for (line_number, line) in enumerate(eachline(path))
            isempty(strip(line)) && continue
            try
                push!(records, JSON.parse(line))
            catch error
                throw(
                    ArgumentError(
                        "Could not parse $(path) at line $(line_number): $(error)",
                    ),
                )
            end
        end
        return records
    end

    """Return a nested JSON value or `default` when one key is absent."""
    function nested_get(value, keys...; default=nothing)
        current = value
        for key in keys
            current isa AbstractDict || return default
            current = get(current, string(key), default)
            current === default && return default
        end
        return current
    end

    """Convert a JSON scalar to a finite `Float64`, or return `nothing`."""
    function finite_float(value)
        value === nothing && return nothing
        value === missing && return nothing
        try
            number = Float64(value)
            return isfinite(number) ? number : nothing
        catch
            return nothing
        end
    end

    """Return a finite numeric column from JSON dictionaries."""
    function numeric_column(records, key::AbstractString)
        values = Float64[]
        for record in records
            number = finite_float(get(record, key, nothing))
            number === nothing || push!(values, number)
        end
        return values
    end

    """Return a stable string label for a JSON field."""
    function string_value(record, key::AbstractString; default="unknown")
        value = get(record, key, default)
        value === nothing && return default
        value === missing && return default
        return string(value)
    end

    """Return numeric `h11` values from JSON dictionaries."""
    function h11_values(records)
        values = Int[]
        for record in records
            value = finite_float(get(record, "h11", nothing))
            value === nothing || push!(values, round(Int, value))
        end
        return values
    end
end

# ╔═╡ 1dadfd63-8a43-44df-8d66-76ae3ec3dcf6
begin
    # PyArrow is used through a short-lived Python process because the
    # production writer stores EFT rows in Parquet. JSONL remains the primary
    # input, so the notebook still provides the run-level plots when PyArrow
    # is not installed in the selected Python environment.
    const PARQUET_READER = raw"""
import json
import sys

import pyarrow.parquet as parquet

path = sys.argv[1]
columns = sys.argv[2].split(",")
table = parquet.read_table(path, columns=columns)
payload = {column: table[column].to_pylist() for column in columns}
print(json.dumps(payload, allow_nan=True, separators=(",", ":")))
"""

    const EFT_COLUMNS = [
        "h11",
        "h21",
        "qcd_volume_scale",
        "qcd_volume",
        "qed_volume",
        "qed_log10_lambda4",
        "assignment_pool_size",
        "sampled_rank",
        "sampled_pool_rank",
        "qed_leading_status",
    ]

    """Find a Python executable that can read the EFT Parquet table."""
    function parquet_python()
        choices = String[]
        configured = get(ENV, "CYAXIVERSE_PYTHON", "")
        isempty(configured) || push!(choices, configured)
        push!(choices, "/opt/homebrew/Caskroom/miniforge/base/envs/cytools/bin/python")
        push!(choices, "python3")
        for choice in choices
            try
                success(`$choice --version`)
                return choice
            catch
                nothing
            end
        end
        return nothing
    end

    """Read selected Parquet columns as a vector of JSON dictionaries."""
    function read_eft_parquet(path)
        path === nothing && return (rows=Dict{String,Any}[], message="No EFT Parquet file was found.")
        python = parquet_python()
        python === nothing && return (
            rows=Dict{String,Any}[],
            message="No Python executable was found. Set CYAXIVERSE_PYTHON to the cytools Python executable.",
        )
        try
            output = read(`$python -c $PARQUET_READER $path $(join(EFT_COLUMNS, ","))`, String)
            payload = JSON.parse(output)
            row_count = isempty(EFT_COLUMNS) ? 0 : length(payload[EFT_COLUMNS[1]])
            rows = Vector{Dict{String,Any}}(undef, row_count)
            for row_index in 1:row_count
                row = Dict{String,Any}()
                for column in EFT_COLUMNS
                    row[column] = payload[column][row_index]
                end
                rows[row_index] = row
            end
            return (rows=rows, message="Loaded $(row_count) EFT rows with $(python).")
        catch error
            return (
                rows=Dict{String,Any}[],
                message="Could not read $(path) with PyArrow: $(sprint(showerror, error))",
            )
        end
    end
end

# ╔═╡ 5e04c9f3-bb7f-47af-b5c1-11099b24e1f3
begin
    """Load both run roots and the optional EFT model table."""
    function load_run_artifacts(stage1_root::AbstractString, stage2_root::AbstractString)
        stage1_manifest_path = first_existing(stage1_root, "run_manifest.json")
        stage1_status_path = first_existing(stage1_root, "frst_terminal_statuses.jsonl")
        stage2_manifest_path = first_existing(stage2_root, "run_manifest.json")
        stage2_status_path = first_existing(
            stage2_root,
            "stage2_terminal_statuses.jsonl",
            "stage2_terminal_statuses.partial.jsonl",
        )
        eft_path = first_existing(stage2_root, "eft_models.parquet")
        eft = read_eft_parquet(eft_path)
        return (
            stage1_manifest=read_json_or_empty(stage1_manifest_path),
            stage1_rows=read_jsonl(stage1_status_path),
            stage1_manifest_path=stage1_manifest_path,
            stage1_status_path=stage1_status_path,
            stage2_manifest=read_json_or_empty(stage2_manifest_path),
            stage2_rows=read_jsonl(stage2_status_path),
            stage2_manifest_path=stage2_manifest_path,
            stage2_status_path=stage2_status_path,
            eft_rows=eft.rows,
            eft_path=eft_path,
            eft_message=eft.message,
        )
    end

    # Keep the button as an explicit dependency for manual refresh in Pluto.
    reload_data_value
    artifacts = load_run_artifacts(stage1_root_value, stage2_root_value)
    stage1_manifest = artifacts.stage1_manifest
    stage1_rows = artifacts.stage1_rows
    stage2_manifest = artifacts.stage2_manifest
    stage2_rows = artifacts.stage2_rows
    eft_rows = artifacts.eft_rows
end

# ╔═╡ 4a87f161-81de-4ad6-a28a-fc1c57df7ec4
begin
    stage1_status = isempty(stage1_manifest) ? "not found" : string(get(stage1_manifest, "status", "unknown"))
    stage2_status = isempty(stage2_manifest) ? "not found" : string(get(stage2_manifest, "status", "unknown"))
    eft_dataset_status = string(
        nested_get(stage2_manifest, "eft", "allocation", "dataset_status"; default="not available"),
    )
    stage1_h11_counts = nested_get(stage1_manifest, "retained_raw_frst_count_by_h11"; default=Dict{String,Any}())
    stage2_accepted = nested_get(stage2_manifest, "accepted_geometry_count"; default="not available")
    eft_allocation = nested_get(stage2_manifest, "eft", "allocation"; default=Dict{String,Any}())
end

# ╔═╡ 9d74eae2-2567-4ed9-a94d-99b1857bb6b5
md"""
## Loaded run summary

| Artifact | Value |
|:--|:--|
| Stage-1 root | `$(stage1_root_value)` |
| Stage-1 manifest status | `$(stage1_status)` |
| Stage-1 terminal records | **$(length(stage1_rows))** |
| Stage-2 root | `$(stage2_root_value)` |
| Stage-2 manifest status | `$(stage2_status)` |
| Stage-2 terminal records | **$(length(stage2_rows))** |
| Stage-2 accepted geometries | **$(stage2_accepted)** |
| EFT table | `$(artifacts.eft_path === nothing ? "not found" : artifacts.eft_path)` |
| EFT dataset status | `$(eft_dataset_status)` |
| EFT rows loaded | **$(length(eft_rows))** |

`$(artifacts.eft_message)`
"""

# ╔═╡ 3e6f14c0-1d6d-47b8-a8bc-8c905f8cec6d
begin
    """Return sorted `h11` labels and status counts for grouped bars."""
    function status_count_table(records; status_key="terminal_status")
        h11s = sort(unique(h11_values(records)))
        statuses = sort(unique(string_value(record, status_key) for record in records))
        counts = zeros(Int, length(h11s), length(statuses))
        h11_index = Dict(value => index for (index, value) in enumerate(h11s))
        status_index = Dict(value => index for (index, value) in enumerate(statuses))
        for record in records
            h11 = finite_float(get(record, "h11", nothing))
            h11 === nothing && continue
            h11_key = round(Int, h11)
            status = string_value(record, status_key)
            counts[h11_index[h11_key], status_index[status]] += 1
        end
        return h11s, statuses, counts
    end

    """Draw grouped status counts by `h11`."""
    function plot_status_counts!(axis, records; title, status_key="terminal_status")
        h11s, statuses, counts = status_count_table(records; status_key=status_key)
        isempty(h11s) && begin
            text!(axis, 0.5, 0.5, text="No records", space=:relative, align=(:center, :center))
            axis.title = title
            return axis
        end
        positions = collect(1:length(h11s))
        bar_width = 0.8 / max(length(statuses), 1)
        colors = [:steelblue, :darkorange, :seagreen, :firebrick, :mediumpurple, :gray]
        for (status_index, status) in enumerate(statuses)
            offset = (status_index - (length(statuses) + 1) / 2) * bar_width
            barplot!(
                axis,
                positions .+ offset,
                counts[:, status_index];
                width=bar_width,
                color=colors[mod1(status_index, length(colors))],
                label=status,
            )
        end
        axis.xticks = (positions, string.(h11s))
        axis.xticklabelrotation = π / 4
        axis.title = title
        axislegend(axis; position=:rt, labelsize=8)
        return axis
    end

    """Draw a histogram or a marker for one numeric distribution."""
    function plot_histogram!(axis, values; title, xlabel, color=:steelblue, bins=30)
        finite_values = filter(isfinite, values)
        if isempty(finite_values)
            text!(axis, 0.5, 0.5, text="No numeric data", space=:relative, align=(:center, :center))
        elseif length(unique(finite_values)) == 1
            value = only(unique(finite_values))
            vlines!(axis, value; color=color, linewidth=3)
            text!(axis, 0.03, 0.94, text=@sprintf("value = %.5g", value), space=:relative)
        else
            hist!(axis, finite_values; bins=min(bins, max(5, round(Int, sqrt(length(finite_values))))), color=color, strokewidth=0)
        end
        axis.title = title
        axis.xlabel = xlabel
        axis.ylabel = "count"
        return axis
    end

    """Return EFT rows filtered by a selected `h11` label."""
    function filter_h11(records, selected)
        selected == "all" && return records
        selected_value = try
            parse(Int, selected)
        catch
            return records
        end
        return [
            record for record in records
            if begin
                value = finite_float(get(record, "h11", nothing))
                value !== nothing && round(Int, value) == selected_value
            end
        ]
    end
end

# ╔═╡ e9719362-0508-44d2-93c8-e9777c3c3156
begin
    fig = Figure(size=(1450, 980))
    ax11 = Axis(fig[1, 1], ylabel="records")
    plot_status_counts!(ax11, stage1_rows; title="Stage 1 terminal status by h11")

    ax12 = Axis(fig[1, 2])
    plot_histogram!(
        ax12,
        log10.(numeric_column(stage1_rows, "file_size_bytes"));
        title="Stage 1 raw-FRST artifact size",
        xlabel="log10(file size / bytes)",
        color=:steelblue,
    )

    ax21 = Axis(fig[2, 1], ylabel="records")
    plot_status_counts!(ax21, stage2_rows; title="Stage 2 geometry status by h11")

    ax22 = Axis(fig[2, 2])
    plot_histogram!(
        ax22,
        numeric_column(stage2_rows, "stage2_elapsed_seconds");
        title="Stage 2 geometry run time",
        xlabel="seconds per input geometry",
        color=:darkorange,
    )

    ax31 = Axis(fig[3, 1])
    plot_histogram!(
        ax31,
        numeric_column(eft_rows, "qed_volume");
        title="EFT QED divisor volume",
        xlabel="final QED divisor volume",
        color=:seagreen,
    )

    ax32 = Axis(fig[3, 2])
    plot_histogram!(
        ax32,
        numeric_column(eft_rows, "qed_log10_lambda4");
        title="EFT QED potential scale",
        xlabel="log10(Λ⁴)",
        color=:mediumpurple,
    )
    fig
end

# ╔═╡ 7cc94a5a-3026-4ff5-a16c-d6f6a2674b47
md"""
The Stage-1 artifact-size plot is a storage diagnostic. The Stage-2 time plot
measures the full per-geometry Stage-2 path, including reconstruction,
Kähler-point construction, potential references, and HDF5 writing. The EFT
plots use accepted rows from `eft_models.parquet` only.
"""

# ╔═╡ 1d54c41f-d62c-4f9d-97a6-c8cfd87e92d3
begin
    distribution_options = [
        "QED volume",
        "QCD volume scale",
        "log10 Lambda^4",
        "assignment-pool size",
        "sampled rank",
    ]
    h11_options = ["all"; string.(sort(unique(h11_values(eft_rows))))]
    @bind selected_distribution PlutoUI.Select(distribution_options; default=first(distribution_options))
    @bind selected_h11 PlutoUI.Select(h11_options; default=first(h11_options))
end

# ╔═╡ 6a3bd7d1-a062-4dde-9e5e-5d5c825691d4
begin
    selected_distribution_value = ismissing(selected_distribution) ? first(distribution_options) : selected_distribution
    selected_h11_value = ismissing(selected_h11) ? first(h11_options) : selected_h11
    distribution_keys = Dict(
        "QED volume" => ("qed_volume", "final QED divisor volume"),
        "QCD volume scale" => ("qcd_volume_scale", "homogeneous QCD radial scale"),
        "log10 Lambda^4" => ("qed_log10_lambda4", "log10(Λ⁴)"),
        "assignment-pool size" => ("assignment_pool_size", "assignment-pool entries"),
        "sampled rank" => ("sampled_rank", "sampled rank"),
    )
    selected_key, selected_label = distribution_keys[selected_distribution_value]
    selected_rows = filter_h11(eft_rows, selected_h11_value)
    selected_values = numeric_column(selected_rows, selected_key)
end

# ╔═╡ 8de48c8a-784e-4988-8a96-72b1ee8b7ff7
begin
    selected_figure = Figure(size=(900, 520))
    selected_axis = Axis(
        selected_figure[1, 1],
        title="EFT distribution: $(selected_distribution_value), h11=$(selected_h11_value)",
    )
    plot_histogram!(
        selected_axis,
        selected_values;
        title="$(selected_distribution_value) ($(length(selected_values)) rows)",
        xlabel=selected_label,
        color=:teal,
        bins=40,
    )
    selected_figure
end

# ╔═╡ b2c2dc3c-f5a1-45e5-9c18-8f77b1a46e79
md"""
## EFT allocation accounting

- Requested minimum rows: **$(nested_get(eft_allocation, "minimum_rows"; default="not available"))**
- Requested maximum rows: **$(nested_get(eft_allocation, "maximum_rows"; default="not available"))**
- Accepted rows: **$(nested_get(eft_allocation, "accepted_count"; default=length(eft_rows)) )**
- Validated assignment capacity: **$(nested_get(eft_allocation, "validated_assignment_capacity"; default="not available"))**
- Minimum shortfall: **$(nested_get(eft_allocation, "minimum_shortfall"; default="not available"))**

The EFT table is an adapted finite model table. A `diagnostic_partial` status
means that the run completed its accounting but did not reach its configured
row target.
"""

# ╔═╡ Cell order:
# ╠═4f3cf4dd-0d6d-4f78-9648-739e6bd7ee15
# ╠═c6de49b0-685c-4a80-8ff4-5d8e8e7ce8ed
# ╠═4a50d8bb-22fb-4fa7-8aac-985526bf5bf4
# ╟─0c363361-206c-45a4-aee0-55ea7d601d33
# ╠═8d1e616b-0c6f-4d20-b4f4-9e614d1f7984
# ╟─145bc6d0-58c2-4ea9-a39d-a7c76e7684d8
# ╠═c4dc28c8-a06a-4ff7-950d-0873ab60b7c0
# ╠═f2f231cb-43d1-4d2f-956d-4dc52cdbcaef
# ╠═1dadfd63-8a43-44df-8d66-76ae3ec3dcf6
# ╠═5e04c9f3-bb7f-47af-b5c1-11099b24e1f3
# ╠═4a87f161-81de-4ad6-a28a-fc1c57df7ec4
# ╟─9d74eae2-2567-4ed9-a94d-99b1857bb6b5
# ╠═3e6f14c0-1d6d-47b8-a8bc-8c905f8cec6d
# ╠═e9719362-0508-44d2-93c8-e9777c3c3156
# ╟─7cc94a5a-3026-4ff5-a16c-d6f6a2674b47
# ╠═1d54c41f-d62c-4f9d-97a6-c8cfd87e92d3
# ╠═6a3bd7d1-a062-4dde-9e5e-5d5c825691d4
# ╠═8de48c8a-784e-4988-8a96-72b1ee8b7ff7
# ╟─b2c2dc3c-f5a1-45e5-9c18-8f77b1a46e79
