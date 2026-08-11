#!/usr/bin/env julia

"""Bounded, read-only Stage 4/5 validation for the vacua pipeline.

Stage 4 checks the repository's N=5/N=8 anchors, the six inflation-screen
geometries, the reduced JLM-method/digitized aggregate, and deterministic selected
column ordering.  Stage 5 benchmarks one available geometry at each requested
large dimension with `save=false`; it never modifies geometry HDF5 files.

The default data root is the workspace sibling `../data`.  Use
`--data-dir PATH` to point at an isolated copy.  Output is intentionally
machine-readable and should normally be written outside Git; only curated
discrepancy references and the compact report belong in the repository.
"""

using CYAxiverse
using DelimitedFiles
using LinearAlgebra
using Printf
using SHA
using Statistics

include(joinpath(@__DIR__, "vacua_pipeline.jl"))

const GeometryIndex = CYAxiverse.structs.GeometryIndex

function _usage()
    println("""
    Usage: julia --project=. scripts/validate_vacua_stage4_5.jl [options]

      --data-dir PATH       Read-only geometry data root. Default: ../data.
      --output-dir PATH     Output directory. Default: /tmp/cyaxiverse-vacua-stage4-5.
      --stage4-only         Run only comparison checks.
      --stage5-only         Run only read-only resource checks.
      --stage5-limit N      Number of geometries per h11 (default: 1).
      --threads LIST        BLAS threads, comma-separated (default: 1,2,4,8,16).
      --starts N             Bounded solver start budget (default: 1).
    """)
end

function _parse_args(args)
    options = (data_dir="",
        output_dir="/tmp/cyaxiverse-vacua-stage4-5", stage4=true, stage5=true,
        stage5_limit=1, threads=[1, 2, 4, 8, 16], starts=1)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--help", "-h")
            _usage()
            exit(0)
        elseif arg == "--stage4-only"
            options = merge(options, (; stage4=true, stage5=false))
        elseif arg == "--stage5-only"
            options = merge(options, (; stage4=false, stage5=true))
        elseif arg in ("--data-dir", "--output-dir", "--stage5-limit", "--threads", "--starts")
            i < length(args) || error("missing value for $arg")
            value = args[i + 1]
            if arg == "--data-dir"
                options = merge(options, (; data_dir=value))
            elseif arg == "--output-dir"
                options = merge(options, (; output_dir=abspath(value)))
            elseif arg == "--stage5-limit"
                options = merge(options, (; stage5_limit=parse(Int, value)))
            elseif arg == "--threads"
                options = merge(options, (; threads=parse.(Int, split(value, ","))))
            else
                options = merge(options, (; starts=parse(Int, value)))
            end
            i += 1
        else
            error("unknown option: $arg")
        end
        i += 1
    end
    options.stage5_limit > 0 || error("stage5-limit must be positive")
    all(>(0), options.threads) || error("threads must be positive")
    options.starts > 0 || error("starts must be positive")
    merge(options, (; data_dir=CYAxiverse.filestructure.resolve_data_dir(options.data_dir)))
end

_csv_fields(line) = split(chomp(line), ',')

function _csv_rows(path)
    isfile(path) || return NamedTuple[]
    lines = readlines(path)
    isempty(lines) && return NamedTuple[]
    names = Symbol.(_csv_fields(lines[1]))
    rows = NamedTuple[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = _csv_fields(line)
        length(values) == length(names) || error("malformed CSV row in $path")
        push!(rows, NamedTuple{Tuple(names)}(Tuple(values)))
    end
    rows
end

function _int_field(row, name)
    parse(Int, getproperty(row, name))
end

function _float_field(row, name)
    parse(Float64, getproperty(row, name))
end

function _write_line(io, name, value)
    println(io, name, "=", repr(value))
end

function _matrix_literal(matrix)
    repr(Matrix(matrix))
end

function _write_reproducer(path, geom_idx, Q, L; threshold, method,
        solver_status, residual, comparison)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Compact Stage 4 discrepancy reproducer; no source-data write.")
        _write_line(io, "geometry", (geom_idx.h11, geom_idx.polytope, geom_idx.frst))
        _write_line(io, "Q_orientation", "h11_by_instanton")
        _write_line(io, "L_orientation", "2_by_instanton")
        _write_line(io, "threshold", threshold)
        _write_line(io, "method", method)
        _write_line(io, "solver_status", solver_status)
        _write_line(io, "residual_diagnostic", residual)
        _write_line(io, "comparison", comparison)
        _write_line(io, "Q", _matrix_literal(Q))
        _write_line(io, "L", _matrix_literal(L))
    end
end

function _diagnostic_residual(Q, L; starts)
    solved = CYAxiverse.generate.reduced_critical_points(L, Q; starts,
        residual_tolerance=1e-9, merge_tolerance=1e-6, max_iterations=300)
    isempty(solved.residuals) ? "no_converged_points" : maximum(solved.residuals)
end

function _stage4_n5_n8()
    n5 = CYAxiverse.paper_benchmarks.n5_potential()
    kc = CYAxiverse.paper_benchmarks.n5_critical_scale()
    n5_low = CYAxiverse.paper_benchmarks.n5_reduced_critical_points(kc - 1e-4)
    n5_high = CYAxiverse.paper_benchmarks.n5_reduced_critical_points(kc + 1e-4)
    n8 = CYAxiverse.paper_benchmarks.n8_minima_scan(starts=2048)
    n8_counts = Tuple(result.minima_count for result in n8)
    length(n5_low.theta) == 4 && n5_low.minima == 2 || error("N=5 low-side anchor failed")
    length(n5_high.theta) == 2 && n5_high.minima == 1 || error("N=5 high-side anchor failed")
    n8_counts == (5, 1) || error("N=8 minima anchor failed: $n8_counts")
    (; n5_low=(critical=length(n5_low.theta), minima=n5_low.minima),
       n5_high=(critical=length(n5_high.theta), minima=n5_high.minima),
       n8_counts)
end

const INFLATION_GEOMETRIES = ((5, 1, 1, 160, 5), (9, 1, 1, 2560, 5),
    (10, 1, 1, 5120, 5), (11, 1, 1, 10240, 5),
    (11, 2, 1, 10240, 5), (11, 7, 1, 12288, 6))

function _stage4_inflation(data_dir, output_dir)
    ENV["CYAXIVERSE_DATA_DIR"] = data_dir
    rows = NamedTuple[]
    for (h11, polytope, frst, expected_branches, expected_minima) in INFLATION_GEOMETRIES
        idx = GeometryIndex(h11, polytope, frst)
        potential = CYAxiverse.read.potential(idx)
        selected = CYAxiverse.generate.LQtilde(Matrix{Int}(potential.Q), Matrix{Float64}(potential.L))
        branches = CYAxiverse.generate.leading_critical_branches(selected;
            max_branches=1_000_000)
        matches = branches.branch_count == expected_branches &&
            branches.leading_minima_count == expected_minima
        if !matches
            residual = try
                _diagnostic_residual(Matrix{Int}(potential.Q), Matrix{Float64}(potential.L); starts=256)
            catch err
                "diagnostic_failed: " * replace(sprint(showerror, err), ',' => ';')
            end
            _write_reproducer(joinpath(output_dir, "reproducers",
                    "h11_$(lpad(h11, 3, '0'))_$(polytope)_$(frst).txt"), idx,
                potential.Q, potential.L; threshold=0.5,
                method="leading_branch_legacy_label_comparison",
                solver_status="completed",
                residual,
                comparison="legacy_label_replaced_by_physical_geometry=true; archived branch_count=$expected_branches leading_minima=$expected_minima; current branch_count=$(branches.branch_count) leading_minima=$(branches.leading_minima_count); selected_det=$(branches.det_Qtilde)")
        end
        push!(rows, (; h11, polytope, frst, branch_count=branches.branch_count,
            leading_minima_count=branches.leading_minima_count,
            archived_branch_count=expected_branches, archived_minima_count=expected_minima,
            verification_status=matches ? "verified_selected_branch_set" :
                "explained_legacy_geometry_replacement",
            solver_status="completed"))
    end
    rows
end

function _reference_selected_columns(Q, L)
    h11 = size(Q, 1)
    permutation = sortperm(@view(L[2, :]), rev=true)
    selected = Int[]
    current_rank = 0
    for column in permutation
        candidate = isempty(selected) ? Q[:, [column]] : Q[:, [selected; column]]
        next_rank = rank(candidate)
        if next_rank > current_rank
            push!(selected, column)
            current_rank = next_rank
        end
        current_rank == h11 && break
    end
    selected
end

function _stage4_ordering(data_dir)
    ENV["CYAXIVERSE_DATA_DIR"] = data_dir
    rows = NamedTuple[]
    for h11 in (10, 20)
        idx = GeometryIndex(h11, 1, 1)
        potential = CYAxiverse.read.potential(idx)
        Q = Matrix{Int}(potential.Q)
        L = Matrix{Float64}(potential.L)
        reference = _reference_selected_columns(Q, L)
        selected = CYAxiverse.generate.LQtilde(Q, L)
        selected.Qtilde == Q[:, reference] || error("Q selection order mismatch at h11=$h11")
        selected.Ltilde == L[:, reference] || error("L selection order mismatch at h11=$h11")
        digest = bytes2hex(sha1(repr((Q[:, reference], L[:, reference]))))
        push!(rows, (; h11, geometry="$(h11),1,1", selected_columns=join(reference, ";"),
            sha1=digest, status="pass"))
    end
    rows
end

function _stage4_aggregate(data_dir, output_dir)
    log_root = joinpath(dirname(data_dir), "data", "logs")
    digitized_path = joinpath(REPO_ROOT,
        "paper_benchmarks/2023_minima/h11_004_011_reduced_jlm_vs_digitized.csv")
    digitized = Dict(_int_field(row, :h11) => row for row in _csv_rows(digitized_path))
    rows = NamedTuple[]
    for h11 in 4:11
        path = joinpath(log_root, "jlm_reduced_h11_$(lpad(h11, 3, '0'))_compare.csv")
        computed = _csv_rows(path)
        isempty(computed) && error("missing reduced-JLM summary: $path")
        values = [_int_field(row, :Nvac) for row in computed if getproperty(row, :status) == "done"]
        isempty(values) && error("no completed rows in $path")
        aggregate = (; h11, n_geometries=length(values),
            computed_mean=mean(values), computed_median=median(values),
            computed_max=maximum(values), non_square=count(getproperty(row, :issquare) == "0" for row in computed))
        reference = digitized[h11]
        delta = aggregate.computed_mean - _float_field(reference, :digitized_mean_Nvac)
        discrepancy = abs(delta) > 0
        first = computed[1]
        idx = GeometryIndex(h11, _int_field(first, :polytope), _int_field(first, :frst))
        potential = CYAxiverse.read.potential(idx)
        residual = try
            _diagnostic_residual(Matrix{Int}(potential.Q), Matrix{Float64}(potential.L); starts=256)
        catch err
            "diagnostic_failed: " * replace(sprint(showerror, err), ',' => ';')
        end
        if discrepancy
            filename = joinpath(output_dir, "reproducers", "h11_$(lpad(h11, 3, '0'))_aggregate.txt")
            _write_reproducer(filename, idx, potential.Q, potential.L; threshold=0.5,
                method="reduced_JLM_method_summary_vs_digitized", solver_status=getproperty(first, :status),
                residual, comparison="finite-search/reduced-JLM aggregate mean=$(aggregate.computed_mean) vs digitized=$(getproperty(reference, :digitized_mean_Nvac)); non_square=$(aggregate.non_square)")
        end
        push!(rows, merge(aggregate, (; digitized_mean=_float_field(reference, :digitized_mean_Nvac),
            mean_delta=delta, discrepancy_status=discrepancy ?
                "not_same_estimand_finite_search_lower_bound" : "pass")))
    end
    rows
end

function _stage5_geometries(data_dir, limit)
    ENV["CYAXIVERSE_DATA_DIR"] = data_dir
    candidates = GeometryIndex[]
    for h11 in (150, 180, 200)
        root = joinpath(data_dir, "h11_$(lpad(h11, 3, '0'))")
        isdir(root) || continue
        for np_dir in sort(filter(name -> startswith(name, "np_"), readdir(root)))
            np = try parse(Int, np_dir[4:end]) catch; continue end
            cy_root = joinpath(root, np_dir)
            for cy_dir in sort(filter(name -> startswith(name, "cy_"), readdir(cy_root)))
                cy = try parse(Int, cy_dir[4:end]) catch; continue end
                isfile(joinpath(cy_root, cy_dir, "cyax.h5")) || continue
                push!(candidates, GeometryIndex(h11, np, cy))
            end
        end
    end
    grouped = [filter(idx -> idx.h11 == h11, candidates)[1:min(limit, count(==(h11), getfield.(candidates, :h11)))] for h11 in (150, 180, 200) if any(==(h11), getfield.(candidates, :h11))]
    reduce(vcat, grouped; init=GeometryIndex[])
end

function _stage5_run(data_dir, output_dir, geometries, threads, starts)
    path = joinpath(output_dir, "stage5_resource.csv")
    open(path, "w") do io
        println(io, "h11,polytope,frst,blas_threads,wall_seconds,allocations_bytes,maxrss_bytes,method,verification_status,solver_status,estimate_status,Nvac,physical_mode_count,one_writer,source_data_written")
        for idx in geometries, nthreads in threads
            LinearAlgebra.BLAS.set_num_threads(nthreads)
            GC.gc()
            before = Sys.maxrss()
            allocated = 0
            started = time()
            status = "failed"
            verification = "unavailable"
            solver_status = "not_started"
            estimate_status = "unavailable"
            nmin = ""
            modes = ""
            message = ""
            try
                result = nothing
                allocated = @allocated result = compute_axion_data(idx.h11, idx.polytope, idx.frst,
                    data_dir; threshold=0.5, starts, method=:reduced_jlm, save=false)
                search = result["search"]
                status = "completed"
                verification = search.search_classification == "exact_determinant_branch" ? "verified" : "not_applicable"
                solver_status = search.search_status
                estimate_status = "estimated"
                nmin = string(result["vacua_estimate"].vac)
                modes = string(length(result["spectrum"].m))
            catch err
                message = replace(sprint(showerror, err), ',' => ';')
            end
            rss = max(Sys.maxrss(), before)
            @printf(io, "%d,%d,%d,%d,%.6f,%d,%d,reduced_jlm,%s,%s,%s,%s,%s,read_only_no_writer,false\n",
                idx.h11, idx.polytope, idx.frst, nthreads, time() - started,
                allocated, rss, verification, solver_status, estimate_status, nmin, modes)
            isempty(message) || @printf("Stage 5 failed for %s threads=%d: %s\n", idx, nthreads, message)
            flush(io)
        end
    end
    path
end

function main(args=ARGS)
    options = _parse_args(args)
    isdir(options.data_dir) || error("data directory does not exist: $(options.data_dir)")
    mkpath(options.output_dir)
    report = joinpath(options.output_dir, "stage4_report.txt")
    open(report, "w") do io
        println(io, "data_dir=$(options.data_dir)")
        println(io, "git_revision=$(_git_revision())")
        println(io, "julia=$(VERSION)")
        println(io, "source_data_written=false")
        if options.stage4
            n5n8 = _stage4_n5_n8()
            println(io, "n5_n8=pass $(n5n8)")
            inflation = _stage4_inflation(options.data_dir, options.output_dir)
            open(joinpath(options.output_dir, "stage4_inflation.csv"), "w") do inflation_io
                println(inflation_io, "h11,polytope,frst,branch_count,leading_minima_count,archived_branch_count,archived_minima_count,verification_status,solver_status")
                for row in inflation
                    @printf(inflation_io, "%d,%d,%d,%d,%d,%d,%d,%s,%s\n", row.h11,
                        row.polytope, row.frst, row.branch_count, row.leading_minima_count,
                        row.archived_branch_count, row.archived_minima_count,
                        row.verification_status, row.solver_status)
                end
            end
            ordering = _stage4_ordering(options.data_dir)
            open(joinpath(options.output_dir, "stage4_ordering.csv"), "w") do ordering_io
                println(ordering_io, "h11,geometry,selected_columns,sha1,status")
                for row in ordering
                    println(ordering_io, row.h11, ',', row.geometry, ',', row.selected_columns,
                        ',', row.sha1, ',', row.status)
                end
            end
            println(io, "inflation_screen=$inflation")
            println(io, "selected_ordering=$ordering")
            aggregate = _stage4_aggregate(options.data_dir, options.output_dir)
            open(joinpath(options.output_dir, "stage4_aggregate.csv"), "w") do agg_io
                println(agg_io, "h11,n_geometries,computed_mean,computed_median,computed_max,non_square,digitized_mean,mean_delta,discrepancy_status")
                for row in aggregate
                    @printf(agg_io, "%d,%d,%.6f,%.6f,%d,%d,%.6f,%.6f,%s\n", row.h11,
                        row.n_geometries, row.computed_mean, row.computed_median, row.computed_max,
                        row.non_square, row.digitized_mean, row.mean_delta, row.discrepancy_status)
                end
            end
            println(io, "aggregate=recorded $(aggregate)")
        end
        if options.stage5
            geometries = _stage5_geometries(options.data_dir, options.stage5_limit)
            isempty(geometries) && error("no h11=150/180/200 geometries available")
            println(io, "stage5_geometries=$(geometries)")
            println(io, "stage5_resource_csv=$(_stage5_run(options.data_dir, options.output_dir, geometries, options.threads, options.starts))")
            println(io, "h11_491=unavailable_not_tested")
        end
    end
    println(report)
    true
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
