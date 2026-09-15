#!/usr/bin/env julia

"""
P0 environment identity and package-load measurement harness.

Modes:
  metadata  Print privacy-safe Julia/runtime/BLAS/thread identity.
  load      Import CYAxiverse once and print elapsed/allocated bytes.
  cold      Instantiate and precompile the active project, then import once.

The caller controls the active project and JULIA_DEPOT_PATH.  In cold mode the
caller must provide P0_BASE_MANIFEST; it is copied to a temporary project and
the CYAxiverse path entry is redirected to this checkout.  Temporary files are
not part of the repository evidence.
"""

using Dates
using Random
using SHA
using LinearAlgebra
using Libdl
using Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const MODE = isempty(ARGS) ? "metadata" : ARGS[1]
const REFERENCE_SHA = "7a40285bb5c313f7e8746b90644d5f45bb67be44"

function safe_project()
    p = Base.active_project()
    return p === nothing ? "none" : basename(dirname(p)) * "/" * basename(p)
end

function print_identity()
    println("identity_timestamp_utc=", Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sss\Z"))
    println("julia_version=", VERSION)
    println("julia_build_commit=", Base.GIT_VERSION_INFO.commit)
    println("julia_build_branch=", Base.GIT_VERSION_INFO.branch)
    println("julia_build_number=", Base.GIT_VERSION_INFO.build_number)
    println("julia_build_date=", Base.GIT_VERSION_INFO.date_string)
    println("julia_build_tagged=", Base.GIT_VERSION_INFO.tagged_commit)
    println("julia_sysimage=", basename(unsafe_string(Base.JLOptions().image_file)))
    println("julia_build=", basename(unsafe_string(Base.JLOptions().julia_bin)))
    println("word_size=", Sys.WORD_SIZE)
    println("architecture=", Sys.MACHINE)
    println("os=", Sys.KERNEL, " ", Sys.ARCH)
    println("cpu_name=", Sys.CPU_NAME)
    println("cpu_threads=", Sys.CPU_THREADS)
    println("julia_threads=", Threads.nthreads())
    println("julia_project=", safe_project())
    println("blas_vendor=", BLAS.vendor())
    println("blas_config=", BLAS.get_config())
    println("blas_threads=", BLAS.get_num_threads())
    println("libblastrampoline=", basename(Libdl.dlpath(LinearAlgebra.libblastrampoline)))
    for name in ("JULIA_NUM_THREADS", "JULIA_THREAD_SLEEP_THRESHOLD", "JULIA_CPU_TARGET",
                 "JULIA_DEPOT_PATH", "JULIA_PROJECT", "JULIA_PKG_PRECOMPILE_AUTO",
                 "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLAS_NUM_THREADS", "LANG", "LC_ALL", "TZ")
        value = get(ENV, name, "<unset>")
        if name == "JULIA_DEPOT_PATH"
            value = "<temporary-depot-configured>"
        end
        println("env_", name, "=", value)
    end
    println("rng_algorithm=Random.MersenneTwister")
    println("rng_seed=0x5eed")
    println("rng_probe=", rand(MersenneTwister(0x5eed), UInt64))
    println("repository_sha=", strip(read(`git -C $REPO_ROOT rev-parse HEAD`, String)))
    println("repository_ref=", strip(read(`git -C $REPO_ROOT symbolic-ref --short HEAD`, String)))
    println("reference_sha=", REFERENCE_SHA)
    println("reference_source_match=", success(`git -C $REPO_ROOT diff --quiet $REFERENCE_SHA HEAD -- Project.toml src`))
    println("project_sha256=", bytes2hex(sha256(read(joinpath(REPO_ROOT, "Project.toml")))))
    manifest = get(ENV, "P0_BASE_MANIFEST", "")
    if !isempty(manifest) && isfile(manifest)
        println("resolved_manifest_sha256=", bytes2hex(sha256(read(manifest))))
        println("resolved_manifest_basename=", basename(manifest))
        println("resolved_manifest_julia_version=", get(Pkg.TOML.parsefile(manifest), "julia_version", "<missing>"))
        println("resolved_manifest_format=", get(Pkg.TOML.parsefile(manifest), "manifest_format", "<missing>"))
        println("resolved_manifest_project_hash=", get(Pkg.TOML.parsefile(manifest), "project_hash", "<missing>"))
    else
        println("resolved_manifest=unavailable")
    end
end

function import_once()
    GC.gc()
    t0 = time_ns()
    allocated = @allocated begin
        @eval Main using CYAxiverse
    end
    elapsed = time_ns() - t0
    println("package_load_elapsed_ns=", elapsed)
    println("package_load_elapsed_s=", elapsed / 1e9)
    println("package_load_allocated_bytes=", allocated)
    mod = only([m for m in values(Base.loaded_modules) if nameof(m) == :CYAxiverse])
    println("loaded_module_path=", basename(pathof(mod)))
    println("loaded_package_version=", Base.pkgversion(mod))
end

function cold_import()
    println("cold_active_project_before=", safe_project())
    t0 = time_ns()
    Pkg.instantiate(; verbose=false)
    t1 = time_ns()
    Pkg.precompile(; strict=true)
    t2 = time_ns()
    println("cold_instantiate_elapsed_ns=", t1 - t0)
    println("cold_instantiate_elapsed_s=", (t1 - t0) / 1e9)
    println("cold_precompile_elapsed_ns=", t2 - t1)
    println("cold_precompile_elapsed_s=", (t2 - t1) / 1e9)
    import_once()
    println("cold_total_environment_build_and_load_elapsed_s=", (time_ns() - t0) / 1e9)
end

print_identity()
if MODE == "load"
    import_once()
elseif MODE == "cold"
    cold_import()
elseif MODE == "metadata"
    nothing
else
    error("unknown mode: ", MODE)
end
