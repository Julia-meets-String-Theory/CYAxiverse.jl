# Environment predicates shared by the test files. A registry install of the
# package has no .git directory, and CI images may lack git or python3.11+.
using Test

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))

"""True only when `root` is itself the top level of a Git work tree."""
function in_git_checkout(root::AbstractString=REPOSITORY_ROOT)
    Sys.which("git") === nothing && return false
    toplevel = try
        readchomp(pipeline(Cmd(["git", "-C", root, "rev-parse", "--show-toplevel"]);
            stderr=devnull))
    catch
        return false
    end
    return realpath(toplevel) == realpath(root)
end

const SYNTHETIC_COMMIT = repeat("1", 40)
const SYNTHETIC_TREE = repeat("2", 40)

"""
Candidate identity for tests that only need a well-formed commit/tree pair.
Outside a Git checkout the fixed synthetic pair is returned and no git process
is started; tests that depend on real repository state use `@requires_git`.
"""
function head_identity(root::AbstractString=REPOSITORY_ROOT)
    in_git_checkout(root) || return (commit=SYNTHETIC_COMMIT, tree=SYNTHETIC_TREE)
    return (commit=strip(read(Cmd(["git", "-C", root, "rev-parse", "HEAD"]), String)),
        tree=strip(read(Cmd(["git", "-C", root, "rev-parse", "HEAD^{tree}"]), String)))
end

macro requires_git(testset)
    quote
        if in_git_checkout()
            $(esc(testset))
        else
            @info "skipping: not inside a Git checkout"
        end
    end
end

"""Python 3.11 added `tomllib`, which the skills checker relies on."""
function python_at_least_3_11()
    python = something(Sys.which("python3.13"), Sys.which("python3"), Some(nothing))
    python === nothing && return false
    version = try
        readchomp(Cmd([python, "-c", "import sys; print(sys.version_info >= (3, 11))"]))
    catch
        return false
    end
    return version == "True"
end
