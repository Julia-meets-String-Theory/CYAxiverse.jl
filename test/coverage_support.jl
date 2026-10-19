"""
Return the extra `julia` flags a test subprocess needs so that its coverage
output stays outside the checkout.

When the test process runs with coverage, `Base.julia_cmd()` hands the same
setting to every child. Julia then writes per-source `*.cov` files next to the
sources at child exit, which dirties the working tree that the scientific
qualification test requires to be clean. Redirect the child's LCOV output to a
temporary `.info` file instead. Return no flags when coverage is off or the
parent already redirects its output.
"""
function child_coverage_args()
    options = Base.JLOptions()
    if options.code_coverage != 0 && options.output_code_coverage == C_NULL
        return ["--code-coverage=$(tempname()).info"]
    end
    String[]
end
