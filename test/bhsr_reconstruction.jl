using Test
include(joinpath(@__DIR__, "..", "scripts", "bhsr_regge_reconstruction.jl"))
include(joinpath(@__DIR__, "..", "scripts", "bhsr_likelihood_reconstruction.jl"))

@testset "CYAX-0121 BHSR analytic reconstruction" begin
    modes = bhsr_nodeless_modes()
    @test [mode.label for mode in modes] == ["|211>", "|322>", "|433>", "|544>", "|655>"]
    @test [bhsr_principal_number(mode) for mode in modes] == 2:6
    @test all(mode -> mode.n_r == 0 && mode.m == mode.l, modes)

    mode211 = only(filter(mode -> mode.label == "|211>", modes))
    alpha = bhsr_alpha(big"10", big"4.3e-12")
    @test isapprox(alpha, big"0.3217847"; rtol = big"1e-6")

    nmax = bhsr_nmax(big"10", mode211)
    @test isapprox(nmax, big"8.34731776078894725e76"; rtol = big"1e-15")
    @test isapprox(bhsr_nmax(big"10", modes[2]), nmax / 2; rtol = big"1e-70")

    @test bhsr_rate_s(big"10", big"4.3e-12", big"0.5", mode211) < 0
    @test bhsr_rate_s(big"10", big"4.3e-12", big"0.99", mode211) > 0
    @test_throws ArgumentError bhsr_rate_s(big"10", big"4.3e-12", big"1.01", mode211)

    critical = bhsr_critical_spin(big"10", big"4.3e-12", mode211, big"1e10")
    @test !ismissing(critical)
    @test big"0.8" < critical < big"1.0"
    @test bhsr_free_field_residual(big"10", big"4.3e-12", critical - big"1e-6",
                                  mode211, big"1e10") < 0
    @test bhsr_free_field_residual(big"10", big"4.3e-12", critical + big"1e-6",
                                  mode211, big"1e10") > 0
    @test_throws ErrorException bhsr_critical_spin(big"10", big"4.3e-12", mode211,
        big"1e10"; max_iterations = 1)

    all_mode_row = bhsr_regge_row(big"10", big"4.3e-12", big"1e10", modes)
    @test all_mode_row.union_spin < critical
    @test first(all_mode_row.per_mode).first == mode211.label
    @test isapprox(all_mode_row.union_spin,
                   big"0.408969756865561523197058083325"; rtol = big"1e-22")

    bose_negative = bhsr_bosenova_details(big"10", big"4.3e-12", mode211, big"-1e-73")
    bose_positive = bhsr_bosenova_details(big"10", big"4.3e-12", mode211, big"1e-73")
    @test bose_negative.route_identity == "REFERENCE_2021_BOSENOVA"
    @test bose_negative.lambda_iiii_signed < 0
    @test bose_negative.lambda_iiii_magnitude == bose_positive.lambda_iiii_magnitude
    @test bose_negative.f_pert_GeV == bose_positive.f_pert_GeV
    @test bose_negative.n_bose == bose_positive.n_bose
    @test bose_negative.omitted_interactions == ("off_diagonal_quartics", "cubic_interactions")
    @test BHSR_BOSENOVA_TIMESCALES_YEARS == ("1e10", "4.5e7", "4.5e6")
    @test !bhsr_bosenova_efficient(-big"1e-30")
    @test !bhsr_bosenova_efficient(big"0")
    @test bhsr_bosenova_efficient(big"1e-30")

    bose_critical = bhsr_bosenova_critical_spin(big"10", big"4.3e-12", mode211,
                                                 big"-1e-73", big"1e10")
    bose_below = bhsr_bosenova_residual(big"10", big"4.3e-12", bose_critical - big"1e-6",
                                        mode211, big"-1e-73", big"1e10")
    bose_above = bhsr_bosenova_residual(big"10", big"4.3e-12", bose_critical + big"1e-6",
                                        mode211, big"-1e-73", big"1e10")
    @test bose_below < 0 < bose_above
    @test !bhsr_bosenova_efficient(bose_below)
    @test bhsr_bosenova_efficient(bose_above)
    bose_row = bhsr_bosenova_regge_row(big"10", big"4.3e-12", big"-1e-73",
                                      big"1e10", modes)
    @test bose_row.route_identity == "REFERENCE_2021_BOSENOVA"
    @test bose_row.union_spin < last(bose_row.per_mode[1])
    @test isapprox(bose_row.union_spin,
                   big"0.408959402721713939807860399649"; rtol = big"1e-24")
    @test_throws ErrorException bhsr_bosenova_critical_spin(big"10", big"4.3e-12",
        mode211, big"-1e-73", big"1e10"; max_iterations = 1)
    @test ismissing(bhsr_bosenova_critical_spin(big"10", big"4.3e-12", mode211,
        big"-1e-73", big"1e-50"))

    grid = bhsr_regge_grid(big"4.3e-12", big"1e10";
                           mass_log10_min = -1, mass_log10_max = 1,
                           points = 3, modes = [mode211])
    @test length(grid) == 3
    @test grid[1].mass_solar == big"0.1"
    @test grid[2].mass_solar == big"1"
    @test grid[3].mass_solar == big"10"
    @test first(grid[3].per_mode).first == mode211.label
    @test grid[3].union_spin == last(grid[3].per_mode[1])

    @test isapprox(bhsr_standard_normal_cdf(big"1"),
                   big"0.841344746068542948585232545632"; rtol = big"1e-28")
    @test bhsr_contour_derivative(x -> x^3, big"2") ≈ big"12" atol = big"1e-25"
    @test bhsr_contour_derivative(x -> x^2, big"0"; lower = 0, upper = 1) ≈ 0 atol = big"1e-25"
    @test bhsr_projected_sigma_y(big"0.3", big"0.4", big"2") ≈ sqrt(big"0.52")
    @test bhsr_projected_sigma_x(big"0.3", big"0.4", big"2") ≈ sqrt(big"0.73")

    direct = bhsr_allowed_probability_direct(big"1", big"3", big"0.3", big"0.4",
                                               x -> 2x + 1)
    @test direct.projection == :y_of_x
    @test direct.status == :evaluated
    @test direct.probability_allowed == big"0.5"
    outside = bhsr_allowed_probability_direct(big"1.1", big"0", big"0.3", big"0.4",
                                               x -> 2x + 1; lower = 0, upper = 1)
    @test outside.status == :outside_contour_support
    @test ismissing(outside.probability_allowed)
    missing_stencil = bhsr_allowed_probability_direct(big"0.5", big"0", big"0.3", big"0.4",
        x -> x == big"0.5" ? big"0.2" : missing; lower = 0, upper = 1)
    @test missing_stencil.status == :incomplete_derivative_support
    @test ismissing(missing_stencil.probability_allowed)
    inverse = bhsr_allowed_probability_inverse(big"0.1", big"0.1", big"0.2",
                                                [big"0", big"2"], [big"1", big"-1"])
    @test inverse.nearest_branch == 1
    @test inverse.inverse_derivative == 1
    @test 0 < inverse.probability_allowed < 1
    @test_throws ArgumentError bhsr_allowed_probability_inverse(big"1", big"0.1", big"0.2",
        [big"0", big"2"], [big"1", big"-1"])
    @test_throws ArgumentError bhsr_allowed_probability_inverse(big"0", big"0.1", big"0.2",
        [big"0", big"0", big"2"], [big"1", big"1", big"-1"])
    @test_throws ArgumentError bhsr_allowed_probability_inverse(big"0", big"0.1", big"0.2",
        [big"0", missing], [big"1", big"-1"])

    cusp = bhsr_union_boundary_function([x -> big"0.2" + x, x -> big"0.2" - x])
    @test cusp(big"0") == big"0.2"
    @test bhsr_contour_derivative(cusp, big"0") ≈ 0 atol = big"1e-25"
    @test bhsr_union_boundary([missing, big"0.4", big"0.3"]) == big"0.3"

    tree = bhsr_probability_tree([[big"0.5", big"0.5"], [big"0.5"]])
    @test tree.single_axion_allowed == [big"0.25", big"0.5"]
    @test tree.geometry_allowed == big"0.125"
    @test tree.geometry_excluded == big"0.875"
    @test !tree.threshold_exceeded
    threshold_equal = bhsr_probability_tree([[big"0.0455"]])
    @test threshold_equal.geometry_excluded == big"0.9545"
    @test !threshold_equal.threshold_exceeded
    incomplete_tree = bhsr_probability_tree([[big"0.9", missing], [big"0.8"]])
    @test ismissing(incomplete_tree.geometry_allowed)
    @test ismissing(incomplete_tree.geometry_excluded)
    @test ismissing(incomplete_tree.threshold_exceeded)

    sigma_row = (name = "censored source row", sigma_mass = big"1",
                 sigma_spin = missing, sigma_mass_source = "source_defined",
                 sigma_spin_source = "reported_lower_bound")
    gate = bhsr_sigma_gate([sigma_row]; expected_rows = 1)
    @test gate.status == :unavailable
    @test gate.unresolved == ["censored source row"]
    sourcewide = bhsr_sourcewide_likelihood_status()
    @test sourcewide.status == :unavailable
    @test length(sourcewide.unresolved_censored_spin_rows) == 6
end
