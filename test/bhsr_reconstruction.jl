using Test
include(joinpath(@__DIR__, "..", "scripts", "bhsr_regge_reconstruction.jl"))
include(joinpath(@__DIR__, "..", "scripts", "bhsr_likelihood_reconstruction.jl"))

function _bhsr_tree_fixture_row(axion_id, bh_id, probability, ensemble;
                                model_id = "geometry-fixture",
                                method_manifest_sha256 = BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
                                topology_method_addendum_sha256 = BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256,
                                source_mode_manifest_sha256 = BHSR_SOURCE_MODE_MANIFEST_SHA256,
                                source_backed = false,
                                status = :evaluated)
    return (; model_id, axion_id, bh_id,
            route_identity = BHSR_APPENDIX_B_ROUTE,
            method_manifest_sha256,
            topology_method_addendum_sha256,
            source_mode_manifest_sha256,
            bh_ensemble_id = ensemble.manifest_id,
            observational_manifest_sha256 = ensemble.observational_manifest_sha256,
            mode_labels = ["|211>"], source_backed, status,
            probability_allowed = probability)
end

@testset "CYAX-0121 continued-fraction validation" begin
    spec = bhsr_cf_method_spec()
    @test spec.route_identity == "CF_2018_VALIDATION"
    @test spec.alpha == big"0.1"
    @test spec.spin == big"0.9"
    @test spec.mode.label == "|211>"
    @test spec.continued_fraction_orders == [64, 128, 256]
    @test spec.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256
    @test bhsr_cf_angular_eigenvalue(1, 1, 0im; truncation = 9) == 2

    frozen = bhsr_cf_manifest_refinement()
    @test frozen.status == :unavailable_refinement_convergence
    @test frozen.refinement_changes.continued_fraction_order.gamma_relative_change > big"1e-8"
    @test frozen.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256
    target_root = only(filter(result -> result.continued_fraction_order == 256 &&
        result.angular_spheroidal_truncation == 17 && result.precision_bits == 256,
        frozen.results))
    @test real(target_root.q_bound_branch) < 0
    @test target_root.root_residual <= spec.root_residual_tolerance
    @test target_root.mode_label == "|211>"

    dolan_099 = bhsr_cf_dolan_table_benchmark(big"0.421", big"0.99";
        orders = [512, 1024, 2048], angular_truncations = [9, 17],
        precision_bits = 256, published_growth_M = big"1.50e-7",
        model_id = "DOLAN-TABLE-I-099")
    @test dolan_099.final_order_relative_change < big"1e-8"
    @test dolan_099.relative_table_difference < big"0.01"
    @test dolan_099.final_growth_M ≈ big"1.5043266535071376587323957061923448e-7" rtol = big"1e-12"
    @test all(result -> real(result.q_bound_branch) < 0, dolan_099.results)

    dolan_098 = bhsr_cf_dolan_table_benchmark(big"0.393", big"0.98";
        orders = [512, 1024, 2048], angular_truncations = [9, 17],
        precision_bits = 256, published_growth_M = big"1.11e-7",
        model_id = "DOLAN-TABLE-I-098")
    @test dolan_098.final_order_relative_change < big"1e-8"
    @test dolan_098.relative_table_difference < big"0.01"
    @test dolan_098.final_growth_M ≈ big"1.1122112979067492477014914411192176e-7" rtol = big"1e-12"
end

function _bhsr_complete_sigma_fixture(ensemble)
    return [(; name = bh_id,
             sigma_mass = big"1", sigma_spin = big"0.1",
             sigma_mass_source = "source_defined", sigma_spin_source = "source_defined",
             sigma_mass_confidence = "1sigma", sigma_spin_confidence = "1sigma",
             sigma_mass_provenance = "arXiv:1805.02016v2 Table I, fixture mass entry",
             sigma_spin_provenance = "arXiv:1805.02016v2 Table I, fixture spin entry")
            for bh_id in ensemble.bh_ids]
end

function _bhsr_contour_fixture(masses, spins, evaluator; model_id = "inverse-fixture",
                               active_modes = fill("|211>", length(masses)),
                               source_grid = false,
                               contour_topology_status = nothing,
                               topology_method_addendum_sha256 = BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256)
    mode = only(filter(item -> item.label == "|211>", bhsr_nodeless_modes()))
    rows = Any[]
    for i in eachindex(masses)
        row = (; model_id, mass_solar = _bhsr_big(masses[i]),
               union_spin = ismissing(spins[i]) ? missing : _bhsr_big(spins[i]),
               union_mode = ismissing(spins[i]) ? missing : active_modes[i],
               per_mode = ["|211>" => (ismissing(spins[i]) ? missing : _bhsr_big(spins[i]))])
        push!(rows, contour_topology_status === nothing ? row :
            merge(row, (; contour_topology_status)))
    end
    return BHSRContourGrid(model_id, "SYNTHETIC_FIXTURE",
        BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
        topology_method_addendum_sha256,
        BHSR_SOURCE_MODE_MANIFEST_SHA256,
        BigFloat.(_bhsr_big.(masses)),
        Union{Missing,BigFloat}[ismissing(spin) ? missing : _bhsr_big(spin) for spin in spins],
        rows, [mode], big"4.3e-12", big"1e10", big"0.1", 256, 140, big"1e-40",
        log10(_bhsr_big(first(masses))), log10(_bhsr_big(last(masses))), source_grid, evaluator)
end

@testset "CYAX-0121 BHSR analytic reconstruction" begin
    modes = bhsr_nodeless_modes()
    @test [mode.label for mode in modes] == ["|211>", "|322>", "|433>", "|544>", "|655>"]
    @test [bhsr_principal_number(mode) for mode in modes] == 2:6
    @test all(mode -> mode.n_r == 0 && mode.m == mode.l, modes)
    @test BHSR_SOURCE_MODE_MANIFEST_SHA256 ==
        bytes2hex(sha256(read(BHSR_SOURCE_MODE_MANIFEST)))
    @test BHSR_NUMERICAL_METHOD_MANIFEST_SHA256 ==
        bytes2hex(sha256(read(BHSR_NUMERICAL_METHOD_MANIFEST)))
    @test BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256 ==
        bytes2hex(sha256(read(BHSR_TOPOLOGY_METHOD_ADDENDUM)))
    @test_throws ArgumentError _bhsr_validate_modes([BHSRMode(0, 6, 6, "|766>")])

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
    low_alpha_free = bhsr_critical_spin_topology(big"1", big"4.3e-12", mode211,
        big"1e10"; model_id = "low-alpha-free-topology")
    low_alpha_free_mid = bhsr_free_field_residual(big"1", big"4.3e-12", big"0.5",
        mode211, big"1e10")
    low_alpha_free_extremal = bhsr_free_field_residual(big"1", big"4.3e-12", big"1",
        mode211, big"1e10")
    @test low_alpha_free_mid ≈ big"2.8136713513e7" rtol = big"1e-11"
    @test low_alpha_free_extremal ≈ big"-172.50682646" rtol = big"1e-11"
    @test length(low_alpha_free.roots) == 2
    @test low_alpha_free.intervals[1][1] ≈ big"0.1281682643098299286975054649" atol = big"1e-27"
    @test low_alpha_free.intervals[1][2] ≈ big"0.9999990663471144156736753704" atol = big"1e-27"
    @test low_alpha_free.topology_status == :bounded_efficiency_interval
    @test low_alpha_free.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256
    @test low_alpha_free.resolution_ladder == [257, 513, 1025]
    @test low_alpha_free.resolution_status == :stable
    @test bhsr_critical_spin(big"1", big"4.3e-12", mode211, big"1e10") ==
        first(low_alpha_free.roots)
    low_alpha_row = bhsr_regge_row(big"1", big"4.3e-12", big"1e10", [mode211];
        model_id = "low-alpha-free-row")
    @test low_alpha_row.union_spin == first(low_alpha_free.roots)
    @test low_alpha_row.union_intervals == low_alpha_free.intervals
    @test low_alpha_row.contour_topology_status == :bounded_efficiency_interval
    @test low_alpha_row.onset_only

    tangent_free_mass = big"0.21554620150829444416845426694618320734"
    tangent_free = bhsr_critical_spin_topology(tangent_free_mass, big"4.3e-12",
        mode211, big"1e10"; model_id = "near-tangent-free-regression")
    free_peak = only(tangent_free.refined_extrema)
    @test free_peak[1] ≈ big"0.5876767437668640383645201551" atol = big"1e-28"
    @test free_peak[2] ≈ big"1.345994205450331216276227928e-24" rtol = big"1e-16"
    @test tangent_free.roots[1] ≈ big"0.5876767437668230485381096634" atol = big"1e-28"
    @test tangent_free.roots[2] ≈ big"0.5876767437669050281909306458" atol = big"1e-28"
    @test tangent_free.topology_status == :bounded_efficiency_interval
    @test tangent_free.resolution_ladder == [257, 513, 1025]
    @test tangent_free.resolution_status == :stable
    @test tangent_free.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256

    near_zero_peak(x) = big"5e-41" - (x - big"0.5")^2
    near_zero_topology = _bhsr_isolate_positive_spin_intervals(near_zero_peak)
    @test near_zero_topology.status == :unavailable_near_tangent
    @test near_zero_topology.resolution_status == :near_tangent
    @test isempty(near_zero_topology.roots)
    @test isempty(near_zero_topology.intervals)
    @test length(near_zero_topology.candidate_roots) == 2
    unresolved_tangent_row = bhsr_regge_row(tangent_free_mass, big"4.3e-12",
        big"1e10", [mode211]; model_id = "near-tangent-row")
    @test unresolved_tangent_row.union_spin == tangent_free.roots[1]
    @test unresolved_tangent_row.contour_topology_status == :bounded_efficiency_interval

    narrow_pocket(x) = big"1.1" * exp(-((x - big"0.501") / big"0.0003")^2) - x
    unresolved_topology = _bhsr_isolate_positive_spin_intervals(narrow_pocket)
    @test unresolved_topology.status == :unavailable_topology_resolution
    @test unresolved_topology.resolution_status == :unstable
    @test isempty(unresolved_topology.intervals)
    @test length(unresolved_topology.candidate_intervals) == 1
    @test_throws ErrorException bhsr_critical_spin(big"10", big"4.3e-12", mode211,
        big"1e10"; max_iterations = 1)

    all_mode_row = bhsr_regge_row(big"10", big"4.3e-12", big"1e10", modes;
                                  model_id = "analytic-10-solar-mass")
    @test all_mode_row.model_id == "analytic-10-solar-mass"
    @test all_mode_row.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256
    @test all_mode_row.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256
    @test all_mode_row.source_mode_manifest_sha256 == BHSR_SOURCE_MODE_MANIFEST_SHA256
    @test all_mode_row.mode_labels == getfield.(modes, :label)
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
    low_alpha_bose = bhsr_bosenova_critical_spin_topology(big"1", big"4.3e-12",
        mode211, big"-1e-73", big"1e10"; model_id = "low-alpha-bose-topology")
    low_alpha_bose_mid = bhsr_bosenova_residual(big"1", big"4.3e-12", big"0.5",
        mode211, big"-1e-73", big"1e10")
    low_alpha_bose_extremal = bhsr_bosenova_residual(big"1", big"4.3e-12", big"1",
        mode211, big"-1e-73", big"1e10")
    @test low_alpha_bose_mid ≈ big"2.5238411056e10" rtol = big"1e-11"
    @test low_alpha_bose_extremal ≈ big"-173.56198036" rtol = big"1e-11"
    @test length(low_alpha_bose.roots) == 2
    @test low_alpha_bose.intervals[1][1] ≈ big"0.1281665047765220283893719384" atol = big"1e-27"
    @test low_alpha_bose.intervals[1][2] ≈ big"0.9999999989528558771827878312" atol = big"1e-27"
    @test low_alpha_bose.topology_status == :bounded_efficiency_interval
    tangent_bose_lambda = big"-1.7180957110518061656012412304276707960189280606045e-65"
    tangent_bose = bhsr_bosenova_critical_spin_topology(big"1", big"4.3e-12",
        mode211, tangent_bose_lambda, big"1e10";
        model_id = "near-tangent-bosenova-regression")
    bose_peak = only(tangent_bose.refined_extrema)
    @test bose_peak[1] ≈ big"0.6257692360535021388397666715" atol = big"1e-28"
    @test bose_peak[2] ≈ big"1.593503808487069909134855922e-27" rtol = big"1e-16"
    @test tangent_bose.roots[1] ≈ big"0.6257692360535008233366463567" atol = big"1e-28"
    @test tangent_bose.roots[2] ≈ big"0.6257692360535034543428869862" atol = big"1e-28"
    @test tangent_bose.topology_status == :bounded_efficiency_interval
    @test tangent_bose.resolution_ladder == [257, 513, 1025]
    @test tangent_bose.resolution_status == :stable
    @test tangent_bose.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256
    bose_row = bhsr_bosenova_regge_row(big"10", big"4.3e-12", big"-1e-73",
                                      big"1e10", modes; model_id = "bosenova-10-solar-mass")
    @test bose_row.model_id == "bosenova-10-solar-mass"
    @test bose_row.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256
    @test bose_row.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256
    @test bose_row.source_mode_manifest_sha256 == BHSR_SOURCE_MODE_MANIFEST_SHA256
    @test bose_row.route_identity == "REFERENCE_2021_BOSENOVA"
    @test bose_row.union_spin < last(bose_row.per_mode[1])
    @test isapprox(bose_row.union_spin,
                   big"0.319053261183905135751996850325"; rtol = big"1e-24")
    @test bose_row.contour_topology_status == :single_onset_to_extremal_spin
    @test length(bose_row.per_mode_topology[4].roots) == 2
    @test bose_row.per_mode_topology[4].topology_status == :bounded_efficiency_interval
    @test_throws ErrorException bhsr_bosenova_critical_spin(big"10", big"4.3e-12",
        mode211, big"-1e-73", big"1e10"; max_iterations = 1)
    @test ismissing(bhsr_bosenova_critical_spin(big"10", big"4.3e-12", mode211,
        big"-1e-73", big"1e-50"))

    grid = bhsr_regge_grid(big"4.3e-12", big"1e10";
                           mass_log10_min = -1, mass_log10_max = 1,
                           points = 3, modes = [mode211],
                           model_id = "three-point-regge-fixture")
    @test length(grid) == 3
    @test grid[1].mass_solar == big"0.1"
    @test grid[2].mass_solar == big"1"
    @test grid[3].mass_solar == big"10"
    @test first(grid[3].per_mode).first == mode211.label
    @test grid[3].union_spin == last(grid[3].per_mode[1])
    @test grid.model_id == "three-point-regge-fixture"
    @test grid.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256
    @test grid.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256
    @test grid.source_mode_manifest_sha256 == BHSR_SOURCE_MODE_MANIFEST_SHA256
    @test !grid.source_grid

    @test isapprox(bhsr_standard_normal_cdf(big"1"),
                   big"0.841344746068542948585232545632"; rtol = big"1e-28")
    @test bhsr_contour_derivative(x -> x^3, big"2") ≈ big"12" atol = big"1e-25"
    @test bhsr_contour_derivative(x -> x^2, big"0"; lower = 0, upper = 1) ≈ 0 atol = big"1e-25"
    @test bhsr_projected_sigma_y(big"0.3", big"0.4", big"2") ≈ sqrt(big"0.52")
    @test bhsr_projected_sigma_x(big"0.3", big"0.4", big"2") ≈ sqrt(big"0.73")

    ensemble = bhsr_bh_identity_ensemble()
    @test ensemble.route_identity == BHSR_APPENDIX_B_ROUTE
    @test length(ensemble.bh_ids) == BHSR_APPENDIX_B_EXPECTED_ROWS
    @test length(unique(ensemble.bh_ids)) == length(ensemble.bh_ids)
    @test ensemble.observational_manifest_sha256 == BHSR_OBSERVATIONAL_MANIFEST_SHA256
    @test_throws ArgumentError bhsr_bh_identity_ensemble("REFERENCE_2021_POPULATION")

    direct_curve = bhsr_union_boundary_function([x -> 2x + 1]; model_id = "direct-fixture")
    direct = bhsr_allowed_probability_direct(big"1", big"3", big"0.3", big"0.4",
        direct_curve; model_id = "direct-fixture", axion_id = "axion-test",
        bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test direct.projection == :y_of_x
    @test direct.status == :nonauthoritative_diagnostic
    @test direct.diagnostic_status == :evaluated
    @test direct.diagnostic_probability_allowed == big"0.5"
    @test ismissing(direct.probability_allowed)
    @test direct.authority_status == :unavailable_source_sigma_set
    @test direct.model_id == "direct-fixture"
    @test direct.axion_id == "axion-test"
    @test direct.bh_id == first(ensemble.bh_ids)
    @test direct.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256
    @test direct.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256
    @test direct.source_mode_manifest_sha256 == BHSR_SOURCE_MODE_MANIFEST_SHA256
    @test !direct.source_backed
    @test direct.contour_provenance_status == :caller_supplied_contour_functions
    outside = bhsr_allowed_probability_direct(big"1.1", big"0", big"0.3", big"0.4",
        direct_curve; lower = 0, upper = 1, model_id = "direct-fixture",
        axion_id = "axion-test", bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test outside.status == :outside_contour_support
    @test ismissing(outside.probability_allowed)
    missing_curve = bhsr_union_boundary_function(
        [x -> x == big"0.5" ? big"0.2" : missing]; model_id = "missing-fixture")
    missing_stencil = bhsr_allowed_probability_direct(big"0.5", big"0", big"0.3", big"0.4",
        missing_curve; lower = 0, upper = 1, model_id = "missing-fixture",
        axion_id = "axion-test", bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test missing_stencil.status == :incomplete_derivative_support
    @test ismissing(missing_stencil.probability_allowed)

    cusp = bhsr_union_boundary_function([x -> big"0.2" + x, x -> big"0.2" - x];
        model_id = "cusp-fixture")
    @test cusp(big"0") == big"0.2"
    @test cusp.model_id == "cusp-fixture"
    @test cusp.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256
    @test bhsr_contour_derivative(cusp, big"0") ≈ 0 atol = big"1e-25"
    @test bhsr_union_boundary([missing, big"0.4", big"0.3"]) == big"0.3"

    source_modes = getfield.(modes, :label)
    constant_curves = [x -> big"0.5" for _ in modes]
    labeled_constant = bhsr_union_boundary_function(constant_curves;
        model_id = "forged-source-closure", mode_labels = source_modes)
    @test labeled_constant.mode_labels == source_modes
    @test !labeled_constant.source_backed
    @test labeled_constant(big"1") == big"0.5"
    direct_forged_closure = bhsr_allowed_probability_direct(big"1", big"0.5",
        big"1", big"0.1", labeled_constant; lower = 0, upper = 2,
        model_id = "forged-source-closure", axion_id = "axion-211",
        bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test direct_forged_closure.status == :nonauthoritative_diagnostic
    @test direct_forged_closure.diagnostic_probability_allowed == big"0.5"
    @test ismissing(direct_forged_closure.probability_allowed)
    @test !direct_forged_closure.source_backed

    manually_forged_union = BHSRUnionContour(constant_curves, "forged-struct",
        BHSR_APPENDIX_B_ROUTE, BHSR_NUMERICAL_METHOD_MANIFEST_SHA256,
        BHSR_SOURCE_MODE_MANIFEST_SHA256, source_modes, true)
    forged_struct_result = bhsr_allowed_probability_direct(big"1", big"0.5",
        big"1", big"0.1", manually_forged_union; lower = 0, upper = 2,
        model_id = "forged-struct", axion_id = "axion-211",
        bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test manually_forged_union.source_backed
    @test !forged_struct_result.source_backed
    @test forged_struct_result.contour_provenance_status == :caller_supplied_contour_functions
    @test ismissing(forged_struct_result.probability_allowed)

    asserted_source_grid = _bhsr_contour_fixture([1, 2, 3], [1, 4, 9], x -> x^2;
        source_grid = true)
    asserted_grid_result = bhsr_allowed_probability_direct(big"2", big"4", big"0.1",
        big"0.2", asserted_source_grid; model_id = "inverse-fixture",
        axion_id = "axion-test", bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test asserted_source_grid.source_grid
    @test !asserted_grid_result.source_backed
    @test asserted_grid_result.contour_provenance_status ==
        :matching_recipe_without_source_target_data
    @test asserted_grid_result.contour_topology_status == :unverified_source_contour_topology
    @test asserted_grid_result.status == :ambiguous_source_spin_region_topology
    stale_topology_grid = _bhsr_contour_fixture([1, 2, 3], [1, 4, 9], x -> x^2;
        topology_method_addendum_sha256 = repeat("0", 64))
    @test_throws ArgumentError bhsr_allowed_probability_direct(big"2", big"4", big"0.1",
        big"0.2", stale_topology_grid; model_id = "inverse-fixture",
        axion_id = "axion-test", bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)

    bounded_source_formula_grid = _bhsr_contour_fixture([1, 2, 3], [1, 4, 9], x -> x^2;
        contour_topology_status = :bounded_efficiency_interval)
    bounded_direct = bhsr_allowed_probability_direct(big"2", big"4", big"0.1",
        big"0.2", bounded_source_formula_grid; model_id = "inverse-fixture",
        axion_id = "axion-test", bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test bounded_direct.status == :ambiguous_source_spin_region_topology
    @test ismissing(bounded_direct.diagnostic_probability_allowed)
    bounded_inverse = bhsr_allowed_probability_inverse(big"0.5", big"0.75", big"0.1",
        big"0.2", bounded_source_formula_grid; model_id = "inverse-fixture",
        axion_id = "axion-test", bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test bounded_inverse.status == :ambiguous_source_spin_region_topology
    @test ismissing(bounded_inverse.diagnostic_probability_allowed)
    tangent_source_formula_grid = _bhsr_contour_fixture([1, 2, 3], [1, 4, 9], x -> x^2;
        contour_topology_status = :unavailable_near_tangent)
    tangent_likelihood = bhsr_allowed_probability_direct(big"2", big"4", big"0.1",
        big"0.2", tangent_source_formula_grid; model_id = "inverse-fixture",
        axion_id = "axion-test", bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test tangent_likelihood.status == :unavailable_near_tangent
    @test ismissing(tangent_likelihood.diagnostic_probability_allowed)
    @test ismissing(asserted_grid_result.probability_allowed)

    boundary_grid = _bhsr_contour_fixture([1, 2, 3], [1, 4, 9], x -> x^2)
    boundary_likelihood = bhsr_allowed_probability_direct(big"1", big"1", big"0.1",
        big"0.2", boundary_grid; model_id = "inverse-fixture", axion_id = "axion-test",
        bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test boundary_likelihood.status == :nonauthoritative_diagnostic
    @test boundary_likelihood.diagnostic_status == :evaluated
    @test ismissing(boundary_likelihood.probability_allowed)
    @test boundary_likelihood.derivative ≈ 2 atol = big"1e-8"
    @test boundary_likelihood.contour_route_identity == "SYNTHETIC_FIXTURE"
    @test boundary_likelihood.contour_mass_support_solar == (big"1", big"3")
    @test boundary_likelihood.likelihood_route_identity == BHSR_APPENDIX_B_ROUTE

    two_root_grid = _bhsr_contour_fixture(
        [big"0.5", 1, big"1.5", 2, big"2.5", 3, big"3.5"],
        [missing, 0, big"0.75", 1, big"0.75", 0, missing],
        x -> 1 <= x <= 3 ? big"1" - (x - 2)^2 : missing)
    inverse = bhsr_allowed_probability_inverse(big"0.5", big"0.75", big"0.1", big"0.2",
        two_root_grid; model_id = "inverse-fixture", axion_id = "axion-test",
        bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test inverse.status == :nonauthoritative_diagnostic
    @test inverse.diagnostic_status == :evaluated
    @test inverse.inverse_roots ≈ [big"1.5", big"2.5"] atol = big"1e-11"
    @test inverse.nearest_branch == 1
    @test inverse.inverse_derivative ≈ 1 atol = big"1e-6"
    @test 0 < inverse.diagnostic_probability_disallowed_between_branches < 1
    @test 0 < inverse.diagnostic_probability_allowed < 1
    @test ismissing(inverse.probability_disallowed_between_branches)
    @test ismissing(inverse.probability_allowed)
    @test inverse.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256
    @test inverse.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256
    @test inverse.source_mode_manifest_sha256 == BHSR_SOURCE_MODE_MANIFEST_SHA256
    @test !inverse.source_backed
    tangent = bhsr_allowed_probability_inverse(big"0.5", big"1", big"0.1", big"0.2",
        two_root_grid; model_id = "inverse-fixture", axion_id = "axion-test",
        bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test tangent.status == :unsupported_single_inverse_branch
    @test ismissing(tangent.probability_allowed)
    no_roots = bhsr_allowed_probability_inverse(big"0.5", big"1.2", big"0.1", big"0.2",
        two_root_grid; model_id = "inverse-fixture", axion_id = "axion-test",
        bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test no_roots.status == :no_inverse_roots
    @test ismissing(no_roots.probability_allowed)
    @test_throws MethodError bhsr_allowed_probability_inverse(big"0.5", big"0.75",
        big"0.1", big"0.2", [big"1.5", big"2.5"])

    many_masses = [big"0.5", 1, big"1.25", big"1.5", big"1.75", 2,
                   big"2.25", big"2.5", big"2.75", 3, big"3.5"]
    many_spins = [missing, 1, big"0.5", 0, big"0.5", 1,
                  big"0.5", 0, big"0.5", 1, missing]
    many_grid = _bhsr_contour_fixture(many_masses, many_spins,
        x -> 1 <= x <= 3 ? big"0.5" + big"0.5" *
            cos(2 * big"3.141592653589793238462643383279502884" * (x - 1)) : missing)
    many_roots = bhsr_allowed_probability_inverse(big"0.5", big"0.5", big"0.1", big"0.2",
        many_grid; model_id = "inverse-fixture", axion_id = "axion-test",
        bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test many_roots.status == :unsupported_inverse_topology
    @test ismissing(many_roots.probability_allowed)
    gap_grid = _bhsr_contour_fixture([big"0.5", 1, big"1.5", 2, big"2.5", 3, big"3.5"],
        [missing, 0, big"0.5", missing, big"0.5", 0, missing], x -> missing)
    gap = bhsr_allowed_probability_inverse(big"0.5", big"0.25", big"0.1", big"0.2",
        gap_grid; model_id = "inverse-fixture", axion_id = "axion-test",
        bh_id = first(ensemble.bh_ids), bh_ensemble = ensemble)
    @test gap.status == :internal_contour_support_gap
    @test ismissing(gap.probability_allowed)

    axion_ids = ["axion-a", "axion-b"]
    complete_tree_rows = [_bhsr_tree_fixture_row(axion, bh, big"0.5", ensemble;
                          source_backed = true)
                          for axion in axion_ids for bh in ensemble.bh_ids]
    tree = bhsr_probability_tree(complete_tree_rows; model_id = "geometry-fixture",
        source_relevant_axion_ids = axion_ids, bh_ensemble = ensemble)
    @test tree.status == :unavailable
    @test tree.source_sigma_gate_status == :unavailable
    @test tree.authority_status == :unavailable_source_sigma_set
    @test ismissing(tree.single_axion_allowed[1])
    @test ismissing(tree.geometry_allowed)
    @test ismissing(tree.geometry_excluded)
    @test ismissing(tree.threshold_exceeded)
    @test ("<source-defined-Gaussian-sigma-set>",
           BHSR_APPENDIX_B_SOURCE_SIGMA_STATUS) in tree.unresolved_terms
    @test tree.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256
    @test tree.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256
    @test tree.source_mode_manifest_sha256 == BHSR_SOURCE_MODE_MANIFEST_SHA256
    @test_throws ArgumentError bhsr_probability_tree(complete_tree_rows;
        model_id = "geometry-fixture", source_relevant_axion_ids = String[], bh_ensemble = ensemble)
    @test_throws ArgumentError bhsr_probability_tree(complete_tree_rows;
        model_id = "geometry-fixture", source_relevant_axion_ids = ["axion-a", "axion-a"],
        bh_ensemble = ensemble)

    partial_tree = bhsr_probability_tree(complete_tree_rows[1:end-1];
        model_id = "geometry-fixture", source_relevant_axion_ids = axion_ids, bh_ensemble = ensemble)
    @test partial_tree.status == :unavailable
    @test length(partial_tree.missing_terms) == 1
    @test ismissing(partial_tree.geometry_allowed)
    duplicate_tree = bhsr_probability_tree(vcat(complete_tree_rows, [first(complete_tree_rows)]);
        model_id = "geometry-fixture", source_relevant_axion_ids = axion_ids, bh_ensemble = ensemble)
    @test duplicate_tree.status == :unavailable
    @test duplicate_tree.duplicate_terms == [("axion-a", first(ensemble.bh_ids))]
    @test ismissing(duplicate_tree.geometry_allowed)
    unresolved_rows = Any[complete_tree_rows...]
    unresolved_rows[1] = merge(first(unresolved_rows),
                               (; probability_allowed = missing, status = :unavailable))
    unresolved_tree = bhsr_probability_tree(unresolved_rows; model_id = "geometry-fixture",
        source_relevant_axion_ids = axion_ids, bh_ensemble = ensemble)
    @test unresolved_tree.status == :unavailable
    @test ("axion-a", first(ensemble.bh_ids)) in unresolved_tree.unresolved_terms
    wrong_manifest_rows = copy(complete_tree_rows)
    wrong_manifest_rows[1] = merge(first(wrong_manifest_rows),
        (; method_manifest_sha256 = repeat("0", 64)))
    wrong_manifest_tree = bhsr_probability_tree(wrong_manifest_rows; model_id = "geometry-fixture",
        source_relevant_axion_ids = axion_ids, bh_ensemble = ensemble)
    @test wrong_manifest_tree.status == :unavailable
    wrong_topology_rows = copy(complete_tree_rows)
    wrong_topology_rows[1] = merge(first(wrong_topology_rows),
        (; topology_method_addendum_sha256 = repeat("0", 64)))
    wrong_topology_tree = bhsr_probability_tree(wrong_topology_rows;
        model_id = "geometry-fixture", source_relevant_axion_ids = axion_ids,
        bh_ensemble = ensemble)
    @test wrong_topology_tree.status == :unavailable
    wrong_model_tree = bhsr_probability_tree(complete_tree_rows; model_id = "different-model",
        source_relevant_axion_ids = axion_ids, bh_ensemble = ensemble)
    @test wrong_model_tree.status == :unavailable
    unknown_bh_rows = copy(complete_tree_rows)
    unknown_bh_rows[1] = merge(first(unknown_bh_rows), (; bh_id = "unlisted-BH"))
    unknown_bh_tree = bhsr_probability_tree(unknown_bh_rows; model_id = "geometry-fixture",
        source_relevant_axion_ids = axion_ids, bh_ensemble = ensemble)
    @test unknown_bh_tree.status == :unavailable
    @test "unlisted-BH" in unknown_bh_tree.unexpected_bh_ids
    arbitrary_subset = bhsr_probability_tree([big"0.5"]; model_id = "geometry-fixture",
        source_relevant_axion_ids = axion_ids, bh_ensemble = ensemble)
    @test arbitrary_subset.status == :unavailable
    @test ismissing(arbitrary_subset.geometry_allowed)

    adversarial_axion_ids = ["axion-" * replace(label, "|" => "") for label in source_modes]
    adversarial_rows = NamedTuple[]
    for axion_id in adversarial_axion_ids, bh_id in ensemble.bh_ids
        diagnostic = bhsr_allowed_probability_direct(big"1", big"0.5", big"1", big"0.1",
            labeled_constant; lower = 0, upper = 2, model_id = "forged-source-closure",
            axion_id, bh_id, bh_ensemble = ensemble)
        # Simulate a caller copying invented sigmas and asserting the result is
        # source-backed/evaluated. The frozen source manifest must still block it.
        push!(adversarial_rows, merge(diagnostic,
            (; model_id = "forged-source-closure", source_backed = true,
               status = :evaluated,
               probability_allowed = diagnostic.diagnostic_probability_allowed)))
    end
    @test length(adversarial_rows) == 5 * 24
    invented_sigma_gate = bhsr_sigma_gate(_bhsr_complete_sigma_fixture(ensemble);
        bh_ensemble = ensemble)
    @test invented_sigma_gate.diagnostic_input_status == :complete
    @test invented_sigma_gate.status == :unavailable
    @test !invented_sigma_gate.source_authoritative
    @test invented_sigma_gate.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256
    forged_complete_tree = bhsr_probability_tree(adversarial_rows;
        model_id = "forged-source-closure", source_relevant_axion_ids = adversarial_axion_ids,
        bh_ensemble = ensemble)
    @test 1 - big"0.5"^(5 * 24) > big"0.9545"
    @test forged_complete_tree.status == :unavailable
    @test forged_complete_tree.authority_status == :unavailable_source_sigma_set
    @test all(ismissing, forged_complete_tree.single_axion_allowed)
    @test ismissing(forged_complete_tree.geometry_allowed)
    @test ismissing(forged_complete_tree.geometry_excluded)
    @test ismissing(forged_complete_tree.threshold_exceeded)

    threshold_rows = Any[_bhsr_tree_fixture_row("axion-one", bh, BigFloat(1), ensemble;
                            source_backed = true)
                      for bh in ensemble.bh_ids]
    threshold_rows[end] = _bhsr_tree_fixture_row("axion-one", last(ensemble.bh_ids),
                                                 big"0.0455", ensemble)
    threshold_equal = bhsr_probability_tree(threshold_rows; model_id = "geometry-fixture",
        source_relevant_axion_ids = ["axion-one"], bh_ensemble = ensemble)
    @test threshold_equal.status == :unavailable
    @test ismissing(threshold_equal.geometry_excluded)
    @test ismissing(threshold_equal.threshold_exceeded)

    sigma_rows = _bhsr_complete_sigma_fixture(ensemble)
    gate = bhsr_sigma_gate(sigma_rows; bh_ensemble = ensemble)
    @test gate.status == :unavailable
    @test gate.diagnostic_input_status == :complete
    @test gate.source_manifest_status == BHSR_APPENDIX_B_SOURCE_SIGMA_STATUS
    @test !gate.source_authoritative
    @test gate.expected_rows == 24
    @test length(gate.expected_bh_ids) == 24
    duplicate_sigma_rows = copy(sigma_rows)
    duplicate_sigma_rows[end] = merge(last(duplicate_sigma_rows), (; name = first(ensemble.bh_ids)))
    duplicate_gate = bhsr_sigma_gate(duplicate_sigma_rows; bh_ensemble = ensemble)
    @test duplicate_gate.status == :unavailable
    @test duplicate_gate.duplicate_ids == [first(ensemble.bh_ids)]
    @test last(ensemble.bh_ids) in duplicate_gate.missing_ids
    @test bhsr_sigma_gate(sigma_rows[1:end-1]; bh_ensemble = ensemble).status == :unavailable
    for invalid_sigma in (big"0", big"-1", BigFloat(NaN), missing)
        invalid_rows = Any[sigma_rows...]
        invalid_rows[1] = merge(first(invalid_rows), (; sigma_mass = invalid_sigma))
        @test bhsr_sigma_gate(invalid_rows; bh_ensemble = ensemble).status == :unavailable
    end
    for invalid_provenance in ((; sigma_mass_provenance = missing),
                               (; sigma_spin_confidence = "90%"),
                               (; sigma_spin_provenance = ""),
                               (; sigma_mass_source = "inferred"))
        invalid_rows = Any[sigma_rows...]
        invalid_rows[1] = merge(first(invalid_rows), invalid_provenance)
        @test bhsr_sigma_gate(invalid_rows; bh_ensemble = ensemble).status == :unavailable
    end
    sourcewide = bhsr_sourcewide_likelihood_status()
    @test sourcewide.status == :unavailable
    @test sourcewide.method_manifest_sha256 == BHSR_NUMERICAL_METHOD_MANIFEST_SHA256
    @test sourcewide.topology_method_addendum_sha256 == BHSR_TOPOLOGY_METHOD_ADDENDUM_SHA256
    @test sourcewide.source_mode_manifest_sha256 == BHSR_SOURCE_MODE_MANIFEST_SHA256
    @test length(sourcewide.unresolved_censored_spin_rows) == 6
    @test sourcewide.contour_mass_support_solar == (big"0.1", big"100")
    @test sourcewide.unsupported_contour_support_bh_ids == [
        "Mrk 335", "Fairall 9", "Mrk 79", "NGC 3783", "MCG-6-30-15",
        "NGC 7469", "Ark 120", "Mrk 110", "NGC 4051",
    ]
end
