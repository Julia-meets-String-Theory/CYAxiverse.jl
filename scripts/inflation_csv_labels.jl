"""Per-row kinetic-normalization labels of the inflation CSV families (CYAX-0232).

Implements the R-007 reader rules with Clarification A3: a row of a current
layout carries a `kinetic_normalization` column and the current version
values; a legacy row (no label column) is inferred as `legacy_c_half` only if
it matches exactly one recorded entry (header, `row_type`, version values)
written by merged physical code. Every other row raises an `ArgumentError`
naming the file, the row and the values.
"""

if !isdefined(@__MODULE__, :INFLATION_CSV_LEGACY_HEADERS)

# Generated from the historical writer sources at the commits recorded in
# specs/0232-kinetic-normalization/layouts/inflation-csv-entries.md
# (CYAX-0232 T041). Do not edit by hand.
const INFLATION_CSV_LEGACY_HEADERS = Dict{String, Vector{String}}(
    "PREP_A" => [
        "contract_version", "data_dir", "max_branches", "h11", "polytope", "frst",
        "status", "error", "instantons", "selected_instantons", "qtilde_det",
        "leading_log_gap", "log_scale_span", "strong_hierarchy", "branch_count",
        "leading_minima_count", "saddle_count", "candidate_slowroll_saddles",
        "least_tachyonic_min_eta", "least_tachyonic_epsilon", "best_min_eta",
        "best_epsilon", "best_abs_min_eta", "mass_min", "mass_max",
        "negative_mass_count", "stage_load_s", "stage_select_s", "stage_hierarchy_s",
        "stage_factor_s", "stage_mass_basis_s", "stage_branches_s", "stage_classify_s",
        "stage_allocated_bytes", "total_seconds"
    ],
    "PREP_B" => [
        "contract_version", "diagnostic_schema_version", "measurement_scope",
        "data_dir", "max_branches", "h11", "polytope", "frst", "status", "error",
        "instantons", "selected_instantons", "qtilde_det", "leading_log_gap",
        "log_scale_span", "strong_hierarchy", "branch_count", "leading_minima_count",
        "saddle_count", "candidate_slowroll_saddles", "least_tachyonic_min_eta",
        "least_tachyonic_epsilon", "best_min_eta", "best_epsilon", "best_abs_min_eta",
        "mass_min", "mass_max", "negative_mass_count", "stage_load_s",
        "stage_select_s", "stage_hierarchy_s", "stage_factor_s", "stage_mass_basis_s",
        "stage_branches_s", "stage_classify_s", "stage_load_output_bytes",
        "stage_select_output_bytes", "stage_hierarchy_output_bytes",
        "stage_factor_output_bytes", "stage_mass_basis_output_bytes",
        "stage_classify_output_bytes", "stage_allocated_bytes", "stage_output_bytes",
        "total_seconds"
    ],
    "PREP_C" => [
        "contract_version", "diagnostic_schema_version", "measurement_scope",
        "data_dir", "max_branches", "h11", "polytope", "frst", "status", "error",
        "instantons", "selected_instantons", "qtilde_det", "negative_mode_range",
        "structured_charge_validated", "structured_fallback_reason",
        "search_classification", "mask_count", "masks_visited", "masks_skipped",
        "lattice_copy_count", "lattice_copies_visited", "leading_log_gap",
        "log_scale_span", "strong_hierarchy", "branch_count", "leading_minima_count",
        "saddle_count", "candidate_slowroll_saddles", "least_tachyonic_min_eta",
        "least_tachyonic_epsilon", "best_min_eta", "best_epsilon", "best_abs_min_eta",
        "mass_min", "mass_max", "negative_mass_count", "stage_load_s",
        "stage_select_s", "stage_hierarchy_s", "stage_factor_s", "stage_mass_basis_s",
        "stage_branches_s", "stage_classify_s", "stage_load_output_bytes",
        "stage_select_output_bytes", "stage_hierarchy_output_bytes",
        "stage_factor_output_bytes", "stage_mass_basis_output_bytes",
        "stage_classify_output_bytes", "stage_allocated_bytes", "stage_output_bytes",
        "total_seconds"
    ],
    "SHARD_1" => [
        "shard_schema_version", "run_id", "shard_index", "shard_count", "attempt",
        "started_unix_s", "finished_unix_s", "contract_version",
        "diagnostic_schema_version", "measurement_scope", "data_dir", "max_branches",
        "h11", "polytope", "frst", "status", "error", "instantons",
        "selected_instantons", "qtilde_det", "leading_log_gap", "log_scale_span",
        "strong_hierarchy", "branch_count", "leading_minima_count", "saddle_count",
        "candidate_slowroll_saddles", "least_tachyonic_min_eta",
        "least_tachyonic_epsilon", "best_min_eta", "best_epsilon", "best_abs_min_eta",
        "mass_min", "mass_max", "negative_mass_count", "stage_load_s",
        "stage_select_s", "stage_hierarchy_s", "stage_factor_s", "stage_mass_basis_s",
        "stage_branches_s", "stage_classify_s", "stage_load_output_bytes",
        "stage_select_output_bytes", "stage_hierarchy_output_bytes",
        "stage_factor_output_bytes", "stage_mass_basis_output_bytes",
        "stage_classify_output_bytes", "stage_allocated_bytes", "stage_output_bytes",
        "total_seconds"
    ],
    "SHARD_2" => [
        "shard_schema_version", "run_id", "shard_index", "shard_count", "attempt",
        "started_unix_s", "finished_unix_s", "contract_version",
        "diagnostic_schema_version", "measurement_scope", "data_dir", "max_branches",
        "h11", "polytope", "frst", "status", "error", "instantons",
        "selected_instantons", "qtilde_det", "negative_mode_range",
        "structured_charge_validated", "structured_fallback_reason",
        "search_classification", "mask_count", "masks_visited", "masks_skipped",
        "lattice_copy_count", "lattice_copies_visited", "leading_log_gap",
        "log_scale_span", "strong_hierarchy", "branch_count", "leading_minima_count",
        "saddle_count", "candidate_slowroll_saddles", "least_tachyonic_min_eta",
        "least_tachyonic_epsilon", "best_min_eta", "best_epsilon", "best_abs_min_eta",
        "mass_min", "mass_max", "negative_mass_count", "stage_load_s",
        "stage_select_s", "stage_hierarchy_s", "stage_factor_s", "stage_mass_basis_s",
        "stage_branches_s", "stage_classify_s", "stage_load_output_bytes",
        "stage_select_output_bytes", "stage_hierarchy_output_bytes",
        "stage_factor_output_bytes", "stage_mass_basis_output_bytes",
        "stage_classify_output_bytes", "stage_allocated_bytes", "stage_output_bytes",
        "total_seconds"
    ],
    "REF" => [
        "row_type", "schema_version", "run_id", "data_dir", "code_commit",
        "max_branches", "negative_mode_range", "random_starts", "random_seed",
        "max_points", "float_tolerance", "high_tolerance", "duplicate_tolerance",
        "working_basis", "physical_basis", "physical_vectors", "scale_status", "h11",
        "polytope", "frst", "search_status", "coverage_status", "refinement_status",
        "search_mode", "branch_source", "branch_estimate", "branch_count",
        "mask_count", "masks_visited", "branch_seed_index", "leading_negative_modes",
        "seed_theta", "float_correction_status", "float_correction_residual",
        "float_correction_iterations", "float_value", "float_gradient_residual",
        "float_gradient_norm", "float_epsilon", "float_min_eta",
        "float_negative_modes", "float_zeroish_modes", "float_positive_modes",
        "high_precision_bits", "high_correction_status", "high_correction_residual",
        "high_correction_iterations", "high_value", "high_gradient_residual",
        "high_gradient_norm", "high_epsilon", "high_min_eta", "high_negative_modes",
        "high_zeroish_modes", "high_positive_modes", "residual_agreement",
        "inertia_agreement", "accepted", "candidate_status", "candidate_reason",
        "wall_seconds", "allocated_bytes", "output_bytes", "failure"
    ],
    "FLOW" => [
        "row_type", "schema_version", "run_id", "data_dir", "code_commit", "h11",
        "polytope", "frst", "candidate_index", "branch_seed_index",
        "leading_negative_modes", "seed_theta", "float_theta", "high_theta",
        "precision_bits", "working_basis", "physical_basis", "scale_status",
        "coordinate_chart", "mass_mode_index", "displacement_sign", "mass_eigenvalue",
        "flow_theta_initial", "flow_theta_final", "flow_status", "flow_end_event",
        "flow_efolds", "flow_slow_roll_efolds", "flow_entry_efolds",
        "flow_exit_efolds", "flow_max_efolds", "flow_step", "flow_accepted",
        "flow_reason", "flow_error", "flow_steps"
    ],
    "PILOT_3" => [
        "row_type", "schema_version", "run_id", "data_root", "geometry_path", "h11",
        "polytope", "frst", "reference_scale", "sampled_scale", "scale_grid",
        "scale_source", "scale_status", "volume_normalization",
        "stored_reference_max_log10_error", "stored_reference_sign_mismatches",
        "leading_log_gap", "log_scale_span", "strong_hierarchy", "search_mode",
        "branch_coverage_status", "branch_estimate", "branch_count", "mask_count",
        "masks_visited", "masks_skipped", "lattice_copy_count",
        "lattice_copies_visited", "estimated_stage_allocated_bytes",
        "max_stage_allocated_bytes", "reference_uncorrected_screen_candidates",
        "branch_seed_index", "leading_negative_modes", "leading_seed_theta",
        "corrected_theta", "correction_status", "correction_residual",
        "correction_iterations", "branch_provenance_id", "branch_match_id",
        "matching_status", "value", "gradient_norm", "epsilon", "min_eta", "max_eta",
        "abs_min_eta", "hessian_min", "hessian_max", "zeroish_modes", "negative_modes",
        "positive_modes", "near_catastrophe", "catastrophe_bracket",
        "catastrophe_reason", "candidate_status", "candidate_reason", "seed_count",
        "corrected_count", "matched_count", "lost_count", "new_count",
        "duplicate_count", "correction_failed_count", "near_catastrophe_brackets",
        "screen_candidate_count", "corrected_candidate_count", "minima_count",
        "wall_seconds", "allocated_bytes", "output_bytes", "failure"
    ],
    "PILOT_4" => [
        "row_type", "schema_version", "run_id", "data_root", "geometry_path", "h11",
        "polytope", "frst", "reference_scale", "sampled_scale", "scale_grid",
        "scale_source", "scale_status", "volume_normalization",
        "domain_certificate_version", "domain_status", "domain_reason",
        "fixed_point_status", "trajectory_status", "coverage_status", "moduli_status",
        "phase_convention", "units", "normalization", "source_identity",
        "precision_bits", "source_numeric_type", "source_precision_bits",
        "target_numeric_type", "target_precision_bits", "conversion_status",
        "conversion_error_bound", "conversion_tolerance", "conversion_comparison",
        "stored_reference_max_log10_error", "stored_reference_sign_mismatches",
        "leading_log_gap", "log_scale_span", "strong_hierarchy", "search_mode",
        "branch_coverage_status", "branch_estimate", "branch_count", "mask_count",
        "masks_visited", "masks_skipped", "lattice_copy_count",
        "lattice_copies_visited", "estimated_stage_allocated_bytes",
        "max_stage_allocated_bytes", "reference_uncorrected_screen_candidates",
        "branch_seed_index", "leading_negative_modes", "leading_seed_theta",
        "corrected_theta", "correction_status", "correction_residual",
        "correction_iterations", "branch_provenance_id", "branch_match_id",
        "matching_status", "value", "gradient_norm", "epsilon", "min_eta", "max_eta",
        "abs_min_eta", "hessian_min", "hessian_max", "zeroish_modes", "negative_modes",
        "positive_modes", "near_catastrophe", "catastrophe_bracket",
        "catastrophe_reason", "candidate_status", "candidate_reason", "seed_count",
        "corrected_count", "matched_count", "lost_count", "new_count",
        "duplicate_count", "correction_failed_count", "near_catastrophe_brackets",
        "screen_candidate_count", "corrected_candidate_count", "minima_count",
        "wall_seconds", "allocated_bytes", "output_bytes", "failure"
    ],
    "PILOT_6" => [
        "row_type", "schema_version", "run_id", "data_root", "geometry_path", "h11",
        "polytope", "frst", "reference_scale", "sampled_scale", "scale_grid",
        "scale_source", "scale_status", "volume_normalization",
        "domain_certificate_version", "domain_status", "domain_reason",
        "physical_scaling_gate_status", "physical_scaling_gate_reason",
        "physical_scaling_gate_provenance", "physical_control_gate_status",
        "physical_control_gate_reason", "physical_control_gate_provenance",
        "physical_viability_status", "physical_viability_reason", "fixed_point_status",
        "trajectory_status", "coverage_status", "moduli_status", "phase_convention",
        "units", "normalization", "source_identity", "precision_bits",
        "source_numeric_type", "source_precision_bits", "target_numeric_type",
        "target_precision_bits", "conversion_status", "conversion_error_bound",
        "conversion_tolerance", "conversion_comparison", "conversion_policy_version",
        "stored_reference_max_log10_error", "stored_reference_sign_mismatches",
        "leading_log_gap", "log_scale_span", "strong_hierarchy", "search_mode",
        "branch_coverage_status", "branch_estimate", "branch_count", "mask_count",
        "masks_visited", "masks_skipped", "lattice_copy_count",
        "lattice_copies_visited", "estimated_stage_allocated_bytes",
        "max_stage_allocated_bytes", "reference_uncorrected_screen_candidates",
        "branch_seed_index", "leading_negative_modes", "leading_seed_theta",
        "corrected_theta", "correction_status", "correction_residual",
        "correction_iterations", "branch_provenance_id", "branch_match_id",
        "matching_status", "value", "gradient_norm", "epsilon", "min_eta", "max_eta",
        "abs_min_eta", "hessian_min", "hessian_max", "zeroish_modes", "negative_modes",
        "positive_modes", "near_catastrophe", "catastrophe_bracket",
        "catastrophe_reason", "candidate_status", "candidate_reason", "seed_count",
        "corrected_count", "matched_count", "lost_count", "new_count",
        "duplicate_count", "correction_failed_count", "near_catastrophe_brackets",
        "screen_candidate_count", "corrected_candidate_count",
        "physical_eligible_count", "minima_count", "wall_seconds", "allocated_bytes",
        "output_bytes", "failure"
    ],
)

const INFLATION_CSV_LEGACY_ENTRIES = (
    (id="INF-PREP-1", header="PREP_A", row_types=nothing,
        versions=Dict{String, String}("contract_version" => "1")),
    (id="INF-PREP-2", header="PREP_A", row_types=nothing,
        versions=Dict{String, String}("contract_version" => "2")),
    (id="INF-PREP-3", header="PREP_B", row_types=nothing,
        versions=Dict{String, String}("contract_version" => "3", "diagnostic_schema_version" => "1")),
    (id="INF-PREP-4", header="PREP_C", row_types=nothing,
        versions=Dict{String, String}("contract_version" => "4", "diagnostic_schema_version" => "1")),
    (id="INF-SHARD-1", header="SHARD_1", row_types=nothing,
        versions=Dict{String, String}("shard_schema_version" => "1", "contract_version" => "3", "diagnostic_schema_version" => "1")),
    (id="INF-SHARD-2", header="SHARD_2", row_types=nothing,
        versions=Dict{String, String}("shard_schema_version" => "2", "contract_version" => "4", "diagnostic_schema_version" => "1")),
    (id="INF-REF-3", header="REF", row_types=("point", "summary"),
        versions=Dict{String, String}("schema_version" => "3")),
    (id="INF-FLOW-1", header="FLOW", row_types=("flow",),
        versions=Dict{String, String}("schema_version" => "1")),
    (id="INF-PILOT-3-REPORT", header="PILOT_3", row_types=("scale",),
        versions=Dict{String, String}("schema_version" => "3")),
    (id="INF-PILOT-3-SHARD", header="PILOT_3", row_types=("branch",),
        versions=Dict{String, String}("schema_version" => "3")),
    (id="INF-PILOT-4-REPORT", header="PILOT_4", row_types=("scale",),
        versions=Dict{String, String}("schema_version" => "4")),
    (id="INF-PILOT-4-SHARD", header="PILOT_4", row_types=("branch",),
        versions=Dict{String, String}("schema_version" => "4")),
    (id="INF-PILOT-6-REPORT", header="PILOT_6", row_types=("scale",),
        versions=Dict{String, String}("schema_version" => "6")),
    (id="INF-PILOT-6-SHARD", header="PILOT_6", row_types=("branch",),
        versions=Dict{String, String}("schema_version" => "6")),
)

const INFLATION_CSV_VERSION_COLUMNS = ("schema_version", "shard_schema_version",
    "contract_version", "diagnostic_schema_version")
# Current version values, by column and (for `schema_version`) by row type.
const INFLATION_CSV_CURRENT_VERSIONS = Dict{String, String}(
    "contract_version" => "5", "diagnostic_schema_version" => "2",
    "shard_schema_version" => "3")
const INFLATION_CSV_CURRENT_SCHEMA_BY_ROW_TYPE = Dict{String, String}(
    "point" => "4", "summary" => "4", "flow" => "2", "scale" => "7", "branch" => "7")

"""
    inflation_csv_row_label(header, fields; path, row) -> (label::String, inferred::Bool)

Return the label of one inflation CSV row (CYAX-0232 R-007, A3).

- Header with a `kinetic_normalization` column (current layouts): every
  version column present must hold its current value (`contract_version` "5",
  `diagnostic_schema_version` "2", `shard_schema_version` "3", and
  `schema_version` "4" for `point`/`summary`, "2" for `flow`, "7" for
  `scale`/`branch` rows). The label must be `"sugra_c1"`, or
  `"a96_reproduction_raw_radian"` in a row of the per-row-labelled diagnostic
  family (a `diagnostic_schema_version` column); returns `(label, false)`.
- Header without the label column: the row must match exactly one recorded
  legacy entry (`INFLATION_CSV_LEGACY_ENTRIES`); returns
  `("legacy_c_half", true)`.

Throws an `ArgumentError` naming `path`, `row` and the values otherwise:
a field-count mismatch, an empty or disallowed label, a contradictory version
value, an unknown row type, an unknown header, or a legacy row matching no
recorded entry (or more than one).
"""
function inflation_csv_row_label(header, fields; path, row)
    header = String[string(column) for column in header]
    fields = String[string(field) for field in fields]
    where_ = "inflation CSV $(path), row $(row)"
    length(fields) == length(header) || throw(ArgumentError(
        "$(where_): $(length(fields)) fields for $(length(header)) columns"))
    value = Dict{String, String}(zip(String.(header), String.(fields)))
    versions = Dict(c => value[c] for c in INFLATION_CSV_VERSION_COLUMNS if haskey(value, c))
    if haskey(value, "kinetic_normalization")
        isempty(versions) && throw(ArgumentError(
            "$(where_): a labelled header with no version column is not a recorded layout"))
        for (column, found) in versions
            expected = column == "schema_version" ?
                get(INFLATION_CSV_CURRENT_SCHEMA_BY_ROW_TYPE, get(value, "row_type", ""), nothing) :
                INFLATION_CSV_CURRENT_VERSIONS[column]
            expected === nothing && throw(ArgumentError(
                "$(where_): unknown row_type $(repr(get(value, "row_type", "")))"))
            found == expected || throw(ArgumentError(
                "$(where_): $(column) $(repr(found)) contradicts the labelled layout " *
                "(expected $(repr(expected)))"))
        end
        label = value["kinetic_normalization"]
        isempty(label) && throw(ArgumentError("$(where_): empty kinetic_normalization"))
        allowed = haskey(value, "diagnostic_schema_version") ?
            ("sugra_c1", "a96_reproduction_raw_radian") : ("sugra_c1",)
        label in allowed || throw(ArgumentError(
            "$(where_): kinetic_normalization $(repr(label)) is not allowed in this family"))
        return (label, false)
    end
    candidates = [entry for entry in INFLATION_CSV_LEGACY_ENTRIES
        if INFLATION_CSV_LEGACY_HEADERS[entry.header] == header]
    isempty(candidates) && throw(ArgumentError(
        "$(where_): header matches no recorded inflation CSV layout"))
    matches = [entry for entry in candidates
        if (entry.row_types === nothing || get(value, "row_type", "") in entry.row_types) &&
            all(get(value, column, nothing) == expected for (column, expected) in entry.versions)]
    length(matches) == 1 || throw(ArgumentError(
        "$(where_): row_type $(repr(get(value, "row_type", nothing))) and versions " *
        "$(versions) match $(length(matches)) recorded entries of this header"))
    ("legacy_c_half", true)
end

end
