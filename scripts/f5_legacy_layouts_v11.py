"""F5 `…-reproduction-1.1` layouts (CYAX-0232 T041 → T061 reader fragment).

Encodes specs/0232-kinetic-normalization/layouts/f5-reproduction-1.1.md by
composition: each subtree macro of the record (PROV, FROZEN, DIAG, ...) is a
`Sub` (required, optional, map nodes; paths relative to the macro root) and is
mounted at its prefix. Per group: a base set, h11-family additions and flag
deltas. Stdlib only.

Flag model (record, "Flag model"):
  K --keep-details, A --orientifold-audit, D --orientifold-reason-diagnostics
  (requires A), E --export-kaehler-points, M --model-stage (implies E).
  E is inert without K; in G1, A is inert without K. A layout lists the
  minimal representative flag set of its path set (M is listed without E;
  K-off layouts never list E; G1 K-off layouts list A only together with D).
h11 families (record, convention d): T (Table 1 h11 2..7), N (other h11),
  P (G4 only: h11 in {4, 5}, population preflight present; G4 T is {2,3,6,7}).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from f5_legacy_layout_api import Layout


# --------------------------------------------------------------------------
# Composition helpers
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Sub:
    """A subtree: required and optional paths and map nodes, relative."""

    r: frozenset = field(default_factory=frozenset)
    o: frozenset = field(default_factory=frozenset)
    m: frozenset = field(default_factory=frozenset)

    def __or__(self, other: "Sub") -> "Sub":
        r = self.r | other.r
        return Sub(r, (self.o | other.o) - r, self.m | other.m)

    def opt(self) -> "Sub":
        """Everything optional (mounted under a null-able or empty-able parent)."""
        return Sub(frozenset(), self.r | self.o, self.m)

    def at(self, prefix: str) -> "Sub":
        """Mount under `prefix` (a dict key path or an `x[]` element prefix)."""
        def j(p: str) -> str:
            return f"{prefix}.{p}" if prefix else p
        return Sub(frozenset(map(j, self.r)), frozenset(map(j, self.o)),
                   frozenset(map(j, self.m)))


def R(*paths: str) -> Sub:
    return Sub(r=frozenset(paths))


def O(*paths: str) -> Sub:
    return Sub(o=frozenset(paths))


def MAP(*paths: str) -> Sub:
    """Map-node paths: the node key itself is required, and listed as a map node."""
    return Sub(r=frozenset(paths), m=frozenset(paths))


def keys(prefix: str, *names: str) -> tuple[str, ...]:
    """`prefix` plus `prefix.name` for each name (parent path included)."""
    return (prefix,) + tuple(f"{prefix}.{n}" for n in names)


def sub_keys(prefix: str, *names: str) -> tuple[str, ...]:
    """`prefix.name` for each name (parent not included; for `x[]` prefixes)."""
    return tuple(f"{prefix}.{n}" for n in names)


def frac(prefix: str) -> tuple[str, ...]:
    """Object branch of a scalar-or-{numerator, denominator} value."""
    return (f"{prefix}.numerator", f"{prefix}.denominator")


def nu(prefix: str = "nu") -> tuple[str, ...]:
    """Always-object `_fraction_vector_to_json` value."""
    return (prefix,) + frac(prefix)


# --------------------------------------------------------------------------
# Macros (record: "Shared subtree macros")
# --------------------------------------------------------------------------

DESTS = {
    "G1": ("export_kaehler_points", "gs", "h11", "julia_binary", "julia_project",
           "keep_details", "limit", "model_stage", "orientifold_audit",
           "orientifold_reason_diagnostics", "output", "parquet_dir", "progress",
           "qcd_divisor_domain", "w0_imag", "w0_real"),
}
DESTS["G2"] = DESTS["G1"] + ("terminal_ledger",)
DESTS["G2b"] = DESTS["G2"]
DESTS["G3"] = DESTS["G2"] + ("shard_count", "shard_index")
DESTS["G4"] = DESTS["G3"]


def PROV(g: str) -> Sub:
    s = R("source_commit", "git_dirty", "package_version",
          *keys("runtime_versions", "python", "cytools", "numpy", "scipy", "sympy", "pyarrow"),
          *keys("cli_arguments", *DESTS[g]),
          *keys("input_partition_manifest", "status", "scan_basis", "scan_reason",
                "directory", "pattern", "partitions"))
    s |= O(*sub_keys("input_partition_manifest.partitions[]", "path", "size_bytes", "sha256", "status"))
    if g != "G1":
        s |= R("working_tree_identity", "input_partition_manifest_version")
        s |= O(*sub_keys("working_tree_identity", "tree_sha256", "diff_sha256",
                         "untracked_sha256", "untracked_file_count"))
    return s


FROZEN = R("status", "surface_records") | O(
    "reason", "unavailable_cones", "frozen_surface_count",
    *sub_keys("surface_records[]", "rays", "frozen_conifold_count", "formula"))

H21_A = R("status", "reasons", "chi_F_I", "h21_minus", "h21_plus", "components") | O(
    *sub_keys("components[]", "dim", "rays", "chi", "kind"))

CHECKS = ("finite_coordinates", "cone_membership", "finite_cy_volume", "positive_cy_volume",
          "finite_curve_volumes", "positive_curve_volumes", "finite_basis_divisor_volumes",
          "positive_basis_divisor_volumes", "finite_prime_divisor_volumes",
          "positive_prime_divisor_volumes", "prime_divisor_volume_lower_bound",
          "finite_effective_divisor_volumes", "positive_effective_divisor_volumes",
          "effective_divisor_volume_lower_bound", "finite_inverse_metric",
          "positive_inverse_metric")
DIAG = R("attempt_index", "point_kind", "point_seed", "attempted", "solver", "point_status",
         "checks", "divisor_volume_tolerance", "prime_divisor_volume_lower_bound",
         "effective_divisor_volume_lower_bound", "divisor_volume_lower_bounds_enforced",
         "volume_backend") | O(
    "coordinate_dimension", "point_norm", "point_sha256", "minimum_kaehler_slack",
    "cy_volume", "minimum_curve_volume", "minimum_basis_divisor_volume",
    "minimum_prime_divisor_volume", "minimum_effective_divisor_volume",
    "minimum_metric_eigenvalue", "failure_reason", *sub_keys("checks", *CHECKS))

KEXP = R("status", "diagnostic") | DIAG.at("diagnostic") | O(
    "reason", "h11", "point", "cy_volume", "prime_divisor_volumes", "inverse_metric",
    "glsm_charge_matrix", *keys("potential_matrix_convention", "Q"))

PROVREC = R("source_kind", "dataset", "dataset_url", "parquet_file", "row_index", "mirror_h11",
            "mirror_h12", "physical_h11", "physical_h21", "vertex_count", "favorable_checked_by")

TRI_A = R("p0", "q0", "lattice_matrix", "torus_shift_numerator", "torus_shift_denominator",
          "lambda_f", "criterion")
TRI_B = TRI_A | R("reconstruction_schema_version", "reconstruction_rule_version", "action_digest")

FCOMP_v1 = R("sigma_rays", "sigma_dimension", *nu(), "nu_binary_source", "integrality_vector",
             "fixed_toric_dimension", "f_vanishes_identically", "hypersurface_component_dimension")
FCOMP_v2 = FCOMP_v1 | R(*keys("fixed_component_integrality", "method", "shortcut_proof",
                              "quotient_annihilator", "quotient_coordinates")) | O(
    *frac("fixed_component_integrality.quotient_coordinates[]"))  # [enc-only]

CHART = keys("chart", "status", "reason_code", "reason", "chart_cone", "cox_coordinates",
             "nonzero_coordinate_rays", "irrelevant_ideal_avoided", "stabilizer_order",
             "stabilizer_method")
IRS = R("status", "reason_code", "reason", "support", "newton_face",
        "restriction_identically_zero") | O(
    "method", "source_anchor", "dual_point_count", "dual_point_source", "ambient_rays",
    "restriction_statement", "covariance",
    *sub_keys("covariance[]", "q", "image_q", "fixed_dual_point", "phase_exponent",
              "coefficient_allowed", "equation"),
    *frac("covariance[].phase_exponent"),
    *sub_keys("support[]", "q", "orbit_q", "image_q", "cox_exponents", "restricted_character",
              "phase_exponent", "chart_nonzero"),
    *frac("support[].restricted_character[]"), *frac("support[].phase_exponent"),
    *frac("newton_face[][]"),  # list of restricted_character lists (GL:1240)
    *CHART) | Sub(m=frozenset({"chart.cox_coordinates"}))
IRS_PLUS = IRS | O("newton_origin", *frac("newton_origin[]"))

CPW = R(*keys("cox_phase_witness", "status", "equation", "alpha", "lattice_vector"),
        *nu("cox_phase_witness.phase_vector")) | O(
    *frac("cox_phase_witness.alpha[]"), *frac("cox_phase_witness.lattice_vector[]"))
FCL5 = R(*keys("fixed_cone_lattice", "rank", "index", "saturated", "stabilizer_order", "method"))

PROV6 = ("auxiliary_cone", "ambient_cone", "ambient_cone_determinant",
         "ambient_cone_smooth_unimodular", "ambient_cartier_support", "restricted_cartier_support")
LOCALEV = R("local_evidence_schema", "method", "source_anchor", "fixed_lattice_basis",
            "sigma_rays", "sigma_dimension", "fixed_lattice_rank", "status", "reason_code",
            "reason") | O(
    *keys("local_cone", "rays", "coordinates", "determinant"),
    "ambient_cone_provenance", *sub_keys("ambient_cone_provenance[]", *PROV6),
    "ambient_cone_determinants", "local_cone_determinant",
    *keys("ambient_provenance_smoothness", "status", "all_unimodular", "method", "determinants"),
    *keys("local_smoothness", "status", "smooth", "method", "determinant"),
    *keys("restricted_cartier", "status", "integral", "method", "support", "provenance"),
    *sub_keys("restricted_cartier.provenance[]", *PROV6),
    # [enc-only]
    *frac("ambient_cone_provenance[].ambient_cartier_support[]"),
    *frac("ambient_cone_provenance[].restricted_cartier_support[]"),
    *frac("restricted_cartier.support[]"),
    *frac("restricted_cartier.provenance[].ambient_cartier_support[]"),
    *frac("restricted_cartier.provenance[].restricted_cartier_support[]"))

FCOMP_v3 = FCOMP_v2 | CPW | FCL5 | R("invariant_restricted_support", "containment_method") \
    | IRS.at("invariant_restricted_support")
FCOMP_v3L = FCOMP_v3 | O("zero_dimensional_local_smoothness_cartier_evidence") \
    | LOCALEV.at("zero_dimensional_local_smoothness_cartier_evidence").opt()

COMP = R(*keys("component", "sigma_rays", "sigma_dimension"), *nu("component.nu"))

STRATA = sub_keys("orbifold_intersection.singular_strata[]", "cone", "orbit_dimension",
                  "face_lattice_point_count", "face_lattice_points")
PCC_v2 = R("component_sigma_rays", "component_sigma_dimension", "fixed_toric_dimension",
           "fixed_lattice_basis", "quotient_annihilator", "quotient_rays",
           "quotient_maximal_cones", "restricted_anticanonical_coefficients",
           *sub_keys("restricted_anticanonical_coefficients[]", "ray", "coefficient"),
           "status", "reason_code", "reason",
           *keys("nefness", "status", "nef", "method", "support_inequality_margins", "witnesses"),
           *sub_keys("nefness.support_inequality_margins[]", "maximal_cone", "ray", "margin"),
           *keys("orbifold_intersection", "status", "avoided", "method")) | COMP | O(
    *sub_keys("orbifold_intersection", "singular_cones", "positive_dimensional_singular_cones",
              "reason", "singular_strata"), *STRATA,
    # [enc-only]
    *frac("restricted_anticanonical_coefficients[].coefficient"),
    *frac("nefness.support_inequality_margins[].margin"),
    *frac("orbifold_intersection.singular_strata[].face_lattice_points[][]"))

CERT_POS = R("component_sigma_rays", "component_sigma_dimension", "fixed_toric_dimension",
             "fixed_lattice_basis", "status", "reason_code", "reason", "nefness",
             "nefness.status", "orbifold_intersection", "orbifold_intersection.status") | O(
    "restricted_support",
    *sub_keys("nefness", "reason_code", "reason", "nef", "method",
              "support_inequality_margins", "witnesses"),
    *sub_keys("nefness.support_inequality_margins[]", "maximal_cone", "ray", "margin"),
    *frac("nefness.support_inequality_margins[].margin"),
    *sub_keys("nefness.witnesses[]", "maximal_cone", "ray", "support_inequality_margin"),
    *frac("nefness.witnesses[].support_inequality_margin"),
    *sub_keys("orbifold_intersection", "reason_code", "reason", "singular_cones", "avoided",
              "method", "positive_dimensional_singular_cones", "singular_strata"),
    *STRATA, *frac("orbifold_intersection.singular_strata[].face_lattice_points[][]"),
    *keys("fixed_cone_lattice", "source_ray_sublattice_saturated",
          "quotient_character_lattice_saturated", "method"),
    "quotient_annihilator", "quotient_rays", "quotient_maximal_cones",
    "restricted_anticanonical_coefficients",
    *sub_keys("restricted_anticanonical_coefficients[]", "ray", "coefficient"),
    *frac("restricted_anticanonical_coefficients[].coefficient"),
    *keys("section_genericity", "status", "scope", "nondegeneracy", "method", "support_count",
          "newton_face", "reason"),
    *frac("section_genericity.newton_face[][]"),
    "restricted_support_coefficients",
    *sub_keys("restricted_support_coefficients[]", "ray", "coefficient"),
    *frac("restricted_support_coefficients[].coefficient")) \
    | IRS_PLUS.at("restricted_support").opt()
PCC_v3 = CERT_POS | COMP

CERT_ZERO = R("fixed_toric_dimension", "local_evidence_schema", "status", "reason_code",
              "reason") | O(
    "local_evidence", *keys("fixed_cone_lattice", "source_ray_sublattice_saturated",
                            "local_cone_determinant", "smooth_unimodular")) \
    | LOCALEV.at("local_evidence").opt()
ZCC = CERT_ZERO | COMP

DVP5 = keys("dual_vertex_parity", "available", "fixed_dual_vertices", "violations",
            "noninteger_pairings", "condition")
SMOOTH_ACC_v2 = R("status", "verdict", "method", "reasons", *DVP5, "positive_component_checks") \
    | PCC_v2.at("positive_component_checks[]").opt()
SMOOTH_ACC_v3 = R("status", "verdict", "method", "reasons", *DVP5, "positive_component_checks",
                  "zero_dimensional_component_checks") \
    | PCC_v3.at("positive_component_checks[]").opt() \
    | ZCC.at("zero_dimensional_component_checks[]").opt()
SMOOTH_FULL_v3 = R("status", "verdict", "method", "reasons",
                   *keys("dual_vertex_parity", "available", "fixed_dual_vertices", "violations"),
                   "positive_component_checks", "zero_dimensional_component_checks") \
    | O("dual_vertex_parity.noninteger_pairings", "dual_vertex_parity.condition") \
    | PCC_v3.at("positive_component_checks[]").opt() \
    | ZCC.at("zero_dimensional_component_checks[]").opt()


def FCE(fcomp: Sub, smooth: Sub) -> Sub:
    return R("fixed_point_components",
             *keys("fixed_point_set", "description", "component_count", "construction"),
             "smoothness") | MAP("fixed_surface_n_s_evidence") \
        | O("fixed_point_set.smoothness_status") \
        | fcomp.at("fixed_point_components[]").opt() | smooth.at("smoothness")


FCE_v2 = FCE(FCOMP_v2, SMOOTH_ACC_v2)
FCE_v3 = FCE(FCOMP_v3L, SMOOTH_ACC_v3)


def TL(g: str, fce: Sub, v2: bool) -> Sub:
    funnel = ("polytope_index", "frst_class_index", "polytope_id", "frst_hash",
              "matrix_attempt_count", "candidate_attempt_count", "status_counts",
              "accepted_witness", "accepted_for_table_1") + (("polytope_normal_form_id",) if v2 else ())
    witness = "class_funnel[].accepted_witness"
    return R("schema_version", "provenance",
             *keys("metadata", "program", "requested_h11", "candidate_schema",
                   "acceptance_scope", "search_scope"),
             "sidecar_path", "sidecar_sha256", "record_count", "class_count", "class_funnel",
             *keys("attempt_accounting", "matrix_validation_attempts", "candidate_attempts",
                   "terminal_records")) \
        | MAP("record_kind_counts", "terminal_status_counts") \
        | PROV(g).at("provenance") \
        | (O(*sub_keys("class_funnel[]", *funnel),
             *sub_keys(witness, "candidate_id", "matrix_id", "lambda_f", "torus_shift",
                       "fixed_component_evidence"),
             *frac(f"{witness}.torus_shift"))
           | Sub(m=frozenset({"class_funnel[].status_counts"}))
           | fce.at(f"{witness}.fixed_component_evidence").opt())


SURF_CERT = ("c2_ambient_restricted", "c2_surface", "c1_surface_squared", "n_s",
             "invariant_lattice_basis", "quotient_annihilator", "quotient_surface_rays",
             "quotient_surface_cones", "surface_cone_provenance",
             "surface_cone_provenance[].quotient_cone", "surface_cone_provenance[].ambient_cones",
             "boundary_ray_scales", "boundary_ray_scales[].quotient_ray",
             "boundary_ray_scales[].scale", "reference_ambient_cone",
             "restricted_divisor_coefficients", "restricted_divisor_coefficients[].ambient_ray",
             "restricted_divisor_coefficients[].coefficients",
             "restricted_divisor_coefficients[].coefficients[].quotient_ray",
             "restricted_divisor_coefficients[].coefficients[].coefficient")
SURF_ENC = ("c2_ambient_restricted", "c2_surface", "c1_surface_squared", "n_s",
            "restricted_divisor_coefficients[].coefficients[].coefficient")
ROW_ATT = R("polytope_index", "frst_class_index", "matrix_id", "lattice_matrix",
            *keys("fixed_component", "sigma_rays"), *nu("fixed_component.nu"),
            "status", "reason_code", "reason",
            *keys("surface_data", "status", "reason_code", "reason", "sigma_rays",
                  "sigma_dimension")) | O(
    *sub_keys("surface_data", *SURF_CERT),
    *(p for e in SURF_ENC for p in frac(f"surface_data.{e}")),  # [enc-only]
    *keys("candidate_context", "candidate_id", "lambda_f", "h11_plus", "h11_minus",
          "candidate_terminal_status"), *nu("candidate_context.torus_shift"))


def ROW_UNRES(fcomp: Sub) -> Sub:
    return R("polytope_index", "frst_class_index", "matrix_id", "candidate_id", "lattice_matrix",
             *nu("torus_shift"), "lambda_f", "h11_plus", "h11_minus",
             "candidate_terminal_status", "fixed_component", "reason_code", "reason") \
        | fcomp.at("fixed_component")


def ROWS(fcomp: Sub) -> Sub:
    return (ROW_ATT.at("surface_attempts[]") | ROW_ATT.at("certified_surfaces[]")
            | ROW_UNRES(fcomp).at("unresolved_components[]")).opt()


def RD_TOP(fcomp: Sub) -> Sub:
    return R("schema_version", "h11", "surface_attempt_count", "skipped_surface_count",
             "certified_surface_count", "unresolved_candidate_component_count",
             "surface_attempts", "unresolved_components", "certified_surfaces") \
        | MAP("reason_counts", "reason_counts_by_h11", "reason_counts_by_polytope",
              "reason_counts_by_frst_class", "reason_counts_by_matrix",
              "reason_counts_by_fixed_component", "unresolved_candidate_reason_counts") \
        | Sub(m=frozenset({"reason_counts_by_h11.<map>"})) | ROWS(fcomp)


def RD_DET(fcomp: Sub) -> Sub:
    return R("polytope_index", "surface_attempts", "unresolved_components",
             "certified_surfaces") | ROWS(fcomp)


ORBIT = R("ordinary_euler", "method", "normalization", "genericity", "restricted_support",
          "orbits", *sub_keys("orbits[]", "cone", "orbit_dimension",
                              "newton_face_affine_dimension", "newton_face_vertices",
                              "newton_face_lattice_point_count", "normalized_lattice_volume",
                              "independent_ehrhart_normalized_lattice_volume", "chi", "method")) \
    | IRS_PLUS.at("restricted_support") \
    | O(*keys("orbits[].newton_face_difference_lattice", "rank", "index", "smith_invariants",
              "method"))

COMPEV = R("sigma_rays", *nu(), "contained_in_hypersurface", "containment_method",
           "invariant_restricted_support", "ambient_component_dimension", "chi",
           "fixed_surface_n_s", "certificate", "method", "orbit_evidence", "euler_status",
           "euler_reason_code", "euler_reason") \
    | IRS.at("invariant_restricted_support") | CPW | FCL5 \
    | (CERT_POS | CERT_ZERO).at("certificate").opt() | ORBIT.at("orbit_evidence").opt()

PAIRING = ("q", *nu("pairing_2t_q"), *nu("total"))
EA = R("schema_version", "reconstruction_rule_version", "construction", "source_gauge",
       *keys("source_anchors", "sheridan_population", "sheridan_example", "moritz_trilayer",
             "moritz_fan", "moritz_h2", "moritz_parity", "moritz_fixed_components",
             "moritz_smoothness", "moritz_hodge"),
       *keys("source_archive_sha256", "sheridan_arXiv_2412_12012v1", "moritz_arXiv_2305_06363v1"),
       *keys("source_archive_paths", "sheridan_arXiv_2412_12012v1", "moritz_arXiv_2305_06363v1"),
       "aggregate_labels_used", "unpublished_witnesses_used", "polytope_id",
       "primal_vertex_index", "p0", "dual_vertex_count", "terminal_status", "reason",
       "pairings", "dual_facet_vertex_indices", "q0_vertex_index", "q0", "hull_checks",
       *keys("action", "lattice_matrix", "lambda_f"), *nu("action.torus_shift"),
       "action_digest", "lattice_matrix", *nu("torus_shift"), "lambda_f",
       *keys("action_involution", "status", "linear_involution", "integral_two_t", "two_t"),
       *frac("action_involution.two_t[]"), "matrix_digest",
       *keys("fan_preservation", "status", "reason")) | O(
    *sub_keys("hull_checks[]", "vertex", "translated_facet_point", "in_facet_hull"),
    *sub_keys("fan_preservation", "simplex_count", "point_count", "frst_hash"),
    "frst_hash", "candidate_id",
    *keys("h2_action", "matrix", "h11_plus", "h11_minus", "proof"),
    *sub_keys("h2_action.proof", "method", "equation", "row_convention", "Q_shape", "P_shape",
              "Q_rank", "selected_column_indices", "selected_minor_determinant",
              "exact_rational_solution", "integral_solution", "exact_residual_zero",
              "exact_residual_max_abs", "involution_verified_exactly"),
    *keys("eq_4_45_parity", "status", "checked_vertices", "violations", "reason", "condition"),
    *sub_keys("eq_4_45_parity.checked_vertices[]", *PAIRING),
    *sub_keys("eq_4_45_parity.violations[]", *PAIRING),
    "fixed_components", "smoothness",
    *keys("fixed_locus_euler", "status", "components", "reason", "reason_code", "chi_F_I",
          "component_count", "certified_component_count", "method", "component_union"),
    *keys("hodge_split", "h11_plus", "h11_minus", "h21_plus", "h21_minus", "chi_fixed_locus",
          "chi_x")) \
    | FCOMP_v3L.at("fixed_components[]").opt() | SMOOTH_FULL_v3.at("smoothness").opt() \
    | COMPEV.at("fixed_locus_euler.components[]").opt()

H21_B = R("status", "reasons", "chi_F_I", "h21_minus", "h21_plus", "components") \
    | O("exact_action") | EA.at("exact_action").opt() | COMPEV.at("components[]").opt()

PREFLIGHT = R("status", "policy", "h11", "handoffs",
              *sub_keys("handoffs[]", "path", "sha256", "bytes_read", "markers"),
              *keys("source_contract", "handoff", "handoff_source_path", "source_path",
                    "source_path_origin", "required_partitions", "expected_directory_digest",
                    "observed_directory_digest", "directory_digest_status", "partitions"),
              *sub_keys("source_contract.partitions[]", "partition", "path", "expected_sha256",
                        "observed_sha256", "size_bytes"),
              *keys("artifacts", "directory", "checksum_manifest", "verified_file_count",
                    "verified_files", "merged_artifact", "gap_analysis_artifact",
                    "artifact_digests", "metadata", "population_complete"),
              *sub_keys("artifacts.verified_files[]", "path", "sha256"),
              *keys("artifacts.artifact_digests.merged_artifact", "path", "sha256"),
              *keys("artifacts.artifact_digests.gap_analysis_artifact", "path", "sha256"),
              *sub_keys("artifacts.metadata", "h11", "favorable_polytopes", "frst_classes",
                        "h21_plus_zero_trilayer_frst_classes"))


# --------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------

COMMITS = {
    "G1": ("b191af1", "4f26e07"),
    "G2": ("31cf465", "06f86be", "d3cebea"),
    "G2b": ("2ae6bc8",),
    "G3": ("f45ea31", "68df2a2", "b1aa7e4", "b50ca89", "5388755", "a2018d9", "64c8450",
           "d151bb7", "f83f509", "9f31d71", "5081059", "fdaeaad", "4119b0d", "c396ea2",
           "7709dff", "925a8c1", "d633127", "6bb9ffc", "2dfa8f3", "ee569fa", "82d8b1d",
           "b3719e3", "b829190"),
    "G4": ("239a3a7", "a59a14c", "8a758cc", "6a610b6", "1ea31f5", "563a367", "2ea2e10",
           "85cb6e1", "20f0084", "1342592", "c0d05c0", "2162e81", "78cb603", "07552d0",
           "d12b97d", "ef3643f", "de0762f", "417accf", "d082e96", "ec7ca38", "df0b114",
           "fd14677", "d72ec45", "ce6d9f4", "975f8c8", "931d40c", "49fc6b4", "be6714b"),
}
FAMILIES = {"G1": ("T", "N"), "G2": ("T", "N"), "G2b": ("T", "N"), "G3": ("T", "N"),
            "G4": ("P", "T", "N")}
FCOMP = {"G1": FCOMP_v1, "G2": FCOMP_v2, "G2b": FCOMP_v2, "G3": FCOMP_v2, "G4": FCOMP_v3}
FCE_OF = {"G2": FCE_v2, "G2b": FCE_v2, "G3": FCE_v2, "G4": FCE_v3}
COUNTS = ("favorable_polytopes", "raw_frsts", "frst_classes", "raw_trilayer_polytopes",
          "raw_trilayer_frst_classes", "nonfrozen_trilayer_frst_classes",
          "h21_plus_zero_trilayer_frst_classes", "identity_torus_action_count",
          "identity_torus_action_cy_count", "identity_valid_o3o7_action_cy_count",
          "source_evidence_inherited_orientifold_cys",
          "source_evidence_h11_minus_zero_orientifold_cys",
          "source_vertex_evidence_inherited_orientifold_cys",
          "source_vertex_evidence_h11_minus_zero_orientifold_cys",
          "kaehler_point_export_accepted_count", "kaehler_point_export_rejected_count")
INPUT = ("source", "parquet_dir", "requested_h11", "favorable_lattice", "record_count",
         "population_complete", "population_completion_basis", "population_completion_reason",
         "population_completion_loaded_favorable_polytopes",
         "population_completion_expected_favorable_polytopes")
TARGETS = ("favorable_polytopes", "frst_classes", "inherited_orientifold_cys",
           "h11_minus_zero_orientifold_cys", "h11_minus_zero_h21_plus_zero_orientifold_cys",
           "models")
DETAILS = ("polytope_index", "provenance", "raw_frst_count", "frst_class_count", "trilayer",
           "frozen_conifold", "frozen_conifold_per_class", "nonfrozen_class_count",
           "h21_plus_zero_per_class", "h21_plus_zero_class_count",
           "identity_torus_action_numerators", "identity_valid_o3o7_action_numerators",
           "orientifold_action_audit", "kaehler_point_export_per_class")
MODEL = ("input_record_count", "total_model_count", "model_count_per_record", "models",
         "convention", "diagnostic_reason")
CONVENTION = ("qcd_divisor_domain", "qcd_divisor_domain_justification", "gs",
              "gs_justification", "w0_real", "w0_imag", "w0_justification")
MODELS_ELEM = ("record_index", "axion_index", "qcd_divisor_index", "lambda",
               "mass_reference_log10_ev", "tau_reference")


def base(g: str, fam: str) -> Sub:
    s = R("schema_version", "paper", "orientifold_source", "schema_metadata", "run_provenance",
          *keys("input", *INPUT), *keys("counts", *COUNTS), "paper_targets",
          *keys("claim_status", "favorable_polytopes", "frst_classes", "h21_plus_zero",
                "model_count"),
          "model_stage", "orientifold_reason_diagnostics", "details") \
        | MAP("schema_metadata.deprecated_aliases") | PROV(g).at("run_provenance")
    if g != "G1":
        s |= R("terminal_ledger")
    if g in ("G3", "G4"):
        s |= R(*keys("input.shard", "index", "count", "is_sharded", "shard_favorable_polytopes",
                     "total_favorable_polytopes"))
    if g == "G4":
        s |= R("population_preflight")
    if fam in ("T", "P"):
        s |= R(*sub_keys("paper_targets", *TARGETS))
    if fam == "P":
        s |= PREFLIGHT.at("population_preflight")
    return s


def layout_sub(g: str, fam: str, k: bool, a: bool, d: bool, e: bool, m: bool) -> Sub:
    s = base(g, fam)
    fcomp = FCOMP[g]
    if k:
        s |= R(*sub_keys("details[]", *DETAILS)) | PROVREC.at("details[].provenance")
        s |= (TRI_B if g == "G4" else TRI_A).at("details[].trilayer").opt()
        s |= FROZEN.at("details[].frozen_conifold").opt()
        s |= FROZEN.at("details[].frozen_conifold_per_class[]").opt()
        s |= (H21_B if g == "G4" else H21_A).at("details[].h21_plus_zero_per_class[]").opt()
    if a:
        if g != "G1":
            s |= TL(g, FCE_OF[g], v2=g != "G2").at("terminal_ledger")
        if k:
            s |= R(*sub_keys("details[].orientifold_action_audit", "inherited", "h11_minus_zero",
                             "h11_minus_zero_classes"))
    if d:
        s |= R("orientifold_reason_diagnostics") | RD_TOP(fcomp).at("orientifold_reason_diagnostics")
        if k:
            s |= R("details[].orientifold_action_audit.reason_diagnostics") \
                | RD_DET(fcomp).at("details[].orientifold_action_audit.reason_diagnostics")
    if (e or m) and k:
        s |= KEXP.at("details[].kaehler_point_export_per_class[]").opt()
    if m:
        s |= R(*sub_keys("model_stage", *MODEL), *sub_keys("model_stage.convention", *CONVENTION))
        s |= O("model_stage.convention.prefactor_P")
        if k:
            s |= O(*sub_keys("model_stage.models[]", *MODELS_ELEM))
    return s


def combos(g: str):
    """Minimal representative flag combinations with distinct path sets."""
    for k in (False, True):
        audit = [(False, False), (True, False), (True, True)]
        if g == "G1" and not k:
            audit = [(False, False), (True, True)]  # A inert without K in G1
        export = [(False, False), (True, False), (False, True)] if k else [(False, False), (False, True)]
        for (a, d), (e, m) in product(audit, export):
            yield k, a, d, e, m


def flag_names(k, a, d, e, m) -> tuple[str, ...]:
    on = []
    if k: on.append("--keep-details")
    if a: on.append("--orientifold-audit")
    if d: on.append("--orientifold-reason-diagnostics")
    if e: on.append("--export-kaehler-points")
    if m: on.append("--model-stage")
    return tuple(sorted(on))


def _build() -> list[Layout]:
    out: list[Layout] = []
    for g in ("G1", "G2", "G2b", "G3", "G4"):
        n = 0
        for fam in FAMILIES[g]:
            for k, a, d, e, m in combos(g):
                n += 1
                s = layout_sub(g, fam, k, a, d, e, m)
                paths = s.r | s.o
                out.append(Layout(
                    id=f"1.1-{g}-L{n:02d}", version="1.1", group=f"1.1-{g}",
                    commits=COMMITS[g], flags=flag_names(k, a, d, e, m), h11_family=fam,
                    required=s.r, optional=s.o,
                    map_nodes=frozenset(p for p in s.m if p in paths or "<map>" in p
                                        and p.rsplit(".<map>", 1)[0] in paths)))
    return out


LAYOUTS: list[Layout] = _build()


if __name__ == "__main__":
    from f5_legacy_layout_api import normalize, synthesize

    ids = [l.id for l in LAYOUTS]
    assert len(ids) == len(set(ids)), "duplicate ids"
    per_group: dict[str, int] = {}
    for l in LAYOUTS:
        per_group[l.group] = per_group.get(l.group, 0) + 1
    expected = {"1.1-G1": 26, "1.1-G2": 30, "1.1-G2b": 30, "1.1-G3": 30, "1.1-G4": 45}
    assert per_group == expected, per_group
    all_commits = [c for g in COMMITS.values() for c in g]
    assert len(all_commits) == 57 and len(set(all_commits)) == 57, len(all_commits)
    seen_sets: dict[tuple, str] = {}
    for l in LAYOUTS:
        assert not (l.required & l.optional), (l.id, sorted(l.required & l.optional)[:5])
        assert normalize(synthesize(l.required), l.map_nodes) == l.required, l.id
        full = l.required | l.optional
        got = normalize(synthesize(full), l.map_nodes)
        assert got == full, (l.id, sorted(full - got)[:5], sorted(got - full)[:5])
        for p in full:
            assert not p.endswith("[]") and not p.endswith(".<map>"), (l.id, p)
        key = (l.group, l.required, l.optional)
        assert key not in seen_sets, (l.id, seen_sets.get(key))
        seen_sets[key] = l.id
    print(f"ok: {len(LAYOUTS)} layouts, per group {per_group}, "
          f"max |R|={max(len(l.required) for l in LAYOUTS)}, "
          f"max |R∪O|={max(len(l.required | l.optional) for l in LAYOUTS)}")
