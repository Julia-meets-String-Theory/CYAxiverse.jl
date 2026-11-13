"""F5 `...-reproduction-1.0` layouts, Part B (1.0 group 5 and 1.0 group 6).

Encodes Part B of specs/0232-kinetic-normalization/layouts/f5-reproduction-1.0.md
(commits fc66261 .. e7a8f51) against the fixed API in f5_legacy_layout_api.py.
Stdlib only.

Conventions (see Part B of the appendix):
- No bare `m.<map>` path and no bare `k[]` path (API `normalize`). The values
  of a map node are normalized under `m.<map>`; every map of this range has
  scalar values or (for `reason_counts_by_h11`) map values, so no path below a
  map node exists, and the inner level `reason_counts_by_h11.<map>` is listed
  as a map node.
- `details[]` element keys are in R in `--keep-details` layouts (a file with
  `details: []` matches the `--keep-details`-off layout with the same other
  flags).
- `flags` lists the minimal set of long CLI flags that are on for the layout.
  Flags that leave the path set unchanged may also be on in a matching file:
  `--orientifold-audit` when neither `--keep-details` nor
  `--orientifold-reason-diagnostics` is on; `--export-kaehler-points` when
  `--keep-details` is off or `--model-stage` is on; and every flag in the
  "Structure-neutral flags" lists of the appendix.
- h11_family: "T" = `--h11` value with a Table 1 row ({2,...,7} in this
  range), "N" = no row (`paper_targets` is null).
"""

from __future__ import annotations

import itertools

from f5_legacy_layout_api import Layout

VERSION = "1.0"
G5 = ("fc66261", "f5f397c", "4d2fc3b", "4c3f7c5", "1620b5a", "f3bf677", "4fcc3fa", "de1220b")
G6 = ("e7a8f51",)


def _j(prefix: str, rels: list[str]) -> list[str]:
    return [prefix + "." + r for r in rels]


# ---------------------------------------------------------------- base -----
TOP = ["schema_version", "paper", "orientifold_source"]
INPUT = ["input"] + _j("input", ["source", "parquet_dir", "requested_h11", "favorable_lattice",
                                 "record_count", "population_complete"])
COUNTS = ["counts"] + _j("counts", [
    "favorable_polytopes", "raw_frsts", "frst_classes", "raw_trilayer_polytopes",
    "raw_trilayer_frst_classes", "nonfrozen_trilayer_frst_classes",
    "h21_plus_zero_trilayer_frst_classes", "identity_torus_action_count",
    "identity_torus_action_cy_count", "identity_valid_o3o7_action_cy_count",
    "source_vertex_evidence_inherited_orientifold_cys",
    "source_vertex_evidence_h11_minus_zero_orientifold_cys",
    "kaehler_point_export_accepted_count", "kaehler_point_export_rejected_count"])
PT = ["paper_targets"]
PT_KEYS = _j("paper_targets", ["favorable_polytopes", "frst_classes", "inherited_orientifold_cys",
                               "h11_minus_zero_orientifold_cys",
                               "h11_minus_zero_h21_plus_zero_orientifold_cys", "models"])
CLAIM = ["claim_status"] + _j("claim_status", ["favorable_polytopes", "frst_classes",
                                               "h21_plus_zero", "model_count"])
BASE_G5 = TOP + INPUT + COUNTS + PT + PT_KEYS + CLAIM + ["model_stage", "details"]
BASE_G6 = BASE_G5 + ["orientifold_reason_diagnostics"]

# --------------------------------------------------------- model stage -----
MS_R = _j("model_stage", ["input_record_count", "total_model_count", "model_count_per_record",
                          "models", "convention", "diagnostic_reason"]) + _j(
    "model_stage.convention", ["qcd_divisor_domain", "qcd_divisor_domain_justification", "gs",
                               "gs_justification", "w0_real", "w0_imag", "w0_justification"])
MS_O = ["model_stage.convention.prefactor_P"]
MODELS_O = _j("model_stage.models[]", ["record_index", "axion_index", "qcd_divisor_index",
                                       "lambda", "mass_reference_log10_ev", "tau_reference"])

# ------------------------------------------------------------- details -----
_FC = ["status", "reason", "surface_records", "unavailable_cones", "frozen_surface_count",
       "surface_records[].rays", "surface_records[].frozen_conifold_count",
       "surface_records[].formula"]
_H21 = ["status", "reasons", "chi_F_I", "h21_minus", "h21_plus", "components",
        "components[].dim", "components[].rays", "components[].chi", "components[].kind"]
DET_R = _j("details[]", [
    "polytope_index", "provenance", "raw_frst_count", "frst_class_count", "trilayer",
    "frozen_conifold", "frozen_conifold_per_class", "nonfrozen_class_count",
    "h21_plus_zero_per_class", "h21_plus_zero_class_count", "identity_torus_action_numerators",
    "identity_valid_o3o7_action_numerators", "orientifold_action_audit",
    "kaehler_point_export_per_class"]) + _j("details[].provenance", [
    "source_kind", "dataset", "dataset_url", "parquet_file", "row_index", "mirror_h11",
    "mirror_h12", "physical_h11", "physical_h21", "vertex_count", "favorable_checked_by"])
DET_O = (
    _j("details[].trilayer", ["p0", "q0", "lattice_matrix", "torus_shift_numerator",
                              "torus_shift_denominator", "lambda_f", "criterion"])
    + _j("details[].frozen_conifold", _FC)
    + _j("details[].frozen_conifold_per_class[]", _FC)
    + _j("details[].h21_plus_zero_per_class[]", _H21)
)
AUDIT_R = _j("details[].orientifold_action_audit", ["inherited", "h11_minus_zero",
                                                    "h11_minus_zero_classes"])
_DIAG = ["attempt_index", "point_kind", "point_seed", "attempted", "solver", "point_status",
         "checks", "divisor_volume_tolerance", "prime_divisor_volume_lower_bound",
         "effective_divisor_volume_lower_bound", "divisor_volume_lower_bounds_enforced",
         "volume_backend", "coordinate_dimension", "point_norm", "point_sha256",
         "minimum_kaehler_slack", "cy_volume", "minimum_curve_volume",
         "minimum_basis_divisor_volume", "minimum_prime_divisor_volume",
         "minimum_effective_divisor_volume", "minimum_metric_eigenvalue", "failure_reason"]
_CHECKS = ["finite_coordinates", "cone_membership", "finite_cy_volume", "positive_cy_volume",
           "finite_curve_volumes", "positive_curve_volumes", "finite_basis_divisor_volumes",
           "positive_basis_divisor_volumes", "finite_prime_divisor_volumes",
           "positive_prime_divisor_volumes", "prime_divisor_volume_lower_bound",
           "finite_effective_divisor_volumes", "positive_effective_divisor_volumes",
           "effective_divisor_volume_lower_bound", "finite_inverse_metric",
           "positive_inverse_metric"]
_KX = "details[].kaehler_point_export_per_class[]"
EXPORT_O = (
    _j(_KX, ["status", "reason", "h11", "point", "cy_volume", "prime_divisor_volumes",
             "inverse_metric", "glsm_charge_matrix", "potential_matrix_convention",
             "potential_matrix_convention.Q", "diagnostic"])
    + _j(_KX + ".diagnostic", _DIAG)
    + _j(_KX + ".diagnostic.checks", _CHECKS)
)

# ------------------------------------------ reason diagnostics (G6 only) ----
_ORD = "orientifold_reason_diagnostics"
ORD_R = _j(_ORD, ["schema_version", "h11", "surface_attempt_count", "skipped_surface_count",
                  "certified_surface_count", "reason_counts", "reason_counts_by_h11",
                  "reason_counts_by_polytope", "reason_counts_by_frst_class",
                  "reason_counts_by_matrix", "reason_counts_by_fixed_component",
                  "unresolved_candidate_component_count", "unresolved_candidate_reason_counts",
                  "surface_attempts", "unresolved_components", "certified_surfaces"])
ORD_MAP_NODES = _j(_ORD, ["reason_counts", "reason_counts_by_h11", "reason_counts_by_h11.<map>",
                          "reason_counts_by_polytope", "reason_counts_by_frst_class",
                          "reason_counts_by_matrix", "reason_counts_by_fixed_component",
                          "unresolved_candidate_reason_counts"])
_SA = [
    "polytope_index", "frst_class_index", "matrix_id", "lattice_matrix", "fixed_component",
    "fixed_component.sigma_rays", "fixed_component.nu", "fixed_component.nu.numerator",
    "fixed_component.nu.denominator", "status", "reason_code", "reason", "surface_data",
    "surface_data.status", "surface_data.reason_code", "surface_data.reason",
    "surface_data.sigma_rays", "surface_data.sigma_dimension",
    "surface_data.c2_ambient_restricted", "surface_data.c2_ambient_restricted.numerator",
    "surface_data.c2_ambient_restricted.denominator", "surface_data.c2_surface",
    "surface_data.c1_surface_squared", "surface_data.c1_surface_squared.numerator",
    "surface_data.c1_surface_squared.denominator", "surface_data.n_s",
    "surface_data.invariant_lattice_basis", "surface_data.quotient_annihilator",
    "surface_data.quotient_surface_rays", "surface_data.quotient_surface_cones",
    "surface_data.surface_cone_provenance",
    "surface_data.surface_cone_provenance[].quotient_cone",
    "surface_data.surface_cone_provenance[].ambient_cones",
    "surface_data.boundary_ray_scales", "surface_data.boundary_ray_scales[].quotient_ray",
    "surface_data.boundary_ray_scales[].scale", "surface_data.reference_ambient_cone",
    "surface_data.restricted_divisor_coefficients",
    "surface_data.restricted_divisor_coefficients[].ambient_ray",
    "surface_data.restricted_divisor_coefficients[].coefficients",
    "surface_data.restricted_divisor_coefficients[].coefficients[].quotient_ray",
    "surface_data.restricted_divisor_coefficients[].coefficients[].coefficient",
    "candidate_context", "candidate_context.candidate_id", "candidate_context.torus_shift",
    "candidate_context.torus_shift.numerator", "candidate_context.torus_shift.denominator",
    "candidate_context.lambda_f", "candidate_context.h11_plus", "candidate_context.h11_minus",
    "candidate_context.candidate_terminal_status"]
_UC = [
    "polytope_index", "frst_class_index", "matrix_id", "candidate_id", "lattice_matrix",
    "torus_shift", "torus_shift.numerator", "torus_shift.denominator", "lambda_f", "h11_plus",
    "h11_minus", "candidate_terminal_status", "fixed_component", "fixed_component.sigma_rays",
    "fixed_component.sigma_dimension", "fixed_component.nu", "fixed_component.nu.numerator",
    "fixed_component.nu.denominator", "fixed_component.nu_binary_source",
    "fixed_component.integrality_vector", "fixed_component.fixed_toric_dimension",
    "fixed_component.f_vanishes_identically", "fixed_component.hypersurface_component_dimension",
    "reason_code", "reason"]


def _rows(prefix: str) -> list[str]:
    return (_j(prefix + ".surface_attempts[]", _SA)
            + _j(prefix + ".unresolved_components[]", _UC)
            + _j(prefix + ".certified_surfaces[]", _SA))


ORD_ROWS_O = _rows(_ORD)
_RD = "details[].orientifold_action_audit.reason_diagnostics"
DRD_R = [_RD] + _j(_RD, ["polytope_index", "surface_attempts", "unresolved_components",
                         "certified_surfaces"])
DRD_O = _rows(_RD)


# --------------------------------------------------------- composition -----
def _states(group: int):
    """Yield (T, K, X, A, D) for every distinct layout, in table order."""
    for T, K in itertools.product((True, False), (False, True)):
        for X in (("none", "E", "M") if K else ("none", "M")):
            for A in (False, True):
                for D in ((False, True) if group == 6 else (False,)):
                    if not K and not D and A:
                        continue  # audit alone is structure-neutral without K and D
                    yield T, K, X, A, D


def _compose(group: int, T: bool, K: bool, X: str, A: bool, D: bool):
    R = list(BASE_G6 if group == 6 else BASE_G5)
    O: list[str] = []
    maps: list[str] = []
    if not T:
        R = [p for p in R if p not in PT_KEYS]
    if X == "M":
        R += MS_R
        O += MS_O
    if K:
        R += DET_R
        O += DET_O
        if A:
            R += AUDIT_R
        if X in ("E", "M"):
            O += EXPORT_O
        if X == "M":
            O += MODELS_O
    if D:
        R += ORD_R
        maps += ORD_MAP_NODES
        if A:
            O += ORD_ROWS_O
            if K:
                R += DRD_R
                O += DRD_O
    flags = []
    if K:
        flags.append("--keep-details")
    if A and (K or D):
        flags.append("--orientifold-audit")
    if D:
        flags.append("--orientifold-reason-diagnostics")
    if X == "E":
        flags.append("--export-kaehler-points")
    if X == "M":
        flags.append("--model-stage")
    return frozenset(R), frozenset(O), frozenset(maps), tuple(sorted(flags))


def _build() -> list[Layout]:
    out: list[Layout] = []
    for group, commits in ((5, G5), (6, G6)):
        for n, state in enumerate(_states(group), 1):
            R, O, maps, flags = _compose(group, *state)
            out.append(Layout(
                id=f"1.0-G{group}-L{n:02d}",
                version=VERSION,
                group=f"1.0-G{group}",
                commits=commits,
                flags=flags,
                h11_family="T" if state[0] else "N",
                required=R,
                optional=O,
                map_nodes=maps,
            ))
    return out


LAYOUTS: list[Layout] = _build()


if __name__ == "__main__":
    from f5_legacy_layout_api import normalize, synthesize

    ids = [layout.id for layout in LAYOUTS]
    assert len(ids) == len(set(ids)), "duplicate ids"
    counts = {g: sum(1 for l in LAYOUTS if l.group == g) for g in ("1.0-G5", "1.0-G6")}
    assert counts == {"1.0-G5": 16, "1.0-G6": 36}, counts
    keys = [(l.group, l.required, l.optional) for l in LAYOUTS]
    assert len(keys) == len(set(keys)), "two layouts in one group share a path set"
    for layout in LAYOUTS:
        R, O = layout.required, layout.optional
        assert not (R & O), (layout.id, sorted(R & O))
        assert not any(p.endswith("[]") or p.endswith("<map>") for p in R | O), layout.id
        assert normalize(synthesize(R), layout.map_nodes) == R, layout.id
        assert normalize(synthesize(R | O), layout.map_nodes) == R | O, layout.id
    # every map node path must be a (map-typed) R path or an inner map level
    for layout in LAYOUTS:
        for m in layout.map_nodes:
            assert m in layout.required or m.endswith(".<map>"), (layout.id, m)
    # a K-on file with details: [] must match the K-off layout of the same other flags
    for layout in LAYOUTS:
        if "--keep-details" not in layout.flags:
            continue
        s = frozenset(p for p in layout.required if not p.startswith("details[]"))
        hits = [l.id for l in LAYOUTS if l.group == layout.group
                and l.required <= s <= (l.required | l.optional)]
        assert hits, (layout.id, "empty-details file matches no layout")
    print("ok", counts, "layouts:", len(LAYOUTS))
