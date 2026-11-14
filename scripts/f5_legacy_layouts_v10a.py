"""F5 `…-reproduction-1.0` layouts, Part A (1.0 groups 1-4), for the T061 reader.

Encodes Part A of specs/0232-kinetic-normalization/layouts/f5-reproduction-1.0.md
by composition from its named path sets. Stdlib only.

Flag convention: `flags` lists the boolean CLI flags that are on, sorted. Where
one layout covers several flag combinations (for example `--model-stage` with or
without `--export-kaehler-points`, or audit/export without `--keep-details`), the
minimal combination is listed. `--h11` is not listed; `h11_family` carries it
("T": h11 has a Table 1 row, including the default 4; "N": no row).
"""

from __future__ import annotations

from f5_legacy_layout_api import Layout

VERSION = "1.0"
NO_MAP_NODES: frozenset[str] = frozenset()  # Part A has no map nodes.


def _paths(*items: str) -> frozenset[str]:
    return frozenset(items)


def _prefixed(prefix: str, names: tuple[str, ...]) -> frozenset[str]:
    return frozenset(f"{prefix}.{name}" for name in names)


# --- Named path sets (Part A "Named path sets") ---------------------------------

PT_CHILDREN = _prefixed(
    "paper_targets",
    (
        "favorable_polytopes",
        "frst_classes",
        "inherited_orientifold_cys",
        "h11_minus_zero_orientifold_cys",
        "h11_minus_zero_h21_plus_zero_orientifold_cys",
        "models",
    ),
)

BASE_1 = (
    _paths("schema_version", "paper", "orientifold_source", "input", "counts",
           "paper_targets", "claim_status", "details")
    | _prefixed(
        "input",
        ("source", "parquet_dir", "requested_h11", "favorable_lattice",
         "record_count", "population_complete"),
    )
    | _prefixed(
        "counts",
        (
            "favorable_polytopes",
            "raw_frsts",
            "frst_classes",
            "raw_trilayer_polytopes",
            "raw_trilayer_frst_classes",
            "nonfrozen_trilayer_frst_classes",
            "identity_torus_action_count",
            "identity_torus_action_cy_count",
            "identity_valid_o3o7_action_cy_count",
            "source_vertex_evidence_inherited_orientifold_cys",
            "source_vertex_evidence_h11_minus_zero_orientifold_cys",
            "kaehler_point_export_accepted_count",
            "kaehler_point_export_rejected_count",
        ),
    )
    | PT_CHILDREN
    | _prefixed("claim_status", ("favorable_polytopes", "frst_classes", "h21_plus_zero"))
)

BASE_2_ADD = _paths("claim_status.model_count", "model_stage")

BASE_3_ADD = _paths("counts.h21_plus_zero_trilayer_frst_classes")

_PROVENANCE = "details[].provenance"
KD_R_1 = (
    _prefixed(
        "details[]",
        (
            "polytope_index",
            "provenance",
            "raw_frst_count",
            "frst_class_count",
            "trilayer",
            "frozen_conifold",
            "frozen_conifold_per_class",
            "nonfrozen_class_count",
            "identity_torus_action_numerators",
            "identity_valid_o3o7_action_numerators",
            "orientifold_action_audit",
            "kaehler_point_export_per_class",
        ),
    )
    | _prefixed(
        _PROVENANCE,
        (
            "source_kind",
            "dataset",
            "dataset_url",
            "parquet_file",
            "row_index",
            "mirror_h11",
            "mirror_h12",
            "physical_h11",
            "physical_h21",
            "vertex_count",
            "favorable_checked_by",
        ),
    )
)

_FROZEN_SHAPE = (
    "status",
    "reason",
    "surface_records",
    "surface_records[].rays",
    "surface_records[].frozen_conifold_count",
    "surface_records[].formula",
    "unavailable_cones",
    "frozen_surface_count",
)
KD_O_1 = (
    _prefixed(
        "details[].trilayer",
        ("p0", "q0", "lattice_matrix", "torus_shift_numerator",
         "torus_shift_denominator", "lambda_f", "criterion"),
    )
    | _prefixed("details[].frozen_conifold", _FROZEN_SHAPE)
    | _prefixed("details[].frozen_conifold_per_class[]", _FROZEN_SHAPE)
)

KD_R_3_ADD = _prefixed("details[]", ("h21_plus_zero_per_class", "h21_plus_zero_class_count"))

KD_O_3_ADD = _prefixed(
    "details[].h21_plus_zero_per_class[]",
    (
        "status",
        "reasons",
        "chi_F_I",
        "h21_minus",
        "h21_plus",
        "components",
        "components[].dim",
        "components[].rays",
        "components[].chi",
        "components[].kind",
    ),
)

AUDIT_R = _prefixed(
    "details[].orientifold_action_audit",
    ("inherited", "h11_minus_zero", "h11_minus_zero_classes"),
)

_EXPORT = "details[].kaehler_point_export_per_class[]"
_DIAG = f"{_EXPORT}.diagnostic"
EXPORT_O = (
    _prefixed(
        _EXPORT,
        (
            "status",
            "reason",
            "h11",
            "point",
            "cy_volume",
            "prime_divisor_volumes",
            "inverse_metric",
            "glsm_charge_matrix",
            "potential_matrix_convention",
            "potential_matrix_convention.Q",
            "diagnostic",
        ),
    )
    | _prefixed(
        _DIAG,
        (
            "attempt_index",
            "point_kind",
            "point_seed",
            "attempted",
            "solver",
            "point_status",
            "checks",
            "divisor_volume_tolerance",
            "prime_divisor_volume_lower_bound",
            "effective_divisor_volume_lower_bound",
            "divisor_volume_lower_bounds_enforced",
            "volume_backend",
            "coordinate_dimension",
            "point_norm",
            "point_sha256",
            "minimum_kaehler_slack",
            "cy_volume",
            "minimum_curve_volume",
            "minimum_basis_divisor_volume",
            "minimum_prime_divisor_volume",
            "minimum_effective_divisor_volume",
            "minimum_metric_eigenvalue",
            "failure_reason",
        ),
    )
    | _prefixed(
        f"{_DIAG}.checks",
        (
            "finite_coordinates",
            "cone_membership",
            "finite_cy_volume",
            "positive_cy_volume",
            "finite_curve_volumes",
            "positive_curve_volumes",
            "finite_basis_divisor_volumes",
            "positive_basis_divisor_volumes",
            "finite_prime_divisor_volumes",
            "positive_prime_divisor_volumes",
            "prime_divisor_volume_lower_bound",
            "finite_effective_divisor_volumes",
            "positive_effective_divisor_volumes",
            "effective_divisor_volume_lower_bound",
            "finite_inverse_metric",
            "positive_inverse_metric",
        ),
    )
)

MS_R = _prefixed(
    "model_stage",
    (
        "input_record_count",
        "total_model_count",
        "model_count_per_record",
        "models",
        "convention",
        "convention.gs",
        "convention.gs_justification",
        "convention.w0_real",
        "convention.w0_imag",
        "convention.w0_justification",
        "diagnostic_reason",
    ),
)

MS_O = _paths("model_stage.convention.prefactor_P")

MS_KD_O = _prefixed(
    "model_stage.models[]",
    ("record_index", "axion_index", "qcd_divisor_index", "lambda",
     "mass_reference_log10_ev", "tau_reference"),
)

# Expected path-set sizes from the Part A headings.
_SET_SIZES = {
    "BASE_1": (BASE_1, 36),
    "BASE_2_ADD": (BASE_2_ADD, 2),
    "BASE_3_ADD": (BASE_3_ADD, 1),
    "PT_CHILDREN": (PT_CHILDREN, 6),
    "KD_R_1": (KD_R_1, 23),
    "KD_O_1": (KD_O_1, 23),
    "KD_R_3_ADD": (KD_R_3_ADD, 2),
    "KD_O_3_ADD": (KD_O_3_ADD, 10),
    "AUDIT_R": (AUDIT_R, 3),
    "EXPORT_O": (EXPORT_O, 50),
    "MS_R": (MS_R, 11),
    "MS_O": (MS_O, 1),
    "MS_KD_O": (MS_KD_O, 6),
}

# --- Groups ----------------------------------------------------------------------

KD = "--keep-details"
AUDIT = "--orientifold-audit"
EXPORT = "--export-kaehler-points"
MS = "--model-stage"

# Group 1 (ec73c15): no --model-stage.
_G1_COMBOS: tuple[tuple[str, ...], ...] = (
    (),
    (KD,),
    (KD, AUDIT),
    (KD, EXPORT),
    (KD, AUDIT, EXPORT),
)
# Groups 2-4: the 8 combinations of Part A's tables (L6-L13 order).
_G2_COMBOS: tuple[tuple[str, ...], ...] = (
    (),
    (MS,),
    (KD,),
    (KD, AUDIT),
    (KD, EXPORT),
    (KD, AUDIT, EXPORT),
    (KD, MS),
    (KD, AUDIT, MS),
)


def _compose(
    flags: tuple[str, ...],
    base_r: frozenset[str],
    kd_r: frozenset[str],
    kd_o: frozenset[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Apply the Part A flag deltas to a group's base and keep-details sets."""

    on = set(flags)
    required = set(base_r)
    optional: set[str] = set()
    effective_export = EXPORT in on or MS in on  # `export_kaehler_points = ... or args.model_stage`
    if MS in on:
        required |= MS_R
        optional |= MS_O
    if KD in on:
        required |= kd_r
        optional |= kd_o
        if AUDIT in on:
            required |= AUDIT_R
        if effective_export:
            optional |= EXPORT_O
        if MS in on:
            optional |= MS_KD_O
    return frozenset(required), frozenset(optional)


def _group(
    group: str,
    commits: tuple[str, ...],
    combos: tuple[tuple[str, ...], ...],
    families: tuple[tuple[str, frozenset[str]], ...],
    kd_r: frozenset[str],
    kd_o: frozenset[str],
) -> list[Layout]:
    layouts = []
    number = 0
    for family, base_r in families:
        for flags in combos:
            number += 1
            required, optional = _compose(flags, base_r, kd_r, kd_o)
            layouts.append(
                Layout(
                    id=f"{group}-L{number:02d}",
                    version=VERSION,
                    group=group,
                    commits=commits,
                    flags=tuple(sorted(flags)),
                    h11_family=family,
                    required=required,
                    optional=optional,
                    map_nodes=NO_MAP_NODES,
                )
            )
    return layouts


B1 = BASE_1
B2 = BASE_1 | BASE_2_ADD
B3 = BASE_1 | BASE_2_ADD | BASE_3_ADD
B3_NO_TARGETS = B3 - PT_CHILDREN
K3R = KD_R_1 | KD_R_3_ADD
K3O = KD_O_1 | KD_O_3_ADD

GROUP_COMMITS = {
    "1.0-G1": ("ec73c15",),
    "1.0-G2": ("aee4c1a",),
    # d36c51b, c2b5851, 8b87ea8 and e24064f share group 3's writer blobs.
    "1.0-G3": ("4d68038", "d36c51b", "c2b5851", "8b87ea8", "e24064f"),
    "1.0-G4": ("efdcbce", "47b1d74"),
}

LAYOUTS: list[Layout] = (
    _group("1.0-G1", GROUP_COMMITS["1.0-G1"], _G1_COMBOS, (("T", B1),), KD_R_1, KD_O_1)
    + _group("1.0-G2", GROUP_COMMITS["1.0-G2"], _G2_COMBOS, (("T", B2),), KD_R_1, KD_O_1)
    + _group("1.0-G3", GROUP_COMMITS["1.0-G3"], _G2_COMBOS, (("T", B3),), K3R, K3O)
    # Group 4: T layouts (L01-L08, equal to group 3's) then N layouts (L09-L16).
    + _group(
        "1.0-G4",
        GROUP_COMMITS["1.0-G4"],
        _G2_COMBOS,
        (("T", B3), ("N", B3_NO_TARGETS)),
        K3R,
        K3O,
    )
)

EXPECTED_COUNTS = {"1.0-G1": 5, "1.0-G2": 8, "1.0-G3": 8, "1.0-G4": 16}


if __name__ == "__main__":
    from collections import Counter

    from f5_legacy_layout_api import normalize, synthesize

    for name, (paths, size) in _SET_SIZES.items():
        assert len(paths) == size, (name, len(paths), size)
    assert PT_CHILDREN <= BASE_1
    assert len(B3_NO_TARGETS) == 33

    ids = [layout.id for layout in LAYOUTS]
    assert len(ids) == len(set(ids)), "duplicate ids"

    counts = Counter(layout.group for layout in LAYOUTS)
    assert dict(counts) == EXPECTED_COUNTS, counts

    for layout in LAYOUTS:
        r, o = layout.required, layout.optional
        assert not (r & o), (layout.id, sorted(r & o))
        assert all(" " not in p and not p.endswith("[]") for p in r | o), layout.id
        assert normalize(synthesize(r), layout.map_nodes) == r, layout.id
        assert normalize(synthesize(r | o), layout.map_nodes) == r | o, layout.id
        assert layout.flags == tuple(sorted(layout.flags)), layout.id

    # Group 4 T layouts equal group 3's (Part A, group 4 layout count note).
    g3 = [l for l in LAYOUTS if l.group == "1.0-G3"]
    g4t = [l for l in LAYOUTS if l.group == "1.0-G4" and l.h11_family == "T"]
    assert [(l.flags, l.required, l.optional) for l in g3] == [
        (l.flags, l.required, l.optional) for l in g4t
    ]

    # Distinct (R, O) pairs over Part A: 5 + 8 + 8 + 8 = 29.
    distinct = {(l.required, l.optional) for l in LAYOUTS}
    assert len(distinct) == 29, len(distinct)

    print(f"ok: {len(LAYOUTS)} layouts, {len(distinct)} distinct path-set pairs")
