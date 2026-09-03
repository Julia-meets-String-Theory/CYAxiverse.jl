"""Compare the historical and corrected fixed-component constructions.

Run this on a bounded h11=2/h11=3 mirror slice before any population rerun.
The historical side reproduces the pre-track-2 exact-``nu`` containment and
half-ray condition. The corrected side calls the production implementation.
"""

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_geometric_data_multitriangulation import load_mirror_polytopes
import inherited_orientifold_candidates as candidates
import reproduce_fuzzy_axions_h11_4 as reproduction


def _decode_vector(value):
    denominator = int(value["denominator"])
    return tuple(Fraction(int(item), denominator) for item in value["numerator"])


def _old_fixed_component_records(matrix, torus_shift, lambda_f, fixed_cone_keys):
    """Reproduce the pre-track-2 half-ray and exact-label construction."""

    nu_representatives = candidates.enumerate_projected_lattice_representatives(
        matrix, -1
    )
    fixed_subspace_dimension = candidates._exact_rank(
        np.eye(4, dtype=int) + np.asarray(matrix, dtype=int)
    )
    admissible = []
    for source_rays in fixed_cone_keys:
        rays = [tuple(int(value) for value in ray) for ray in source_rays]
        sigma_dimension = (
            candidates._exact_rank(np.asarray(rays, dtype=int)) if rays else 0
        )
        for nu_record in nu_representatives:
            nu = nu_record["vector"]
            integrality_vector = candidates._fraction_sum(
                [torus_shift, nu]
                + [tuple(Fraction(value, 2) for value in ray) for ray in rays]
            )
            if not candidates._is_integral(integrality_vector):
                continue
            ambient_dimension = fixed_subspace_dimension - sigma_dimension
            if ambient_dimension < 0:
                continue
            vanishes_identically = (sigma_dimension + int(lambda_f)) % 2 == 1
            if ambient_dimension == 0 and not vanishes_identically:
                continue
            admissible.append(
                {
                    "sigma_rays": [list(ray) for ray in rays],
                    "sigma_dimension": sigma_dimension,
                    "nu": candidates._fraction_vector_to_json(nu),
                    "nu_binary_source": nu_record["binary_source"],
                    "integrality_vector": candidates._integer_vector(integrality_vector),
                    "fixed_toric_dimension": ambient_dimension,
                    "f_vanishes_identically": bool(vanishes_identically),
                    "hypersurface_component_dimension": (
                        ambient_dimension
                        if vanishes_identically
                        else max(ambient_dimension - 1, 0)
                    ),
                }
            )

    retained = []
    for component in admissible:
        rays = frozenset(tuple(ray) for ray in component["sigma_rays"])
        nu_key = json.dumps(component["nu"], sort_keys=True)
        if any(
            frozenset(tuple(ray) for ray in other["sigma_rays"]) < rays
            and json.dumps(other["nu"], sort_keys=True) == nu_key
            for other in admissible
        ):
            continue
        retained.append(component)
    return sorted(
        retained,
        key=lambda item: (
            item["sigma_dimension"],
            tuple(tuple(ray) for ray in item["sigma_rays"]),
            tuple(item["nu"]["numerator"]),
        ),
    )


def _component_key(component):
    return json.dumps(
        {
            "sigma_rays": component["sigma_rays"],
            "nu": component["nu"],
        },
        sort_keys=True,
    )


def _component_delta(old_components, new_components):
    old_by_key = {_component_key(component): component for component in old_components}
    new_by_key = {_component_key(component): component for component in new_components}
    added = [new_by_key[key] for key in sorted(set(new_by_key) - set(old_by_key))]
    removed = [old_by_key[key] for key in sorted(set(old_by_key) - set(new_by_key))]

    for component in added:
        method = component.get("fixed_component_integrality", {}).get("method")
        component["delta_reason"] = (
            "general_quotient_eq_4.30_accepts_label_rejected_by_old_half_ray"
            if method == "general_quotient_lattice_eq_4.30"
            else "new_component_label"
        )
    for component in removed:
        if any(
            any(
                tuple(old["sigma_rays"]) == tuple(new["sigma_rays"])
                and candidates._nu_equal_mod_span(
                    _decode_vector(old["nu"]),
                    _decode_vector(new["nu"]),
                    tuple(tuple(ray) for ray in old["sigma_rays"]),
                )
                for new in new_components
            )
            for old in (component,)
        ):
            reason = "phase_labels_reduced_modulo_sigma_span"
        elif any(
            candidates._fixed_component_is_contained_in(component, new)
            for new in new_components
        ):
            reason = "contained_in_retained_proper_face_component"
        else:
            reason = "not_present_after_corrected_enumeration"
        component["delta_reason"] = reason

    merged = []
    for new in new_components:
        equivalent_old = [
            old
            for old in old_components
            if tuple(old["sigma_rays"]) == tuple(new["sigma_rays"])
            and candidates._nu_equal_mod_span(
                _decode_vector(old["nu"]),
                _decode_vector(new["nu"]),
                tuple(tuple(ray) for ray in new["sigma_rays"]),
            )
        ]
        if len(equivalent_old) > 1:
            merged.append(
                {
                    "corrected_component": new,
                    "historical_components": equivalent_old,
                    "reason": "phase_labels_reduced_modulo_sigma_span",
                }
            )
    return {
        "old_count": len(old_components),
        "new_count": len(new_components),
        "added": added,
        "removed": removed,
        "merged": merged,
    }


def _status_from_smoothness(smoothness):
    return (
        "accepted_verified_orientifold"
        if smoothness["status"] == "smooth"
        else smoothness["status"]
    )


def _compare_h11(h11, parquet_dir, limit):
    records = load_mirror_polytopes(
        parquet_dir,
        h11=h11,
        limit=limit,
        favorable=True,
    )
    candidate_deltas = []
    for polytope_index, (poly, provenance) in enumerate(records):
        _, classes = reproduction._frst_classes(poly)
        for frst_class_index, triangulation in enumerate(classes):
            triangulation_cones = candidates._triangulation_cones(poly, triangulation)
            topology = dict(
                reproduction.extract_topology(triangulation.get_cy(), triangulation)
            )
            topology["fixed_surface_n_s"] = candidates.identity_fixed_surface_n_s_table(
                triangulation_cones, triangulation
            )
            topology["compute_general_fixed_surface_n_s"] = True
            topology["non_smooth_facet_dual_vertices"] = candidates.facets_with_non_smooth_cones(
                poly, triangulation
            )
            dual_vertices = candidates._extract_dual_vertices(poly)
            current_records = candidates.enumerate_orientifold_candidates(
                poly,
                triangulation,
                topology,
            )
            fixed_cone_cache = {}
            for record in current_records:
                if record.get("record_kind") != "candidate" or record.get("torus_shift") is None:
                    continue
                matrix = np.asarray(record["lattice_matrix"], dtype=int)
                matrix_key = tuple(int(value) for value in matrix.flatten())
                if matrix_key not in fixed_cone_cache:
                    fixed_cone_cache[matrix_key] = candidates._pointwise_invariant_cone_keys(
                        triangulation_cones, matrix
                    )
                fixed_cone_keys = fixed_cone_cache[matrix_key]
                torus_shift = _decode_vector(record["torus_shift"])
                lambda_f = int(record["lambda_f"])
                old_components = _old_fixed_component_records(
                    matrix,
                    torus_shift,
                    lambda_f,
                    fixed_cone_keys,
                )
                new_components = record["fixed_point_components"]
                delta = _component_delta(old_components, new_components)
                matrix_topology = dict(topology)
                matrix_topology["fixed_surface_n_s"] = record.get(
                    "fixed_surface_n_s_evidence", topology.get("fixed_surface_n_s", {})
                )
                old_smoothness = candidates.classify_smoothness(
                    matrix,
                    torus_shift,
                    lambda_f,
                    record["auxiliary_fan"],
                    old_components,
                    matrix_topology,
                    dual_vertices,
                )
                old_status = _status_from_smoothness(old_smoothness)
                current_status = record["terminal_status"]
                status_changed = old_status != current_status
                delta.update(
                    {
                        "candidate_id": record["candidate_id"],
                        "matrix_id": record["matrix_id"],
                        "polytope_index": polytope_index,
                        "frst_class_index": frst_class_index,
                        "lambda_f": lambda_f,
                        "torus_shift": record["torus_shift"],
                        "historical_terminal_status": old_status,
                        "corrected_terminal_status": current_status,
                        "acceptance_change": (
                            {
                                "historical": old_status,
                                "corrected": current_status,
                                "source_rule": "fixed-component enumeration changed smoothness input",
                            }
                            if status_changed
                            else None
                        ),
                    }
                )
                candidate_deltas.append(delta)
    return {
        "h11": int(h11),
        "mirror_directory": str(Path(parquet_dir).expanduser()),
        "limit": int(limit),
        "polytope_count": len(records),
        "candidate_count": len(candidate_deltas),
        "component_deltas": candidate_deltas,
        "acceptance_changes": [
            item["acceptance_change"]
            for item in candidate_deltas
            if item["acceptance_change"] is not None
        ],
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h11-2-parquet", required=True)
    parser.add_argument("--h11-3-parquet", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def main(argv=None):
    args = _parse_args(argv)
    root = SCRIPT_DIR.parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": "cyaxiverse-fixed-component-delta-1.0",
        "source": {
            "paper": "https://arxiv.org/html/2305.06363",
            "equations": ["4.30", "4.32", "4.33", "4.34", "4.35"],
            "historical_rule": "half-ray integrality for every pointwise invariant cone plus exact-nu containment",
            "corrected_rule": "exact quotient-lattice integrality, phase reduction modulo span(sigma), and proper-face containment",
        },
        "provenance": {
            "source_commit": commit,
            "git_dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
            "bounded_before_population_rerun": True,
        },
        "fixtures": [
            _compare_h11(2, args.h11_2_parquet, args.limit),
            _compare_h11(3, args.h11_3_parquet, args.limit),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
