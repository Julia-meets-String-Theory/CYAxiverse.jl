"""Focused regression tests for the orientifold axiverse database bridge.

Covers `load_ledger_accepted_classes`'s two preserved ledger shapes: a
single-shard summary with `class_funnel` at the top level (h11=2/3), and a
merged, sharded artifact with the same list nested at
`terminal_ledger.class_funnel` (h11=4's `h4.merged.json.zst`).
"""

import hashlib
import itertools
import json
from argparse import Namespace
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from build_orientifold_axion_database import (
    canonical_prime_divisor_volumes,
    action_witness_digest,
    certify_and_build_pending,
    default_run_manifest_path,
    enrich_accepted_witness_matrices,
    EXACT_ACTION_H21_PLUS_STATUS,
    exact_action_h21_diagnostic,
    exact_hodge_split_from_euler,
    find_exact_trilayer_witnesses,
    find_accepted_o3o7_witness,
    load_ledger_accepted_classes,
    load_mpcp_certificates,
    _lookup_mpcp_certificate,
    main,
    orientifold_action_digest,
    lattice_matrix_digest,
    population_set_audit,
    require_clean_git_source,
    require_exact_action_h21_plus_validation,
    _witness_matches_ledger,
    write_run_manifest,
    write_json_report,
)
from glimmers_raw_frst import stable_hash
import orientifold_general_l_geometry as general_l
from toric_fixed_component_euler import (
    component_euler_from_certificate,
    exact_fixed_locus_euler,
    normalized_lattice_volume,
    normalized_lattice_volume_ehrhart,
    transverse_component_euler_orbit,
)
from mpcp_bounded_analysis import build_replay_certificate, point_identity


def _smooth_toric_certificate(rays, maximal_cones, coefficients):
    dimension = len(rays[0])
    if dimension == 1:
        newton_face = [[-1], [0], [1]]
    elif tuple(coefficients) in {(2, 0, 1, 0), (1, 0, 1, 0)}:
        lower = -2 if tuple(coefficients) == (2, 0, 1, 0) else -1
        upper = 2 if tuple(coefficients) == (2, 0, 1, 0) else 2
        newton_face = [[value, 0] for value in range(lower, upper)]
    elif tuple(rays) == ((1, 0), (0, 1), (-1, -2)):
        newton_face = (
            [[value, -1] for value in range(-1, 4)]
            + [[value, 0] for value in range(-1, 2)]
            + [[-1, 1]]
        )
    else:
        newton_face = [
            [x, y] for x in range(-1, 2) for y in range(-1, 2)
        ]
    return {
        "status": "certified",
        "fixed_toric_dimension": len(rays[0]),
        "quotient_rays": [list(ray) for ray in rays],
        "quotient_maximal_cones": [
            [list(rays[index]) for index in cone] for cone in maximal_cones
        ],
        "restricted_anticanonical_coefficients": [
            {"ray": list(ray), "coefficient": {"numerator": coefficient, "denominator": 1}}
            for ray, coefficient in zip(rays, coefficients)
        ],
        "nefness": {"status": "certified", "nef": True},
        "orbifold_intersection": {
            "status": "certified", "avoided": True,
        },
        "section_genericity": {"status": "certified"},
        # Explicit fixture support.  Production certificates obtain this
        # from exact dual lattice points and Eq. (4.42), never by rebuilding
        # a complete line system.
        "restricted_support": {
            "status": "certified",
            "support": [{"q": [0] * dimension, "chart_nonzero": True}],
            "newton_face": newton_face,
            "restriction_identically_zero": False,
            "chart": {"status": "certified"},
        },
    }


def _write_compressed_ledger(directory: Path, name: str, payload: dict) -> None:
    """Write one zstd-compressed ledger fixture and its SHA256SUMS.txt entry."""
    path = directory / name
    encoded = json.dumps(payload).encode("utf-8")
    subprocess.run(
        ["zstd", "-q", "-f", "-o", str(path)],
        input=encoded,
        check=True,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sums_path = directory / "SHA256SUMS.txt"
    with open(sums_path, "a", encoding="utf-8") as stream:
        stream.write(f"{digest}  {name}\n")


_ACCEPTED_ENTRY = {
    "accepted_for_table_1": True,
    "accepted_witness": {"candidate_id": "c1", "lambda_f": 1, "torus_shift": {
        "numerator": [0, 0, 0, 1], "denominator": 2}},
    "candidate_attempt_count": 4,
    "frst_class_index": 0,
    "frst_hash": "deadbeef",
    "matrix_attempt_count": 1,
    "polytope_id": "lattice-points-sha256:aaaa",
    "polytope_index": 0,
    "polytope_normal_form_id": "normal-form-sha256:bbbb",
    "status_counts": {"accepted_verified_orientifold": 1},
}
_REJECTED_ENTRY = {
    "accepted_for_table_1": False,
    "accepted_witness": None,
    "candidate_attempt_count": 4,
    "frst_class_index": 0,
    "frst_hash": "cafef00d",
    "matrix_attempt_count": 1,
    "polytope_id": "lattice-points-sha256:cccc",
    "polytope_index": 1,
    "polytope_normal_form_id": "normal-form-sha256:dddd",
    "status_counts": {"fixed_point_set_non_smooth": 4},
}


def _valid_mpcp_certificate_fixture():
    """Build one source-keyed certificate that passes the replay validator."""
    points = np.asarray([
        [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 1, 0], [0, 0, 0, 1],
    ], dtype=int)
    source = {
        "source_sha256": "source-fixture-sha",
        "parquet_sha256": "source-fixture-sha",
        "source_row": 21,
        "polytope_id": f"lattice-points-sha256:{point_identity(points)}",
        "global_points": points.tolist(),
    }
    action = {
        "lattice_matrix": np.eye(4, dtype=int).tolist(),
        "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1},
        "lambda_f": 1,
    }
    report = {
        "schema_version": "cyaxiverse-bounded-mpcp-replay-1.3",
        "runtime_provenance": {"cytools_version_guard": {
            "status": "verified", "expected": "1.4.12", "observed": "1.4.12",
        }},
        "selected_frst": {"identity": "fixture-frst"},
        "source_identity": source,
    }
    action_record = {
        "terminal_status": "refined_action_evaluated",
        "candidate_index": 0,
        "frst_hash": "fixture-frst",
        "action": action,
        "fixed_locus_euler": {"status": "computed", "chi_F_I": 0, "components": []},
        "refined_glsm": {
            "status": "refined_h2_action_verified", "h2_matrix": [[1]], "proof": {},
        },
        "hodge_split": {
            "h11_plus": 1, "h11_minus": 0, "h21_plus": 0,
            "h21_minus": 0, "chi_fixed_locus": 0, "chi_x": 0,
        },
    }
    certificate = build_replay_certificate(
        26, {"source": source}, report, action_record
    )
    assert certificate is not None
    return certificate


class LoadLedgerAcceptedClassesTests(unittest.TestCase):
    @staticmethod
    def _mock_git_run(status_output=""):
        def run(command, **kwargs):
            if command[1] == "status":
                stdout = status_output
            elif command[1] == "diff":
                stdout = b""
            else:
                stdout = "known-commit\n" if "commit" in command[-1] else "known-tree\n"
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        return run

    def test_generation_requires_clean_known_git_source(self):
        with mock.patch(
            "build_orientifold_axion_database.subprocess.run",
            side_effect=self._mock_git_run(),
        ):
            self.assertEqual(
                require_clean_git_source("/fixture/repo"),
                ("known-commit", mock.ANY),
            )
        with mock.patch(
            "build_orientifold_axion_database.subprocess.run",
            side_effect=self._mock_git_run(" M scripts/dirty.py\n"),
        ):
            with self.assertRaisesRegex(RuntimeError, "dirty Git worktree"):
                require_clean_git_source("/fixture/repo")

    def test_generation_rejects_unknown_git_commit(self):
        def no_commit(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with mock.patch(
            "build_orientifold_axion_database.subprocess.run",
            side_effect=no_commit,
        ):
            with self.assertRaisesRegex(RuntimeError, "known HEAD commit"):
                require_clean_git_source("/fixture/repo")

    def test_population_release_remains_blocked_until_population_verification(self):
        self.assertEqual(EXACT_ACTION_H21_PLUS_STATUS, "not_validated")
        with self.assertRaisesRegex(RuntimeError, "population release is blocked"):
            require_exact_action_h21_plus_validation()

    def test_eq_4_51_published_h11_2_rearrangement(self):
        split = exact_hodge_split_from_euler(
            h11=2, h21=132, h11_minus=0,
            chi_fixed_locus=272, chi_x=-260,
        )
        self.assertEqual(
            split,
            {
                "h11_plus": 2, "h11_minus": 0,
                "h21_plus": 0, "h21_minus": 132,
                "chi_fixed_locus": 272, "chi_x": -260,
            },
        )

    def test_eq_4_51_nonidentity_h11_minus_term(self):
        # Analytic nonidentity fixture: the h11_minus term must not be dropped.
        split = exact_hodge_split_from_euler(
            h11=3, h21=5, h11_minus=1,
            chi_fixed_locus=4, chi_x=-4,
        )
        self.assertEqual(split["h11_plus"], 2)
        self.assertEqual(split["h21_minus"], 2)
        self.assertEqual(split["h21_plus"], 3)

    def test_smooth_p1_contained_and_transverse_euler(self):
        # Independent geometry: chi(P1)=2 and an anticanonical section has
        # degree two, hence consists of two points.
        certificate = _smooth_toric_certificate(
            ((1,), (-1,)), ((0,), (1,)), (1, 1)
        )
        self.assertEqual(component_euler_from_certificate(certificate, contained=True), 2)
        self.assertEqual(component_euler_from_certificate(certificate, contained=False), 2)
        orbit_chi, evidence = transverse_component_euler_orbit(certificate)
        self.assertEqual(orbit_chi, 2)
        self.assertTrue(all(
            item["normalized_lattice_volume"]
            == item["independent_ehrhart_normalized_lattice_volume"]
            for item in evidence["orbits"]
        ))

    def test_normalized_volume_convention_has_no_extra_factorial(self):
        simplex_2 = ((0, 0), (1, 0), (0, 1))
        simplex_3 = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        doubled_simplex_3 = tuple(tuple(2 * value for value in point) for point in simplex_3)
        self.assertEqual(normalized_lattice_volume(simplex_2), 1)
        self.assertEqual(normalized_lattice_volume(simplex_3), 1)
        self.assertEqual(normalized_lattice_volume(doubled_simplex_3), 8)
        self.assertEqual(normalized_lattice_volume_ehrhart(simplex_2), 1)
        self.assertEqual(normalized_lattice_volume_ehrhart(simplex_3), 1)
        self.assertEqual(normalized_lattice_volume_ehrhart(doubled_simplex_3), 8)

    def test_weighted_projective_orbifold_curve_ordinary_euler(self):
        # P(1,1,2): the generic anticanonical curve is elliptic and avoids
        # the A1 torus fixed point. Orbit additivity gives -8 + 8 = 0.
        orbifold = _smooth_toric_certificate(
            ((1, 0), (0, 1), (-1, -2)),
            ((0, 1), (1, 2), (2, 0)),
            (1, 1, 1),
        )
        orbit_chi, evidence = transverse_component_euler_orbit(orbifold)
        self.assertEqual(orbit_chi, 0)
        self.assertEqual(evidence["ordinary_euler"], 0)

        # Crepant smooth refinement by the primitive ray (0,-1). The strict
        # transform is again anticanonical and adjunction independently gives
        # Euler zero.
        refinement = _smooth_toric_certificate(
            ((1, 0), (0, 1), (-1, -2), (0, -1)),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            (1, 1, 1, 1),
        )
        self.assertEqual(
            component_euler_from_certificate(refinement, contained=False),
            orbit_chi,
        )

    def test_lower_dimensional_face_torus_product_has_zero_dense_euler(self):
        # The nef class (2,0) on P1xP1 has a one-dimensional Newton segment.
        # Its dense-orbit zero locus is two copies of C*, hence Euler zero;
        # boundary orbit contributions give the compact divisor Euler four.
        certificate = _smooth_toric_certificate(
            ((1, 0), (0, 1), (-1, 0), (0, -1)),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            (1, 0, 1, 0),
        )
        orbit_chi, evidence = transverse_component_euler_orbit(certificate)
        self.assertEqual(orbit_chi, 4)
        self.assertTrue(any(
            item["method"] == "lower_dimensional_newton_face_torus_product_zero_euler"
            for item in evidence["orbits"]
        ))

    def test_zero_dimensional_component_branches(self):
        point = {"status": "certified", "fixed_toric_dimension": 0}
        self.assertEqual(component_euler_from_certificate(point, contained=True), 1)
        self.assertEqual(component_euler_from_certificate(point, contained=False), 0)

    def test_contained_zero_dimensional_component_requires_local_evidence(self):
        component = {
            "sigma_rays": [],
            "nu": {"numerator": [0, 0, 0, 0], "denominator": 1},
            "fixed_toric_dimension": 0,
            "f_vanishes_identically": True,
        }
        result = exact_fixed_locus_euler([], np.eye(4, dtype=int), [component])
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("local smoothness and Cartier evidence", result["reason"])

    def test_singular_contained_points_det2_det4_are_unavailable(self):
        base = {
            "sigma_rays": [],
            "nu": {"numerator": [0, 0, 0, 0], "denominator": 1},
            "fixed_toric_dimension": 0,
            "f_vanishes_identically": True,
        }
        for determinant in (2, 4):
            with self.subTest(determinant=determinant):
                component = dict(
                    base,
                    zero_dimensional_local_smoothness_cartier_evidence={
                        "status": "certified",
                        "ambient_cone_determinant": determinant,
                        "local_smoothness": {"status": "certified", "smooth": True},
                        "restricted_cartier": {"status": "certified", "integral": True},
                    },
                )
                result = exact_fixed_locus_euler(
                    [], np.eye(4, dtype=int), [component]
                )
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(
                    result["reason_code"], "zero_dimensional_local_fan_not_smooth"
                )

    def test_fractional_local_cone_determinant_is_unavailable(self):
        component = {
            "sigma_rays": [],
            "nu": {"numerator": [0, 0, 0, 0], "denominator": 1},
            "fixed_toric_dimension": 0,
            "f_vanishes_identically": True,
            "zero_dimensional_local_smoothness_cartier_evidence": {
                "status": "certified",
                "ambient_cone_determinant": 1.5,
                "local_smoothness": {"status": "certified", "smooth": True},
                "restricted_cartier": {"status": "certified", "integral": True},
            },
        }
        result = exact_fixed_locus_euler([], np.eye(4, dtype=int), [component])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason_code"], "zero_dimensional_local_fan_not_smooth")

    def test_contained_zero_dimensional_point_requires_and_uses_certified_contract(self):
        component = {
            "sigma_rays": [],
            "nu": {"numerator": [0, 0, 0, 0], "denominator": 1},
            "fixed_toric_dimension": 0,
            "f_vanishes_identically": True,
            "zero_dimensional_local_smoothness_cartier_evidence": {
                "status": "certified",
                "ambient_cone_determinant": 1,
                "local_smoothness": {"status": "certified", "smooth": True},
                "restricted_cartier": {"status": "certified", "integral": True},
            },
        }
        result = exact_fixed_locus_euler([], np.eye(4, dtype=int), [component])
        self.assertEqual(result["status"], "computed")
        self.assertEqual(result["chi_F_I"], 1)

    def test_nonidentity_fixture_contained_and_transverse_components(self):
        # Hand-computable quotient-star fans under a nonidentity action:
        # contained P1 has chi=2; anticanonical (2,2) in P1xP1 is elliptic.
        action = np.diag([1, 1, -1, -1])
        self.assertFalse(np.array_equal(action, np.eye(4, dtype=int)))
        p1 = _smooth_toric_certificate(
            ((1,), (-1,)), ((0,), (1,)), (1, 1)
        )
        p1xp1 = _smooth_toric_certificate(
            ((1, 0), (0, 1), (-1, 0), (0, -1)),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            (1, 1, 1, 1),
        )
        contained = component_euler_from_certificate(p1, contained=True)
        transverse = component_euler_from_certificate(p1xp1, contained=False)
        self.assertEqual((contained, transverse, contained + transverse), (2, 0, 2))

    def test_nonidentity_action_to_components_to_euler_end_to_end(self):
        # Ambient fan of (P1)^4. L fixes the first factor and negates the
        # other three. Thus H_- has 2^3 phase labels and each maximal fixed
        # component is a P1. lambda_f=1 contains all eight P1s; lambda_f=0
        # intersects each in an anticanonical degree-two divisor. Both have
        # hand-derived total Euler 8*2=16.
        ambient_cones = []
        for signs in itertools.product((-1, 1), repeat=4):
            ambient_cones.append(
                tuple(
                    tuple(signs[index] if coordinate == index else 0 for coordinate in range(4))
                    for index in range(4)
                )
            )
        matrix = np.diag([1, -1, -1, -1])
        shift = (0, 0, 0, 0)
        auxiliary = general_l.build_auxiliary_fan(ambient_cones, matrix)
        fixed_cones = general_l._pointwise_invariant_cone_keys(ambient_cones, matrix)
        dual_points = [(-1, 0, 0, 0), (1, 0, 0, 0)]
        ambient_rays = tuple(sorted({ray for cone in ambient_cones for ray in cone}))
        for lambda_f, contained in ((1, True), (0, False)):
            components = general_l._fixed_component_records(
                auxiliary,
                matrix,
                shift,
                lambda_f,
                fixed_cone_keys=fixed_cones,
                dual_points=dual_points,
                ambient_rays=ambient_rays,
                fan_cones=ambient_cones,
            )
            self.assertEqual(len(components), 8)
            self.assertTrue(all(item["f_vanishes_identically"] is contained for item in components))
            result = exact_fixed_locus_euler(auxiliary, matrix, components)
            self.assertEqual(result["status"], "computed")
            self.assertEqual(result["chi_F_I"], 16)
            self.assertEqual([item["chi"] for item in result["components"]], [2] * 8)

    def test_simplicial_orbifold_component_fails_unavailable(self):
        certificate = _smooth_toric_certificate(
            ((1, 0), (0, 1), (-1, -2)),
            ((0, 1), (1, 2), (2, 0)),
            (1, 1, 1),
        )
        with self.assertRaisesRegex(ValueError, "not smooth unimodular"):
            component_euler_from_certificate(certificate, contained=True)

    def test_contained_surface_requires_moritz_n_s_zero(self):
        certificate = _smooth_toric_certificate(
            ((1, 0), (0, 1), (-1, 0), (0, -1)),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            (1, 1, 1, 1),
        )
        component = {
            "sigma_rays": [],
            "sigma_dimension": 0,
            "nu": {"numerator": [0, 0, 0, 0], "denominator": 1},
            "fixed_toric_dimension": 2,
            "f_vanishes_identically": True,
        }
        key = general_l._component_key(component)
        with mock.patch(
            "toric_fixed_component_euler.general_l._positive_component_section_certificate",
            return_value=certificate,
        ):
            missing = exact_fixed_locus_euler([], np.eye(4, dtype=int), [component])
            self.assertEqual(missing["status"], "unavailable")
            certified = exact_fixed_locus_euler(
                [], np.eye(4, dtype=int), [component],
                fixed_surface_n_s_evidence={key: 0},
            )
        self.assertEqual(certified["status"], "computed")
        self.assertEqual(certified["chi_F_I"], 4)

    def test_eq_4_51_rejects_nonintegral_split(self):
        with self.assertRaisesRegex(ValueError, "not divisible by four"):
            exact_hodge_split_from_euler(
                h11=2, h21=3, h11_minus=0,
                chi_fixed_locus=1, chi_x=0,
            )

    def test_general_L_without_geometry_fails_unavailable(self):
        class FakeTriangulation:
            pass

        witness = {
            "lattice_matrix": np.diag([-1, 1, 1, 1]).tolist(),
            "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1},
            "lambda_f": 1,
            "h11_minus": 1,
            "smoothness": {"status": "smooth"},
        }
        result = exact_action_h21_diagnostic(None, FakeTriangulation(), witness)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("component construction failed", result["reason"])

    def test_arbitrary_summary_witness_does_not_hide_valid_live_action(self):
        class FakeTriangulation:
            def get_cy(self):
                return object()

        unavailable = {
            "candidate_id": "first", "matrix_candidate_id": "m1",
            "terminal_status": "accepted_verified_orientifold",
            "h11_minus": 0, "lambda_f": 1,
            "lattice_matrix": np.eye(4, dtype=int).tolist(),
            "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1},
        }
        valid = {
            "candidate_id": "second", "matrix_candidate_id": "m2",
            "terminal_status": "accepted_verified_orientifold",
            "h11_minus": 0, "lambda_f": 1,
            "lattice_matrix": np.eye(4, dtype=int).tolist(),
            "torus_shift": {"numerator": [0, 0, 0, 1], "denominator": 2},
        }
        ledger_entry = {"accepted_witness": {"candidate_id": "first", "matrix_id": "m1"}}
        with mock.patch("build_orientifold_axion_database.mg.extract_topology", return_value={}), \
             mock.patch("build_orientifold_axion_database.ioc._triangulation_cones", return_value=[]), \
             mock.patch("build_orientifold_axion_database.ioc.identity_fixed_surface_n_s_table", return_value={}), \
             mock.patch("build_orientifold_axion_database.ioc.facets_with_non_smooth_cones", return_value=set()), \
             mock.patch("build_orientifold_axion_database.ioc.enumerate_orientifold_candidates", return_value=[unavailable, valid]), \
             mock.patch("build_orientifold_axion_database.exact_action_h21_diagnostic", side_effect=[
                 {"status": "unavailable", "reason": "fixture gap"},
                 {"status": "validated", "h21_plus": 0, "action_digest": "aaa"},
             ]):
            result = find_accepted_o3o7_witness(object(), FakeTriangulation(), ledger_entry)
        self.assertEqual([item["candidate_id"] for item in result], ["second"])
        self.assertEqual(result[0]["orientifold_action_digest"], "aaa")

    def test_no_exact_live_action_still_fails_closed(self):
        class FakeTriangulation:
            def get_cy(self):
                return object()

        record = {
            "candidate_id": "only", "matrix_candidate_id": "m1",
            "terminal_status": "accepted_verified_orientifold",
            "h11_minus": 0, "lambda_f": 1,
            "lattice_matrix": np.eye(4, dtype=int).tolist(),
            "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1},
        }
        with mock.patch("build_orientifold_axion_database.mg.extract_topology", return_value={}), \
             mock.patch("build_orientifold_axion_database.ioc._triangulation_cones", return_value=[]), \
             mock.patch("build_orientifold_axion_database.ioc.identity_fixed_surface_n_s_table", return_value={}), \
             mock.patch("build_orientifold_axion_database.ioc.facets_with_non_smooth_cones", return_value=set()), \
             mock.patch("build_orientifold_axion_database.ioc.enumerate_orientifold_candidates", return_value=[record]), \
             mock.patch("build_orientifold_axion_database.exact_action_h21_diagnostic", return_value={"status": "unavailable"}):
            result, evidence = find_accepted_o3o7_witness(
                object(), FakeTriangulation(), {}, return_all_evidence=True
            )
        self.assertEqual(result, [])
        self.assertEqual(evidence[0]["exact_action_h21_evidence"]["status"], "unavailable")

    def test_exact_action_requires_smoothness_evidence(self):
        witness = {
            "lattice_matrix": np.eye(4, dtype=int).tolist(),
            "torus_shift": {"numerator": [1, 0, 0, 0], "denominator": 2},
            "lambda_f": 1,
            "h11_minus": 0,
        }
        result = exact_action_h21_diagnostic(None, None, witness)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("smoothness certificate", result["reason"])

    def test_canonical_ledger_action_requires_exact_live_lattice_match(self):
        ledger_witness = {
            "lattice_matrix": np.eye(2, dtype=int).tolist(),
            "torus_shift": {"numerator": [0, 1], "denominator": 2},
            "lambda_f": 1,
        }
        live = dict(ledger_witness, matrix_candidate_id="live-matrix")
        self.assertEqual(orientifold_action_digest(live), orientifold_action_digest(ledger_witness))
        self.assertTrue(_witness_matches_ledger(live, ledger_witness))
        mismatched = dict(live, lattice_matrix=[[1, 0], [0, -1]])
        self.assertFalse(_witness_matches_ledger(mismatched, ledger_witness))
        self.assertFalse(_witness_matches_ledger(live, {"lambda_f": 1}))

    def test_full_action_witness_digest_contains_class_and_action_fields(self):
        witness = {
            "polytope_id": "poly",
            "frst_hash": "frst",
            "frst_class_index": 3,
            "matrix_id": "matrix",
            "candidate_id": "candidate",
            "lattice_matrix": np.eye(4, dtype=int).tolist(),
            "torus_shift": {"numerator": [0, 0, 0, 1], "denominator": 2},
            "lambda_f": 1,
        }
        digest = action_witness_digest(witness)
        self.assertEqual(len(digest), 64)
        changed_class = dict(witness, frst_class_index=4)
        changed_shift = dict(
            witness,
            torus_shift={"numerator": [0, 0, 0, 0], "denominator": 1},
        )
        self.assertNotEqual(digest, action_witness_digest(changed_class))
        self.assertNotEqual(digest, action_witness_digest(changed_shift))
        with self.assertRaisesRegex(ValueError, "exact integer entries"):
            lattice_matrix_digest(np.eye(4, dtype=float) * 0.5)

    def test_source_trilayer_witness_flattens_provenance_fields_for_writer(self):
        action = {
            "lattice_matrix": np.eye(4, dtype=int).tolist(),
            "torus_shift": {"numerator": [1, 0, 0, 0], "denominator": 2},
            "lambda_f": 1,
        }
        candidate = {
            "terminal_status": "accepted_exact_trilayer_action",
            "candidate_id": "source-candidate",
            "action": action,
            "action_digest": orientifold_action_digest(action),
            "torus_shift": action["torus_shift"],
            "lambda_f": 1,
            "h2_action": {
                "matrix": np.eye(2, dtype=int).tolist(),
                "proof": {"equation": "M Q = Q P"},
                "h11_plus": 2,
                "h11_minus": 0,
            },
            "hodge_split": {
                "h11_plus": 2,
                "h11_minus": 0,
                "h21_plus": 0,
                "h21_minus": 1,
                "chi_x": 0,
            },
            "fixed_locus_euler": {"chi_F_I": 8},
            "smoothness": {"status": "smooth"},
        }
        reconstruction = {
            "provenance": {"schema_version": "fixture"},
            "candidates": [candidate],
        }
        certificate = _valid_mpcp_certificate_fixture()
        with mock.patch(
            "build_orientifold_axion_database.reconstruct_trilayer_actions",
            return_value=reconstruction,
        ) as reconstruct, mock.patch(
            "build_orientifold_axion_database.compute_polytope_id",
            return_value="poly",
        ), mock.patch(
            "build_orientifold_axion_database.compute_triangulation_hash",
            return_value="frst",
        ):
            witnesses = find_exact_trilayer_witnesses(
                mock.Mock(points=lambda: np.zeros((1, 4), dtype=int)),
                mock.Mock(simplices=lambda: np.zeros((1, 5), dtype=int)),
                frst_class_index=0, mpcp_certificate=certificate,
                source_record={"source": certificate["source"]},
            )
        reconstruction_kwargs = reconstruct.call_args.kwargs
        self.assertIs(reconstruction_kwargs["mpcp_certificate"], certificate)
        self.assertEqual(reconstruction_kwargs["source_record"]["source"], certificate["source"])
        self.assertEqual(len(witnesses), 1)
        witness = witnesses[0]
        self.assertEqual(witness["involution_type"], "O3/O7")
        np.testing.assert_array_equal(witness["h2_involution_matrix"], np.eye(2, dtype=int))
        self.assertEqual(witness["h2_action_proof"]["equation"], "M Q = Q P")
        self.assertEqual(witness["h11_plus"], 2)
        self.assertEqual(witness["h11_minus"], 0)

    def test_matrix_catalog_enriches_only_lattice_matrix_with_verified_source(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            raw = directory / "terminal.jsonl"
            raw.write_text('{"immutable":"fixture"}\n')
            raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
            matrix = np.eye(4, dtype=int).tolist()
            polytope_id, frst_hash = "poly", "frst"
            matrix_id = stable_hash(
                [polytope_id, frst_hash, tuple(value for row in matrix for value in row)]
            )
            catalog = {
                "schema_version": "cyaxiverse-accepted-matrix-catalog-1.1",
                "record_role": "class_level_matrix_catalog",
                "source_sha256": raw_sha,
                "accepted_count": 1,
                "missing": [], "ambiguous": [], "stable_id_recompute_failures": [],
                "catalog": [{
                    "polytope_id": polytope_id, "frst_hash": frst_hash,
                    "frst_class_index": 0,
                    "matrix_id": matrix_id, "lattice_matrix": matrix,
                    "matrix_digest": lattice_matrix_digest(matrix),
                    "source_line_numbers": [7],
                }],
            }
            catalog_path = directory / "catalog.json.zst"
            subprocess.run(
                ["zstd", "-q", "-f", "-19", "-o", str(catalog_path)],
                input=json.dumps(catalog).encode(), check=True,
            )
            accepted = [{
                "polytope_id": polytope_id, "frst_hash": frst_hash,
                "frst_class_index": 0,
                "accepted_witness": {
                    "matrix_id": matrix_id, "candidate_id": "candidate-1",
                    "torus_shift": {"numerator": [0, 0, 0, 0], "denominator": 1},
                    "lambda_f": 1,
                },
            }]
            enriched, provenance = enrich_accepted_witness_matrices(
                accepted, catalog_path, raw
            )
            witness = enriched[0]["accepted_witness"]
            self.assertEqual(witness["lattice_matrix"], matrix)
            self.assertEqual(
                witness["torus_shift"],
                {"numerator": [0, 0, 0, 0], "denominator": 1},
            )
            self.assertEqual(witness["lambda_f"], 1)
            self.assertEqual(witness["matrix_digest"], lattice_matrix_digest(matrix))
            self.assertTrue(witness["action_witness_digest"])
            self.assertEqual(provenance["terminal_ledger_sha256"], raw_sha)

            missing = [{
                "polytope_id": "missing", "frst_hash": frst_hash,
                "frst_class_index": 0,
                "accepted_witness": {"matrix_id": matrix_id},
            }]
            with self.assertRaisesRegex(RuntimeError, "no unique matrix catalog row"):
                enrich_accepted_witness_matrices(missing, catalog_path, raw)

            conflicting = json.loads(json.dumps(accepted))
            conflicting[0]["accepted_witness"]["lattice_matrix"] = np.diag(
                [1, 1, 1, -1]
            ).tolist()
            with self.assertRaisesRegex(RuntimeError, "conflicts with matrix catalog"):
                enrich_accepted_witness_matrices(conflicting, catalog_path, raw)

            duplicate_catalog = dict(catalog, catalog=catalog["catalog"] * 2)
            subprocess.run(
                ["zstd", "-q", "-f", "-19", "-o", str(catalog_path)],
                input=json.dumps(duplicate_catalog).encode(), check=True,
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate matrix catalog key"):
                enrich_accepted_witness_matrices(accepted, catalog_path, raw)

            malformed_matrix = np.eye(3, dtype=int).tolist()
            malformed_id = stable_hash([
                polytope_id, frst_hash,
                tuple(value for row in malformed_matrix for value in row),
            ])
            malformed_catalog = dict(catalog, catalog=[dict(
                catalog["catalog"][0], matrix_id=malformed_id,
                lattice_matrix=malformed_matrix,
            )])
            malformed_accepted = [{
                "polytope_id": polytope_id, "frst_hash": frst_hash,
                "frst_class_index": 0,
                "accepted_witness": {"matrix_id": malformed_id},
            }]
            subprocess.run(
                ["zstd", "-q", "-f", "-19", "-o", str(catalog_path)],
                input=json.dumps(malformed_catalog).encode(), check=True,
            )
            with self.assertRaisesRegex(RuntimeError, "non-4x4 L"):
                enrich_accepted_witness_matrices(malformed_accepted, catalog_path, raw)

            bad_matrix = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            bad_id = stable_hash([
                polytope_id, frst_hash,
                tuple(value for row in bad_matrix for value in row),
            ])
            bad_catalog = dict(catalog, catalog=[dict(
                catalog["catalog"][0], matrix_id=bad_id, lattice_matrix=bad_matrix
            )])
            bad_accepted = [{
                "polytope_id": polytope_id, "frst_hash": frst_hash,
                "frst_class_index": 0,
                "accepted_witness": {"matrix_id": bad_id},
            }]
            subprocess.run(
                ["zstd", "-q", "-f", "-19", "-o", str(catalog_path)],
                input=json.dumps(bad_catalog).encode(), check=True,
            )
            with self.assertRaisesRegex(RuntimeError, "non-involutive L"):
                enrich_accepted_witness_matrices(bad_accepted, catalog_path, raw)

            raw.write_text("changed\n")
            with self.assertRaisesRegex(RuntimeError, "terminal-ledger SHA256"):
                enrich_accepted_witness_matrices(accepted, catalog_path, raw)

    def test_generation_guard_runs_before_manifest_output(self):
        with tempfile.TemporaryDirectory() as directory_name:
            output = Path(directory_name) / "database"
            argv = [
                "build_orientifold_axion_database.py",
                "--parquet-dir", str(Path(directory_name) / "mirror"),
                "--ledger-population-dir", str(Path(directory_name) / "population"),
                "--ledger-name", "ledger.json.zst",
                "--db-root", str(output),
            ]
            with mock.patch("build_orientifold_axion_database.sys.argv", argv):
                with mock.patch(
                    "build_orientifold_axion_database.require_clean_git_source",
                    side_effect=RuntimeError("dirty Git worktree"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "dirty Git worktree"):
                        main()
            self.assertFalse(output.exists())

    def test_manifest_reference_links_compressed_file_digest(self):
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "manifest.json.zst"
            payload = {"schema_version": "test-1.0", "configuration": {"h11": 2}}
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            payload["manifest_payload_sha256"] = hashlib.sha256(encoded).hexdigest()
            reference = write_run_manifest(path, payload)
            self.assertEqual(reference["manifest_file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(reference["manifest_payload_sha256"], payload["manifest_payload_sha256"])
            self.assertEqual(write_run_manifest(path, payload), reference)

    def test_report_is_json_zst_level19_and_refuses_changed_overwrite(self):
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "report.json.zst"
            report = {"h11": 2, "status": "selected"}
            reference = write_json_report(path, report)
            decoded = subprocess.run(
                ["zstd", "-dc", str(path)], check=True, capture_output=True
            ).stdout
            self.assertEqual(json.loads(decoded)["schema_version"],
                             "cyaxiverse-orientifold-bridge-report-1.0")
            expected = subprocess.run(
                ["zstd", "-q", "-19", "-c"],
                input=(json.dumps({"h11": 2, "status": "selected",
                                   "schema_version":
                                   "cyaxiverse-orientifold-bridge-report-1.0"},
                                  indent=2, sort_keys=True) + "\n").encode(),
                check=True, capture_output=True,
            ).stdout
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(reference["file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(write_json_report(path, report), reference)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_json_report(path, {"h11": 3})
            with self.assertRaisesRegex(ValueError, r"\.json\.zst"):
                write_json_report(Path(directory_name) / "report.json", report)

    def test_full_stage_writes_final_report_once_after_build(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            report_path = directory / "report.json.zst"
            manifest_path = directory / "manifest.json.zst"
            argv = [
                "build_orientifold_axion_database.py", "--h11", "2",
                "--parquet-dir", str(directory / "mirror"),
                "--ledger-population-dir", str(directory / "population"),
                "--ledger-name", "ledger.json.zst",
                "--matrix-catalog", str(directory / "catalog.json.zst"),
                "--matrix-terminal-ledger", str(directory / "terminal.jsonl"),
                "--db-root", str(directory / "database"),
                "--manifest", str(manifest_path),
                "--report", str(report_path),
            ]
            manifest = {
                "manifest_file_sha256": "manifest-file",
                "manifest_payload_sha256": "manifest-payload",
                "manifest_path": str(manifest_path),
            }
            audit = {
                "live_minus_ledger": [], "ledger_minus_live": [],
                "live_duplicate_keys": [], "ledger_duplicate_keys": [],
                "equal": True,
            }
            with mock.patch("build_orientifold_axion_database.sys.argv", argv), \
                 mock.patch("build_orientifold_axion_database.require_clean_git_source", return_value=("commit", "tree")), \
                 mock.patch("build_orientifold_axion_database.run_population_preflight", return_value=None), \
                 mock.patch("build_orientifold_axion_database.require_exact_action_h21_plus_validation"), \
                 mock.patch("build_orientifold_axion_database.sha256_of_file", return_value="ledger-sha"), \
                 mock.patch("build_orientifold_axion_database.build_run_manifest", return_value=manifest), \
                 mock.patch("build_orientifold_axion_database.write_run_manifest", return_value=manifest), \
                 mock.patch("build_orientifold_axion_database.load_ledger_accepted_classes", return_value=([], {}, "ledger-sha")), \
                 mock.patch("build_orientifold_axion_database.enrich_accepted_witness_matrices", return_value=([], {})), \
                 mock.patch("build_orientifold_axion_database.select_and_verify_trilayer_population", return_value=([], [], 0, [])), \
                 mock.patch("build_orientifold_axion_database.population_set_audit", return_value=audit), \
                 mock.patch("build_orientifold_axion_database.PAPER_TRILAYER_TARGETS", {}), \
                 mock.patch("build_orientifold_axion_database.build_database", return_value=([], [])), \
                 mock.patch("build_orientifold_axion_database.write_json_report") as write_report:
                main()
            self.assertEqual(write_report.call_count, 1)
            self.assertIn("build_results", write_report.call_args.args[1])

    def test_certificate_input_is_required_before_population_selection(self):
        with self.assertRaisesRegex(FileNotFoundError, "certificate input is missing"):
            load_mpcp_certificates(Path("/private/tmp/does-not-exist-mpcp.json.zst"))
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "invalid.json"
            path.write_text(json.dumps({"certificates": [{}]}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "certificate 0 is invalid"):
                load_mpcp_certificates(path)

    def test_valid_certificate_is_loaded_and_joined_by_polytope_and_frst(self):
        certificate = _valid_mpcp_certificate_fixture()
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "certificates.json"
            path.write_text(json.dumps({"certificates": [certificate]}), encoding="utf-8")
            loaded = load_mpcp_certificates(path)
        self.assertEqual(loaded["count"], 1)
        loaded_certificate = loaded["certificates"][0]
        self.assertEqual(
            _lookup_mpcp_certificate(
                [loaded_certificate],
                polytope_id=loaded_certificate["source"]["polytope_id"],
                frst_hash=loaded_certificate["frst"]["frst_hash"],
            ),
            loaded_certificate,
        )
        self.assertIsNone(
            _lookup_mpcp_certificate(
                [loaded_certificate],
                polytope_id=loaded_certificate["source"]["polytope_id"],
                frst_hash="different-frst",
            )
        )

    def test_preflight_failure_precedes_geometry_loading(self):
        with tempfile.TemporaryDirectory() as directory_name:
            argv = [
                "build_orientifold_axion_database.py", "--h11", "4",
                "--parquet-dir", str(Path(directory_name) / "mirror"),
                "--ledger-population-dir", str(Path(directory_name) / "population"),
                "--ledger-name", "ledger.json.zst",
                "--db-root", str(Path(directory_name) / "database"),
            ]
            with mock.patch("build_orientifold_axion_database.sys.argv", argv), \
                 mock.patch("build_orientifold_axion_database.require_clean_git_source"), \
                 mock.patch(
                     "build_orientifold_axion_database.run_population_preflight",
                     side_effect=RuntimeError("preflight gate"),
                 ), \
                 mock.patch(
                     "build_orientifold_axion_database.mg.load_mirror_polytopes",
                 ) as load_geometry:
                with self.assertRaisesRegex(RuntimeError, "preflight gate"):
                    main()
            load_geometry.assert_not_called()

    def test_pending_stage_forwards_loaded_certificate_to_reconstruction(self):
        certificate = _valid_mpcp_certificate_fixture()
        record = {
            "poly_index": 0,
            "class_index": 0,
            "polytope_id": certificate["source"]["polytope_id"],
            "frst_hash": certificate["frst"]["frst_hash"],
            "poly": mock.Mock(
                points=lambda: np.asarray(certificate["source"]["global_points"])
            ),
            "triangulation": mock.Mock(),
            "ledger_entry": {},
        }
        args = Namespace(h11=4)
        with mock.patch(
            "build_orientifold_axion_database.require_exact_action_h21_plus_validation",
        ), mock.patch(
            "build_orientifold_axion_database._selected_records_by_key",
            return_value={(0, 0): record},
        ), mock.patch(
            "build_orientifold_axion_database.find_exact_trilayer_witnesses",
            return_value=[],
        ) as find_witnesses, mock.patch(
            "build_orientifold_axion_database._exact_trilayer_topology",
            return_value={},
        ), mock.patch(
            "build_orientifold_axion_database.reconstruct_trilayer_actions",
            return_value={"candidates": []},
        ):
            certified, not_certified = certify_and_build_pending(
                args, {(0, 0)}, "ledger.zst", "ledger-sha", str(Path.cwd()),
                1, {}, {}, mpcp_certificates=[certificate],
            )
        self.assertEqual(certified, [])
        self.assertEqual(len(not_certified), 1)
        self.assertIs(
            find_witnesses.call_args.kwargs["mpcp_certificate"], certificate
        )

    def test_default_manifest_path_is_configuration_specific(self):
        common = dict(
            h11=2, stage="full", ledger_name="ledger.zst", db_root="/tmp/db",
            parquet_dir="/tmp/mirror", ledger_population_dir="/tmp/population",
            pending_classes=None, np_index_start=None, certified_trilayer_count=None,
            report=None, manifest=None, np_start=None, np_end=None,
            expected_mismatch_classes=None,
        )
        first = default_run_manifest_path(Namespace(**common))
        changed = dict(common, stage="select")
        second = default_run_manifest_path(Namespace(**changed))
        self.assertNotEqual(first, second)

    def test_population_set_audit_reports_both_directions(self):
        live = [
            {"polytope_id": "p1", "frst_hash": "f1"},
            {"polytope_id": "p2", "frst_hash": "f2"},
        ]
        ledger = [
            {"polytope_id": "p2", "frst_hash": "f2"},
            {"polytope_id": "p3", "frst_hash": "f3"},
        ]
        audit = population_set_audit(live, ledger)
        self.assertEqual(audit["live_minus_ledger"], ["p1::f1"])
        self.assertEqual(audit["ledger_minus_live"], ["p3::f3"])
        self.assertEqual(audit["live_entry_count"], 2)
        self.assertEqual(audit["ledger_entry_count"], 2)
        self.assertEqual(audit["live_duplicate_keys"], [])
        self.assertEqual(audit["ledger_duplicate_keys"], [])
        self.assertFalse(audit["equal"])

    def test_prime_divisor_volumes_use_glsm_columns_not_basis_selector(self):
        # The middle prime divisor is deliberately not a basis divisor.  The
        # old basis-column slicing returned a wrong vector (and could return
        # zero); the canonical GLSM formula is unambiguous.
        glsm = np.asarray([[1, 2, -1], [0, 3, 4]], dtype=np.int64)
        tau = np.asarray([5.0, 7.0])
        labels = np.asarray([10, 11, 12], dtype=np.int64)
        np.testing.assert_allclose(
            canonical_prime_divisor_volumes(glsm, tau, labels),
            [5.0, 31.0, 23.0],
        )

    def test_prime_divisor_volumes_reject_shape_mismatch(self):
        with self.assertRaises(ValueError):
            canonical_prime_divisor_volumes(
                np.eye(2, dtype=np.int64), [1.0, 2.0], [1]
            )

    def test_flat_class_funnel_shape(self):
        # h11=2/3 style: a single-shard summary with class_funnel at the
        # top level.
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            payload = {
                "schema_version": "test-1.0",
                "class_funnel": [_ACCEPTED_ENTRY, _REJECTED_ENTRY],
            }
            _write_compressed_ledger(directory, "flat.json.zst", payload)
            accepted, ledger, sha256 = load_ledger_accepted_classes(
                str(directory / "flat.json.zst"), str(directory / "SHA256SUMS.txt")
            )
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]["polytope_id"], _ACCEPTED_ENTRY["polytope_id"])
            self.assertEqual(ledger["schema_version"], "test-1.0")
            self.assertEqual(len(sha256), 64)

    def test_nested_terminal_ledger_class_funnel_shape(self):
        # h11=4 style: a merged, sharded artifact with class_funnel nested
        # under terminal_ledger.
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            payload = {
                "schema_version": "test-merged-1.0",
                "shard_count": 4,
                "terminal_ledger": {
                    "class_count": 2,
                    "class_funnel": [_ACCEPTED_ENTRY, _REJECTED_ENTRY],
                },
            }
            _write_compressed_ledger(directory, "merged.json.zst", payload)
            accepted, ledger, sha256 = load_ledger_accepted_classes(
                str(directory / "merged.json.zst"), str(directory / "SHA256SUMS.txt")
            )
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]["polytope_id"], _ACCEPTED_ENTRY["polytope_id"])
            self.assertEqual(ledger["shard_count"], 4)
            self.assertEqual(len(sha256), 64)

    def test_unrecognized_shape_raises(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            payload = {"schema_version": "test-1.0", "something_else": []}
            _write_compressed_ledger(directory, "malformed.json.zst", payload)
            with self.assertRaises(RuntimeError):
                load_ledger_accepted_classes(
                    str(directory / "malformed.json.zst"),
                    str(directory / "SHA256SUMS.txt"),
                )

    def test_sha256_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            payload = {"class_funnel": [_ACCEPTED_ENTRY]}
            _write_compressed_ledger(directory, "tampered.json.zst", payload)
            # Corrupt the recorded checksum so it no longer matches the file.
            sums_path = directory / "SHA256SUMS.txt"
            sums_path.write_text("0" * 64 + "  tampered.json.zst\n")
            with self.assertRaises(RuntimeError):
                load_ledger_accepted_classes(
                    str(directory / "tampered.json.zst"), str(sums_path)
                )


if __name__ == "__main__":
    unittest.main()
